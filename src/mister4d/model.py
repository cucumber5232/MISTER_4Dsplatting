"""Dynamic scene representation: Gaussians that are local in time.

Every Gaussian is an ordinary 3D Gaussian plus three temporal attributes:

- ``times``: the instant (in frames) at which it is centred;
- ``durations``: how long it lives, as the standard deviation (in frames) of a
  temporal bell curve that modulates its opacity;
- ``velocities``: a linear motion (metres per frame) around that instant.

At frame ``f`` the scene is the set of Gaussians alive at ``f``, each moved by
``velocity * (f - time)`` and faded by its temporal weight. Nothing has to be
tracked across the whole take: a fast limb is simply described by Gaussians
that live for a few frames. Parts that do not move can be described by
long-lived Gaussians.

A static scene (plain 3DGS) is the special case with the temporal attributes
switched off.

Written from scratch for this project on top of PyTorch; the only third-party
rendering code involved is gsplat (Apache-2.0).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .rasterizer import ALPHA_MIN, eval_sh, num_sh_bases, rgb_to_sh0

MIN_DURATION = 0.35  # frames; shorter lives would alias between frames


def estimate_velocities(xyz: np.ndarray, frames: np.ndarray, smooth: int = 8) -> np.ndarray:
    """Initial velocity (metres per frame) of every point from the motion of the point clouds.

    Each frame has its own points, sampled independently on the surface of
    that instant. The displacement to the nearest point of the next and of the
    previous frame tells how the surface moved there, along its normal: exactly
    the component that makes Gaussians of neighbouring frames line up instead
    of leaving a ghost. Displacements no larger than the sampling distance are
    noise and count as "not moving".
    """
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, dtype=np.float64)
    velocities = np.zeros_like(xyz)
    ids = np.unique(frames)
    clouds = {f: xyz[frames == f] for f in ids}
    trees = {f: cKDTree(clouds[f]) for f in ids}
    for i, f in enumerate(ids):
        sel = np.nonzero(frames == f)[0]
        pts = clouds[f]
        spacing = np.median(trees[f].query(pts, k=min(2, len(pts)))[0][:, -1]) if len(pts) > 1 else 0.0
        steps = []
        for j, sign in ((i + 1, 1.0), (i - 1, -1.0)):
            if 0 <= j < len(ids):
                g = ids[j]
                _, nearest = trees[g].query(pts)
                steps.append(sign * (clouds[g][nearest] - pts) / abs(float(g) - float(f)))
        if not steps:
            continue
        motion = np.mean(steps, axis=0)
        if len(pts) > smooth:  # neighbours on the surface move together
            _, nb = trees[f].query(pts, k=smooth)
            motion = motion[nb].mean(axis=1)
        motion[np.linalg.norm(motion, axis=1) < 1.5 * spacing] = 0.0
        velocities[sel] = motion
    return velocities.astype(np.float32)


def init_gaussian_params(xyz: np.ndarray, rgb: np.ndarray, frames: np.ndarray, sh_degree: int,
                         init_opacity: float = 0.1, init_duration: float = 1.0,
                         velocities: np.ndarray | None = None) -> nn.ParameterDict:
    """Isotropic Gaussians sized by the distance to their neighbours of the same frame."""
    from scipy.spatial import cKDTree

    xyz = np.asarray(xyz, dtype=np.float32)
    frames = np.asarray(frames, dtype=np.float32)
    n = len(xyz)
    spacing = np.full(n, 0.01, dtype=np.float32)
    for f in np.unique(frames):
        sel = np.nonzero(frames == f)[0]
        k = min(4, len(sel))
        if k > 1:
            dist, _ = cKDTree(xyz[sel]).query(xyz[sel], k=k)
            spacing[sel] = dist[:, 1:].mean(axis=1)
    spacing = np.clip(spacing, 1e-4, None)
    quats = torch.zeros(n, 4)
    quats[:, 0] = 1.0
    return nn.ParameterDict(
        {
            "means": nn.Parameter(torch.from_numpy(xyz)),
            "scales": nn.Parameter(torch.log(torch.from_numpy(spacing))[:, None].repeat(1, 3)),
            "quats": nn.Parameter(quats),
            "opacities": nn.Parameter(torch.full((n,), math.log(init_opacity / (1 - init_opacity)))),
            "sh0": nn.Parameter(rgb_to_sh0(torch.from_numpy(np.asarray(rgb, dtype=np.float32)))[:, None, :]),
            "shN": nn.Parameter(torch.zeros(n, num_sh_bases(sh_degree) - 1, 3)),
            "times": nn.Parameter(torch.from_numpy(frames.copy())),
            "durations": nn.Parameter(torch.full((n,), math.log(init_duration))),
            "velocities": nn.Parameter(torch.zeros(n, 3) if velocities is None
                                       else torch.from_numpy(np.asarray(velocities, dtype=np.float32))),
        }
    )


class SceneView:
    """The tensors of a scene on one device, ready to be rendered.

    On the device that owns the parameters this is just a window on them. On
    any other device it holds differentiable copies, which is how one training
    step is spread over several GPUs: gradients flow back to the owner.
    """

    def __init__(self, params: dict[str, Tensor], sh_degree: int, static: bool):
        self.params = params
        self.sh_degree = sh_degree
        self.static = static

    def state_at(self, frame: float | Tensor) -> dict[str, Tensor]:
        """Activated parameters of the Gaussians alive at ``frame`` (a frame index, may be fractional).

        ``frame`` may be a 0-dim tensor: the result is then differentiable with
        respect to time, which is how the time offset of a camera is learned.
        ``active`` holds their indices in the full parameter arrays (None = all).
        """
        p = self.params
        opacities = torch.sigmoid(p["opacities"])
        if self.static:
            active, means = None, p["means"]
            pick = lambda x: x  # noqa: E731
        else:
            dt = frame - p["times"]
            sigma = torch.exp(p["durations"]).clamp(min=MIN_DURATION)
            opacities = opacities * torch.exp(-0.5 * (dt / sigma) ** 2)
            active = torch.nonzero(opacities.detach() >= ALPHA_MIN).squeeze(-1)
            pick = lambda x: x[active]  # noqa: E731
            means = pick(p["means"]) + pick(p["velocities"]) * pick(dt)[:, None]
            opacities = pick(opacities)
        return {
            "means": means,
            "quats": F.normalize(pick(p["quats"]), dim=-1),
            "scales": torch.exp(pick(p["scales"])),
            "opacities": opacities,
            "sh": torch.cat([pick(p["sh0"]), pick(p["shN"])], dim=1),
            "velocities": None if self.static else pick(p["velocities"]),
            "active": active,
        }

    def colors(self, state: dict[str, Tensor], cam_centers: Tensor, active_degree: int | None = None) -> Tensor:
        """View-dependent colours [C, M, 3] ([M, 3] at SH degree 0)."""
        degree = self.sh_degree if active_degree is None else min(active_degree, self.sh_degree)
        dirs = state["means"].detach()[None, :, :] - cam_centers[:, None, :]
        return eval_sh(degree, state["sh"], dirs)


class DynamicScene(nn.Module):
    def __init__(self, params: nn.ParameterDict, sh_degree: int, static: bool = False):
        super().__init__()
        self.params = params
        self.sh_degree = sh_degree
        self.static = static

    @property
    def num_gaussians(self) -> int:
        return int(self.params["means"].shape[0])

    @property
    def device(self) -> torch.device:
        return self.params["means"].device

    def view(self, device: torch.device | str | None = None) -> SceneView:
        device = self.device if device is None else torch.device(device)
        if device == self.device:
            return SceneView(dict(self.params.items()), self.sh_degree, self.static)
        return SceneView({k: v.to(device) for k, v in self.params.items()}, self.sh_degree, self.static)

    def state_at(self, frame: float | Tensor) -> dict[str, Tensor]:
        return self.view().state_at(frame)


def render_scene(scene: DynamicScene | SceneView, frame: float | Tensor, viewmats: Tensor, Ks_opencv: Tensor,
                 width: int, height: int, backend: str = "auto", antialiased: bool = False,
                 active_sh_degree: int | None = None):
    """Render the scene at ``frame`` from C cameras.

    ``Ks_opencv`` are intrinsics in the calibration-file convention.
    Returns premultiplied rgb [C,H,W,3], alpha [C,H,W,1], depth [C,H,W,1],
    the rasterizer info dict and the state of the Gaussians alive at ``frame``.
    """
    from .rasterizer import opencv_to_render_K, render

    view = scene.view() if isinstance(scene, DynamicScene) else scene
    state = view.state_at(frame)
    cam_centers = -torch.einsum("cji,cj->ci", viewmats[:, :3, :3], viewmats[:, :3, 3])
    colors = view.colors(state, cam_centers, active_sh_degree)
    rgb, alpha, depth, info = render(
        state["means"], state["quats"], state["scales"], state["opacities"], colors,
        viewmats, opencv_to_render_K(Ks_opencv), width, height,
        backend=backend, antialiased=antialiased,
    )
    return rgb, alpha, depth, info, state


def save_checkpoint(path, scene: DynamicScene, config: dict, meta: dict, step: int) -> None:
    torch.save(
        {
            "format": "mister4d-checkpoint",
            "version": 2,
            "step": step,
            "config": config,
            "meta": meta,
            "sh_degree": scene.sh_degree,
            "static": scene.static,
            "params": {k: v.detach().cpu() for k, v in scene.params.items()},
        },
        path,
    )


def load_checkpoint(path, device: str | torch.device = "cpu") -> tuple[DynamicScene, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("format") != "mister4d-checkpoint" or ckpt.get("version") != 2:
        raise ValueError(f"{path}: not a mister4d checkpoint (version 2)")
    params = nn.ParameterDict({k: nn.Parameter(v) for k, v in ckpt["params"].items()})
    scene = DynamicScene(params, ckpt["sh_degree"], ckpt["static"]).to(device)
    scene.eval()
    return scene, ckpt
