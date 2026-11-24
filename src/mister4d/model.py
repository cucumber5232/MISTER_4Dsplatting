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

A take can also carry its *scenery*: the Gaussians of the empty set, trained
once as a static scene and then frozen. They are drawn together with every
instant and are never optimised again, so that what moves is described only by
the Gaussians of the take, and the set is the same from the first frame to the
last.

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
SCENERY_KEYS = ("means", "quats", "scales", "opacities", "sh")  # activated values, see scenery_of


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


def _rotmat_to_quat(R: Tensor) -> Tensor:
    """Rotation matrices [N,3,3] to quaternions wxyz [N,4]."""
    m00, m11, m22 = R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]
    w = torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=1e-8)) / 2.0
    x = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-8)) / 2.0
    y = torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=1e-8)) / 2.0
    z = torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=1e-8)) / 2.0
    big = torch.stack([w, x, y, z], dim=-1).argmax(dim=-1)
    q = torch.zeros(R.shape[0], 4, dtype=R.dtype)
    for k in range(4):
        sel = big == k
        if not sel.any():
            continue
        r = R[sel]
        if k == 0:
            d = 4.0 * w[sel]
            q[sel] = torch.stack([w[sel], (r[:, 2, 1] - r[:, 1, 2]) / d, (r[:, 0, 2] - r[:, 2, 0]) / d,
                                  (r[:, 1, 0] - r[:, 0, 1]) / d], dim=-1)
        elif k == 1:
            d = 4.0 * x[sel]
            q[sel] = torch.stack([(r[:, 2, 1] - r[:, 1, 2]) / d, x[sel], (r[:, 0, 1] + r[:, 1, 0]) / d,
                                  (r[:, 0, 2] + r[:, 2, 0]) / d], dim=-1)
        elif k == 2:
            d = 4.0 * y[sel]
            q[sel] = torch.stack([(r[:, 0, 2] - r[:, 2, 0]) / d, (r[:, 0, 1] + r[:, 1, 0]) / d, y[sel],
                                  (r[:, 1, 2] + r[:, 2, 1]) / d], dim=-1)
        else:
            d = 4.0 * z[sel]
            q[sel] = torch.stack([(r[:, 1, 0] - r[:, 0, 1]) / d, (r[:, 0, 2] + r[:, 2, 0]) / d,
                                  (r[:, 1, 2] + r[:, 2, 1]) / d, z[sel]], dim=-1)
    return q


def init_gaussian_params(xyz: np.ndarray, rgb: np.ndarray, frames: np.ndarray, sh_degree: int,
                         init_opacity: float = 0.1, init_duration: float = 1.0,
                         velocities: np.ndarray | None = None, flatten: float = 1.0) -> nn.ParameterDict:
    """Gaussians sized by the distance to their neighbours of the same frame.

    Isotropic by default. With ``flatten`` < 1 they start as discs lying on the
    surface the points describe: that much thinner across it than along it.
    """
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
    log_scales = torch.log(torch.from_numpy(spacing))[:, None].repeat(1, 3)
    if flatten < 1.0:
        normals = np.zeros((n, 3), dtype=np.float32)
        normals[:, 2] = 1.0
        for f in np.unique(frames):
            sel = np.nonzero(frames == f)[0]
            k = min(10, len(sel))
            if k >= 4:  # the direction in which the neighbours of a point spread least
                _, nb = cKDTree(xyz[sel]).query(xyz[sel], k=k)
                local = xyz[sel][nb] - xyz[sel][nb].mean(axis=1, keepdims=True)
                _, vecs = np.linalg.eigh(np.einsum("nki,nkj->nij", local, local))
                normals[sel] = vecs[:, :, 0]
        z = torch.from_numpy(normals)
        helper = torch.where(z[:, 2:3].abs() < 0.9, torch.tensor([[0.0, 0.0, 1.0]]), torch.tensor([[1.0, 0.0, 0.0]]))
        x = F.normalize(torch.cross(helper.expand_as(z), z, dim=-1), dim=-1)
        y = torch.cross(z, x, dim=-1)
        quats = _rotmat_to_quat(torch.stack([x, y, z], dim=-1))  # the local z axis is the normal
        log_scales[:, 2] += math.log(flatten)
    return nn.ParameterDict(
        {
            "means": nn.Parameter(torch.from_numpy(xyz)),
            "scales": nn.Parameter(log_scales),
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

    def __init__(self, params: dict[str, Tensor], sh_degree: int, static: bool,
                 scenery: dict[str, Tensor] | None = None, scenery_degree: int = 0):
        self.params = params
        self.sh_degree = sh_degree
        self.static = static
        self.scenery = scenery  # frozen Gaussians of the empty set, or None
        self.scenery_degree = scenery_degree

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

    def scenery_colors(self, cam_centers: Tensor) -> Tensor:
        """Colours of the scenery [C, S, 3] ([S, 3] at SH degree 0)."""
        dirs = self.scenery["means"][None, :, :] - cam_centers[:, None, :]
        return eval_sh(self.scenery_degree, self.scenery["sh"], dirs)


def scenery_of(scene: "DynamicScene", min_opacity: float = ALPHA_MIN) -> tuple[dict[str, Tensor], int]:
    """A trained static scene as scenery: its visible Gaussians, activated and detached, and their SH degree."""
    if not scene.static:
        raise ValueError("the scenery must be a static scene")
    state = scene.view().state_at(0)
    keep = state["opacities"].detach() >= min_opacity
    return {k: state[k].detach()[keep].clone() for k in SCENERY_KEYS}, scene.sh_degree


class DynamicScene(nn.Module):
    def __init__(self, params: nn.ParameterDict, sh_degree: int, static: bool = False,
                 scenery: dict[str, Tensor] | None = None, scenery_degree: int = 0):
        super().__init__()
        self.params = params
        self.sh_degree = sh_degree
        self.static = static
        self.scenery_degree = int(scenery_degree)
        self._scenery_on: dict[str, dict[str, Tensor]] = {}  # copies on other devices, made once
        for key in SCENERY_KEYS:  # buffers: they follow the scene from device to device and are never optimised
            self.register_buffer("scenery_" + key, None if scenery is None else scenery[key].detach().clone(),
                                 persistent=False)

    @property
    def num_gaussians(self) -> int:
        """Gaussians being optimised (the scenery is not counted)."""
        return int(self.params["means"].shape[0])

    @property
    def num_scenery(self) -> int:
        return 0 if self.scenery_means is None else int(self.scenery_means.shape[0])

    def scenery(self, device: torch.device | str | None = None) -> dict[str, Tensor] | None:
        if self.scenery_means is None:
            return None
        device = self.device if device is None else torch.device(device)
        if device == self.scenery_means.device:
            return {k: getattr(self, "scenery_" + k) for k in SCENERY_KEYS}
        if str(device) not in self._scenery_on:
            self._scenery_on[str(device)] = {k: getattr(self, "scenery_" + k).to(device) for k in SCENERY_KEYS}
        return self._scenery_on[str(device)]

    @property
    def device(self) -> torch.device:
        return self.params["means"].device

    def view(self, device: torch.device | str | None = None) -> SceneView:
        device = self.device if device is None else torch.device(device)
        scenery = self.scenery(device)
        if device == self.device:
            return SceneView(dict(self.params.items()), self.sh_degree, self.static, scenery, self.scenery_degree)
        return SceneView({k: v.to(device) for k, v in self.params.items()}, self.sh_degree, self.static,
                         scenery, self.scenery_degree)

    def state_at(self, frame: float | Tensor) -> dict[str, Tensor]:
        return self.view().state_at(frame)


def render_scene(scene: DynamicScene | SceneView, frame: float | Tensor, viewmats: Tensor, Ks_opencv: Tensor,
                 width: int, height: int, backend: str = "auto", antialiased: bool = False,
                 active_sh_degree: int | None = None, moments: bool = False):
    """Render the scene at ``frame`` from C cameras.

    ``Ks_opencv`` are intrinsics in the calibration-file convention.
    Returns premultiplied rgb [C,H,W,3], alpha [C,H,W,1], depth [C,H,W,1],
    the rasterizer info dict and the state of the Gaussians alive at ``frame``.
    With ``moments`` the info also holds ``moments`` [C,H,W,2]: the blended depth
    and squared depth of what every pixel is made of (not divided by alpha).

    The scenery of the scene, if it has one, is drawn too: its Gaussians come
    after those of the state in what is handed to the rasterizer, and
    ``info["count"]`` says how many belong to the state.
    """
    from .rasterizer import opencv_to_render_K, render

    view = scene.view() if isinstance(scene, DynamicScene) else scene
    state = view.state_at(frame)
    cam_centers = -torch.einsum("cji,cj->ci", viewmats[:, :3, :3], viewmats[:, :3, 3])
    colors = view.colors(state, cam_centers, active_sh_degree)
    means, quats, scales, opacities = state["means"], state["quats"], state["scales"], state["opacities"]
    if view.scenery is not None:
        behind = view.scenery_colors(cam_centers)
        cams = viewmats.shape[0]
        if colors.dim() != behind.dim():  # one of the two depends on the point of view
            colors = colors if colors.dim() == 3 else colors[None].expand(cams, -1, -1)
            behind = behind if behind.dim() == 3 else behind[None].expand(cams, -1, -1)
        colors = torch.cat([colors, behind], dim=-2)
        means = torch.cat([means, view.scenery["means"]])
        quats = torch.cat([quats, view.scenery["quats"]])
        scales = torch.cat([scales, view.scenery["scales"]])
        opacities = torch.cat([opacities, view.scenery["opacities"]])
    if moments:  # depth and squared depth of every Gaussian, blended like two more colours
        z = torch.einsum("cj,mj->cm", viewmats[:, 2, :3], means) + viewmats[:, 2, 3:4]
        if colors.dim() == 2:
            colors = colors[None].expand(z.shape[0], -1, -1)
        colors = torch.cat([colors, z[..., None], (z * z)[..., None]], dim=-1)
    rgb, alpha, depth, info = render(
        means, quats, scales, opacities, colors,
        viewmats, opencv_to_render_K(Ks_opencv), width, height,
        backend=backend, antialiased=antialiased,
    )
    if view.scenery is not None:
        info["count"] = state["means"].shape[0]
    if moments:
        info["moments"], rgb = rgb[..., 3:5], rgb[..., :3]
    return rgb, alpha, depth, info, state


def save_checkpoint(path, scene: DynamicScene, config: dict, meta: dict, step: int) -> None:
    """Write a scene. Its scenery is not copied into the file when ``meta["scenery_file"]`` says where it is
    (a path relative to the folder of the checkpoint): a long take has one scenery for all its segments."""
    data = {
        "format": "mister4d-checkpoint",
        "version": 2,
        "step": step,
        "config": config,
        "meta": meta,
        "sh_degree": scene.sh_degree,
        "static": scene.static,
        "params": {k: v.detach().cpu() for k, v in scene.params.items()},
    }
    if scene.num_scenery and not meta.get("scenery_file"):
        data["scenery"] = {k: v.detach().cpu() for k, v in scene.scenery().items()}
        data["scenery_degree"] = scene.scenery_degree
    torch.save(data, path)


def load_checkpoint(path, device: str | torch.device = "cpu") -> tuple[DynamicScene, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("format") != "mister4d-checkpoint" or ckpt.get("version") != 2:
        raise ValueError(f"{path}: not a mister4d checkpoint (version 2)")
    params = nn.ParameterDict({k: nn.Parameter(v) for k, v in ckpt["params"].items()})
    scenery, degree = ckpt.get("scenery"), int(ckpt.get("scenery_degree", 0))
    reference = ckpt["meta"].get("scenery_file")
    if scenery is None and reference:
        from pathlib import Path

        source = Path(path).parent / reference
        if not source.exists():
            raise FileNotFoundError(f"{path}: the scenery of this take is missing ({source})")
        scenery, degree = scenery_of(load_checkpoint(source)[0])
    scene = DynamicScene(params, ckpt["sh_degree"], ckpt["static"], scenery, degree).to(device)
    scene.eval()
    return scene, ckpt
