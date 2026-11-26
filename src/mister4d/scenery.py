"""The empty set as a scene of its own: its images and its starting points.

A take can be reconstructed whole, set included, instead of cutting the
subject out of it. The set does not move, so it is reconstructed once, from
one image per camera with nobody in it (the *plate*), as a static scene; the
take is then trained on top of it, on the full images (see ``trainer.py``).

This module prepares what that needs, inside the dataset:

- ``background/<cam>.png``: the plates. Those recorded with the empty set are
  used as they are; a missing one is estimated from the take, as the median
  over time of every pixel (what is there when the subject is not);
- ``change/<cam>/<frame>.png``: where every frame differs from its plate,
  shadows included. Only the visual hull uses them, to place the first
  Gaussians of what moves; the training never sees a mask;
- ``scenery/``: a one-frame dataset of the plates, with starting points found
  by plane sweeping between neighbouring cameras.

Plane sweeping needs no learned model: a plane is moved through the scene, the
plates of the neighbouring cameras are warped onto the reference camera
through it, and every pixel keeps the plane on which the images agree best.
Two families of planes are swept: facing the camera (walls, backdrops) and
horizontal (floors, which a camera-facing plane cannot follow when the
cameras are far apart).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .camera import Rig
from .dataset import frame_name, read_meta, update_meta, write_meta, write_points_ply
from .masks import foreground_mask


def _plate_path(root: Path, cam: str) -> Path | None:
    return next((p for p in sorted((root / "background").glob(f"{cam}.*"))), None) \
        if (root / "background").is_dir() else None


def estimate_plates(data_dir: str | Path, max_frames: int = 25, log=print) -> str:
    """Make sure every camera has a plate in ``background/``; returns "recorded", "estimated" or "mixed"."""
    root = Path(data_dir)
    meta = read_meta(root)
    ids = meta["frame_ids"]
    picked = [ids[i] for i in sorted(set(np.linspace(0, len(ids) - 1, min(len(ids), max_frames)).round().astype(int)))]
    made = []
    for cam in meta["cameras"]:
        if _plate_path(root, cam) is not None:
            continue
        frames = [cv2.imread(str(next((root / "images" / cam).glob(f"{frame_name(fid)}.*"))), cv2.IMREAD_COLOR)
                  for fid in picked]
        (root / "background").mkdir(exist_ok=True)
        cv2.imwrite(str(root / "background" / f"{cam}.png"), np.median(np.stack(frames), axis=0).astype(np.uint8))
        made.append(cam)
    source = "recorded" if not made else "estimated" if len(made) == len(meta["cameras"]) else "mixed"
    if made:
        log(f"[scenery] no empty-set image for {', '.join(made)}: estimated from the take as the median of "
            f"{len(picked)} frames (whatever stays still for the whole take becomes part of the set)")
    update_meta(root, plates=source)
    return source


def change_masks(data_dir: str | Path, threshold: float = 24.0, log=print) -> None:
    """``change/<cam>/<frame>.png``: where each frame differs from its plate, shadows included."""
    root = Path(data_dir)
    meta = read_meta(root)
    for cam in meta["cameras"]:
        plate = cv2.imread(str(_plate_path(root, cam)), cv2.IMREAD_COLOR)
        (root / "change" / cam).mkdir(parents=True, exist_ok=True)
        share = []
        for fid in meta["frame_ids"]:
            img = cv2.imread(str(next((root / "images" / cam).glob(f"{frame_name(fid)}.*"))), cv2.IMREAD_COLOR)
            mask = foreground_mask(img, plate, threshold, suppress_shadows=False)
            cv2.imwrite(str(root / "change" / cam / f"{frame_name(fid)}.png"), mask)
            share.append(float((mask > 127).mean()))
        log(f"[scenery] {cam}: {np.mean(share):.1%} of the image changes during the take")


# ---------------------------------------------------------------------------
# Plane sweep
# ---------------------------------------------------------------------------


def _box(x: torch.Tensor, k: int) -> torch.Tensor:
    return F.avg_pool2d(x, k, stride=1, padding=k // 2, count_include_pad=False)


class _SweepCamera:
    """A camera at the working resolution, with its plate in grey levels."""

    def __init__(self, cam, plate_bgr: np.ndarray, width: int, device):
        scale = width / cam.width
        self.w, self.h = int(round(cam.width * scale)), int(round(cam.height * scale))
        sx, sy = self.w / cam.width, self.h / cam.height
        K = np.array(cam.K, dtype=np.float64)
        self.fx, self.fy = K[0, 0] * sx, K[1, 1] * sy
        self.cx, self.cy = (K[0, 2] + 0.5) * sx - 0.5, (K[1, 2] + 0.5) * sy - 0.5
        small = cv2.resize(plate_bgr, (self.w, self.h), interpolation=cv2.INTER_AREA)
        self.rgb = small[..., ::-1].astype(np.float32) / 255.0
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        self.grey = torch.from_numpy(grey).to(device)[None, None]
        self.R = torch.tensor(np.asarray(cam.R), dtype=torch.float32, device=device)
        self.t = torch.tensor(np.asarray(cam.t), dtype=torch.float32, device=device)
        self.center = torch.tensor(np.asarray(cam.center), dtype=torch.float32, device=device)
        v, u = torch.meshgrid(torch.arange(self.h, device=device, dtype=torch.float32),
                              torch.arange(self.w, device=device, dtype=torch.float32), indexing="ij")
        rays_cam = torch.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy, torch.ones_like(u)], dim=-1)
        self.rays = rays_cam @ self.R  # world directions with unit depth: point = centre + depth * ray
        self.forward = self.R[2]

    def project(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sampling grid for ``grid_sample`` and the mask of the points that fall inside the image."""
        cam = points @ self.R.T + self.t
        z = cam[..., 2]
        u = self.fx * cam[..., 0] / z.clamp(min=1e-6) + self.cx
        v = self.fy * cam[..., 1] / z.clamp(min=1e-6) + self.cy
        inside = (z > 0.05) & (u >= 0) & (u <= self.w - 1) & (v >= 0) & (v <= self.h - 1)
        grid = torch.stack([2.0 * u / (self.w - 1) - 1.0, 2.0 * v / (self.h - 1) - 1.0], dim=-1)
        return grid, inside


@torch.no_grad()
def _sweep(ref: _SweepCamera, others: list[_SweepCamera], normals: torch.Tensor, offsets: torch.Tensor,
           window: int = 7, batch: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """For every pixel of ``ref``: the depth on the best plane ``normal . x = offset`` and how badly images agree there.

    The cost is one minus the normalised cross-correlation of a small window,
    averaged over the cameras that see the point (2 = no camera does).
    """
    mean_r = _box(ref.grey, window)
    var_r = (_box(ref.grey * ref.grey, window) - mean_r * mean_r).clamp(min=0.0)
    best_cost = torch.full((ref.h, ref.w), 2.0, device=ref.grey.device)
    best_depth = torch.zeros(ref.h, ref.w, device=ref.grey.device)
    for start in range(0, len(offsets), batch):
        n, d = normals[start:start + batch], offsets[start:start + batch]
        along = torch.einsum("hwc,bc->bhw", ref.rays, n)
        depth = (d - n @ ref.center)[:, None, None] / torch.where(along.abs() < 1e-6, torch.full_like(along, 1e-6), along)
        ahead = depth > 0.05
        points = ref.center + depth.clamp(min=0.05)[..., None] * ref.rays[None]
        total = torch.zeros_like(depth)
        seen = torch.zeros_like(depth)
        for other in others:
            grid, inside = other.project(points)
            inside = (inside & ahead).float()[:, None]
            warped = F.grid_sample(other.grey.expand(len(d), -1, -1, -1), grid, mode="bilinear",
                                   padding_mode="zeros", align_corners=True)
            usable = (_box(inside, window) > 0.9).float()[:, 0]
            mean_w = _box(warped, window)
            var_w = (_box(warped * warped, window) - mean_w * mean_w).clamp(min=0.0)
            cov = _box(ref.grey * warped, window) - mean_r * mean_w
            ncc = (cov / torch.sqrt(var_r * var_w + 1e-6))[:, 0]
            total += (1.0 - ncc) * usable
            seen += usable
        cost = torch.where(seen > 0, total / seen.clamp(min=1.0), torch.full_like(total, 2.0))
        cost, which = cost.min(dim=0)
        better = cost < best_cost
        best_cost = torch.where(better, cost, best_cost)
        best_depth = torch.where(better, depth.gather(0, which[None])[0], best_depth)
    return best_depth, best_cost


SECTORS = 72  # directions around the centre in which the reach of the set is measured


def envelope_of(points: np.ndarray, center: np.ndarray, cameras: np.ndarray) -> dict:
    """The outside of a set, from the points two cameras agree on: its floor and how far it reaches.

    ``floor`` is the most common height of the points below the cameras (None if there is none).
    ``reach`` is, for each of ``SECTORS`` directions around ``center``, the horizontal distance of
    the farthest group of points among those within 12 degrees of it: the walls, or as far as the
    floor is seen. On a plain wall the cameras agree only along its lower edge, so the points of
    the floor and of the bottom of the walls are what tells where the set ends. A few points
    alone farther out are wrong matches and do not count; a few points of a wall behind a lamp
    do. The cameras stand inside the set: a direction whose points all lie nearer than the
    cameras says nothing about its walls and takes the reach of the directions next to it. When
    in doubt the envelope is too wide: see ``_on_envelope``.
    """
    center = np.asarray(center, dtype=np.float64)
    floor = None
    low = points[points[:, 2] < cameras[:, 2].min() - 0.2, 2] if len(points) else np.zeros(0)
    if len(low) >= 100:
        hist, edges = np.histogram(low, bins=np.arange(low.min() - 0.02, low.max() + 0.04, 0.02))
        floor = float(edges[int(np.argmax(hist))] + 0.01)
    rel = points[:, :2] - center[:2]
    radius = np.hypot(rel[:, 0], rel[:, 1])
    sector = np.floor((np.arctan2(rel[:, 1], rel[:, 0]) + np.pi) / (2 * np.pi) * SECTORS).astype(int) % SECTORS
    ring = float(np.hypot(*(cameras[:, :2] - center[:2]).T).max())
    step = 0.25

    def farthest_groups(use: np.ndarray) -> np.ndarray:
        found = np.full(SECTORS, np.nan)
        for k in range(SECTORS):
            near = (sector - k + SECTORS // 2) % SECTORS - SECTORS // 2  # sectors from k, either way
            here = radius[use & (np.abs(near) <= 2)]
            if len(here) < 20:
                continue
            hist, edges = np.histogram(here, bins=np.arange(0.0, here.max() + 2 * step, step))
            crowded = np.flatnonzero(hist >= max(5, 0.005 * len(here)))
            if len(crowded) and edges[crowded[-1] + 1] > 1.15 * ring:
                last = here[(here >= edges[crowded[-1]]) & (here < edges[crowded[-1] + 1])]
                found[k] = float(np.median(last)) + 0.05
        return found

    reach = farthest_groups(np.ones(len(points), dtype=bool))
    if np.isnan(reach).all():
        reach[:] = 2.0 * ring
    else:
        known = np.flatnonzero(~np.isnan(reach))
        turns = np.concatenate([known - SECTORS, known, known + SECTORS])
        reach = np.interp(np.arange(SECTORS), turns, np.tile(reach[known], 3))
        around = np.stack([np.roll(reach, k) for k in range(-2, 3)])
        reach = np.median(around, axis=0)  # one direction with a few wrong points does not move the wall
    reach = np.maximum(reach, 1.15 * ring)
    return {"center": center[:2].tolist(), "floor": floor, "reach": reach.tolist(), "ring": ring,
            "top": float(cameras[:, 2].max() + 1.0)}


def envelope_reach(envelope: dict, xy: np.ndarray) -> np.ndarray:
    """How far the set reaches, from its centre, in the direction of each point ``xy`` [N, 2]."""
    reach = np.asarray(envelope["reach"], dtype=np.float64)
    rel = np.asarray(xy, dtype=np.float64) - np.asarray(envelope["center"])
    at = (np.arctan2(rel[..., 1], rel[..., 0]) + np.pi) / (2 * np.pi) * SECTORS - 0.5
    lo = np.floor(at).astype(int)
    frac = at - lo
    return reach[lo % SECTORS] * (1.0 - frac) + reach[(lo + 1) % SECTORS] * frac


def _on_envelope(rays: np.ndarray, origin: np.ndarray, envelope: dict, far: float) -> np.ndarray:
    """Depth at which each ray leaves the set: at its floor or at its reach, whichever comes first.

    This is where a pixel is put when the cameras do not agree on its depth (plain walls, what
    one camera alone sees). It is the farthest place it can be. A point put too far hides
    nothing; one left in mid air becomes a veil in front of the views between the cameras.
    """
    origin = np.asarray(origin, dtype=np.float64)
    d = rays.reshape(-1, 3).astype(np.float64)
    level = np.hypot(d[:, 0], d[:, 1])
    depth = np.full(len(d), far)
    # Where the ray crosses the reach of the set: the reach depends on the direction of the
    # crossing point, so it is found in a few rounds from a first guess.
    o = origin[:2] - np.asarray(envelope["center"])
    a = level ** 2
    b = d[:, 0] * o[0] + d[:, 1] * o[1]
    guess = np.full(len(d), float(np.median(envelope["reach"])))
    for _ in range(6):
        c = o @ o - guess ** 2
        t = (-b + np.sqrt(np.maximum(b * b - a * c, 0.0))) / np.maximum(a, 1e-12)
        guess = envelope_reach(envelope, origin[:2] + t[:, None] * d[:, :2])
    depth = np.where(level > 1e-6, np.minimum(depth, t), depth)
    if envelope["floor"] is not None:
        down = d[:, 2] < -1e-6
        depth = np.where(down, np.minimum(depth, (envelope["floor"] - origin[2]) / np.where(down, d[:, 2], -1.0)), depth)
    return np.clip(depth, 0.05, far).reshape(rays.shape[:-1]).astype(np.float32)


def inside_envelope(envelope: dict, xyz: np.ndarray, margin: float) -> np.ndarray:
    """Which points are in the room of the set, more than ``margin`` from its floor and from its reach."""
    xyz = np.asarray(xyz, dtype=np.float64)
    rel = xyz[:, :2] - np.asarray(envelope["center"])
    inside = np.hypot(rel[:, 0], rel[:, 1]) < envelope_reach(envelope, xyz[:, :2]) - margin
    if envelope["floor"] is not None:
        inside &= xyz[:, 2] > envelope["floor"] + margin
    return inside


def in_the_rig(envelope: dict, xyz: np.ndarray, margin: float = 0.3) -> np.ndarray:
    """Which points are between the cameras and the subject: inside the ring of the cameras, in the room.

    That is, nearer to the centre than the farthest camera and than the walls (less ``margin``),
    more than ``margin`` above the floor and below ``top`` (a metre above the highest camera).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    rel = xyz[:, :2] - np.asarray(envelope["center"])
    limit = np.minimum(envelope["ring"], envelope_reach(envelope, xyz[:, :2]) - margin)
    inside = (np.hypot(rel[:, 0], rel[:, 1]) < limit) & (xyz[:, 2] < envelope["top"])
    if envelope["floor"] is not None:
        inside &= xyz[:, 2] > envelope["floor"] + margin
    return inside


class RigGuard:
    """What an empty set may hold between the cameras and the subject: only what they agree on.

    The space inside the ring of the cameras is empty in an empty set, but for objects that stand
    there. Those have a texture that at least three cameras find at the same depth (``support``).
    Anything else there is a guess the images cannot contradict: on a plain wall a grey Gaussian
    in mid air looks right from every training camera, and from a camera between them it is a
    grey veil over the subject. ``stray`` tells those apart: in the rig, farther than ``reach``
    from every supporting point. Outside the rig (walls, floor, what is behind the cameras)
    nothing is removed: there a Gaussian at the wrong depth hides nothing.
    """

    def __init__(self, envelope: dict, support: np.ndarray, reach: float = 0.1, voxel: float = 0.05):
        from scipy import ndimage

        self.envelope, self.voxel = envelope, voxel
        self.support = np.asarray(support, dtype=np.float32).reshape(-1, 3)
        pts = self.support.astype(np.float64)
        k = int(np.ceil(reach / voxel))
        if len(pts) == 0:
            self.lo, self.grid = np.zeros(3), np.zeros((1, 1, 1), dtype=bool)
            return
        self.lo = pts.min(axis=0) - (k + 1) * voxel
        shape = np.floor((pts.max(axis=0) + (k + 1) * voxel - self.lo) / voxel).astype(int) + 1
        grid = np.zeros(shape, dtype=bool)
        grid[tuple(np.floor((pts - self.lo) / voxel).astype(int).T)] = True
        ball = np.linalg.norm(np.indices((2 * k + 1,) * 3) - k, axis=0) <= k + 1e-9
        self.grid = ndimage.binary_dilation(grid, structure=ball)

    def supported(self, xyz: np.ndarray) -> np.ndarray:
        idx = np.floor((np.asarray(xyz, dtype=np.float64) - self.lo) / self.voxel).astype(int)
        ok = ((idx >= 0) & (idx < np.array(self.grid.shape))).all(axis=1)
        out = np.zeros(len(idx), dtype=bool)
        out[ok] = self.grid[tuple(idx[ok].T)]
        return out

    def stray(self, xyz: np.ndarray) -> np.ndarray:
        return in_the_rig(self.envelope, xyz) & ~self.supported(xyz)

    def save(self, folder: Path) -> None:
        np.savez_compressed(Path(folder) / "rig_support.npz", support=self.support)

    @classmethod
    def load(cls, folder: Path, envelope: dict) -> "RigGuard":
        path = Path(folder) / "rig_support.npz"
        return cls(envelope, np.load(path)["support"] if path.exists() else np.zeros((0, 3)))


@torch.no_grad()
def sweep_points(rig: Rig, cams: list[str], plates: dict[str, np.ndarray], center: np.ndarray,
                 work_width: int = 480, depths: int = 768, height_step: float = 0.01, per_camera: int = 60_000,
                 neighbours: int = 2, max_cost: float = 0.5, device: str | torch.device = "cpu",
                 seed: int = 0, log=print, details: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Points of the empty set with their colours, from plane sweeps between neighbouring cameras.

    Where the plates agree on a depth (textured surfaces), and a second camera
    finds the same surface there, the point is put on it. Elsewhere (plain
    walls, anything only one camera sees) it is put on the envelope of the
    set: its floor, or as far as the sure points reach on that side. Points
    between the cameras and the subject are kept only near what three cameras
    agree on (see ``RigGuard``). ``details``, if given, receives ``sure`` (for
    every point, whether two cameras agreed on it), ``envelope`` and ``guard``.
    """
    device = torch.device(device)
    views = {c: _SweepCamera(rig.cameras[c], plates[c], min(work_width, rig.cameras[c].width), device) for c in cams}
    center = np.asarray(center, dtype=np.float64)
    reach = float(np.mean([np.linalg.norm(rig.cameras[c].center - center) for c in cams]))
    up = torch.tensor([0.0, 0.0, 1.0], device=device)
    rng = np.random.default_rng(seed)
    found = {}
    for cam in cams:
        ref = views[cam]
        near = sorted((c for c in cams if c != cam),
                      key=lambda c: float(np.linalg.norm(rig.cameras[c].center - rig.cameras[cam].center)))
        others = [views[c] for c in near[:neighbours]]
        # Planes facing the camera, evenly spaced in inverse depth, and horizontal planes one
        # centimetre apart: between cameras far from each other a coarser step misses the match.
        inv = torch.linspace(1.0 / (8.0 * reach), 1.0 / (0.25 * reach), depths, device=device)
        facing = ref.forward[None].expand(depths, -1)
        facing_off = ref.forward @ ref.center + 1.0 / inv
        level_off = torch.arange(float(center[2] - 0.9 * reach), float(center[2] + 0.6 * reach), height_step,
                                 device=device)
        level = up[None].expand(len(level_off), -1)
        depth, cost = _sweep(ref, others, torch.cat([facing, level]), torch.cat([facing_off, level_off]))
        # Too flat a neighbourhood has no depth to speak of, whatever the correlation says.
        mean = _box(ref.grey, 7)
        texture = (_box(ref.grey * ref.grey, 7) - mean * mean)[0, 0]
        sure = (cost < max_cost) & (texture > 1e-4) & (depth > 0.05) & (depth < 12.0 * reach)
        found[cam] = (depth, sure)
    # A depth counts only if another camera, looking at the same point, found a surface there too.
    agreed_on = {}
    for cam in cams:
        ref = views[cam]
        depth, sure = found[cam]
        points = ref.center + depth.clamp(min=0.05)[..., None] * ref.rays
        votes = torch.zeros_like(depth)
        for other in cams:
            if other == cam:
                continue
            view = views[other]
            grid, inside = view.project(points)
            other_depth, other_sure = found[other]
            seen = F.grid_sample(other_depth[None, None], grid[None], mode="nearest", align_corners=True)[0, 0]
            ok = F.grid_sample(other_sure[None, None].float(), grid[None], mode="nearest", align_corners=True)[0, 0] > 0.5
            z = (points @ view.R.T + view.t)[..., 2]
            votes += (inside & ok & ((seen - z).abs() < 0.03 * z)).float()
        agreed_on[cam] = ((sure & (votes >= 1)).cpu().numpy(), points.cpu().numpy(),
                          (sure & (votes >= 2)).cpu().numpy())
    centers = np.array([rig.cameras[c].center for c in cams], dtype=np.float64)
    envelope = envelope_of(np.concatenate([p[m] for m, p, _ in agreed_on.values()]), center, centers)
    strong = np.concatenate([p[m] for _, p, m in agreed_on.values()])
    guard = RigGuard(envelope, strong[in_the_rig(envelope, strong)])
    all_xyz, all_rgb, all_sure, shares = [], [], [], []
    for cam in cams:
        ref = views[cam]
        sure = agreed_on[cam][0]
        shares.append(float(sure.mean()))
        outside = _on_envelope(ref.rays.cpu().numpy(), centers[cams.index(cam)], envelope, 12.0 * reach)
        depth = np.where(sure, found[cam][0].cpu().numpy(), outside)
        weight = np.where(sure, 1.0, 0.35).ravel()
        count = min(per_camera, weight.size)
        pick = rng.choice(weight.size, size=count, replace=False, p=weight / weight.sum())
        rays = ref.rays.cpu().numpy().reshape(-1, 3)[pick]
        all_xyz.append(np.asarray(rig.cameras[cam].center, dtype=np.float64) + depth.ravel()[pick, None] * rays)
        all_rgb.append(ref.rgb.reshape(-1, 3)[pick])
        all_sure.append(sure.ravel()[pick])
    xyz, rgb = np.concatenate(all_xyz), np.concatenate(all_rgb)
    # The cameras overlap: one point per cell of a fine grid is enough.
    cell = max(0.01, reach / 400.0)
    _, first = np.unique(np.round(xyz / cell).astype(np.int64), axis=0, return_index=True)
    first.sort()
    xyz, rgb, flags = xyz[first], rgb[first], np.concatenate(all_sure)[first]
    stray = guard.stray(xyz)
    xyz, rgb, flags = xyz[~stray], rgb[~stray], flags[~stray]
    if details is not None:
        details["sure"], details["envelope"], details["guard"] = flags, envelope, guard
    floor = "no floor found" if envelope["floor"] is None else f"floor at {envelope['floor']:.2f} m"
    log(f"[scenery] plane sweep on {len(cams)} cameras: two cameras agree on the depth of "
        f"{np.mean(shares):.0%} of the pixels; {floor}, the set reaches "
        f"{min(envelope['reach']):.1f}-{max(envelope['reach']):.1f} m from its centre; "
        f"{len(guard.support)} points between the cameras that three agree on; {len(xyz)} starting points")
    return xyz.astype(np.float32), rgb.astype(np.float32)


def prepare_scenery(data_dir: str | Path, exclude: list[str] | None = None, points: int = 400_000,
                    work_width: int = 480, device: str | torch.device | None = None, log=print) -> Path:
    """Write ``<data>/scenery``: the plates as a one-frame dataset, with starting points.

    The cameras in ``exclude`` keep their plate (to measure the result on it)
    but give no points: they are the ones left out of the training.
    """
    root = Path(data_dir)
    meta = read_meta(root)
    rig = Rig.load(root / "calibration.json")
    cams = [c for c in meta["cameras"] if c not in (exclude or [])]
    if len(cams) < 2:
        raise RuntimeError("the scenery needs at least two cameras")
    out = root / "scenery"
    if out.exists():
        shutil.rmtree(out)
    plates = {}
    for cam in meta["cameras"]:
        path = _plate_path(root, cam)
        if path is None:
            raise FileNotFoundError(f"no plate for {cam} in {root / 'background'}")
        plates[cam] = cv2.imread(str(path), cv2.IMREAD_COLOR)
        (out / "images" / cam).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "images" / cam / f"{frame_name(0)}.png"), plates[cam])
    shutil.copy(root / "calibration.json", out / "calibration.json")

    vol = rig.volume or {"center": [0.0, 0.0, 0.0]}
    center = (np.array(meta["bbox_min"]) + np.array(meta["bbox_max"])) / 2 if "bbox_min" in meta \
        else np.array(vol["center"], dtype=np.float64)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    details = {}
    xyz, rgb = sweep_points(rig, cams, plates, center, work_width=work_width,
                            per_camera=max(2000, points // len(cams)), device=device, log=log, details=details)
    write_points_ply(out / "init_points.ply", xyz, rgb, np.zeros(len(xyz), dtype=np.int32))
    details["guard"].save(out)
    # The box of the set, centred where the take is: learning rates and the smallest size of a
    # Gaussian are measured from there in both trainings.
    half = np.percentile(np.abs(xyz - center), 98.0, axis=0) + 0.25
    write_meta(out, meta["fps"], [0], meta["cameras"],
               {"source": "empty set", "plates": meta.get("plates", "recorded"), "init_cameras": cams,
                "envelope": details["envelope"],
                "bbox_min": (center - half).tolist(), "bbox_max": (center + half).tolist()})
    log(f"[scenery] the set spans {np.round(2 * half, 1).tolist()} m -> {out}")
    return out
