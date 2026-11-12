"""Initial point clouds from the visual hull of the masks, frame by frame.

With calibrated cameras and foreground masks, at every instant the subject
lies inside the intersection of the silhouette cones. The surface of that
intersection is where the Gaussians of that instant are born: every frame
gets its own starting geometry, so nothing has to be tracked through time.

The cameras are fixed, so the projection of every voxel is computed once and
each frame only costs a mask lookup.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage

from .camera import Rig
from .dataset import frame_name, read_meta, update_meta, write_points_ply


class _Grid:
    """A voxel grid with the pixel of every voxel in every camera precomputed."""

    def __init__(self, rig: Rig, cams: list[str], lo: np.ndarray, hi: np.ndarray, voxel: float):
        self.voxel = voxel
        self.axes = [np.arange(lo[i] + voxel / 2, hi[i], voxel) for i in range(3)]
        self.shape = tuple(len(a) for a in self.axes)
        grid = np.stack(np.meshgrid(*self.axes, indexing="ij"), axis=-1).reshape(-1, 3)
        self.lookup = {}
        for cam in cams:
            c = rig.cameras[cam]
            uv, depth = c.project(grid)
            u, v = np.round(uv[:, 0]).astype(np.int64), np.round(uv[:, 1]).astype(np.int64)
            ok = (depth > 0) & (u >= 0) & (u < c.width) & (v >= 0) & (v < c.height)
            self.lookup[cam] = (np.nonzero(ok)[0], v[ok], u[ok])
        self.seen = np.zeros(len(grid), dtype=np.uint8)
        for idx, _, _ in self.lookup.values():
            self.seen[idx] += 1

    def occupancy(self, masks: dict[str, np.ndarray], min_views: int) -> np.ndarray:
        """Voxels inside the silhouette of every camera that sees them."""
        inside = np.zeros(len(self.seen), dtype=np.uint8)
        for cam, (idx, v, u) in self.lookup.items():
            inside[idx] += masks[cam][v, u] > 127
        return ((inside == self.seen) & (self.seen >= min_views)).reshape(self.shape)

    def centers(self, occupied: np.ndarray) -> np.ndarray:
        ix, iy, iz = np.nonzero(occupied)
        return np.stack([self.axes[0][ix], self.axes[1][iy], self.axes[2][iz]], axis=-1)

    def bounds(self, occupied: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = self.centers(occupied)
        return pts.min(axis=0), pts.max(axis=0)


def _load_masks(root: Path, cams: list[str], frame_id: int) -> dict[str, np.ndarray]:
    return {cam: cv2.imread(str(root / "masks" / cam / f"{frame_name(frame_id)}.png"), cv2.IMREAD_GRAYSCALE)
            for cam in cams}


def visual_hull(data_dir: str | Path, voxel: float = 0.015, frame_step: int = 1,
                exclude: list[str] | None = None, min_views: int | None = None,
                total_points: int = 600_000, max_points_per_frame: int = 15_000,
                center=None, size=None, seed: int = 0, log=print) -> Path:
    """Write ``init_points.ply``: surface points of the visual hull of every ``frame_step``-th frame."""
    root = Path(data_dir)
    meta = read_meta(root)
    rig = Rig.load(root / "calibration.json")
    cams = [c for c in meta["cameras"] if c not in (exclude or [])]
    if len(cams) < 2:
        raise RuntimeError("the visual hull needs at least two cameras")
    if not (root / "masks" / cams[0]).is_dir():
        raise FileNotFoundError(f"no masks in {root / 'masks'}: run 'mister4d masks' first")
    ids = meta["frame_ids"]
    vol = rig.volume or {"center": [0, 0, 0], "size": [3, 3, 3]}
    center = np.array(center if center is not None else vol["center"], dtype=np.float64)
    size = np.array(size if size is not None else vol["size"], dtype=np.float64)
    min_views = min(len(cams), max(2, len(cams) - 1)) if min_views is None else min_views
    frames = list(range(0, len(ids), max(frame_step, 1)))

    # Pass 1, coarse, whole capture volume: where is the subject during the take?
    coarse = _Grid(rig, cams, center - size / 2, center + size / 2, max(voxel * 2.5, float(size.max()) / 120))
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    probes = sorted({frames[i] for i in np.linspace(0, len(frames) - 1, min(len(frames), 12)).round().astype(int)})
    for fi in probes:
        occ = coarse.occupancy(_load_masks(root, cams, ids[fi]), min_views)
        if occ.any():
            a, b = coarse.bounds(occ)
            lo, hi = np.minimum(lo, a), np.maximum(hi, b)
    if not np.isfinite(lo).all():
        raise RuntimeError(
            "empty visual hull: the masks do not intersect inside the capture volume. Check the "
            "calibration, the masks preview and the volume (--center / --size)")
    edge = center - size / 2, center + size / 2
    touching = [n for i, n in enumerate("xyz")
                if lo[i] - coarse.voxel < edge[0][i] or hi[i] + coarse.voxel > edge[1][i]]
    if touching:
        log(f"[init] WARNING: the hull touches the border of the capture volume along {touching}: "
            "with few cameras the silhouette cones are open-ended; enlarge or move the volume "
            "(--center / --size) if the subject is cut")
    margin = 0.25
    lo, hi = np.maximum(lo - margin, edge[0]), np.minimum(hi + margin, edge[1])

    # Pass 2, fine, only around the subject, every selected frame.
    fine = _Grid(rig, cams, lo, hi, voxel)
    per_frame = int(np.clip(total_points // len(frames), 500, max_points_per_frame))
    rng = np.random.default_rng(seed)
    all_xyz, all_rgb, all_frames = [], [], []
    box_lo, box_hi = np.full(3, np.inf), np.full(3, -np.inf)
    empty = []
    for fi in frames:
        occupied = fine.occupancy(_load_masks(root, cams, ids[fi]), min_views)
        if not occupied.any():
            empty.append(ids[fi])
            continue
        a, b = fine.bounds(occupied)
        box_lo, box_hi = np.minimum(box_lo, a), np.maximum(box_hi, b)
        shell = occupied & ~ndimage.binary_erosion(occupied)
        xyz = fine.centers(shell if shell.sum() >= 200 else occupied)
        if len(xyz) > per_frame:
            xyz = xyz[rng.choice(len(xyz), per_frame, replace=False)]
        xyz = xyz + rng.uniform(-voxel / 2, voxel / 2, xyz.shape)

        # Colour: average of the cameras whose silhouette contains the point.
        color_sum, weight = np.zeros((len(xyz), 3)), np.zeros(len(xyz))
        for cam in cams:
            c = rig.cameras[cam]
            img = cv2.imread(str(next((root / "images" / cam).glob(f"{frame_name(ids[fi])}.*"))), cv2.IMREAD_COLOR)
            uv, depth = c.project(xyz)
            u, v = np.round(uv[:, 0]).astype(np.int64), np.round(uv[:, 1]).astype(np.int64)
            ok = (depth > 0) & (u >= 0) & (u < c.width) & (v >= 0) & (v < c.height)
            color_sum[ok] += img[v[ok], u[ok], ::-1] / 255.0
            weight[ok] += 1
        all_xyz.append(xyz)
        all_rgb.append(color_sum / np.maximum(weight, 1)[:, None])
        all_frames.append(np.full(len(xyz), fi, dtype=np.int32))

    if not all_xyz:
        raise RuntimeError("empty visual hull in every frame: check masks and calibration")
    if empty:
        log(f"[init] WARNING: empty hull in {len(empty)} frame(s) (first: {empty[:5]}): "
            "check the masks of those frames")
    xyz, rgb, frame_idx = np.concatenate(all_xyz), np.concatenate(all_rgb), np.concatenate(all_frames)
    out = root / "init_points.ply"
    write_points_ply(out, xyz, rgb, frame_idx)
    update_meta(root, bbox_min=(box_lo - 0.10).tolist(), bbox_max=(box_hi + 0.10).tolist(),
                init_frame_step=int(max(frame_step, 1)), init_cameras=cams)
    log(f"[init] visual hull from {len(cams)} cameras, {len(all_xyz)} frames, voxel {voxel * 100:.1f} cm: "
        f"{len(xyz)} points ({len(xyz) // len(all_xyz)} per frame), "
        f"subject box {np.round(box_hi - box_lo, 2).tolist()} m -> {out}")
    return out
