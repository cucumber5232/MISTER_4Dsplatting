"""Prepared multi-view dataset: what the trainer consumes.

Layout (produced by ``mister4d prepare`` or by ``mister4d demo``)::

    <data>/
      calibration.json            pinhole cameras (images already undistorted)
      meta.json                   fps, frame ids, camera names
      images/<cam>/<frame>.jpg    one folder per camera, same frame ids everywhere
      masks/<cam>/<frame>.png     optional foreground masks (255 = subject)
      background/<cam>.jpg        optional clean plate of the empty set
      init_points.ply             optional initial point cloud (one set of points per frame)
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

from .camera import Rig


def frame_name(frame_id: int) -> str:
    return f"{int(frame_id):06d}"


def write_meta(data_dir: str | Path, fps: float, frame_ids: list[int], cameras: list[str],
               extra: dict | None = None) -> None:
    meta = {"format": "mister4d-dataset", "version": 1, "fps": float(fps),
            "frame_ids": [int(i) for i in frame_ids], "cameras": list(cameras)}
    meta.update(extra or {})
    (Path(data_dir) / "meta.json").write_text(json.dumps(meta, indent=2))


def read_meta(data_dir: str | Path) -> dict:
    path = Path(data_dir) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run 'mister4d prepare' first")
    return json.loads(path.read_text())


def update_meta(data_dir: str | Path, **fields) -> dict:
    meta = read_meta(data_dir)
    meta.update(fields)
    (Path(data_dir) / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


class MultiViewDataset:
    def __init__(self, data_dir: str | Path, cache_gb: float = 8.0):
        self.root = Path(data_dir)
        self.meta = read_meta(self.root)
        self.rig = Rig.load(self.root / "calibration.json")
        if not all(c.is_pinhole for c in self.rig.cameras.values()):
            raise ValueError("the dataset calibration still has lens distortion: "
                             "run 'mister4d prepare' to undistort the images")
        self.cameras = [c for c in self.meta["cameras"] if c in self.rig.cameras]
        self.frame_ids: list[int] = self.meta["frame_ids"]
        self.fps = float(self.meta["fps"])
        self.has_masks = (self.root / "masks" / self.cameras[0]).is_dir()
        first = self.rig.cameras[self.cameras[0]]
        self.width, self.height = first.width, first.height
        for name in self.cameras:
            c = self.rig.cameras[name]
            if (c.width, c.height) != (self.width, self.height):
                raise ValueError("all cameras must have the same resolution after 'prepare'")
        item_bytes = self.width * self.height * (4 if self.has_masks else 3)
        self._capacity = max(8, int(cache_gb * 1e9 / item_bytes))
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()  # frames are decoded by several threads

    @property
    def num_frames(self) -> int:
        return len(self.frame_ids)

    def _find(self, kind: str, cam: str, frame_id: int) -> Path | None:
        for ext in ("jpg", "png", "jpeg"):
            p = self.root / kind / cam / f"{frame_name(frame_id)}.{ext}"
            if p.exists():
                return p
        return None

    def load(self, frame_index: int, cam: str) -> tuple[np.ndarray, np.ndarray | None]:
        """uint8 RGB image [H, W, 3] and uint8 mask [H, W] (or None)."""
        key = (frame_index, cam)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        frame_id = self.frame_ids[frame_index]
        path = self._find("images", cam, frame_id)
        if path is None:
            raise FileNotFoundError(f"missing image: camera {cam}, frame {frame_id}")
        img = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        mask = None
        if self.has_masks:
            mpath = self._find("masks", cam, frame_id)
            if mpath is None:
                raise FileNotFoundError(f"missing mask: camera {cam}, frame {frame_id}")
            mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if img.shape[:2] != (self.height, self.width):
            raise ValueError(f"{path}: image size {img.shape[1]}x{img.shape[0]} does not match "
                             f"the calibration ({self.width}x{self.height})")
        with self._lock:
            self._cache[key] = (img, mask)
            if len(self._cache) > self._capacity:
                self._cache.popitem(last=False)
        return img, mask

    def batch(self, frame_index: int, cams: list[str], device) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Float images [C, H, W, 3] and masks [C, H, W, 1] (or None) on ``device``."""
        imgs, masks = [], []
        for cam in cams:
            img, mask = self.load(frame_index, cam)
            imgs.append(torch.from_numpy(img))
            if mask is not None:
                masks.append(torch.from_numpy(mask))
        images = torch.stack(imgs).to(device).float() / 255.0
        mask_t = None
        if masks:
            mask_t = torch.stack(masks).to(device).float()[..., None] / 255.0
        return images, mask_t

    def camera_tensors(self, cams: list[str], device) -> tuple[torch.Tensor, torch.Tensor]:
        """World->camera matrices [C, 4, 4] and OpenCV intrinsics [C, 3, 3]."""
        viewmats = torch.tensor(np.stack([self.rig.cameras[c].world_to_cam() for c in cams]),
                                dtype=torch.float32, device=device)
        Ks = torch.tensor(np.stack([self.rig.cameras[c].K for c in cams]),
                          dtype=torch.float32, device=device)
        return viewmats, Ks

    def scene_bounds(self) -> tuple[np.ndarray, float]:
        """Centre of the capture volume and a characteristic radius (metres)."""
        vol = self.rig.volume or {}
        if "bbox_min" in self.meta:
            lo, hi = np.array(self.meta["bbox_min"]), np.array(self.meta["bbox_max"])
            center, size = (lo + hi) / 2, hi - lo
        elif vol:
            center, size = np.array(vol["center"], dtype=float), np.array(vol["size"], dtype=float)
        else:
            center, size = np.zeros(3), np.full(3, 2.0)
        return center, float(np.linalg.norm(size) / 2)


# ---------------------------------------------------------------------------
# Point cloud I/O (binary little-endian PLY: position, 8-bit colour, frame index)
# ---------------------------------------------------------------------------

_PLY_DTYPE = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                       ("red", "u1"), ("green", "u1"), ("blue", "u1"), ("frame", "<i4")])


def write_points_ply(path: str | Path, xyz: np.ndarray, rgb: np.ndarray, frames: np.ndarray) -> None:
    """``frames`` is the index (position in the take) at which each point was observed."""
    data = np.empty(len(xyz), dtype=_PLY_DTYPE)
    xyz = np.asarray(xyz, dtype=np.float32)
    rgb8 = np.clip(np.asarray(rgb) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    data["x"], data["y"], data["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    data["red"], data["green"], data["blue"] = rgb8[:, 0], rgb8[:, 1], rgb8[:, 2]
    data["frame"] = np.asarray(frames, dtype=np.int32)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(xyz)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "property int frame\n"
        "end_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())


def read_points_ply(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reads the files written by write_points_ply: xyz, rgb in 0..1, frame index."""
    raw = Path(path).read_bytes()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii")
    if "binary_little_endian" not in header or "property int frame" not in header:
        raise ValueError(f"{path}: not a mister4d point cloud (run 'mister4d init' again)")
    n = int(header.split("element vertex")[1].split()[0])
    data = np.frombuffer(raw, dtype=_PLY_DTYPE, count=n, offset=end)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=-1).astype(np.float32)
    rgb = np.stack([data["red"], data["green"], data["blue"]], axis=-1).astype(np.float32) / 255.0
    return xyz, rgb, data["frame"].astype(np.int32)
