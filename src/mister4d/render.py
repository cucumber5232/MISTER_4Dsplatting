"""Novel-view video: a virtual camera moving around the subject while time plays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .camera import Camera, look_at
from .model import load_checkpoint, render_scene
from .rasterizer import resolve_backend
from .trainer import pick_device
from .video import VideoWriter


def orbit_path(cameras: list[Camera], center: np.ndarray, num: int, margin_deg: float = 10.0):
    """Camera poses sweeping the part of the circle that the rig actually covers.

    With cameras all around the subject the path is a full turn; with cameras
    on an arc it goes back and forth over that arc (the far side was never
    observed and would only show artefacts).
    """
    offsets = np.array([c.center - center for c in cameras])
    az = np.sort(np.arctan2(offsets[:, 1], offsets[:, 0]))
    radius = float(np.mean(np.linalg.norm(offsets[:, :2], axis=1)))
    height = float(np.mean(offsets[:, 2]))
    gaps = np.diff(np.concatenate([az, [az[0] + 2 * np.pi]]))
    k = int(np.argmax(gaps))
    u = np.arange(num) / num
    if gaps[k] < np.radians(120.0):
        angles = az[0] + 2 * np.pi * u
    else:
        start = az[(k + 1) % len(az)] - np.radians(margin_deg)
        span = 2 * np.pi - gaps[k] + 2 * np.radians(margin_deg)
        angles = start + span * (0.5 - 0.5 * np.cos(2 * np.pi * u))
    poses = []
    for a in angles:
        eye = center + np.array([radius * np.cos(a), radius * np.sin(a), height])
        poses.append(look_at(eye, center))
    return poses


def ping_pong(i: int, n: int) -> int:
    """0, 1, ..., n-1, n-2, ..., 1, 0, 1, ... so that looping motion has no jump."""
    if n <= 1:
        return 0
    period = 2 * (n - 1)
    k = i % period
    return k if k < n else period - k


@torch.no_grad()
def render_orbit(run_dir: str | Path, out_path: str | Path, seconds: float = 8.0,
                 fps: float | None = None, scale: float = 1.0, background=(0.08, 0.08, 0.1),
                 fixed_camera: str | None = None, device: str = "auto", log=print) -> Path:
    dev = pick_device(device)
    scene, ckpt = load_checkpoint(Path(run_dir) / "ckpt.pt", dev)
    meta = ckpt["meta"]
    backend = resolve_backend(ckpt["config"].get("backend", "auto"), dev)
    cams = [Camera.from_dict(n, meta["cameras"][n]) for n in meta["train_cameras"]]
    ref = cams[0]
    if fixed_camera is not None:
        ref = Camera.from_dict(fixed_camera, meta["cameras"][fixed_camera])
    fps = fps or meta["fps"]
    total = max(1, int(round(seconds * fps)))
    num_frames = len(meta["frame_ids"])
    width, height = int(round(ref.width * scale)), int(round(ref.height * scale))
    K = ref.K.copy()
    K[0] *= width / ref.width
    K[1] *= height / ref.height
    K_t = torch.tensor(K, dtype=torch.float32, device=dev)[None]
    center = np.array(meta["center"])
    poses = [(ref.R, ref.t)] * total if fixed_camera else orbit_path(cams, center, total)
    bg = torch.tensor(background, dtype=torch.float32, device=dev)

    with VideoWriter(out_path, width, height, fps) as writer:
        for i, (R, t) in enumerate(poses):
            view = np.eye(4)
            view[:3, :3], view[:3, 3] = R, t
            viewmat = torch.tensor(view, dtype=torch.float32, device=dev)[None]
            frame = ping_pong(int(round(i * meta["fps"] / fps)), num_frames)
            rgb, alpha, *_ = render_scene(scene, frame, viewmat, K_t, width, height, backend=backend)
            img = (rgb[0] + (1.0 - alpha[0]) * bg).clamp(0, 1)
            writer.write((img * 255 + 0.5).to(torch.uint8).cpu().numpy())
    log(f"[render] {total} frames at {width}x{height} -> {out_path}")
    return Path(out_path)
