"""Video of the result: a virtual camera around the subject, a camera of the rig, or a path of your own.

Besides the short preview of the demo, this is the output meant for editing:
any resolution and frame rate, the whole take played once from start to end,
and a transparent background, as ProRes 4444 (``.mov``) or as a sequence of
PNG images (a folder), to lay the subject over other footage.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .camera import Camera, look_at
from .model import render_scene
from .rasterizer import resolve_backend
from .run import Run
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


def load_camera_path(path: str | Path) -> list[dict]:
    """Key views of a camera move: ``[{"time": seconds, "eye": [x, y, z], "target": [x, y, z], "fov": degrees}]``.

    ``time`` is the moment of the video at which the camera is there; the
    player copies a view in this form ("Copia la vista").
    """
    keys = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(keys, dict):
        keys = keys.get("keys", [keys])
    if not keys:
        raise ValueError(f"{path}: no key views")
    out = []
    for i, key in enumerate(keys):
        try:
            out.append({"time": float(key.get("time", i)), "eye": np.array(key["eye"], dtype=np.float64),
                        "target": np.array(key["target"], dtype=np.float64), "fov": float(key.get("fov", 40.0))})
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: key view {i} needs 'eye' and 'target' ([x, y, z])") from exc
    return sorted(out, key=lambda k: k["time"])


def path_view(keys: list[dict], seconds: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Eye, target and vertical field of view at ``seconds``: eased between the two key views around it."""
    if seconds <= keys[0]["time"] or len(keys) == 1:
        k = keys[0]
        return k["eye"], k["target"], k["fov"]
    if seconds >= keys[-1]["time"]:
        k = keys[-1]
        return k["eye"], k["target"], k["fov"]
    j = next(i for i in range(1, len(keys)) if keys[i]["time"] >= seconds)
    a, b = keys[j - 1], keys[j]
    u = (seconds - a["time"]) / max(b["time"] - a["time"], 1e-9)
    u = u * u * (3.0 - 2.0 * u)  # no jerk at the key views
    return (a["eye"] + (b["eye"] - a["eye"]) * u, a["target"] + (b["target"] - a["target"]) * u,
            a["fov"] + (b["fov"] - a["fov"]) * u)


class ImageSequenceWriter:
    """Frames as numbered PNG files in a folder (RGBA when the background is transparent)."""

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self._count = 0

    def write(self, frame: np.ndarray) -> None:
        code = cv2.COLOR_RGBA2BGRA if frame.shape[2] == 4 else cv2.COLOR_RGB2BGR
        cv2.imwrite(str(self.folder / f"frame_{self._count:06d}.png"), cv2.cvtColor(frame, code))
        self._count += 1

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


@torch.no_grad()
def render_orbit(run_dir: str | Path, out_path: str | Path, seconds: float = 8.0,
                 fps: float | None = None, scale: float = 1.0, background=(0.08, 0.08, 0.1),
                 fixed_camera: str | None = None, device: str = "auto", log=print, *,
                 take: bool = False, speed: float = 1.0, transparent: bool = False,
                 width: int | None = None, height: int | None = None,
                 path: str | Path | None = None) -> Path:
    """Render a video of a trained take.

    The camera orbits the subject, or stands where ``fixed_camera`` of the rig
    stood, or follows the key views of ``path``. By default the take is played
    back and forth for ``seconds``; with ``take`` it is played once, start to
    end (``speed`` 0.5 = half speed). The output is an ``.mp4``, a ProRes
    ``.mov``, or a folder of PNG images; the last two can be ``transparent``.
    """
    dev = pick_device(device)
    run = Run(run_dir, dev)
    meta = run.meta
    backend = resolve_backend(run.config.get("backend", "auto"), dev)
    antialiased = bool(run.config.get("antialiased", False))
    cams = [Camera.from_dict(n, meta["cameras"][n]) for n in meta["train_cameras"]]
    ref = cams[0]
    if fixed_camera is not None:
        if fixed_camera not in meta["cameras"]:
            raise ValueError(f"no camera '{fixed_camera}' in this take (have {sorted(meta['cameras'])})")
        ref = Camera.from_dict(fixed_camera, meta["cameras"][fixed_camera])
    keys = load_camera_path(path) if path is not None else None

    fps = fps or meta["fps"]
    num_frames = 1 if run.static_frame is not None else run.num_frames
    rate = meta["fps"] / fps * speed  # frames of the take per frame of the video
    if take and num_frames > 1:
        total = max(1, int(np.ceil((num_frames - 1) / rate)) + 1)
    else:
        total = max(1, int(round(seconds * fps)))

    # Image size: the reference camera scaled, or what was asked for.
    if height is None and width is None:
        out_w, out_h = int(round(ref.width * scale)), int(round(ref.height * scale))
    else:
        out_h = int(height if height is not None else round(width * ref.height / ref.width))
        out_w = int(width if width is not None else round(out_h * ref.width / ref.height))
    zoom = out_h / ref.height
    K = np.array([[ref.K[0, 0] * zoom, 0.0, ref.K[0, 2] * out_w / ref.width],
                  [0.0, ref.K[1, 1] * zoom, ref.K[1, 2] * zoom], [0.0, 0.0, 1.0]])
    center = np.array(meta["center"])
    orbit = None if (fixed_camera or keys) else orbit_path(cams, center, total)
    bg = torch.tensor(background, dtype=torch.float32, device=dev)

    out_path = Path(out_path)
    if out_path.suffix == "":
        writer = ImageSequenceWriter(out_path)
    else:
        writer = VideoWriter(out_path, out_w, out_h, fps, alpha=transparent)
    with writer:
        for i in range(total):
            if keys is not None:
                eye, target, fov = path_view(keys, i / fps)
                R, t = look_at(eye, target)
                f = (out_h / 2.0) / np.tan(np.radians(fov) / 2.0)
                K = np.array([[f, 0.0, (out_w - 1) / 2.0], [0.0, f, (out_h - 1) / 2.0], [0.0, 0.0, 1.0]])
            elif orbit is not None:
                R, t = orbit[i]
            else:
                R, t = ref.R, ref.t
            view = np.eye(4)
            view[:3, :3], view[:3, 3] = R, t
            viewmat = torch.tensor(view, dtype=torch.float32, device=dev)[None]
            K_t = torch.tensor(K, dtype=torch.float32, device=dev)[None]
            if run.static_frame is not None:
                frame = run.static_frame
            elif take:
                frame = min(int(round(i * rate)), num_frames - 1)
            else:
                frame = ping_pong(int(round(i * rate)), num_frames)
            # Whole frames: the instants the scene was trained on.
            rgb, alpha, *_ = render_scene(run.scene_at(frame), frame, viewmat, K_t, out_w, out_h,
                                          backend=backend, antialiased=antialiased)
            if transparent:  # colour not multiplied by alpha, as editing software expects
                img = torch.cat([rgb[0] / alpha[0].clamp(min=1e-4), alpha[0]], dim=-1).clamp(0, 1)
            else:
                img = (rgb[0] + (1.0 - alpha[0]) * bg).clamp(0, 1)
            writer.write((img * 255 + 0.5).to(torch.uint8).cpu().numpy())
    what = "PNG images" if out_path.suffix == "" else "frames"
    log(f"[render] {total} {what} at {out_w}x{out_h}, {fps:g} fps"
        f"{', transparent background' if transparent else ''} -> {out_path}")
    return out_path
