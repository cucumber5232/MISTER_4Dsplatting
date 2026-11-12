"""Turn extracted frames + calibration into a training dataset.

Images are undistorted once, here, and the calibration written next to them is
a pure pinhole model flagged ``undistorted``: nothing downstream may correct
the lens again. Intrinsics are updated consistently with every resize.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .camera import Camera, Rig
from .dataset import frame_name, write_meta
from .ingest import read_ingest


def _frame_files(folder: Path) -> dict[int, Path]:
    return {int(p.stem): p for p in sorted(folder.iterdir())
            if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.stem.isdigit()}


class Rectifier:
    """Undistort + downscale for one camera, with the matching pinhole intrinsics."""

    def __init__(self, cam: Camera, downscale: float, already_undistorted: bool):
        w, h = cam.width, cam.height
        self.in_size = (w, h)
        self.maps = None
        K = cam.K.copy()
        if not already_undistorted and not cam.is_pinhole:
            K, _ = cv2.getOptimalNewCameraMatrix(cam.K, cam.dist, (w, h), 0)
            self.maps = cv2.initUndistortRectifyMap(cam.K, cam.dist, None, K, (w, h), cv2.CV_32FC1)
        self.out_size = (max(1, int(round(w / downscale))), max(1, int(round(h / downscale))))
        sx, sy = self.out_size[0] / w, self.out_size[1] / h
        # Pixel centres sit at integer coordinates: shift by half a pixel around the scaling.
        K = K.copy()
        K[0, 0] *= sx
        K[0, 1] *= sx
        K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5
        K[1, 1] *= sy
        K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
        self.camera = Camera(cam.name, self.out_size[0], self.out_size[1], K, np.zeros(5),
                             cam.R, cam.t, info=cam.info)

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if (img.shape[1], img.shape[0]) != self.in_size:
            raise ValueError(
                f"{self.camera.name}: frame is {img.shape[1]}x{img.shape[0]} but the calibration "
                f"was computed at {self.in_size[0]}x{self.in_size[1]}; calibrate at the recording "
                "resolution and with the same zoom/focus")
        if self.maps is not None:
            img = cv2.remap(img, self.maps[0], self.maps[1], cv2.INTER_CUBIC)
        if (img.shape[1], img.shape[0]) != self.out_size:
            img = cv2.resize(img, self.out_size, interpolation=cv2.INTER_AREA)
        return img


def prepare_dataset(take_dir: str | Path, calibration: str | Path, out_dir: str | Path,
                    empty_dir: str | Path | None = None, downscale: float = 1.0,
                    max_frames: int | None = None, ext: str = "jpg", log=print) -> Path:
    take_dir, out = Path(take_dir), Path(out_dir)
    manifest = read_ingest(take_dir)
    rig = Rig.load(calibration)
    cams = list(manifest["cameras"])
    not_calibrated = [c for c in cams if c not in rig.cameras]
    if not_calibrated:
        raise RuntimeError(f"cameras {not_calibrated} are not in the calibration file "
                           f"(calibrated: {rig.names}); file names must match between clips")
    frame_ids = manifest["frame_ids"]
    if max_frames and len(frame_ids) > max_frames:
        frame_ids = frame_ids[:max_frames]

    out_cams = {}
    params = [cv2.IMWRITE_JPEG_QUALITY, 97] if ext == "jpg" else []
    for cam in cams:
        rect = Rectifier(rig.cameras[cam], downscale, rig.undistorted)
        out_cams[cam] = rect.camera
        files = _frame_files(take_dir / "frames" / cam)
        (out / "images" / cam).mkdir(parents=True, exist_ok=True)
        for fid in frame_ids:
            if fid not in files:
                raise RuntimeError(f"{cam}: frame {fid} listed in ingest.json but missing on disk")
            img = rect(cv2.imread(str(files[fid]), cv2.IMREAD_COLOR))
            cv2.imwrite(str(out / "images" / cam / f"{frame_name(fid)}.{ext}"), img, params)
        if empty_dir is not None:
            empty = _frame_files(Path(empty_dir) / "frames" / cam)
            if not empty:
                raise RuntimeError(f"no empty-set frames for camera {cam} in {empty_dir}")
            pick = list(empty.values())
            pick = [pick[i] for i in np.linspace(0, len(pick) - 1, min(len(pick), 15)).round().astype(int)]
            plate = np.median(np.stack([cv2.imread(str(p), cv2.IMREAD_COLOR) for p in pick]), axis=0)
            (out / "background").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "background" / f"{cam}.png"), rect(plate.astype(np.uint8)))
        log(f"[prepare] {cam}: {len(frame_ids)} frames -> {rect.out_size[0]}x{rect.out_size[1]}"
            f"{' (undistorted)' if rect.maps is not None else ''}")

    Rig(out_cams, volume=rig.volume, undistorted=True, report=rig.report).save(out / "calibration.json")
    write_meta(out, manifest["fps"] / max(manifest.get("step", 1), 1), frame_ids, cams,
               {"source": str(take_dir), "downscale": downscale})
    return out
