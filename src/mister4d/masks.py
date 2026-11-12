"""Foreground masks by comparison with the empty set.

Built in and dependency-free: every frame is compared with the clean plate of
the same camera recorded without the subject. It needs what the capture guide
asks for anyway (fixed cameras, locked exposure, stable light, matte backdrop).
Masks made with any other tool can be dropped into ``masks/<cam>/`` instead.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .dataset import frame_name, read_meta


def foreground_mask(image: np.ndarray, plate: np.ndarray, threshold: float = 24.0,
                    suppress_shadows: bool = True, min_area: float = 5e-4,
                    feather: float = 0.7) -> np.ndarray:
    """uint8 mask (255 = subject) of a BGR frame against its BGR clean plate."""
    h, w = image.shape[:2]
    blur = 5 if min(h, w) >= 400 else 3  # just enough to tame sensor noise
    a = cv2.GaussianBlur(image, (blur, blur), 0).astype(np.float32)
    b = cv2.GaussianBlur(plate, (blur, blur), 0).astype(np.float32)
    diff = np.abs(a - b).max(axis=2)
    fg = diff > threshold
    if suppress_shadows:
        # A shadow darkens the backdrop without changing its hue much.
        ratio = (a * b).sum(axis=2) / np.maximum((b * b).sum(axis=2), 1.0)
        na = a / np.maximum(np.linalg.norm(a, axis=2, keepdims=True), 1.0)
        nb = b / np.maximum(np.linalg.norm(b, axis=2, keepdims=True), 1.0)
        hue_shift = np.linalg.norm(na - nb, axis=2)
        shadow = (ratio > 0.45) & (ratio < 0.98) & (hue_shift < 0.05) & (diff < 3.0 * threshold)
        fg &= ~shadow
    # Clean-up radius grows with the resolution; on tiny images it is skipped,
    # there a limb is only a couple of pixels wide.
    k = int(round(min(h, w) / 240)) | 1
    mask = fg.astype(np.uint8)
    if k > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area * h * w
    mask = keep[labels].astype(np.uint8) * 255
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
    return mask


def compute_masks(data_dir: str | Path, threshold: float = 24.0, suppress_shadows: bool = True,
                  log=print) -> None:
    root = Path(data_dir)
    meta = read_meta(root)
    previews = []
    for cam in meta["cameras"]:
        plate_path = next((p for p in (root / "background").glob(f"{cam}.*")), None) \
            if (root / "background").is_dir() else None
        if plate_path is None:
            raise FileNotFoundError(
                f"no clean plate for {cam} in {root / 'background'}: record the empty set and pass "
                "--empty to 'mister4d prepare', or provide your own masks in masks/<cam>/")
        plate = cv2.imread(str(plate_path), cv2.IMREAD_COLOR)
        (root / "masks" / cam).mkdir(parents=True, exist_ok=True)
        coverage = []
        row = []
        ids = meta["frame_ids"]
        for i, fid in enumerate(ids):
            img_path = next((root / "images" / cam).glob(f"{frame_name(fid)}.*"))
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            mask = foreground_mask(img, plate, threshold, suppress_shadows)
            cv2.imwrite(str(root / "masks" / cam / f"{frame_name(fid)}.png"), mask)
            coverage.append(float((mask > 127).mean()))
            if i in (0, len(ids) // 2, len(ids) - 1):
                tint = img.copy()
                tint[mask <= 127] = (tint[mask <= 127] * 0.25).astype(np.uint8)
                scale = 360.0 / tint.shape[1]
                row.append(cv2.resize(tint, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
        previews.append(np.concatenate(row, axis=1))
        log(f"[masks] {cam}: subject covers {np.mean(coverage):.1%} of the image "
            f"(min {np.min(coverage):.1%}, max {np.max(coverage):.1%})")
        if np.min(coverage) < 0.002:
            log(f"[masks] WARNING {cam}: almost empty mask in some frames; lower --threshold")
        if np.max(coverage) > 0.6:
            log(f"[masks] WARNING {cam}: mask covers most of the image; the light or the camera "
                "changed after the empty set was recorded, or --threshold is too low")
    width = min(p.shape[1] for p in previews)
    cv2.imwrite(str(root / "masks_preview.jpg"), np.concatenate([p[:, :width] for p in previews], axis=0))
    log(f"[masks] preview written to {root / 'masks_preview.jpg'} (check it before training)")
