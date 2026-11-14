"""Quality check against cameras that were not used for training.

The capture guide asks to judge the result from an independent point of view.
With a small rig that means leaving one camera out of the training
(``holdout``): the numbers of that camera are the honest ones, at the price of
a reduced coverage, which the report states explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .dataset import MultiViewDataset
from .losses import psnr, ssim
from .model import render_scene
from .rasterizer import resolve_backend
from .run import Run
from .trainer import apply_camera_colors, pick_device


METRICS = ("psnr", "psnr_subject", "ssim", "iou", "flicker", "flicker_at_cuts")


def _crop_box(region: torch.Tensor, margin: int = 8) -> tuple[int, int, int, int] | None:
    ys, xs = torch.nonzero(region, as_tuple=True)
    if ys.numel() == 0:
        return None
    h, w = region.shape
    return (max(int(ys.min()) - margin, 0), min(int(ys.max()) + margin + 1, h),
            max(int(xs.min()) - margin, 0), min(int(xs.max()) + margin + 1, w))


@torch.no_grad()
def evaluate_run(run_dir: str | Path, data_dir: str | Path | None = None, frame_step: int = 1,
                 num_images: int = 4, device: str = "auto", log=print) -> dict:
    run = Path(run_dir)
    dev = pick_device(device)
    take = Run(run, dev)
    meta = take.meta
    backend = resolve_backend(take.config.get("backend", "auto"), dev)
    antialiased = bool(take.config.get("antialiased", False))
    ds = MultiViewDataset(data_dir or meta["data_dir"])
    out = run / "eval"
    out.mkdir(exist_ok=True)
    frames = list(range(0, ds.num_frames, max(frame_step, 1)))
    if meta.get("static_frame") is not None:
        frames = [meta["static_frame"]]  # a static scene only describes that instant
    shown = set(np.linspace(0, len(frames) - 1, min(num_images, len(frames))).round().astype(int).tolist())

    per_camera = {}
    for group, cams in (("holdout", meta["holdout_cameras"]), ("train", meta["train_cameras"])):
        for cam in cams:
            viewmats, Ks = ds.camera_tensors([cam], dev)
            rows = []
            previous = None  # (frame index, render, ground truth) of the frame before
            for k, fi in enumerate(frames):
                gt, mask = ds.batch(fi, [cam], dev)
                shift = take.time_offsets(fi).get(cam, 0.0)  # learned sub-frame offset
                rgb, alpha, *_ = render_scene(take.scene_at(fi), fi + shift, viewmats, Ks, ds.width, ds.height,
                                              backend=backend, antialiased=antialiased)
                # A training camera is compared with the scene as that camera sees it.
                rgb = apply_camera_colors(rgb, take.camera_colors(fi).get(cam))
                if mask is not None:  # compare subject only, on white
                    gt = gt * mask + (1.0 - mask)
                    pred = (rgb + (1.0 - alpha)).clamp(0, 1)
                    region = (mask[0, ..., 0] > 0.5) | (alpha[0, ..., 0] > 0.5)
                    inter = ((mask > 0.5) & (alpha > 0.5)).sum().item()
                    union = ((mask > 0.5) | (alpha > 0.5)).sum().item()
                    iou = inter / union if union else 1.0
                else:
                    pred = rgb.clamp(0, 1)
                    region = torch.ones_like(alpha[0, ..., 0], dtype=torch.bool)
                    iou = float("nan")
                row = {"frame": ds.frame_ids[fi], "psnr": psnr(pred, gt), "ssim": float(ssim(pred, gt)),
                       "iou": iou}
                box = _crop_box(region)
                if box is not None:
                    y0, y1, x0, x1 = box
                    row["psnr_subject"] = psnr(pred[:, y0:y1, x0:x1], gt[:, y0:y1, x0:x1])
                    if previous is not None and previous[0] == fi - 1:
                        # Change between consecutive frames that is not in the footage:
                        # 0 for a result that moves exactly like the subject, whatever its
                        # static errors. In 8-bit levels, inside the box of the subject.
                        extra = (pred - previous[1]) - (gt - previous[2])
                        row["flicker"] = float(extra[:, y0:y1, x0:x1].abs().mean() * 255.0)
                        if take.is_cut(fi):  # where two segments of a long take meet
                            row["flicker_at_cuts"] = row["flicker"]
                previous = (fi, pred, gt)
                rows.append(row)
                if k in shown:
                    err = ((pred - gt).abs().mean(dim=-1, keepdim=True) * 4.0).clamp(0, 1).expand(-1, -1, -1, 3)
                    strip = torch.cat([gt[0], pred[0], err[0]], dim=1).cpu().numpy()
                    cv2.imwrite(str(out / f"{group}_{cam}_{ds.frame_ids[fi]:06d}.jpg"),
                                cv2.cvtColor((strip * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
            keys = [k for k in ("psnr", "psnr_subject", "ssim", "iou", "flicker", "flicker_at_cuts")
                    if any(k in r for r in rows)]
            per_camera[cam] = {"group": group, "frames": rows,
                               **{k: float(np.nanmean([r[k] for r in rows if k in r])) for k in keys}}
            log(f"[eval] {group:7s} {cam}: " + "  ".join(f"{k}={per_camera[cam][k]:.3f}" for k in keys))

    def mean_of(group, key):
        vals = [c[key] for c in per_camera.values() if c["group"] == group and key in c]
        return float(np.mean(vals)) if vals else None

    summary = {
        "train_cameras": meta["train_cameras"], "holdout_cameras": meta["holdout_cameras"],
        "frames_evaluated": len(frames), "gaussians": take.num_gaussians, "segments": len(take.segments),
        "holdout": {k: mean_of("holdout", k) for k in METRICS},
        "train": {k: mean_of("train", k) for k in METRICS},
        "per_camera": per_camera,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    def fmt(v):
        return "n/a" if v is None else f"{v:.3f}"

    lines = [
        "# Evaluation", "",
        f"- training cameras ({len(meta['train_cameras'])}): {', '.join(meta['train_cameras'])}",
        f"- held-out cameras ({len(meta['holdout_cameras'])}): {', '.join(meta['holdout_cameras']) or 'none'}",
        f"- frames evaluated: {len(frames)}; Gaussians: {take.num_gaussians}"
        + (f"; segments: {len(take.segments)}" if len(take.segments) > 1 else ""),
    ]
    if meta.get("coverage"):
        from .camera import describe_coverage

        lines.append(f"- coverage of the training cameras: {describe_coverage(meta['coverage'])}")
    lines.append("")
    if meta["holdout_cameras"]:
        lines += ["The held-out cameras were never seen during training (nor by the visual hull if "
                  "`init --exclude` was used): their numbers measure novel-view quality. Training "
                  f"used only {len(meta['train_cameras'])} cameras, so coverage is reduced with "
                  "respect to the full rig.", ""]
    else:
        lines += ["**No camera was held out**: the numbers below only say how well the training "
                  "views are reproduced, not how the scene looks from a new point of view.", ""]
    lines += ["| group | PSNR | PSNR subject | SSIM | silhouette IoU | flicker |", "|---|---|---|---|---|---|"]
    for group in ("holdout", "train"):
        s = summary[group]
        lines.append(f"| {group} | {fmt(s['psnr'])} | {fmt(s['psnr_subject'])} | {fmt(s['ssim'])} "
                     f"| {fmt(s['iou'])} | {fmt(s['flicker'])} |")
    lines += ["", "PSNR subject = PSNR inside the bounding box of the subject (the full-frame value is "
              "inflated by the uniform background). Flicker = change between consecutive frames that "
              "is not in the footage, in 8-bit levels inside the box of the subject (lower is better; "
              "needs every frame, i.e. --frame-step 1). Images: ground truth | render | error x4."]
    for group in ("holdout", "train"):
        if len(take.segments) > 1 and summary[group]["flicker_at_cuts"] is not None:
            lines += ["", f"Flicker where two segments meet ({group}): {fmt(summary[group]['flicker_at_cuts'])} "
                      f"(elsewhere {fmt(summary[group]['flicker'])} on average)."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    log(f"[eval] report in {out}")
    return summary
