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
from .model import load_checkpoint, render_scene
from .rasterizer import resolve_backend
from .trainer import pick_device


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
    scene, ckpt = load_checkpoint(run / "ckpt.pt", dev)
    meta = ckpt["meta"]
    backend = resolve_backend(ckpt["config"].get("backend", "auto"), dev)
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
            for k, fi in enumerate(frames):
                gt, mask = ds.batch(fi, [cam], dev)
                rgb, alpha, *_ = render_scene(scene, fi, viewmats, Ks, ds.width, ds.height,
                                              backend=backend)
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
                rows.append(row)
                if k in shown:
                    err = ((pred - gt).abs().mean(dim=-1, keepdim=True) * 4.0).clamp(0, 1).expand(-1, -1, -1, 3)
                    strip = torch.cat([gt[0], pred[0], err[0]], dim=1).cpu().numpy()
                    cv2.imwrite(str(out / f"{group}_{cam}_{ds.frame_ids[fi]:06d}.jpg"),
                                cv2.cvtColor((strip * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
            keys = [k for k in ("psnr", "psnr_subject", "ssim", "iou") if all(k in r for r in rows)]
            per_camera[cam] = {"group": group, "frames": rows,
                               **{k: float(np.nanmean([r[k] for r in rows])) for k in keys}}
            log(f"[eval] {group:7s} {cam}: " + "  ".join(f"{k}={per_camera[cam][k]:.3f}" for k in keys))

    def mean_of(group, key):
        vals = [c[key] for c in per_camera.values() if c["group"] == group and key in c]
        return float(np.mean(vals)) if vals else None

    summary = {
        "train_cameras": meta["train_cameras"], "holdout_cameras": meta["holdout_cameras"],
        "frames_evaluated": len(frames), "gaussians": scene.num_gaussians,
        "holdout": {k: mean_of("holdout", k) for k in ("psnr", "psnr_subject", "ssim", "iou")},
        "train": {k: mean_of("train", k) for k in ("psnr", "psnr_subject", "ssim", "iou")},
        "per_camera": per_camera,
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))

    def fmt(v):
        return "n/a" if v is None else f"{v:.3f}"

    lines = [
        "# Evaluation", "",
        f"- training cameras ({len(meta['train_cameras'])}): {', '.join(meta['train_cameras'])}",
        f"- held-out cameras ({len(meta['holdout_cameras'])}): {', '.join(meta['holdout_cameras']) or 'none'}",
        f"- frames evaluated: {len(frames)}; Gaussians: {scene.num_gaussians}", "",
    ]
    if meta["holdout_cameras"]:
        lines += ["The held-out cameras were never seen during training (nor by the visual hull if "
                  "`init --exclude` was used): their numbers measure novel-view quality. Training "
                  f"used only {len(meta['train_cameras'])} cameras, so coverage is reduced with "
                  "respect to the full rig.", ""]
    else:
        lines += ["**No camera was held out**: the numbers below only say how well the training "
                  "views are reproduced, not how the scene looks from a new point of view.", ""]
    lines += ["| group | PSNR | PSNR subject | SSIM | silhouette IoU |", "|---|---|---|---|---|"]
    for group in ("holdout", "train"):
        s = summary[group]
        lines.append(f"| {group} | {fmt(s['psnr'])} | {fmt(s['psnr_subject'])} | {fmt(s['ssim'])} | {fmt(s['iou'])} |")
    lines += ["", "PSNR subject = PSNR inside the bounding box of the subject (the full-frame value is "
              "inflated by the uniform background). Images: ground truth | render | error x4."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    log(f"[eval] report in {out}")
    return summary
