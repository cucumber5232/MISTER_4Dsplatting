"""How real does a trained take look? Measures against ground truth where the dataset has it.

usage: python scripts/quality_check.py RUN [--data DATASET] [--out DIR] [--frames 0,1,2] [--sheet-cams ver01,ver02]

Besides the usual image scores (subject PSNR, SSIM, silhouette IoU, flicker) it reports what the
eye reads as "fog" and "things flying around":

  spill      rendered alpha outside the true silhouette, as a share of the silhouette area
  holes      missing alpha inside the true silhouette
  soft       share of the pixels around the subject whose alpha is neither empty nor full
  depth_cm   median distance between rendered and true depth where the subject is drawn
  thick_cm   median spread in depth of what a pixel is made of (a surface is thin, fog is thick)
  needles    share of the visible Gaussians at least three times longer in one direction than in the next
  floaters   share of the opacity that sits where there is nothing: outside the true silhouette
             of some camera, or more than 4 cm in front of the true surface

Cameras are split in "train" and "verify" (never used for the visual hull nor for training).
The dataset needs truth/<cam>/<frame>.npz (depth, mask) and truth/cameras.json for depth and
floaters (scripts/make_benchmark.py writes them); without them the rest is computed against
the masks of the dataset. Development tool: run it with the Python of .venv.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from mister4d.dataset import MultiViewDataset, frame_name
from mister4d.losses import psnr, ssim
from mister4d.model import render_scene
from mister4d.rasterizer import opencv_to_render_K, render, resolve_backend
from mister4d.run import Run
from mister4d.trainer import pick_device

p = argparse.ArgumentParser()
p.add_argument("run")
p.add_argument("--data", default=None)
p.add_argument("--out", default=None)
p.add_argument("--frames", default="")
p.add_argument("--sheet-cams", default="")
p.add_argument("--sheet-frames", default="")
p.add_argument("--front-cm", type=float, default=4.0)
p.add_argument("--device", default="auto")
p.add_argument("--quiet", action="store_true")
a = p.parse_args()

dev = pick_device(a.device)
take = Run(a.run, dev)
meta = take.meta
ds = MultiViewDataset(a.data or meta["data_dir"])
backend = resolve_backend(take.config.get("backend", "auto"), dev)
antialiased = bool(take.config.get("antialiased", False))
out = Path(a.out or Path(a.run) / "evalq")
out.mkdir(parents=True, exist_ok=True)
W, H = ds.width, ds.height

truth_dir = ds.root / "truth"
groups = {"train": list(meta["train_cameras"]), "verify": list(meta["holdout_cameras"])}
if (truth_dir / "cameras.json").exists():
    listed = json.loads((truth_dir / "cameras.json").read_text())
    groups["verify"] = [c for c in listed.get("verify", []) if c not in groups["train"]] or groups["verify"]

if meta.get("static_frame") is not None:
    frames = [meta["static_frame"]]
elif a.frames:
    frames = [int(f) for f in a.frames.split(",")]
else:
    frames = list(range(ds.num_frames))


def truth(cam: str, fi: int):
    """True depth (metres, 0 = nothing) and coverage (0..1) of ``cam`` at frame index ``fi``, or None."""
    for name in (frame_name(ds.frame_ids[fi]), str(ds.frame_ids[fi]), frame_name(fi), str(fi)):
        f = truth_dir / cam / f"{name}.npz"
        if f.exists():
            z = np.load(f)
            return z["depth"].astype(np.float32), z["mask"].astype(np.float32) / 255.0
    return None


def box_of(region: np.ndarray, margin: int = 8):
    ys, xs = np.nonzero(region)
    if len(ys) == 0:
        return None
    return (max(ys.min() - margin, 0), min(ys.max() + margin + 1, H), max(xs.min() - margin, 0), min(xs.max() + margin + 1, W))


kernel3 = np.ones((3, 3), np.uint8)
kernel5 = np.ones((5, 5), np.uint8)
rows = {g: {c: [] for c in cams} for g, cams in groups.items()}
sheet_cams = [c for c in a.sheet_cams.split(",") if c] or groups["verify"][:3] or groups["train"][:2]
sheet_frames = [int(f) for f in a.sheet_frames.split(",")] if a.sheet_frames else [frames[len(frames) // 2]]
tiles = []
floaters, visible, opaque, needles = [], [], [], []

with torch.no_grad():
    for fi in frames:
        scene = take.scene_at(fi)
        state = scene.view().state_at(fi)
        means, opac = state["means"].cpu(), state["opacities"].cpu()
        seen = opac >= 0.02
        visible.append(int(seen.sum()))
        opaque.append(float((opac[seen] > 0.5).float().mean()) if seen.any() else 0.0)
        ordered = state["scales"].cpu()[seen].sort(dim=-1, descending=True).values  # a needle: one axis far longer than the next
        needles.append(float((ordered[:, 0] / ordered[:, 1].clamp(min=1e-9) > 3.0).float().mean()) if seen.any() else 0.0)
        flagged = {"train": torch.zeros_like(opac, dtype=torch.bool), "verify": torch.zeros_like(opac, dtype=torch.bool)}
        have_truth = False
        for group, cams in groups.items():
            for cam in cams:
                viewmats, Ks = ds.camera_tensors([cam], dev)
                gt, mask = ds.batch(fi, [cam], dev)
                rgb, alpha, depth, _, st = render_scene(scene, fi, viewmats, Ks, W, H, backend=backend, antialiased=antialiased)
                rgb, alpha, depth, gt = rgb.cpu(), alpha.cpu(), depth.cpu(), gt.cpu()
                mask = None if mask is None else mask.cpu()
                t = truth(cam, fi)
                if t is not None:
                    have_truth = True
                    tdepth, tmask = t
                    m = torch.from_numpy(tmask)[None, ..., None]
                elif mask is not None:
                    tdepth, tmask, m = None, mask[0, ..., 0].numpy(), mask
                else:
                    continue
                target = gt * m + (1.0 - m)
                pred = (rgb + (1.0 - alpha)).clamp(0, 1)
                al = alpha[0, ..., 0].numpy()
                solid = tmask > 0.5
                area = max(float(solid.sum()), 1.0)
                box = box_of(solid | (al > 0.5))
                near = cv2.dilate(solid.astype(np.uint8), kernel5, iterations=3) > 0  # the subject and a rim around it
                outside = cv2.dilate(solid.astype(np.uint8), kernel3) == 0
                inside = cv2.erode(solid.astype(np.uint8), kernel3) > 0
                row = {"frame": int(ds.frame_ids[fi]), "psnr": psnr(pred, target), "ssim": float(ssim(pred, target)),
                       "iou": float((solid & (al > 0.5)).sum() / max((solid | (al > 0.5)).sum(), 1)),
                       "spill": float(al[outside].sum() / area), "holes": float((1.0 - al[inside]).sum() / area),
                       "soft": float(((al > 0.05) & (al < 0.95) & near).sum() / max(near.sum(), 1))}
                if box is not None:
                    y0, y1, x0, x1 = box
                    row["psnr_subject"] = psnr(pred[:, y0:y1, x0:x1], target[:, y0:y1, x0:x1])
                # Thickness: what a pixel is made of, spread along the ray.
                z_cam = (st["means"] @ viewmats[0, :3, :3].T + viewmats[0, :3, 3])[:, 2]
                moments = torch.stack([z_cam, z_cam * z_cam, torch.zeros_like(z_cam)], dim=-1)
                mom, _, _, _ = render(st["means"], st["quats"], st["scales"], st["opacities"], moments, viewmats,
                                      opencv_to_render_K(Ks), W, H, backend=backend, antialiased=antialiased)
                drawn = al > 0.5
                mom = mom.cpu()
                mean_z = mom[0, ..., 0].numpy() / np.maximum(al, 1e-6)
                var_z = np.maximum(mom[0, ..., 1].numpy() / np.maximum(al, 1e-6) - mean_z ** 2, 0.0)
                if (drawn & solid).any():
                    row["thick_cm"] = float(np.median(np.sqrt(var_z[drawn & solid])) * 100.0)
                    if tdepth is not None:
                        err = np.abs(depth[0, ..., 0].numpy() - tdepth)[drawn & solid & (tdepth > 0)]
                        if err.size:
                            row["depth_cm"] = float(np.median(err) * 100.0)
                            row["depth90_cm"] = float(np.percentile(err, 90) * 100.0)
                rows[group][cam].append(row | {"_pred": pred, "_target": target})
                # Where is each Gaussian, as this camera knows the truth?
                if tdepth is not None:
                    view_cpu = viewmats[0].cpu()
                    cam_pts = means @ view_cpu[:3, :3].T + view_cpu[:3, 3]
                    z = cam_pts[:, 2]
                    K = Ks[0].cpu()
                    u = (K[0, 0] * cam_pts[:, 0] / z.clamp(min=1e-6) + K[0, 2]).round().long()
                    v = (K[1, 1] * cam_pts[:, 1] / z.clamp(min=1e-6) + K[1, 2]).round().long()
                    ok = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
                    far = np.where(tdepth > 0, tdepth, np.inf).astype(np.float32)
                    nearest = torch.from_numpy(cv2.erode(far, kernel5))  # the closest true surface around the pixel
                    around = torch.from_numpy(cv2.dilate((tmask > 0.02).astype(np.uint8), kernel5) > 0)
                    uu, vv = u.clamp(0, W - 1), v.clamp(0, H - 1)
                    empty = ok & ~around[vv, uu]
                    front = ok & around[vv, uu] & (z < nearest[vv, uu] - a.front_cm / 100.0)
                    flagged[group] |= empty | front
                if cam in sheet_cams and fi in sheet_frames and box is not None:
                    y0, y1, x0, x1 = box
                    err_img = ((pred - target).abs().mean(dim=-1, keepdim=True) * 4.0).clamp(0, 1).expand(-1, -1, -1, 3)
                    parts = [target[0], pred[0], err_img[0], alpha[0].expand(-1, -1, 3)]
                    strip = torch.cat([x[y0:y1, x0:x1] for x in parts], dim=1).numpy()
                    strip = cv2.cvtColor((strip * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    k = max(1, int(round(420 / strip.shape[0])))
                    strip = cv2.resize(strip, None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST if k <= 2 else cv2.INTER_CUBIC)
                    cv2.putText(strip, f"{cam} ({group}) frame {fi}: truth | render | error x4 | alpha", (6, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1, cv2.LINE_AA)
                    tiles.append(strip)
        if have_truth:
            total = float(opac.sum().clamp(min=1e-9))
            floaters.append({"train": float(opac[flagged["train"]].sum() / total),
                             "any": float(opac[flagged["train"] | flagged["verify"]].sum() / total)})

KEYS = ("psnr_subject", "ssim", "iou", "spill", "holes", "soft", "depth_cm", "depth90_cm", "thick_cm", "flicker")
summary = {"run": str(a.run), "frames": len(frames), "gaussians": take.num_gaussians,
           "visible_per_frame": float(np.mean(visible)), "opaque_share": float(np.mean(opaque)), "needle_share": float(np.mean(needles)), "cameras": groups}
for group, cams in rows.items():
    per_cam = {}
    for cam, rs in cams.items():
        for prev, cur in zip(rs, rs[1:]):  # change between consecutive frames that is not in the footage
            if cur["frame"] - prev["frame"] == 1:
                extra = (cur["_pred"] - prev["_pred"]) - (cur["_target"] - prev["_target"])
                cur["flicker"] = float(extra.abs().mean() * 255.0 * (W * H) / max(float((cur["_target"] < 0.999).any(-1).sum()), 1.0))
        for r in rs:
            r.pop("_pred"), r.pop("_target")
        per_cam[cam] = {k: float(np.mean([r[k] for r in rs if k in r])) for k in KEYS if any(k in r for r in rs)}
    summary[group] = {k: float(np.mean([c[k] for c in per_cam.values() if k in c])) for k in KEYS
                      if any(k in c for c in per_cam.values())}
    summary[group + "_per_camera"] = per_cam
if floaters:
    summary["floaters_train"] = float(np.mean([f["train"] for f in floaters]))
    summary["floaters_any"] = float(np.mean([f["any"] for f in floaters]))
(out / "metrics.json").write_text(json.dumps(summary, indent=2))
if tiles:
    width = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 4, 0, width - t.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tiles]
    cv2.imwrite(str(out / "sheet.png"), np.concatenate(tiles, axis=0))


def line(group):
    s = summary.get(group, {})
    pct = lambda k: f"{100 * s[k]:.1f}%" if k in s else "-"  # noqa: E731
    num = lambda k, d=2: f"{s[k]:.{d}f}" if k in s else "-"  # noqa: E731
    return (f"{group:6s} psnr={num('psnr_subject')} ssim={num('ssim', 3)} iou={num('iou', 3)} spill={pct('spill')} "
            f"holes={pct('holes')} soft={pct('soft')} depth={num('depth_cm', 1)}cm thick={num('thick_cm', 1)}cm "
            f"flicker={num('flicker')}")


if not a.quiet:
    print(line("verify"))
    print(line("train"))
    fl = (f"floaters any={100 * summary['floaters_any']:.1f}% train={100 * summary['floaters_train']:.1f}%"
          if floaters else "floaters n/a (no truth)")
    print(f"{fl} | gaussians={summary['gaussians']} visible/frame={summary['visible_per_frame']:.0f} "
          f"opaque(>0.5)={100 * summary['opaque_share']:.0f}% needles={100 * summary['needle_share']:.0f}% -> {out}")
