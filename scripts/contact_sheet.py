"""Contact sheet of a trained take: viewpoints along and beyond the arc of the cameras, at some instants.

usage: python scripts/contact_sheet.py RUN OUT.png [--scale 2] [--zoom 2.2] [--width 400] [--steps 6] [--frames 0,7,14]

The labels say how far each viewpoint is from the nearest training camera. With --scale and
--zoom above 1 the scene is seen larger than it was trained, which is where thin or
half-transparent Gaussians show. Development tool: run it with the Python of .venv.
"""
import argparse
import json

import cv2
import numpy as np
import torch

from mister4d.camera import Camera, look_at
from mister4d.model import render_scene
from mister4d.rasterizer import resolve_backend
from mister4d.run import Run
from mister4d.trainer import pick_device

p = argparse.ArgumentParser()
p.add_argument("run"); p.add_argument("out")
p.add_argument("--scale", type=float, default=3.0)
p.add_argument("--frames", default="")
p.add_argument("--steps", type=int, default=7)
p.add_argument("--extra", type=float, default=40.0, help="degrees beyond the ends of the rig arc")
p.add_argument("--bg", type=float, default=1.0)
p.add_argument("--zoom", type=float, default=1.0)
p.add_argument("--width", type=int, default=0)
a = p.parse_args()

dev = pick_device("auto")
run = Run(a.run, dev)
meta = run.meta
cams = [Camera.from_dict(n, meta["cameras"][n]) for n in meta["train_cameras"]]
ref = cams[0]
center = np.array(meta["center"])
off = np.array([c.center - center for c in cams])
az = np.degrees(np.arctan2(off[:, 1], off[:, 0]))
radius = float(np.mean(np.linalg.norm(off[:, :2], axis=1))); height = float(np.mean(off[:, 2]))
lo, hi = az.min() - a.extra, az.max() + a.extra
angles = np.linspace(lo, hi, a.steps)
frames = [int(f) for f in a.frames.split(",")] if a.frames else list(np.linspace(0, run.num_frames - 1, 4).astype(int))
H = int(ref.height * a.scale); W = a.width or int(ref.width * a.scale)
foc = ref.K[1, 1] * a.scale * a.zoom
K = np.array([[foc, 0, (W - 1) / 2], [0, foc, (H - 1) / 2], [0, 0, 1.0]])
backend = resolve_backend("auto", dev)
rows = []
with torch.no_grad():
    for f in frames:
        row = []
        for ang in angles:
            r = np.radians(ang)
            eye = center + np.array([radius * np.cos(r), radius * np.sin(r), height])
            st = run.scene_at(f).state_at(f)
            w = st["opacities"][:, None]
            target = ((st["means"] * w).sum(0) / w.sum()).cpu().numpy()
            R, t = look_at(eye, target)
            view = np.eye(4); view[:3, :3], view[:3, 3] = R, t
            rgb, alpha, *_ = render_scene(run.scene_at(f), f, torch.tensor(view, dtype=torch.float32, device=dev)[None],
                                          torch.tensor(K, dtype=torch.float32, device=dev)[None], W, H, backend=backend)
            img = (rgb[0] + (1 - alpha[0]) * a.bg).clamp(0, 1).cpu().numpy()
            img = (img[..., ::-1] * 255).astype(np.uint8).copy()
            near = np.abs(az - ang).min()
            cv2.putText(img, f"f{f} az{ang:.0f} ({near:.0f} deg from a camera)", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (0, 0, 200) if near > 5 else (0, 130, 0), 1, cv2.LINE_AA)
            row.append(img)
        rows.append(np.concatenate(row, axis=1))
cv2.imwrite(a.out, np.concatenate(rows, axis=0))
print("cameras az", np.round(az, 1), "sheet", a.out, W, H)
