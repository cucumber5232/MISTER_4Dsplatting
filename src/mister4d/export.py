"""Export a trained scene for playback.

Three formats, all a sequence of frames holding the Gaussians alive at that
instant:

- ``html``: one self-contained .html file (viewer + data) that opens with a
  double click: no server, no installation. The thing to hand to someone else;
- ``web``: a folder with the same viewer and one compact binary per frame
  (17 bytes per Gaussian: quantized position, rotation, scale, colour and
  opacity), served by ``mister4d view``. Lighter to load for long takes;
- ``ply``: one standard 3DGS ``.ply`` per frame, for third-party tools
  (uncompressed: much heavier).
"""

from __future__ import annotations

import base64
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .model import load_checkpoint
from .rasterizer import SH_C0

TEX_WIDTH = 1024
VIEWER_DIR = Path(__file__).parent / "viewer"


def _frame_list(meta: dict, frame_step: int) -> list[int]:
    if meta.get("static_frame") is not None:
        return [meta["static_frame"]]
    return list(range(0, len(meta["frame_ids"]), max(frame_step, 1)))


@torch.no_grad()
def _alive(scene, frame: int, min_opacity: float) -> dict[str, np.ndarray]:
    """Gaussians visible at ``frame``, as numpy arrays (opacity already includes the temporal fade)."""
    state = scene.state_at(frame)
    keep = state["opacities"] >= min_opacity
    return {k: state[k][keep].cpu().numpy() for k in ("means", "quats", "scales", "opacities", "sh")}


@torch.no_grad()
def _bake(run_dir: str | Path, frame_step: int, min_opacity: float) -> tuple[dict, list[bytes]]:
    """Scene description and one quantized binary blob per frame."""
    scene, ckpt = load_checkpoint(Path(run_dir) / "ckpt.pt", "cpu")
    meta = ckpt["meta"]
    frames = _frame_list(meta, frame_step)
    baked = [_alive(scene, f, min_opacity) for f in frames]
    counts = [len(b["means"]) for b in baked]
    if max(counts) == 0:
        raise RuntimeError("nothing to export: no Gaussian is visible in any frame")
    filled = [b for b in baked if len(b["means"])]
    lo = np.min([b["means"].min(axis=0) for b in filled], axis=0) - 0.01
    hi = np.max([b["means"].max(axis=0) for b in filled], axis=0) + 0.01
    ls_min = float(min(np.log(b["scales"]).min() for b in filled))
    ls_max = float(max(np.log(b["scales"]).max() for b in filled)) + 1e-6
    tex_h = max(1, -(-max(counts) // TEX_WIDTH))
    slots = TEX_WIDTH * tex_h

    blobs = []
    for b in baked:
        n = len(b["means"])
        pos = np.zeros((slots, 3), dtype=np.uint16)
        rot = np.full((slots, 4), 128, dtype=np.uint8)
        scl = np.zeros((slots, 3), dtype=np.uint8)
        col = np.zeros((slots, 4), dtype=np.uint8)
        if n:
            pos[:n] = np.round((b["means"] - lo) / (hi - lo) * 65535.0)
            rot[:n] = np.round(np.clip(b["quats"], -1, 1) * 127.5 + 127.5)
            scl[:n] = np.round((np.log(b["scales"]) - ls_min) / (ls_max - ls_min) * 255.0)
            col[:n, :3] = np.round(np.clip(b["sh"][:, 0, :] * SH_C0 + 0.5, 0, 1) * 255.0)
            col[:n, 3] = np.round(np.clip(b["opacities"], 0, 1) * 255.0)
        blobs.append(pos.tobytes() + rot.tobytes() + scl.tobytes() + col.tobytes())

    cameras = []
    for cam_name in meta["train_cameras"] + meta["holdout_cameras"]:
        c = meta["cameras"][cam_name]
        cameras.append({"name": cam_name, "width": c["width"], "height": c["height"],
                        "K": c["K"], "R": c["R"], "t": c["t"],
                        "holdout": cam_name in meta["holdout_cameras"]})
    scene_json = {
        "format": "mister4d-web", "version": 2,
        "tex_width": TEX_WIDTH, "tex_height": tex_h,
        "num_frames": len(blobs), "fps": meta["fps"] / max(frame_step, 1),
        "frames": [f"frames/{f:06d}.bin" for f in frames], "counts": counts,
        "bbox_min": lo.tolist(), "bbox_max": hi.tolist(),
        "log_scale_min": ls_min, "log_scale_max": ls_max,
        "center": meta["center"], "up": [0, 0, 1], "cameras": cameras,
    }
    return scene_json, blobs


def export_web(run_dir: str | Path, out_dir: str | Path, frame_step: int = 1,
               min_opacity: float = 0.02, log=print) -> Path:
    """Folder with the viewer, ``scene.json`` and one binary per frame."""
    scene_json, blobs = _bake(run_dir, frame_step, min_opacity)
    out = Path(out_dir)
    if (out / "frames").exists():
        shutil.rmtree(out / "frames")
    (out / "frames").mkdir(parents=True)
    for name, blob in zip(scene_json["frames"], blobs):
        (out / name).write_bytes(blob)
    (out / "scene.json").write_text(json.dumps(scene_json))
    for f in VIEWER_DIR.iterdir():
        if f.is_file():
            shutil.copy(f, out / f.name)
    counts = scene_json["counts"]
    size_mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    log(f"[export] web player: {len(blobs)} frames, {int(np.mean(counts))} Gaussians per frame "
        f"(max {max(counts)}), {size_mb:.1f} MB -> {out}")
    log(f"[export] open it with: mister4d view {out}")
    return out


def export_html(run_dir: str | Path, out_file: str | Path, frame_step: int = 1,
                min_opacity: float = 0.02, log=print) -> Path:
    """One .html file with viewer and data inside: opens with a double click."""
    scene_json, blobs = _bake(run_dir, frame_step, min_opacity)
    page = (VIEWER_DIR / "index.html").read_text(encoding="utf-8")
    tag = '<script src="viewer.js"></script>'
    if tag not in page:
        raise RuntimeError("viewer/index.html does not reference viewer.js as expected")
    # "<" never appears literally inside the JSON block, so it cannot close the element early.
    parts = ['<script id="mister4d-scene" type="application/json">'
             + json.dumps(scene_json).replace("<", "\\u003c") + "</script>"]
    for i, blob in enumerate(blobs):
        parts.append(f'<script id="mister4d-frame-{i}" type="application/octet-stream">'
                     + base64.b64encode(blob).decode("ascii") + "</script>")
    parts.append("<script>\n" + (VIEWER_DIR / "viewer.js").read_text(encoding="utf-8") + "\n</script>")
    out = Path(out_file)
    if out.suffix.lower() not in (".html", ".htm"):
        out = out.with_name(out.name + ".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page.replace(tag, "\n".join(parts)), encoding="utf-8")
    counts = scene_json["counts"]
    size_mb = out.stat().st_size / 1e6
    log(f"[export] single-file player: {len(blobs)} frames, {int(np.mean(counts))} Gaussians per "
        f"frame (max {max(counts)}), {size_mb:.1f} MB -> {out}")
    if size_mb > 300:
        log("[export] WARNING: large file, browsers may take long to open it: use --frame-step "
            "or the folder format (--format web)")
    return out


@torch.no_grad()
def export_ply(run_dir: str | Path, out_dir: str | Path, frame_step: int = 1,
               min_opacity: float = 0.02, log=print) -> Path:
    """One standard 3DGS .ply per frame (x y z, normals, f_dc, f_rest, opacity, scale, rot)."""
    scene, ckpt = load_checkpoint(Path(run_dir) / "ckpt.pt", "cpu")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for f in _frame_list(ckpt["meta"], frame_step):
        b = _alive(scene, f, min_opacity)
        n = len(b["means"])
        rest = b["sh"][:, 1:, :].transpose(0, 2, 1).reshape(n, -1)  # channel-major, as 3DGS expects
        fields = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
                  + [f"f_rest_{i}" for i in range(rest.shape[1])] + ["opacity"]
                  + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
        opacity = np.clip(b["opacities"], 1e-6, 1 - 1e-6)
        table = np.concatenate([b["means"], np.zeros((n, 3)), b["sh"][:, 0, :], rest,
                                np.log(opacity / (1 - opacity))[:, None],  # stored as a logit
                                np.log(b["scales"]), b["quats"]], axis=1).astype("<f4")
        header = ("ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
                  + "".join(f"property float {name}\n" for name in fields) + "end_header\n")
        with open(out / f"frame_{f:06d}.ply", "wb") as fh:
            fh.write(header.encode("ascii"))
            fh.write(table.tobytes())
    files = list(out.glob("*.ply"))
    log(f"[export] {len(files)} .ply files, {sum(f.stat().st_size for f in files) / 1e6:.1f} MB -> {out}")
    return out


def serve(directory: str | Path, port: int = 8080, host: str = "127.0.0.1") -> None:
    """Serve an exported web player (browsers refuse to load it from file://)."""
    import functools
    import http.server

    directory = Path(directory).resolve()
    if not (directory / "scene.json").exists():
        raise FileNotFoundError(f"{directory} is not a web export (no scene.json)")
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(directory))
    with http.server.ThreadingHTTPServer((host, port), handler) as httpd:
        print(f"[view] http://{host}:{port}/  (Ctrl+C to stop)")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
