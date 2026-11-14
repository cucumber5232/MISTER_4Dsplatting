"""Export a trained scene for playback and for other tools.

- ``html``: one self-contained .html file (player + scene) that opens with a
  double click: no server, no installation. The thing to hand to someone else,
  for takes of a few seconds;
- ``web``: a folder with the same player and the scene in chunks that are
  loaded while playing. It opens with a double click on ``index.html`` too
  (or from any web server), and plays takes of any length holding only the
  part being shown;
- ``engine``: a folder with the same chunks as plain binary files and a flat
  ``scene.json``, to be read by a game engine (see ``unity/Mister4D``);
- ``ply``: one standard 3DGS ``.ply`` per frame, for third-party tools
  (uncompressed: much heavier).

Player format ("mister4d-web", version 3)
-----------------------------------------
The scene is not baked frame by frame: every Gaussian is stored once, with
the instant it is centred on, how long it lives and how it moves, and the
player evaluates it at the time being shown, exactly as the training does.
Storing a Gaussian once instead of once per frame it is visible in is what
keeps the files light, and it makes slow motion real instead of repeated
frames.

One chunk per segment of the take; in a chunk the Gaussians are sorted by
time (the few long-lived ones first), 23 bytes each, as five blocks that are
uploaded to the GPU as they are:

    A  uint16 x4   position x y z (in the box of the chunk), time
    B  uint8  x4   velocity x y z (signed, 128 = still; the code is squared, so that
                   slow motion, the common case, is stored finely), life
    C  uint8  x4   rotation quaternion w x y z
    D  uint8  x4   log-scale x y z, peak opacity
    E  uint8  x3   colour r g b

The chunk is then gzip-compressed. ``scene.json`` holds the ranges needed to
turn the integers back into metres, frames and so on.
"""

from __future__ import annotations

import base64
import gzip
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .model import MIN_DURATION
from .rasterizer import SH_C0
from .run import Run

VIEWER_DIR = Path(__file__).parent / "viewer"
ALWAYS = 255  # life code of a Gaussian that does not depend on time (static scene)


@torch.no_grad()
def _alive(scene, frame: float, min_opacity: float) -> dict[str, np.ndarray]:
    """Gaussians visible at ``frame``, as numpy arrays (opacity already includes the temporal fade)."""
    state = scene.state_at(frame)
    keep = state["opacities"] >= min_opacity
    return {k: state[k][keep].cpu().numpy() for k in ("means", "quats", "scales", "opacities", "sh")}


def _quantize(values: np.ndarray, lo, hi, levels: int) -> np.ndarray:
    span = np.maximum(np.asarray(hi, dtype=np.float64) - lo, 1e-9)
    return np.clip(np.round((values - lo) / span * levels), 0, levels)


@torch.no_grad()
def pack_chunk(scene, first: int, last: int, min_opacity: float = 0.02) -> tuple[dict, bytes]:
    """The Gaussians a scene needs to show the frames ``[first, last)``, packed for the player.

    Returns the description of the chunk and its bytes (not yet compressed).
    """
    p = {k: v.detach().cpu().numpy() for k, v in scene.params.items()}
    opacity = 1.0 / (1.0 + np.exp(-p["opacities"].astype(np.float64)))
    static = bool(scene.static)
    if static:
        time, sigma = np.zeros_like(opacity), np.ones_like(opacity)
        reach = np.full_like(opacity, np.inf)
        keep = opacity >= min_opacity
    else:
        time = p["times"].astype(np.float64)
        sigma = np.maximum(np.exp(p["durations"].astype(np.float64)), MIN_DURATION)
        # Frames from its instant after which a Gaussian has faded below ``min_opacity``.
        reach = sigma * np.sqrt(2.0 * np.log(np.maximum(opacity / min_opacity, 1.0)))
        keep = (opacity >= min_opacity) & (time + reach >= first) & (time - reach < last)
    if not keep.any():
        raise RuntimeError(f"nothing to export: no Gaussian is visible in frames {first}-{last - 1}")

    # Long-lived Gaussians first (the player always considers them), the others by time
    # (the player looks only at those around the instant being shown).
    cap = np.inf if static else max(float(np.percentile(reach[keep], 95.0)), 2.0)
    is_long = keep & (reach > cap if not static else True)
    order = np.concatenate([np.nonzero(is_long)[0],
                            np.nonzero(keep & ~is_long)[0][np.argsort(time[keep & ~is_long], kind="stable")]])
    n, n_long = len(order), int(is_long.sum())
    means, time, sigma, reach, opacity = p["means"][order], time[order], sigma[order], reach[order], opacity[order]
    vel = np.zeros_like(means) if static else p["velocities"][order]
    quats = p["quats"][order] / np.maximum(np.linalg.norm(p["quats"][order], axis=1, keepdims=True), 1e-12)
    log_scale = p["scales"][order].astype(np.float64)
    rgb = np.clip(p["sh0"][order][:, 0, :] * SH_C0 + 0.5, 0.0, 1.0)

    box_lo, box_hi = means.min(axis=0) - 0.01, means.max(axis=0) + 0.01
    t_lo, t_span = float(time.min()), max(float(time.max() - time.min()), 1e-6)
    vel_max = max(float(np.percentile(np.abs(vel), 99.9)), 1e-6)
    dur_lo, dur_hi = float(np.log(MIN_DURATION)), max(float(np.log(sigma.max())), float(np.log(MIN_DURATION)) + 1e-6)
    ls_lo, ls_hi = float(log_scale.min()), float(log_scale.max()) + 1e-6

    a = np.empty((n, 4), dtype="<u2")
    a[:, :3] = _quantize(means, box_lo, box_hi, 65535)
    a[:, 3] = _quantize(time, t_lo, t_lo + t_span, 65535)
    b = np.empty((n, 4), dtype=np.uint8)
    b[:, :3] = np.sign(vel) * np.round(127.0 * np.sqrt(np.minimum(np.abs(vel) / vel_max, 1.0))) + 128
    b[:, 3] = ALWAYS if static else _quantize(np.log(sigma), dur_lo, dur_hi, 254)
    c = np.round(np.clip(quats, -1, 1) * 127.5 + 127.5).astype(np.uint8)
    d = np.empty((n, 4), dtype=np.uint8)
    d[:, :3] = _quantize(log_scale, ls_lo, ls_hi, 255)
    d[:, 3] = np.round(np.clip(opacity, 0, 1) * 255.0)
    e = np.round(rgb * 255.0).astype(np.uint8)

    # How many are visible at once: what the device that plays the scene has to draw.
    if static:
        alive = [n]
    else:
        alive = [int(np.count_nonzero(np.abs(f - time) <= reach)) for f in range(first, last)]
    info = {
        "first": int(first), "last": int(last), "count": n, "long": n_long,
        "reach": None if static else round(cap, 4),
        "bbox_min": box_lo.tolist(), "bbox_max": box_hi.tolist(),
        "time": [t_lo, t_span], "vel_max": vel_max, "life": [dur_lo, dur_hi], "log_scale": [ls_lo, ls_hi],
        "alive_max": max(alive), "alive_mean": int(np.mean(alive)),
    }
    return info, a.tobytes() + b.tobytes() + c.tobytes() + d.tobytes() + e.tobytes()


def _scene_description(run: Run, chunks: list[dict], min_opacity: float) -> dict:
    meta = run.meta
    cameras = []
    for cam_name in meta["train_cameras"] + meta["holdout_cameras"]:
        c = meta["cameras"][cam_name]
        cameras.append({"name": cam_name, "width": c["width"], "height": c["height"],
                        "K": c["K"], "R": c["R"], "t": c["t"],
                        "holdout": cam_name in meta["holdout_cameras"]})
    static = run.static_frame is not None
    return {
        "format": "mister4d-web", "version": 3,
        "num_frames": 1 if static else run.num_frames, "fps": meta["fps"], "static": static,
        "min_opacity": min_opacity,  # below this a Gaussian is not drawn: the exporter counted on it
        "bbox_min": np.min([c["bbox_min"] for c in chunks], axis=0).tolist(),
        "bbox_max": np.max([c["bbox_max"] for c in chunks], axis=0).tolist(),
        "gaussians": sum(c["count"] for c in chunks),
        "alive_max": max(c["alive_max"] for c in chunks),
        "center": meta["center"], "up": [0, 0, 1], "cameras": cameras, "chunks": chunks,
    }


def _chunks(run: Run, min_opacity: float):
    """Description and compressed bytes of every chunk, one at a time (a long take never sits in memory)."""
    if run.static_frame is not None:
        scene, _, _, _ = run.segment(0)
        info, raw = pack_chunk(scene, 0, 1, min_opacity)
        yield info, gzip.compress(raw, compresslevel=9, mtime=0)
        return
    for i in range(len(run.segments)):
        scene, _, first, last = run.segment(i)
        info, raw = pack_chunk(scene, first, last, min_opacity)
        yield info, gzip.compress(raw, compresslevel=9, mtime=0)


def _report(kind: str, scene: dict, size_mb: float, out: Path, log) -> None:
    seconds = scene["num_frames"] / scene["fps"]
    rate = f", {size_mb / seconds:.1f} MB per second of take" if scene["num_frames"] > 1 else ""
    log(f"[export] {kind}: {scene['num_frames']} frames in {len(scene['chunks'])} chunk(s), "
        f"{scene['gaussians']} Gaussians, at most {scene['alive_max']} visible at once, "
        f"{size_mb:.1f} MB{rate} -> {out}")


def export_web(run_dir: str | Path, out_dir: str | Path, frame_step: int = 1,
               min_opacity: float = 0.02, log=print) -> Path:
    """Folder with the player and the scene in chunks loaded on demand.

    The chunks are small scripts, not binary files: a page opened from disk
    may load scripts but not fetch data, so the folder works with a double
    click on ``index.html`` as well as from a web server.
    """
    run = Run(run_dir)
    out = Path(out_dir)
    if (out / "chunks").exists():
        shutil.rmtree(out / "chunks")
    (out / "chunks").mkdir(parents=True)
    chunks = []
    for i, (info, data) in enumerate(_chunks(run, min_opacity)):
        info["file"] = f"chunks/{i:04d}.js"
        info["bytes"] = len(data)
        (out / info["file"]).write_text(f'mister4dChunk({i},"{base64.b64encode(data).decode("ascii")}");\n',
                                        encoding="ascii")
        chunks.append(info)
    scene = _scene_description(run, chunks, min_opacity)
    (out / "scene.json").write_text(json.dumps(scene))
    (out / "scene.js").write_text("mister4dScene(" + json.dumps(scene) + ");\n", encoding="utf-8")
    for f in VIEWER_DIR.iterdir():
        if f.is_file():
            shutil.copy(f, out / f.name)
    page = (out / "index.html").read_text(encoding="utf-8")
    tag = '<script src="viewer.js"></script>'
    if tag not in page:
        raise RuntimeError("viewer/index.html does not reference viewer.js as expected")
    (out / "index.html").write_text(page.replace(tag, tag + '\n<script src="scene.js"></script>'), encoding="utf-8")
    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    _report("player folder", scene, size_mb, out, log)
    log(f"[export] open {out / 'index.html'} with a double click, or serve the folder (mister4d view {out})")
    return out


def export_html(run_dir: str | Path, out_file: str | Path, frame_step: int = 1,
                min_opacity: float = 0.02, log=print) -> Path:
    """One .html file with player and scene inside: opens with a double click."""
    run = Run(run_dir)
    page = (VIEWER_DIR / "index.html").read_text(encoding="utf-8")
    tag, css = '<script src="viewer.js"></script>', '<link rel="stylesheet" href="theme.css">'
    if tag not in page or css not in page:
        raise RuntimeError("viewer/index.html does not reference viewer.js and theme.css as expected")
    page = page.replace(css, "<style>\n" + (VIEWER_DIR / "theme.css").read_text(encoding="utf-8") + "</style>")
    chunks, blocks = [], []
    for i, (info, data) in enumerate(_chunks(run, min_opacity)):
        info["bytes"] = len(data)
        chunks.append(info)
        blocks.append(f'<script id="mister4d-chunk-{i}" type="application/octet-stream">'
                      + base64.b64encode(data).decode("ascii") + "</script>")
    scene = _scene_description(run, chunks, min_opacity)
    # "<" never appears literally inside the JSON block, so it cannot close the element early.
    parts = ['<script id="mister4d-scene" type="application/json">'
             + json.dumps(scene).replace("<", "\\u003c") + "</script>", *blocks,
             "<script>\n" + (VIEWER_DIR / "viewer.js").read_text(encoding="utf-8") + "\n</script>"]
    out = Path(out_file)
    if out.suffix.lower() not in (".html", ".htm"):
        out = out.with_name(out.name + ".html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page.replace(tag, "\n".join(parts)), encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    _report("single-file player", scene, size_mb, out, log)
    if size_mb > 150:
        log("[export] WARNING: a single file this large is slow to open and stays whole in memory: "
            "for long takes use the folder format (--format web), which loads the take piece by piece")
    return out


def export_engine(run_dir: str | Path, out_dir: str | Path, frame_step: int = 1,
                  min_opacity: float = 0.02, log=print) -> Path:
    """Folder for a game engine: ``scene.json`` and one gzip-compressed binary file per chunk.

    The bytes of a chunk are those of the player format (see the top of this
    file). The description is flat and has only numbers, strings and arrays
    of numbers, so that the simplest JSON readers of the engines can take it:
    no nested arrays, no nulls, no names that are keywords somewhere.
    """
    run = Run(run_dir)
    out = Path(out_dir)
    if (out / "chunks").exists():
        shutil.rmtree(out / "chunks")
    (out / "chunks").mkdir(parents=True)
    chunks = []
    for i, (info, data) in enumerate(_chunks(run, min_opacity)):
        name = f"chunks/{i:04d}.bin"
        (out / name).write_bytes(data)
        chunks.append({
            "file": name, "bytes": len(data), "first": info["first"], "last": info["last"],
            "count": info["count"], "long_lived": info["long"],
            "reach": -1.0 if info["reach"] is None else info["reach"],  # -1: every Gaussian is always a candidate
            "bbox_min": info["bbox_min"], "bbox_max": info["bbox_max"],
            "time_start": info["time"][0], "time_span": info["time"][1], "vel_max": info["vel_max"],
            "life_log_min": info["life"][0], "life_log_max": info["life"][1],
            "scale_log_min": info["log_scale"][0], "scale_log_max": info["log_scale"][1],
            "alive_max": info["alive_max"],
        })
    static = run.static_frame is not None
    scene = {
        "format": "mister4d-engine", "version": 1,
        "num_frames": 1 if static else run.num_frames, "fps": run.meta["fps"],
        "min_opacity": min_opacity, "gaussians": sum(c["count"] for c in chunks),
        "alive_max": max(c["alive_max"] for c in chunks),
        "bbox_min": np.min([c["bbox_min"] for c in chunks], axis=0).tolist(),
        "bbox_max": np.max([c["bbox_max"] for c in chunks], axis=0).tolist(),
        "up_axis": "z", "handedness": "right", "units": "metres",
        "chunks": chunks,
    }
    (out / "scene.json").write_text(json.dumps(scene, indent=1))
    size_mb = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    seconds = scene["num_frames"] / scene["fps"]
    log(f"[export] engine folder: {scene['num_frames']} frames in {len(chunks)} chunk(s), "
        f"{scene['gaussians']} Gaussians, at most {scene['alive_max']} visible at once, {size_mb:.1f} MB"
        + (f", {size_mb / seconds:.1f} MB per second of take" if scene["num_frames"] > 1 else "") + f" -> {out}")
    return out


@torch.no_grad()
def export_ply(run_dir: str | Path, out_dir: str | Path, frame_step: int = 1,
               min_opacity: float = 0.02, log=print) -> Path:
    """One standard 3DGS .ply per frame (x y z, normals, f_dc, f_rest, opacity, scale, rot)."""
    run = Run(run_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = [run.static_frame] if run.static_frame is not None else range(0, run.num_frames, max(frame_step, 1))
    for f in frames:
        b = _alive(run.scene_at(f), f, min_opacity)
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
    """Serve an exported player folder on a local port (it also opens from disk with a double click)."""
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
