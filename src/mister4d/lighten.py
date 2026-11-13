"""A lighter copy of a trained take, for devices that cannot draw the full one.

A stand-alone headset draws far fewer Gaussians per frame than a workstation.
Training to a small budget from the start would waste the work already done;
dropping Gaussians from the finished scene without telling the others leaves
holes. So:

1. every Gaussian gets a score: how much it actually contributes to the
   images of the training cameras (the sum of its blending weights over all
   pixels, measured, not guessed from its size);
2. in every instant the same share of Gaussians, the least useful ones, is
   dropped, until no more than the budget is visible at once;
3. the survivors are refined for a short while against the footage, so that
   they take over what the dropped ones did.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import MultiViewDataset
from .model import MIN_DURATION, load_checkpoint, save_checkpoint
from .rasterizer import opencv_to_render_K, render
from .run import Run, segment_dir, write_manifest
from .trainer import TrainConfig, _train_window, pick_devices, warmup_backends


def _reach(scene, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Instant of every Gaussian and for how many frames around it its opacity stays above ``floor``."""
    p = scene.params
    opacity = torch.sigmoid(p["opacities"]).detach().cpu().numpy().astype(np.float64)
    if scene.static:
        return np.zeros_like(opacity), np.where(opacity >= floor, np.inf, -1.0)
    time_ = p["times"].detach().cpu().numpy().astype(np.float64)
    sigma = np.maximum(np.exp(p["durations"].detach().cpu().numpy().astype(np.float64)), MIN_DURATION)
    reach = sigma * np.sqrt(2.0 * np.log(np.maximum(opacity / floor, 1.0)))
    return time_, np.where(opacity >= floor, reach, -1.0)


def visible_at_once(scene, first: int, last: int, floor: float = 0.02, keep: np.ndarray | None = None) -> int:
    """The largest number of Gaussians visible in one frame of ``[first, last)``."""
    time_, reach = _reach(scene, floor)
    if keep is not None:
        time_, reach = time_[keep], reach[keep]
    return max(int(np.count_nonzero(np.abs(f - time_) <= reach)) for f in range(first, last))


def contribution(scene, ds: MultiViewDataset, cams: list[str], frames, device, backend: str,
                 antialiased: bool = False) -> torch.Tensor:
    """How much every Gaussian contributes to the images of ``cams`` over ``frames``.

    A rendered pixel is the sum of the colours of the Gaussians covering it,
    each multiplied by its blending weight. The derivative of the sum of all
    pixels with respect to the colour of a Gaussian is therefore exactly the
    sum of its weights: one backward pass per instant measures them all.
    """
    view = scene.view(device)
    viewmats, Ks = ds.camera_tensors(cams, device)
    Ks = opencv_to_render_K(Ks)
    score = torch.zeros(scene.num_gaussians, device=device)
    for frame in frames:
        with torch.no_grad():
            state = view.state_at(frame)
        if state["means"].shape[0] == 0:
            continue
        colors = torch.ones(state["means"].shape[0], 3, device=device, requires_grad=True)
        rgb, *_ = render(state["means"], state["quats"], state["scales"], state["opacities"], colors,
                         viewmats, Ks, ds.width, ds.height, backend=backend, antialiased=antialiased)
        (grad,) = torch.autograd.grad(rgb.sum(), colors)
        weight = grad[:, 0]
        if state["active"] is None:
            score += weight
        else:
            score[state["active"]] += weight
    return score.cpu()


def choose(scene, score: torch.Tensor, first: int, last: int, budget: int, floor: float = 0.02) -> np.ndarray:
    """Which Gaussians to keep so that at most ``budget`` are visible at once: the best of every instant."""
    n = scene.num_gaussians
    keep = np.ones(n, dtype=bool)
    time_, _ = _reach(scene, floor)
    bins = np.round(time_).astype(np.int64)
    score = score.numpy()
    share = 1.0
    for _ in range(6):
        peak = visible_at_once(scene, first, last, floor, keep)
        if peak <= budget:
            break
        share *= budget / peak * 0.98
        keep[:] = False
        for b in np.unique(bins):
            members = np.nonzero(bins == b)[0]
            count = max(1, int(np.ceil(share * len(members))))
            keep[members[np.argsort(-score[members], kind="stable")[:count]]] = True
    return keep


def lighten(run_dir: str | Path, out_dir: str | Path, budget: int, data_dir: str | Path | None = None,
            iterations: int = -1, devices: str = "auto", log=print) -> Path:
    """Write to ``out_dir`` a copy of the take with at most ``budget`` Gaussians visible at once.

    ``iterations`` is the length of the refinement of every segment (default:
    15% of its training). Returns the checkpoint or the manifest, like ``train``.
    """
    if budget < 100:
        raise ValueError("the budget is the number of Gaussians visible at once: at least 100")
    src = Run(run_dir, "cpu", keep=1)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.resolve() == Path(run_dir).resolve():
        raise ValueError("the lighter copy needs a folder of its own")
    data_dir = Path(data_dir or src.meta["data_dir"])
    ds = MultiViewDataset(data_dir)
    fields = {f.name for f in dataclasses.fields(TrainConfig)}
    base = TrainConfig(**{k: v for k, v in src.config.items() if k in fields})
    devs = pick_devices(devices)
    backend = warmup_backends(devs, base.backend)[0]
    cams = src.meta["train_cameras"]
    many = len(src.segments) > 1
    (out_dir / "run.json").unlink(missing_ok=True)
    t_start = time.time()
    done, summaries = [], []
    for i in range(len(src.segments)):
        scene, ckpt, first, last = src.segment(i)
        meta = ckpt["meta"]
        seg = segment_dir(out_dir, i) if many else out_dir
        seg.mkdir(parents=True, exist_ok=True)
        stat = (src.segments[i][0] / "ckpt.pt").stat()
        steps = iterations if iterations > 0 else max(300, int(0.15 * int(ckpt["config"].get("iterations", 7000))))
        tag = {"source": f"{stat.st_size}:{stat.st_mtime_ns}", "budget": int(budget), "iterations": steps}
        tag_file, summary_file = seg / "segment.json", seg / "train_summary.json"
        if many:
            log(f"[lighten] segment={i + 1}/{len(src.segments)} frames={first}-{last - 1}")
        if (seg / "ckpt.pt").exists() and tag_file.exists() and summary_file.exists() \
                and json.loads(tag_file.read_text()) == tag:
            summary = json.loads(summary_file.read_text())
            log(f"[lighten] segment {i + 1} is already done: kept")
        else:
            tag_file.unlink(missing_ok=True)
            frames = [meta["static_frame"]] if scene.static else range(first, last)
            before = visible_at_once(scene, first, last)
            score = contribution(scene, ds, cams, frames, devs[0], backend, bool(base.antialiased))
            # A little under the budget: the refinement makes some Gaussians last longer.
            keep = choose(scene, score, first, last, int(budget * 0.92))
            log(f"[lighten] visible at once {before} -> {visible_at_once(scene, first, last, keep=keep)}, "
                f"Gaussians {scene.num_gaussians} -> {int(keep.sum())}; refining for {steps} steps")
            start = {k: v.detach()[torch.from_numpy(keep)] for k, v in scene.params.items()}
            cfg = dataclasses.replace(base, iterations=steps, devices=devices, densify=False,
                                      eval_every=max(steps // 2, 1))
            offsets = meta.get("time_offsets") or None
            path, summary = _train_window(data_dir, seg, cfg, window=tuple(meta.get("window") or (first, last)),
                                          core=(first, last), fixed_offsets=offsets, start=start)
            refined, saved = load_checkpoint(path)
            if visible_at_once(refined, first, last) > budget:  # the budget is a promise: trim what went over
                score = contribution(refined, ds, cams, frames, devs[0], backend, bool(base.antialiased))
                keep = torch.from_numpy(choose(refined, score, first, last, budget))
                refined.params = torch.nn.ParameterDict(
                    {k: torch.nn.Parameter(v.detach()[keep]) for k, v in refined.params.items()})
                save_checkpoint(path, refined, saved["config"], saved["meta"], saved["step"])
                summary["gaussians"] = refined.num_gaussians
                summary_file.write_text(json.dumps(summary, indent=2))
            log(f"[lighten] done: at most {visible_at_once(refined, first, last)} Gaussians visible at once")
            tag_file.write_text(json.dumps(tag, indent=2))
        summaries.append(summary)
        done.append({"dir": seg.name, "first": first, "last": last, "gaussians": summary["gaussians"]})
    if not many:
        return out_dir / "ckpt.pt"
    total = {**summaries[0], "iterations": sum(x["iterations"] for x in summaries),
             "seconds": round(time.time() - t_start, 1), "gaussians": sum(x["gaussians"] for x in summaries),
             "frames": src.num_frames, "segments": len(done)}
    (out_dir / "train_summary.json").write_text(json.dumps(total, indent=2))
    return write_manifest(out_dir, done, frames=src.num_frames, budget=int(budget))
