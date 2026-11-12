"""The whole pipeline on a synthetic capture session, in one call.

Works the same on Linux and Windows: no shell script, no cameras. With a GPU
it runs at full resolution with the default training options; without one it
falls back to small images and the CPU reference rasterizer (a few minutes).
"""

from __future__ import annotations

from pathlib import Path

import torch

from .charuco import BoardSpec, calibrate_rig
from .evaluate import evaluate_run
from .export import export_html
from .hull import visual_hull
from .ingest import ingest_clip
from .masks import compute_masks
from .prepare import prepare_dataset
from .render import render_orbit
from .synth import SynthConfig, make_raw_session
from .trainer import TrainConfig, train

SKIP_AFTER_SYNC = 8  # frames dropped after the clap / flash used for synchronisation


def run_demo(out_dir: str | Path, cfg: SynthConfig, holdout: str = "cam02", log=print) -> Path:
    out = Path(out_dir)
    gpu = torch.cuda.is_available()

    make_raw_session(out, cfg, log=log)

    # 1. Check the files, align the cameras in time, extract frames.
    for clip, sync in (("calib", "audio"), ("empty", "none"), ("take", "audio")):
        ingest_clip(sorted((out / "raw" / clip).glob("*.mp4")), out / "work" / clip, sync=sync,
                    start=SKIP_AFTER_SYNC if sync != "none" else 0, log=log)

    # 2. Calibrate the rig from the ChArUco clip.
    calibration = out / "work" / "calibration.json"
    calibrate_rig(out / "work" / "calib" / "frames", BoardSpec(), log=log).save(calibration)

    # 3. Undistort / resize, masks from the empty set, visual hull of every frame.
    data = out / "data"
    prepare_dataset(out / "work" / "take", calibration, data, empty_dir=out / "work" / "empty",
                    downscale=1 if gpu else 4, ext="png", log=log)
    compute_masks(data, log=log)
    visual_hull(data, exclude=[holdout], total_points=300_000 if gpu else 60_000, log=log)

    # 4. Train with one camera held out, then measure on it.
    if gpu:
        train_cfg = TrainConfig(holdout=[holdout])
    else:
        train_cfg = TrainConfig(holdout=[holdout], devices="cpu", views_per_step=3, iterations=1500,
                                densify_start=400, max_gaussians=80_000, eval_every=500)
    run = out / "run"
    train(data, run, train_cfg)
    evaluate_run(run, log=log)

    # 5. Outputs.
    render_orbit(run, out / "orbit.mp4", seconds=6, scale=1 if gpu else 2, log=log)
    player = export_html(run, out / "player.html", log=log)
    log("")
    log(f"Report:  {run / 'eval' / 'summary.md'}")
    log(f"Video:   {out / 'orbit.mp4'}")
    log(f"Player:  {player}  (open it with a double click)")
    return player
