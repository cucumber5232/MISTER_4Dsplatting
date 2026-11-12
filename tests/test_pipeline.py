import json
import shutil

import cv2
import numpy as np
import pytest
import torch

from mister4d.dataset import MultiViewDataset, read_points_ply
from mister4d.evaluate import evaluate_run
from mister4d.export import export_html, export_ply, export_web
from mister4d.hull import visual_hull
from mister4d.masks import compute_masks
from mister4d.model import load_checkpoint
from mister4d.render import ping_pong, render_orbit
from mister4d.trainer import TrainConfig, load_config, train

FAST = dict(iterations=60, views_per_step=3, densify_start=20, densify_every=10,
            log_every=20, eval_every=20, holdout=["cam04"])
quiet = lambda *_: None  # noqa: E731


def _log(run):
    return [json.loads(line) for line in (run / "train_log.jsonl").read_text().splitlines()]


@pytest.mark.slow
@pytest.mark.parametrize("devices", ["cpu", "cpu,cpu"])
def test_training_improves_held_out_view(tiny_dataset, tmp_path, devices):
    """One device, and the multi-device code path with two (virtual) devices."""
    run = tmp_path / "run"
    train(tiny_dataset, run, TrainConfig(devices=devices, **FAST))
    log = _log(run)
    assert log[-1]["psnr_holdout"] > log[0]["psnr_holdout"] + 0.5
    assert log[-1]["loss"] < log[0]["loss"]
    scene, ckpt = load_checkpoint(run / "ckpt.pt")
    assert not scene.static and ckpt["meta"]["holdout_cameras"] == ["cam04"]
    # Every frame of the take is described by some Gaussians.
    assert all(scene.state_at(f)["active"].numel() > 100 for f in range(5))


@pytest.mark.slow
def test_evaluate_render_and_export(tiny_dataset, tmp_path):
    run = tmp_path / "run"
    train(tiny_dataset, run, TrainConfig(devices="cpu", **FAST))

    summary = evaluate_run(run, log=quiet)
    assert summary["holdout"]["psnr"] is not None and summary["holdout"]["iou"] > 0.5
    assert "No camera was held out" not in (run / "eval" / "summary.md").read_text()

    video = render_orbit(run, tmp_path / "orbit.mp4", seconds=0.3, log=quiet)
    assert video.stat().st_size > 1000

    web = export_web(run, tmp_path / "web", log=quiet)
    info = json.loads((web / "scene.json").read_text())
    slots = info["tex_width"] * info["tex_height"]
    assert info["num_frames"] == 5 and len(info["counts"]) == 5 and max(info["counts"]) <= slots
    for name in info["frames"]:
        assert (web / name).stat().st_size == slots * 17
    assert (web / "index.html").exists() and (web / "viewer.js").exists()
    # Positions and opacities decode back to what the model renders at that frame.
    scene, _ = load_checkpoint(run / "ckpt.pt")
    with torch.no_grad():
        state = scene.state_at(2)
    keep = state["opacities"] >= 0.02
    count = info["counts"][2]
    assert count == int(keep.sum())
    raw = (web / info["frames"][2]).read_bytes()
    lo, hi = np.array(info["bbox_min"]), np.array(info["bbox_max"])
    pos = np.frombuffer(raw, dtype=np.uint16, count=slots * 3).reshape(-1, 3)[:count]
    assert np.abs(pos / 65535.0 * (hi - lo) + lo - state["means"][keep].numpy()).max() < 1e-3
    rgba = np.frombuffer(raw, dtype=np.uint8, offset=slots * 13).reshape(-1, 4)[:count]
    assert np.abs(rgba[:, 3] / 255.0 - state["opacities"][keep].numpy()).max() < 0.01

    # The single-file player carries the same viewer and the same bytes, inline.
    import base64
    import re

    page = export_html(run, tmp_path / "player.html", log=quiet).read_text()
    assert 'src="viewer.js"' not in page and "function sortBackToFront" in page
    assert json.loads(re.search(r'<script id="mister4d-scene"[^>]*>(.*?)</script>', page, re.S).group(1)) == info
    embedded = re.findall(r'<script id="mister4d-frame-(\d+)"[^>]*>(.*?)</script>', page, re.S)
    assert [int(i) for i, _ in embedded] == [0, 1, 2, 3, 4]
    assert base64.b64decode(embedded[2][1]) == raw

    ply = export_ply(run, tmp_path / "ply", frame_step=2, log=quiet)
    files = sorted(ply.glob("*.ply"))
    assert len(files) == 3
    header = files[0].read_bytes()[:500].decode("ascii", errors="ignore")
    assert "element vertex" in header and "property float rot_3" in header


def test_static_mode_and_config_overrides(tiny_dataset, tmp_path):
    cfg = load_config(None, ["static=true", "iterations=8", "holdout=cam04", "devices=cpu",
                             "densify=false", "log_every=4"])
    assert cfg.static and cfg.holdout == ["cam04"]
    train(tiny_dataset, tmp_path / "run", cfg)
    scene, ckpt = load_checkpoint(tmp_path / "run" / "ckpt.pt")
    assert scene.static and ckpt["meta"]["static_frame"] == 2
    _, _, frames = read_points_ply(tiny_dataset / "init_points.ply")
    assert scene.num_gaussians == int((frames == 2).sum())  # only the points of that instant
    web = export_web(tmp_path / "run", tmp_path / "web", log=quiet)
    assert json.loads((web / "scene.json").read_text())["num_frames"] == 1
    with pytest.raises(ValueError, match="unknown config keys"):
        load_config(None, ["no_such_option=1"])


def test_masks_from_clean_plate_match_the_truth(tiny_dataset, tmp_path):
    data = tmp_path / "data"
    shutil.copytree(tiny_dataset, data)
    truth = {p: cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in (data / "masks").rglob("*.png")}
    shutil.rmtree(data / "masks")
    compute_masks(data, log=quiet)
    ious = []
    for path, gt in truth.items():
        est = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127
        ious.append((est & (gt > 127)).sum() / max((est | (gt > 127)).sum(), 1))
    assert np.mean(ious) > 0.8
    assert (data / "masks_preview.jpg").exists()


def test_visual_hull_follows_the_subject_frame_by_frame(tiny_dataset, tmp_path):
    data = tmp_path / "data"
    shutil.copytree(tiny_dataset, data)
    body, _, body_frames = read_points_ply(data / "init_points.ply")  # points on the true surface
    visual_hull(data, voxel=0.03, exclude=["cam04"], total_points=5000, log=quiet)
    hull, colours, frames = read_points_ply(data / "init_points.ply")
    meta = json.loads((data / "meta.json").read_text())
    assert meta["init_cameras"] == ["cam01", "cam02", "cam03"]
    assert sorted(set(frames.tolist())) == [0, 1, 2, 3, 4]
    assert colours.min() >= 0 and colours.max() <= 1
    lo, hi = np.array(meta["bbox_min"]), np.array(meta["bbox_max"])
    assert ((body >= lo) & (body <= hi)).all(axis=1).mean() > 0.97
    from scipy.spatial import cKDTree

    for f in (0, 4):
        # Every part of the true body, the raised hand included, is inside the hull of its frame.
        gap, _ = cKDTree(hull[frames == f]).query(body[body_frames == f])
        assert gap.max() < 0.12, f"frame {f}: hull misses part of the subject"
    # The hand moves between the first and the last frame, and the hull moves with it.
    top = [hull[frames == f][:, 2].max() for f in (0, 4)]
    assert top[1] > top[0] + 0.1
    assert MultiViewDataset(data).scene_bounds()[1] < 2.0


def test_ping_pong_has_no_jump():
    assert [ping_pong(i, 4) for i in range(9)] == [0, 1, 2, 3, 2, 1, 0, 1, 2]
    assert ping_pong(5, 1) == 0
