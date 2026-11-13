import dataclasses
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
from mister4d.lighten import contribution, lighten, visible_at_once
from mister4d.masks import compute_masks
from mister4d.model import load_checkpoint
from mister4d.render import ping_pong, render_orbit
from mister4d.run import Run, plan_segments
from mister4d.trainer import TrainConfig, load_config, train

FAST = dict(iterations=60, views_per_step=3, densify_start=20, densify_every=10,
            log_every=20, eval_every=20, holdout=["cam04"])
quiet = lambda *_: None  # noqa: E731


def _log(run):
    return [json.loads(line) for line in (run / "train_log.jsonl").read_text().splitlines()]


def _chunk_bytes(text):
    """The bytes of a chunk of the player format, from its base64 text (inline or in a chunk script)."""
    import base64
    import gzip
    import re

    found = re.search(r'mister4dChunk\(\d+,"([^"]*)"\)', text)
    return gzip.decompress(base64.b64decode(found.group(1) if found else text))


def _decode_at(chunk, data, frame):
    """Position and opacity at ``frame`` of every Gaussian of a chunk: what the player computes."""
    n = chunk["count"]
    a = np.frombuffer(data, dtype="<u2", count=4 * n).reshape(n, 4).astype(np.float64)
    b = np.frombuffer(data, dtype=np.uint8, count=4 * n, offset=8 * n).reshape(n, 4).astype(np.float64)
    d = np.frombuffer(data, dtype=np.uint8, count=4 * n, offset=16 * n).reshape(n, 4).astype(np.float64)
    lo, hi = np.array(chunk["bbox_min"]), np.array(chunk["bbox_max"])
    dt = frame - (chunk["time"][0] + a[:, 3] / 65535.0 * chunk["time"][1])
    sigma = np.exp(chunk["life"][0] + (chunk["life"][1] - chunk["life"][0]) * b[:, 3] / 254.0)
    always = b[:, 3] == 255
    dt[always] = 0.0
    fade = np.where(always, 1.0, np.exp(-0.5 * (dt / sigma) ** 2))
    code = (b[:, :3] - 128.0) / 127.0
    pos = lo + a[:, :3] / 65535.0 * (hi - lo) + code * np.abs(code) * chunk["vel_max"] * dt[:, None]
    return pos, d[:, 3] / 255.0 * fade


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

    # Video for editing: the take played once, transparent background, PNG images or ProRes 4444.
    frames = render_orbit(run, tmp_path / "png", take=True, transparent=True, log=quiet)
    images = sorted(frames.glob("*.png"))
    assert len(images) == 5  # one per frame of the take
    rgba = cv2.imread(str(images[2]), cv2.IMREAD_UNCHANGED)
    assert rgba.shape[2] == 4 and 0 < (rgba[..., 3] > 127).mean() < 0.9  # subject opaque, the rest clear
    mov = render_orbit(run, tmp_path / "soggetto.mov", seconds=0.2, transparent=True, height=64, log=quiet)
    from mister4d.video import probe_video
    assert probe_video(mov).pix_fmt.startswith("yuva444") and probe_video(mov).height == 64
    with pytest.raises(Exception, match="transparent"):
        render_orbit(run, tmp_path / "no.mp4", seconds=0.1, transparent=True, log=quiet)
    keys = tmp_path / "path.json"
    keys.write_text(json.dumps([{"time": 0, "eye": [2.5, 0, 1.2], "target": [0, 0, 1], "fov": 40},
                                {"time": 0.2, "eye": [0, 2.5, 1.2], "target": [0, 0, 1], "fov": 30}]))
    assert render_orbit(run, tmp_path / "move.mp4", seconds=0.2, path=keys, log=quiet).stat().st_size > 1000

    web = export_web(run, tmp_path / "web", log=quiet)
    info = json.loads((web / "scene.json").read_text())
    assert info["version"] == 3 and info["num_frames"] == 5 and len(info["chunks"]) == 1
    assert (web / "index.html").exists() and (web / "viewer.js").exists() and (web / "scene.js").exists()
    assert 'src="scene.js"' in (web / "index.html").read_text()
    chunk = info["chunks"][0]
    data = _chunk_bytes((web / chunk["file"]).read_text())
    assert len(data) == 23 * chunk["count"]
    # Every Gaussian is stored once, and decodes back to what the model renders at a frame.
    scene, _ = load_checkpoint(run / "ckpt.pt")
    assert chunk["count"] <= scene.num_gaussians
    with torch.no_grad():
        state = scene.state_at(2)
    keep = state["opacities"] >= info["min_opacity"]
    pos, opacity = _decode_at(chunk, data, 2.0)
    shown = opacity >= info["min_opacity"]
    assert abs(int(shown.sum()) - int(keep.sum())) <= 0.02 * int(keep.sum()) + 2  # but for rounding at the threshold
    assert chunk["alive_max"] >= int(keep.sum()) - 2
    from scipy.spatial import cKDTree
    dist, nearest = cKDTree(state["means"][keep].numpy()).query(pos[shown])
    assert np.percentile(dist, 99) < 2e-3  # millimetres
    assert np.abs(opacity[shown] - state["opacities"][keep].numpy()[nearest]).mean() < 0.02

    # The single-file player carries the same player and the same bytes, inline.
    import re

    page = export_html(run, tmp_path / "player.html", log=quiet).read_text()
    assert 'src="viewer.js"' not in page and "function visibleBackToFront" in page
    inline = json.loads(re.search(r'<script id="mister4d-scene"[^>]*>(.*?)</script>', page, re.S).group(1))
    assert inline["chunks"][0]["count"] == chunk["count"] and "file" not in inline["chunks"][0]
    embedded = re.findall(r'<script id="mister4d-chunk-(\d+)"[^>]*>(.*?)</script>', page, re.S)
    assert [int(i) for i, _ in embedded] == [0]
    assert _chunk_bytes(embedded[0][1]) == data

    ply = export_ply(run, tmp_path / "ply", frame_step=2, log=quiet)
    files = sorted(ply.glob("*.ply"))
    assert len(files) == 3
    header = files[0].read_bytes()[:500].decode("ascii", errors="ignore")
    assert "element vertex" in header and "property float rot_3" in header


@pytest.mark.slow
def test_time_offsets_are_learned_stored_and_used(tiny_dataset, tmp_path):
    run = tmp_path / "run"
    train(tiny_dataset, run, TrainConfig(devices="cpu", learn_time_offsets=True, **FAST))
    _, ckpt = load_checkpoint(run / "ckpt.pt")
    offsets = ckpt["meta"]["time_offsets"]
    assert list(offsets) == ["cam01", "cam02", "cam03"] and offsets["cam01"] == 0.0  # the reference
    assert all(abs(v) < 1.0 for v in offsets.values())
    assert any(v != 0.0 for v in offsets.values())
    assert ckpt["meta"]["coverage"]["largest_gap_deg"] > 0
    summary = evaluate_run(run, log=quiet)
    assert summary["train"]["flicker"] is not None and "coverage" in (run / "eval" / "summary.md").read_text()


def test_segments_have_equal_lengths():
    assert plan_segments(40, 60) == [(0, 40)]
    assert plan_segments(90, 60) == [(0, 90)]  # one and a half segments: still one piece
    assert plan_segments(150, 60) == [(0, 75), (75, 150)]
    assert plan_segments(1800, 60)[-1] == (1740, 1800) and len(plan_segments(1800, 60)) == 30
    assert plan_segments(300, 0) == [(0, 300)]  # never split


@pytest.mark.slow
def test_a_long_take_is_trained_in_segments_and_continues_where_it_stopped(tiny_dataset, tmp_path, capsys):
    run = tmp_path / "run"
    cfg = TrainConfig(devices="cpu", segment_frames=2, segment_overlap=1, **{**FAST, "iterations": 30})
    assert train(tiny_dataset, run, cfg).name == "run.json"
    manifest = json.loads((run / "run.json").read_text())
    assert [(s["first"], s["last"]) for s in manifest["segments"]] == [(0, 2), (2, 5)]
    assert not (run / "ckpt.pt").exists()
    first, _ = load_checkpoint(run / "seg_000" / "ckpt.pt")
    assert first.params["times"].max() < 2.5  # its own frames plus one of context, not the rest of the take
    total = json.loads((run / "train_summary.json").read_text())
    assert total["frames"] == 5 and total["segments"] == 2 and total["iterations"] == 60

    take = Run(run)
    assert take.num_frames == 5 and len(take.segments) == 2 and take.index_of(1) == 0 and take.index_of(2) == 1
    assert take.is_cut(2) and not take.is_cut(1)
    assert take.num_gaussians == total["gaussians"]
    assert all((take.scene_at(f).state_at(f)["opacities"] > 0.02).sum() > 50 for f in range(5))

    # Reading the result does not depend on the segments.
    summary = evaluate_run(run, log=quiet)
    assert summary["segments"] == 2 and summary["train"]["flicker_at_cuts"] is not None
    web = export_web(run, tmp_path / "web", log=quiet)
    info = json.loads((web / "scene.json").read_text())
    assert [(c["first"], c["last"]) for c in info["chunks"]] == [(0, 2), (2, 5)]
    assert all((web / c["file"]).exists() for c in info["chunks"])
    assert len(list(export_ply(run, tmp_path / "ply", log=quiet).glob("*.ply"))) == 5
    assert render_orbit(run, tmp_path / "take.mp4", take=True, log=quiet).stat().st_size > 1000

    # The same command again keeps what is there; an interrupted job loses only its last segment.
    stamp = (run / "seg_000" / "ckpt.pt").stat().st_mtime_ns
    (run / "seg_001" / "ckpt.pt").unlink()
    (run / "run.json").unlink()
    with pytest.raises(FileNotFoundError, match="did not finish"):
        Run(run)
    capsys.readouterr()
    train(tiny_dataset, run, cfg)
    assert "segment 1 is already trained" in capsys.readouterr().out
    assert (run / "seg_000" / "ckpt.pt").stat().st_mtime_ns == stamp and (run / "seg_001" / "ckpt.pt").exists()
    # Other options, or data prepared again, and nothing is kept.
    train(tiny_dataset, run, dataclasses.replace(cfg, lambda_dssim=0.1))
    assert (run / "seg_000" / "ckpt.pt").stat().st_mtime_ns != stamp


@pytest.mark.slow
def test_a_lighter_copy_keeps_what_contributes_most(tiny_dataset, tmp_path):
    run = tmp_path / "run"
    train(tiny_dataset, run, TrainConfig(devices="cpu", **FAST))
    scene, ckpt = load_checkpoint(run / "ckpt.pt")
    before = visible_at_once(scene, 0, 5)

    # The score of a Gaussian is what it adds to the images: none for those that cannot be seen.
    ds = MultiViewDataset(tiny_dataset)
    score = contribution(scene, ds, ckpt["meta"]["train_cameras"], range(5), "cpu", "torch")
    assert score.shape == (scene.num_gaussians,) and float(score.min()) >= 0 and float(score.max()) > 0
    hidden = scene.params["opacities"].detach() < -8
    assert float(score[hidden].sum()) <= 1e-3 * float(score.sum())

    budget = before // 2
    light = tmp_path / "light"
    assert lighten(run, light, budget, iterations=20, devices="cpu", log=quiet).name == "ckpt.pt"
    small, _ = load_checkpoint(light / "ckpt.pt")
    assert visible_at_once(small, 0, 5) <= budget  # the budget is a promise
    assert small.num_gaussians < 0.7 * scene.num_gaussians
    assert all(small.state_at(f)["active"].numel() > 50 for f in range(5))  # no instant left empty
    # It is a take like any other: it can be measured, exported, lightened again.
    assert evaluate_run(light, log=quiet)["holdout"]["iou"] > 0.4
    info = json.loads((export_web(light, tmp_path / "web", log=quiet) / "scene.json").read_text())
    assert info["alive_max"] <= budget
    with pytest.raises(ValueError, match="folder of its own"):
        lighten(run, run, budget, devices="cpu", log=quiet)


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
    info = json.loads((web / "scene.json").read_text())
    assert info["num_frames"] == 1 and info["static"] and info["chunks"][0]["long"] == info["chunks"][0]["count"]
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
