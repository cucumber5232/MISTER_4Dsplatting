"""The whole scene: the empty set as frozen scenery, the take trained on top of it, no masks."""

import json
import shutil

import cv2
import numpy as np
import pytest
import torch

from mister4d import cli
from mister4d.camera import Camera, Rig, look_at
from mister4d.dataset import MultiViewDataset, read_meta, read_points_ply
from mister4d.evaluate import evaluate_run
from mister4d.export import ALWAYS, export_web, pack_chunk
from mister4d.model import DynamicScene, init_gaussian_params, load_checkpoint, render_scene, scenery_of
from mister4d.scenery import change_masks, estimate_plates, sweep_points

quiet = lambda *_: None  # noqa: E731


def _plain_scene(n: int, static: bool, seed: int, colour: float, z: float, side: float = 1.0) -> DynamicScene:
    rng = np.random.default_rng(seed)
    xyz = (rng.random((n, 3)) - 0.5).astype(np.float32) * [side, side, 0.2] + [0.0, 0.0, z]
    params = init_gaussian_params(xyz.astype(np.float32), np.full((n, 3), colour), np.zeros(n), 0, init_opacity=0.8)
    return DynamicScene(params, 0, static=static)


def test_the_scenery_is_drawn_with_the_scene_and_left_out_of_what_is_trained():
    back = _plain_scene(300, True, 0, 0.2, 0.0, side=2.0)  # a wide floor ...
    scenery, degree = scenery_of(back)
    front = _plain_scene(200, False, 1, 0.9, 0.5, side=0.5)  # ... and something small above it
    both = DynamicScene(front.params, 0, scenery=scenery, scenery_degree=degree)
    assert both.num_gaussians == 200 and both.num_scenery == 300
    R, t = look_at(np.array([0.0, 0.0, 3.0]), np.zeros(3), up=(0.0, 1.0, 0.0))
    view = np.eye(4, dtype=np.float32)
    view[:3, :3], view[:3, 3] = R, t
    viewmats = torch.tensor(view)[None]
    Ks = torch.tensor([[[60.0, 0, 31.5], [0, 60.0, 23.5], [0, 0, 1]]])
    with torch.no_grad():
        alone, alpha_alone, _, info_alone, state_alone = render_scene(front, 0, viewmats, Ks, 64, 48)
        rgb, alpha, _, info, state = render_scene(both, 0, viewmats, Ks, 64, 48)
    assert state["means"].shape[0] == state_alone["means"].shape[0] == 200  # the state is what moves, only
    assert info["count"] == 200 and info["means2d"].shape[1] == 500 and "count" not in info_alone
    assert float(alpha.mean()) > float(alpha_alone.mean()) + 0.05  # the set fills what the take leaves empty
    # The take is in front: where it is solid the picture is the same with and without the set.
    solid = alpha_alone[..., 0] > 0.9
    assert solid.any() and float((rgb - alone)[solid].abs().max()) < 0.05


def test_a_scene_remembers_where_its_scenery_is(tmp_path):
    from mister4d.model import save_checkpoint

    back = _plain_scene(50, True, 0, 0.2, 0.0)
    (tmp_path / "scenery").mkdir()
    save_checkpoint(tmp_path / "scenery" / "ckpt.pt", back, {}, {}, 0)
    front = DynamicScene(_plain_scene(20, False, 1, 0.9, 0.5).params, 0, False, *scenery_of(back))
    save_checkpoint(tmp_path / "ckpt.pt", front, {}, {"scenery_file": "scenery/ckpt.pt"}, 0)
    assert "scenery" not in torch.load(tmp_path / "ckpt.pt", weights_only=True)  # one copy, in its own file
    loaded, _ = load_checkpoint(tmp_path / "ckpt.pt")
    assert loaded.num_gaussians == 20 and loaded.num_scenery == 50
    save_checkpoint(tmp_path / "whole.pt", front, {}, {}, 0)  # no file to point to: it travels inside
    assert load_checkpoint(tmp_path / "whole.pt")[0].num_scenery == 50
    (tmp_path / "scenery" / "ckpt.pt").unlink()
    with pytest.raises(FileNotFoundError, match="scenery"):
        load_checkpoint(tmp_path / "ckpt.pt")


def test_plates_are_taken_from_the_take_when_the_empty_set_was_not_recorded(tiny_dataset, tmp_path):
    data = tmp_path / "data"
    shutil.copytree(tiny_dataset, data)
    recorded = {p.stem: cv2.imread(str(p)) for p in (data / "background").iterdir()}
    assert estimate_plates(data, log=quiet) == "recorded"  # they are there: nothing to do
    shutil.rmtree(data / "background")
    assert estimate_plates(data, log=quiet) == "estimated" and read_meta(data)["plates"] == "estimated"
    for cam, plate in recorded.items():
        made = cv2.imread(str(data / "background" / f"{cam}.png"))
        close = np.abs(made.astype(np.float32) - plate.astype(np.float32)).max(axis=2) < 16
        assert close.mean() > 0.85  # the set, but for where the subject stood all the time
    change_masks(data, log=quiet)
    changed = cv2.imread(str(next((data / "change" / "cam01").iterdir())), cv2.IMREAD_GRAYSCALE)
    assert changed.shape == (72, 96) and 0.0 < (changed > 127).mean() < 0.5


@pytest.mark.slow
def test_plane_sweeps_put_the_set_where_a_camera_that_gave_no_points_sees_it():
    from mister4d.human import StudioView

    width, height, radius, target = 240, 135, 3.0, np.array([0.0, 0.0, 1.08])
    fy = height * radius / 2.36

    def camera(name, degrees):
        a = np.radians(degrees)
        R, t = look_at(np.array([radius * np.cos(a), radius * np.sin(a), 1.4]), target)
        return Camera(name, width, height, [[fy, 0, (width - 1) / 2], [0, fy, (height - 1) / 2], [0, 0, 1]],
                      np.zeros(5), R, t)

    cams = {f"cam{i + 1:02d}": camera(f"cam{i + 1:02d}", a) for i, a in enumerate((-75.0, -25.0, 25.0, 75.0))}
    check = camera("check", 0.0)  # in the middle of a gap of 50 degrees
    rig = Rig(cams)
    plates = {n: cv2.cvtColor((StudioView(c).background * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
              for n, c in {**cams, "check": check}.items()}
    xyz, rgb = sweep_points(rig, list(cams), plates, target, work_width=width, depths=384, height_step=0.02,
                            per_camera=20_000, log=quiet)
    assert len(xyz) > 20_000 and np.isfinite(xyz).all()
    # The studio is a floor and a wall six metres away: nearly all the points are inside it.
    assert (np.linalg.norm(xyz[:, :2], axis=1) < 8.0).mean() > 0.9 and (np.abs(xyz[:, 2] - 2.0) < 4.0).mean() > 0.9
    uv, depth = check.project(xyz.astype(np.float64))
    u, v = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
    ok = (depth > 0.05) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    seen = np.zeros((height, width), dtype=bool)
    seen[v[ok], u[ok]] = True
    assert seen.mean() > 0.6
    error = np.abs(rgb[ok][:, ::-1] * 255.0 - plates["check"][v[ok], u[ok]].astype(np.float32)).mean(axis=1)
    assert np.median(error) < 20.0  # 8-bit levels: the colours land where that camera has them


@pytest.mark.slow
def test_a_take_is_reconstructed_whole_without_masks(tiny_dataset, tmp_path):
    data = tmp_path / "data"
    shutil.copytree(tiny_dataset, data)
    shutil.rmtree(data / "masks")
    (data / "init_points.ply").unlink()
    assert cli.main(["init", str(data), "--full-scene", "--exclude", "cam04", "--points", "2000",
                     "--scenery-points", "3000"]) == 0
    assert not (data / "masks").exists() and (data / "change" / "cam01").is_dir()
    set_xyz, _, set_frames = read_points_ply(data / "scenery" / "init_points.ply")
    assert len(set_xyz) > 1500 and (set_frames == 0).all()
    assert MultiViewDataset(data / "scenery").num_frames == 1

    run = tmp_path / "run"
    options = ["--holdout", "cam04", "--devices", "cpu", "--iterations", "40", "--set", "views_per_step=3",
               "densify_start=15", "densify_every=10", "log_every=20", "eval_every=20", "scenery_iterations=60"]
    assert cli.main(["train", str(data), "--out", str(run), "--full-scene", *options]) == 0
    scenery_log = [json.loads(line) for line in (run / "scenery" / "train_log.jsonl").read_text().splitlines()]
    assert scenery_log[-1]["psnr_holdout"] > scenery_log[0]["psnr_holdout"]  # the set, from a camera not used

    scene, ckpt = load_checkpoint(run / "ckpt.pt")
    assert ckpt["meta"]["scenery_file"] == "scenery/ckpt.pt" and ckpt["config"]["full_scene"]
    assert scene.num_scenery > 1000 and scene.num_gaussians > 500 and not scene.static
    summary = json.loads((run / "train_summary.json").read_text())
    assert summary["scenery"] == scene.num_scenery and summary["gaussians"] == scene.num_gaussians

    # The whole image is covered, and the take adds the subject to what the set alone shows.
    ds = MultiViewDataset(data)
    viewmats, Ks = ds.camera_tensors(["cam01"], "cpu")
    gt, _ = ds.batch(2, ["cam01"], "cpu")
    with torch.no_grad():
        rgb, alpha, *_ = render_scene(scene, 2, viewmats, Ks, ds.width, ds.height)
        empty, *_ = render_scene(load_checkpoint(run / "scenery" / "ckpt.pt")[0], 0, viewmats, Ks, ds.width, ds.height)
    assert float(alpha.mean()) > 0.9
    assert float((rgb - gt).abs().mean()) < float((empty - gt).abs().mean())

    # Starting again keeps the scenery; the measures are on whole images.
    assert cli.main(["train", str(data), "--out", str(run), "--full-scene", *options]) == 0
    measures = evaluate_run(run, log=quiet)
    assert measures["holdout"]["psnr"] is not None and measures["holdout"]["iou"] is None

    # In the player format the scenery comes first and is always there.
    info, raw = pack_chunk(scene, 0, 5)
    n = info["count"]
    assert info["scenery"] > 1000 and info["long"] >= info["scenery"] and n > info["scenery"]
    life = np.frombuffer(raw, dtype=np.uint8)[8 * n:12 * n].reshape(n, 4)[:, 3]
    assert (life[:info["scenery"]] == ALWAYS).all() and (life[info["long"]:] != ALWAYS).all()
    web = export_web(run, tmp_path / "web", log=quiet)
    assert json.loads((web / "scene.json").read_text())["chunks"][0]["scenery"] == info["scenery"]
