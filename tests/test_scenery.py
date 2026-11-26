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
from mister4d.scenery import (RigGuard, _on_envelope, change_masks, envelope_of, estimate_plates, in_the_rig,
                              inside_envelope, sweep_points)

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


def test_the_set_is_trained_without_decay_and_shared_by_takes_that_differ_in_their_own_options():
    import dataclasses

    from mister4d.trainer import TrainConfig, scenery_config

    cameras = ["cam01", "cam02", "ver01"]
    plain = TrainConfig(full_scene=True, holdout=["ver01", "ver09"], scenery_iterations=900)
    own = scenery_config(plain, cameras)
    assert plain.opacity_decay > 0 and own.opacity_decay == 0.0  # the take fades what is not needed, the set not
    assert own.static and not own.full_scene and own.iterations == 900 and own.holdout == ["ver01"]
    assert own.max_gaussians == plain.scenery_max_gaussians and own.max_scale == plain.scenery_max_scale
    other = dataclasses.replace(plain, opacity_decay=0.0, opacity_decay_logit=True, lambda_opacity_entropy=0.005,
                                lambda_ray_spread=0.1, init_opacity=0.3, iterations=123)
    assert scenery_config(other, cameras) == own  # the same set: it is trained once
    assert scenery_config(dataclasses.replace(plain, scenery_opacity_decay=0.002), cameras).opacity_decay == 0.002
    assert scenery_config(dataclasses.replace(plain, lambda_needle=0.0), cameras) != own  # this shapes the set too


def test_the_test_rig_can_have_cameras_that_look_down_on_the_subject():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "make_benchmark", Path(__file__).resolve().parents[1] / "scripts" / "make_benchmark.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    rig, train, verify = bench.bench_rig(160, 90, train=4, arc=150.0, high=2)
    assert train == [f"cam{i:02d}" for i in range(1, 7)] and set(rig.cameras) == set(train) | set(verify)
    low, _, _ = bench.bench_rig(160, 90, train=4, arc=150.0)
    assert all(np.allclose(rig.cameras[n].t, low.cameras[n].t) for n in low.cameras)  # the others do not move
    for name in train[4:]:
        cam = rig.cameras[name]
        eye = -cam.R.T @ cam.t
        assert abs(eye[2] - 2.6) < 1e-6 and abs(np.hypot(eye[0], eye[1]) - bench.RADIUS) < 1e-6
        uv, depth = cam.project(np.array([bench.TARGET, [0.0, 0.0, 0.05], [0.0, 0.0, 1.75]]))
        assert (depth > 0).all() and np.allclose(uv[0], [cam.K[0][2], cam.K[1][2]], atol=1e-6)
        assert (uv[:, 0] > 0).all() and (uv[:, 0] < 159).all() and (uv[:, 1] > 0).all() and (uv[:, 1] < 89).all()


def _room(rng, wall_from=-180.0, wall_to=180.0):
    """Points on the floor of a round room of 5 m and on its wall, in the directions asked for."""
    a, r = rng.uniform(-np.pi, np.pi, 6000), 5.0 * np.sqrt(rng.uniform(0, 1, 6000))
    floor = np.stack([r * np.cos(a), r * np.sin(a), rng.normal(0.0, 0.004, 6000)], axis=1)
    a = np.radians(rng.uniform(wall_from, wall_to, 6000))
    wall = np.stack([5.0 * np.cos(a), 5.0 * np.sin(a), rng.uniform(0.0, 3.0, 6000)], axis=1)
    return np.concatenate([floor, wall])


def test_the_envelope_of_a_set_is_its_floor_and_its_walls_also_where_no_point_shows_them():
    rng = np.random.default_rng(0)
    cameras = np.array([[2.0 * np.cos(a), 2.0 * np.sin(a), 1.4] for a in np.radians([-60.0, 0.0, 60.0])])
    points = _room(rng, -180.0, 90.0)  # a quarter of the wall gave no point
    a, r = rng.uniform(-np.pi, np.pi, 60), rng.uniform(8.0, 20.0, 60)  # wrong matches, far and scattered
    stray = np.stack([r * np.cos(a), r * np.sin(a), rng.uniform(0.0, 3.0, 60)], axis=1)
    in_front = np.array([[2.6, 0.2 * k, 1.0] for k in range(-3, 4)])  # a few wrong points in mid air
    env = envelope_of(np.concatenate([points, stray, in_front]), np.zeros(3), cameras)
    assert abs(env["floor"]) < 0.02 and env["ring"] == pytest.approx(2.0) and env["top"] == pytest.approx(2.4)
    assert min(env["reach"]) > 4.8 and max(env["reach"]) < 5.3  # neither the stray points nor the gap move it

    # A ray is carried to where it leaves the set: at the floor, or at the wall.
    origin = cameras[1]
    rays = np.array([[[-1.0, 0.0, 0.0], [-1.0, 0.0, -0.5], [-1.0, 0.0, 0.2], [0.0, 0.0, -1.0]]])
    depth = _on_envelope(rays, origin, env, far=50.0)[0]
    assert abs(depth[0] - (2.0 + env["reach"][0])) < 0.15  # level, across the room, to the wall beyond
    assert abs(depth[1] - 2.8) < 0.05 and abs(depth[3] - 1.4) < 0.02  # down to the floor, 1.4 m below
    assert abs(depth[2] - depth[0]) < 0.15  # upwards: the wall again, there is no ceiling to stop at

    inside = inside_envelope(env, np.array([[0.0, 0.0, 1.0], [4.95, 0.0, 1.0], [0.0, 0.0, 0.1], [7.0, 0.0, 1.0]]), 0.3)
    assert inside.tolist() == [True, False, False, False]

    # With nothing above the floor the set reaches as far as its floor is seen; with no point at
    # all, twice as far as the cameras.
    a, r = rng.uniform(-np.pi, np.pi, 40000), 5.0 * np.sqrt(rng.uniform(0, 1, 40000))
    bare = envelope_of(np.stack([r * np.cos(a), r * np.sin(a), np.zeros(40000)], axis=1), np.zeros(3), cameras)
    assert 4.5 < min(bare["reach"]) and max(bare["reach"]) < 5.3
    assert envelope_of(np.zeros((0, 3)), np.zeros(3), cameras)["reach"][0] == pytest.approx(4.0)


def test_between_the_cameras_an_empty_set_keeps_only_what_three_cameras_agree_on(tmp_path):
    env = {"center": [0.0, 0.0], "floor": 0.0, "reach": [6.0] * 72, "ring": 3.0, "top": 3.4}
    lamp = np.array([[1.0, 1.0, 1.0 + 0.02 * k] for k in range(20)])  # a pole the cameras agree on
    guard = RigGuard(env, lamp)
    points = np.array([[0.0, 0.0, 1.0],    # where the subject stands: nothing there
                       [2.0, 0.0, 1.5],    # in front of a camera: a veil
                       [1.05, 1.0, 1.2],   # on the pole
                       [0.0, 0.0, 0.1],    # the floor
                       [5.9, 0.0, 1.4],    # the wall, behind the cameras
                       [4.5, 0.0, 1.4],    # beyond the ring: hides nothing the cameras look at
                       [0.0, 0.0, 4.0]])   # high above the rig
    assert in_the_rig(env, points).tolist() == [True, True, True, False, False, False, False]
    assert guard.stray(points).tolist() == [True, True, False, False, False, False, False]
    # A wall nearer than the ring (the subject in front of a backdrop): the backdrop is kept.
    near = {**env, "reach": [2.0] * 72}
    assert not in_the_rig(near, np.array([[1.95, 0.0, 1.0]]))[0] and in_the_rig(near, np.array([[1.5, 0.0, 1.0]]))[0]
    guard.save(tmp_path)
    again = RigGuard.load(tmp_path, env)
    assert (again.stray(points) == guard.stray(points)).all()
    assert RigGuard.load(tmp_path / "nothing", env).stray(points).tolist() == [True, True, True] + [False] * 4


def test_the_orbit_of_the_video_can_stay_inside_the_arc_of_the_cameras():
    from mister4d.render import orbit_path

    center = np.array([0.0, 0.0, 1.0])
    cams = []
    for i, degrees in enumerate(np.linspace(-90.0, 90.0, 7)):
        a = np.radians(degrees)
        R, t = look_at(center + [3.0 * np.cos(a), 3.0 * np.sin(a), 0.4], center)
        cams.append(Camera(f"cam{i}", 64, 48, [[60.0, 0, 31.5], [0, 60.0, 23.5], [0, 0, 1]], np.zeros(5), R, t))

    def reach(margin):
        eyes = np.array([-R.T @ t for R, t in orbit_path(cams, center, 200, margin)])
        return np.degrees(np.arctan2(eyes[:, 1], eyes[:, 0]))

    assert abs(reach(10.0).max() - 100.0) < 0.5 and abs(reach(10.0).min() + 100.0) < 0.5
    assert abs(reach(-5.0).max() - 85.0) < 0.5 and abs(reach(-5.0).min() + 85.0) < 0.5


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
    # Hardly any is left in mid air where the subject and the cameras are: there it would be a veil.
    from mister4d.human import _stand

    radius = np.hypot(xyz[:, 0], xyz[:, 1])
    off = np.minimum(np.abs(xyz[:, 2]), np.abs(radius - 6.0))
    for middle, R, axes, *_ in _stand():  # the lamp behind the subject is there
        off = np.minimum(off, np.maximum(np.linalg.norm((xyz - middle) @ R, axis=1) - float(np.max(axes)), 0.0))
    assert ((off > 0.4) & (radius < 4.5) & (xyz[:, 2] > 0.3) & (xyz[:, 2] < 3.0)).mean() < 0.03
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
