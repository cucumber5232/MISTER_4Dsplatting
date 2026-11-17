"""What keeps a scene solid and clean: no needles, no fog left behind, a surface per pixel."""

import math

import numpy as np
import pytest
import torch

from mister4d.dataset import MultiViewDataset
from mister4d.losses import needle_penalty
from mister4d.model import init_gaussian_params, load_checkpoint, render_scene
from mister4d.rasterizer import quat_to_rotmat
from mister4d.trainer import TrainConfig, train

FAST = dict(iterations=60, views_per_step=3, densify_start=20, densify_every=10,
            log_every=20, eval_every=20, holdout=["cam04"])


def test_needles_are_penalized_and_discs_are_not():
    ball = torch.tensor([[0.02, 0.02, 0.02]])
    disc = torch.tensor([[0.03, 0.03, 0.002]])
    needle = torch.tensor([[0.06, 0.004, 0.004]])
    assert float(needle_penalty(ball)) == 0.0 and float(needle_penalty(disc)) == 0.0
    assert float(needle_penalty(needle)) > 5.0
    assert float(needle_penalty(torch.cat([ball, disc, needle]))) == pytest.approx(float(needle_penalty(needle)) / 3)


def test_discs_start_on_the_surface_the_points_describe():
    rng = np.random.default_rng(0)
    xyz = np.zeros((400, 3), dtype=np.float32)
    xyz[:, :2] = rng.random((400, 2))  # points on the floor
    params = init_gaussian_params(xyz, np.full((400, 3), 0.5), np.zeros(400), 0, flatten=0.25)
    scales = params["scales"].detach()
    assert torch.allclose(scales[:, 2], scales[:, 0] + math.log(0.25), atol=1e-5)  # thin across...
    assert torch.allclose(scales[:, 0], scales[:, 1])  # ...round along
    normals = quat_to_rotmat(params["quats"].detach())[:, :, 2]  # the thin axis, in the world
    assert float(normals[:, 2].abs().min()) > 0.99  # is the vertical: the discs lie on the floor
    round_ones = init_gaussian_params(xyz, np.full((400, 3), 0.5), np.zeros(400), 0)
    assert torch.allclose(round_ones["scales"][:, 0], round_ones["scales"][:, 2])


def test_the_depth_of_what_a_pixel_is_made_of_can_be_rendered(tiny_dataset, tmp_path):
    run = tmp_path / "run"
    train(tiny_dataset, run, TrainConfig(devices="cpu", **{**FAST, "iterations": 30}))
    scene, _ = load_checkpoint(run / "ckpt.pt")
    ds = MultiViewDataset(tiny_dataset)
    viewmats, Ks = ds.camera_tensors(["cam01", "cam02"], "cpu")
    with torch.no_grad():
        rgb, alpha, depth, info, _ = render_scene(scene, 2, viewmats, Ks, ds.width, ds.height, moments=True)
        plain, alpha2, *_ = render_scene(scene, 2, viewmats, Ks, ds.width, ds.height)
    assert rgb.shape[-1] == 3 and torch.allclose(rgb, plain, atol=1e-6) and torch.allclose(alpha, alpha2)
    mom = info["moments"]
    assert mom.shape == (2, ds.height, ds.width, 2)
    drawn = alpha[..., 0] > 0.5
    assert drawn.any()
    # The first is the blended depth: the same the rasterizer returns, before dividing by alpha.
    assert torch.allclose((mom[..., 0] / alpha[..., 0])[drawn], depth[..., 0][drawn], rtol=1e-4, atol=1e-4)
    # The two together give how far apart in depth the Gaussians of a pixel are: never negative.
    apart = alpha[..., 0] * mom[..., 1] - mom[..., 0] ** 2
    assert float(apart.min()) > -1e-4 and float(apart[drawn].mean()) > 0.0


@pytest.mark.slow
def test_nothing_faint_and_no_needle_is_left_at_the_end(tiny_dataset, tmp_path):
    run = tmp_path / "run"
    cfg = TrainConfig(devices="cpu", opacity_decay=0.03, **{**FAST, "iterations": 120})
    assert cfg.prune_to_end and cfg.lambda_needle > 0  # both are on by default
    train(tiny_dataset, run, cfg)
    scene, _ = load_checkpoint(run / "ckpt.pt")
    assert float(torch.sigmoid(scene.params["opacities"].detach()).min()) >= cfg.prune_opacity
    ordered = torch.exp(scene.params["scales"].detach()).sort(dim=-1, descending=True).values
    needles = (ordered[:, 0] / ordered[:, 1] > 3.0).float().mean()
    assert float(needles) < 0.1
    assert all(scene.state_at(f)["active"].numel() > 100 for f in range(5))

    faint = tmp_path / "faint"  # the same without the cleaning: fog is left behind
    train(tiny_dataset, faint, TrainConfig(devices="cpu", opacity_decay=0.03, prune_to_end=False,
                                           **{**FAST, "iterations": 120}))
    left, _ = load_checkpoint(faint / "ckpt.pt")
    assert float(torch.sigmoid(left.params["opacities"].detach()).min()) < cfg.prune_opacity
    assert left.num_gaussians > scene.num_gaussians


@pytest.mark.slow
def test_the_optional_surface_terms_still_give_a_scene(tiny_dataset, tmp_path):
    run = tmp_path / "run"
    cfg = TrainConfig(devices="cpu", lambda_ray_spread=0.01, lambda_opacity_entropy=0.005, lambda_flat=0.005,
                      init_flatten=0.3, min_scale_pixels=0.5, surface_start=0.2, **{**FAST, "iterations": 80})
    train(tiny_dataset, run, cfg)
    log = [__import__("json").loads(line) for line in (run / "train_log.jsonl").read_text().splitlines()]
    assert log[-1]["psnr_holdout"] > log[0]["psnr_holdout"]
    scene, ckpt = load_checkpoint(run / "ckpt.pt")
    ds = MultiViewDataset(tiny_dataset)
    cams = ckpt["meta"]["train_cameras"]
    distance = np.mean([np.linalg.norm(ds.rig.cameras[c].center - np.array(ckpt["meta"]["center"])) for c in cams])
    pixel = distance / np.mean([ds.rig.cameras[c].K[0, 0] for c in cams])  # metres a pixel covers at the subject
    assert float(torch.exp(scene.params["scales"].detach()).min()) >= 0.5 * pixel * 0.999  # nothing thinner than half a pixel
    assert all(scene.state_at(f)["active"].numel() > 100 for f in range(5))
