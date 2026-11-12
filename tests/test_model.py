import math

import numpy as np
import torch

from mister4d.model import DynamicScene, init_gaussian_params


def _scene(static=False):
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    params = init_gaussian_params(xyz, np.full((3, 3), 0.5), np.array([2, 2, 9]), sh_degree=0,
                                  init_opacity=0.5, init_duration=1.0)
    return DynamicScene(params, 0, static)


def test_gaussians_fade_in_time_and_move_with_their_velocity():
    scene = _scene()
    with torch.no_grad():
        scene.params["velocities"][0] = torch.tensor([0.1, 0.0, 0.0])
    at_birth = scene.state_at(2)
    assert at_birth["active"].tolist() == [0, 1]  # the third one lives around frame 9
    assert torch.allclose(at_birth["opacities"], torch.tensor([0.5, 0.5]))
    assert torch.allclose(at_birth["means"][0], torch.zeros(3))

    later = scene.state_at(3.5)
    fade = math.exp(-0.5 * 1.5**2)
    assert torch.allclose(later["opacities"], torch.tensor([0.5 * fade, 0.5 * fade]), atol=1e-6)
    assert torch.allclose(later["means"][0], torch.tensor([0.15, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(later["means"][1], torch.tensor([1.0, 0.0, 0.0]))
    assert scene.state_at(9)["active"].tolist() == [2]
    assert scene.state_at(30)["active"].numel() == 0


def test_gradients_reach_the_temporal_parameters():
    scene = _scene()
    state = scene.state_at(2.7)
    (state["opacities"].sum() + state["means"].sum()).backward()
    for name in ("times", "durations", "velocities", "opacities", "means"):
        grad = scene.params[name].grad
        assert grad is not None and grad[:2].abs().sum() > 0, name
    assert scene.params["times"].grad[2] == 0  # not alive at that instant


def test_static_scene_ignores_time():
    scene = _scene(static=True)
    state = scene.state_at(123.0)
    assert state["active"] is None and state["velocities"] is None
    assert state["means"].shape == (3, 3)
    assert torch.allclose(state["opacities"], torch.full((3,), 0.5))


def test_initial_size_comes_from_neighbours_of_the_same_frame():
    rng = np.random.default_rng(0)
    near = rng.normal(0, 0.01, (50, 3))
    far = rng.normal(0, 1.0, (50, 3))  # same place, another instant, much sparser
    params = init_gaussian_params(np.concatenate([near, far]), np.zeros((100, 3)),
                                  np.repeat([0, 5], 50), sh_degree=1)
    scales = torch.exp(params["scales"][:, 0])
    assert scales[:50].mean() < 0.1 * scales[50:].mean()
    assert params["shN"].shape == (100, 3, 3)
