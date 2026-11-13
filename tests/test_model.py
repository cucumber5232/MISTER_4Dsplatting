import math

import numpy as np
import torch

from mister4d.model import DynamicScene, estimate_velocities, init_gaussian_params


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


def _sphere(rng, n, center):
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v * 0.2 + center, v


def test_velocities_follow_a_moving_surface_and_ignore_a_still_one():
    """Each frame is sampled independently: only the motion along the normal can be seen."""
    rng = np.random.default_rng(0)
    step = np.array([0.05, 0.0, 0.0])  # metres per frame
    clouds, normals = zip(*[_sphere(rng, 3000, f * step) for f in range(4)])
    frames = np.repeat(np.arange(4), 3000)
    v = estimate_velocities(np.concatenate(clouds), frames).reshape(4, 3000, 3)
    for f in range(4):
        front_or_back = np.abs(normals[f][:, 0]) > 0.8  # the surface moves along its normal here
        assert 0.03 < v[f][front_or_back, 0].mean() < 0.06
        assert np.abs(v[f][front_or_back, 1:]).mean() < 0.02  # only the normal component is seen

    still = np.concatenate([_sphere(rng, 3000, np.zeros(3))[0] for _ in range(4)])
    assert (np.linalg.norm(estimate_velocities(still, frames), axis=1) > 0).mean() < 0.02


def test_time_is_differentiable_so_a_camera_offset_can_be_learned():
    scene = _scene()
    with torch.no_grad():
        scene.params["velocities"][0] = torch.tensor([0.1, 0.0, 0.0])
    when = torch.tensor(2.4, requires_grad=True)
    state = scene.state_at(when)
    state["means"][0, 0].backward(retain_graph=True)
    assert abs(float(when.grad) - 0.1) < 1e-6  # moving the clock moves the Gaussian by its velocity
    when.grad = None
    state["opacities"].sum().backward()
    assert float(when.grad) < 0  # and, past its instant, makes it fade
