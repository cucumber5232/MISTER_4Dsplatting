import pytest
import torch

from mister4d.rasterizer import (ALPHA_MAX, ALPHA_MIN, TRANSMITTANCE_MIN, eval_sh,
                                 project_gaussians, render)


def _scene(n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(n, 3, generator=g) * 0.3 + torch.tensor([0.2, 0.0, 3.0])
    quats = torch.randn(n, 4, generator=g)
    scales = torch.rand(n, 3, generator=g) * 0.1 + 0.02
    opacities = torch.rand(n, generator=g) * 0.9 + 0.05
    colors = torch.rand(n, 3, generator=g)
    viewmats = torch.eye(4)[None].repeat(2, 1, 1)
    viewmats[1, 0, 3] = 0.2
    Ks = torch.tensor([[[60.0, 0, 18.5], [0, 60.0, 14.5], [0, 0, 1]]]).repeat(2, 1, 1)
    return means, quats, scales, opacities, colors, viewmats, Ks


def test_matches_per_pixel_definition():
    """The batched rasterizer equals a literal front-to-back loop over every pixel."""
    w, h = 37, 29  # deliberately not multiples of the tile size
    means, quats, scales, opacities, colors, viewmats, Ks = _scene()
    rgb, alpha, _, _ = render(means, quats, scales, opacities, colors, viewmats, Ks, w, h, backend="torch")
    m2, con, dep, rad, _ = project_gaussians(means, quats, scales, viewmats, Ks, w, h)
    for c in range(2):
        visible = torch.nonzero((rad[c] > 0).all(-1)).squeeze(-1)
        order = visible[torch.argsort(dep[c, visible])].tolist()
        for i in range(0, h, 3):
            for j in range(0, w, 3):
                trans, colour = 1.0, torch.zeros(3)
                for g in order:
                    dx, dy = j + 0.5 - m2[c, g, 0], i + 0.5 - m2[c, g, 1]
                    sigma = 0.5 * (con[c, g, 0] * dx * dx + con[c, g, 2] * dy * dy) + con[c, g, 1] * dx * dy
                    a = min(ALPHA_MAX, float(opacities[g] * torch.exp(-sigma)))
                    if sigma < 0 or a < ALPHA_MIN:
                        continue
                    if trans * (1 - a) <= TRANSMITTANCE_MIN:
                        break
                    colour += colors[g] * a * trans
                    trans *= 1 - a
                assert torch.allclose(rgb[c, i, j], colour, atol=1e-5)
                assert abs(float(alpha[c, i, j, 0]) - (1 - trans)) < 1e-5


def test_gradients_reach_gaussians_and_screen_positions():
    means, quats, scales, opacities, colors, viewmats, Ks = _scene()
    means.requires_grad_(True)
    rgb, alpha, _, info = render(means, quats, scales, opacities, colors, viewmats, Ks, 32, 24, backend="torch")
    info["means2d"].retain_grad()
    (rgb.sum() + alpha.sum()).backward()
    assert means.grad.abs().sum() > 0
    assert info["means2d"].grad.shape == (2, means.shape[0], 2)
    assert info["radii"].shape == (2, means.shape[0], 2)


def test_projection_matches_gsplat_reference():
    """Same numbers as gsplat's own PyTorch reference: both backends agree on the math."""
    ref = pytest.importorskip("gsplat.cuda._torch_impl")
    means, quats, scales, _, _, viewmats, Ks = _scene()
    m2, con, dep, rad, comp = project_gaussians(means, quats, scales, viewmats, Ks, 160, 120)
    covars, _ = ref._quat_scale_to_covar_preci(quats, scales, True, False)
    r2, m2_ref, dep_ref, con_ref, comp_ref = ref._fully_fused_projection(
        means, covars, viewmats, Ks, 160, 120, calc_compensations=True)
    assert torch.allclose(m2, m2_ref, atol=1e-3)
    assert torch.allclose(con, con_ref, atol=1e-4)
    assert torch.allclose(dep, dep_ref, atol=1e-5)
    assert torch.equal(rad, r2)
    assert torch.allclose(comp, comp_ref, atol=1e-5)


def test_spherical_harmonics_match_gsplat_reference():
    ref = pytest.importorskip("gsplat.cuda._torch_impl")
    g = torch.Generator().manual_seed(0)
    coeffs = torch.randn(50, 16, 3, generator=g)
    dirs = torch.randn(2, 50, 3, generator=g)
    unit = torch.nn.functional.normalize(dirs, dim=-1)
    for degree in range(4):
        expected = torch.clamp(ref._spherical_harmonics(degree, unit, coeffs[None].expand(2, -1, -1, -1)) + 0.5, min=0)
        assert torch.allclose(eval_sh(degree, coeffs, dirs), expected, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_gsplat_backend_matches_reference_on_every_gpu():
    from mister4d.doctor import gpu_selftest

    for i in range(torch.cuda.device_count()):
        assert gpu_selftest(torch.device(f"cuda:{i}")) < 0.02
