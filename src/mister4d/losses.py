"""Image losses and regularizers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _gaussian_window(size: int, sigma: float, device, dtype) -> Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    g = g / g.sum()
    return g[:, None] * g[None, :]


def ssim(a: Tensor, b: Tensor, window_size: int = 11, sigma: float = 1.5) -> Tensor:
    """Mean structural similarity of image batches [B, H, W, C] in 0..1."""
    a = a.permute(0, 3, 1, 2)
    b = b.permute(0, 3, 1, 2)
    channels = a.shape[1]
    win = _gaussian_window(window_size, sigma, a.device, a.dtype)
    win = win.expand(channels, 1, window_size, window_size).contiguous()
    pad = window_size // 2

    def blur(x):
        return F.conv2d(x, win, padding=pad, groups=channels)

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a * mu_a
    var_b = blur(b * b) - mu_b * mu_b
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a * mu_a + mu_b * mu_b + c1) * (var_a + var_b + c2)
    return (num / den).mean()


def psnr(a: Tensor, b: Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    return 99.0 if mse <= 1e-12 else float(-10.0 * torch.log10(torch.tensor(mse)))


def scale_regularizer(scales: Tensor, max_scale: float, max_anisotropy: float) -> Tensor:
    """Penalize Gaussians that are too large or too needle-like."""
    largest = scales.max(dim=-1).values
    smallest = scales.min(dim=-1).values
    too_big = F.relu(largest - max_scale).mean() / max_scale
    too_thin = F.relu(largest / smallest.clamp(min=1e-8) - max_anisotropy).mean()
    return too_big + 0.01 * too_thin


def needle_penalty(scales: Tensor, max_ratio: float = 3.0) -> Tensor:
    """Penalize Gaussians much longer in one direction than in the other two.

    A disc lying on a surface is fine; a needle is what a Gaussian becomes when it hides
    along the line of sight of one camera, and it shows as a streak from anywhere else.
    """
    ordered = scales.sort(dim=-1, descending=True).values
    return F.relu(ordered[:, 0] / ordered[:, 1].clamp(min=1e-8) - max_ratio).mean()
