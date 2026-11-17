"""Gaussian rasterization behind one small interface.

Two interchangeable backends:

- ``gsplat``: the CUDA rasterizer of the gsplat library (Apache-2.0). This is
  the one to use for real work.
- ``torch``: a slow, dense, pure-PyTorch reference written for this project.
  It follows the same math as gsplat so that the whole pipeline can be tested
  on a machine without a GPU. Only practical for small images.

Both take the same inputs and return the same ``info`` dictionary, which is
what the gsplat densification strategies expect.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.999
TRANSMITTANCE_MIN = 1e-4


def resolve_backend(backend: str, device: torch.device) -> str:
    if backend == "auto":
        return "gsplat" if device.type == "cuda" else "torch"
    if backend == "gsplat" and device.type != "cuda":
        raise RuntimeError("the gsplat backend needs a CUDA device; use backend='torch' on CPU")
    if backend not in ("gsplat", "torch"):
        raise ValueError(f"unknown backend '{backend}'")
    return backend


def opencv_to_render_K(K: Tensor) -> Tensor:
    """OpenCV intrinsics (pixel centre at integer coords) -> rasterizer intrinsics.

    Both backends sample pixel (i, j) at (j + 0.5, i + 0.5), so the principal
    point moves by half a pixel.
    """
    K = K.clone()
    K[..., 0, 2] += 0.5
    K[..., 1, 2] += 0.5
    return K


def render(
    means: Tensor,  # [N, 3]
    quats: Tensor,  # [N, 4] wxyz, not necessarily normalized
    scales: Tensor,  # [N, 3] already activated (metres)
    opacities: Tensor,  # [N] already activated (0..1)
    colors: Tensor,  # [N, D] or [C, N, D]: colours, or any quantity to blend like a colour
    viewmats: Tensor,  # [C, 4, 4] world -> camera
    Ks: Tensor,  # [C, 3, 3] rasterizer convention, see opencv_to_render_K
    width: int,
    height: int,
    backend: str = "auto",
    antialiased: bool = False,
    near: float = 0.01,
    far: float = 1e10,
) -> tuple[Tensor, Tensor, Tensor, dict]:
    """Render C views. Returns rgb [C,H,W,D], alpha [C,H,W,1], depth [C,H,W,1], info.

    ``rgb`` is premultiplied (no background added): composite with
    ``rgb + (1 - alpha) * background``.
    """
    backend = resolve_backend(backend, means.device)
    if backend == "gsplat":
        return _render_gsplat(means, quats, scales, opacities, colors, viewmats, Ks,
                              width, height, antialiased, near, far)
    return _render_torch(means, quats, scales, opacities, colors, viewmats, Ks,
                         width, height, antialiased, near, far)


def _render_gsplat(means, quats, scales, opacities, colors, viewmats, Ks,
                   width, height, antialiased, near, far):
    from gsplat import rasterization

    out, alpha, info = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        near_plane=near,
        far_plane=far,
        sh_degree=None,
        packed=False,
        render_mode="RGB+ED",
        rasterize_mode="antialiased" if antialiased else "classic",
    )
    return out[..., :-1], alpha, out[..., -1:], info


# --------------------------------------------------------------------------
# Pure PyTorch reference
# --------------------------------------------------------------------------


def quat_to_rotmat(quats: Tensor) -> Tensor:
    """Quaternions wxyz [..., 4] -> rotation matrices [..., 3, 3]."""
    w, x, y, z = torch.unbind(F.normalize(quats, dim=-1), dim=-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(quats.shape[:-1] + (3, 3))


def project_gaussians(means, quats, scales, viewmats, Ks, width, height,
                      eps2d: float = 0.3, near: float = 0.01, far: float = 1e10):
    """EWA projection of 3D Gaussians onto C image planes.

    Returns means2d [C,N,2], conics [C,N,3], depths [C,N], radii [C,N,2] (int,
    0 where culled) and the antialiasing compensation factor [C,N].
    """
    rot = quat_to_rotmat(quats)
    M = rot * scales[:, None, :]
    covars = M @ M.transpose(-1, -2)  # [N,3,3]

    R = viewmats[:, :3, :3]  # [C,3,3]
    t = viewmats[:, :3, 3]  # [C,3]
    means_c = torch.einsum("cij,nj->cni", R, means) + t[:, None, :]
    covars_c = torch.einsum("cij,njk,clk->cnil", R, covars, R)

    tx, ty, tz = torch.unbind(means_c, dim=-1)
    tz = torch.where(tz.abs() < 1e-8, torch.full_like(tz, 1e-8), tz)
    fx, fy = Ks[:, 0, 0, None], Ks[:, 1, 1, None]
    cx, cy = Ks[:, 0, 2, None], Ks[:, 1, 2, None]

    # Clamp the point used for the Jacobian to slightly outside the frustum.
    tan_x, tan_y = 0.5 * width / fx, 0.5 * height / fy
    jx = tz * torch.minimum(torch.maximum(tx / tz, -(cx / fx + 0.3 * tan_x)),
                            (width - cx) / fx + 0.3 * tan_x)
    jy = tz * torch.minimum(torch.maximum(ty / tz, -(cy / fy + 0.3 * tan_y)),
                            (height - cy) / fy + 0.3 * tan_y)
    zero = torch.zeros_like(tz)
    J = torch.stack([fx / tz, zero, -fx * jx / tz**2,
                     zero, fy / tz, -fy * jy / tz**2], dim=-1).reshape(tz.shape + (2, 3))
    cov2d = J @ covars_c @ J.transpose(-1, -2)

    a, b, c = cov2d[..., 0, 0], cov2d[..., 0, 1], cov2d[..., 1, 1]
    det_orig = a * c - b * b
    a, c = a + eps2d, c + eps2d
    det = (a * c - b * b).clamp(min=1e-10)
    compensation = torch.sqrt((det_orig / det).clamp(min=0.0))
    conics = torch.stack([c / det, -b / det, a / det], dim=-1)

    means2d = torch.stack([fx * tx / tz + cx, fy * ty / tz + cy], dim=-1)
    depths = means_c[..., 2]

    with torch.no_grad():
        radius = torch.stack([torch.ceil(3.33 * torch.sqrt(a)),
                              torch.ceil(3.33 * torch.sqrt(c))], dim=-1)
        valid = (depths > near) & (depths < far)
        inside = (
            (means2d[..., 0] + radius[..., 0] > 0)
            & (means2d[..., 0] - radius[..., 0] < width)
            & (means2d[..., 1] + radius[..., 1] > 0)
            & (means2d[..., 1] - radius[..., 1] < height)
        )
        radius[~(valid & inside)] = 0.0
        radii = radius.int()
    return means2d, conics, depths, radii, compensation


def _render_torch(means, quats, scales, opacities, colors, viewmats, Ks,
                  width, height, antialiased, near, far, tile: int = 8):
    """Dense reference rasterizer.

    Splats are sorted front to back once per camera; the image is processed one
    row of tiles at a time, with the splats of every tile padded to a common
    count so that the whole row is a single batch of tensor operations.
    """
    C = viewmats.shape[0]
    dev, dt = means.device, means.dtype
    means2d, conics, depths, radii, comp = project_gaussians(
        means, quats, scales, viewmats, Ks, width, height, near=near, far=far
    )
    if colors.dim() == 2:
        colors = colors[None].expand(C, -1, -1)

    rows, cols = math.ceil(height / tile), math.ceil(width / tile)
    offsets = torch.arange(tile, device=dev, dtype=dt) + 0.5
    off_y, off_x = (o.reshape(-1) for o in torch.meshgrid(offsets, offsets, indexing="ij"))
    tile_x0 = torch.arange(cols, device=dev, dtype=dt) * tile
    px = tile_x0[:, None] + off_x[None, :]  # [cols, P]

    def pad(x):  # one extra, fully transparent splat used as filler
        return torch.cat([x, x.new_zeros((1,) + x.shape[1:])])

    out_rgb, out_alpha, out_depth = [], [], []
    for c in range(C):
        rgb = torch.zeros(rows * tile, cols * tile, colors.shape[-1], dtype=dt, device=dev)
        alpha = torch.zeros(rows * tile, cols * tile, 1, dtype=dt, device=dev)
        depth = torch.zeros(rows * tile, cols * tile, 1, dtype=dt, device=dev)
        visible = torch.nonzero((radii[c] > 0).all(dim=-1)).squeeze(-1)
        if visible.numel() > 0:
            order = visible[torch.argsort(depths[c, visible])]  # front to back
            m2, rad = means2d[c, order], radii[c, order].to(dt)
            count = order.numel()
            opa = opacities[order] * comp[c, order] if antialiased else opacities[order]
            m2_p, con_p, col_p = pad(m2), pad(conics[c, order]), pad(colors[c, order])
            dep_p, opa_p = pad(depths[c, order]), pad(opa)
            in_col = ((m2[None, :, 0] + rad[None, :, 0] > tile_x0[:, None])
                      & (m2[None, :, 0] - rad[None, :, 0] < tile_x0[:, None] + tile))
            for ty in range(rows):
                y0 = ty * tile
                in_row = (m2[:, 1] + rad[:, 1] > y0) & (m2[:, 1] - rad[:, 1] < y0 + tile)
                hits = in_col & in_row[None, :]  # [cols, count]
                per_tile = hits.sum(dim=1)
                busy = torch.nonzero(per_tile > 0).squeeze(-1)  # tiles with something to draw
                if busy.numel() == 0:
                    continue
                hits, per_tile = hits[busy], per_tile[busy]
                most = int(per_tile.max())
                # nonzero() is row-major, so inside each tile the depth order is kept.
                t_idx, g_idx = torch.nonzero(hits, as_tuple=True)
                rank = torch.arange(t_idx.numel(), device=dev) - (per_tile.cumsum(0) - per_tile)[t_idx]
                index = torch.full((busy.numel(), most), count, dtype=torch.long, device=dev)
                index[t_idx, rank] = g_idx

                centre, con = m2_p[index], con_p[index]  # [cols, K, 2], [cols, K, 3]
                dx = px[busy][:, None, :] - centre[..., 0:1]  # [T, K, P]
                dy = (y0 + off_y)[None, None, :] - centre[..., 1:2]
                sigma = (0.5 * (con[..., 0:1] * dx * dx + con[..., 2:3] * dy * dy)
                         + con[..., 1:2] * dx * dy)
                a = torch.clamp(opa_p[index][..., None] * torch.exp(-sigma), max=ALPHA_MAX)
                a = torch.where((sigma >= 0) & (a >= ALPHA_MIN), a, torch.zeros_like(a))
                trans = torch.cumprod(1.0 - a, dim=1)  # transmittance after each splat
                before = torch.cat([torch.ones_like(trans[:, :1]), trans[:, :-1]], dim=1)
                w = a * before * (trans > TRANSMITTANCE_MIN)

                def place(values):  # [T, P, ch] -> [tile, cols * tile, ch]
                    ch = values.shape[-1]
                    full = torch.zeros(cols, tile * tile, ch, dtype=dt, device=dev)
                    full[busy] = values
                    return full.reshape(cols, tile, tile, ch).permute(1, 0, 2, 3).reshape(tile, cols * tile, ch)

                rgb[y0:y0 + tile] = place(torch.einsum("tkp,tkc->tpc", w, col_p[index]))
                alpha[y0:y0 + tile] = place(w.sum(dim=1)[..., None])
                depth[y0:y0 + tile] = place(torch.einsum("tkp,tk->tp", w, dep_p[index])[..., None])
        out_rgb.append(rgb[:height, :width])
        out_alpha.append(alpha[:height, :width])
        out_depth.append(depth[:height, :width])

    rgb, alpha, depth = torch.stack(out_rgb), torch.stack(out_alpha), torch.stack(out_depth)
    depth = depth / alpha.clamp(min=1e-10)
    info = {
        "means2d": means2d,
        "radii": radii,
        "depths": depths,
        "width": width,
        "height": height,
        "n_cameras": C,
        "gaussian_ids": None,
    }
    return rgb, alpha, depth, info


# --------------------------------------------------------------------------
# Spherical harmonics (real basis, the ordering used by 3DGS .ply files)
# --------------------------------------------------------------------------

SH_C0 = 0.5 * math.sqrt(1.0 / math.pi)
_C1 = math.sqrt(3.0 / (4.0 * math.pi))
_C2 = [
    0.5 * math.sqrt(15.0 / math.pi),
    0.25 * math.sqrt(5.0 / math.pi),
    0.25 * math.sqrt(15.0 / math.pi),
]
_C3 = [
    0.25 * math.sqrt(35.0 / (2.0 * math.pi)),
    0.5 * math.sqrt(105.0 / math.pi),
    0.25 * math.sqrt(21.0 / (2.0 * math.pi)),
    0.25 * math.sqrt(7.0 / math.pi),
    0.25 * math.sqrt(105.0 / math.pi),
]


def num_sh_bases(degree: int) -> int:
    return (degree + 1) ** 2


def rgb_to_sh0(rgb: Tensor) -> Tensor:
    return (rgb - 0.5) / SH_C0


def eval_sh(degree: int, coeffs: Tensor, dirs: Tensor) -> Tensor:
    """Evaluate SH colours.

    coeffs: [N, K, 3]; dirs: [..., N, 3] (need not be normalized).
    Returns colours [..., N, 3], clamped to be non-negative.
    """
    out = SH_C0 * coeffs[:, 0]
    if degree > 0:
        d = F.normalize(dirs, dim=-1)
        x, y, z = d[..., 0:1], d[..., 1:2], d[..., 2:3]
        out = out + _C1 * (-y * coeffs[:, 1] + z * coeffs[:, 2] - x * coeffs[:, 3])
        if degree > 1:
            xx, yy, zz = x * x, y * y, z * z
            out = out + (
                _C2[0] * x * y * coeffs[:, 4]
                - _C2[0] * y * z * coeffs[:, 5]
                + _C2[1] * (2 * zz - xx - yy) * coeffs[:, 6]
                - _C2[0] * x * z * coeffs[:, 7]
                + _C2[2] * (xx - yy) * coeffs[:, 8]
            )
            if degree > 2:
                out = out + (
                    -_C3[0] * y * (3 * xx - yy) * coeffs[:, 9]
                    + _C3[1] * x * y * z * coeffs[:, 10]
                    - _C3[2] * y * (4 * zz - xx - yy) * coeffs[:, 11]
                    + _C3[3] * z * (2 * zz - 3 * xx - 3 * yy) * coeffs[:, 12]
                    - _C3[2] * x * (4 * zz - xx - yy) * coeffs[:, 13]
                    + _C3[4] * z * (xx - yy) * coeffs[:, 14]
                    - _C3[0] * x * (xx - 3 * yy) * coeffs[:, 15]
                )
    return torch.clamp(out + 0.5, min=0.0)
