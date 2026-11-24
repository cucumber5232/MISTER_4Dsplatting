"""Environment check: tools, GPUs, and a numerical self-test of the GPU rasterizer."""

from __future__ import annotations

import shutil
import subprocess

import torch


def _line(ok: bool | None, text: str) -> None:
    mark = {True: "[ ok ]", False: "[FAIL]", None: "[warn]"}[ok]
    print(f"{mark} {text}")


def gpu_selftest(device: torch.device) -> float:
    """Render one random scene with gsplat on ``device`` and with the CPU reference.

    Returns the largest absolute difference between the two images (0..1 scale).
    """
    from .rasterizer import render

    gen = torch.Generator().manual_seed(0)
    n, w, h = 2000, 160, 120
    means = torch.randn(n, 3, generator=gen) * 0.4 + torch.tensor([0.0, 0.0, 3.0])
    quats = torch.randn(n, 4, generator=gen)
    scales = torch.rand(n, 3, generator=gen) * 0.05 + 0.01
    opacities = torch.rand(n, generator=gen) * 0.9 + 0.05
    colors = torch.rand(n, 3, generator=gen)
    viewmats = torch.eye(4)[None].repeat(2, 1, 1)
    viewmats[1, 0, 3] = 0.3
    Ks = torch.tensor([[[150.0, 0, 80], [0, 150.0, 60], [0, 0, 1]]]).repeat(2, 1, 1)
    args = (means, quats, scales, opacities, colors, viewmats, Ks)
    ref_rgb, ref_alpha, *_ = render(*args, w, h, backend="torch")
    gpu_rgb, gpu_alpha, *_ = render(*(a.to(device) for a in args), w, h, backend="gsplat")
    return float(max((gpu_rgb.cpu() - ref_rgb).abs().max(), (gpu_alpha.cpu() - ref_alpha).abs().max()))


def run() -> int:
    import cv2
    import gsplat

    failures = 0
    print(f"mister4d doctor  (torch {torch.__version__}, gsplat {gsplat.__version__}, "
          f"opencv {cv2.__version__})")
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        _line(path is not None, f"{tool}: {path or 'not found in PATH (install ffmpeg)'}")
        failures += path is None
    _line(hasattr(cv2, "aruco") and hasattr(cv2.aruco, "CharucoDetector"),
          "OpenCV ChArUco detector available")

    if not torch.cuda.is_available():
        _line(None, "no CUDA device visible to PyTorch: training would run on the slow CPU "
                    "reference rasterizer (tests only). Install a CUDA build of PyTorch.")
        return failures
    count = torch.cuda.device_count()
    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        _line(True, f"cuda:{i}: {props.name}, {props.total_memory / 2**30:.0f} GB")
    nvcc = shutil.which("nvcc")
    if nvcc:
        version = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip()
        _line(True, f"nvcc: {version.splitlines()[-1] if version else nvcc}")
    else:
        _line(None, "nvcc not found: fine with a prebuilt gsplat wheel, otherwise gsplat cannot "
                    "compile its CUDA kernels on first use")
    try:  # gsplat hides why its CUDA code does not load: ask for it directly
        from gsplat import csrc  # noqa: F401

        _line(True, "gsplat: its CUDA code is loaded")
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic tool
        _line(False, f"gsplat: its CUDA code does not load: {type(exc).__name__}: {exc}")
        failures += 1
    for i in range(count):
        try:
            diff = gpu_selftest(torch.device(f"cuda:{i}"))
            ok = diff < 0.02
            _line(ok, f"cuda:{i}: gsplat render matches the CPU reference (max diff {diff:.4f})")
            failures += not ok
        except Exception as exc:  # noqa: BLE001 - this is a diagnostic tool
            _line(False, f"cuda:{i}: gsplat render failed: {type(exc).__name__}: {exc}")
            failures += 1
    if count > 1 and failures == 0:
        _line(True, f"{count} GPUs usable: 'mister4d train' uses all of them by default")
    return failures
