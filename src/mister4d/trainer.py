"""Training loop for the dynamic Gaussian scene (single or multi GPU).

Multi-GPU scheme: the parameters live on the first device. At every step each
device receives differentiable copies of them, renders a different batch (one
instant seen by several cameras) and returns its loss; a single backward pass
sends the gradients of all devices back to the owner. With two GPUs a step
therefore sees twice as many images in about the same wall-clock time.
Frames are decoded by background threads so that the GPUs never wait on disk.

Long takes are trained in segments of a couple of seconds (see ``run.py``):
each segment is a training of its own, so memory and time per training do not
grow with the length of the take, and an interrupted job continues from the
first segment that is missing.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import yaml

from .camera import describe_coverage, silhouette_coverage
from .dataset import MultiViewDataset, read_meta, read_points_ply
from .losses import needle_penalty, psnr, scale_regularizer, ssim
from .model import (DynamicScene, estimate_velocities, init_gaussian_params, load_checkpoint, render_scene,
                    save_checkpoint, scenery_of)
from .rasterizer import resolve_backend
from .run import MANIFEST, SEGMENT_FRAMES, plan_segments, segment_dir, write_manifest


@dataclass
class TrainConfig:
    # --- data ---------------------------------------------------------------
    holdout: list[str] = field(default_factory=list)  # cameras kept out of training
    views_per_step: int = 4  # cameras rendered per step and per device (same instant)
    cache_gb: float = 16.0  # RAM budget for decoded frames
    loader_threads: int = 8  # threads decoding frames ahead of the GPUs

    # --- schedule -----------------------------------------------------------
    iterations: int = -1  # -1 = 250 per frame of the take (of each segment, for a long take), at least 7000
    segment_frames: int = SEGMENT_FRAMES  # a longer take is trained in segments of about this many frames (0 = never)
    segment_overlap: int = 2  # frames of context a segment sees beyond each of its ends

    # --- the whole scene ----------------------------------------------------
    full_scene: bool = False  # True: reconstruct the set too (no masks): first the empty set, then the take on it
    scenery_iterations: int = -1  # steps for the empty set; -1 = 7000
    scenery_max_gaussians: int = 1_500_000
    scenery_max_scale: float = 1.0  # metres: floors and walls are made of large Gaussians
    scenery_opacity_decay: float = 0.0  # the decay below, for the empty set: there it leaves it see-through
    scenery_prune_scale: float = 3.0

    # --- model --------------------------------------------------------------
    static: bool = False  # True: plain 3DGS of one instant
    static_frame: int = -1  # index of that instant in the take; -1 = middle
    sh_degree: int = 0  # 0 = view-independent colour (safest with few cameras)
    sh_interval: int = 1000  # steps between SH degree increases
    init_duration: float = 1.0  # initial life of a Gaussian, in units of the init frame spacing
    init_velocities: bool = True  # start from the motion estimated between the init point clouds
    init_flatten: float = 1.0  # < 1: start from discs lying on the visual hull, this much thinner across it
    learn_time_offsets: bool = False  # cameras not exposed together: learn their sub-frame offsets
    init_opacity: float = 0.1
    init_random_points: int = 20000  # used only when there is no init_points.ply

    # --- losses -------------------------------------------------------------
    lambda_dssim: float = 0.2
    lambda_mask: float = 0.5  # rendered alpha vs foreground mask
    lambda_scale: float = 0.1
    lambda_velocity: float = 0.05  # keeps motion that no camera can see from drifting
    max_scale: float = 0.05  # metres; larger Gaussians are penalized
    max_anisotropy: float = 10.0
    lambda_ray_spread: float = 0.0  # a pixel should be made of one surface, not of layers spread in depth
    lambda_opacity_entropy: float = 0.0  # a Gaussian should be either there or not: no fog
    lambda_flat: float = 0.0  # Gaussians should be thin across the surface they lie on
    lambda_needle: float = 0.02  # no Gaussian much longer in one direction than in the other two
    min_scale_pixels: float = 0.5  # no Gaussian thinner than this many pixels of the training images (0 = off)
    surface_start: float = 0.3  # the three terms above act from this share of the iterations
    random_background: bool = True

    # --- optimisation -------------------------------------------------------
    lr_means: float = 1.6e-4  # multiplied by the rig radius
    lr_velocities: float = 1.6e-4  # metres per frame, multiplied by the rig radius
    lr_final_factor: float = 0.01
    lr_scales: float = 5e-3
    lr_quats: float = 1e-3
    lr_opacities: float = 5e-2
    lr_sh0: float = 2.5e-3
    lr_shN: float = 1.25e-4
    lr_times: float = 2e-3  # frames
    lr_durations: float = 5e-3
    lr_time_offsets: float = 5e-3  # frames
    learn_camera_colors: bool = False  # cameras that do not see the same colours: learn the gains of each
    lr_camera_colors: float = 5e-3

    # --- densification ------------------------------------------------------
    densify: bool = True
    densify_start: int = 500
    densify_stop: int = -1  # -1 = 60% of the iterations
    densify_every: int = 100
    grow_grad2d: float = 2e-4
    prune_opacity: float = 0.005
    prune_scale: float = 0.15  # metres; larger Gaussians are removed
    opacity_reset_every: int = 0  # 0 = never (resets hurt with few cameras)
    opacity_decay: float = 0.005  # share of its opacity a Gaussian loses each time it is drawn (0 = off)
    opacity_decay_stop: float = 1.0  # the decay acts until this share of the iterations
    # The share above pulls a solid Gaussian down much harder than a faint one: at opacity 0.9 it
    # is as much as the optimiser can give back in a step, so nothing stays solid. With this the
    # same amount is taken in logit space: faint Gaussians fade as before, solid ones hardly at all.
    opacity_decay_logit: bool = False
    prune_to_end: bool = True  # keep removing faint Gaussians after densification stops, and at the end
    max_gaussians: int = 2_000_000

    # --- runtime ------------------------------------------------------------
    devices: str = "auto"  # auto = every visible GPU; or "cuda:0", "cuda:0,cuda:1", "cpu"
    backend: str = "auto"  # auto | gsplat | torch
    antialiased: bool = False
    seed: int = 0
    log_every: int = 100
    eval_every: int = 1000
    save_every: int = 0  # 0 = only at the end


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> TrainConfig:
    """Defaults <- YAML file <- ``key=value`` overrides."""
    values: dict = {}
    if path:
        values.update(yaml.safe_load(Path(path).read_text()) or {})
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like key=value")
        key, raw = item.split("=", 1)
        values[key.strip()] = yaml.safe_load(raw)
    known = {f.name for f in dataclasses.fields(TrainConfig)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    if isinstance(values.get("holdout"), str):
        values["holdout"] = [c for c in values["holdout"].split(",") if c]
    return TrainConfig(**values)


def pick_devices(spec: str = "auto") -> list[torch.device]:
    if spec == "auto":
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        return [torch.device(f"cuda:{i}") for i in range(count)] or [torch.device("cpu")]
    return [torch.device(d.strip()) for d in spec.split(",") if d.strip()]


def pick_device(spec: str = "auto") -> torch.device:
    return pick_devices(spec)[0]


def warmup_backends(devices: list[torch.device], backend: str) -> list[str]:
    """Resolve the backend of every device and load gsplat's CUDA code once.

    gsplat builds / loads its CUDA extension on first use; doing it here, in
    the main thread, gives a readable error and avoids racing worker threads.
    """
    backends = [resolve_backend(backend, d) for d in devices]
    for dev, name in zip(devices, backends):
        if name != "gsplat":
            continue
        try:
            from .rasterizer import render

            eye = torch.eye(4, device=dev)[None]
            K = torch.tensor([[[20.0, 0, 8], [0, 20.0, 8], [0, 0, 1]]], device=dev)
            render(torch.tensor([[0.0, 0, 2]], device=dev), torch.tensor([[1.0, 0, 0, 0]], device=dev),
                   torch.full((1, 3), 0.1, device=dev), torch.tensor([0.5], device=dev),
                   torch.full((1, 3), 0.5, device=dev), eye, K, 16, 16, backend="gsplat")
        except Exception as exc:  # noqa: BLE001 - report anything that went wrong
            raise RuntimeError(
                f"the gsplat CUDA backend failed on {dev}: {exc}\nRun 'mister4d doctor' for a "
                "diagnosis (gsplat compiles its CUDA kernels on first use and needs the CUDA "
                "toolkit and a C++ compiler, or a prebuilt wheel)") from exc
    return backends


class BatchLoader:
    """Samples the batches of every step and decodes their frames in background."""

    def __init__(self, ds: MultiViewDataset, train_cams: list[str], cfg: TrainConfig,
                 iterations: int, jobs_per_step: int, static_frame: int | None,
                 window: tuple[int, int] | None = None):
        self.ds, self.cams, self.cfg, self.iterations = ds, train_cams, cfg, iterations
        self.jobs_per_step, self.static_frame = jobs_per_step, static_frame
        self.window = window or (0, ds.num_frames)
        self._rng = random.Random(cfg.seed)
        self._queue: queue.Queue = queue.Queue(maxsize=6)
        self._stop = threading.Event()
        self._pool = ThreadPoolExecutor(max(1, cfg.loader_threads))
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _sample(self) -> list[tuple[int, list[str]]]:
        per_job = min(self.cfg.views_per_step, len(self.cams))
        if self.static_frame is not None:
            # One instant only: deal the cameras out so that devices get different views.
            order = self._rng.sample(self.cams, len(self.cams))
            jobs = []
            for j in range(self.jobs_per_step):
                picked = [order[(j * per_job + k) % len(order)] for k in range(per_job)]
                jobs.append((self.static_frame, list(dict.fromkeys(picked))))
            return jobs
        return [(self._rng.randrange(*self.window), self._rng.sample(self.cams, per_job))
                for _ in range(self.jobs_per_step)]

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def _run(self) -> None:
        try:
            for _ in range(self.iterations):
                batch = []
                for frame_index, cams in self._sample():
                    arrays = list(self._pool.map(lambda c, f=frame_index: self.ds.load(f, c), cams))
                    batch.append((frame_index, cams, arrays))
                if not self._put(batch):
                    return
        except BaseException as exc:  # noqa: BLE001 - forwarded to the training thread
            self._put(exc)

    def next(self) -> list[tuple[int, list[str], list]]:
        item = self._queue.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._pool.shutdown(wait=False)


def _to_device(arrays: list, device) -> tuple[torch.Tensor, torch.Tensor | None]:
    images = torch.from_numpy(np.stack([a[0] for a in arrays])).to(device).float() / 255.0
    if arrays[0][1] is None:
        return images, None
    masks = torch.from_numpy(np.stack([a[1] for a in arrays])).to(device).float()[..., None] / 255.0
    return images, masks


def _merge_info(results: list[tuple[dict, torch.Tensor | None]], total: int, device) -> dict:
    """One densification record out of the rasterizer outputs of all devices.

    Every device rendered only the Gaussians alive at its instant; their
    screen-space gradients and radii are scattered back to the full set.
    """
    grads, radii = [], []
    for info, active in results:
        g = info["means2d"].grad
        g = (torch.zeros_like(info["means2d"]) if g is None else g).to(device)
        r = info["radii"].to(device)
        if "count" in info:  # the scenery was drawn after them: it is not densified
            g, r = g[:, :info["count"]], r[:, :info["count"]]
        if active is not None:
            idx = active.to(device)
            full_g = torch.zeros(g.shape[0], total, 2, dtype=g.dtype, device=device)
            full_r = torch.zeros(r.shape[0], total, 2, dtype=r.dtype, device=device)
            full_g[:, idx], full_r[:, idx] = g, r
            g, r = full_g, full_r
        grads.append(g)
        radii.append(r)
    grads = torch.cat(grads)  # [sum C, N, 2]
    holder = torch.zeros_like(grads, requires_grad=True)
    holder.grad = grads
    return {"means2d": holder, "radii": torch.cat(radii),
            "width": results[0][0]["width"], "height": results[0][0]["height"],
            "n_cameras": grads.shape[0], "gaussian_ids": None}


def _initial_points(ds: MultiViewDataset, cfg: TrainConfig, static_frame: int | None, window: tuple[int, int]):
    """Positions, colours and frame index of the starting points of ``window``, plus their frame spacing."""
    ply = ds.root / "init_points.ply"
    if ply.exists():
        xyz, rgb, frames = read_points_ply(ply)
        step = int(ds.meta.get("init_frame_step", 1))
        if static_frame is not None:  # keep the points of the closest initialized frame
            nearest = frames[np.argmin(np.abs(frames - static_frame))]
            keep = frames == nearest
            xyz, rgb, frames = xyz[keep], rgb[keep], np.full(int(keep.sum()), static_frame)
        else:
            keep = (frames >= window[0]) & (frames < window[1])
            xyz, rgb, frames = xyz[keep], rgb[keep], frames[keep]
        if len(xyz):
            return xyz, rgb, frames, step
        print(f"[train] WARNING: no starting points for frames {window[0]}-{window[1] - 1}; "
              "starting from random points in the capture volume")
    else:
        print("[train] WARNING: no init_points.ply (run 'mister4d init'); "
              "starting from random points in the capture volume")
    vol = ds.rig.volume or {"center": [0, 0, 0], "size": [2, 2, 2]}
    rng = np.random.default_rng(cfg.seed)
    center, size = np.array(vol["center"], float), np.array(vol["size"], float)
    n = cfg.init_random_points
    xyz = (center + (rng.random((n, 3)) - 0.5) * size).astype(np.float32)
    frames = np.full(n, static_frame) if static_frame is not None else rng.integers(window[0], window[1], n)
    return xyz, np.full_like(xyz, 0.5), frames, 1


def apply_camera_colors(rgb: torch.Tensor, gains: list | None) -> torch.Tensor:
    """A rendered image as the camera with these learned colour ``gains`` (red, green, blue) sees it."""
    if not gains:
        return rgb
    return rgb * torch.tensor(gains, dtype=rgb.dtype, device=rgb.device)


@torch.no_grad()
def evaluate_views(scene, ds, frame_indices, cams, device, backend, antialiased=False,
                   time_offsets: dict | None = None, camera_colors: dict | None = None,
                   masked: bool = True) -> dict:
    """PSNR against the ground truth: the subject on white, or the whole image when not ``masked``."""
    values = []
    for cam in cams:
        viewmats, Ks = ds.camera_tensors([cam], device)
        shift = (time_offsets or {}).get(cam, 0.0)
        for fi in frame_indices:
            gt, mask = ds.batch(fi, [cam], device)
            rgb, alpha, *_ = render_scene(scene, fi + shift, viewmats, Ks, ds.width, ds.height,
                                          backend=backend, antialiased=antialiased)
            rgb = apply_camera_colors(rgb, (camera_colors or {}).get(cam))
            if mask is not None and masked:
                gt = gt * mask + (1.0 - mask)
                rgb = rgb + (1.0 - alpha)
            values.append(psnr(rgb.clamp(0, 1), gt))
    return {"psnr": float(np.mean(values))}


def _train_window(data_dir: str | Path, out_dir: str | Path, cfg: TrainConfig,
                  window: tuple[int, int] | None = None, core: tuple[int, int] | None = None,
                  fixed_offsets: dict | None = None, start: dict | None = None,
                  fixed_colors: dict | None = None, scenery: tuple[dict, int] | None = None,
                  scenery_file: str | None = None) -> tuple[Path, dict]:
    """One training: the whole take, or the frames ``window`` of a long one.

    ``core`` are the frames this scene will be used for (the window without
    the context at its ends). ``fixed_offsets`` and ``fixed_colors`` are the time
    offsets and the colours of the cameras learned on an earlier segment.
    ``start`` are the parameters of an existing scene to refine instead of
    starting from the visual hull (no densification then). ``scenery`` are the
    frozen Gaussians of the empty set and their SH degree, drawn under every
    instant; ``scenery_file`` is where their checkpoint is, relative to
    ``out_dir``. Returns the checkpoint and its summary.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    devices = pick_devices(cfg.devices)
    master = devices[0]
    backends = warmup_backends(devices, cfg.backend)
    ds = MultiViewDataset(data_dir, cache_gb=cfg.cache_gb, masks=not cfg.full_scene and scenery is None)

    unknown = [c for c in cfg.holdout if c not in ds.cameras]
    if unknown:
        raise ValueError(f"holdout cameras not in the dataset: {unknown} (have {ds.cameras})")
    train_cams = [c for c in ds.cameras if c not in cfg.holdout]
    val_cams = [c for c in ds.cameras if c in cfg.holdout]
    if len(train_cams) < 2:
        raise ValueError("at least 2 training cameras are required")
    if len(train_cams) < 3:
        print("[train] WARNING: fewer than 3 training cameras: expect a very rough result")

    center, radius = ds.scene_bounds()
    cam_dist = float(np.mean([np.linalg.norm(ds.rig.cameras[c].center - center) for c in train_cams]))
    static = cfg.static or ds.num_frames == 1
    static_frame = None
    if static:
        static_frame = min(ds.num_frames // 2 if cfg.static_frame < 0 else cfg.static_frame, ds.num_frames - 1)
    window = window or (0, ds.num_frames)
    core = core or window
    iterations = cfg.iterations
    if iterations <= 0:
        iterations = 7000 if static else max(7000, 250 * (window[1] - window[0]))

    if start is not None:
        params = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in start.items()})
    else:
        xyz, rgb, frames, frame_step = _initial_points(ds, cfg, static_frame, window)
        velocities = estimate_velocities(xyz, frames) if cfg.init_velocities and not static else None
        # On a scenery a faint subject lets the set show through it: there it starts solid.
        opacity = max(cfg.init_opacity, 0.5) if scenery is not None else cfg.init_opacity
        params = init_gaussian_params(xyz, rgb, frames, cfg.sh_degree, opacity,
                                      cfg.init_duration * frame_step, velocities, cfg.init_flatten)
    scene = DynamicScene(params, cfg.sh_degree, static, *(scenery or (None, 0))).to(master)
    # With the set in the scene the images are compared whole: no mask enters the training.
    masked = ds.has_masks and not cfg.full_scene and scenery is None

    # What the cameras cannot resolve must not be modelled: a Gaussian thinner than a pixel of the
    # training images costs nothing there and shows as a hair or a speck when the view is enlarged.
    focal = float(np.mean([ds.rig.cameras[c].K[0, 0] for c in train_cams]))
    scale_floor = math.log(cfg.min_scale_pixels * cam_dist / focal) if cfg.min_scale_pixels > 0 else None

    refine = 0.1 if start is not None else 1.0  # an existing scene only needs small moves
    lrs = {"means": cfg.lr_means * cam_dist * refine, "scales": cfg.lr_scales, "quats": cfg.lr_quats,
           "opacities": cfg.lr_opacities, "sh0": cfg.lr_sh0, "shN": cfg.lr_shN,
           "times": cfg.lr_times * refine, "durations": cfg.lr_durations,
           "velocities": cfg.lr_velocities * cam_dist * refine}
    optimizers = {
        name: torch.optim.Adam([{"params": scene.params[name], "lr": lr, "name": name}], eps=1e-15)
        for name, lr in lrs.items()
    }
    gamma = cfg.lr_final_factor ** (1.0 / max(iterations, 1))
    schedulers = [torch.optim.lr_scheduler.ExponentialLR(optimizers[k], gamma=gamma)
                  for k in ("means", "velocities")]

    # Sub-frame synchronisation: one time offset per training camera, relative
    # to the first one. The scene is continuous in time, so a camera that was
    # exposed a fraction of a frame late is simply rendered that much later.
    offsets = offset_opt = None
    if fixed_offsets and not static:  # learned on the first segment: a camera keeps its clock
        offsets = torch.tensor([float(fixed_offsets.get(c, 0.0)) for c in train_cams], device=master)
        offset_free = torch.ones(len(train_cams), device=master)
    elif cfg.learn_time_offsets and not static:
        offsets = torch.nn.Parameter(torch.zeros(len(train_cams), device=master))
        offset_opt = torch.optim.Adam([offsets], lr=cfg.lr_time_offsets)
        offset_free = torch.ones(len(train_cams), device=master)
        offset_free[0] = 0.0  # the reference camera defines the clock

    # Colours: cameras differ in exposure and white balance, and a scene fitted to images that
    # disagree changes tint with the point of view. Every camera gets a gain per channel; the
    # scene keeps the average look of the cameras (the gains are kept centred on "none").
    color_gain = color_opt = None  # logarithm of the gains, cameras x channels
    if fixed_colors:
        color_gain = torch.tensor([fixed_colors.get(c, [1.0] * 3) for c in train_cams], device=master).log()
    elif cfg.learn_camera_colors:
        color_gain = torch.nn.Parameter(torch.zeros(len(train_cams), 3, device=master))
        color_opt = torch.optim.Adam([color_gain], lr=cfg.lr_camera_colors)

    def learned_colors() -> dict[str, list[float]]:
        if color_gain is None:
            return {}
        gain = color_gain.detach().exp().cpu()
        return {c: [round(float(v), 5) for v in gain[i]] for i, c in enumerate(train_cams)}

    def learned_offsets() -> dict[str, float]:
        if offsets is None:
            return {}
        return {c: float(v) for c, v in zip(train_cams, (offsets.detach() * offset_free).cpu())}

    # Densification: gsplat's default strategy (clone / split / prune). Two
    # instances share one state so that growth can be switched off at the cap.
    strategy_state = None
    if cfg.densify and start is None:
        from gsplat.strategy import DefaultStrategy
        from gsplat.strategy.ops import remove, reset_opa

        stop = int(iterations * 0.6) if cfg.densify_stop < 0 else cfg.densify_stop
        common = dict(prune_opa=cfg.prune_opacity, refine_start_iter=cfg.densify_start,
                      refine_stop_iter=stop, refine_every=cfg.densify_every,
                      reset_every=10**9)  # opacity resets are handled below
        grow = DefaultStrategy(grow_grad2d=cfg.grow_grad2d, **common)
        hold = DefaultStrategy(grow_grad2d=1e10, **common)
        grow.check_sanity(scene.params, optimizers)
        strategy_state = grow.initialize_state(scene_scale=cam_dist)

    meta = {
        "data_dir": str(Path(data_dir).resolve()),
        "fps": ds.fps,
        "frame_ids": ds.frame_ids,
        "train_cameras": train_cams,
        "holdout_cameras": val_cams,
        "center": [float(v) for v in center],
        "radius": radius,
        "backend": backends[0],
        "static_frame": static_frame,
        "segment": list(core),  # the frames this scene is used for
        "window": list(window),  # the frames it was trained on
        # Everything the render / export commands need, so that a checkpoint
        # stays usable without the dataset next to it.
        "cameras": {name: cam.to_dict() for name, cam in ds.rig.cameras.items()},
        "volume": ds.rig.volume,
        "scenery_file": scenery_file,
    }
    for cam in meta["cameras"].values():
        cam.pop("info", None)
    cfg_dict = dataclasses.asdict(cfg)
    cfg_dict["iterations"] = iterations
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg_dict, sort_keys=False))

    print(f"[train] devices={[str(d) for d in devices]} backend={backends[0]} "
          f"cameras={train_cams} holdout={val_cams}")
    print(f"[train] frames={window[1] - window[0]} size={ds.width}x{ds.height} "
          f"masks={'yes' if masked else 'NO'} gaussians={scene.num_gaussians} "
          + (f"scenery={scene.num_scenery} " if scene.num_scenery else "") +
          f"model={'static, frame ' + str(static_frame) if static else 'dynamic'} "
          f"iterations={iterations} images/step={len(devices) * min(cfg.views_per_step, len(train_cams))}")
    coverage = silhouette_coverage([ds.rig.cameras[c] for c in train_cams], center)
    meta["coverage"] = coverage
    print(f"[train] coverage of the training cameras: {describe_coverage(coverage)}")
    if master.type == "cpu":
        print("[train] WARNING: no GPU in use: the CPU reference rasterizer is only meant for "
              "tests on tiny images")
    if not masked and not cfg.full_scene and scenery is None and ds.meta.get("source") != "empty set":
        print("[train] WARNING: training without masks: the background of every camera "
              "will be fitted too and will produce floaters")

    cam_index = {c: i for i, c in enumerate(train_cams)}
    cam_tensors = {str(d): ds.camera_tensors(train_cams, d) for d in set(devices)}
    eval_frames = [static_frame] if static else sorted({core[0], (core[0] + core[1]) // 2, core[1] - 1})
    eval_cams = val_cams or train_cams
    loader = BatchLoader(ds, train_cams, cfg, iterations, len(devices), static_frame, window)
    workers = ThreadPoolExecutor(len(devices)) if len(devices) > 1 else None
    log_file = open(out_dir / "train_log.jsonl", "w")
    t_start = time.time()
    ema_loss = None

    def run_job(slot: int, job, active_sh: int, shaping: bool):
        """Forward pass of one batch on one device."""
        frame_index, cams, arrays = job
        dev = devices[slot]
        gt, mask = _to_device(arrays, dev)
        if not masked:
            mask = None
        viewmats, Ks = cam_tensors[str(dev)]
        sel = [cam_index[c] for c in cams]
        view = scene.view(dev)
        spread = shaping and cfg.lambda_ray_spread > 0
        options = dict(backend=backends[slot], antialiased=cfg.antialiased, active_sh_degree=active_sh,
                       moments=spread)
        if offsets is None:
            renders = [render_scene(view, frame_index, viewmats[sel], Ks[sel], ds.width, ds.height, **options)]
        else:  # every camera at its own instant
            shift = (offsets * offset_free).to(dev)
            renders = [render_scene(view, frame_index + shift[i], viewmats[[i]], Ks[[i]],
                                    ds.width, ds.height, **options) for i in sel]
        rgb = torch.cat([r[0] for r in renders])
        alpha = torch.cat([r[1] for r in renders])
        if color_gain is not None:  # as each camera sees it
            rgb = rgb * color_gain.to(dev)[sel].exp()[:, None, None, :]
        infos = [(r[3], r[4]["active"]) for r in renders]
        if strategy_state is not None:
            for info, _ in infos:
                if info["means2d"].requires_grad:
                    info["means2d"].retain_grad()
        if mask is not None:
            bg = torch.rand(3, device=dev) if cfg.random_background else torch.ones(3, device=dev)
            pred = rgb + (1.0 - alpha) * bg
            target = gt * mask + (1.0 - mask) * bg
        else:
            # Without masks every pixel shows something. Over a colour that changes at every step
            # a see-through scene is an error: it has to become solid, and cannot make up for
            # what is missing with brighter colours that only look right from the training cameras.
            pred = rgb + (1.0 - alpha) * torch.rand(3, device=dev) if cfg.random_background else rgb
            target = gt
        l1 = (pred - target).abs().mean()
        loss = (1.0 - cfg.lambda_dssim) * l1 + cfg.lambda_dssim * (1.0 - ssim(pred, target))
        if mask is not None and cfg.lambda_mask > 0:
            loss = loss + cfg.lambda_mask * (alpha - mask).abs().mean()
        for state in (r[4] for r in renders):
            if state["means"].shape[0] == 0:
                continue
            if cfg.lambda_scale > 0:
                loss = loss + cfg.lambda_scale / len(renders) * scale_regularizer(
                    state["scales"], cfg.max_scale, cfg.max_anisotropy)
            if state["velocities"] is not None and cfg.lambda_velocity > 0:
                loss = loss + cfg.lambda_velocity / len(renders) * state["velocities"].norm(dim=-1).mean()
            if cfg.lambda_needle > 0:
                loss = loss + cfg.lambda_needle / len(renders) * needle_penalty(state["scales"])
            if shaping and cfg.lambda_flat > 0:
                loss = loss + cfg.lambda_flat / len(renders) * (
                    state["scales"].min(dim=-1).values / state["scales"].max(dim=-1).values.detach()).mean()
            if shaping and cfg.lambda_opacity_entropy > 0:
                idx = state["active"]
                base = torch.sigmoid(view.params["opacities"] if idx is None else view.params["opacities"][idx])
                base = base.clamp(1e-6, 1.0 - 1e-6)
                entropy = -(base * base.log() + (1.0 - base) * (1.0 - base).log())
                loss = loss + cfg.lambda_opacity_entropy / len(renders) * entropy.mean()
        if spread:
            # Blended depth and squared depth give, pixel by pixel, how far apart in depth the
            # Gaussians that make it are: sum_ij w_i w_j (z_i - z_j)^2 / 2 = alpha * E[z^2] - E[z]^2.
            mom = torch.cat([r[3]["moments"] for r in renders])
            apart = (alpha * mom[..., 1:2] - mom[..., 0:1] ** 2).clamp(min=0.0)
            loss = loss + cfg.lambda_ray_spread * apart.sum() / (alpha.detach().sum() + 1.0) / 0.05 ** 2
        return loss, l1.detach(), infos

    try:
        for step in range(iterations):
            active_sh = min(cfg.sh_degree, step // max(cfg.sh_interval, 1))
            shaping = step >= cfg.surface_start * iterations
            jobs = loader.next()
            if workers is None:
                results = [run_job(0, jobs[0], active_sh, shaping)]
            else:
                results = list(workers.map(lambda s: run_job(s, jobs[s], active_sh, shaping), range(len(devices))))
            loss = sum(r[0].to(master) for r in results) / len(results)
            l1 = float(np.mean([r[1].item() for r in results]))

            loss.backward()
            for opt in optimizers.values():
                opt.step()
                opt.zero_grad(set_to_none=True)
            for sched in schedulers:
                sched.step()
            if scale_floor is not None:
                scene.params["scales"].data.clamp_(min=scale_floor)
            if offset_opt is not None:
                if step >= iterations // 10:  # once the scene has taken shape
                    offset_opt.step()
                offset_opt.zero_grad(set_to_none=True)
            if color_opt is not None:
                # From the first step: the starting colours are the average of the cameras, so what
                # each one sees differently shows at once; later the scene would have absorbed it.
                color_opt.step()
                color_opt.zero_grad(set_to_none=True)
                with torch.no_grad():  # no camera is the reference: the scene is their average
                    color_gain -= color_gain.mean(dim=0, keepdim=True)
            if cfg.opacity_decay > 0 and start is None and step < cfg.opacity_decay_stop * iterations:
                # Every Gaussian that was drawn loses a little opacity: those the images need win it
                # back at once, those that only fill space the cameras cannot tell apart fade away
                # and are pruned. (Not when refining a finished scene: there nothing prunes.)
                with torch.no_grad():
                    opacity = scene.params["opacities"]
                    seen = []
                    for result in results:
                        for info, active in result[2]:
                            radii = info["radii"][:, :info["count"]] if "count" in info else info["radii"]
                            shown = (radii > 0).all(dim=-1).any(dim=0)  # in the picture of at least one camera
                            seen.append((torch.nonzero(shown).squeeze(-1) if active is None else active[shown]).to(master))
                    seen = torch.cat(seen).unique()
                    if cfg.opacity_decay_logit:
                        opacity[seen] -= cfg.opacity_decay
                    else:
                        faded = torch.sigmoid(opacity[seen]) * (1.0 - cfg.opacity_decay)
                        opacity[seen] = torch.logit(faded.clamp(1e-6, 1.0 - 1e-6))

            if strategy_state is not None:
                info = _merge_info([pair for r in results for pair in r[2]], scene.num_gaussians, master)
                active = grow if scene.num_gaussians < cfg.max_gaussians else hold
                active.step_post_backward(scene.params, optimizers, strategy_state, step, info,
                                          packed=False)
                if cfg.densify_start < step < active.refine_stop_iter and step % cfg.densify_every == 0:
                    with torch.no_grad():
                        too_big = torch.exp(scene.params["scales"]).max(dim=-1).values > cfg.prune_scale
                    if 0 < int(too_big.sum()) < scene.num_gaussians:
                        remove(scene.params, optimizers, strategy_state, too_big)
                if (cfg.opacity_reset_every > 0 and step > 0 and step % cfg.opacity_reset_every == 0
                        and step < active.refine_stop_iter):
                    reset_opa(scene.params, optimizers, strategy_state, value=cfg.prune_opacity * 2.0)
                if cfg.prune_to_end and step >= active.refine_stop_iter and (
                        step % cfg.densify_every == 0 or step == iterations - 1):
                    with torch.no_grad():  # what has faded since densification stopped
                        faint = torch.sigmoid(scene.params["opacities"]) < cfg.prune_opacity
                    if 0 < int(faint.sum()) < scene.num_gaussians:
                        remove(scene.params, optimizers, strategy_state, faint)

            ema_loss = loss.item() if ema_loss is None else 0.9 * ema_loss + 0.1 * loss.item()
            last = step == iterations - 1
            if step % cfg.log_every == 0 or last:
                record = {"step": step, "loss": round(ema_loss, 5), "l1": round(l1, 5),
                          "gaussians": scene.num_gaussians, "seconds": round(time.time() - t_start, 1)}
                if (cfg.eval_every > 0 and step % cfg.eval_every == 0) or last:
                    record["psnr_" + ("holdout" if val_cams else "train")] = round(
                        evaluate_views(scene, ds, eval_frames, eval_cams, master, backends[0],
                                       cfg.antialiased, learned_offsets(), learned_colors(), masked)["psnr"], 2)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                print("[train] " + "  ".join(f"{k}={v}" for k, v in record.items()), flush=True)
            if cfg.save_every > 0 and step > 0 and step % cfg.save_every == 0:
                save_checkpoint(out_dir / f"ckpt_{step:06d}.pt", scene, cfg_dict, meta, step)
    finally:
        loader.close()
        if workers is not None:
            workers.shutdown(wait=False)
        log_file.close()

    meta["time_offsets"] = learned_offsets()
    meta["camera_colors"] = learned_colors()
    if color_opt is not None:
        print("[train] learned colour gains (r g b): " + ", ".join(
            f"{c} " + " ".join(f"{v:.2f}" for v in gain) for c, gain in meta["camera_colors"].items()))
    if offsets is not None:
        ms = 1000.0 / ds.fps
        print("[train] learned time offsets relative to " + train_cams[0] + ": " + ", ".join(
            f"{c} {v:+.2f} frames ({v * ms:+.1f} ms)" for c, v in meta["time_offsets"].items() if c != train_cams[0]))
    ckpt_path = out_dir / "ckpt.pt"
    save_checkpoint(ckpt_path, scene, cfg_dict, meta, iterations)
    # The outcome in a small text file, for tools that should not have to load the checkpoint.
    summary = {
        "iterations": iterations, "seconds": round(time.time() - t_start, 1),
        "gaussians": scene.num_gaussians, "scenery": scene.num_scenery, "frames": core[1] - core[0], "fps": ds.fps,
        "train_cameras": train_cams, "holdout_cameras": val_cams, "devices": [str(d) for d in devices],
        "time_offsets": meta["time_offsets"], "camera_colors": meta["camera_colors"], "coverage": coverage}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[train] done in {time.time() - t_start:.0f}s -> {ckpt_path}")
    return ckpt_path, summary


# Options that change how a training runs but not what it produces.
_NOT_IN_RESULT = ("devices", "loader_threads", "cache_gb", "log_every", "eval_every", "save_every")


def _fingerprint(data_dir: Path) -> str:
    """Changes whenever the dataset is prepared, masked or initialised again."""
    parts = []
    for name in ("meta.json", "calibration.json", "init_points.ply", "masks_preview.jpg"):
        f = data_dir / name
        parts.append(f"{name}:{f.stat().st_size}:{f.stat().st_mtime_ns}" if f.exists() else f"{name}:-")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def scenery_config(cfg: TrainConfig, cameras: list[str]) -> TrainConfig:
    """The options the empty set is trained with, from those of a take.

    The opacity decay has its own value, off by default. In a take a Gaussian is drawn in the few
    steps that show its instant; in a static scene it is drawn at every step, so the same decay
    is many times stronger: floors and backdrops stay see-through and lose most of their
    Gaussians. What only concerns the take goes back to its default: two takes that differ in
    that share one set.
    """
    plain = TrainConfig()
    return dataclasses.replace(
        cfg, full_scene=False, static=True, static_frame=0, iterations=cfg.scenery_iterations,
        max_gaussians=cfg.scenery_max_gaussians, max_scale=cfg.scenery_max_scale,
        prune_scale=cfg.scenery_prune_scale, holdout=[c for c in cfg.holdout if c in cameras],
        learn_time_offsets=False, learn_camera_colors=False, opacity_decay=cfg.scenery_opacity_decay,
        opacity_decay_stop=plain.opacity_decay_stop, opacity_decay_logit=plain.opacity_decay_logit,
        lambda_opacity_entropy=plain.lambda_opacity_entropy, lambda_ray_spread=plain.lambda_ray_spread,
        lambda_flat=plain.lambda_flat, init_flatten=plain.init_flatten, init_opacity=plain.init_opacity,
        init_duration=plain.init_duration, lambda_velocity=plain.lambda_velocity,
        segment_frames=plain.segment_frames, segment_overlap=plain.segment_overlap)


def _train_scenery(data_dir: Path, out_dir: Path, cfg: TrainConfig, resume: bool) -> Path:
    """Reconstruct the empty set as a static scene in ``out_dir/scenery``; returns its checkpoint."""
    source = data_dir / "scenery"
    if not (source / "meta.json").exists():
        raise FileNotFoundError(f"no {source}: run 'mister4d init {data_dir} --full-scene' first")
    target = out_dir / "scenery"
    target.mkdir(parents=True, exist_ok=True)
    own = scenery_config(cfg, read_meta(source)["cameras"])
    tag = json.loads(json.dumps({
        "fingerprint": _fingerprint(source),
        "config": {k: v for k, v in dataclasses.asdict(own).items() if k not in _NOT_IN_RESULT}}))
    tag_file = target / "segment.json"
    if resume and (target / "ckpt.pt").exists() and tag_file.exists() and json.loads(tag_file.read_text()) == tag:
        print("[train] the scenery is already trained with these data and options: kept", flush=True)
    else:
        tag_file.unlink(missing_ok=True)
        print("[train] scenery: the empty set, as a static scene", flush=True)
        _train_window(source, target, own)
        tag_file.write_text(json.dumps(tag, indent=2))
    return target / "ckpt.pt"


def train(data_dir: str | Path, out_dir: str | Path, cfg: TrainConfig, resume: bool = True) -> Path:
    """Train a take; returns its checkpoint, or the manifest of its segments.

    A take longer than about one and a half ``cfg.segment_frames`` is trained
    in segments. With ``resume`` the segments already trained from the same
    data with the same options are kept: starting the same command again
    continues an interrupted job.
    """
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_frames = len(read_meta(data_dir)["frame_ids"])
    static = cfg.static or num_frames == 1
    segments = [(0, num_frames)] if static else plan_segments(num_frames, cfg.segment_frames)
    manifest = out_dir / MANIFEST
    manifest.unlink(missing_ok=True)  # it is there only while every segment is
    scenery, scenery_ckpt = None, None
    if cfg.full_scene and not static:
        scenery_ckpt = _train_scenery(data_dir, out_dir, cfg, resume)
        scenery = scenery_of(load_checkpoint(scenery_ckpt)[0])
        print(f"[train] take: what moves, on a scenery of {scenery[0]['means'].shape[0]} Gaussians", flush=True)
    if len(segments) == 1:
        return _train_window(data_dir, out_dir, cfg, scenery=scenery,
                             scenery_file="scenery/ckpt.pt" if scenery else None)[0]

    (out_dir / "ckpt.pt").unlink(missing_ok=True)  # of an earlier, shorter take
    identity = json.loads(json.dumps({
        "fingerprint": _fingerprint(data_dir),
        "config": {k: v for k, v in dataclasses.asdict(cfg).items() if k not in _NOT_IN_RESULT}}))
    print(f"[train] long take: {num_frames} frames in {len(segments)} segments", flush=True)
    t_start = time.time()
    done, summaries, offsets, colors = [], [], None, None
    for i, (first, last) in enumerate(segments):
        seg = segment_dir(out_dir, i)
        window = (max(first - cfg.segment_overlap, 0), min(last + cfg.segment_overlap, num_frames))
        tag = {**identity, "segment": [first, last], "window": list(window), "fixed_offsets": offsets,
               "fixed_colors": colors}
        tag_file, summary_file = seg / "segment.json", seg / "train_summary.json"
        print(f"[train] segment={i + 1}/{len(segments)} frames={first}-{last - 1}", flush=True)
        kept = (resume and (seg / "ckpt.pt").exists() and tag_file.exists() and summary_file.exists()
                and json.loads(tag_file.read_text()) == tag)
        if kept:
            summary = json.loads(summary_file.read_text())
            print(f"[train] segment {i + 1} is already trained with these data and options: kept", flush=True)
        else:
            seg.mkdir(parents=True, exist_ok=True)
            tag_file.unlink(missing_ok=True)
            _, summary = _train_window(data_dir, seg, cfg, window=window, core=(first, last), fixed_offsets=offsets,
                                       fixed_colors=colors, scenery=scenery,
                                       scenery_file="../scenery/ckpt.pt" if scenery else None)
            tag_file.write_text(json.dumps(tag, indent=2))
        if i == 0 and cfg.learn_time_offsets:
            offsets = summary["time_offsets"]  # one clock per camera for the whole take
        if i == 0 and cfg.learn_camera_colors:
            colors = summary["camera_colors"]  # and one look
        summaries.append(summary)
        done.append({"dir": seg.name, "first": first, "last": last, "gaussians": summary["gaussians"]})

    total = {**summaries[0],
             "iterations": sum(x["iterations"] for x in summaries),
             "seconds": round(sum(x["seconds"] for x in summaries), 1),
             "gaussians": sum(x["gaussians"] for x in summaries),
             "frames": num_frames, "segments": len(segments)}
    (out_dir / "train_summary.json").write_text(json.dumps(total, indent=2))
    write_manifest(out_dir, done, frames=num_frames)
    print(f"[train] all {len(segments)} segments done in {time.time() - t_start:.0f}s -> {manifest}", flush=True)
    return manifest
