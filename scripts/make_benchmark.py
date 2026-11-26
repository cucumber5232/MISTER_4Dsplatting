"""Benchmark datasets with ground truth: the clothed human of mister4d.human on the studio set.

    python scripts/make_benchmark.py dataset OUT --width 960 --height 540 --frames 12
    python scripts/make_benchmark.py raw OUT --width 960 --height 540 --frames 30
    python scripts/make_benchmark.py sheets DATASET   # contact sheets of an existing dataset
    python scripts/make_benchmark.py set RUN          # the trained set of a whole scene against the studio

Development tool: it needs the package installed (run it with the Python of .venv).

A dataset is what the pipeline has after "prepare" (pinhole cameras), plus:
  background/<cam>.png          the empty set
  masks/<cam>/<frame>.png       from "mister4d masks" (plate difference), NOT the true ones
  init_points.ply               from "mister4d init --exclude <verification cameras>"
  scenery/                      with --full-scene, instead of the masks: the empty set and its starting points
  truth/<cam>/<frame>.npz       depth (float32, metres along the optical axis, 0 = no subject)
                                and mask (uint8, true coverage 0..255)
  truth/cameras.json            {"train": [...], "verify": [...]}
Training cameras cam01.. sit on an arc around the subject like the demo rig;
verification cameras ver01.. are in the gaps, beyond one end and higher up,
and must never be used for init or training.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from mister4d import cli
from mister4d.camera import Camera, Rig, look_at
from mister4d.dataset import frame_name, write_meta
from mister4d.human import StudioView, human_at, load_human
from mister4d.synth import SynthConfig, make_raw_session

TARGET = np.array([0.0, 0.0, 1.08])
RADIUS = 3.0
VIEW_METRES = 2.36  # height seen at the subject: a 1.77 m person fills 75% of the image
FPS = 30.0


def bench_rig(width: int, height: int, train: int = 4, arc: float = 150.0,
              between: int | None = None, high: int = 0) -> tuple[Rig, list[str], list[str]]:
    """Training cameras on an arc like the demo rig, verification cameras between and around them.

    ``between`` limits how many of the gaps between training cameras get a verification camera
    (default: all of them); with many cameras a few, evenly spread, are enough. ``high`` adds that
    many training cameras 2.6 m above the floor, looking down: without them nothing sees the
    subject from above.
    """
    fy = height * RADIUS / VIEW_METRES
    scale = width / 960.0
    cams = {}

    def add(name, az, z=None, elevation=None, k=0):
        az = np.radians(az)
        if elevation is None:
            eye = np.array([RADIUS * np.cos(az), RADIUS * np.sin(az), z])
        else:
            el = np.radians(elevation)
            d = np.hypot(RADIUS, 1.4 - TARGET[2])
            eye = TARGET + d * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        R, t = look_at(eye, TARGET)
        K = [[fy, 0, (width - 1) / 2 + 3.0 * (k - 1) * scale], [0, fy, (height - 1) / 2 - 2.0 * k * scale], [0, 0, 1]]
        cams[name] = Camera(name, width, height, K, np.zeros(5), R, t)

    az = np.linspace(-arc / 2, arc / 2, train) if train > 1 else np.array([0.0])
    names_train = [f"cam{i + 1:02d}" for i in range(train)]
    for i, name in enumerate(names_train):
        add(name, az[i], z=1.4 + (0.25 if i % 2 else -0.1), k=i)
    for j in range(high):
        names_train.append(f"cam{train + j + 1:02d}")
        add(names_train[-1], -arc / 2 + arc * (j + 0.5) / high, z=2.6, k=train + j)
    verify = [(a + b) / 2 for a, b in zip(az[:-1], az[1:])]  # the middle of every gap
    if between is not None and between < len(verify):
        verify = [verify[i] for i in sorted(set(np.linspace(0, len(verify) - 1, max(between, 1)).round().astype(int)))]
    k = 0
    for a in verify:
        k += 1
        add(f"ver{k:02d}", a, z=1.4, k=k)
    add(f"ver{k + 1:02d}", az[-1] + 25.0, z=1.4, k=k + 1)  # beyond one end of the arc
    base = np.degrees(np.arctan2(1.4 - TARGET[2], RADIUS))
    add(f"ver{k + 2:02d}", 0.0, elevation=base + 25.0, k=k + 2)  # middle of the arc, 25 degrees higher
    names_verify = [n for n in cams if n.startswith("ver")]
    volume = {"center": [0.0, 0.0, 1.1], "size": [2.4, 2.4, 2.4]}
    return Rig(cams, volume=volume, undistorted=True), names_train, names_verify


def hand_speed(frame: int) -> float:
    """Speed of the tip of the right middle finger between frame - 1 and frame, m/s."""
    h = load_human()
    tip = int(np.argmax(h.skin_bones[:, 0] == h.index["finger3-3.R"]))
    return float(np.linalg.norm(human_at(frame / FPS).verts[tip] - human_at((frame - 1) / FPS).verts[tip]) * FPS)


def fastest_window(count: int, last: int = 60) -> int:
    speeds = np.array([hand_speed(f) for f in range(1, last + 1)])
    means = [speeds[i:i + count].mean() for i in range(len(speeds) - count + 1)]
    return int(np.argmax(means)) + 1


def _render_camera(job) -> str:
    """Everything one camera sees: the empty set, every frame with its noise, the truth."""
    out, name, cam, ids, noise, seed = job
    rng = np.random.default_rng(seed)
    view = StudioView(Camera.from_dict(name, cam))
    cv2.imwrite(str(out / "background" / f"{name}.png"),
                cv2.cvtColor((view.background * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR))
    for fid in ids:
        image, cover, depth = view.render(fid / FPS)
        noisy = image * 255.0 + rng.normal(0.0, noise, size=image.shape)
        cv2.imwrite(str(out / "images" / name / f"{frame_name(fid)}.png"),
                    cv2.cvtColor(np.clip(noisy + 0.5, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        np.savez_compressed(out / "truth" / name / f"{frame_name(fid)}.npz", depth=depth.astype(np.float32),
                            mask=(cover * 255.0 + 0.5).astype(np.uint8))
    return name


def make_dataset(out: Path, width: int, height: int, first: int, frames: int, train: int = 4,
                 arc: float = 150.0, noise: float = 1.5, seed: int = 0, points: int | None = None,
                 full_scene: bool = False, between: int | None = None, jobs: int = 1, high: int = 0) -> Path:
    out = Path(out)
    rig, names_train, names_verify = bench_rig(width, height, train, arc, between, high)
    ids = list(range(first, first + frames))
    t0 = time.time()
    (out / "background").mkdir(parents=True, exist_ok=True)
    for name in rig.cameras:
        for sub in ("images", "truth"):
            (out / sub / name).mkdir(parents=True, exist_ok=True)
    work = [(out, name, cam.to_dict(), ids, noise, seed + 1000 * i) for i, (name, cam) in enumerate(rig.cameras.items())]
    if jobs > 1:  # one camera per process: the rendering is the slow part
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(min(jobs, len(work))) as pool:
            for name in pool.map(_render_camera, work):
                print(f"[bench] {name} rendered ({time.time() - t0:.0f} s)", flush=True)
    else:
        for job in work:
            _render_camera(job)
    rig.save(out / "calibration.json")
    write_meta(out, FPS, ids, rig.names, {"source": "synthetic human (make_benchmark.py)",
                                          "hand_speed_m_s": [round(hand_speed(f), 2) for f in ids]})
    (out / "truth" / "cameras.json").write_text(json.dumps({"train": names_train, "verify": names_verify}, indent=2))
    print(f"[bench] {out}: {len(rig.names)} cameras x {frames} frames at {width}x{height} "
          f"in {time.time() - t0:.0f} s")
    # Masks and initial points exactly as a user would get them. For the whole scene there are no
    # masks: the set gets its own starting points and the take starts from what changes.
    init = ["init", str(out), "--exclude", *names_verify]
    if points:
        init += ["--points", str(points)]
    if full_scene:
        assert cli.main(init + ["--full-scene"]) == 0
    else:
        assert cli.main(["masks", str(out)]) == 0
        assert cli.main(init) == 0
        report_masks(out)
    return out


def report_masks(out: Path) -> None:
    """How far the masks of the pipeline are from the true coverage."""
    meta = json.loads((out / "meta.json").read_text())
    ious = []
    for cam in meta["cameras"]:
        for fid in meta["frame_ids"]:
            mask = cv2.imread(str(out / "masks" / cam / f"{frame_name(fid)}.png"), cv2.IMREAD_GRAYSCALE) > 127
            truth = np.load(out / "truth" / cam / f"{frame_name(fid)}.npz")["mask"] > 127
            ious.append((mask & truth).sum() / max((mask | truth).sum(), 1))
    print(f"[bench] masks from the pipeline against true coverage: IoU mean {np.mean(ious):.3f}, "
          f"worst {np.min(ious):.3f}")


def make_raw(out: Path, width: int, height: int, frames: int, cameras: int = 4, seed: int = 0) -> Path:
    cfg = SynthConfig(num_cameras=cameras, width=width, height=height, take_frames=frames, seed=seed,
                      subject="human")
    make_raw_session(out, cfg)
    return Path(out)


def check_raw(out: Path) -> float:
    """Ingest and calibrate the raw session as the demo does; returns the RMS error in pixels."""
    assert cli.main(["ingest", str(out / "raw" / "calib"), "--out", str(out / "work" / "calib"),
                     "--sync", "audio", "--start", "8"]) == 0
    assert cli.main(["calibrate", str(out / "work" / "calib"), "--out", str(out / "work" / "calibration.json")]) == 0
    rig, truth = Rig.load(out / "work" / "calibration.json"), Rig.load(out / "gt" / "calibration.json")
    print(f"[bench] calibration RMS {rig.report['rms_px']:.3f} px; focal length error "
          + ", ".join(f"{n} {abs(rig.cameras[n].K[0, 0] / truth.cameras[n].K[0, 0] - 1) * 100:.2f}%" for n in rig.names))
    return float(rig.report["rms_px"])


# ---------------------------------------------------------------------------
# Contact sheets
# ---------------------------------------------------------------------------


def _label(img: np.ndarray, text: str) -> np.ndarray:
    img = np.ascontiguousarray(img)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _grid(tiles: list[np.ndarray], cols: int) -> np.ndarray:
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.concatenate([np.concatenate(tiles[i:i + cols], axis=1) for i in range(0, len(tiles), cols)], axis=0)


def _turbo(depth: np.ndarray) -> np.ndarray:
    on = depth > 0
    lo, hi = (depth[on].min(), depth[on].max()) if on.any() else (0.0, 1.0)
    d = np.clip((depth - lo) / max(hi - lo, 1e-6), 0, 1)
    out = cv2.applyColorMap((255 - d * 220).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[~on] = 0
    return out


def sheets(data: Path, out_dir: Path | None = None, scale: float = 1.0) -> list[Path]:
    """Images, masks (pipeline in green, missed in red, extra in blue) and depth of a dataset."""
    data = Path(data)
    out_dir = Path(out_dir or data)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((data / "meta.json").read_text())
    ids, cams = meta["frame_ids"], meta["cameras"]
    mid = ids[len(ids) // 2]
    written = []

    def tile(cam, fid, what):
        truth = np.load(data / "truth" / cam / f"{frame_name(fid)}.npz")
        if what == "image":
            img = cv2.imread(str(data / "images" / cam / f"{frame_name(fid)}.png"))
        elif what == "mask":
            mask = cv2.imread(str(data / "masks" / cam / f"{frame_name(fid)}.png"), cv2.IMREAD_GRAYSCALE) > 127
            true = truth["mask"] > 127
            img = np.zeros(mask.shape + (3,), np.uint8)
            img[mask & true] = (90, 200, 90)
            img[~mask & true] = (40, 40, 255)
            img[mask & ~true] = (255, 120, 40)
        else:
            img = _turbo(truth["depth"])
        if scale != 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST if what != "image" else cv2.INTER_CUBIC)
        return _label(img, f"{cam} f{fid}")

    for what in ("image", "mask", "depth"):
        path = out_dir / f"{data.name}_{what}s.png"
        cv2.imwrite(str(path), _grid([tile(c, mid, what) for c in cams], 3))
        written.append(path)
    if len(ids) > 3:  # the motion, as seen by the first two training cameras and one verification camera
        pick = [cams[0], cams[1], [c for c in cams if c.startswith("ver")][1]]
        path = out_dir / f"{data.name}_motion.png"
        cv2.imwrite(str(path), _grid([tile(c, f, "image") for c in pick for f in ids[::2]], len(ids[::2])))
        written.append(path)
    return written


def closeups(out: Path, frames=(0, 14, 26, 50), height: int = 900) -> Path:
    """The subject as the benchmark cameras see it, large enough to judge how real it looks."""
    rig, names_train, names_verify = bench_rig(height * 16 // 9, height)
    rows = []
    for fid in frames:
        row = []
        for name in (names_train[0], names_train[1], names_verify[1], names_train[2], names_train[3], names_verify[4]):
            cam = rig.cameras[name]
            image, cover, _ = StudioView(cam).render(fid / FPS)
            cols = np.nonzero(cover.max(axis=0) > 0.5)[0]
            centre = int(np.clip((cols.min() + cols.max()) // 2, height * 0.3, cam.width - height * 0.3))
            crop = image[:, centre - int(height * 0.3):centre + int(height * 0.3)]
            row.append(_label(cv2.cvtColor((crop * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR), f"{name} f{fid}"))
        rows.append(np.concatenate(row, axis=1))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.concatenate(rows, axis=0))
    return out


def studio_distance(xyz: np.ndarray) -> np.ndarray:
    """Distance (m) of points from the nearest surface of the studio: its floor, its wall, the lamp."""
    from mister4d.human import WALL_RADIUS, _stand

    xyz = np.asarray(xyz, dtype=np.float64)
    r = np.hypot(xyz[:, 0], xyz[:, 1])
    beyond = np.maximum(r - WALL_RADIUS, 0.0)
    off = np.minimum(np.hypot(xyz[:, 2], beyond), np.hypot(r - WALL_RADIUS, np.minimum(xyz[:, 2], 0.0)))
    for center, R, radii, *_ in _stand():
        off = np.minimum(off, np.maximum(np.linalg.norm((xyz - center) @ R, axis=1) - float(np.max(radii)), 0.0))
    return off


def studio_depth(cam) -> np.ndarray:
    """Depth of the empty studio along the pixels of ``cam`` (floor and wall; the lamp is left out)."""
    from mister4d.human import WALL_RADIUS

    v, u = np.mgrid[0:cam.height, 0:cam.width].astype(np.float64)
    K = np.asarray(cam.K)
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1) @ np.asarray(cam.R)
    o = np.asarray(cam.center)
    a = rays[..., 0] ** 2 + rays[..., 1] ** 2
    b = rays[..., 0] * o[0] + rays[..., 1] * o[1]
    wall = (-b + np.sqrt(np.maximum(b * b - a * (o[0] ** 2 + o[1] ** 2 - WALL_RADIUS ** 2), 0.0))) / np.maximum(a, 1e-12)
    floor = np.where(rays[..., 2] < -1e-6, -o[2] / np.minimum(rays[..., 2], -1e-6), np.inf)
    return np.minimum(wall, floor)


def check_set(run: Path, device: str = "auto", keep: int = 200_000) -> dict:
    """The set of a whole scene against the studio it was rendered from.

    How much of it hangs in mid air where the subject and the cameras are, and, from every
    verification camera, how much of the picture is drawn in front of where the set is: a veil
    there does not show in the measures of the training cameras. Writes ``evalq/set.json`` and
    ``evalq/set_points.npz`` (up to ``keep`` Gaussians: position, opacity, size), small enough
    to send.
    """
    import torch

    from mister4d.dataset import MultiViewDataset
    from mister4d.model import load_checkpoint, render_scene
    from mister4d.rasterizer import resolve_backend
    from mister4d.trainer import pick_device

    dev = pick_device(device)
    scene, ckpt = load_checkpoint(run / "scenery" / "ckpt.pt", dev)
    source = Path(ckpt["meta"]["data_dir"])
    ds = MultiViewDataset(source, masks=False)
    names = json.loads((source.parent / "truth" / "cameras.json").read_text())
    backend = resolve_backend(ckpt["config"].get("backend", "auto"), dev)
    means = scene.params["means"].detach().cpu().numpy()
    opacity = torch.sigmoid(scene.params["opacities"].detach()).cpu().numpy().ravel()
    size = torch.exp(scene.params["scales"].detach()).max(dim=-1).values.cpu().numpy()
    r = np.hypot(means[:, 0], means[:, 1])
    air = (studio_distance(means) > 0.4) & (r < 5.5) & (means[:, 2] > 0.3) & (means[:, 2] < 3.5)
    out = {"gaussians": int(len(means)), "in_air": int(air.sum()),
           "in_air_opacity_share": float(opacity[air].sum() / max(opacity.sum(), 1e-9)),
           "in_air_largest_m": float(size[air].max()) if air.any() else 0.0, "cameras": {}}
    for group in ("train", "verify"):
        for name in names[group]:
            cam = ds.rig.cameras[name]
            viewmats, Ks = ds.camera_tensors([name], dev)
            gt, _ = ds.batch(0, [name], dev)
            with torch.no_grad():
                rgb, alpha, depth, *_ = render_scene(scene, 0, viewmats, Ks, ds.width, ds.height, backend=backend)
            drawn = (depth[0, ..., 0] / alpha[0, ..., 0].clamp(min=1e-3)).cpu().numpy()
            true = studio_depth(cam)
            near = float(np.linalg.norm(means - np.asarray(cam.center), axis=1).min()) if len(means) else 0.0
            around = np.linalg.norm(means - np.asarray(cam.center), axis=1) < 1.0
            out["cameras"][name] = {
                "group": group, "psnr": float(-10.0 * torch.log10(((rgb.clamp(0, 1) - gt) ** 2).mean()).item()),
                "alpha": float(alpha.mean()), "depth_error_m": float(np.median(np.abs(drawn - true))),
                "in_front_share": float(((true - drawn) > 0.5).mean()),
                "nearest_gaussian_m": near, "opacity_within_1m": float(opacity[around].sum())}
    for group in ("train", "verify"):
        rows = [v for v in out["cameras"].values() if v["group"] == group]
        out[group] = {k: float(np.mean([row[k] for row in rows])) for k in ("psnr", "depth_error_m", "in_front_share")}
    target = run / "evalq"
    target.mkdir(parents=True, exist_ok=True)
    (target / "set.json").write_text(json.dumps(out, indent=2))
    pick = np.random.default_rng(0).choice(len(means), size=min(keep, len(means)), replace=False)
    np.savez_compressed(target / "set_points.npz", xyz=means[pick].astype(np.float16),
                        opacity=np.round(opacity[pick] * 255).astype(np.uint8), size=size[pick].astype(np.float16))
    print(f"[bench] set: {out['gaussians']} Gaussians, {out['in_air']} in mid air in the room "
          f"({100 * out['in_air_opacity_share']:.1f}% of the opacity)")
    for name, row in out["cameras"].items():
        if row["group"] == "verify":
            print(f"[bench]   {name}: psnr {row['psnr']:.2f}  drawn in front of the set {100 * row['in_front_share']:.1f}%  "
                  f"depth error {row['depth_error_m']:.2f} m  nearest Gaussian {row['nearest_gaussian_m']:.2f} m")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    p = sub.add_parser("dataset")
    p.add_argument("out")
    p.add_argument("--width", type=int, default=480)
    p.add_argument("--height", type=int, default=270)
    p.add_argument("--first", type=int, default=None, help="first frame of the action (default: the fastest part)")
    p.add_argument("--frames", type=int, default=3)
    p.add_argument("--train-cameras", type=int, default=4)
    p.add_argument("--arc", type=float, default=150.0)
    p.add_argument("--noise", type=float, default=1.5)
    p.add_argument("--points", type=int, default=None, help="total initial points (default: that of 'mister4d init')")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--full-scene", action="store_true",
                   help="prepare the whole scene, set included (train with 'mister4d train --full-scene')")
    p.add_argument("--verify-between", type=int, default=None,
                   help="verification cameras in the gaps between training cameras (default: one per gap)")
    p.add_argument("--jobs", type=int, default=1, help="cameras rendered at the same time, one process each")
    p.add_argument("--high", type=int, default=0, help="more training cameras, 2.6 m above the floor")
    p = sub.add_parser("raw")
    p.add_argument("out")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--cameras", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p = sub.add_parser("sheets")
    p.add_argument("data")
    p.add_argument("--out", default=None)
    p.add_argument("--scale", type=float, default=1.0)
    p = sub.add_parser("closeups")
    p.add_argument("out", help="image to write")
    p = sub.add_parser("set", help="the trained set of a whole scene against the studio")
    p.add_argument("run")
    p.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.what == "dataset":
        first = args.first if args.first is not None else fastest_window(args.frames)
        make_dataset(Path(args.out), args.width, args.height, first, args.frames, args.train_cameras, args.arc,
                     args.noise, args.seed, args.points, args.full_scene, args.verify_between, args.jobs,
                     args.high)
    elif args.what == "raw":
        check_raw(make_raw(Path(args.out), args.width, args.height, args.frames, args.cameras, args.seed))
    elif args.what == "sheets":
        print(*sheets(Path(args.data), args.out, args.scale), sep="\n")
    elif args.what == "set":
        check_set(Path(args.run), args.device)
    else:
        print(closeups(Path(args.out)))


if __name__ == "__main__":
    main()
