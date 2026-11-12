"""Synthetic multi-camera recordings of an animated mannequin.

Lets the whole pipeline run (and be tested) without any hardware: the
generator ray-casts a figure made of ellipsoids from a ring of virtual cameras
and writes exactly what a real session would provide:

- ``raw/calib/<cam>.mp4``  a ChArUco board carried through the volume
- ``raw/empty/<cam>.mp4``  the empty set
- ``raw/take/<cam>.mp4``   the subject moving
- ``gt/``                  ground truth calibration and start offsets

The cameras have lens distortion, start recording at different instants and
carry an audio track with a clap, so calibration, undistortion and
synchronisation are all exercised. Nothing here is used on real footage.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera, Rig, look_at
from .charuco import BoardSpec
from .dataset import frame_name, write_meta, write_points_ply
from .video import VideoWriter

AUDIO_RATE = 16000


@dataclass
class SynthConfig:
    num_cameras: int = 4
    width: int = 960
    height: int = 540
    fps: float = 30.0
    take_frames: int = 40
    calib_frames: int = 90
    empty_frames: int = 8
    rig_radius: float = 3.0
    rig_height: float = 1.4
    arc_degrees: float = 150.0  # angular spread of the cameras around the subject
    distortion_k1: float = -0.06
    noise: float = 1.5  # sensor noise, 8-bit standard deviation
    max_offset: int = 4  # cameras start recording up to this many frames apart
    event_frame: int = 6  # global frame of the clap / flash (after every camera started)
    seed: int = 0


# ---------------------------------------------------------------------------
# Cameras
# ---------------------------------------------------------------------------


def make_rig(cfg: SynthConfig, distorted: bool = True) -> Rig:
    target = np.array([0.0, 0.0, 1.0])
    fy = cfg.height * cfg.rig_radius / 2.7
    cams = {}
    half = np.radians(cfg.arc_degrees) / 2.0
    for i in range(cfg.num_cameras):
        frac = 0.5 if cfg.num_cameras == 1 else i / (cfg.num_cameras - 1)
        az = -half + 2.0 * half * frac
        height = cfg.rig_height + (0.25 if i % 2 else -0.1)
        eye = np.array([cfg.rig_radius * np.cos(az), cfg.rig_radius * np.sin(az), height])
        R, t = look_at(eye, target)
        K = np.array([[fy, 0, (cfg.width - 1) / 2 + 3.0 * (i - 1)],
                      [0, fy, (cfg.height - 1) / 2 - 2.0 * i],
                      [0, 0, 1.0]])
        dist = [cfg.distortion_k1 if distorted else 0.0, 0, 0, 0, 0]
        name = f"cam{i + 1:02d}"
        cams[name] = Camera(name, cfg.width, cfg.height, K, dist, R, t)
    volume = {"center": [0.0, 0.0, 1.1], "size": [2.4, 2.4, 2.4]}
    return Rig(cams, volume=volume, undistorted=not distorted)


def camera_rays(cam: Camera, supersample: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Ray origin [3] and directions [H*s, W*s, 3] in world coordinates."""
    s = supersample
    u = (np.arange(cam.width * s) + 0.5) / s - 0.5
    v = (np.arange(cam.height * s) + 0.5) / s - 0.5
    uu, vv = np.meshgrid(u, v)
    pix = np.stack([uu, vv], axis=-1).reshape(-1, 1, 2).astype(np.float64)
    norm = cv2.undistortPoints(pix, cam.K, cam.dist).reshape(-1, 2)
    d_cam = np.concatenate([norm, np.ones((len(norm), 1))], axis=1)
    d_world = d_cam @ cam.R  # R^T applied to every row
    d_world /= np.linalg.norm(d_world, axis=1, keepdims=True)
    return cam.center, d_world.reshape(cam.height * s, cam.width * s, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

LIGHT = np.array([0.45, 0.25, 0.86], dtype=np.float32)
LIGHT /= np.linalg.norm(LIGHT)


def render_background(origin: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Matte floor and a neutral cylindrical backdrop. Returns RGB float [H, W, 3]."""
    o = origin.astype(np.float32)
    dz = dirs[..., 2]
    rgb = np.empty(dirs.shape, dtype=np.float32)

    # Backdrop: vertical cylinder of radius 6 m.
    a = dirs[..., 0] ** 2 + dirs[..., 1] ** 2 + 1e-12
    b = o[0] * dirs[..., 0] + o[1] * dirs[..., 1]
    c = o[0] ** 2 + o[1] ** 2 - 36.0
    s_wall = (-b + np.sqrt(np.maximum(b * b - a * c, 0.0))) / a
    hz = o[2] + s_wall * dz
    ang = np.arctan2(o[1] + s_wall * dirs[..., 1], o[0] + s_wall * dirs[..., 0])
    wall = 0.52 + 0.04 * np.sin(ang * 9.0) * np.cos(hz * 2.5) - 0.02 * hz
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = wall * 0.98, wall, wall * 1.03

    # Floor: plane z = 0 with a faint tile pattern.
    with np.errstate(divide="ignore", invalid="ignore"):
        s_floor = np.where(dz < -1e-6, -o[2] / dz, np.inf)
    floor = s_floor < s_wall
    fx = o[0] + s_floor[floor] * dirs[..., 0][floor]
    fy = o[1] + s_floor[floor] * dirs[..., 1][floor]
    tiles = ((np.floor(fx / 0.5) + np.floor(fy / 0.5)) % 2) * 0.05
    shade = 0.42 + tiles
    rgb[floor] = np.stack([shade * 1.02, shade, shade * 0.96], axis=-1)
    return rgb


def _limb(a: np.ndarray, b: np.ndarray, radius: float, color) -> dict:
    axis = b - a
    length = np.linalg.norm(axis)
    x = axis / length
    helper = np.array([0.0, 0.0, 1.0]) if abs(x[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    y = np.cross(helper, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return {"center": (a + b) / 2, "R": np.stack([x, y, z], axis=1),
            "radii": np.array([length / 2 + radius * 0.6, radius, radius]), "color": color}


def _blob(center, radii, color) -> dict:
    return {"center": np.asarray(center, float), "R": np.eye(3),
            "radii": np.asarray(radii, float), "color": color}


def mannequin(phase: float) -> list[dict]:
    """Ellipsoids of the figure at ``phase`` in [0, 1] (start to end of the take)."""
    skin, shirt, pants, shoes = (0.90, 0.72, 0.60), (0.78, 0.22, 0.18), (0.18, 0.25, 0.52), (0.12, 0.12, 0.12)
    ease = phase * phase * (3.0 - 2.0 * phase)
    parts = [
        _blob([0, 0, 1.25], [0.13, 0.19, 0.30], shirt),
        _blob([0, 0, 0.95], [0.13, 0.18, 0.14], pants),
        _blob([0.01, 0, 1.68], [0.10, 0.09, 0.12], skin),
    ]
    for side in (1.0, -1.0):
        hip = np.array([0.0, 0.10 * side, 0.92])
        knee = np.array([0.02, 0.11 * side, 0.50])
        ankle = np.array([0.0, 0.11 * side, 0.08])
        parts += [_limb(hip, knee, 0.085, pants), _limb(knee, ankle, 0.06, pants),
                  _blob([0.07, 0.11 * side, 0.04], [0.13, 0.05, 0.04], shoes)]
    # Left arm hangs with a small swing; right arm is raised and waves.
    for side in (1.0, -1.0):
        shoulder = np.array([0.0, 0.23 * side, 1.47])
        if side > 0:
            lift = np.radians(-78.0 + 6.0 * np.sin(2 * np.pi * phase))
            bend = np.radians(8.0)
        else:
            lift = np.radians(-75.0 + 130.0 * ease)
            bend = np.radians(20.0 + 30.0 * ease + 25.0 * np.sin(2 * np.pi * 2.0 * phase) * ease)
        upper = np.array([0.10 * np.cos(lift), side * np.cos(lift), np.sin(lift)])
        upper /= np.linalg.norm(upper)
        fore_angle = lift + bend
        fore = np.array([0.25 * np.cos(fore_angle), side * np.cos(fore_angle), np.sin(fore_angle)])
        fore /= np.linalg.norm(fore)
        elbow = shoulder + 0.30 * upper
        wrist = elbow + 0.27 * fore
        parts += [_limb(shoulder, elbow, 0.05, shirt), _limb(elbow, wrist, 0.042, skin),
                  _blob(wrist + 0.05 * fore, [0.05, 0.05, 0.05], skin)]
    # Whole-body turn and sway.
    yaw = np.radians(14.0) * np.sin(2 * np.pi * phase)
    cz, sz = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1.0]])
    shift = np.array([0.04 * np.sin(2 * np.pi * phase), 0.0, 0.0])
    for p in parts:
        p["center"] = Rz @ p["center"] + shift
        p["R"] = Rz @ p["R"]
    return parts


def render_subject(origin: np.ndarray, dirs: np.ndarray, parts: list[dict]):
    """Ray-cast the ellipsoids. Returns RGB [H, W, 3] and a hit mask [H, W]."""
    h, w, _ = dirs.shape
    d = dirs.reshape(-1, 3)
    best = np.full(len(d), np.inf, dtype=np.float32)
    rgb = np.zeros((len(d), 3), dtype=np.float32)
    for p in parts:
        R, radii = p["R"].astype(np.float32), p["radii"].astype(np.float32)
        o_l = ((origin - p["center"]) @ p["R"] / p["radii"]).astype(np.float32)
        d_l = (d @ R) / radii
        a = np.einsum("ij,ij->i", d_l, d_l)
        b = d_l @ o_l
        disc = b * b - a * (o_l @ o_l - 1.0)
        hit = disc >= 0
        s = np.where(hit, (-b - np.sqrt(np.maximum(disc, 0.0))) / a, np.inf)
        closer = hit & (s > 0) & (s < best)
        if not closer.any():
            continue
        local = o_l + s[closer, None] * d_l[closer]
        normal = (local / radii) @ R.T
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        diffuse = np.clip(normal @ LIGHT, 0.0, 1.0)
        stripes = 1.0 + 0.22 * np.sign(np.sin(local[:, 0] * 9.0) * np.sin(local[:, 2] * 7.0 + 1.0))
        shade = (0.42 + 0.58 * diffuse) * stripes
        rgb[closer] = np.asarray(p["color"], dtype=np.float32) * shade[:, None]
        best[closer] = s[closer]
    return rgb.reshape(h, w, 3), np.isfinite(best).reshape(h, w)


def render_board(origin: np.ndarray, dirs: np.ndarray, texture: np.ndarray,
                 R_b: np.ndarray, t_b: np.ndarray, size_m: tuple[float, float]):
    """Ray-cast a planar board. Board frame: x right, y down, z into the board.

    ``R_b``/``t_b`` map board coordinates to world. Returns grey values and
    coverage, both [H, W] float.
    """
    h, w, _ = dirs.shape
    normal = R_b[:, 2]
    denom = dirs.reshape(-1, 3) @ normal
    with np.errstate(divide="ignore", invalid="ignore"):
        s = ((t_b - origin) @ normal) / denom
    pts = origin + s[:, None] * dirs.reshape(-1, 3)
    local = (pts - t_b) @ R_b  # board coordinates
    ppm_x = texture.shape[1] / size_m[0]
    ppm_y = texture.shape[0] / size_m[1]
    map_x = (local[:, 0] * ppm_x - 0.5).astype(np.float32).reshape(h, w)
    map_y = (local[:, 1] * ppm_y - 0.5).astype(np.float32).reshape(h, w)
    inside = ((s > 0) & (local[:, 0] >= 0) & (local[:, 0] <= size_m[0])
              & (local[:, 1] >= 0) & (local[:, 1] <= size_m[1])).reshape(h, w)
    grey = cv2.remap(texture, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # The printed side faces -z; from behind the panel is plain cardboard.
    front = (denom > 0).reshape(h, w)
    grey = np.where(front, grey.astype(np.float32) / 255.0, 0.35)
    return grey, inside.astype(np.float32)


def board_pose(i: int, n: int, cfg: SynthConfig, spec: BoardSpec) -> tuple[np.ndarray, np.ndarray]:
    """Pose of the calibration board at calibration frame i (board -> world)."""
    half = np.radians(cfg.arc_degrees) / 2.0 + np.radians(8.0)
    u = i / max(n - 1, 1)
    az = half * np.sin(2 * np.pi * 1.5 * u)  # sweeps across the camera arc
    facing = np.array([np.cos(az), np.sin(az), 0.0])  # towards the cameras
    pitch = np.radians(22.0) * np.sin(2 * np.pi * 3.3 * u + 0.4)
    roll = np.radians(18.0) * np.sin(2 * np.pi * 2.1 * u + 1.1)
    z = -(facing * np.cos(pitch) + np.array([0, 0, 1.0]) * np.sin(pitch))
    x = np.cross(np.array([0, 0, 1.0]), -z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    cr, sr = np.cos(roll), np.sin(roll)
    x, y = cr * x + sr * y, -sr * x + cr * y
    R_b = np.stack([x, y, z], axis=1)
    reach = 0.2 + 1.2 * (0.5 + 0.5 * np.sin(2 * np.pi * 2.7 * u))
    side = np.cross(np.array([0, 0, 1.0]), facing)
    center = (facing * reach + side * 0.75 * np.sin(2 * np.pi * 4.3 * u)
              + np.array([0, 0, 1.1 + 0.5 * np.sin(2 * np.pi * 1.9 * u + 2.0)]))
    w, h = spec.size_m
    t_b = center - R_b @ np.array([w / 2, h / 2, 0.0])
    return R_b, t_b


def paint_board(cam: Camera, rays2, background: np.ndarray, texture: np.ndarray,
                R_b: np.ndarray, t_b: np.ndarray, spec: BoardSpec) -> np.ndarray:
    """Image of the calibration board in front of the background (2x supersampled)."""
    grey, cover = render_board(*rays2, texture, R_b, t_b, spec.size_m)
    size = (cam.width, cam.height)
    grey = cv2.resize(grey, size, interpolation=cv2.INTER_AREA)
    cover = cv2.resize(cover, size, interpolation=cv2.INTER_AREA)[..., None]
    return background * (1 - cover) + grey[..., None] * cover


def write_calibration_frames(out_dir: str | Path, cfg: SynthConfig, spec: BoardSpec | None = None) -> Rig:
    """Calibration images straight to ``<out>/<cam>/<frame>.png`` (no video). Returns the true rig."""
    out = Path(out_dir)
    rig = make_rig(cfg, distorted=True)
    spec = spec or BoardSpec()
    texture = spec.image(pixels_per_metre=1000)
    rng = np.random.default_rng(cfg.seed)
    for name, cam in rig.cameras.items():
        (out / name).mkdir(parents=True, exist_ok=True)
        background = render_background(*camera_rays(cam))
        rays2 = camera_rays(cam, supersample=2)
        for i in range(cfg.calib_frames):
            R_b, t_b = board_pose(i, cfg.calib_frames, cfg, spec)
            img = paint_board(cam, rays2, background, texture, R_b, t_b, spec)
            cv2.imwrite(str(out / name / f"{frame_name(i)}.png"),
                        cv2.cvtColor(_to_uint8(img, rng, cfg.noise), cv2.COLOR_RGB2BGR))
    return rig


def _to_uint8(img: np.ndarray, rng: np.random.Generator, noise: float) -> np.ndarray:
    out = img * 255.0
    if noise > 0:
        out = out + rng.normal(0.0, noise, size=img.shape).astype(np.float32)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def make_prepared_dataset(out_dir: str | Path, cfg: SynthConfig, init_points: int = 2000) -> Path:
    """Write a ready-to-train dataset (pinhole cameras, exact masks, ``init_points`` per frame)."""
    out = Path(out_dir)
    rig = make_rig(cfg, distorted=False)
    rng = np.random.default_rng(cfg.seed)
    frame_ids = list(range(cfg.take_frames))
    for name, cam in rig.cameras.items():
        origin, dirs = camera_rays(cam)
        background = render_background(origin, dirs)
        (out / "images" / name).mkdir(parents=True, exist_ok=True)
        (out / "masks" / name).mkdir(parents=True, exist_ok=True)
        (out / "background").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "background" / f"{name}.jpg"),
                    cv2.cvtColor(_to_uint8(background, rng, 0.0), cv2.COLOR_RGB2BGR))
        for fid in frame_ids:
            phase = fid / max(cfg.take_frames - 1, 1)
            rgb, hit = render_subject(origin, dirs, mannequin(phase))
            img = np.where(hit[..., None], rgb, background)
            cv2.imwrite(str(out / "images" / name / f"{frame_name(fid)}.png"),
                        cv2.cvtColor(_to_uint8(img, rng, cfg.noise), cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out / "masks" / name / f"{frame_name(fid)}.png"),
                        hit.astype(np.uint8) * 255)
    rig.save(out / "calibration.json")
    write_meta(out, cfg.fps, frame_ids, rig.names, {"source": "synthetic"})
    if init_points > 0:  # points on the true surface of every frame: a stand-in for the visual hull
        clouds = [surface_points(mannequin(f / max(cfg.take_frames - 1, 1)), init_points, rng) for f in frame_ids]
        write_points_ply(out / "init_points.ply", np.concatenate([c[0] for c in clouds]),
                         np.concatenate([c[1] for c in clouds]), np.repeat(frame_ids, len(clouds[0][0])))
    return out


def surface_points(parts: list[dict], n: int, rng: np.random.Generator):
    """Random points on the ellipsoid surfaces (a stand-in for a visual hull)."""
    per = max(n // len(parts), 1)
    xyz, rgb = [], []
    for p in parts:
        v = rng.normal(size=(per, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        xyz.append((v * p["radii"]) @ p["R"].T + p["center"])
        rgb.append(np.tile(np.asarray(p["color"]), (per, 1)))
    return np.concatenate(xyz).astype(np.float32), np.concatenate(rgb).astype(np.float32)


def _audio_track(cfg: SynthConfig, total_frames: int, rng: np.random.Generator) -> np.ndarray:
    """Room sound shared by all cameras: ambience plus one clap."""
    n = int((total_frames / cfg.fps + 0.5) * AUDIO_RATE)
    ambience = np.convolve(rng.normal(0, 0.02, n), np.ones(24) / 24, mode="same")
    start = int(cfg.event_frame / cfg.fps * AUDIO_RATE)
    length = int(0.08 * AUDIO_RATE)
    clap = rng.normal(0, 1.0, length) * np.exp(-np.arange(length) / (0.012 * AUDIO_RATE))
    ambience[start:start + length] += 0.8 * clap
    return ambience.astype(np.float32)


def _write_wav(path: Path, samples: np.ndarray) -> None:
    from scipy.io import wavfile

    wavfile.write(str(path), AUDIO_RATE, np.clip(samples * 32767, -32768, 32767).astype(np.int16))


def make_raw_session(out_dir: str | Path, cfg: SynthConfig, log=print) -> Path:
    """Write videos as they would come out of a real (imperfect) capture session."""
    out = Path(out_dir)
    rig = make_rig(cfg, distorted=True)
    rng = np.random.default_rng(cfg.seed)
    spec = BoardSpec()
    texture = spec.image(pixels_per_metre=1000)
    offsets = {name: int(rng.integers(0, cfg.max_offset + 1)) for name in rig.names}
    offsets[rig.names[0]] = 0
    (out / "gt").mkdir(parents=True, exist_ok=True)
    rig.save(out / "gt" / "calibration.json")
    (out / "gt" / "offsets.json").write_text(json.dumps(
        {"start_offset_frames": offsets, "event_frame": cfg.event_frame, "fps": cfg.fps}, indent=2))

    rays = {n: camera_rays(c) for n, c in rig.cameras.items()}
    rays2 = {n: camera_rays(c, supersample=2) for n, c in rig.cameras.items()}
    backgrounds = {n: render_background(*rays[n]) for n in rig.names}
    tmp = out / "tmp_audio"
    tmp.mkdir(exist_ok=True)

    def clip(kind: str, frames: int, synced: bool, painter):
        (out / "raw" / kind).mkdir(parents=True, exist_ok=True)
        audio = _audio_track(cfg, frames + cfg.max_offset, rng) if synced else None
        for name in rig.names:
            off = offsets[name] if synced else 0
            wav = None
            if audio is not None:
                a0 = int(off / cfg.fps * AUDIO_RATE)
                local = audio[a0:a0 + int((frames / cfg.fps + 0.2) * AUDIO_RATE)]
                wav = tmp / f"{kind}_{name}.wav"
                _write_wav(wav, local + rng.normal(0, 0.004, len(local)).astype(np.float32))
            with VideoWriter(out / "raw" / kind / f"{name}.mp4", cfg.width, cfg.height,
                             cfg.fps, audio=wav) as writer:
                for local_frame in range(frames):
                    g = local_frame + off  # global frame index
                    img = painter(name, g)
                    if synced and g == cfg.event_frame:
                        img = np.clip(img + 0.35, 0, 1)  # flash
                    writer.write(_to_uint8(img, rng, cfg.noise))
            log(f"[demo] wrote raw/{kind}/{name}.mp4")

    def paint_empty(name, g):
        return backgrounds[name]

    def paint_take(name, g):
        first = cfg.event_frame + 3  # the action starts after the sync event
        phase = float(np.clip((g - first) / max(cfg.take_frames - 1 - first, 1), 0, 1))
        rgb, hit = render_subject(*rays[name], mannequin(phase))
        return np.where(hit[..., None], rgb, backgrounds[name])

    def paint_calib(name, g):
        R_b, t_b = board_pose(g, cfg.calib_frames + cfg.max_offset, cfg, spec)
        return paint_board(rig.cameras[name], rays2[name], backgrounds[name], texture, R_b, t_b, spec)

    clip("empty", cfg.empty_frames, False, paint_empty)
    clip("calib", cfg.calib_frames, True, paint_calib)
    clip("take", cfg.take_frames, True, paint_take)
    for f in tmp.glob("*.wav"):
        f.unlink()
    tmp.rmdir()
    return out
