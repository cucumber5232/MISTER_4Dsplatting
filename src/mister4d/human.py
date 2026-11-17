"""A clothed, animated human for the synthetic capture generator.

The mannequin of ``synth.py`` is enough to exercise the pipeline, not to judge
whether a reconstruction looks real. This module poses a textured human mesh
(linear blend skinning on the MakeHuman skeleton) and renders it with a small
software rasterizer: colour, coverage and depth for any camera of the project,
lens distortion included, on a studio set where the subject casts a soft
shadow on the floor.

The mesh is built from MakeHuman data released as CC0 by
``scripts/build_human_asset.py`` (see ``assets/HUMAN.md``). Nothing here is
used on real footage.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera, rodrigues

ASSET = Path(__file__).parent / "assets" / "human.npz"
MOTION_SECONDS = 2.0  # the action lasts this long, then plays backwards and so on

# Per material, in the order used by the asset: how far diffuse light wraps past the
# terminator (red, green, blue), strength and exponent of the highlight.
MATERIALS = {
    "skin": ((0.42, 0.26, 0.20), 0.05, 24.0),
    "top": ((0.20, 0.20, 0.20), 0.0, 1.0),
    "trousers": ((0.15, 0.15, 0.15), 0.0, 1.0),
    "shoes": ((0.10, 0.10, 0.10), 0.04, 30.0),
    "hair": ((0.25, 0.25, 0.25), 0.10, 14.0),
    "eyes": ((0.30, 0.30, 0.30), 0.60, 160.0),
}


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


# Studio lights: direction towards the light and colour.
KEY_DIR, KEY_RGB = _unit([0.56, 0.48, 0.67]), np.array([0.56, 0.535, 0.50], dtype=np.float32)
FILL_DIR, FILL_RGB = _unit([0.55, -0.70, 0.45]), np.array([0.235, 0.25, 0.275], dtype=np.float32)
BACK_DIR, BACK_RGB = _unit([-0.74, 0.60, 0.30]), np.array([0.25, 0.24, 0.225], dtype=np.float32)
SKY_RGB = np.array([0.36, 0.37, 0.39], dtype=np.float32)  # ambient light from above ...
GROUND_RGB = np.array([0.20, 0.195, 0.185], dtype=np.float32)  # ... and bounced off the floor
KEY_SIZE = 0.16  # tangent of the angular radius of the key light: sets the penumbra
WALL_RADIUS = 6.0


# ---------------------------------------------------------------------------
# Asset
# ---------------------------------------------------------------------------


@dataclass
class Human:
    verts: np.ndarray  # [V, 3] rest pose, metres, +Z up, facing +X, standing on z = 0
    tris: np.ndarray  # [2Q, 3]; triangles 2q and 2q + 1 are the halves of quad q
    material: np.ndarray  # [Q] index into MATERIALS
    block: np.ndarray  # [Q] texture block of the quad inside the atlas of its material
    ao: np.ndarray  # [V] baked ambient occlusion, 0..1
    bones: list[str]
    parents: np.ndarray  # [B], -1 for the root; parents come first
    heads: np.ndarray  # [B, 3] joint positions at rest
    skin_bones: np.ndarray  # [V, 4]
    skin_weights: np.ndarray  # [V, 4]
    textures: list  # per material: list of mip levels (atlas [h, w, 3] float32, block size)
    texel: np.ndarray  # [M] size of a texel of the finest level, metres

    def __post_init__(self):
        self.index = {name: i for i, name in enumerate(self.bones)}

    def joint(self, name: str) -> np.ndarray:
        return self.heads[self.index[name]].astype(np.float64)


def _mip(atlas: np.ndarray, n: int) -> np.ndarray:
    """Halve the resolution of every n x n block; block edges keep their own texels."""
    h, w, _ = atlas.shape
    m = (n - 1) // 2 + 1
    b = atlas.reshape(h // n, n, w // n, n, 3)
    for axis in (1, 3):
        b = np.moveaxis(b, axis, 0)
        out = b[::2].copy()
        out[1:-1] = 0.25 * b[1:-2:2] + 0.5 * b[2:-1:2] + 0.25 * b[3::2]
        b = np.moveaxis(out, 0, axis)
    return np.ascontiguousarray(b).reshape(h // n * m, w // n * m, 3)


@functools.lru_cache(maxsize=1)
def load_human() -> Human:
    if not ASSET.exists():
        raise FileNotFoundError(f"{ASSET} is missing: build it with scripts/build_human_asset.py")
    data = np.load(ASSET, allow_pickle=False)
    quads = data["quads"].astype(np.int32)
    tris = np.stack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=1).reshape(-1, 3)
    material = data["material"].astype(np.int32)
    block = np.zeros(len(quads), dtype=np.int32)
    textures = []
    for m in range(len(MATERIALS)):
        sel = material == m
        block[sel] = np.arange(int(sel.sum()))
        atlas = cv2.imdecode(data[f"texture{m}"], cv2.IMREAD_COLOR)
        atlas = cv2.cvtColor(atlas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        n = int(data["texture_block"][m])
        levels = [(atlas, n)]
        while n > 2:
            atlas, n = _mip(atlas, n), (n - 1) // 2 + 1
            levels.append((atlas, n))
        textures.append(levels)
    weights = data["skin_weights"].astype(np.float32)
    return Human(verts=data["verts"].astype(np.float32), tris=tris, material=material, block=block,
                 ao=data["ao"].astype(np.float32) / 255.0, bones=[str(b) for b in data["bones"]],
                 parents=data["parents"].astype(np.int32), heads=data["heads"].astype(np.float32),
                 skin_bones=data["skin_bones"].astype(np.int32),
                 skin_weights=weights / weights.sum(axis=1, keepdims=True),
                 textures=textures, texel=data["texel"].astype(np.float32))


# ---------------------------------------------------------------------------
# Pose and motion
# ---------------------------------------------------------------------------


def _rot(axis, degrees: float) -> np.ndarray:
    return rodrigues(_unit(axis) * np.radians(degrees))


def _between(a, b, fraction: float = 1.0) -> np.ndarray:
    """Smallest rotation taking direction ``a`` to ``b`` (or a fraction of it)."""
    a, b = _unit(a), _unit(b)
    axis = np.cross(a, b)
    s = np.linalg.norm(axis)
    if s < 1e-9:
        return np.eye(3)
    return rodrigues(axis / s * np.arctan2(s, a @ b) * fraction)


def _smooth(t: float, t0: float, t1: float) -> float:
    """0 before t0, 1 after t1, eased in between."""
    u = float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


class _Rig:
    """Rotations of the bones about their joints, in the axes of the rest pose."""

    def __init__(self, human: Human):
        self.h = human
        self.local: dict[str, np.ndarray] = {}
        self.pivot = human.joint("spine05")  # the pelvis turns and shifts about this point
        self.shift = np.zeros(3)

    def frame(self, bone: str) -> tuple[np.ndarray, np.ndarray]:
        """Rest -> posed transform of a bone: x' = A @ x + c."""
        h = self.h
        chain = []
        i = h.index[bone]
        while i >= 0:
            chain.append(i)
            i = h.parents[i]
        A = self.local.get(h.bones[chain[-1]], np.eye(3))
        c = self.pivot - A @ self.pivot + self.shift
        for i in reversed(chain[:-1]):
            R = self.local.get(h.bones[i])
            if R is not None:
                head = h.heads[i].astype(np.float64)
                c = A @ (head - R @ head) + c
                A = A @ R
        return A, c

    def point(self, bone: str, rest_point) -> np.ndarray:
        A, c = self.frame(bone)
        return A @ np.asarray(rest_point, dtype=np.float64) + c

    def aim(self, bone: str, parent: str, rest_dir, world_dir, twist: float = 0.0) -> None:
        """Turn ``bone`` so that ``rest_dir`` points along ``world_dir``."""
        A, _ = self.frame(parent)
        R = _between(rest_dir, A.T @ _unit(world_dir))
        self.local[bone] = R @ _rot(rest_dir, twist) if twist else R

    def transforms(self) -> np.ndarray:
        """[B, 3, 4] for every bone."""
        h = self.h
        out = np.zeros((len(h.bones), 3, 4))
        for i, name in enumerate(h.bones):
            R = self.local.get(name)
            if h.parents[i] < 0:
                A = R if R is not None else np.eye(3)
                c = self.pivot - A @ self.pivot + self.shift
            else:
                A, c = out[h.parents[i], :, :3], out[h.parents[i], :, 3]
                if R is not None:
                    head = h.heads[i].astype(np.float64)
                    c = A @ (head - R @ head) + c
                    A = A @ R
            out[i, :, :3], out[i, :, 3] = A, c
        return out


def _arm(rig: _Rig, side: str, upper, fore, twist: float = 0.0, lift: float = 0.0, hand=None) -> None:
    """Aim upper arm and forearm (directions in the frame of the chest)."""
    h = rig.h
    chest, _ = rig.frame("spine01")
    sign = 1.0 if side == "L" else -1.0
    if lift:
        rig.local[f"clavicle.{side}"] = _rot([1, 0, 0], sign * lift)
    shoulder, elbow, wrist = (h.joint(f"{b}.{side}") for b in ("upperarm01", "lowerarm01", "wrist"))
    A, _ = rig.frame(f"clavicle.{side}")
    full = _between(elbow - shoulder, A.T @ chest @ _unit(upper), 1.0)
    part = _between(elbow - shoulder, A.T @ chest @ _unit(upper), 0.3)
    rig.local[f"shoulder01.{side}"] = part  # the deltoid follows the arm part of the way
    rig.local[f"upperarm01.{side}"] = part.T @ full
    rig.aim(f"lowerarm01.{side}", f"upperarm02.{side}", wrist - elbow, chest @ _unit(fore))
    if twist:
        rig.local[f"lowerarm02.{side}"] = _rot(wrist - elbow, sign * twist)
    if hand is not None:
        rig.local[f"wrist.{side}"] = hand


def _fingers(rig: _Rig, side: str, curl: float) -> None:
    """Close the four fingers by ``curl`` degrees per joint (about the line of the knuckles)."""
    h = rig.h
    knuckles = h.joint(f"finger5-1.{side}") - h.joint(f"finger2-1.{side}")
    along = h.joint(f"finger3-2.{side}") - h.joint(f"finger3-1.{side}")
    palm = np.cross(knuckles, along)
    thumb = h.joint(f"finger1-3.{side}") - h.joint(f"finger3-1.{side}")
    sign = 1.0 if palm @ thumb > 0 else -1.0  # the thumb is on the side of the palm
    for f in (2, 3, 4, 5):
        for j, amount in ((1, 0.7), (2, 1.0), (3, 0.6)):
            rig.local[f"finger{f}-{j}.{side}"] = _rot(knuckles, sign * curl * amount * (0.8 + 0.07 * f))
    rig.local[f"finger1-2.{side}"] = _rot(knuckles, sign * curl * 0.3)


def _leg(rig: _Rig, side: str, ankle, yaw: float = 0.0, pitch: float = 0.0) -> None:
    """Two-bone inverse kinematics: put the ankle at ``ankle`` with the foot flat (or pitched)."""
    h = rig.h
    hip0, knee0, ankle0 = (h.joint(f"{b}.{side}") for b in ("upperleg01", "lowerleg01", "foot"))
    A, c = rig.frame(f"pelvis.{side}")
    hip = A @ hip0 + c
    l1, l2 = np.linalg.norm(knee0 - hip0), np.linalg.norm(ankle0 - knee0)
    d = np.asarray(ankle, dtype=np.float64) - hip
    dist = min(np.linalg.norm(d), (l1 + l2) * 0.9995)
    d = _unit(d)
    foot = _rot([0, 0, 1], yaw) @ _rot([0, 1, 0], pitch)
    forward = foot @ np.array([1.0, 0.0, 0.0])
    pole = _unit(forward - (forward @ d) * d)
    bend = np.arccos(np.clip((l1 * l1 + dist * dist - l2 * l2) / (2 * l1 * dist), -1.0, 1.0))
    knee = hip + l1 * (np.cos(bend) * d + np.sin(bend) * pole)
    rig.aim(f"upperleg01.{side}", f"pelvis.{side}", knee0 - hip0, knee - hip)
    rig.aim(f"lowerleg01.{side}", f"upperleg02.{side}", ankle0 - knee0, hip + dist * d - knee)
    A, _ = rig.frame(f"lowerleg02.{side}")
    rig.local[f"foot.{side}"] = A.T @ foot


def human_pose(human: Human, seconds: float) -> np.ndarray:
    """Bone transforms [B, 3, 4] of the action at a given time.

    Two seconds: the weight goes onto the left leg, the right arm comes up
    fast and waves, the head turns, the right foot takes a small step. Longer
    takes play the action back and forth.
    """
    t = seconds % (2.0 * MOTION_SECONDS)
    t = t if t <= MOTION_SECONDS else 2.0 * MOTION_SECONDS - t
    rig = _Rig(human)
    ankle_z = human.joint("foot.L")[2]

    shift = _smooth(t, 0.0, 0.55)  # weight onto the left leg
    step = _smooth(t, 1.15, 1.60)  # the right foot travels
    settle = _smooth(t, 1.45, 2.0)  # weight back between the feet
    raise_ = _smooth(t, 0.08, 0.70)
    lower = _smooth(t, 1.35, 2.0)
    wave = np.sin(2.0 * np.pi * 2.3 * (t - 0.62)) * _smooth(t, 0.58, 0.74) * (1.0 - _smooth(t, 1.25, 1.50))

    sway = 0.045 * shift - 0.035 * settle
    rig.shift = np.array([0.012 * shift + 0.075 * settle, sway,
                          -0.012 - 0.010 * shift + 0.006 * settle])
    rig.local["root"] = _rot([0, 0, 1], 3.0 * shift + 4.0 * settle) @ _rot([1, 0, 0], -2.5 * shift + 2.0 * settle)
    for i, bone in enumerate(("spine05", "spine04", "spine03", "spine02", "spine01")):
        rig.local[bone] = (_rot([1, 0, 0], 1.0 * shift - 0.9 * settle + 0.6 * raise_ * (1 - lower))
                           @ _rot([0, 0, 1], -1.2 * raise_ * (1 - lower) - 0.6 * settle)
                           @ _rot([0, 1, 0], 0.4 + 0.3 * settle))
    turn = -24.0 * _smooth(t, 0.25, 0.80) + 34.0 * _smooth(t, 1.20, 1.90)
    nod = 4.0 * _smooth(t, 0.25, 0.80) - 6.0 * _smooth(t, 1.20, 1.90)
    for bone, part in (("neck01", 0.15), ("neck02", 0.15), ("neck03", 0.15), ("head", 0.55)):
        rig.local[bone] = _rot([0, 0, 1], turn * part) @ _rot([0, 1, 0], nod * part)

    # Left arm hangs and swings a little; right arm is raised, waves, comes half down.
    swing = 4.0 * np.sin(np.pi * t) * (1.0 - 0.5 * settle)
    _arm(rig, "L", _rot([0, 1, 0], swing) @ np.array([0.03, 0.20, -1.0]),
         _rot([0, 1, 0], swing * 1.5) @ np.array([0.22, 0.10, -1.0]), twist=10.0)
    _fingers(rig, "L", 26.0)
    up = raise_ * (1.0 - 0.72 * lower)
    upper = _rot([1, 0, 0], -97.0 * up) @ _rot([0, 1, 0], -14.0 * up) @ np.array([0.03, -0.20, -1.0])
    fore_rest = np.array([0.22, -0.10, -1.0])
    tilt = 8.0 + 30.0 * wave
    fore_up = _rot([1, 0, 0], tilt) @ np.array([0.16, 0.0, 1.0])
    fore = _unit(_unit(fore_rest) * (1.0 - up) + fore_up * up + np.array([0.5, 0.0, 0.0]) * up * (1.0 - up))
    flick = _rot([1, 0, 0], 14.0 * np.sin(2.0 * np.pi * 2.3 * (t - 0.68)) * abs(wave))
    _arm(rig, "R", upper, fore, twist=10.0 - 70.0 * up, lift=16.0 * up, hand=flick)
    _fingers(rig, "R", 26.0 - 20.0 * up)

    # Feet: planted, then the right one steps forward.
    lift = np.sin(np.pi * step) ** 2
    _leg(rig, "L", [0.0, 0.105, ankle_z], yaw=8.0)
    _leg(rig, "R", [0.20 * step, -0.105 - 0.03 * step, ankle_z + 0.055 * lift], yaw=-8.0 - 6.0 * step,
         pitch=14.0 * lift * (1.0 - step) - 9.0 * lift * step)
    return rig.transforms()


def pose_vertices(human: Human, transforms: np.ndarray) -> np.ndarray:
    """Linear blend skinning: posed vertices [V, 3]."""
    m = (transforms[human.skin_bones] * human.skin_weights[:, :, None, None]).sum(axis=1)  # [V, 3, 4]
    return (np.einsum("vij,vj->vi", m[:, :, :3], human.verts) + m[:, :, 3]).astype(np.float32)


def vertex_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    a, b, c = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    face = np.cross(b - a, c - a)
    out = np.zeros_like(verts)
    for k in range(3):
        for axis in range(3):
            out[:, axis] += np.bincount(tris[:, k], weights=face[:, axis], minlength=len(verts))
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------------------
# Rasterizer
# ---------------------------------------------------------------------------


def rasterize(px: np.ndarray, key: np.ndarray, tris: np.ndarray, width: int, height: int):
    """Z-buffer on a regular grid of samples.

    ``px`` [V, 2]: vertex positions in grid units (sample (row i, column j) is at
    x = j, y = i). ``key`` [V]: a quantity that is linear on screen and larger
    when closer (1 / depth for a perspective view). Returns, for the covered
    samples only: flat sample index, triangle, weights of its three corners
    (affine on screen) [N, 3].
    """
    p = px[tris].astype(np.float32)
    x, y = p[..., 0], p[..., 1]
    area = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    x0 = np.clip(np.ceil(x.min(axis=1)), 0, width).astype(np.int32)
    x1 = np.clip(np.floor(x.max(axis=1)), -1, width - 1).astype(np.int32)
    y0 = np.clip(np.ceil(y.min(axis=1)), 0, height).astype(np.int32)
    y1 = np.clip(np.floor(y.max(axis=1)), -1, height - 1).astype(np.int32)
    keep = np.nonzero((x1 >= x0) & (y1 >= y0) & (np.abs(area) > 1e-10))[0]
    nx = x1[keep] - x0[keep] + 1
    count = nx * (y1[keep] - y0[keep] + 1)
    first = np.cumsum(count) - count
    which = np.repeat(np.arange(len(keep), dtype=np.int32), count)
    k = np.arange(int(count.sum()), dtype=np.int32) - first[which].astype(np.int32)
    t = keep[which]
    nxw = nx[which]
    xi = x0[t] + k % nxw
    yi = y0[t] + k // nxw
    xf, yf = xi.astype(np.float32), yi.astype(np.float32)
    xa, ya, xb, yb, xc, yc = x[t, 0], y[t, 0], x[t, 1], y[t, 1], x[t, 2], y[t, 2]
    inv = 1.0 / area[t]
    w0 = ((xb - xf) * (yc - yf) - (xc - xf) * (yb - yf)) * inv
    w1 = ((xc - xf) * (ya - yf) - (xa - xf) * (yc - yf)) * inv
    w2 = 1.0 - w0 - w1
    inside = np.nonzero((w0 >= 0) & (w1 >= 0) & (w2 >= 0))[0]
    t = t[inside]
    w = np.stack([w0[inside], w1[inside], w2[inside]], axis=1)
    pid = yi[inside] * width + xi[inside]
    depth_key = (w * key[tris[t]].astype(np.float32)).sum(axis=1)
    order = np.argsort(depth_key, kind="stable")  # far to near: the nearest is written last
    winner = np.full(width * height, -1, dtype=np.int32)
    winner[pid[order]] = order
    sample = np.nonzero(winner >= 0)[0]
    slot = winner[sample]
    return sample, t[slot], w[slot]


# ---------------------------------------------------------------------------
# Light: shadow of the subject on itself and on the floor
# ---------------------------------------------------------------------------

SHADOW_CELL = 0.004  # metres per cell of the shadow maps


class _Shadow:
    """What a distant light sees of the subject: who is in front, and the shade on the floor."""

    def __init__(self, verts: np.ndarray, tris: np.ndarray, light: np.ndarray, size: float):
        light = _unit(light)
        self.light = light
        e1 = _unit(np.cross([0.0, 0.0, 1.0], light)) if abs(light[2]) < 0.999 else np.array([1.0, 0.0, 0.0])
        e2 = np.cross(light, e1)
        self.axes = np.stack([e1, e2]).astype(np.float32)
        uv = verts @ self.axes.T
        depth = -(verts @ light.astype(np.float32))
        margin = 48
        self.origin = uv.min(axis=0) - margin * SHADOW_CELL
        w, h = (np.ceil((uv.max(axis=0) - self.origin) / SHADOW_CELL).astype(int) + margin)
        sample, t, wts = rasterize((uv - self.origin) / SHADOW_CELL, -depth, tris, int(w), int(h))
        self.depth = np.full(w * h, np.inf, dtype=np.float32)
        self.depth[sample] = (wts * depth[tris[t]]).sum(axis=1)
        self.depth = self.depth.reshape(h, w)
        # Shade on the floor: the outline of the subject, blurred more where the part that
        # casts it is higher above the floor.
        cover = np.zeros(w * h, dtype=np.float32)
        cover[sample] = 1.0
        above = np.zeros(w * h, dtype=np.float32)
        above[sample] = (wts * verts[:, 2][tris[t]]).sum(axis=1) / max(light[2], 0.2)
        half = (w // 2, h // 2)  # the shade is soft: half the resolution is plenty
        cover = cv2.resize(cover.reshape(h, w), half, interpolation=cv2.INTER_AREA)
        above = cv2.resize(above.reshape(h, w), half, interpolation=cv2.INTER_AREA)
        spread = cv2.GaussianBlur(above, (0, 0), 8) / np.maximum(cv2.GaussianBlur(cover, (0, 0), 8), 1e-4)
        far = cv2.GaussianBlur(above, (0, 0), 24) / np.maximum(cv2.GaussianBlur(cover, (0, 0), 24), 1e-4)
        near = cv2.GaussianBlur(cover, (0, 0), 8) > 0.02
        radius = np.where(near, spread, far) * size / (2.0 * SHADOW_CELL) * 0.5  # sigma of the blur, cells
        sigmas = [0.6, 2.0, 4.0, 8.0, 16.0, 32.0]
        stack = np.stack([cv2.GaussianBlur(cover, (0, 0), s) for s in sigmas])
        level = np.clip(np.log2(np.maximum(radius, 1e-3) / 2.0) + 1.0, 0.0, len(sigmas) - 1.001)
        lo = np.floor(level).astype(np.int32)
        frac = level - lo
        rows, cols = np.indices(cover.shape)
        self.floor = stack[lo, rows, cols] * (1.0 - frac) + stack[lo + 1, rows, cols] * frac

    def lit(self, points: np.ndarray, normals: np.ndarray) -> np.ndarray:
        """Fraction of the light reaching surface points (soft edges from four samples)."""
        uv = (points @ self.axes.T - self.origin) / SHADOW_CELL
        depth = -(points @ self.light.astype(np.float32))
        facing = np.clip(normals @ self.light.astype(np.float32), 0.05, 1.0)
        bias = 0.012 + np.minimum(SHADOW_CELL * np.sqrt(1.0 - facing * facing) / facing * 1.5, 0.03)
        h, w = self.depth.shape
        x0 = np.clip(np.floor(uv[:, 0] - 0.5).astype(np.int32), 0, w - 3)
        y0 = np.clip(np.floor(uv[:, 1] - 0.5).astype(np.int32), 0, h - 3)
        fx = np.clip(uv[:, 0] - 0.5 - x0, 0.0, 1.0)
        fy = np.clip(uv[:, 1] - 0.5 - y0, 0.0, 1.0)
        out = np.zeros(len(points), dtype=np.float32)
        for dy, wy in ((0, 1.0 - fy), (1, 1.0), (2, fy)):
            for dx, wx in ((0, 1.0 - fx), (1, 1.0), (2, fx)):
                out += wy * wx * (depth <= self.depth[y0 + dy, x0 + dx] + bias)
        return out * 0.25

    def on_floor(self, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
        """Shade (0..1) at points of the floor, given as two maps of the same shape."""
        u = (fx * self.axes[0, 0] + fy * self.axes[0, 1] - self.origin[0]) / (2.0 * SHADOW_CELL) - 0.25
        v = (fx * self.axes[1, 0] + fy * self.axes[1, 1] - self.origin[1]) / (2.0 * SHADOW_CELL) - 0.25
        return cv2.remap(self.floor, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)


@dataclass
class Instant:
    """The posed subject at one instant: shared by all the cameras."""

    verts: np.ndarray
    normals: np.ndarray
    key: _Shadow
    overhead: _Shadow


@functools.lru_cache(maxsize=96)
def human_at(seconds: float) -> Instant:
    human = load_human()
    verts = pose_vertices(human, human_pose(human, seconds))
    return Instant(verts, vertex_normals(verts, human.tris).astype(np.float32),
                   _Shadow(verts, human.tris, KEY_DIR, KEY_SIZE),
                   _Shadow(verts, human.tris, np.array([0.0, 0.0, 1.0]), 0.55))


# ---------------------------------------------------------------------------
# The set
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _grain() -> np.ndarray:
    """Tileable noise, three scales in the three channels, values around 0."""
    rng = np.random.default_rng(7)
    white = rng.normal(size=(512, 512)).astype(np.float32)
    wide = np.pad(white, 64, mode="wrap")  # blur across the seam so that the tile repeats
    out = [cv2.GaussianBlur(wide, (0, 0), s)[64:-64, 64:-64] for s in (1.2, 5.0, 20.0)]
    return np.stack([o / o.std() for o in out], axis=-1)


def _noise(x: np.ndarray, y: np.ndarray, scale: float) -> np.ndarray:
    """Three channels of noise at 2D positions; ``scale`` = metres covered by the tile."""
    g = _grain()
    u = (np.mod(x / scale, 1.0) * 512.0).astype(np.float32)
    v = (np.mod(y / scale, 1.0) * 512.0).astype(np.float32)
    return cv2.remap(g, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _stand() -> list[tuple[np.ndarray, np.ndarray, np.ndarray, tuple, float]]:
    """A lamp on a tripod behind the subject, as ellipsoids: centre, axes, radii, colour, glow."""
    base = np.array([-2.5, 2.35, 0.0])
    parts = []

    def rod(a, b, radius, color):
        a, b = base + np.asarray(a, float), base + np.asarray(b, float)
        x = _unit(b - a)
        y = _unit(np.cross([0.3, 0.2, 1.0], x))
        parts.append(((a + b) / 2, np.stack([x, y, np.cross(x, y)], axis=1),
                      np.array([np.linalg.norm(b - a) / 2 * 1.15, radius, radius]), color, 0.0))

    black = (0.05, 0.05, 0.055)
    rod([0, 0, 0.45], [0, 0, 2.05], 0.018, black)
    for k in range(3):
        a = np.radians(90.0 + 120.0 * k)
        rod([0, 0, 0.62], [0.42 * np.cos(a), 0.42 * np.sin(a), 0.0], 0.012, black)
    parts.append((base + [0.0, 0.0, 2.27], np.eye(3), np.array([0.26, 0.26, 0.24]), (1.0, 0.97, 0.90), 1.0))
    return parts


def studio_set(origin: np.ndarray, dirs: np.ndarray) -> dict:
    """The empty set along rays: matte grey plank floor, grey wall with a skirting board, a lamp.

    Returns ``rgb`` [H, W, 3], the points where the rays meet the floor
    (``fx``, ``fy``) and what the key light and the ambient light contribute
    there (``key``, ``ambient``): the shadow of the subject removes part of them.
    """
    o = origin.astype(np.float32)
    dx, dy, dz = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    a = dx * dx + dy * dy + 1e-12
    b = o[0] * dx + o[1] * dy
    c = o[0] ** 2 + o[1] ** 2 - WALL_RADIUS ** 2
    s_wall = (-b + np.sqrt(np.maximum(b * b - a * c, 0.0))) / a
    with np.errstate(divide="ignore", invalid="ignore"):
        s_floor = np.where(dz < -1e-6, -o[2] / dz, np.inf).astype(np.float32)
    floor = s_floor < s_wall
    s = np.where(floor, s_floor, s_wall)
    fx, fy, fz = o[0] + s * dx, o[1] + s * dy, o[2] + s * dz

    # Floor: planks along x, each with its own tint, and wood grain.
    width, length = 0.16, 1.4
    row = np.floor(fy / width)
    along = fx / length + np.modf(row * 0.37)[0] + 5.0
    board = row * 13.0 + np.floor(along)
    tint = np.modf(np.sin(board * 12.9898) * 43758.5453)[0]
    across, lengthwise = np.modf(fy / width + 100.0)[0], np.modf(along)[0]
    gap = np.minimum(np.minimum(across, 1.0 - across) * width,
                     np.minimum(lengthwise, 1.0 - lengthwise) * length)  # distance from the plank edge
    grain = _noise(fx * 0.12, fy, 1.7)
    cloud = _noise(fx, fy, 4.3)
    shade = (1.0 + 0.055 * tint + 0.02 * grain[..., 0] + 0.02 * grain[..., 1] + 0.02 * cloud[..., 2])
    shade = shade * (1.0 - 0.30 * np.exp(-(gap / 0.0035) ** 2))
    wood = np.stack([0.60 * shade, 0.575 * shade, 0.545 * shade], axis=-1)
    pool = 1.0 / (1.0 + (fx * fx + fy * fy) / 60.0)  # the lights are aimed at the middle
    key = wood * (KEY_RGB * float(KEY_DIR[2])) * pool[..., None]
    ambient = wood * SKY_RGB * 0.80
    lit_floor = key + ambient + wood * (FILL_RGB * float(FILL_DIR[2]) + BACK_RGB * float(BACK_DIR[2])) * pool[..., None]

    # Wall: matte grey, brighter where the lights spill, darker towards the ceiling.
    ang = np.arctan2(fy, fx)
    paper = _noise(ang * WALL_RADIUS, fz, 5.1)
    wall = (0.60 - 0.035 * fz + 0.05 * np.cos(ang - 2.6) + 0.012 * paper[..., 2] + 0.006 * paper[..., 1])
    wall = np.stack([wall * 0.97, wall, wall * 1.04], axis=-1)
    board_top = 0.11
    skirting = np.array([0.80, 0.79, 0.76], dtype=np.float32) * (0.92 + 0.08 * np.clip(fz / board_top, 0, 1))[..., None]
    wall = np.where((fz < board_top)[..., None], skirting, wall)
    wall *= (1.0 - 0.22 * np.exp(-np.maximum(fz - board_top, 0.0) / 0.05) * (fz >= board_top))[..., None]
    corner = np.exp(-np.maximum(WALL_RADIUS - np.hypot(fx, fy), 0.0) / 0.12)  # the floor is darker along the wall
    rgb = np.where(floor[..., None], lit_floor * (1.0 - 0.25 * corner[..., None]), wall)

    # The lamp and its tripod.
    d = dirs.reshape(-1, 3)
    out = rgb.reshape(-1, 3)
    nearest = s.reshape(-1).copy()
    hidden = np.zeros(len(d), dtype=bool)
    for center, R, radii, color, glow in _stand():
        R32, r32 = R.astype(np.float32), radii.astype(np.float32)
        o_l = ((origin - center) @ R / radii).astype(np.float32)
        d_l = (d @ R32) / r32
        qa = np.einsum("ij,ij->i", d_l, d_l)
        qb = d_l @ o_l
        disc = qb * qb - qa * (o_l @ o_l - 1.0)
        t_hit = (-qb - np.sqrt(np.maximum(disc, 0.0))) / qa
        hit = (disc >= 0) & (t_hit > 0) & (t_hit < nearest)
        if not hit.any():
            continue
        local = o_l + t_hit[hit, None] * d_l[hit]
        normal = (local / r32) @ R32.T
        normal /= np.linalg.norm(normal, axis=1, keepdims=True)
        facing = np.abs(np.einsum("ij,ij->i", normal, d[hit]))
        if glow:
            col = np.asarray(color, np.float32) * (0.80 + 0.20 * facing)[:, None]
        else:
            diffuse = (np.clip(normal @ KEY_DIR.astype(np.float32), 0, 1)[:, None] * KEY_RGB
                       + (0.5 + 0.5 * normal[:, 2:3]) * SKY_RGB)
            col = np.asarray(color, np.float32) * diffuse + 0.10 * (1.0 - facing)[:, None] ** 3
        out[hit] = col
        nearest[hit] = t_hit[hit]
        hidden |= hit
    # Its feet darken the floor a little.
    foot = np.exp(-((fx + 2.5) ** 2 + (fy - 2.35) ** 2) / 0.09)
    rgb = out.reshape(rgb.shape) * np.where(floor & ~hidden.reshape(floor.shape), 1.0 - 0.18 * foot, 1.0)[..., None]
    on_floor = (floor & ~hidden.reshape(floor.shape))[..., None]
    return {"rgb": rgb.astype(np.float32), "fx": fx, "fy": fy,
            "key": np.where(on_floor, key, 0.0).astype(np.float32),
            "ambient": np.where(on_floor, ambient, 0.0).astype(np.float32)}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _camera_rays(cam: Camera, supersample: int) -> tuple[np.ndarray, np.ndarray]:
    from .synth import camera_rays

    return camera_rays(cam, supersample)


def _texture(human: Human, quad: np.ndarray, s: np.ndarray, t: np.ndarray, spacing: float) -> np.ndarray:
    """Colour at position (s, t) in 0..1 of each quad. ``spacing``: distance between samples, metres."""
    out = np.zeros((len(quad), 3), dtype=np.float32)
    material = human.material[quad]
    for m, levels in enumerate(human.textures):
        sel = np.nonzero(material == m)[0]
        if not len(sel):
            continue
        level = int(np.clip(np.floor(np.log2(max(spacing / float(human.texel[m]), 1e-6)) + 0.5), 0, len(levels) - 1))
        atlas, n = levels[level]
        cols = atlas.shape[1] // n
        blk = human.block[quad[sel]]
        x = s[sel] * (n - 1)
        y = t[sel] * (n - 1)
        x0 = np.minimum(x.astype(np.int32), n - 2)
        y0 = np.minimum(y.astype(np.int32), n - 2)
        ax, ay = (x - x0)[:, None], (y - y0)[:, None]
        col = blk % cols * n + x0
        row = blk // cols * n + y0
        top = atlas[row, col] * (1.0 - ax) + atlas[row, col + 1] * ax
        bottom = atlas[row + 1, col] * (1.0 - ax) + atlas[row + 1, col + 1] * ax
        out[sel] = top * (1.0 - ay) + bottom * ay
    return out


def _wrap(cosine: np.ndarray, wrap: np.ndarray) -> np.ndarray:
    """Diffuse term that lets some light past the terminator, per channel."""
    return np.clip((cosine[:, None] + wrap) / (1.0 + wrap), 0.0, 1.0) ** 1.2


def shade_samples(human: Human, instant: Instant, tri: np.ndarray, bary: np.ndarray,
                  eye: np.ndarray, back: np.ndarray, spacing: float) -> np.ndarray:
    """Colour of surface samples given their triangle and (true) barycentric coordinates."""
    corners = human.tris[tri]
    quad = tri >> 1
    second = (tri & 1).astype(bool)
    s = np.where(second, bary[:, 1], bary[:, 1] + bary[:, 2])
    t = np.where(second, bary[:, 1] + bary[:, 2], bary[:, 2])
    albedo = _texture(human, quad, s, t, spacing)
    b = bary[:, :, None]
    point = (instant.verts[corners] * b).sum(axis=1)
    normal = (instant.normals[corners] * b).sum(axis=1)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
    normal[back] *= -1.0
    occlusion = (human.ao[corners] * bary).sum(axis=1)[:, None]
    params = np.array([list(w) + [k, e] for w, k, e in MATERIALS.values()], dtype=np.float32)[human.material[quad]]
    wrap = params[:, :3]
    key_cos = normal @ KEY_DIR.astype(np.float32)
    shadow = 1.0 - instant.key.lit(point, normal)
    shadow *= np.clip((key_cos + 0.15) / 0.4, 0.0, 1.0)  # near the terminator the wrap does the shading
    key = _wrap(key_cos, wrap) * (1.0 - shadow)[:, None]
    light = (key * KEY_RGB
             + _wrap(normal @ FILL_DIR.astype(np.float32), wrap) * FILL_RGB * (0.4 + 0.6 * occlusion)
             + _wrap(normal @ BACK_DIR.astype(np.float32), wrap * 0.3) * BACK_RGB * (0.3 + 0.7 * occlusion)
             + (GROUND_RGB + (SKY_RGB - GROUND_RGB) * (0.5 + 0.5 * normal[:, 2:3])) * occlusion)
    view = eye.astype(np.float32) - point
    view /= np.linalg.norm(view, axis=1, keepdims=True)
    half = view + KEY_DIR.astype(np.float32)
    half /= np.linalg.norm(half, axis=1, keepdims=True)
    gloss = params[:, 3] * np.clip((normal * half).sum(axis=1), 0.0, 1.0) ** params[:, 4] * (1.0 - shadow)
    rim = (1.0 - np.clip((normal * view).sum(axis=1), 0.0, 1.0)) ** 4 * 0.05 * occlusion[:, 0]
    return albedo * light + (gloss + rim)[:, None]


class StudioView:
    """One camera on the set: what does not change during a take is computed once."""

    def __init__(self, cam: Camera, supersample: int = 3):
        self.cam = cam
        self.ss = int(supersample)
        h, w = cam.height, cam.width
        origin, dirs = _camera_rays(cam, 2)
        full = studio_set(origin, dirs)

        def small(a):
            return cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)

        self.background = small(full["rgb"])
        self._key, self._ambient = small(full["key"]), small(full["ambient"])
        self._fx, self._fy = small(full["fx"]), small(full["fy"])

    def render(self, seconds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Image [H, W, 3] in 0..1, coverage of the subject [H, W] in 0..1, depth [H, W].

        Depth is measured along the optical axis at the centre of the pixel,
        0 where the centre does not fall on the subject.
        """
        human, cam, ss = load_human(), self.cam, self.ss
        instant = human_at(float(seconds))
        h, w = cam.height, cam.width
        px, z = cam.project(instant.verts)
        grid = (px + 0.5) * ss - 0.5  # sample (i, j) of the fine grid sits at ((j + .5) / ss - .5, ...)
        ahead = grid[z > 0.05] if (z > 0.05).any() else grid[:1]
        lo = np.clip(np.floor(ahead.min(axis=0)), 0, [w * ss - 1, h * ss - 1]).astype(int)
        hi = np.clip(np.ceil(ahead.max(axis=0)), 0, [w * ss - 1, h * ss - 1]).astype(int)
        gw, gh = int(hi[0] - lo[0] + 1), int(hi[1] - lo[1] + 1)
        front = np.nonzero(z[human.tris].min(axis=1) > 0.05)[0]  # drop what is behind the camera
        tris = human.tris[front]
        inv_z = (1.0 / np.maximum(z, 1e-3)).astype(np.float32)
        sample, tri, wts = rasterize(grid - lo, inv_z, tris, gw, gh)
        q = wts * inv_z[tris[tri]]
        depth = 1.0 / q.sum(axis=1)
        bary = q * depth[:, None]
        p = grid[tris[tri]]
        tri = front[tri]
        back = ((p[:, 1, 0] - p[:, 0, 0]) * (p[:, 2, 1] - p[:, 0, 1])
                - (p[:, 2, 0] - p[:, 0, 0]) * (p[:, 1, 1] - p[:, 0, 1])) > 0  # seen from inside
        spacing = float(np.median(depth)) / (float(cam.K[0, 0]) * ss)
        color = shade_samples(human, instant, tri, bary, cam.center, back, spacing)

        gx = sample % gw + lo[0]
        gy = sample // gw + lo[1]
        pixel = (gy // ss) * w + gx // ss
        cover = (np.bincount(pixel, minlength=w * h) / float(ss * ss)).astype(np.float32).reshape(h, w)
        rgb = np.stack([np.bincount(pixel, weights=color[:, k], minlength=w * h) for k in range(3)], axis=-1)
        rgb = (rgb / float(ss * ss)).astype(np.float32).reshape(h, w, 3)
        centre = (gx % ss == ss // 2) & (gy % ss == ss // 2) if ss % 2 else np.zeros(len(sample), dtype=bool)
        if not ss % 2:  # no sample at the pixel centre: take the nearest surface of the pixel
            order = np.argsort(-depth)
            depth_map = np.zeros(w * h, dtype=np.float32)
            depth_map[pixel[order]] = depth[order]
            depth_map[cover.reshape(-1) < 0.5] = 0.0
        else:
            depth_map = np.zeros(w * h, dtype=np.float32)
            depth_map[pixel[centre]] = depth[centre]

        floor = self.background - self._key * instant.key.on_floor(self._fx, self._fy)[..., None] \
            - 0.45 * self._ambient * instant.overhead.on_floor(self._fx, self._fy)[..., None]
        image = floor * (1.0 - cover[..., None]) + rgb
        return np.clip(image, 0.0, 1.0), cover, depth_map.reshape(h, w)


def surface_points(seconds: float, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random points on the posed surface with their colours (a stand-in for a visual hull)."""
    human = load_human()
    instant = human_at(float(seconds))
    v = instant.verts
    a, b, c = (v[human.tris[:, k]] for k in range(3))
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1).astype(np.float64)
    tri = rng.choice(len(area), size=n, p=area / area.sum())
    u = rng.random((n, 2))
    flip = u.sum(axis=1) > 1.0
    u[flip] = 1.0 - u[flip]
    bary = np.concatenate([1.0 - u.sum(axis=1, keepdims=True), u], axis=1).astype(np.float32)
    eye = np.array([4.0, 0.0, 1.4])
    color = shade_samples(human, instant, tri, bary, eye, np.zeros(n, dtype=bool), 0.02)
    points = (v[human.tris[tri]] * bary[:, :, None]).sum(axis=1)
    return points.astype(np.float32), np.clip(color, 0.0, 1.0).astype(np.float32)
