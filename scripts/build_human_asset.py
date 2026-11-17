"""Build src/mister4d/assets/human.npz from a clone of MakeHuman (development tool).

    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/makehumancommunity/makehuman
    python scripts/build_human_asset.py makehuman

Only data that MakeHuman releases as CC0 is read: the base mesh with its
helper geometry (``data/3dobjs/base.obj``), one shape target, the default
skeleton and its skinning weights. Clothes, shoes and hair are derived from
that geometry here, and every texture is painted by this script, so the asset
contains nothing else. The package itself only needs the resulting file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mister4d.human import ASSET, MATERIALS, rasterize, vertex_normals  # noqa: E402

SOLE = 0.024  # thickness of the shoe sole: the figure stands this much higher
HAIR = np.array([0.16, 0.10, 0.065])


# ---------------------------------------------------------------------------
# MakeHuman files
# ---------------------------------------------------------------------------


def to_world(v: np.ndarray) -> np.ndarray:
    """MakeHuman (decimetres, y up, facing +z) -> metres, z up, facing +x."""
    return np.stack([v[:, 2], v[:, 0], v[:, 1]], axis=1) * 0.1


def load_obj(path: Path):
    verts, faces, groups = [], [], []
    group = ""
    for line in open(path):
        if line.startswith("v "):
            verts.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("g "):
            group = line.split()[1]
        elif line.startswith("f "):
            faces.append([int(p.split("/")[0]) - 1 for p in line.split()[1:]])
            groups.append(group)
    return np.array(verts), np.array(faces), np.array(groups)


def load_target(path: Path, count: int) -> np.ndarray:
    delta = np.zeros((count, 3))
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        i, x, y, z = line.split()
        delta[int(i)] = float(x), float(y), float(z)
    return delta


def load_rig(data: Path, verts: np.ndarray):
    """Bones (parents first), joint positions and dense skinning weights [V, B]."""
    skel = json.loads((data / "rigs" / "default.mhskel").read_text())
    names, todo = [], ["root"]
    while todo:  # breadth first: parents come before children
        name = todo.pop(0)
        names.append(name)
        todo += sorted(b for b, d in skel["bones"].items() if d["parent"] == name)
    index = {n: i for i, n in enumerate(names)}
    parents = np.array([index.get(skel["bones"][n]["parent"], -1) for n in names])
    heads = np.array([verts[skel["joints"][skel["bones"][n]["head"]]].mean(axis=0) for n in names])
    tails = np.array([verts[skel["joints"][skel["bones"][n]["tail"]]].mean(axis=0) for n in names])
    weights = np.zeros((len(verts), len(names)), dtype=np.float32)
    table = json.loads((data / "rigs" / "default_weights.mhw").read_text())["weights"]
    for bone, pairs in table.items():
        if pairs:
            idx, w = np.array(pairs).T
            weights[idx.astype(int), index[bone]] = w
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-9)
    return names, parents, heads, tails, weights


# ---------------------------------------------------------------------------
# Quad meshes
# ---------------------------------------------------------------------------


class Mesh:
    """Quads with per-vertex data: position ``v``, skinning ``w`` and painting attributes."""

    def __init__(self, v, quads, w, attrs=None):
        used = np.unique(quads)
        remap = np.full(int(quads.max()) + 1, -1)
        remap[used] = np.arange(len(used))
        self.v = np.asarray(v, dtype=np.float64)[used]
        self.q = remap[quads]
        self.w = np.asarray(w, dtype=np.float32)[used]
        self.attrs = {k: np.asarray(a, dtype=np.float64)[used] for k, a in (attrs or {}).items()}

    @property
    def tris(self):
        return np.stack([self.q[:, [0, 1, 2]], self.q[:, [0, 2, 3]]], axis=1).reshape(-1, 3)

    def normals(self):
        return vertex_normals(self.v, self.tris)

    def edges(self):
        e = np.concatenate([self.q[:, [i, (i + 1) % 4]] for i in range(4)])
        return np.sort(e, axis=1)

    def boundary(self) -> np.ndarray:
        """Indices of the vertices on an open edge."""
        e, count = np.unique(self.edges(), axis=0, return_counts=True)
        return np.unique(e[count == 1])

    def loops(self) -> list[np.ndarray]:
        """Open edges grouped into connected rings of vertices."""
        e, count = np.unique(self.edges(), axis=0, return_counts=True)
        e = e[count == 1]
        label = np.arange(len(self.v))
        for _ in range(len(e)):  # tiny graphs: plain label propagation
            low = np.minimum(label[e[:, 0]], label[e[:, 1]])
            if (label[e[:, 0]] == low).all() and (label[e[:, 1]] == low).all():
                break
            np.minimum.at(label, e[:, 0], low)
            np.minimum.at(label, e[:, 1], low)
        on = np.unique(e)
        return [on[label[on] == k] for k in np.unique(label[on])]

    def neighbours(self):
        e = np.unique(self.edges(), axis=0)
        return np.concatenate([e, e[:, ::-1]])

    def smooth(self, values: np.ndarray, rounds: int, amount: float = 0.5, shrink: bool = True,
               edges_only: bool = False) -> np.ndarray:
        """Laplacian smoothing of per-vertex values; open edges only slide along themselves."""
        pairs = self.neighbours()
        edge = np.zeros(len(self.v), dtype=bool)
        edge[self.boundary()] = True
        keep = ~(edge[pairs[:, 0]] & ~edge[pairs[:, 1]])  # an edge vertex ignores inner neighbours
        if edges_only:
            keep &= edge[pairs[:, 0]]
        pairs = pairs[keep]
        count = np.bincount(pairs[:, 0], minlength=len(self.v))[:, None]
        out = np.array(values, dtype=np.float64).reshape(len(self.v), -1)
        for i in range(rounds):
            for step in ((amount,) if shrink else (amount, -amount * 1.04)):  # Taubin: no shrinkage
                mean = np.stack([np.bincount(pairs[:, 0], weights=out[pairs[:, 1], k], minlength=len(self.v))
                                 for k in range(out.shape[1])], axis=1) / np.maximum(count, 1)
                out = out + step * np.where(count > 0, mean - out, 0.0)
        return out.reshape(np.shape(values))

    def subdivide(self) -> "Mesh":
        """Every quad becomes four (new vertices at edge midpoints and face centres)."""
        nv = len(self.v)
        corners = np.concatenate([self.q[:, [i, (i + 1) % 4]] for i in range(4)])
        e, inverse = np.unique(np.sort(corners, axis=1), axis=0, return_inverse=True)
        mid = nv + inverse.reshape(4, -1).T  # [Q, 4]: midpoint of edge i -> i + 1
        centre = nv + len(e) + np.arange(len(self.q))

        def grow(a):
            return np.concatenate([a, a[e].mean(axis=1), a[self.q].mean(axis=1)])

        q = self.q
        quads = np.concatenate([
            np.stack([q[:, 0], mid[:, 0], centre, mid[:, 3]], axis=1),
            np.stack([mid[:, 0], q[:, 1], mid[:, 1], centre], axis=1),
            np.stack([centre, mid[:, 1], q[:, 2], mid[:, 2]], axis=1),
            np.stack([mid[:, 3], centre, mid[:, 2], q[:, 3]], axis=1)])
        return Mesh(grow(self.v), quads, grow(self.w), {k: grow(a) for k, a in self.attrs.items()})

    def select(self, quad_mask: np.ndarray) -> "Mesh":
        return Mesh(self.v, self.q[quad_mask], self.w, self.attrs)

    def distance_to(self, points: np.ndarray) -> np.ndarray:
        return cKDTree(points).query(self.v)[0]


def smoothstep(x, a, b):
    u = np.clip((np.asarray(x, dtype=np.float64) - a) / (b - a), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


class Noise:
    """Smooth 3D noise in -1..1 from a lattice of random values."""

    def __init__(self, seed: int):
        self.lattice = np.random.default_rng(seed).uniform(-1.0, 1.0, size=(64, 64, 64))

    def __call__(self, p: np.ndarray, period: float, octaves: int = 1, stretch=(1.0, 1.0, 1.0)) -> np.ndarray:
        """``period``: size of the largest features, metres."""
        shape = p.shape[:-1]
        q = (p.reshape(-1, 3) / (period * np.asarray(stretch))).T
        out, amp, total = np.zeros(q.shape[1]), 1.0, 0.0
        for o in range(octaves):
            out += amp * map_coordinates(self.lattice, q * 2.0 ** o + 17.3 * o, order=1, mode="wrap")
            total += amp
            amp *= 0.5
        return (out / total * 1.8).reshape(shape)


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------


def group_weight(names, weights, prefixes) -> np.ndarray:
    cols = [i for i, n in enumerate(names) if n.startswith(tuple(prefixes))]
    return weights[:, cols].sum(axis=1)


def descendants(names, parents, root) -> list[int]:
    out = [names.index(root)]
    for i, p in enumerate(parents):
        if p in out:
            out.append(i)
    return out


def build(data: Path, log=print) -> dict:
    verts, faces, groups = load_obj(data / "3dobjs" / "base.obj")
    verts = verts + load_target(data / "targets" / "macrodetails" / "caucasian-male-young.target", len(verts))
    verts = to_world(verts)
    body_faces = faces[groups == "body"]
    verts[:, 2] -= verts[np.unique(body_faces), 2].min()
    names, parents, heads, tails, weights = load_rig(data, verts)
    joint = {n: heads[i] for i, n in enumerate(names)}
    log(f"body height {verts[np.unique(body_faces), 2].max():.3f} m, {len(names)} bones")

    w_head = weights[:, descendants(names, parents, "neck01")].sum(axis=1)
    w_hand = group_weight(names, weights, ("wrist", "metacarpal", "finger"))
    w_arm = group_weight(names, weights, ("shoulder01", "upperarm", "lowerarm")) + w_hand
    w_fore = group_weight(names, weights, ("lowerarm",))
    w_foot = group_weight(names, weights, ("foot", "toe"))
    w_leg = group_weight(names, weights, ("upperleg", "lowerleg")) + w_foot
    w_shin = group_weight(names, weights, ("lowerleg",))
    side = np.where(verts[:, 1] >= 0, "L", "R")

    def along(a: str, b: str) -> np.ndarray:
        """Position of every vertex along the line between two joints of its side (0 at a, 1 at b)."""
        out = np.zeros(len(verts))
        for s in "LR":
            pa, pb = joint[f"{a}.{s}"], joint[f"{b}.{s}"]
            sel = side == s
            out[sel] = (verts[sel] - pa) @ (pb - pa) / ((pb - pa) @ (pb - pa))
        return out

    fore_s = along("lowerarm01", "wrist")
    arm_s = along("upperarm01", "wrist")
    shin_s = along("lowerleg01", "foot")
    z_hip = joint["upperleg01.L"][2]
    z_ankle = joint["foot.L"][2]
    z_neck = joint["neck01"][2]
    attrs = {"arm": w_arm, "arm_s": arm_s, "fore_s": fore_s, "shin": w_shin, "shin_s": shin_s, "leg": w_leg,
             "foot": w_foot, "hand": w_hand, "head": w_head}
    for key, prefixes in (("oris", ("oris",)), ("oculi", ("oculi",)), ("orbicularis", ("orbicularis",)),
                          ("levator", ("levator",)), ("risorius", ("risorius",)), ("jaw", ("jaw", "special04")),
                          ("temporalis", ("temporalis",)), ("special", ("special0",)), ("eye", ("eye",)),
                          ("tip", ("finger1-3", "finger2-3", "finger3-3", "finger4-3", "finger5-3"))):
        attrs[key] = group_weight(names, weights, prefixes)
    full = Mesh(verts, faces[np.isin(groups, ["body", "helper-tights", "helper-l-eye", "helper-r-eye"])], weights, attrs)
    # ``full`` keeps the vertex order of the file for the parts selected below.
    def part(group_names, vertex_ok) -> Mesh:
        f = faces[np.isin(groups, group_names)]
        return Mesh(verts, f[vertex_ok[f].all(axis=1)], weights, attrs)

    def part_any(group_names, vertex_ok) -> Mesh:
        f = faces[np.isin(groups, group_names)]
        return Mesh(verts, f[vertex_ok[f].any(axis=1)], weights, attrs)

    # Skin that shows: head and neck down into the collar, hands and the end of the forearms.
    shows = (w_head > 0.05) | ((verts[:, 2] > z_neck - 0.07) & (w_arm < 0.3)) | (w_hand > 0.02) \
        | ((w_fore > 0.5) & (fore_s > 0.62))
    skin = part_any(["body"], shows)
    eyes = part(["helper-l-eye", "helper-r-eye"], np.ones(len(verts), dtype=bool))

    suit = part(["helper-tights"], np.ones(len(verts), dtype=bool))
    z_hem, z_waist, z_cuff = z_hip - 0.035, z_hip + 0.11, z_ankle + 0.012
    is_shoe = (verts[:, 2] < z_ankle + 0.055) & (w_leg > 0.5)
    def collar(x):  # round neck: lower at the front
        return z_neck + 0.016 - 0.45 * (x + 0.02)

    is_top = ((verts[:, 2] > z_hem) | (w_arm > 0.5)) & (w_leg < 0.9) & (verts[:, 2] < collar(verts[:, 0]))
    is_trousers = (verts[:, 2] < z_waist) & (verts[:, 2] > z_cuff) & (w_arm < 0.5)
    top = part(["helper-tights"], is_top)
    trousers = part(["helper-tights"], is_trousers)
    shoes = part(["helper-tights"], is_shoe)
    log(f"quads: skin {len(skin.q)}, top {len(top.q)}, trousers {len(trousers.q)}, shoes {len(shoes.q)}")
    del suit, full

    top = shape_top(top, joint, z_hem, collar)
    trousers = shape_trousers(trousers, joint, z_waist, z_cuff)
    shoes = shape_shoes(shoes, joint, names)
    hair = shape_hair(skin, joint)
    parts = {"skin": skin, "top": top, "trousers": trousers, "shoes": shoes, "hair": hair, "eyes": eyes}
    return {"parts": parts, "names": names, "parents": parents, "heads": heads, "joint": joint}


def ring_distance(mesh: Mesh, pick) -> np.ndarray:
    """Distance of every vertex from the open edges chosen by ``pick(points of the ring)``."""
    rings = [mesh.v[r] for r in mesh.loops() if pick(mesh.v[r])]
    if not rings:
        return np.full(len(mesh.v), 9.0)
    return mesh.distance_to(np.concatenate(rings))


def shape_top(m: Mesh, joint, z_hem: float, collar) -> Mesh:
    """A sweatshirt: the suit moved off the body, with tighter ribbed edges and a few folds."""
    edge = m.boundary()
    m.v[edge[np.abs(m.v[edge, 2] - z_hem) < 0.05], 2] = z_hem
    neck = edge[(m.v[edge, 2] > joint["neck01"][2] - 0.10) & (np.abs(m.v[edge, 1]) < 0.12)]
    m.v = m.smooth(m.v, 4, edges_only=True)  # the cuts follow the quads: straighten them
    m.v[edge[np.abs(m.v[edge, 2] - z_hem) < 0.05], 2] = z_hem
    m.v[neck, 2] = collar(m.v[neck, 0])
    m = m.subdivide()
    m.attrs["hem"] = ring_distance(m, lambda p: p[:, 2].mean() < z_hem + 0.03)
    m.attrs["cuff"] = ring_distance(m, lambda p: np.abs(p[:, 1]).mean() > 0.3)
    m.attrs["collar"] = ring_distance(m, lambda p: p[:, 2].mean() > joint["neck01"][2] - 0.10 and np.abs(p[:, 1]).max() < 0.2)
    n = m.normals()
    noise = Noise(3)
    arm = smoothstep(m.attrs["arm"], 0.3, 0.8)
    loose = 0.013 + 0.004 * arm
    loose = loose - 0.008 * (1 - smoothstep(m.attrs["cuff"], 0.02, 0.06)) + 0.006 * np.exp(-((m.attrs["cuff"] - 0.085) / 0.03) ** 2)
    loose = loose - 0.004 * (1 - smoothstep(m.attrs["hem"], 0.03, 0.07)) + 0.007 * np.exp(-((m.attrs["hem"] - 0.10) / 0.04) ** 2)
    loose = loose - 0.006 * (1 - smoothstep(m.attrs["collar"], 0.012, 0.03))
    # Fill the hollow of the waist and the back: a sweatshirt hangs from chest and shoulders.
    waist = np.exp(-((m.v[:, 2] - (z_hem + 0.20)) / 0.12) ** 2) * (1 - arm)
    loose = loose + 0.012 * waist
    folds = noise(m.v, 0.11, 2, stretch=(1.0, 1.0, 2.2)) * 0.0045 + noise(m.v, 0.045, 1) * 0.0015
    folds *= smoothstep(np.minimum(m.attrs["collar"], m.attrs["cuff"]), 0.02, 0.08)
    m.v = m.v + n * (loose + folds)[:, None]
    m.v = m.smooth(m.v, 2, shrink=False)
    return m


def shape_trousers(m: Mesh, joint, z_waist: float, z_cuff: float) -> Mesh:
    """Straight-leg jeans: the suit moved off the body and widened below the knee."""
    edge = m.boundary()
    m.v[edge[np.abs(m.v[edge, 2] - z_waist) < 0.05], 2] = z_waist
    m.v = m.smooth(m.v, 3, edges_only=True)
    m.v[edge[np.abs(m.v[edge, 2] - z_waist) < 0.05], 2] = z_waist
    m.v[edge[np.abs(m.v[edge, 2] - z_cuff) < 0.04], 2] = z_cuff
    m = m.subdivide()
    m.attrs["cuff"] = ring_distance(m, lambda p: p[:, 2].mean() < z_cuff + 0.05)
    m.attrs["waist"] = ring_distance(m, lambda p: p[:, 2].mean() > z_waist - 0.05)
    noise = Noise(5)
    m.v = m.v + m.normals() * 0.009
    for s in "LR":
        knee, ankle = joint[f"lowerleg01.{s}"], joint[f"foot.{s}"]
        axis = (ankle - knee) / np.linalg.norm(ankle - knee)
        sel = ((m.v[:, 1] >= 0) == (s == "L")) & (m.attrs["shin"] > 0.02)
        rel = m.v[sel] - knee
        t = rel @ axis
        radial = rel - t[:, None] * axis
        r = np.linalg.norm(radial, axis=1)
        u = np.clip(t / np.linalg.norm(ankle - knee), 0.0, 1.2)
        wide = 0.074 - 0.006 * u  # leg of the trousers: about 21 cm across at the bottom
        target = np.maximum(r, wide)
        blend = smoothstep(u, 0.0, 0.45) * smoothstep(m.attrs["shin"][sel], 0.02, 0.5)
        m.v[sel] += radial / np.maximum(r, 1e-6)[:, None] * ((target - r) * blend)[:, None]
    n = m.normals()
    folds = noise(m.v, 0.09, 2, stretch=(1.6, 1.6, 0.8)) * 0.003
    stack = np.sin(m.v[:, 2] / 0.021 + 2.5 * noise(m.v, 0.12)) * 0.0035 * (1 - smoothstep(m.attrs["cuff"], 0.03, 0.16))
    m.v = m.v + n * (folds + stack)[:, None]
    m.v = m.smooth(m.v, 1, shrink=False)
    return m


def shape_shoes(m: Mesh, joint, names) -> Mesh:
    """Trainers: the feet of the suit, inflated, smoothed, standing on a flat sole."""
    m = m.subdivide()
    m.attrs["top"] = ring_distance(m, lambda p: True)
    m.v = m.smooth(m.v, 3, shrink=False)
    n = m.normals()
    m.v = m.v + n * (0.007 + 0.003 * (1 - smoothstep(m.attrs["top"], 0.0, 0.03)))[:, None]
    # A sole under the foot: the lowest band is pulled down, the underside becomes a flat
    # tread and a rim sticks out around it.
    under = -n[:, 2] > 0.5
    height = m.v[:, 2] - m.v[:, 2].min()
    m.v[:, 2] = np.maximum(m.v[:, 2] - SOLE * (1 - smoothstep(height, 0.004, 0.022)), -SOLE)
    m.v[under & (height < 0.03), 2] = -SOLE
    rim = (1 - smoothstep(height, 0.004, 0.020)) * ~under
    flat = n * np.array([1.0, 1.0, 0.0])
    m.v = m.v + flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-6) * (0.004 * rim)[:, None]
    m.attrs["sole"] = smoothstep(-m.normals()[:, 2], 0.55, 0.9)
    m.attrs["height"] = m.v[:, 2] + SOLE
    # The shoe is rigid: it follows the foot, not the toes.
    foot = {s: names.index(f"foot.{s}") for s in "LR"}
    toes = [i for i, name in enumerate(names) if name.startswith("toe")]
    for s in "LR":
        mine = [i for i in toes if names[i].endswith(s)]
        m.w[:, foot[s]] += m.w[:, mine].sum(axis=1)
        m.w[:, mine] = 0.0
    return m


def hair_fields(p: np.ndarray, joint) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where hair grows, for points on the head.

    Returns degrees above the hairline, degrees above the edge of the longer
    hair on top (the sides and the back are cut short), and the direction from
    the centre of the skull.
    """
    centre = np.array([joint["eye.L"][0] - 0.075, 0.0, joint["eye.L"][2] + 0.015])
    d = p - centre
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    around = np.degrees(np.arctan2(np.abs(d[..., 1]), d[..., 0]))  # 0 = front, 180 = back
    height = np.degrees(np.arcsin(np.clip(d[..., 2], -1, 1)))
    line = np.interp(around, [0, 40, 62, 70, 76, 83, 89, 102, 115, 135, 180],
                     [31, 33, 26, 14, -7, -7, 5, 4, -14, -46, -58])  # with sideburns, down to the nape
    margin = np.interp(around, [0, 40, 62, 76, 89, 115, 150, 180], [0, 4, 9, 34, 22, 24, 34, 36])
    ear = np.stack([np.full_like(p[..., 0], joint["eye.L"][0] - 0.09), np.abs(p[..., 1]),
                    p[..., 2]], axis=-1) - np.array([0.0, 0.079, joint["eye.L"][2] - 0.008])
    ear_x = p[..., 0] - (joint["eye.L"][0] - 0.09)
    near_ear = np.sqrt(ear_x ** 2 + ear[..., 1] ** 2 + ear[..., 2] ** 2)
    above = height - line
    above = np.where((near_ear < 0.034) & (np.abs(p[..., 1]) > 0.069), -20.0, above)
    return above, height - line - margin, d


def shape_hair(skin: Mesh, joint) -> Mesh:
    """Short hair: the top of the scalp moved outwards, with strands; thin towards its edge."""
    above, top, _ = hair_fields(skin.v, joint)
    scalp = (top > -12) & (above > -8) & (skin.attrs["head"] > 0.5)
    m = skin.select(scalp[skin.q].any(axis=1)).subdivide().subdivide()
    noise = Noise(11)
    above, top, _ = hair_fields(m.v, joint)  # cut the edge on the fine mesh: no steps
    keep = (np.minimum(top, above) + 1.5 * noise(m.v, 0.008) > 0)
    m = m.select(keep[m.q].all(axis=1))
    above, top, d = hair_fields(m.v, joint)
    m.attrs["edge"] = m.distance_to(m.v[m.boundary()])
    grow = smoothstep(m.attrs["edge"], 0.0, 0.045)
    thick = 0.0025 + 0.013 * grow * (0.6 + 0.4 * smoothstep(d[:, 2], 0.3, 0.9))
    thick += 0.011 * smoothstep(d[:, 0], 0.15, 0.7) * smoothstep(m.attrs["edge"], 0.0, 0.03)  # a bit of a quiff
    strands = noise(m.v, 0.03, 2, stretch=(1.0, 0.3, 1.0)) * 0.003 * smoothstep(m.attrs["edge"], 0.0, 0.02)
    m.v = m.v + m.normals() * (thick + strands)[:, None]
    return m


# ---------------------------------------------------------------------------
# Painting
# ---------------------------------------------------------------------------


def block_points(values: np.ndarray, quads: np.ndarray, n: int) -> np.ndarray:
    """Per-vertex values [V, k] at the n x n texels of every quad -> [Q, n, n, k].

    Texel (i, j) sits at (s, t) = (j, i) / (n - 1) and the quad is split along
    the diagonal from corner 0 to corner 2, exactly as the renderer does.
    """
    c = values[quads].astype(np.float64)  # [Q, 4, k]
    t, s = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="ij")
    s, t = s[None, :, :, None], t[None, :, :, None]
    c0, c1, c2, c3 = (c[:, k][:, None, None, :] for k in range(4))
    first = c0 + (c1 - c0) * s + (c2 - c1) * t
    second = c0 + (c2 - c3) * s + (c3 - c0) * t
    return np.where(s >= t, first, second)


class Canvas:
    """What a painter gets: positions, normals and attributes at every texel of a part."""

    def __init__(self, mesh: Mesh, n: int):
        self.p = block_points(mesh.v, mesh.q, n)
        normal = block_points(mesh.normals(), mesh.q, n)
        self.n = normal / np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1e-9)
        self.attr = {k: block_points(a[:, None], mesh.q, n)[..., 0] for k, a in mesh.attrs.items()}
        self.x, self.y, self.z = self.p[..., 0], self.p[..., 1], self.p[..., 2]


def mix(a, b, t):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return a + (b - a) * np.asarray(t)[..., None]


def segment_distance(y, z, points) -> np.ndarray:
    """Distance in the (y, z) plane from a polyline given as [(y, z), ...]."""
    best = np.full(np.shape(y), np.inf)
    for (ya, za), (yb, zb) in zip(points[:-1], points[1:]):
        t = np.clip(((y - ya) * (yb - ya) + (z - za) * (zb - za)) / ((yb - ya) ** 2 + (zb - za) ** 2), 0, 1)
        best = np.minimum(best, np.hypot(y - ya - t * (yb - ya), z - za - t * (zb - za)))
    return best


def paint_skin(c: Canvas, joint) -> np.ndarray:
    noise = Noise(21)
    eye, a = joint["eye.L"], c.attr
    ay = np.abs(c.y)
    front = smoothstep(c.n[..., 0], 0.0, 0.5)
    rgb = np.broadcast_to(np.array([0.78, 0.585, 0.485]), c.p.shape).copy()
    rgb *= (1.0 + 0.03 * noise(c.p, 0.05, 2) + 0.018 * noise(c.p, 0.0035))[..., None]

    # Blood shows through on cheeks, nose, ears, knuckles and finger tips.
    red = np.array([0.80, 0.42, 0.36])
    cheek = np.exp(-((c.x - (eye[0] - 0.012)) ** 2 + (ay - 0.047) ** 2 + (c.z - (eye[2] - 0.045)) ** 2) / 0.022 ** 2)
    nose = np.exp(-((c.x - (eye[0] + 0.045)) ** 2 + ay ** 2 + (c.z - (eye[2] - 0.038)) ** 2) / 0.016 ** 2)
    ears = smoothstep(ay, 0.074, 0.084) * a["head"]
    rgb = mix(rgb, red, np.clip(0.30 * cheek + 0.28 * nose + 0.30 * ears + 0.30 * a["tip"] * a["hand"], 0, 0.5))
    palm = a["hand"] * smoothstep(noise(c.p, 0.02), -0.2, 0.6) * 0.10
    rgb = mix(rgb, red, palm)

    # Eye sockets, lash line, brows.
    rgb *= (1.0 - 0.10 * np.exp(-((ay - eye[1]) ** 2 + (c.z - eye[2]) ** 2) / 0.024 ** 2) * front * a["head"])[..., None]
    lid = smoothstep(a["orbicularis"], 0.55, 0.95)
    rgb = mix(rgb, np.array([0.30, 0.19, 0.15]), 0.55 * lid)
    brow = segment_distance(ay, c.z, [(eye[1] + 0.030, eye[2] + 0.013), (eye[1] + 0.012, eye[2] + 0.0225),
                                      (eye[1] - 0.014, eye[2] + 0.0195)])
    hairs = 0.75 + 0.25 * noise(c.p, 0.0025, 1, stretch=(1.0, 3.0, 1.0))
    width = 0.0042 - 0.0012 * smoothstep(ay, eye[1], eye[1] + 0.03)
    rgb = mix(rgb, HAIR * 1.1, 0.88 * (1 - smoothstep(brow, width * 0.55, width * 1.25)) * hairs * front * a["head"])

    # Lips, and the shadow of a shaven beard.
    mouth_z = 0.5 * (joint["oris05"][2] + joint["oris01"][2]) + 0.0035
    upper = (c.z - mouth_z) / (0.0085 - 0.004 * (ay / 0.026) ** 2 + 0.0015 * np.exp(-(ay / 0.004) ** 2) * -1.0)
    lower = (mouth_z - c.z) / 0.0105
    shape = (ay / 0.0265) ** 2 + np.where(c.z > mouth_z, upper, lower) ** 2
    lips = (1 - smoothstep(shape, 0.75, 1.15)) * smoothstep(a["oris"], 0.35, 0.6) * front
    rgb = mix(rgb, np.array([0.66, 0.36, 0.34]), 0.85 * lips)
    rgb *= (1.0 - 0.45 * np.exp(-((c.z - mouth_z) / 0.0012) ** 2) * (ay < 0.026) * lips)[..., None]
    beard = np.clip(smoothstep(a["jaw"], 0.25, 0.7) + smoothstep(a["oris"], 0.3, 0.6), 0, 1) * (1 - lips)
    beard *= smoothstep(mouth_z + 0.022 - c.z, 0.0, 0.012) * (0.85 + 0.15 * noise(c.p, 0.002))
    beard *= smoothstep(c.z, joint["oris01"][2] - 0.062, joint["oris01"][2] - 0.040) * smoothstep(c.x, eye[0] - 0.09, eye[0] - 0.04)
    rgb = mix(rgb, np.array([0.50, 0.42, 0.38]), 0.26 * beard * a["head"])

    # Hair cut short on the sides and the back, fading into the skin.
    above, _, _ = hair_fields(c.p, joint)
    short = smoothstep(above + 1.5 * noise(c.p, 0.012), -3.0, 6.0) * (a["head"] > 0.5)
    stubble = 0.88 + 0.12 * noise(c.p, 0.0022)
    rgb = mix(rgb, HAIR * 1.08, 0.94 * short * stubble)
    import os
    if os.environ.get("MH_DEBUG"):
        keys = os.environ["MH_DEBUG"].split(",")
        rgb = np.stack([np.clip(c.attr[k], 0, 1) if k in c.attr else np.zeros(c.x.shape) for k in (keys + ["", ""])[:3]], axis=-1) * 0.8 + 0.2
    return np.clip(rgb, 0, 1)


def chest_print(size: int = 512) -> np.ndarray:
    """The emblem on the chest, 0..1."""
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), int(size * 0.46), 255, int(size * 0.045), cv2.LINE_AA)
    text, font = "4D", cv2.FONT_HERSHEY_DUPLEX
    scale = size / 512 * 7.0
    (tw, th), _ = cv2.getTextSize(text, font, scale, int(size * 0.05))
    cv2.putText(img, text, ((size - tw) // 2, (size + th) // 2), font, scale, 255, int(size * 0.05), cv2.LINE_AA)
    return img.astype(np.float32) / 255.0


def paint_top(c: Canvas, joint) -> np.ndarray:
    noise = Noise(31)
    a = c.attr
    base, cream = np.array([0.56, 0.115, 0.13]), np.array([0.93, 0.91, 0.85])
    arm = smoothstep(a["arm"], 0.35, 0.65)
    chest_z = joint["spine01"][2] + 0.085
    # Two bands across chest and back, two rings on each upper arm.
    band = np.maximum(1 - smoothstep(np.abs(c.z - (chest_z - 0.135)), 0.016, 0.019),
                      1 - smoothstep(np.abs(c.z - (chest_z - 0.178)), 0.006, 0.009)) * (1 - arm)
    ring = np.maximum(1 - smoothstep(np.abs(a["arm_s"] - 0.30), 0.028, 0.034),
                      1 - smoothstep(np.abs(a["arm_s"] - 0.385), 0.011, 0.017)) * arm
    rgb = mix(base, cream, np.maximum(band, ring))
    # Emblem on the chest.
    size = 0.19
    u = (c.y / size + 0.5) * 511.0
    v = ((chest_z - c.z) / size + 0.5) * 511.0
    emblem = map_coordinates(chest_print(), [v.ravel(), u.ravel()], order=1, mode="constant").reshape(u.shape)
    rgb = mix(rgb, cream, emblem * smoothstep(c.n[..., 0], 0.1, 0.4) * (1 - arm))
    # Ribbed edges, knit.
    rib = 1 - smoothstep(np.minimum(np.minimum(a["cuff"] - 0.03, a["hem"] - 0.035), a["collar"] - 0.014), 0.0, 0.004)
    rgb = mix(rgb, base * 0.72, rib)
    weave = 1.0 + 0.05 * noise(c.p, 0.0028) + 0.05 * noise(c.p, 0.03, 2) + 0.03 * noise(c.p, 0.2)
    return rgb * weave[..., None]


def leg_frame(c: Canvas, joint):
    """Distance from the axis of the leg and position around it (0 = outer seam)."""
    r_out, r_fwd = np.zeros(c.x.shape), np.zeros(c.x.shape)
    for s in "LR":
        sel = (c.y >= 0) == (s == "L")
        hip, knee, ankle = joint[f"upperleg01.{s}"], joint[f"lowerleg01.{s}"], joint[f"foot.{s}"]
        best = np.full(c.x.shape, np.inf)
        for pa, pb in ((hip, knee), (knee, ankle)):
            axis = (pb - pa) / np.linalg.norm(pb - pa)
            rel = c.p - pa
            t = np.clip(rel @ axis, -0.2, np.linalg.norm(pb - pa))
            radial = rel - t[..., None] * axis
            dist = np.linalg.norm(radial, axis=-1)
            out = np.array([0.0, 1.0 if s == "L" else -1.0, 0.0])
            out = out - (out @ axis) * axis
            out /= np.linalg.norm(out)
            use = sel & (dist < best)
            best = np.where(use, dist, best)
            r_out = np.where(use, radial @ out, r_out)
            r_fwd = np.where(use, radial @ np.cross(out, axis) * (1.0 if s == "L" else -1.0), r_fwd)
    return r_out, r_fwd


def paint_trousers(c: Canvas, joint) -> np.ndarray:
    noise = Noise(41)
    a = c.attr
    dark, light = np.array([0.105, 0.15, 0.27]), np.array([0.33, 0.43, 0.60])
    z_hip, z_knee = joint["upperleg01.L"][2], joint["lowerleg01.L"][2]
    # Worn lighter on the thighs, the knees and the seat.
    wear = 0.30 + 0.30 * smoothstep(c.n[..., 0], -0.2, 0.9) * np.exp(-((c.z - (z_knee + 0.20)) / 0.22) ** 2)
    wear += 0.22 * smoothstep(-c.n[..., 0], 0.2, 0.9) * np.exp(-((c.z - (z_hip - 0.05)) / 0.10) ** 2)
    wear += 0.12 * noise(c.p, 0.07, 2) + 0.10 * noise(c.p, 0.006, 1, stretch=(1.0, 1.0, 9.0))
    rgb = mix(dark, light, np.clip(wear, 0, 1))
    rgb *= (1.0 + 0.06 * noise(c.p, 0.0022))[..., None]
    # Seams along the legs, hem, back pockets.
    r_out, r_fwd = leg_frame(c, joint)
    legs = smoothstep(z_hip - 0.04 - c.z, 0.0, 0.03)
    for side_of_leg in (r_out > 0, r_out < 0):
        seam = np.abs(r_fwd) * np.where(side_of_leg, 1.0, 9.0)
        rgb *= (1.0 - 0.35 * np.exp(-(seam / 0.0022) ** 2) * legs)[..., None]
        rgb = mix(rgb, light * 1.1, 0.5 * np.exp(-((seam - 0.006) / 0.0018) ** 2) * legs * side_of_leg)
    rgb = mix(rgb, np.array([0.72, 0.56, 0.30]), 0.6 * np.exp(-((a["cuff"] - 0.016) / 0.0016) ** 2))
    rgb *= (1.0 - 0.25 * np.exp(-((a["cuff"] - 0.019) / 0.002) ** 2))[..., None]
    back = smoothstep(-c.n[..., 0], 0.3, 0.6)
    py, pz = np.abs(c.y) - 0.085, c.z - (z_hip - 0.035)
    pocket = np.maximum(np.abs(py) - 0.062, np.abs(pz + 0.25 * np.abs(py)) - 0.068)
    rgb = mix(rgb, np.array([0.72, 0.56, 0.30]), 0.55 * np.exp(-((pocket + 0.004) / 0.0018) ** 2) * back)
    rgb *= (1.0 - 0.22 * np.exp(-(pocket / 0.0025) ** 2) * back)[..., None]
    return rgb


def paint_shoes(c: Canvas, joint) -> np.ndarray:
    noise = Noise(51)
    a = c.attr
    upper, rubber = np.array([0.13, 0.135, 0.15]), np.array([0.88, 0.87, 0.84])
    rgb = np.broadcast_to(upper, c.p.shape) * (1.0 + 0.08 * noise(c.p, 0.004) + 0.06 * noise(c.p, 0.03))[..., None]
    toe = c.x.max()
    cap = smoothstep(c.x, toe - 0.050, toe - 0.042) * (a["height"] < 0.066)
    side = 1 - smoothstep(a["height"], 0.027, 0.030)
    rgb = mix(rgb, rubber, np.maximum(cap, side))
    rgb = mix(rgb, np.array([0.22, 0.22, 0.23]), smoothstep(a["sole"], 0.4, 0.7))
    rgb *= (1.0 - 0.3 * np.exp(-((a["height"] - 0.0285) / 0.0012) ** 2))[..., None]
    # Laces over the instep, padded collar.
    foot_y = np.where(c.y >= 0, joint["foot.L"][1], joint["foot.R"][1])
    instep = smoothstep(c.n[..., 2], 0.35, 0.6) * (np.abs(c.y - foot_y) < 0.016) * (c.x > joint["foot.L"][0] + 0.035) \
        * (c.x < toe - 0.07)
    laces = (np.modf(c.x / 0.014 + np.abs(c.y - foot_y) / 0.02 + 50.0)[0] < 0.42)
    rgb = mix(rgb, rubber * 0.95, instep * laces)
    rgb = mix(rgb, np.array([0.30, 0.30, 0.32]), 1 - smoothstep(a["top"], 0.006, 0.010))
    return rgb


def paint_hair(c: Canvas, joint) -> np.ndarray:
    noise = Noise(61)
    _, _, d = hair_fields(c.p, joint)
    crown = np.array([-0.35, 0.05, 0.93]) / np.linalg.norm([-0.35, 0.05, 0.93])
    e1 = np.cross(crown, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(crown, e1)
    around = np.arctan2(d @ e2, d @ e1)
    away = np.arccos(np.clip(d @ crown, -1, 1))
    q = np.stack([np.cos(around) * 0.05, np.sin(around) * 0.05, away * 0.012], axis=-1)
    strands = noise(q, 0.0035, 2) * 0.6 + noise(q, 0.012) * 0.4
    rgb = HAIR * (1.0 + 0.45 * strands)[..., None]
    return rgb * (0.85 + 0.25 * smoothstep(c.attr["edge"], 0.0, 0.04))[..., None]


def paint_eyes(c: Canvas, joint) -> np.ndarray:
    noise = Noise(71)
    white = np.array([0.90, 0.89, 0.87])
    rgb = np.zeros(c.p.shape)
    for s in "LR":
        d = c.p - joint[f"eye.{s}"]
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        angle = np.degrees(np.arccos(np.clip(d[..., 0], -1, 1)))
        fibres = 0.85 + 0.25 * noise(d * 0.012, 0.0012, 1)
        iris = np.array([0.36, 0.23, 0.11]) * (0.55 + 0.6 * smoothstep(angle, 10, 24))[..., None] * fibres[..., None]
        iris = mix(iris, np.array([0.03, 0.025, 0.02]), 1 - smoothstep(angle, 9.5, 11.5))  # pupil
        iris = mix(iris, np.array([0.10, 0.07, 0.05]), smoothstep(angle, 25.0, 29.0))  # darker rim
        ball = mix(white, np.array([0.78, 0.60, 0.56]), 0.5 * smoothstep(angle, 45, 85))
        col = mix(iris, ball, smoothstep(angle, 28.5, 30.5))
        rgb = np.where(((c.y >= 0) == (s == "L"))[..., None], col, rgb)
    return rgb


PAINTERS = {"skin": (paint_skin, 9), "top": (paint_top, 9), "trousers": (paint_trousers, 9),
            "shoes": (paint_shoes, 9), "hair": (paint_hair, 5), "eyes": (paint_eyes, 9)}


def atlas(blocks: np.ndarray) -> np.ndarray:
    """Tile the painted blocks [Q, n, n, 3] into one image, row after row."""
    q, n = blocks.shape[:2]
    cols = max(1, int(np.ceil(np.sqrt(q))))
    rows = int(np.ceil(q / cols))
    out = np.zeros((rows * cols, n, n, 3))
    out[:q] = blocks
    return out.reshape(rows, cols, n, n, 3).transpose(0, 2, 1, 3, 4).reshape(rows * n, cols * n, 3)


# ---------------------------------------------------------------------------
# Ambient occlusion
# ---------------------------------------------------------------------------


def ambient_occlusion(verts: np.ndarray, tris: np.ndarray, normals: np.ndarray, directions: int = 96) -> np.ndarray:
    """Share of the sky each vertex sees (cosine weighted), from many depth maps of the figure."""
    cell = 0.006
    i = np.arange(directions) + 0.5
    zs = 1.0 - 1.15 * i / directions  # mostly from above, some from just below the horizon
    phi = i * np.pi * (3.0 - np.sqrt(5.0))
    dirs = np.stack([np.sqrt(1 - zs * zs) * np.cos(phi), np.sqrt(1 - zs * zs) * np.sin(phi), zs], axis=1)
    seen, total = np.zeros(len(verts)), np.zeros(len(verts))
    for d in dirs:
        e1 = np.cross([0.0, 0.0, 1.0], d) if abs(d[2]) < 0.99 else np.array([1.0, 0.0, 0.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(d, e1)
        uv = verts @ np.stack([e1, e2]).T
        depth = -(verts @ d)
        origin = uv.min(axis=0) - 2 * cell
        w, h = np.ceil((uv.max(axis=0) - origin) / cell).astype(int) + 2
        grid = (uv - origin) / cell
        sample, t, wts = rasterize(grid, -depth, tris, int(w), int(h))
        zbuf = np.full(w * h, np.inf)
        zbuf[sample] = (wts * depth[tris[t]]).sum(axis=1)
        at = np.clip(np.round(grid).astype(int), 0, [w - 1, h - 1])
        visible = depth <= zbuf[at[:, 1] * w + at[:, 0]] + 0.012
        weight = np.clip(normals @ d, 0.0, None) * (0.4 + 0.6 * (0.5 + 0.5 * d[2]))
        seen += weight * visible
        total += weight
    return seen / np.maximum(total, 1e-6)


# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("makehuman", help="clone of https://github.com/makehumancommunity/makehuman")
    ap.add_argument("--out", default=str(ASSET))
    args = ap.parse_args()
    data = Path(args.makehuman) / "makehuman" / "data"
    figure = build(data)
    parts, joint = figure["parts"], figure["joint"]

    verts, quads, material, weights, textures, texel, blocks = [], [], [], [], [], [], []
    offset = 0
    for m, name in enumerate(MATERIALS):
        mesh = parts[name]
        painter, n = PAINTERS[name]
        colour = np.clip(painter(Canvas(mesh, n), joint), 0.0, 1.0)
        image = (atlas(colour) * 255.0 + 0.5).astype(np.uint8)
        textures.append(cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))[1])
        edge = np.linalg.norm(mesh.v[mesh.q] - mesh.v[np.roll(mesh.q, 1, axis=1)], axis=-1)
        texel.append(float(np.median(edge)) / (n - 1))
        blocks.append(n)
        verts.append(mesh.v)
        quads.append(mesh.q + offset)
        material.append(np.full(len(mesh.q), m))
        weights.append(mesh.w)
        offset += len(mesh.v)
        print(f"{name}: {len(mesh.v)} vertices, {len(mesh.q)} quads, texture {image.shape[1]}x{image.shape[0]}, "
              f"{texel[-1] * 1000:.1f} mm per texel, {len(textures[-1]) / 1e6:.2f} MB")
    verts = np.concatenate(verts)
    verts[:, 2] += SOLE
    heads = figure["heads"] + np.array([0.0, 0.0, SOLE])
    quads = np.concatenate(quads)
    weights = np.concatenate(weights)
    tris = np.stack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=1).reshape(-1, 3)
    normals = vertex_normals(verts.astype(np.float32), tris)
    ao = ambient_occlusion(verts, tris, normals)
    whole = Mesh(verts, quads, weights)
    assert len(whole.v) == len(verts)
    ao = np.clip(whole.smooth(ao, 1), 0.0, 1.0) ** 0.9

    top4 = np.argsort(-weights, axis=1)[:, :4]
    w4 = np.take_along_axis(weights, top4, axis=1)
    w4 /= w4.sum(axis=1, keepdims=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, verts=verts.astype(np.float32), quads=quads.astype(np.uint16 if len(verts) < 65536 else np.int32),
        material=np.concatenate(material).astype(np.uint8), ao=(ao * 255.0 + 0.5).astype(np.uint8),
        bones=np.array(figure["names"]), parents=figure["parents"].astype(np.int16), heads=heads.astype(np.float32),
        skin_bones=top4.astype(np.uint8), skin_weights=(w4 * 65535.0 + 0.5).astype(np.uint16),
        texture_block=np.array(blocks, dtype=np.uint8), texel=np.array(texel, dtype=np.float32),
        **{f"texture{m}": t for m, t in enumerate(textures)})
    print(f"{out}: {len(verts)} vertices, {len(quads)} quads, {out.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
