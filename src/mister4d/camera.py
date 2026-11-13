"""Camera model and calibration file I/O.

Conventions used everywhere in this project (and written into every
calibration file, so that nobody has to guess):

- units: metres
- world: right-handed, +Z up
- camera frame: OpenCV (x right, y down, z forward)
- extrinsics: world -> camera, ``x_cam = R @ x_world + t``
- intrinsics ``K``: OpenCV pixel convention, the *centre* of the top-left
  pixel is (0, 0)
- distortion: OpenCV ``(k1, k2, p1, p2, k3)``
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONVENTIONS = {
    "units": "m",
    "world_up": "+Z",
    "camera_frame": "opencv (x right, y down, z forward)",
    "extrinsics": "world_to_camera: x_cam = R @ x_world + t",
    "pixel_origin": "centre of the top-left pixel is (0, 0)",
    "distortion": "opencv (k1, k2, p1, p2, k3)",
}


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Axis-angle vectors [..., 3] -> rotation matrices [..., 3, 3]."""
    rvec = np.asarray(rvec, dtype=np.float64)
    theta = np.linalg.norm(rvec, axis=-1, keepdims=True)
    k = rvec / np.maximum(theta, 1e-12)
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zero = np.zeros_like(kx)
    skew = np.stack(
        [zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], axis=-1
    ).reshape(rvec.shape[:-1] + (3, 3))
    theta = theta[..., None]
    eye = np.broadcast_to(np.eye(3), skew.shape)
    return eye + np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)


def distort_normalized(xy: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Apply OpenCV radial/tangential distortion to normalized coords [..., 2]."""
    k1, k2, p1, p2, k3 = (float(v) for v in np.asarray(dist).reshape(-1)[:5])
    x, y = xy[..., 0], xy[..., 1]
    r2 = x * x + y * y
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.stack([xd, yd], axis=-1)


def look_at(eye, target, up=(0.0, 0.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """World->camera (R, t) for a camera at ``eye`` looking at ``target``."""
    eye = np.asarray(eye, dtype=np.float64)
    z = np.asarray(target, dtype=np.float64) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, dtype=np.float64))
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    return R, -R @ eye


@dataclass
class Camera:
    name: str
    width: int
    height: int
    K: np.ndarray
    dist: np.ndarray = field(default_factory=lambda: np.zeros(5))
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.zeros(3))
    info: dict = field(default_factory=dict)

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        d = np.zeros(5)
        src = np.asarray(self.dist, dtype=np.float64).reshape(-1)[:5]
        d[: len(src)] = src
        self.dist = d
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def forward(self) -> np.ndarray:
        """Optical axis in world coordinates."""
        return self.R[2]

    @property
    def is_pinhole(self) -> bool:
        return bool(np.all(self.dist == 0.0))

    def world_to_cam(self) -> np.ndarray:
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def project(self, pts_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world points [N, 3] -> pixel coords [N, 2] and depths [N]."""
        pc = np.asarray(pts_world, dtype=np.float64) @ self.R.T + self.t
        z = pc[:, 2]
        safe = np.where(np.abs(z) < 1e-9, 1e-9, z)
        xy = pc[:, :2] / safe[:, None]
        if not self.is_pinhole:
            xy = distort_normalized(xy, self.dist)
        u = self.K[0, 0] * xy[:, 0] + self.K[0, 1] * xy[:, 1] + self.K[0, 2]
        v = self.K[1, 1] * xy[:, 1] + self.K[1, 2]
        return np.stack([u, v], axis=-1), z

    def to_dict(self) -> dict:
        out = {
            "width": int(self.width),
            "height": int(self.height),
            "K": self.K.tolist(),
            "dist": self.dist.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
        }
        if self.info:
            out["info"] = self.info
        return out

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "Camera":
        return cls(
            name=name,
            width=int(d["width"]),
            height=int(d["height"]),
            K=d["K"],
            dist=d.get("dist", [0.0] * 5),
            R=d.get("R", np.eye(3)),
            t=d.get("t", np.zeros(3)),
            info=d.get("info", {}),
        )


@dataclass
class Rig:
    """A set of calibrated cameras sharing one world frame."""

    cameras: dict[str, Camera]
    # Capture volume in world coordinates: {"center": [x, y, z], "size": [x, y, z]}
    volume: dict = field(default_factory=dict)
    # True once images have been undistorted: guards against double correction.
    undistorted: bool = False
    report: dict = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return list(self.cameras.keys())

    def save(self, path: str | Path) -> None:
        data = {
            "format": "mister4d-calibration",
            "version": 1,
            "conventions": CONVENTIONS,
            "undistorted": bool(self.undistorted),
            "volume": self.volume,
            "cameras": {n: c.to_dict() for n, c in self.cameras.items()},
            "report": self.report,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Rig":
        data = json.loads(Path(path).read_text())
        if data.get("format") != "mister4d-calibration":
            raise ValueError(f"{path}: not a mister4d calibration file")
        cams = {n: Camera.from_dict(n, d) for n, d in data["cameras"].items()}
        return cls(
            cameras=cams,
            volume=data.get("volume", {}),
            undistorted=bool(data.get("undistorted", False)),
            report=data.get("report", {}),
        )


def rig_center_and_up(cams: list[Camera]) -> tuple[np.ndarray, np.ndarray, str]:
    """Point closest to all optical axes, the vertical direction, and how it was found.

    The vertical is estimated assuming the cameras are mounted without roll:
    then one image axis of every camera is horizontal, and "up" is the
    direction perpendicular to all of them. Works for landscape and portrait
    mounting and does not depend on how much the cameras look down.
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for c in cams:
        d = c.forward / np.linalg.norm(c.forward)
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ c.center
    center = np.linalg.lstsq(A, b, rcond=None)[0]

    def spread(axis: int):
        vals, vecs = np.linalg.eigh(sum(np.outer(c.R[axis], c.R[axis]) for c in cams))
        return vals, vecs[:, 0]  # eigenvalues ascending; vecs[:, 0] = least covered direction

    (x_vals, x_normal), (y_vals, y_normal) = spread(0), spread(1)
    if x_vals[1] >= y_vals[1]:  # image x axes fan out horizontally: landscape
        up, mode = x_normal, "landscape"
        if up @ -np.mean([c.R[1] for c in cams], axis=0) < 0:
            up = -up
    else:  # image y axes fan out horizontally: cameras rotated by 90 degrees
        up, mode = y_normal, "portrait"
        if up @ np.mean([c.center - center for c in cams], axis=0) < 0:
            up = -up  # sign is a guess: cameras are assumed not lower than what they look at
    return center, up / np.linalg.norm(up), mode


def silhouette_coverage(cams: list[Camera], center) -> dict:
    """How well the outlines seen by these cameras pin down the shape of the subject.

    Each camera constrains the subject only across its line of sight. Seen from
    above, two cameras whose lines of sight are g degrees apart leave a shape
    that can be 1 / cos(g / 2) times wider than the real one when looked at
    from the middle of that gap (a round limb becomes a lozenge). Cameras
    facing each other count once: they see the same outline.

    Returns the largest gap between lines of sight, the two cameras around it
    and the corresponding worst-case widening factor.
    """
    center = np.asarray(center, dtype=np.float64)
    sight = []
    for c in cams:
        d = center - c.center
        angle = float(np.degrees(np.arctan2(d[1], d[0])) % 180.0)
        sight.append((0.0 if angle > 180.0 - 1e-6 else angle, c.name))
    sight.sort()
    worst = (-1.0, "", "")
    for i, (angle, name) in enumerate(sight):
        last = i + 1 == len(sight)
        nxt, other = sight[0] if last else sight[i + 1]
        gap = nxt + 180.0 - angle if last else nxt - angle  # the circle of lines of sight closes at 180
        if gap > worst[0]:
            worst = (gap, name, other)
    gap = min(worst[0], 180.0)
    widening = 1.0 / max(np.cos(np.radians(gap) / 2.0), 0.05)
    return {"largest_gap_deg": round(gap, 1), "between": [worst[1], worst[2]],
            "widening": round(float(widening), 3)}


def describe_coverage(cov: dict) -> str:
    return (f"largest gap between lines of sight {cov['largest_gap_deg']:.0f} deg "
            f"({cov['between'][0]} / {cov['between'][1]}): seen from the middle of it the subject "
            f"can look up to {(cov['widening'] - 1) * 100:.0f}% wider than it is")
