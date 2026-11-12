"""Rig calibration with a ChArUco board (OpenCV).

Workflow implemented here:

1. detect the board in every frame of every camera;
2. intrinsics + distortion of each camera from its own detections;
3. relative poses between cameras from frames in which two cameras see the
   board at the same instant, chained along a spanning tree;
4. joint refinement of all camera poses and board poses (bundle adjustment);
5. definition of a metric world frame with +Z up.

The board gives the metric scale, so its squares must be measured on the
actual print.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

from .camera import Camera, Rig, distort_normalized, rig_center_and_up, rodrigues

MIN_CORNERS = 6


@dataclass
class BoardSpec:
    squares_x: int = 5
    squares_y: int = 7
    square_m: float = 0.12
    marker_m: float = 0.09
    dictionary: str = "DICT_4X4_50"

    @property
    def size_m(self) -> tuple[float, float]:
        return self.squares_x * self.square_m, self.squares_y * self.square_m

    def board(self) -> "cv2.aruco.CharucoBoard":
        if not hasattr(cv2.aruco, self.dictionary):
            raise ValueError(f"unknown ArUco dictionary '{self.dictionary}'")
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        return cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_m,
                                      self.marker_m, dictionary)

    def image(self, pixels_per_metre: float = 4000.0) -> np.ndarray:
        """Greyscale image of the board, exactly ``size_m`` large."""
        w, h = self.size_m
        size = (int(round(w * pixels_per_metre)), int(round(h * pixels_per_metre)))
        return self.board().generateImage(size, marginSize=0, borderBits=1)

    def corners_3d(self) -> np.ndarray:
        """Inner chessboard corners in board coordinates [K, 3]."""
        return np.asarray(self.board().getChessboardCorners(), dtype=np.float64).reshape(-1, 3)

    def to_dict(self) -> dict:
        return {"squares_x": self.squares_x, "squares_y": self.squares_y,
                "square_m": self.square_m, "marker_m": self.marker_m,
                "dictionary": self.dictionary}


class Detector:
    def __init__(self, spec: BoardSpec):
        self.spec = spec
        self._detector = cv2.aruco.CharucoDetector(spec.board())

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns corner ids [M] and pixel positions [M, 2] (possibly empty)."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = self._detector.detectBoard(gray)
        if ids is None or len(ids) < MIN_CORNERS:
            return np.zeros(0, dtype=np.int64), np.zeros((0, 2))
        return ids.reshape(-1).astype(np.int64), corners.reshape(-1, 2).astype(np.float64)


def detect_frames(frames_dir: str | Path, spec: BoardSpec, cameras: list[str] | None = None,
                  log=print) -> tuple[dict, dict]:
    """Detect the board in ``<frames_dir>/<cam>/<frame>.jpg``.

    Returns ``{cam: {frame_id: (ids, pixels)}}`` and ``{cam: (width, height)}``.
    """
    frames_dir = Path(frames_dir)
    cams = cameras or sorted(p.name for p in frames_dir.iterdir() if p.is_dir())
    detector = Detector(spec)
    detections, sizes = {}, {}
    for cam in cams:
        files = sorted(p for p in (frames_dir / cam).iterdir()
                       if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if not files:
            raise FileNotFoundError(f"no frames found in {frames_dir / cam}")
        found = {}
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            sizes.setdefault(cam, (img.shape[1], img.shape[0]))
            ids, pix = detector.detect(img)
            if len(ids):
                found[int(f.stem)] = (ids, pix)
        detections[cam] = found
        log(f"[calibrate] {cam}: board found in {len(found)}/{len(files)} frames")
    return detections, sizes


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------

DISTORTION_FLAGS = {
    "none": (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3),
    "radial1": cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3,
    "radial2": cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
    "full": 0,
}


def coverage_fraction(pixel_sets: list[np.ndarray], size: tuple[int, int], grid: int = 8) -> float:
    """Fraction of an 8x8 grid over the image that contains at least one corner."""
    hit = np.zeros((grid, grid), dtype=bool)
    for pix in pixel_sets:
        gx = np.clip((pix[:, 0] / size[0] * grid).astype(int), 0, grid - 1)
        gy = np.clip((pix[:, 1] / size[1] * grid).astype(int), 0, grid - 1)
        hit[gy, gx] = True
    return float(hit.mean())


def calibrate_intrinsics(found: dict, size: tuple[int, int], spec: BoardSpec,
                         distortion: str = "radial2", max_views: int = 60):
    """Returns K, dist(5), RMS reprojection error (px), coverage, number of views."""
    corners = spec.corners_3d()
    frames = sorted(found)
    if len(frames) < 5:
        raise RuntimeError(f"only {len(frames)} frames with the board: at least 5 are needed")
    if len(frames) > max_views:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, max_views).round().astype(int)]
    obj = [corners[found[f][0]].astype(np.float32) for f in frames]
    img = [found[f][1].astype(np.float32) for f in frames]
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, None, None,
                                             flags=DISTORTION_FLAGS[distortion])
    dist5 = np.zeros(5)
    flat = np.asarray(dist).reshape(-1)[:5]
    dist5[: len(flat)] = flat
    return K, dist5, float(rms), coverage_fraction(img, size), len(frames)


# ---------------------------------------------------------------------------
# Extrinsics
# ---------------------------------------------------------------------------


def _pose_matrix(rvec, tvec) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rodrigues(np.asarray(rvec).reshape(3))
    T[:3, 3] = np.asarray(tvec).reshape(3)
    return T


def _project(K, dist, rvec, tvec, pts) -> np.ndarray:
    pc = pts @ rodrigues(rvec).T + tvec
    xy = distort_normalized(pc[:, :2] / pc[:, 2:3], dist)
    return np.stack([K[0, 0] * xy[:, 0] + K[0, 2], K[1, 1] * xy[:, 1] + K[1, 2]], axis=-1)


def board_poses(found: dict, K, dist, spec: BoardSpec, max_error_px: float = 3.0) -> dict:
    """Board -> camera pose for every frame of one camera: {frame: 4x4}."""
    corners = spec.corners_3d()
    out = {}
    for frame, (ids, pix) in found.items():
        obj = corners[ids]
        ok, rvec, tvec = cv2.solvePnP(obj, pix, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            continue
        ok, rvec, tvec = cv2.solvePnP(obj, pix, K, dist, rvec, tvec, useExtrinsicGuess=True,
                                      flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        err = np.linalg.norm(_project(K, dist, rvec.reshape(3), tvec.reshape(3), obj) - pix, axis=1)
        if np.sqrt(np.mean(err**2)) <= max_error_px and float(tvec.reshape(3)[2]) > 0:
            out[frame] = _pose_matrix(rvec, tvec)
    return out


def _medoid(transforms: list[np.ndarray]) -> np.ndarray:
    """The transform closest to all the others (robust to a few bad estimates)."""
    if len(transforms) == 1:
        return transforms[0]
    cost = np.zeros(len(transforms))
    for i, A in enumerate(transforms):
        for B in transforms:
            cos = (np.trace(A[:3, :3].T @ B[:3, :3]) - 1.0) / 2.0
            cost[i] += np.arccos(np.clip(cos, -1, 1)) + np.linalg.norm(A[:3, 3] - B[:3, 3])
    return transforms[int(np.argmin(cost))]


def chain_cameras(poses: dict, reference: str) -> tuple[dict, dict]:
    """Camera poses relative to ``reference`` through shared board observations.

    ``poses`` is ``{cam: {frame: T_cam_board}}``. Returns ``{cam: T_cam_ref}``
    and the number of shared frames for every camera pair.
    """
    cams = list(poses)
    links = {}
    for i, a in enumerate(cams):
        for b in cams[i + 1:]:
            shared = sorted(set(poses[a]) & set(poses[b]))
            if shared:
                links[(a, b)] = shared
    result = {reference: np.eye(4)}
    while len(result) < len(cams):
        # Attach the camera with the strongest link to the already solved set.
        best = None
        for (a, b), shared in links.items():
            for known, new in ((a, b), (b, a)):
                if known in result and new not in result and (best is None or len(shared) > len(best[2])):
                    best = (known, new, shared)
        if best is None:
            missing = [c for c in cams if c not in result]
            raise RuntimeError(
                f"cameras {missing} never see the board together with an already linked camera: "
                "record more board poses visible from adjacent cameras at the same time")
        known, new, shared = best
        rel = _medoid([poses[new][f] @ np.linalg.inv(poses[known][f]) for f in shared])
        result[new] = rel @ result[known]
    return result, {f"{a}-{b}": len(s) for (a, b), s in links.items()}


def bundle_adjust(detections: dict, intr: dict, cam_T: dict, poses: dict, spec: BoardSpec,
                  reference: str, max_frames: int = 150, log=print):
    """Refine camera poses and board poses together. Returns refined poses and stats."""
    corners = spec.corners_3d()
    cams = list(cam_T)
    frames = sorted({f for c in cams for f in poses[c]})
    if len(frames) > max_frames:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, max_frames).round().astype(int)]
    free = [c for c in cams if c != reference]

    def to_vec(T):
        return np.concatenate([Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])

    # Initial board pose in the reference frame, from any camera that sees it.
    board0 = {}
    for f in frames:
        c = next(c for c in cams if f in poses[c])
        board0[f] = np.linalg.inv(cam_T[c]) @ poses[c][f]
    x0 = np.concatenate([to_vec(cam_T[c]) for c in free] + [to_vec(board0[f]) for f in frames])

    obs = []  # (camera index or -1 for the reference, frame index, corner ids, pixels)
    for c in cams:
        for fi, f in enumerate(frames):
            if f in poses[c]:
                ids, pix = detections[c][f]
                obs.append((free.index(c) if c != reference else -1, fi, c, ids, pix))
    n_res = 2 * sum(len(o[3]) for o in obs)
    n_cam = 6 * len(free)

    def residuals(x):
        out = np.empty(n_res)
        k = 0
        for ci, fi, c, ids, pix in obs:
            bp = x[n_cam + 6 * fi: n_cam + 6 * fi + 6]
            pts = corners[ids] @ rodrigues(bp[:3]).T + bp[3:]
            if ci >= 0:
                cp = x[6 * ci: 6 * ci + 6]
                pts = pts @ rodrigues(cp[:3]).T + cp[3:]
            K, dist = intr[c]
            xy = distort_normalized(pts[:, :2] / pts[:, 2:3], dist)
            proj = np.stack([K[0, 0] * xy[:, 0] + K[0, 2], K[1, 1] * xy[:, 1] + K[1, 2]], axis=-1)
            out[k: k + 2 * len(ids)] = (proj - pix).reshape(-1)
            k += 2 * len(ids)
        return out

    sparsity = lil_matrix((n_res, len(x0)), dtype=int)
    k = 0
    for ci, fi, _, ids, _ in obs:
        rows = slice(k, k + 2 * len(ids))
        sparsity[rows, n_cam + 6 * fi: n_cam + 6 * fi + 6] = 1
        if ci >= 0:
            sparsity[rows, 6 * ci: 6 * ci + 6] = 1
        k += 2 * len(ids)

    before = float(np.sqrt(np.mean(residuals(x0) ** 2)))
    sol = least_squares(residuals, x0, jac_sparsity=sparsity, loss="huber", f_scale=1.5,
                        x_scale="jac", method="trf", max_nfev=60)
    res = residuals(sol.x).reshape(-1, 2)
    after = float(np.sqrt(np.mean(res**2)))
    log(f"[calibrate] bundle adjustment: RMS {before:.3f} px -> {after:.3f} px "
        f"({len(frames)} board poses, {n_res // 2} corners)")

    refined = {reference: np.eye(4)}
    for i, c in enumerate(free):
        refined[c] = _pose_matrix(sol.x[6 * i: 6 * i + 3], sol.x[6 * i + 3: 6 * i + 6])
    boards = {f: _pose_matrix(sol.x[n_cam + 6 * i: n_cam + 6 * i + 3],
                              sol.x[n_cam + 6 * i + 3: n_cam + 6 * i + 6])
              for i, f in enumerate(frames)}
    per_cam, k = {}, 0
    for _, _, c, ids, _ in obs:
        per_cam.setdefault(c, []).append(res[k: k + len(ids)])
        k += len(ids)
    stats = {c: {"rms_px": float(np.sqrt(np.mean(np.concatenate(v) ** 2))),
                 "max_px": float(np.abs(np.concatenate(v)).max()),
                 "corners": int(sum(len(e) for e in v))} for c, v in per_cam.items()}
    return refined, boards, after, stats


# ---------------------------------------------------------------------------
# World frame
# ---------------------------------------------------------------------------


def world_from_rig(cam_T: dict, log=print) -> np.ndarray:
    """Reference -> world transform using only the rig geometry.

    Origin: the point the cameras look at. +Z: the vertical estimated from the
    camera orientations (cameras assumed mounted without roll). +X: towards the
    reference camera. The height of the floor is unknown in this mode.
    """
    tmp = [Camera("c", 1, 1, np.eye(3), R=T[:3, :3], t=T[:3, 3]) for T in cam_T.values()]
    center, up, mode = rig_center_and_up(tmp)
    log(f"[calibrate] world frame from the rig: origin at the point the cameras look at, "
        f"vertical from the camera orientations ({mode} mounting assumed, no roll). "
        "Use --floor-frame for a measured origin and floor plane.")
    to_ref = tmp[0].center - center
    x = to_ref - up * (to_ref @ up)
    x /= np.linalg.norm(x)
    y = np.cross(up, x)
    R = np.stack([x, y, up], axis=0)  # reference -> world rotation
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ center
    return T


def world_from_floor_board(board_T: np.ndarray, cam_T: dict, spec: BoardSpec) -> np.ndarray:
    """Reference -> world transform from a board lying on the floor.

    Origin: centre of the board. +Z: board normal, on the side of the cameras.
    """
    w, h = spec.size_m
    origin = board_T[:3, :3] @ np.array([w / 2, h / 2, 0.0]) + board_T[:3, 3]
    x, z = board_T[:3, 0], -board_T[:3, 2]  # printed side faces -z of the board
    centers = np.array([-T[:3, :3].T @ T[:3, 3] for T in cam_T.values()])
    if np.mean((centers - origin) @ z) < 0:
        z = -z
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=0)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ origin
    return T


def calibrate_rig(frames_dir: str | Path, spec: BoardSpec, reference: str | None = None,
                  distortion: str = "radial2", floor_frame: int | None = None,
                  intrinsics_from: Rig | None = None, log=print) -> Rig:
    """Full calibration from synchronized frames of a moving ChArUco board."""
    detections, sizes = detect_frames(frames_dir, spec, log=log)
    cams = list(detections)
    if len(cams) < 2:
        raise RuntimeError("at least two cameras are needed")
    reference = reference or cams[0]
    if reference not in cams:
        raise ValueError(f"reference camera '{reference}' not found (have {cams})")

    intr, info = {}, {}
    for c in cams:
        if intrinsics_from is not None and c in intrinsics_from.cameras:
            src = intrinsics_from.cameras[c]
            if (src.width, src.height) != sizes[c]:
                raise ValueError(f"{c}: provided intrinsics are for {src.width}x{src.height}, "
                                 f"frames are {sizes[c][0]}x{sizes[c][1]}")
            intr[c] = (src.K, src.dist)
            info[c] = {"intrinsics": "provided"}
            continue
        K, dist, rms, cover, views = calibrate_intrinsics(detections[c], sizes[c], spec, distortion)
        intr[c] = (K, dist)
        info[c] = {"intrinsics_rms_px": round(rms, 4), "intrinsics_views": views,
                   "image_coverage": round(cover, 3)}
        log(f"[calibrate] {c}: fx={K[0, 0]:.1f} fy={K[1, 1]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f} "
            f"k1={dist[0]:+.4f} rms={rms:.3f}px coverage={cover:.0%}")
        if cover < 0.3:
            log(f"[calibrate] WARNING {c}: the board covered only {cover:.0%} of the image; "
                "focal length and distortion are weakly constrained (move the board to the edges)")

    poses = {c: board_poses(detections[c], *intr[c], spec) for c in cams}
    cam_T, links = chain_cameras(poses, reference)
    cam_T, boards, rms, stats = bundle_adjust(detections, intr, cam_T, poses, spec, reference, log=log)

    if floor_frame is not None:
        if floor_frame not in boards:
            raise RuntimeError(f"the board was not detected in frame {floor_frame} (floor frame)")
        world_T = world_from_floor_board(boards[floor_frame], cam_T, spec)
        volume = {"center": [0.0, 0.0, 1.2], "size": [3.0, 3.0, 2.4]}
        mode = "floor_board"
    else:
        world_T = world_from_rig(cam_T, log=log)
        volume = {"center": [0.0, 0.0, 0.0], "size": [3.0, 3.0, 3.0]}
        mode = "rig"

    cameras = {}
    inv_world = np.linalg.inv(world_T)
    for c in cams:
        T = cam_T[c] @ inv_world  # world -> camera
        K, dist = intr[c]
        info[c].update({k: round(v, 4) if isinstance(v, float) else v for k, v in stats.get(c, {}).items()})
        cameras[c] = Camera(c, sizes[c][0], sizes[c][1], K, dist, T[:3, :3], T[:3, 3], info=info[c])
    report = {"board": spec.to_dict(), "reference_camera": reference, "world_mode": mode,
              "rms_px": round(rms, 4), "shared_frames": links,
              "opencv_version": cv2.__version__}
    return Rig(cameras, volume=volume, undistorted=False, report=report)
