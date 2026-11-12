import cv2
import numpy as np

from mister4d.camera import Camera, Rig, look_at, rig_center_and_up, rodrigues
from mister4d.prepare import Rectifier


def _camera(dist=(-0.08, 0.02, 0, 0, 0)):
    R, t = look_at([3.0, 1.0, 1.5], [0.0, 0.0, 1.0])
    K = [[600.0, 0, 479.5], [0, 600.0, 269.5], [0, 0, 1]]
    return Camera("cam", 960, 540, K, dist, R, t)


def test_look_at_points_the_camera_at_the_target():
    cam = _camera(dist=np.zeros(5))
    uv, depth = cam.project(np.array([[0.0, 0.0, 1.0]]))
    assert np.allclose(uv[0], [479.5, 269.5], atol=1e-6)
    assert depth[0] > 0
    assert np.allclose(cam.center, [3.0, 1.0, 1.5])
    # +Z of the world is "up" in the image: a higher point has a smaller v.
    assert cam.project(np.array([[0.0, 0.0, 1.5]]))[0][0, 1] < 269.5


def test_projection_matches_opencv():
    cam = _camera()
    pts = np.random.default_rng(0).uniform(-0.8, 0.8, (50, 3)) + [0, 0, 1.0]
    rvec, _ = cv2.Rodrigues(cam.R)
    expected, _ = cv2.projectPoints(pts, rvec, cam.t, cam.K, cam.dist)
    assert np.allclose(cam.project(pts)[0], expected.reshape(-1, 2), atol=1e-6)
    assert np.allclose(rodrigues(rvec.reshape(3)), cam.R, atol=1e-9)


def test_rig_roundtrip(tmp_path):
    rig = Rig({"cam": _camera()}, volume={"center": [0, 0, 1], "size": [2, 2, 2]})
    rig.save(tmp_path / "calibration.json")
    loaded = Rig.load(tmp_path / "calibration.json")
    assert np.allclose(loaded.cameras["cam"].K, rig.cameras["cam"].K)
    assert np.allclose(loaded.cameras["cam"].dist, rig.cameras["cam"].dist)
    assert loaded.volume == rig.volume and loaded.undistorted is False


def test_rectifier_keeps_intrinsics_consistent():
    """A 3D point lands where the undistorted, resized image actually shows it."""
    cam = _camera()
    pts = np.random.default_rng(1).uniform(-0.7, 0.7, (40, 3)) + [0, 0, 1.0]
    distorted_px = cam.project(pts)[0]
    for scale in (1.0, 2.0, 4.0):
        rect = Rectifier(cam, scale, already_undistorted=False)
        assert rect.camera.is_pinhole and (rect.camera.width, rect.camera.height) == (
            round(960 / scale), round(540 / scale))
        K_full, _ = cv2.getOptimalNewCameraMatrix(cam.K, cam.dist, (960, 540), 0)
        undist = cv2.undistortPoints(distorted_px.reshape(-1, 1, 2), cam.K, cam.dist, P=K_full).reshape(-1, 2)
        expected = (undist + 0.5) / scale - 0.5  # pixel centres at integer coordinates
        assert np.allclose(rect.camera.project(pts)[0], expected, atol=2e-2)


def test_rectifier_does_not_undistort_twice():
    cam = _camera()
    rect = Rectifier(cam, 1.0, already_undistorted=True)
    assert rect.maps is None and np.allclose(rect.camera.K, cam.K)


def test_vertical_from_rig_ignores_camera_pitch():
    cams = []
    for az, height in ((-1.0, 1.2), (-0.2, 2.2), (0.6, 0.9)):
        R, t = look_at([3 * np.cos(az), 3 * np.sin(az), height], [0, 0, 1.0])
        cams.append(Camera("c", 10, 10, np.eye(3), R=R, t=t))
    center, up, mode = rig_center_and_up(cams)
    assert mode == "landscape"
    assert np.allclose(up, [0, 0, 1], atol=1e-6)
    assert np.allclose(center, [0, 0, 1.0], atol=1e-6)
