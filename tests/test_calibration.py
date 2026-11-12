import numpy as np
import pytest

from mister4d.charuco import BoardSpec, calibrate_rig
from mister4d.synth import SynthConfig, write_calibration_frames


def _align(est, gt):
    """Rigid transform (no scale) taking the estimated camera centres onto the true ones."""
    a, b = est - est.mean(0), gt - gt.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    R = (u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt).T
    return R, gt.mean(0) - R @ est.mean(0)


@pytest.mark.slow
def test_charuco_calibration_recovers_the_rig(tmp_path):
    cfg = SynthConfig(num_cameras=3, width=800, height=450, calib_frames=60, arc_degrees=110.0)
    truth = write_calibration_frames(tmp_path, cfg)
    rig = calibrate_rig(tmp_path, BoardSpec(), log=lambda *_: None)

    assert rig.report["rms_px"] < 0.5
    est = np.array([rig.cameras[n].center for n in truth.names])
    gt = np.array([truth.cameras[n].center for n in truth.names])
    R, t = _align(est, gt)
    # The board fixes the metric scale: no scale factor is fitted here.
    assert np.linalg.norm(est @ R.T + t - gt, axis=1).max() < 0.03
    for name in truth.names:
        e, g = rig.cameras[name], truth.cameras[name]
        assert abs(e.K[0, 0] / g.K[0, 0] - 1) < 0.02
        assert abs(e.dist[0] - g.dist[0]) < 0.03
        rel = e.R @ R.T @ g.R.T
        angle = np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
        assert angle < 0.6
    # World frame from the rig: +Z is the true vertical, origin where the cameras look.
    assert np.allclose(R @ [0, 0, 1], [0, 0, 1], atol=0.02)
    assert np.linalg.norm(t - [0, 0, 1.0]) < 0.05


def test_disconnected_rig_is_reported():
    from mister4d.charuco import chain_cameras

    poses = {"a": {0: np.eye(4), 1: np.eye(4)}, "b": {1: np.eye(4)}, "c": {7: np.eye(4)}}
    with pytest.raises(RuntimeError, match="never see the board together"):
        chain_cameras(poses, "a")
