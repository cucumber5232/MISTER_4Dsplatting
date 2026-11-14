"""The reader for Unity (unity/Mister4D): what can be checked without Unity.

- the engine export holds the same bytes as the player;
- the C# that reads a chunk, chooses the Gaussians of an instant and sorts them
  is compiled with mono, when it is installed, and compared with the exporter;
- the projection of the shader is transcribed here line by line and compared
  with the pinhole camera the rest of the project uses.

The shader itself and the component are not compiled here: that takes Unity.
"""

import gzip
import json
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from mister4d.export import export_engine, export_web
from mister4d.trainer import TrainConfig, train

ROOT = Path(__file__).parent.parent
quiet = lambda *_: None  # noqa: E731


@pytest.fixture(scope="module")
def engine_export(tiny_dataset, tmp_path_factory):
    work = tmp_path_factory.mktemp("unity")
    cfg = TrainConfig(devices="cpu", iterations=30, views_per_step=3, densify_start=10, densify_every=10,
                      log_every=30, eval_every=30, segment_frames=2, segment_overlap=1)
    train(tiny_dataset, work / "run", cfg)
    return work / "run", export_engine(work / "run", work / "engine", log=quiet)


def _blocks(chunk, data):
    n = chunk["count"]
    a = np.frombuffer(data, dtype="<u2", count=4 * n).reshape(n, 4).astype(np.float64)
    b = np.frombuffer(data, dtype=np.uint8, count=4 * n, offset=8 * n).reshape(n, 4).astype(np.float64)
    d = np.frombuffer(data, dtype=np.uint8, count=4 * n, offset=16 * n).reshape(n, 4).astype(np.float64)
    return a, b, d


@pytest.mark.slow
def test_the_engine_export_holds_the_bytes_of_the_player(engine_export, tmp_path):
    run, folder = engine_export
    scene = json.loads((folder / "scene.json").read_text())
    assert scene["format"] == "mister4d-engine" and scene["version"] == 1 and len(scene["chunks"]) == 2
    web = json.loads((export_web(run, tmp_path / "web", log=quiet) / "scene.json").read_text())
    assert scene["alive_max"] == web["alive_max"] and scene["gaussians"] == web["gaussians"]

    def flat(value):  # what the simplest JSON readers of the engines accept
        if isinstance(value, dict):
            return all(flat(v) for v in value.values())
        if isinstance(value, list):
            return all(isinstance(v, (int, float)) for v in value) or all(isinstance(v, dict) and flat(v) for v in value)
        return isinstance(value, (int, float, str)) and value is not None

    assert flat(scene)
    import base64
    import re

    for mine, theirs in zip(scene["chunks"], web["chunks"]):
        data = gzip.decompress((folder / mine["file"]).read_bytes())
        assert len(data) == 23 * mine["count"] and mine["long_lived"] == theirs["long"]
        text = (tmp_path / "web" / theirs["file"]).read_text()
        assert gzip.decompress(base64.b64decode(re.search(r'"([^"]*)"', text).group(1))) == data


@pytest.mark.slow
def test_the_csharp_reader_chooses_and_sorts_like_the_exporter(engine_export, tmp_path):
    if not (shutil.which("mcs") and shutil.which("mono")):
        pytest.skip("mono is not installed: the C# reader cannot be compiled here")
    _, folder = engine_export
    exe = tmp_path / "reader.exe"
    subprocess.run(["mcs", f"-out:{exe}", str(ROOT / "tests" / "unity" / "ReaderCheck.cs"),
                    str(ROOT / "unity" / "Mister4D" / "Mister4DChunk.cs")], check=True, capture_output=True)
    scene = json.loads((folder / "scene.json").read_text())
    cases = [(0.0, 1.0, 0.0, 0.0), (1.0, 0.3, -0.9, 0.2), (2.0, -0.6, 0.5, -0.4), (4.0, 0.0, 0.0, 1.0), (3.0, 0.2, 0.7, 0.1)]
    for index, c in enumerate(scene["chunks"]):
        config = tmp_path / f"chunk{index}.txt"
        keys = ("count", "long_lived", "first", "last", "reach", "time_start", "time_span", "vel_max",
                "life_log_min", "life_log_max")
        config.write_text("\n".join(
            [f"{k}={c[k]}" for k in keys] + [f"file={folder / c['file']}", f"min_opacity={scene['min_opacity']}",
                                           "bbox_min=" + ",".join(map(str, c["bbox_min"])),
                                           "bbox_max=" + ",".join(map(str, c["bbox_max"]))]) + "\n")
        out = tmp_path / f"chunk{index}.out"
        subprocess.run(["mono", str(exe), str(config), str(out)] + [str(v) for case in cases for v in case], check=True)

        data = gzip.decompress((folder / c["file"]).read_bytes())
        n = c["count"]
        a, b, d = _blocks(c, data)
        lo, hi = np.array(c["bbox_min"]), np.array(c["bbox_max"])
        time = c["time_start"] + a[:, 3] / 65535.0 * c["time_span"]
        sigma = np.exp(c["life_log_min"] + (c["life_log_max"] - c["life_log_min"]) * b[:, 3] / 254.0)
        raw, at = out.read_bytes(), 0
        seen = 0
        for t, dx, dy, dz in cases:
            (m,) = struct.unpack_from("<i", raw, at)
            order = np.frombuffer(raw, dtype="<i4", count=m, offset=at + 4)
            at += 4 + 4 * m
            dt = t - time
            opacity = d[:, 3] / 255.0 * np.exp(-0.5 * (dt / sigma) ** 2)
            chosen = np.zeros(n, dtype=bool)
            chosen[order] = True
            assert len(set(order.tolist())) == m
            # Everything clearly visible is there, nothing clearly invisible is (at the threshold either is right).
            assert not np.any((opacity > scene["min_opacity"] * 1.02) & ~chosen)
            assert not np.any(chosen & ~(opacity > scene["min_opacity"] * 0.98))
            code = (b[:, :3] - 128.0) / 127.0
            position = lo + a[:, :3] / 65535.0 * (hi - lo) + code * np.abs(code) * c["vel_max"] * dt[:, None]
            along = (position @ np.array([dx, dy, dz]))[order]
            if m > 1:  # back to front, within one step of the 16-bit depth
                assert np.all(np.diff(along) <= (along.max() - along.min()) / 65535.0 * 1.01 + 1e-9)
            seen += m
        assert seen > 100


def _quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def test_the_projection_of_the_unity_shader_matches_the_reference_camera():
    """unity/Mister4D/Resources/MisterSplat4D.shader, line by line, against the OpenCV pinhole camera.

    The capture frame is right-handed with z up; Unity is left-handed with y
    up, its eye looks along -z and the image y points up. The same Gaussian
    seen by the same camera must land on the same pixels with the same shape.
    """
    rng = np.random.default_rng(3)
    width, height, fx, fy = 1832.0, 1920.0, 900.0, 880.0
    swap = np.array([[1.0, 0, 0], [0, 0, 1], [0, 1, 0]])  # capture -> Unity: y and z change place
    for _ in range(50):
        # A Gaussian in the capture frame.
        p = rng.uniform(-1, 1, 3)
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        s = np.exp(rng.uniform(-5, -2, 3))
        R = _quat_to_matrix(q)
        cov = (R * s) @ (R * s).T

        # Reference: a camera of the rig. x right, y down, z forward; pixels u right, v down.
        eye_c = rng.uniform(-1, 1, 3) + np.array([3.0, 0, 0.5])
        forward = -eye_c / np.linalg.norm(eye_c) + rng.normal(scale=0.05, size=3)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        Rc = np.stack([right, down, forward])
        X, Y, Z = Rc @ (p - eye_c)
        J = np.array([[fx / Z, 0, -fx * X / Z**2], [0, fy / Z, -fy * Y / Z**2]])
        ref_px = np.array([fx * X / Z, fy * Y / Z])  # from the principal point
        ref_cov = J @ Rc @ cov @ Rc.T @ J.T

        # The same camera in Unity, the object placed at the origin with no rotation.
        V = np.stack([swap @ right, swap @ -down, -(swap @ forward)])  # rows: right, up, back
        cam_u = swap @ eye_c
        P00, P11 = 2 * fx / width, 2 * fy / height

        # --- the vertex shader ---
        p_object = p[[0, 2, 1]]
        cov_object = np.array([[cov[0, 0], cov[0, 2], cov[0, 1]],
                               [cov[2, 0], cov[2, 2], cov[2, 1]],
                               [cov[1, 0], cov[1, 2], cov[1, 1]]])
        eye = V @ (p_object - cam_u)
        depth = -eye[2]
        to_eye = V @ np.eye(3)
        cov_eye = to_eye @ cov_object @ to_eye.T
        focal = 0.5 * np.array([width, height]) * np.array([P00, abs(P11)])
        ju = np.array([focal[0] / depth, 0.0, focal[0] * eye[0] / depth**2])
        jv = np.array([0.0, focal[1] / depth, focal[1] * eye[1] / depth**2])
        ca, cb, cc = ju @ cov_eye @ ju, ju @ cov_eye @ jv, jv @ cov_eye @ jv
        clip = np.array([P00 * eye[0], P11 * eye[1], 0.0, depth])  # x, y and w of the projection
        centre_px = clip[:2] / clip[3] * np.array([width, height]) / 2  # from the centre, y up
        # --- end of the vertex shader ---

        assert depth == pytest.approx(Z)
        assert centre_px == pytest.approx([ref_px[0], -ref_px[1]], abs=1e-6)
        # Same ellipse; only the sign of the mixed term changes, because y points the other way.
        assert [ca, cb, cc] == pytest.approx([ref_cov[0, 0], -ref_cov[0, 1], ref_cov[1, 1]], rel=1e-9, abs=1e-12)
