import shutil
import subprocess

import numpy as np
import pytest

from mister4d.ingest import collect_videos, ingest_clip
from mister4d.video import VideoWriter, probe_video

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

FPS, RATE = 30.0, 16000
OFFSETS = {"cam01": 0, "cam02": 3, "cam03": 1}  # frames each camera starts late
EVENT = 5  # global frame of the clap / flash


def _write_clip(folder, frames=24):
    from scipy.io import wavfile

    rng = np.random.default_rng(0)
    total = frames + max(OFFSETS.values())
    room = rng.normal(0, 0.02, int((total / FPS + 0.5) * RATE))
    start = int(EVENT / FPS * RATE)
    room[start:start + 800] += rng.normal(0, 0.8, 800) * np.exp(-np.arange(800) / 150.0)
    paths = []
    for name, off in OFFSETS.items():
        a0 = int(off / FPS * RATE)
        audio = room[a0:a0 + int((frames / FPS + 0.2) * RATE)] + rng.normal(0, 0.004, int((frames / FPS + 0.2) * RATE))
        wav = folder / f"{name}.wav"
        wavfile.write(str(wav), RATE, np.clip(audio * 32767, -32768, 32767).astype(np.int16))
        path = folder / f"{name}.mp4"
        with VideoWriter(path, 64, 36, FPS, audio=wav) as writer:
            for local in range(frames):
                g = local + off
                img = np.full((36, 64, 3), 60 + 2 * g, dtype=np.uint8)  # brightness encodes global time
                if g == EVENT:
                    img[:] = 250
                writer.write(img)
        paths.append(path)
    return paths


@pytest.mark.parametrize("method", ["audio", "flash"])
def test_sync_recovers_start_offsets(tmp_path, method):
    import cv2

    manifest = ingest_clip(_write_clip(tmp_path), tmp_path / "clip", sync=method, start=EVENT + 1,
                           log=lambda *_: None)
    shifts = {n: c["shift_frames"] for n, c in manifest["cameras"].items()}
    assert shifts == OFFSETS
    # After alignment the same index shows the same instant in every camera.
    for fid in manifest["frame_ids"][:6]:
        levels = [cv2.imread(str(tmp_path / "clip" / "frames" / n / f"{fid:06d}.jpg")).mean() for n in OFFSETS]
        assert max(levels) - min(levels) < 1.5


def test_dropped_frame_is_reported_not_renumbered(tmp_path):
    paths = _write_clip(tmp_path)
    broken = tmp_path / "broken" / "cam02.mp4"
    broken.parent.mkdir()
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(paths[1]), "-vf", r"select=not(eq(n\,9))",
           "-fps_mode", "passthrough", "-an", str(broken)]
    if subprocess.run(cmd, capture_output=True).returncode != 0:
        cmd[cmd.index("-fps_mode"):cmd.index("-fps_mode") + 2] = ["-vsync", "0"]
        subprocess.run(cmd, check=True, capture_output=True)
    assert probe_video(broken).missing_frames() == [9]

    manifest = ingest_clip([paths[0], broken, paths[2]], tmp_path / "clip",
                           manual_offsets="cam02=3,cam03=1", log=lambda *_: None)
    assert manifest["cameras"]["cam02"]["dropped_frames"] == [9]
    assert any("dropped frame" in w for w in manifest["warnings"])
    # Local frame 9 of cam02 is global instant 12: gone for every camera.
    first_global = max(OFFSETS.values())
    assert (12 - first_global) not in manifest["frame_ids"]
    for name in OFFSETS:
        files = sorted(p.stem for p in (tmp_path / "clip" / "frames" / name).iterdir())
        assert files == [f"{i:06d}" for i in manifest["frame_ids"]]


def test_different_frame_rates_are_rejected(tmp_path):
    a, b = tmp_path / "a.mp4", tmp_path / "b.mp4"
    for path, fps in ((a, 30.0), (b, 25.0)):
        with VideoWriter(path, 64, 36, fps) as writer:
            for _ in range(6):
                writer.write(np.zeros((36, 64, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="frame rate"):
        ingest_clip([a, b], tmp_path / "clip", log=lambda *_: None)


def test_clip_can_be_given_as_folder_or_wildcard(tmp_path):
    """Windows shells do not expand wildcards: the program does it itself."""
    paths = sorted(_write_clip(tmp_path, frames=12))
    assert collect_videos([tmp_path]) == paths  # the .wav files next to them are ignored
    assert collect_videos([str(tmp_path / "cam0*.mp4")]) == paths
    assert collect_videos(paths[:1]) == paths[:1]
    with pytest.raises(FileNotFoundError):
        collect_videos([tmp_path / "missing.mp4"])
    with pytest.raises(FileNotFoundError):
        collect_videos([str(tmp_path / "*.mov")])
    manifest = ingest_clip([tmp_path], tmp_path / "clip", manual_offsets="cam02=3,cam03=1",
                           log=lambda *_: None)
    assert sorted(manifest["cameras"]) == ["cam01", "cam02", "cam03"]
