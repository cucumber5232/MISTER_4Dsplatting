"""Thin wrappers around the ffmpeg / ffprobe command line tools.

ffmpeg is used as an external program (no linking), so its licence does not
propagate to this code base.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np


class VideoError(RuntimeError):
    pass


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise VideoError(f"'{tool}' not found in PATH: install ffmpeg")
    return path


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, **kw)
    if proc.returncode != 0:
        tail = proc.stderr.decode(errors="replace")[-800:]
        raise VideoError(f"command failed: {' '.join(cmd[:6])} ...\n{tail}")
    return proc


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    fps_fraction: str
    codec: str
    pix_fmt: str
    duration: float
    has_audio: bool
    pts: np.ndarray  # presentation time of every frame, seconds, sorted

    @property
    def num_frames(self) -> int:
        return int(len(self.pts))

    def timeline(self) -> np.ndarray:
        """Index of each frame on a regular grid of step 1/fps.

        A dropped frame leaves a hole in the returned indices instead of being
        silently renumbered.
        """
        if self.num_frames == 0:
            return np.zeros(0, dtype=np.int64)
        return np.round((self.pts - self.pts[0]) * self.fps).astype(np.int64)

    def missing_frames(self) -> list[int]:
        idx = self.timeline()
        if len(idx) == 0:
            return []
        present = set(idx.tolist())
        return [i for i in range(int(idx[-1]) + 1) if i not in present]

    def jitter_ms(self) -> float:
        """Largest deviation of a frame time from the regular grid."""
        if self.num_frames < 2:
            return 0.0
        ideal = self.pts[0] + self.timeline() / self.fps
        return float(np.max(np.abs(self.pts - ideal)) * 1000.0)


def probe_video(path: str | Path) -> VideoInfo:
    ffprobe = _require("ffprobe")
    path = str(path)
    meta = json.loads(
        _run([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", path]).stdout
    )
    vstreams = [s for s in meta["streams"] if s.get("codec_type") == "video"]
    if not vstreams:
        raise VideoError(f"{path}: no video stream")
    v = vstreams[0]
    rate = Fraction(v.get("r_frame_rate", "0/1"))
    if rate == 0:
        rate = Fraction(v.get("avg_frame_rate", "0/1"))
    if rate == 0:
        raise VideoError(f"{path}: cannot determine the frame rate")
    out = _run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "packet=pts_time", "-of", "csv=p=0", path]
    ).stdout.decode()
    pts = []
    for line in out.splitlines():
        token = line.strip().strip(",")
        try:
            pts.append(float(token))
        except ValueError:
            continue
    pts = np.sort(np.asarray(pts, dtype=np.float64))
    return VideoInfo(
        path=path,
        width=int(v["width"]),
        height=int(v["height"]),
        fps=float(rate),
        fps_fraction=f"{rate.numerator}/{rate.denominator}",
        codec=v.get("codec_name", "?"),
        pix_fmt=v.get("pix_fmt", "?"),
        duration=float(meta.get("format", {}).get("duration", 0.0) or 0.0),
        has_audio=any(s.get("codec_type") == "audio" for s in meta["streams"]),
        pts=pts,
    )


def extract_frames(
    path: str | Path,
    out_dir: str | Path,
    first: int,
    last: int,
    ext: str = "jpg",
) -> list[Path]:
    """Decode frames ``first..last`` (decode order numbers) into ``out_dir``.

    No frame rate conversion is applied: one output file per decoded frame.
    """
    ffmpeg = _require("ffmpeg")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quality = ["-q:v", "1", "-qmin", "1"] if ext == "jpg" else []
    base = [ffmpeg, "-v", "error", "-y", "-i", str(path), "-an",
            "-vf", f"select=between(n\\,{first}\\,{last})"]
    tail = [*quality, "-start_number", "0", str(out_dir / f"%06d.{ext}")]
    try:
        _run([*base, "-fps_mode", "passthrough", *tail])
    except VideoError:
        _run([*base, "-vsync", "0", *tail])  # ffmpeg < 5.1
    return sorted(out_dir.glob(f"*.{ext}"))


def read_audio(path: str | Path, sample_rate: int = 16000, max_seconds: float | None = None) -> np.ndarray:
    """Mono audio track as float32 in [-1, 1]."""
    ffmpeg = _require("ffmpeg")
    cmd = [ffmpeg, "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate)]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-f", "s16le", "-"]
    raw = _run(cmd).stdout
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def frame_brightness(path: str | Path, max_seconds: float | None = None) -> np.ndarray:
    """Mean luma of every frame (cheap: decoded at 32x18)."""
    ffmpeg = _require("ffmpeg")
    cmd = [ffmpeg, "-v", "error", "-i", str(path), "-an"]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += ["-vf", "scale=32:18,format=gray", "-fps_mode", "passthrough", "-f", "rawvideo", "-"]
    try:
        raw = _run(cmd).stdout
    except VideoError:
        cmd[cmd.index("-fps_mode"):cmd.index("-fps_mode") + 2] = ["-vsync", "0"]
        raw = _run(cmd).stdout
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 18 * 32)
    return frames.mean(axis=1).astype(np.float32)


def _h264_available(ffmpeg: str) -> bool:
    out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True).stdout
    return b"libx264" in out


class VideoWriter:
    """Write uint8 frames to a video through an ffmpeg pipe.

    ``.mp4`` is H.264, for viewing. ``.mov`` is ProRes, the usual intermediate
    of editing software: 422 HQ, or 4444 with an alpha channel when ``alpha``
    is set (frames are then RGBA, colour not multiplied by alpha).
    """

    def __init__(self, path: str | Path, width: int, height: int, fps: float,
                 audio: str | Path | None = None, crf: int = 16, alpha: bool = False):
        ffmpeg = _require("ffmpeg")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        prores = Path(path).suffix.lower() == ".mov"
        if alpha and not prores:
            raise VideoError("a transparent background needs a .mov file (ProRes 4444) or a folder of PNG images")
        cmd = [ffmpeg, "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgba" if alpha else "rgb24",
               "-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
        if audio is not None:
            cmd += ["-i", str(audio), "-c:a", "aac", "-shortest"]
        if prores:
            cmd += ["-c:v", "prores_ks", "-profile:v", "4444" if alpha else "hq",
                    "-pix_fmt", "yuva444p10le" if alpha else "yuv422p10le", "-vendor", "apl0"]
        else:
            # libx264 needs even dimensions; pad by one pixel if necessary.
            cmd += ["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
            if _h264_available(ffmpeg):
                cmd += ["-c:v", "libx264", "-preset", "fast", "-crf", str(crf), "-pix_fmt", "yuv420p"]
            else:
                cmd += ["-c:v", "mpeg4", "-q:v", "2"]
        cmd += [str(path)]
        self._shape = (height, width, 4 if alpha else 3)
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        if frame.shape != self._shape or frame.dtype != np.uint8:
            raise ValueError(f"expected uint8 {self._shape}, got {frame.dtype} {frame.shape}")
        self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self._proc.stdin.close()
        err = self._proc.stderr.read()
        if self._proc.wait() != 0:
            raise VideoError(f"ffmpeg encode failed: {err.decode(errors='replace')[-500:]}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        if exc_type is None:
            self.close()
        else:
            self._proc.kill()
