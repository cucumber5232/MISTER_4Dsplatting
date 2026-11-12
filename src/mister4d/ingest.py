"""Video ingest: check the files, align the cameras in time, extract frames.

One call handles one *clip* (calibration, empty set or take): one video per
camera, the camera name being the file name without extension.

Design rules taken from the capture guide:

- frames are never aligned by arrival order: every frame gets an index on a
  common timeline derived from its presentation timestamp;
- a dropped frame leaves a hole in that index, it is reported and the instant
  is excluded for every camera instead of being renumbered away;
- the original files are only read, and their checksum is recorded.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from .video import (VideoInfo, extract_frames, frame_brightness, probe_video,
                    read_audio, sha256_file)

AUDIO_RATE = 16000
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".mxf", ".m4v", ".mts", ".webm"}


def collect_videos(inputs: list[str | Path]) -> list[Path]:
    """Video files named by ``inputs``: files, folders or wildcard patterns.

    Wildcards are expanded here because Windows shells pass them through as is.
    """
    import glob

    found: list[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            found += sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
        elif glob.has_magic(str(item)):
            found += sorted(Path(p) for p in glob.glob(str(item)))
        else:
            found.append(path)
    missing = [str(p) for p in found if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"video not found: {missing}")
    if not found:
        raise FileNotFoundError(f"no video files in {[str(i) for i in inputs]}")
    return found


def estimate_offsets_audio(videos: dict[str, Path], fps: float, max_seconds: float = 120.0):
    """Start offset of every camera from the cross-correlation of the audio tracks.

    Returns ``{cam: seconds}`` (how much later than the reference the camera
    started recording) and ``{cam: confidence}``. Typical use: a clap.
    """
    from scipy.signal import correlate, correlation_lags

    names = list(videos)
    tracks = {}
    for name in names:
        a = read_audio(videos[name], AUDIO_RATE, max_seconds)
        if len(a) < AUDIO_RATE // 4:
            raise RuntimeError(f"{videos[name]}: no usable audio track for audio sync")
        a = a - a.mean()
        tracks[name] = a / (np.abs(a).max() + 1e-9)
    ref = tracks[names[0]]
    offsets, confidence = {names[0]: 0.0}, {}
    for name in names[1:]:
        corr = correlate(ref, tracks[name], mode="full", method="fft")
        lags = correlation_lags(len(ref), len(tracks[name]), mode="full")
        peak = int(np.argmax(np.abs(corr)))
        # A sound at time t in the reference appears at t - offset in this camera.
        offsets[name] = float(lags[peak]) / AUDIO_RATE
        confidence[name] = float(np.abs(corr[peak]) / (np.median(np.abs(corr)) + 1e-12))
    return offsets, confidence


def estimate_offsets_flash(videos: dict[str, Path], fps: float, max_seconds: float = 120.0):
    """Start offsets from the frame with the largest jump in brightness (a flash)."""
    names = list(videos)
    event, confidence = {}, {}
    for name in names:
        luma = frame_brightness(videos[name], max_seconds)
        if len(luma) < 3:
            raise RuntimeError(f"{videos[name]}: too short for flash sync")
        jump = np.diff(luma)
        k = int(np.argmax(jump))
        event[name] = k + 1
        # How far the jump stands out from the ordinary frame-to-frame changes.
        typical = 1.4826 * np.median(np.abs(jump - np.median(jump)))
        confidence[name] = float(jump[k] / (typical + 0.5))
    ref = event[names[0]]
    return {n: (ref - event[n]) / fps for n in names}, confidence


def parse_manual_offsets(text: str, names: list[str]) -> dict[str, float]:
    """``cam02=3,cam03=-1`` -> offsets in frames (missing cameras: 0)."""
    out = {n: 0.0 for n in names}
    for item in filter(None, (s.strip() for s in text.split(","))):
        cam, value = item.split("=")
        if cam.strip() not in out:
            raise ValueError(f"offset given for unknown camera '{cam}' (have {names})")
        out[cam.strip()] = float(value)
    return out


def ingest_clip(video_paths: list[str | Path], out_dir: str | Path, sync: str = "none",
                manual_offsets: str | None = None, start: int = 0, end: int | None = None,
                step: int = 1, ext: str = "jpg", checksums: bool = True, log=print) -> dict:
    """Probe, synchronise and extract. Writes ``frames/<cam>/<index>.<ext>`` and ``ingest.json``.

    ``start``/``end``/``step`` select frames on the common timeline (after sync).
    """
    out_dir = Path(out_dir)
    video_paths = collect_videos(list(video_paths))
    videos = {Path(p).stem: Path(p) for p in sorted(map(str, video_paths))}
    if len(videos) != len(video_paths):
        raise ValueError("two videos share the same file name: camera names must be unique")
    names = list(videos)
    warnings: list[str] = []

    infos: dict[str, VideoInfo] = {}
    for name in names:
        infos[name] = probe_video(videos[name])
        v = infos[name]
        log(f"[ingest] {name}: {v.width}x{v.height} {v.fps:.3f} fps {v.codec} "
            f"{v.num_frames} frames, audio={'yes' if v.has_audio else 'no'}")
    ref = infos[names[0]]
    for name in names[1:]:
        v = infos[name]
        if abs(v.fps - ref.fps) > 1e-3:
            raise RuntimeError(f"{name}: frame rate {v.fps} differs from {names[0]} ({ref.fps}); "
                               "all cameras must record at the same rate")
        if (v.width, v.height) != (ref.width, ref.height):
            warnings.append(f"{name}: resolution {v.width}x{v.height} differs from "
                            f"{names[0]} ({ref.width}x{ref.height})")
    fps = ref.fps

    # --- time alignment ------------------------------------------------------
    confidence: dict[str, float] = {}
    if manual_offsets:
        frames_off = parse_manual_offsets(manual_offsets, names)
        seconds = {n: frames_off[n] / fps for n in names}
        sync = "manual"
    elif sync == "audio":
        seconds, confidence = estimate_offsets_audio(videos, fps)
    elif sync == "flash":
        seconds, confidence = estimate_offsets_flash(videos, fps)
    elif sync == "none":
        seconds = {n: 0.0 for n in names}
    else:
        raise ValueError(f"unknown sync method '{sync}'")
    shift = {n: int(round(seconds[n] * fps)) for n in names}
    residual_ms = {n: (seconds[n] * fps - shift[n]) / fps * 1000.0 for n in names}
    for n in names:
        if sync in ("audio", "flash") and confidence.get(n, 99.0) < 8.0:
            warnings.append(f"{n}: weak {sync} sync peak (confidence {confidence[n]:.1f}); "
                            "check the offset by eye")
        if abs(residual_ms[n]) > 0.25 * 1000.0 / fps:
            warnings.append(f"{n}: sub-frame sync error of {residual_ms[n]:+.1f} ms cannot be "
                            "removed in software (the exposures are not simultaneous)")
    if sync == "none":
        log("[ingest] no time alignment: assuming all cameras started on the same frame "
            "(hardware trigger / genlock)")

    # --- common timeline -----------------------------------------------------
    timelines = {n: infos[n].timeline() + shift[n] for n in names}
    for n in names:
        missing = infos[n].missing_frames()
        if missing:
            warnings.append(f"{n}: {len(missing)} dropped frame(s) at local index "
                            f"{missing[:10]}{'...' if len(missing) > 10 else ''}")
        jitter = infos[n].jitter_ms()
        if jitter > 0.2 * 1000.0 / fps:
            warnings.append(f"{n}: frame timestamps deviate up to {jitter:.1f} ms from a regular "
                            "grid (variable frame rate?)")
    first = max(int(t[0]) for t in timelines.values())
    last = min(int(t[-1]) for t in timelines.values())
    if last < first:
        raise RuntimeError("after time alignment the videos do not overlap")
    lo = first + max(start, 0)
    hi = last if end is None else min(last, first + end)
    wanted = [g for g in range(lo, hi + 1, max(step, 1))]
    present = {n: set(timelines[n].tolist()) for n in names}
    common = [g for g in wanted if all(g in present[n] for n in names)]
    dropped = len(wanted) - len(common)
    if dropped:
        warnings.append(f"{dropped} instant(s) excluded because at least one camera has no frame there")
    if not common:
        raise RuntimeError("no instant is covered by every camera")
    common_set = set(common)

    # --- extraction ----------------------------------------------------------
    cameras = {}
    for n in names:
        tl = timelines[n]
        keep = np.nonzero((tl >= common[0]) & (tl <= common[-1]))[0]
        tmp = out_dir / "frames" / f".tmp_{n}"
        if tmp.exists():
            shutil.rmtree(tmp)
        files = extract_frames(videos[n], tmp, int(keep[0]), int(keep[-1]), ext)
        if len(files) != len(keep):
            raise RuntimeError(f"{n}: expected {len(keep)} decoded frames, got {len(files)}")
        dest = out_dir / "frames" / n
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        for f, k in zip(files, keep):
            g = int(tl[k])
            if g in common_set:
                f.rename(dest / f"{g - common[0]:06d}.{ext}")
        shutil.rmtree(tmp)
        sample = cv2.imread(str(next(iter(sorted(dest.iterdir())))))
        v = infos[n]
        cameras[n] = {
            "file": str(videos[n]),
            "sha256": sha256_file(videos[n]) if checksums else None,
            "bytes": videos[n].stat().st_size,
            "codec": v.codec, "pix_fmt": v.pix_fmt, "fps": v.fps_fraction,
            "decoded_size": [int(sample.shape[1]), int(sample.shape[0])],
            "frames_in_file": v.num_frames,
            "dropped_frames": v.missing_frames(),
            "shift_frames": shift[n],
            "sync_residual_ms": round(residual_ms[n], 3),
            "sync_confidence": round(confidence[n], 2) if n in confidence else None,
        }
        log(f"[ingest] {n}: shift {shift[n]:+d} frames "
            f"(residual {residual_ms[n]:+.2f} ms), {len(common)} frames extracted")

    manifest = {
        "format": "mister4d-ingest", "version": 1, "fps": fps, "sync_method": sync,
        "frame_ids": [g - common[0] for g in common], "step": step,
        "cameras": cameras, "warnings": warnings,
    }
    (out_dir / "ingest.json").write_text(json.dumps(manifest, indent=2))
    for w in warnings:
        log(f"[ingest] WARNING {w}")
    return manifest


def read_ingest(clip_dir: str | Path) -> dict:
    path = Path(clip_dir) / "ingest.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: run 'mister4d ingest' first")
    return json.loads(path.read_text())
