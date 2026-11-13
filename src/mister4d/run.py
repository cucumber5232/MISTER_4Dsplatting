"""A trained take on disk: one checkpoint, or a sequence of segments.

A long take is trained in segments of a couple of seconds, each one a scene
of its own in a sub-folder; ``run.json`` lists them. Segments keep the memory
and the time of one training bounded whatever the length of the take, and an
interrupted job restarts from the first segment that is missing.

Everything that reads a result (evaluation, video, export) goes through
``Run`` and does not need to know how many segments there are.
"""

from __future__ import annotations

import bisect
import json
from collections import OrderedDict
from pathlib import Path

from .model import DynamicScene, load_checkpoint

MANIFEST = "run.json"
SEGMENT_FRAMES = 60  # default length of a segment: 2 s at 30 fps


def plan_segments(num_frames: int, segment_frames: int) -> list[tuple[int, int]]:
    """``[first, last)`` of every segment: equal lengths, as close as possible to ``segment_frames``.

    A take up to one and a half segments long stays in one piece.
    """
    if segment_frames <= 0 or num_frames <= segment_frames + segment_frames // 2:
        return [(0, num_frames)]
    count = max(1, round(num_frames / segment_frames))
    edges = [round(i * num_frames / count) for i in range(count + 1)]
    return [(edges[i], edges[i + 1]) for i in range(count)]


def segment_dir(run_dir: str | Path, index: int) -> Path:
    return Path(run_dir) / f"seg_{index:03d}"


def write_manifest(run_dir: str | Path, segments: list[dict], **extra) -> Path:
    path = Path(run_dir) / MANIFEST
    path.write_text(json.dumps({"format": "mister4d-run", "version": 1, "segments": segments, **extra}, indent=2))
    return path


class Run:
    """Read access to a trained take, segment by segment, with at most ``keep`` scenes in memory."""

    def __init__(self, run_dir: str | Path, device="cpu", keep: int = 2):
        self.dir = Path(run_dir)
        self.device = device
        self.keep = max(1, keep)
        self._open: OrderedDict[int, tuple[DynamicScene, dict]] = OrderedDict()
        manifest = self.dir / MANIFEST
        if manifest.exists():
            data = json.loads(manifest.read_text())
            if data.get("format") != "mister4d-run":
                raise ValueError(f"{manifest}: not a mister4d run")
            self.segments = [(self.dir / s["dir"], int(s["first"]), int(s["last"])) for s in data["segments"]]
            self._counts = [int(s.get("gaussians", 0)) for s in data["segments"]]
        elif (self.dir / "ckpt.pt").exists():
            self.segments = [(self.dir, 0, -1)]
            self._counts = [0]
        elif any(self.dir.glob("seg_*")):
            raise FileNotFoundError(f"{self.dir}: the training of this take did not finish; "
                                    "run 'mister4d train' again with the same options to continue it")
        else:
            raise FileNotFoundError(f"{self.dir}: no trained scene here (no {MANIFEST}, no ckpt.pt)")
        scene, ckpt = self._load(0)
        self.meta, self.config = ckpt["meta"], ckpt["config"]
        if self.segments[0][2] < 0:
            self.segments[0] = (self.dir, 0, len(self.meta["frame_ids"]))
            self._counts[0] = scene.num_gaussians
        self._starts = [s[1] for s in self.segments]

    @property
    def num_frames(self) -> int:
        return len(self.meta["frame_ids"])

    @property
    def static_frame(self) -> int | None:
        return self.meta.get("static_frame")

    @property
    def num_gaussians(self) -> int:
        return int(sum(self._counts))

    def index_of(self, frame: float) -> int:
        """The segment that describes ``frame``."""
        i = bisect.bisect_right(self._starts, float(frame)) - 1
        return min(max(i, 0), len(self.segments) - 1)

    def _load(self, index: int) -> tuple[DynamicScene, dict]:
        if index in self._open:
            self._open.move_to_end(index)
            return self._open[index]
        while len(self._open) >= self.keep:
            self._open.popitem(last=False)
        self._open[index] = load_checkpoint(self.segments[index][0] / "ckpt.pt", self.device)
        return self._open[index]

    def segment(self, index: int) -> tuple[DynamicScene, dict, int, int]:
        """Scene, checkpoint, first and last (exclusive) frame of one segment."""
        scene, ckpt = self._load(index)
        return scene, ckpt, self.segments[index][1], self.segments[index][2]

    def scene_at(self, frame: float) -> DynamicScene:
        return self._load(self.index_of(frame))[0]

    def time_offsets(self, frame: float) -> dict:
        """Learned sub-frame offset of every camera, for the segment of ``frame``."""
        return self._load(self.index_of(frame))[1]["meta"].get("time_offsets") or {}

    def is_cut(self, frame: int) -> bool:
        """True for the first frame of every segment but the first: where two trainings meet."""
        return frame in self._starts[1:]
