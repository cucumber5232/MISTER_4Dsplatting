"""GPU trials on the benchmark person: baseline, one change at a time, summaries for the repository.

Run it with the Python of .venv, from the folder of the main checkout:

    .venv\\Scripts\\python ..\\prove-gpu\\risultati\\prove_gpu.py                  every trial not yet done
    .venv\\Scripts\\python ..\\prove-gpu\\risultati\\prove_gpu.py --list           names of the trials
    .venv\\Scripts\\python ..\\prove-gpu\\risultati\\prove_gpu.py --only base      only these trials
    .venv\\Scripts\\python ..\\prove-gpu\\risultati\\prove_gpu.py --redo --only base

(on Linux: .venv/bin/python ../prove-gpu/risultati/prove_gpu.py)

Every trial trains the same dataset (scripts/make_benchmark.py, cameras ver01.. held out) with the
default options plus one change, measures it (scripts/quality_check.py, mister4d eval), draws the
enlarged contact sheet (scripts/contact_sheet.py) and copies summaries, images and a NOTE.md into
<results>/<trial>/, where <results> is the folder of this script unless --results says otherwise.
<results>/RIEPILOGO.md compares all the trials found there.

Data and checkpoints stay in the checkout (bench/, runs/, sessions/: ignored by git). Nothing is
committed or pushed from here. A trial already measured in <results> is skipped unless --redo is
given, so an interrupted batch continues where it stopped and a failed trial is tried again. Running
a trial again empties its run folder (runs/<trial>; sessions/persona for "catena") and replaces its
results; a run folder this script did not make is never emptied without --redo. --smoke runs
everything tiny, to test this script on a machine without a GPU: its numbers say nothing about
quality.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

HOLDOUT = ["ver01", "ver02", "ver03", "ver04", "ver05"]
MARK = ".prove_gpu"  # left in the run folders this script makes: it never empties any other
# What this script writes into the results folder of a trial (and may replace there).
OWN_FILES = ("NOTE.md", "errore.txt", "metrics.json", "summary.md", "eval_metrics.json", "train_summary.json",
             "config.yaml", "gaussiane.json", "sheet.jpg", "sheet.png", "viste.jpg", "viste.png", "viste_dettaglio.png")


@dataclass
class Trial:
    name: str
    what: str  # one line, in Italian: it goes into the documents
    sets: list[str] = field(default_factory=list)  # --set key=value
    frame_step: int = 1  # > 1: the visual hull is rebuilt every N frames, on a copy of the dataset
    devices: str = ""  # e.g. "cuda:0": train on these devices instead of all
    demo: bool = False  # the whole chain from the videos (mister4d demo --subject human --run)


TRIALS = [
    Trial("base", "opzioni di default, codice così com'è"),
    Trial("una_gpu", "come base, ma su una sola GPU (cuda:0): dice quanto conta la seconda", devices="cuda:0"),
    Trial("rumore", "come base, con un altro seme (seed=1): dice quanto oscillano le misure da sole", ["seed=1"]),
    Trial("catena", "catena intera dai video: mister4d demo --subject human --run", demo=True),
    Trial("prima", "com'era prima delle correzioni contro aghi e nebbia",
          ["lambda_needle=0", "min_scale_pixels=0", "prune_to_end=false"]),
    Trial("decay_0", "decadimento dell'opacità spento", ["opacity_decay=0"]),
    Trial("decay_0015", "decadimento dell'opacità 0,015", ["opacity_decay=0.015"]),
    Trial("entropia_0002", "entropia dell'opacità 0,002", ["lambda_opacity_entropy=0.002"]),
    Trial("entropia_0005", "entropia dell'opacità 0,005", ["lambda_opacity_entropy=0.005"]),
    Trial("entropia_001", "entropia dell'opacità 0,01", ["lambda_opacity_entropy=0.01"]),
    Trial("spread_0005", "spessore lungo il raggio 0,005", ["lambda_ray_spread=0.005"]),
    Trial("spread_002", "spessore lungo il raggio 0,02", ["lambda_ray_spread=0.02"]),
    Trial("dischi", "Gaussiane che partono come dischi: init_flatten 0,3 e lambda_flat 0,005",
          ["init_flatten=0.3", "lambda_flat=0.005"]),
    Trial("passo3", "nuvola di partenza ogni 3 frame (init --frame-step 3)", frame_step=3),
]

WARN = re.compile(r"warn|error|traceback|fail|exception|nan\b", re.IGNORECASE)
TAG = re.compile(r"^\[([a-z_]+)\]")


def it(value, digits: int = 2) -> str:
    """A number the way the documents of the project write it (decimal comma)."""
    if value is None:
        return "-"
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value, digits: int = 1) -> str:
    return "-" if value is None else it(100.0 * value, digits) + "%"


def show(cmd: list[str]) -> str:
    """A command as one would type it, with the interpreter written the short way."""
    parts = [str(c) for c in cmd]
    if parts and Path(parts[0]) == Path(sys.executable):
        parts[0] = ".venv\\Scripts\\python" if os.name == "nt" else ".venv/bin/python"
    return subprocess.list2cmdline(parts) if os.name == "nt" else shlex.join(parts)


class GpuMonitor(threading.Thread):
    """Samples nvidia-smi while a command runs: memory in use and load of every GPU."""

    QUERY = ["--query-gpu=index,name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]

    def __init__(self, period: float = 2.0):
        super().__init__(daemon=True)
        self.exe = shutil.which("nvidia-smi")
        self.period = period
        self.samples: dict[int, dict] = {}
        self._halt = threading.Event()

    @staticmethod
    def parse(text: str) -> list[tuple[int, str, float, float, float]]:
        rows = []
        for line in text.strip().splitlines():
            cells = [c.strip() for c in line.split(",")]
            if len(cells) != 5:
                continue
            try:
                rows.append((int(cells[0]), cells[1], float(cells[2]), float(cells[3]), float(cells[4])))
            except ValueError:  # "[N/A]" on some drivers
                continue
        return rows

    def sample(self) -> list[tuple[int, str, float, float, float]]:
        if not self.exe:
            return []
        try:
            out = subprocess.run([self.exe, *self.QUERY], capture_output=True, text=True, timeout=10).stdout
            return self.parse(out or "")
        except Exception:  # noqa: BLE001 - measuring must never stop a trial
            return []

    def run(self) -> None:
        while not self._halt.is_set():
            for index, name, load, used, total in self.sample():
                gpu = self.samples.setdefault(index, {"name": name, "total_mb": total, "load": [], "used_mb": []})
                gpu["load"].append(load)
                gpu["used_mb"].append(used)
            self._halt.wait(self.period)

    def stop(self) -> None:
        self._halt.set()
        self.join(timeout=15)

    def summary(self, idle: dict[int, float]) -> list[dict]:
        rows = []
        for index in sorted(self.samples):
            gpu = self.samples[index]
            if not gpu["load"]:
                continue
            rows.append({"index": index, "name": gpu["name"], "total_mb": gpu["total_mb"],
                         "peak_mb": max(gpu["used_mb"]), "idle_mb": idle.get(index),
                         "mean_load": sum(gpu["load"]) / len(gpu["load"]),
                         "busy_share": sum(v >= 20.0 for v in gpu["load"]) / len(gpu["load"]),
                         "samples": len(gpu["load"])})
        return rows


def run_logged(cmd: list[str], log_path: Path, cwd: Path, monitor: bool = False) -> dict:
    """Run a command, echo its output, keep it in ``log_path`` with the seconds since the start."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    watcher = GpuMonitor() if monitor else None
    idle = {row[0]: row[3] for row in watcher.sample()} if watcher else {}
    if watcher:
        watcher.start()
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    print(f"\n$ {show(cmd)}", flush=True)
    t0 = time.time()
    lines: list[tuple[float, str]] = []
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"$ {show(cmd)}\n")
        proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            stamp = time.time() - t0
            lines.append((stamp, line))
            log.write(f"{stamp:9.1f}  {line}\n")
            log.flush()
            print(line, flush=True)
        code = proc.wait()
    if watcher:
        watcher.stop()
    return {"cmd": show(cmd), "code": code, "seconds": time.time() - t0, "lines": lines,
            "gpus": watcher.summary(idle) if watcher else [], "log": log_path}


def steps_of(lines: list[tuple[float, str]]) -> list[tuple[str, float]]:
    """Seconds spent under every ``[tag]`` of a log, in order of appearance.

    The time between two tagged lines goes to the tag of the second one: the steps of the pipeline
    print a line when a piece of work is done (the trainer also while it works).
    """
    spans: dict[str, float] = {}
    last = 0.0
    for stamp, line in lines:
        m = TAG.match(line)
        if m:
            spans[m.group(1)] = spans.get(m.group(1), 0.0) + stamp - last
            last = stamp
    return list(spans.items())


def notices(lines: list[tuple[float, str]]) -> list[str]:
    """Warnings and errors of a log, each once."""
    seen: dict[str, None] = {}
    for _, line in lines:
        if WARN.search(line):
            seen.setdefault(line.strip()[:300], None)
    return list(seen)


def scene_stats(run_dir: Path) -> dict:
    """Opacity and shape of the Gaussians of a trained take, read on the CPU."""
    import torch

    from mister4d.run import Run

    take = Run(run_dir, "cpu")
    scene = take.scene_at(0)
    with torch.no_grad():
        peak = torch.sigmoid(scene.params["opacities"])
        out = {"gaussians_first_segment": int(peak.numel()),
               "peak_opacity_median": float(peak.median()),
               "peak_opacity_below_0.05": float((peak < 0.05).float().mean()),
               "peak_opacity_above_0.5": float((peak > 0.5).float().mean()),
               "duration_median_frames": float(torch.exp(scene.params["durations"]).median())}
        frames = sorted({0, take.num_frames // 2, take.num_frames - 1})
        if take.static_frame is not None:
            frames = [take.static_frame]
        alive, median, needles, thin = [], [], [], []
        for f in frames:
            state = take.scene_at(f).view().state_at(f)
            opacity = state["opacities"]
            if opacity.numel() == 0:
                continue
            ordered = state["scales"].sort(dim=-1, descending=True).values
            alive.append(int(opacity.numel()))
            median.append(float(opacity.median()))
            needles.append(float((ordered[:, 0] / ordered[:, 1].clamp(min=1e-9) > 3.0).float().mean()))
            thin.append(float(ordered[:, 2].median() * 1000.0))
        if alive:
            out.update({"frames_sampled": frames, "alive_per_frame": sum(alive) / len(alive),
                        "alive_opacity_median": sum(median) / len(median),
                        "alive_needle_share": sum(needles) / len(needles),
                        "smallest_axis_median_mm": sum(thin) / len(thin)})
    return out


def read_image(path: Path):
    """An image as OpenCV reads it; through NumPy, so that any folder name works on Windows too."""
    import cv2
    import numpy as np

    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def copy_sheet(src: Path, dst: Path, max_side: int, quality: int) -> float:
    """Copy a sheet as a JPEG at most ``max_side`` pixels long. Returns the scale used.

    At full size a contact sheet is a PNG of several megabytes: too much for a repository that
    should stay light. The original stays next to the run.
    """
    import cv2

    image = read_image(src)
    if image is None:
        raise FileNotFoundError(src)
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"cannot encode {src}")
    dst.write_bytes(data.tobytes())
    return scale


def detail_png(sheet: Path, dst: Path, rows: int, cols: int, side: int = 320) -> bool:
    """Full-size centre of two views of the middle row of a contact sheet: the outermost and one by a camera.

    The copies of the sheets are reduced; thin or faint Gaussians only show at full size, without
    the marks of JPEG.
    """
    import cv2
    import numpy as np

    image = read_image(sheet)
    if image is None or rows < 1 or cols < 1:
        return False
    tile_h, tile_w = image.shape[0] // rows, image.shape[1] // cols
    side = min(side, tile_h, tile_w)
    crops = []
    for col in sorted({0, cols // 2}):
        y0 = (rows // 2) * tile_h + (tile_h - side) // 2
        x0 = col * tile_w + (tile_w - side) // 2
        crops.append(image[y0:y0 + side, x0:x0 + side])
    ok, data = cv2.imencode(".png", np.concatenate(crops, axis=1), [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if ok:
        dst.write_bytes(data.tobytes())
    return bool(ok)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def git_head(repo: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, capture_output=True,
                              text=True, timeout=20).stdout.strip() or "?"
    except (OSError, subprocess.SubprocessError):
        return "?"


class Batch:
    def __init__(self, args):
        self.args = args
        self.repo = Path(args.repo).resolve()
        self.results = Path(args.results).resolve()
        self.data = (self.repo / args.data).resolve()
        self.runs = (self.repo / args.runs).resolve()
        self.py = sys.executable
        self.cli = [self.py, "-m", "mister4d.cli"]
        self.head = git_head(self.repo)
        if args.smoke:
            self.size = dict(width=160, height=90, frames=4)
        else:
            self.size = dict(width=args.width, height=args.height, frames=args.frames)

    # --- pieces -------------------------------------------------------------------------

    def rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.repo))
        except ValueError:
            return str(path)

    def environment(self) -> None:
        """AMBIENTE.md: what machine and what versions these trials ran on."""
        self.results.mkdir(parents=True, exist_ok=True)
        doctor = run_logged([*self.cli, "doctor"], self.runs / "doctor.log", self.repo)
        if doctor["code"] != 0:
            print(f"[prove] WARNING: 'mister4d doctor' reports {doctor['code']} problem(s): the trials may fail", flush=True)
        smi = shutil.which("nvidia-smi")
        gpus = ""
        if smi:
            gpus = subprocess.run([smi, "--query-gpu=index,name,driver_version,memory.total", "--format=csv"],
                                  capture_output=True, text=True).stdout.strip()
        text = ["# Ambiente delle prove", "",
                f"- data: {time.strftime('%Y-%m-%d %H:%M')}",
                f"- sistema: {platform.platform()}",
                f"- Python: {platform.python_version()}",
                f"- commit di main provato: `{self.head}`",
                f"- formato delle prove: {self.size['width']}x{self.size['height']}, {self.size['frames']} frame"
                + (" (PROVA RIDOTTA dello script: i numeri non dicono nulla sulla qualità)" if self.args.smoke else ""),
                "", "## mister4d doctor", "", "```", *[line for _, line in doctor["lines"]], "```", "",
                f"Codice di uscita: {doctor['code']} (0 = nessun problema).", ""]
        if gpus:
            text += ["## nvidia-smi", "", "```", gpus, "```", ""]
        else:
            text += ["`nvidia-smi` non trovato: nessuna misura della memoria GPU.", ""]
        (self.results / "AMBIENTE.md").write_text("\n".join(text), encoding="utf-8")

    def claim(self, folder: Path) -> None:
        """An empty run folder for a trial; a folder this script did not make is left alone."""
        if folder.exists() and any(folder.iterdir()) and not (folder / MARK).exists() and not self.args.redo:
            raise RuntimeError(f"{self.rel(folder)} exists and was not made by this script: "
                               "move it away, or pass --redo to replace it")
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / MARK).write_text("made by prove_gpu.py: emptied when the trial runs again\n", encoding="utf-8")

    def dataset(self) -> None:
        if (self.data / "init_points.ply").exists() and (self.data / "truth" / "cameras.json").exists():
            cams = read_json(self.data / "calibration.json").get("cameras", {})
            first = next(iter(cams.values()), {}) if isinstance(cams, dict) else {}
            have = (first.get("width"), first.get("height"), len(read_json(self.data / "meta.json").get("frame_ids", [])))
            want = (self.size["width"], self.size["height"], self.size["frames"])
            if have != want:
                raise SystemExit(f"{self.data} is {have[0]}x{have[1]} with {have[2]} frames, not {want[0]}x{want[1]} "
                                 f"with {want[2]}: remove it, or pass --data / --width --height --frames")
            print(f"[prove] dataset already there: {self.data}")
            return
        cmd = [self.py, "scripts/make_benchmark.py", "dataset", self.rel(self.data), "--width", self.size["width"],
               "--height", self.size["height"], "--frames", self.size["frames"]]
        if self.args.smoke:
            cmd += ["--points", 6000]
        res = run_logged(cmd, self.runs / "make_benchmark.log", self.repo)
        if res["code"] != 0:
            raise SystemExit("make_benchmark.py failed: see " + str(res["log"]))
        (self.results / "dataset.txt").write_text(
            f"$ {res['cmd']}\n{res['seconds']:.0f} s\n" + "\n".join(l for _, l in res["lines"] if l.startswith("[bench]")) + "\n",
            encoding="utf-8")

    def dataset_for(self, trial: Trial, notes: list[str]) -> Path:
        """The dataset of a trial: the common one, or a copy with its own starting points."""
        if trial.frame_step <= 1:
            return self.data
        copy = self.data.with_name(self.data.name + f"_passo{trial.frame_step}")
        if not copy.exists():
            shutil.copytree(self.data, copy, ignore=shutil.ignore_patterns("init_points.ply"))
        cmd = [*self.cli, "init", self.rel(copy), "--frame-step", trial.frame_step, "--exclude", *HOLDOUT]
        if self.args.smoke:
            cmd += ["--points", 6000]
        res = run_logged(cmd, self.runs / trial.name / "log_init.txt", self.repo)
        notes.append(f"Il dataset è una copia di `{self.rel(self.data)}` (`{self.rel(copy)}`), così le altre prove "
                     "non cambiano. Stesse immagini, stesse maschere.")
        notes.append(f"`{res['cmd']}` ({res['seconds']:.0f} s)")
        if res["code"] != 0:
            raise RuntimeError("init failed")
        return copy

    def measure(self, run_dir: Path, out: Path, commands: list[str], problems: list[str]) -> dict:
        """Quality check, enlarged contact sheet, mister4d eval; copies what belongs in the repository."""
        frames = int(read_json(run_dir / "train_summary.json").get("frames", 0))
        picked = sorted({0, frames // 2, max(frames - 1, 0)})
        steps = [
            ("quality", [self.py, "scripts/quality_check.py", self.rel(run_dir)]),
            ("sheet", [self.py, "scripts/contact_sheet.py", self.rel(run_dir), self.rel(run_dir / "viste.png"),
                       "--scale", 2, "--zoom", 2.2, "--width", 500, "--steps", 6,
                       "--frames", ",".join(str(f) for f in picked)]),
            ("eval", [*self.cli, "eval", self.rel(run_dir)]),
        ]
        if self.args.smoke:  # same commands, small pictures
            steps[1][1][steps[1][1].index("--width") + 1] = 120
        seconds = {}
        for name, cmd in steps:
            res = run_logged(cmd, run_dir / f"log_{name}.txt", self.repo)
            commands.append(res["cmd"])
            seconds[name] = res["seconds"]
            problems += [f"{name}: {n}" for n in notices(res["lines"])]
            if res["code"] != 0:
                problems.append(f"{name}: il comando è uscito con codice {res['code']} (vedi {self.rel(res['log'])})")
        for src, dst in ((run_dir / "evalq" / "metrics.json", "metrics.json"), (run_dir / "eval" / "summary.md", "summary.md"),
                         (run_dir / "eval" / "metrics.json", "eval_metrics.json"),
                         (run_dir / "train_summary.json", "train_summary.json"), (run_dir / "config.yaml", "config.yaml")):
            if src.exists():
                shutil.copyfile(src, out / dst)
        if (out / "eval_metrics.json").exists():  # the per-frame rows are long and not needed here
            full = read_json(out / "eval_metrics.json")
            for cam in full.get("per_camera", {}).values():
                cam.pop("frames", None)
            (out / "eval_metrics.json").write_text(json.dumps(full, indent=1), encoding="utf-8")
        images = {}
        for src, name in ((run_dir / "evalq" / "sheet.png", "sheet"), (run_dir / "viste.png", "viste")):
            if not src.exists():
                continue
            if self.args.png:  # the originals, whatever they weigh
                shutil.copyfile(src, out / f"{name}.png")
                images[f"{name}.png"] = 1.0
            else:
                images[f"{name}.jpg"] = copy_sheet(src, out / f"{name}.jpg", self.args.image_side, self.args.jpeg_quality)
        if (run_dir / "viste.png").exists() and not self.args.png and detail_png(
                run_dir / "viste.png", out / "viste_dettaglio.png", rows=len(picked), cols=6):
            images["viste_dettaglio.png"] = 1.0
        return {"seconds": seconds, "images": images}

    # --- one trial ----------------------------------------------------------------------

    def trial(self, trial: Trial) -> None:
        out = self.results / trial.name
        if (out / "NOTE.md").exists() and (out / "metrics.json").exists() and not self.args.redo:
            print(f"[prove] {trial.name}: already in {out} (use --redo to run it again)")
            return
        out.mkdir(parents=True, exist_ok=True)
        for name in OWN_FILES:  # of an earlier attempt
            (out / name).unlink(missing_ok=True)
        t0 = time.time()
        notes: list[str] = []
        problems: list[str] = []
        commands: list[str] = []
        train: dict = {}
        measured: dict = {"seconds": {}, "images": {}}
        steps: list[tuple[str, float]] = []
        run_dir = self.runs / trial.name
        try:
            if trial.demo:
                session = self.repo / "sessions" / "persona"
                run_dir = session / "run"
                cmd = [*self.cli, "demo", "--subject", "human", "--run", "--out", self.rel(session)]
                if self.args.smoke:
                    cmd += ["--frames", 14]
                self.claim(session)
                train = run_logged(cmd, self.runs / trial.name / "log_demo.txt", self.repo, monitor=True)
                steps = steps_of(train["lines"])
            else:
                self.claim(run_dir)
                data = self.dataset_for(trial, notes)
                cmd = [*self.cli, "train", self.rel(data), "--out", self.rel(run_dir), "--holdout", *HOLDOUT, "--fresh"]
                if self.args.iterations:
                    cmd += ["--iterations", self.args.iterations]
                devices = self.args.devices or (trial.devices if not self.args.smoke else "")
                if devices:
                    cmd += ["--devices", devices]
                sets = list(trial.sets) + (["densify_start=40", "densify_every=40", "eval_every=50", "log_every=25"]
                                           if self.args.smoke else [])
                if sets:
                    cmd += ["--set", *sets]
                train = run_logged(cmd, run_dir / "log_train.txt", self.repo, monitor=True)
            commands.append(train["cmd"])
            problems += [f"training: {n}" for n in notices(train["lines"])]
            if train["code"] != 0:
                raise RuntimeError(f"the command ended with code {train['code']}")
            measured = self.measure(run_dir, out, commands, problems)
        except Exception as exc:  # noqa: BLE001 - a failed trial must not stop the others
            problems.append(f"PROVA NON COMPLETATA: {type(exc).__name__}: {exc}")
            if train.get("lines"):
                (out / "errore.txt").write_text("\n".join(l for _, l in train["lines"][-60:]) + "\n", encoding="utf-8")
        stats = {}
        try:
            if (run_dir / "ckpt.pt").exists() or (run_dir / "run.json").exists():
                stats = scene_stats(run_dir)
                (out / "gaussiane.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"statistiche delle Gaussiane non calcolate: {type(exc).__name__}: {exc}")
        self.note(trial, out, run_dir, commands, train, measured, steps, stats, notes, problems, time.time() - t0)

    def note(self, trial, out, run_dir, commands, train, measured, steps, stats, notes, problems, total) -> None:
        summary = read_json(out / "train_summary.json")
        metrics = read_json(out / "metrics.json")
        log = [json.loads(l) for l in (run_dir / "train_log.jsonl").read_text().splitlines()] \
            if (run_dir / "train_log.jsonl").exists() else []
        text = [f"# {trial.name}", "", trial.what + ".", "",
                f"- data: {time.strftime('%Y-%m-%d %H:%M')}; commit di main: `{self.head}`"]
        if self.args.smoke:
            text.append("- **PROVA RIDOTTA dello script, senza GPU: i numeri non dicono nulla sulla qualità.**")
        text += ["", "## Comandi", "", "```", *commands, "```", "", "## Tempi", ""]
        if summary:
            iters, secs = summary.get("iterations", 0), summary.get("seconds", 0.0)
            text.append(f"- training: {iters} passi in {it(secs, 0)} s = {it(secs / 60.0, 1)} min, "
                        f"{it(secs / max(iters, 1), 3)} s per passo ({it(max(iters, 1) / max(secs, 1e-9), 1)} passi/s)")
            if len(log) > 3:
                half = [r for r in log if r["step"] >= iters // 2]
                if len(half) > 1 and half[-1]["step"] > half[0]["step"]:
                    rate = (half[-1]["seconds"] - half[0]["seconds"]) / (half[-1]["step"] - half[0]["step"])
                    text.append(f"- nella seconda metà (scena già cresciuta): {it(rate, 3)} s per passo")
            text.append(f"- dispositivi usati dal training: {', '.join(summary.get('devices', [])) or '-'}")
            text.append(f"- Gaussiane alla fine: {summary.get('gaussians', '-')}")
        if steps:
            text.append("- secondi per passo della catena, letti dal registro: "
                        + ", ".join(f"{tag} {it(sec, 0)}" for tag, sec in steps))
        if train:
            text.append(f"- comando di training completo: {it(train['seconds'] / 60.0, 1)} min")
        for name, sec in measured["seconds"].items():
            text.append(f"- {name}: {it(sec, 0)} s")
        text.append(f"- prova intera: {it(total / 60.0, 1)} min")
        text += ["", "## GPU", ""]
        gpus = train.get("gpus") or []
        if gpus:
            for g in gpus:
                idle = f", {it(g['idle_mb'], 0)} MiB prima di partire" if g["idle_mb"] is not None else ""
                text.append(f"- GPU {g['index']} ({g['name']}): picco {it(g['peak_mb'], 0)} MiB su {it(g['total_mb'], 0)}{idle}; "
                            f"carico medio {it(g['mean_load'], 0)}%, sopra il 20% nel {pct(g['busy_share'], 0)} dei campioni "
                            f"({g['samples']} campioni di nvidia-smi)")
            working = [g for g in gpus if g["busy_share"] >= 0.5]
            text.append(f"- GPU al lavoro per almeno metà del tempo: {len(working)} su {len(gpus)}")
        else:
            text.append("- nessuna misura: `nvidia-smi` non trovato (prova senza GPU).")
        text += ["", "## Misure", ""]
        if metrics:
            for group, label in (("verify", "camere di verifica"), ("train", "camere di training")):
                s = metrics.get(group, {})
                if s:
                    text.append(f"- {label}: PSNR soggetto {it(s.get('psnr_subject'))} dB, SSIM {it(s.get('ssim'), 3)}, "
                                f"IoU {it(s.get('iou'), 3)}, spill {pct(s.get('spill'))}, holes {pct(s.get('holes'))}, "
                                f"soft {pct(s.get('soft'))}, depth {it(s.get('depth_cm'), 1)} cm, thick {it(s.get('thick_cm'), 1)} cm, "
                                f"sfarfallio {it(s.get('flicker'))}")
            text.append(f"- floaters: {pct(metrics.get('floaters_any'))} (solo camere di training: {pct(metrics.get('floaters_train'))}); "
                        f"needles {pct(metrics.get('needle_share'))}; opaque {pct(metrics.get('opaque_share'))}; "
                        f"Gaussiane visibili per frame {it(metrics.get('visible_per_frame'), 0)}")
        else:
            text.append("- nessuna misura (vedi «Avvisi ed errori»).")
        if stats:
            text.append(f"- opacità mediana delle Gaussiane vive in un frame: {it(stats.get('alive_opacity_median'), 3)}; "
                        f"opacità di picco mediana: {it(stats.get('peak_opacity_median'), 3)}; "
                        f"con picco sotto 0,05: {pct(stats.get('peak_opacity_below_0.05'))}; "
                        f"ad ago tra tutte le vive: {pct(stats.get('alive_needle_share'))}; "
                        f"asse più corto mediano {it(stats.get('smallest_axis_median_mm'), 1)} mm; "
                        f"durata mediana {it(stats.get('duration_median_frames'), 2)} frame")
        text += ["", "## Immagini", ""]
        for name, scale in measured["images"].items():
            if name == "viste_dettaglio.png":
                text.append(f"- `{name}`: il centro di due viste di `viste.png` a grandezza piena, frame di mezzo: "
                            "la più esterna all'arco e una accanto a una camera")
            elif name.endswith(".jpg"):
                source = run_dir / ("evalq/sheet.png" if name.startswith("sheet") else "viste.png")
                text.append(f"- `{name}`: copia JPEG " + ("a grandezza piena" if scale >= 0.999 else
                                                           f"al {it(100 * scale, 0)}% della grandezza")
                            + f" di `{self.rel(source)}`")
            else:
                text.append(f"- `{name}`: l'originale")
        if not measured["images"]:
            text.append("- nessuna.")
        text += ["", "Giudizio a occhio: da scrivere dopo aver guardato le immagini.", ""]
        if notes:
            text += ["## Note", "", *[f"- {n}" for n in notes], ""]
        text += ["## Avvisi ed errori", ""]
        text += [f"- {p}" for p in problems] or ["- nessuno nei registri."]
        (out / "NOTE.md").write_text("\n".join(text) + "\n", encoding="utf-8")
        print(f"[prove] {trial.name}: {out / 'NOTE.md'}")

    # --- comparison ---------------------------------------------------------------------

    def compare(self) -> None:
        rows = []
        for trial in TRIALS:
            out = self.results / trial.name
            metrics, summary, stats = (read_json(out / n) for n in ("metrics.json", "train_summary.json", "gaussiane.json"))
            if not metrics:
                continue
            v, t = metrics.get("verify", {}), metrics.get("train", {})
            secs, iters = summary.get("seconds"), summary.get("iterations")
            rows.append("| " + " | ".join([
                f"[{trial.name}]({trial.name}/NOTE.md)", trial.what, str(summary.get("gaussians", metrics.get("gaussians", "-"))),
                it(secs / 60.0, 1) if secs else "-", it(secs / iters, 3) if secs and iters else "-",
                it(v.get("psnr_subject")), pct(v.get("spill")), pct(v.get("soft")), it(v.get("thick_cm"), 1),
                it(v.get("depth_cm"), 1), it(v.get("flicker")), pct(metrics.get("floaters_any")),
                pct(metrics.get("needle_share")), pct(metrics.get("opaque_share")), it(stats.get("alive_opacity_median"), 3),
                it(t.get("psnr_subject")), it(t.get("flicker"))]) + " |")
        head = ["# Riepilogo delle prove", "",
                f"Dataset: persona di prova, {self.size['width']}x{self.size['height']}, {self.size['frames']} frame, "
                "camere cam01..cam04 per il training, ver01..ver05 solo per la verifica.", ""]
        if self.args.smoke:
            head += ["**PROVA RIDOTTA dello script, senza GPU: i numeri non dicono nulla sulla qualità.**", ""]
        head += ["Le colonne da «PSNR» a «sfarfallio» sono misurate sulle camere di verifica. «Opacità mediana» è "
                 "quella delle Gaussiane vive in un frame. Le ultime due colonne sono le camere di training.", "",
                 "| prova | modifica | Gaussiane | min | s/passo | PSNR | spill | soft | thick cm | depth cm | sfarfallio "
                 "| floaters | needles | opaque | opacità mediana | PSNR training | sfarfallio training |",
                 "|" + "---|" * 17]
        (self.results / "RIEPILOGO.md").write_text("\n".join(head + rows) + "\n", encoding="utf-8")
        print(f"[prove] {self.results / 'RIEPILOGO.md'}: {len(rows)} trials")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=".", help="folder of the main checkout (default: the current folder)")
    ap.add_argument("--results", default=str(Path(__file__).resolve().parent),
                    help="where summaries and images are collected (default: the folder of this script)")
    ap.add_argument("--data", default="bench/D", help="benchmark dataset, created if missing")
    ap.add_argument("--runs", default="runs", help="folder of the training runs")
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--only", nargs="*", help="run only these trials")
    ap.add_argument("--skip", nargs="*", default=[], help="leave these trials out")
    ap.add_argument("--redo", action="store_true", help="run again the trials already in the results")
    ap.add_argument("--list", action="store_true", help="print the trials and stop")
    ap.add_argument("--iterations", type=int, help="training steps (default: those of the training itself)")
    ap.add_argument("--devices", help='e.g. "cuda:0" to train on one GPU only (default: all)')
    ap.add_argument("--image-side", type=int, default=1600,
                    help="longest side, in pixels, of the JPEG copies of the sheets put into the results")
    ap.add_argument("--jpeg-quality", type=int, default=85, help="quality of those copies")
    ap.add_argument("--png", action="store_true",
                    help="copy the original PNG sheets instead (several megabytes for every trial)")
    ap.add_argument("--smoke", action="store_true", help="tiny sizes and few steps: tests this script, not the quality")
    args = ap.parse_args()
    for stream in (sys.stdout, sys.stderr):  # a console that cannot show a character must not stop the batch
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    if args.list:
        for trial in TRIALS:
            extra = " ".join(trial.sets) or ("init --frame-step %d" % trial.frame_step if trial.frame_step > 1 else "")
            extra = extra or (f"--devices {trial.devices}" if trial.devices else "")
            print(f"{trial.name:15s} {trial.what}" + (f"  [{extra}]" if extra else ""))
        return 0
    names = {t.name for t in TRIALS}
    unknown = [n for n in (args.only or []) + args.skip if n not in names]
    if unknown:
        ap.error(f"unknown trials: {unknown} (see --list)")
    if not (Path(args.repo) / "scripts" / "make_benchmark.py").exists():
        ap.error(f"{Path(args.repo).resolve()} is not the checkout of the project: run this from its folder or pass --repo")
    if args.smoke and not args.iterations:
        args.iterations = 150

    batch = Batch(args)
    t0 = time.time()
    batch.environment()
    batch.dataset()
    for trial in TRIALS:
        if (args.only and trial.name not in args.only) or trial.name in args.skip:
            continue
        batch.trial(trial)
        batch.compare()
    batch.compare()
    print(f"[prove] done in {(time.time() - t0) / 60.0:.1f} min -> {batch.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
