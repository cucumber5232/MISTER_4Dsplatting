"""MISTER 4D Studio: the whole pipeline behind a guided page in the browser.

For operators who do not use a terminal. ``mister4d app`` (or the launcher in
the repository root) starts a small web server on this computer and opens the
page; nothing leaves the machine. The page walks through the same steps as the
command line, in the same order, and runs each of them as a sub-process of the
normal ``mister4d`` commands: there is one implementation of the pipeline, the
Studio only fills in the arguments and shows the outcome in plain words.

Only the Python standard library is used.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .ingest import VIDEO_EXTENSIONS

ASSETS = Path(__file__).parent / "studio"
THEME = Path(__file__).parent / "viewer" / "theme.css"
MEMORY = Path.home() / ".mister4d" / "studio.json"  # remembers the last project

STEPS = ["video", "calibrazione", "preparazione", "ricostruzione", "risultato"]
CLIPS = {"calib": "calibrazione", "empty": "vuoto", "take": "ripresa"}  # key -> work folder
DEFAULT_OPTIONS = {
    "sync": "none", "skip_frames": 0,
    "square_cm": 12.0, "marker_cm": 9.0, "squares_x": 5, "squares_y": 7,
    "dictionary": "DICT_4X4_50", "floor_frame": None,
    "downscale": 1, "mask_threshold": 24, "keep_shadows": False,
    "holdout": "", "learn_time_offsets": False, "iterations": None, "devices": "",
}

# What a failure usually means, in the operator's words. First match wins.
HINTS = [
    (r"never see the board together", "Alcune camere non vedono mai il pannello insieme a un'altra camera. "
     "Registra di nuovo il pannello facendolo passare davanti a camere vicine nello stesso momento."),
    (r"only \d+ frames with the board", "Il pannello è stato riconosciuto in troppo pochi fotogrammi. Controlla "
     "che sia a fuoco, ben illuminato e che le misure e il numero di riquadri indicati siano quelli giusti."),
    (r"frame rate .* differs", "Le camere non registrano tutte allo stesso numero di fotogrammi al secondo."),
    (r"no usable audio", "Un video non ha l'audio: scegli un altro tipo di sincronizzazione."),
    (r"do not overlap", "Dopo l'allineamento i video non hanno istanti in comune: controlla la sincronizzazione."),
    (r"not in the calibration file", "Nella ripresa c'è una camera che non compare nella calibrazione: "
     "i file devono avere lo stesso nome in tutte le cartelle."),
    (r"calibration was computed at", "La ripresa ha una risoluzione diversa da quella della calibrazione: "
     "vanno registrate con le stesse impostazioni."),
    (r"empty visual hull", "Le sagome delle camere non si incrociano. Controlla l'anteprima delle maschere "
     "e, se è sbagliata, la calibrazione."),
    (r"no clean plate|no empty-set frames", "Manca il set vuoto per una camera."),
    (r"gsplat CUDA backend failed", "La scheda grafica non riesce a eseguire il motore di rendering. "
     "Rilancia l'installazione e controlla i driver NVIDIA."),
    (r"out of memory", "La memoria della scheda grafica non basta: riduci la risoluzione di lavoro."),
    (r"No space left|not enough space", "Lo spazio sul disco è finito."),
    (r"ffmpeg.* not found|'ffprobe' not found", "FFmpeg non è installato su questo computer."),
    (r"no video files|video not found", "In una delle cartelle non ci sono video."),
]


def explain(lines: list[str]) -> str:
    text = "\n".join(lines[-40:])
    for pattern, hint in HINTS:
        if re.search(pattern, text, re.IGNORECASE):
            return hint
    return "Il passo si è interrotto. Il motivo tecnico è nei dettagli qui sotto."


def list_cameras(folder: str | os.PathLike | None) -> list[str]:
    """Camera names (file names without extension) of the videos in a folder."""
    if not folder or not Path(folder).is_dir():
        return []
    return sorted(p.stem for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def _finite(value):
    """NaN and infinity (a metric that could not be measured) are not JSON for a browser."""
    if isinstance(value, float) and value != value or value in (float("inf"), float("-inf")):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class Job:
    """A queue of steps running in the background, one command at a time."""

    def __init__(self, steps: list[str]):
        self.steps = steps
        self.step = steps[0]
        self.label = ""
        self.lines: deque[str] = deque(maxlen=400)
        self.progress: float | None = None
        self.detail = ""
        self.started = time.time()
        self.eta: float | None = None
        self.process: subprocess.Popen | None = None
        self.cancelled = False
        self._total = 0

    def read(self, line: str) -> None:
        self.lines.append(line)
        found = re.search(r"\[train\].*iterations=(\d+)", line)
        if found:
            self._total = int(found.group(1))
        found = re.search(r"\[train\] step=(\d+).*seconds=([\d.]+)", line)
        if found and self._total:
            step, seconds = int(found.group(1)), float(found.group(2))
            self.progress = min(1.0, (step + 1) / self._total)
            self.detail = f"Passo {step + 1} di {self._total}"
            if step > 0:
                self.eta = seconds / step * (self._total - step)

    def view(self) -> dict:
        return {"step": self.step, "label": self.label, "progress": self.progress, "detail": self.detail,
                "elapsed": time.time() - self.started, "eta": self.eta, "tail": list(self.lines)[-12:]}


class Studio:
    """State of the application: the open project, its options, the running job."""

    def __init__(self):
        self.lock = threading.RLock()
        self.project: Path | None = None
        self.videos = {"calib": "", "empty": "", "take": ""}
        self.options = dict(DEFAULT_OPTIONS)
        self.status = {s: "todo" for s in STEPS}
        self.problems: dict[str, str] = {}
        self.job: Job | None = None
        self.system: dict = {"checked": False}
        threading.Thread(target=self._check_system, daemon=True).start()
        last = (_read_json(MEMORY) or {}).get("last_project")
        if last and (Path(last) / "progetto.json").exists():
            self.open_project(last)

    # ----------------------------------------------------------- project ---

    def _check_system(self) -> None:
        info = {"checked": True, "ffmpeg": shutil.which("ffmpeg") is not None, "gpus": [], "error": ""}
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    info["gpus"].append({"name": props.name, "memory_gb": round(props.total_memory / 2**30)})
        except Exception as exc:  # noqa: BLE001 - shown to the user as is
            info["error"] = f"{type(exc).__name__}: {exc}"
        with self.lock:
            self.system = info

    def open_project(self, path: str) -> None:
        folder = Path(path).expanduser()
        if not folder.is_absolute():
            raise ValueError("Indica il percorso completo della cartella.")
        folder.mkdir(parents=True, exist_ok=True)
        with self.lock:
            if self.job:
                raise ValueError("Attendi la fine del passo in corso prima di cambiare progetto.")
            saved = _read_json(folder / "progetto.json") or {}
            self.project = folder
            self.videos = {**{"calib": "", "empty": "", "take": ""}, **saved.get("videos", {})}
            self.options = {**DEFAULT_OPTIONS, **saved.get("options", {})}
            self.status = {s: saved.get("status", {}).get(s, "todo") for s in STEPS}
            for step in STEPS:  # a job interrupted by closing the program never finished
                if self.status[step] == "running":
                    self.status[step] = "todo"
            self.problems = dict(saved.get("problems", {}))
            self.save()
        try:
            MEMORY.parent.mkdir(parents=True, exist_ok=True)
            MEMORY.write_text(json.dumps({"last_project": str(folder)}), encoding="utf-8")
        except OSError:
            pass

    def save(self) -> None:
        if self.project is None:
            return
        data = {"version": 1, "videos": self.videos, "options": self.options,
                "status": self.status, "problems": self.problems}
        (self.project / "progetto.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _invalidate_from(self, step: str) -> None:
        """Whatever was computed from ``step`` on is no longer up to date."""
        for later in STEPS[STEPS.index(step):]:
            if self.status[later] != "todo":
                self.status[later] = "todo"
            self.problems.pop(later, None)

    def set_videos(self, videos: dict) -> None:
        with self.lock:
            self._require_idle()
            changed = False
            for key in self.videos:
                if key in videos and str(videos[key]) != self.videos[key]:
                    self.videos[key] = str(videos[key])
                    changed = True
            if changed:
                self._invalidate_from("video")
                if self.options.get("holdout") not in list_cameras(self.videos["take"]):
                    self.options["holdout"] = ""
            self.save()

    # Options and the first step whose result they change.
    OPTION_STEP = {"sync": "video", "skip_frames": "video",
                   "square_cm": "calibrazione", "marker_cm": "calibrazione", "squares_x": "calibrazione",
                   "squares_y": "calibrazione", "dictionary": "calibrazione", "floor_frame": "calibrazione",
                   "downscale": "preparazione", "mask_threshold": "preparazione", "keep_shadows": "preparazione",
                   "holdout": "ricostruzione", "learn_time_offsets": "ricostruzione",
                   "iterations": "ricostruzione", "devices": "ricostruzione"}

    def set_options(self, options: dict) -> None:
        with self.lock:
            self._require_idle()
            first = None
            for key, value in options.items():
                if key in self.OPTION_STEP and self.options.get(key) != value:
                    self.options[key] = value
                    step = self.OPTION_STEP[key]
                    if first is None or STEPS.index(step) < STEPS.index(first):
                        first = step
            if first:
                self._invalidate_from(first)
            self.save()

    def _require_idle(self) -> None:
        if self.job:
            raise ValueError("Attendi la fine del passo in corso.")

    # ------------------------------------------------------------- paths ---

    @property
    def work(self) -> Path:
        return self.project / "lavoro"

    @property
    def data(self) -> Path:
        return self.project / "dati"

    @property
    def result(self) -> Path:
        return self.project / "risultato"

    def video_report(self) -> dict:
        cams = {key: list_cameras(self.videos[key]) for key in self.videos}
        problems = []
        names = {"calib": "pannello di calibrazione", "empty": "set vuoto", "take": "ripresa del soggetto"}
        chosen = [key for key in cams if cams[key]]
        if len(chosen) == 3:
            reference = set(cams["take"])
            for key in ("calib", "empty"):
                missing = sorted(reference - set(cams[key]))
                if missing:
                    problems.append(f"Nel {names[key]} manca: {', '.join(missing)}. I file devono avere "
                                    "lo stesso nome in tutte e tre le cartelle.")
            if len(reference) < 2:
                problems.append("Servono almeno due camere (tre o più per un risultato utile).")
        return {"folders": dict(self.videos), "cameras": cams, "problems": problems,
                "ready": len(chosen) == 3 and not problems}

    # ---------------------------------------------------------- commands ---

    @property
    def reduced(self) -> bool:
        """No GPU on this computer: only a reduced reconstruction is feasible."""
        return bool(self.system.get("checked")) and not self.system.get("gpus")

    def commands(self, step: str) -> list[tuple[str, list[str]]]:
        """What a step runs: (description for the operator, command line)."""
        o, cli = self.options, [sys.executable, "-u", "-m", "mister4d.cli"]
        work, data, run = self.work, self.data, self.result / "run"
        if step == "esempio":
            return [("Creo le riprese di esempio (un manichino animato visto da 4 camere)",
                     cli + ["demo", "--out", str(self.project / "esempio")])]
        if step == "video":
            if not self.video_report()["ready"]:
                raise ValueError("Indica prima le tre cartelle con i video.")
            sync = ["--sync", str(o["sync"]), "--start", str(int(o["skip_frames"] or 0))]
            return [
                ("Controllo i video del pannello ed estraggo i fotogrammi",
                 cli + ["ingest", self.videos["calib"], "--out", str(work / "calibrazione")] + sync),
                ("Controllo i video del set vuoto ed estraggo i fotogrammi",
                 cli + ["ingest", self.videos["empty"], "--out", str(work / "vuoto")]),
                ("Controllo i video della ripresa ed estraggo i fotogrammi",
                 cli + ["ingest", self.videos["take"], "--out", str(work / "ripresa")] + sync),
            ]
        if step == "calibrazione":
            args = ["calibrate", str(work / "calibrazione"), "--out", str(work / "calibrazione.json"),
                    "--square", str(float(o["square_cm"]) / 100), "--marker", str(float(o["marker_cm"]) / 100),
                    "--squares-x", str(int(o["squares_x"])), "--squares-y", str(int(o["squares_y"])),
                    "--dictionary", str(o["dictionary"])]
            if o.get("floor_frame") not in (None, ""):
                args += ["--floor-frame", str(int(o["floor_frame"]))]
            return [("Cerco il pannello nei fotogrammi e calcolo ottica e posizione di ogni camera", cli + args)]
        if step == "preparazione":
            masks = ["masks", str(data), "--threshold", str(float(o["mask_threshold"]))]
            if o.get("keep_shadows"):
                masks.append("--keep-shadows")
            return [
                ("Correggo la distorsione delle ottiche e preparo le immagini",
                 cli + ["prepare", str(work / "ripresa"), "--calibration", str(work / "calibrazione.json"),
                        "--empty", str(work / "vuoto"), "--out", str(data), "--downscale", str(o["downscale"])]),
                ("Separo il soggetto dallo sfondo", cli + masks),
            ]
        if step == "ricostruzione":
            init = ["init", str(data)]
            train = ["train", str(data), "--out", str(run)]
            overrides = []
            if o.get("holdout"):
                init += ["--exclude", str(o["holdout"])]
                train += ["--holdout", str(o["holdout"])]
            if o.get("learn_time_offsets"):
                overrides.append("learn_time_offsets=true")
            iterations, devices = o.get("iterations"), o.get("devices")
            if self.reduced:  # the settings of ``mister4d demo`` without a GPU: coarse, but it ends
                init += ["--points", "60000"]
                iterations, devices = iterations or 1500, devices or "cpu"
                overrides += ["views_per_step=3", "densify_start=400", "max_gaussians=80000", "eval_every=500"]
            if iterations:
                train += ["--iterations", str(int(iterations))]
            if devices:
                train += ["--devices", str(devices)]
            if overrides:
                train += ["--set"] + overrides
            return [("Ricavo la forma di partenza dalle sagome", cli + init),
                    ("Ottimizzo la scena 4D", cli + train)]
        if step == "risultato":
            cmds = []
            if (_read_json(run / "train_summary.json") or {}).get("holdout_cameras"):
                cmds.append(("Misuro la qualità dalla camera di verifica", cli + ["eval", str(run)]))
            scale = ["--scale", "2"] if self.reduced else []  # the reduced images are small
            cmds += [("Creo il video dimostrativo",
                      cli + ["render", str(run), "--out", str(self.result / "orbita.mp4"), "--seconds", "8"] + scale),
                     ("Creo il player", cli + ["export", str(run), "--out", str(self.result / "player.html")])]
            return cmds
        raise ValueError(f"passo sconosciuto: {step}")

    # --------------------------------------------------------------- jobs ---

    def start(self, steps: list[str]) -> None:
        with self.lock:
            if self.project is None:
                raise ValueError("Scegli prima la cartella del progetto.")
            self._require_idle()
            if not steps or any(step not in STEPS + ["esempio"] for step in steps):
                raise ValueError("Passo sconosciuto.")
            for step in steps:
                if step in STEPS:
                    before = STEPS[:STEPS.index(step)]
                    undone = [s for s in before if self.status[s] != "done" and s not in steps]
                    if undone:
                        raise ValueError("Prima vanno completati i passi precedenti.")
            self.commands(steps[0])  # fails now, with a readable message, if something is missing
            self.job = Job(steps)
            if steps[0] in STEPS:  # the answer to this very request already says "running"
                self._invalidate_from(steps[0])
                self.status[steps[0]] = "running"
            threading.Thread(target=self._run, args=(self.job,), daemon=True).start()

    def _run(self, job: Job) -> None:
        log_dir = self.project / "registro"
        log_dir.mkdir(exist_ok=True)
        for step in job.steps:
            with self.lock:
                job.step, job.progress, job.detail, job.eta = step, None, "", None
                job.started = time.time()
                job.lines.clear()
                if step in STEPS:
                    self._invalidate_from(step)
                    self.status[step] = "running"
                    self.save()
                else:
                    self.problems.pop(step, None)
            ok = True
            try:
                with open(log_dir / f"{step}.log", "w", encoding="utf-8") as log:
                    for label, argv in self.commands(step):
                        job.label, job.progress, job.detail, job.eta = label, None, "", None
                        log.write(f"$ {' '.join(argv[3:])}\n")
                        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                        job.process = subprocess.Popen(
                            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, creationflags=flags,
                            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
                        for line in job.process.stdout:
                            line = line.rstrip()
                            log.write(line + "\n")
                            log.flush()
                            job.read(line)
                        if job.process.wait() != 0 or job.cancelled:
                            ok = False
                            break
            except (OSError, ValueError) as exc:
                job.lines.append(f"error: {exc}")
                ok = False
            with self.lock:
                if step in STEPS:
                    self.status[step] = "done" if ok else ("todo" if job.cancelled else "error")
                    if not ok and not job.cancelled:
                        self.problems[step] = explain(list(job.lines))
                elif step == "esempio" and ok:
                    raw = self.project / "esempio" / "raw"
                    self.videos = {"calib": str(raw / "calib"), "empty": str(raw / "empty"), "take": str(raw / "take")}
                    self.options.update(sync="audio", skip_frames=8)
                    self._invalidate_from("video")
                elif step == "esempio" and not job.cancelled:
                    self.problems["esempio"] = explain(list(job.lines))
                self.save()
            if not ok:
                break
        with self.lock:
            self.job = None

    def cancel(self) -> None:
        with self.lock:
            if self.job and self.job.process and self.job.process.poll() is None:
                self.job.cancelled = True
                self.job.process.terminate()

    # ---------------------------------------------------------- summaries ---

    def summary(self, step: str) -> dict | None:
        """What a finished step produced, read back from its files."""
        try:
            if step == "video":
                clips = {}
                for key, folder in CLIPS.items():
                    manifest = _read_json(self.work / folder / "ingest.json")
                    if not manifest:
                        return None
                    cams = manifest["cameras"]
                    clips[key] = {
                        "frames": len(manifest["frame_ids"]), "fps": manifest["fps"],
                        "cameras": {n: {"shift": c["shift_frames"], "residual_ms": c["sync_residual_ms"],
                                        "dropped": len(c["dropped_frames"])} for n, c in cams.items()},
                    }
                return {"clips": clips}
            if step == "calibrazione":
                calib = _read_json(self.work / "calibrazione.json")
                report = calib["report"]
                return {"rms_px": report["rms_px"], "coverage": report.get("coverage"),
                        "cameras": {n: c.get("info", {}).get("rms_px") for n, c in calib["cameras"].items()}}
            if step == "preparazione":
                meta = _read_json(self.data / "meta.json")
                calib = _read_json(self.data / "calibration.json")
                first = next(iter(calib["cameras"].values()))
                preview = self.data / "masks_preview.jpg"
                return {"frames": len(meta["frame_ids"]), "cameras": meta["cameras"],
                        "size": [first["width"], first["height"]],
                        "preview": f"/project/dati/masks_preview.jpg?v={int(preview.stat().st_mtime)}"}
            if step == "ricostruzione":
                return _read_json(self.result / "run" / "train_summary.json")
            if step == "risultato":
                metrics = _read_json(self.result / "run" / "eval" / "metrics.json")
                out = {"player": (self.result / "player.html").exists(),
                       "video": (self.result / "orbita.mp4").exists(), "quality": None}
                if out["video"]:
                    out["video_url"] = f"/project/risultato/orbita.mp4?v={int((self.result / 'orbita.mp4').stat().st_mtime)}"
                if metrics and metrics.get("holdout_cameras"):
                    out["quality"] = {"cameras": metrics["holdout_cameras"], **metrics["holdout"]}
                return out
        except (OSError, KeyError, TypeError, StopIteration):
            return None
        return None

    def state(self) -> dict:
        with self.lock:
            project = None
            if self.project is not None:
                free = shutil.disk_usage(self.project).free / 1e9
                project = {"path": str(self.project), "name": self.project.name, "free_gb": round(free)}
            steps = []
            for i, step in enumerate(STEPS):
                status = self.status[step]
                blocked = self.project is None or any(self.status[s] != "done" for s in STEPS[:i])
                steps.append({"id": step, "status": status, "blocked": blocked and status != "running",
                              "problem": self.problems.get(step, ""),
                              "summary": self.summary(step) if status == "done" else None})
            return {
                "project": project, "system": {**self.system, "reduced": self.reduced},
                "videos": self.video_report() if self.project else None,
                "options": self.options, "steps": steps,
                "example_problem": self.problems.get("esempio", ""),
                "job": self.job.view() if self.job else None,
            }

    def log(self, step: str) -> str:
        if self.project is None or step not in STEPS + ["esempio"]:
            return ""
        try:
            return (self.project / "registro" / f"{step}.log").read_text(encoding="utf-8", errors="replace")[-20000:]
        except OSError:
            return ""


def pick_folder(title: str) -> str | None:
    """Native "choose a folder" dialog. Runs in its own process: Tk wants a main thread."""
    code = ("import tkinter, tkinter.filedialog as d, sys\n"
            "r = tkinter.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
            "print(d.askdirectory(title=sys.argv[1], mustexist=False) or '')\n")
    try:
        done = subprocess.run([sys.executable, "-c", code, title], capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None  # no display or no Tk: the page falls back to typing the path
    return done.stdout.strip()


def open_in_file_manager(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - a folder of the project, opened in Explorer
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def make_handler(studio: Studio, token: str):
    class Handler(BaseHTTPRequestHandler):
        server_version = "MISTER4DStudio"

        def log_message(self, *_):  # keep the console quiet
            pass

        # The page can start programs and open folders on this computer, so it
        # only answers to itself: local host name, and a token that pages of
        # other sites cannot know.
        def _local(self) -> bool:
            host = (self.headers.get("Host") or "").split(":")[0]
            return host in ("127.0.0.1", "localhost")

        def _send(self, code: int, body: bytes, kind: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload, code: int = 200) -> None:
            body = json.dumps(_finite(payload), allow_nan=False).encode("utf-8")
            self._send(code, body, "application/json; charset=utf-8")

        def _file(self, path: Path) -> None:
            kind = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            size = path.stat().st_size
            start, end = 0, size - 1
            match = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range") or "")
            if match and (match.group(1) or match.group(2)):  # videos are fetched in ranges
                if match.group(1):
                    start = int(match.group(1))
                    end = min(int(match.group(2)), end) if match.group(2) else end
                else:
                    start = max(0, size - int(match.group(2)))
                if start > end:
                    return self._send(416, b"", kind)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            else:
                self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Accept-Ranges", "bytes")
            # A page inside the project folder (the player) runs in an origin of
            # its own: it cannot read the token of the Studio nor call its API.
            self.send_header("Content-Security-Policy", "sandbox allow-scripts allow-popups allow-downloads")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining > 0:
                    chunk = f.read(min(1 << 16, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError):
                        return
                    remaining -= len(chunk)

        def do_GET(self):  # noqa: N802
            if not self._local():
                return self._send(403, b"forbidden", "text/plain")
            path = unquote(urlparse(self.path).path)
            if path == "/":
                page = (ASSETS / "app.html").read_text(encoding="utf-8").replace("{{TOKEN}}", token)
                return self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            if path in ("/app.js", "/app.css"):
                kind = "text/javascript" if path.endswith(".js") else "text/css"
                return self._send(200, (ASSETS / path[1:]).read_bytes(), f"{kind}; charset=utf-8")
            if path == "/theme.css":
                return self._send(200, THEME.read_bytes(), "text/css; charset=utf-8")
            if path.startswith("/project/") and studio.project is not None:
                root = studio.project.resolve()
                target = (root / path[len("/project/"):]).resolve()
                if root in target.parents and target.is_file():
                    return self._file(target)
                return self._send(404, b"not found", "text/plain")
            if path.startswith("/api/"):
                if self.headers.get("X-Studio-Token") != token:
                    return self._json({"error": "sessione non valida: ricarica la pagina"}, 403)
                if path == "/api/state":
                    return self._json(studio.state())
                if path.startswith("/api/log/"):
                    return self._json({"text": studio.log(path.rsplit("/", 1)[1])})
            self._send(404, b"not found", "text/plain")

        def do_POST(self):  # noqa: N802
            if not self._local() or self.headers.get("X-Studio-Token") != token:
                return self._json({"error": "sessione non valida: ricarica la pagina"}, 403)
            try:
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                path = urlparse(self.path).path
                if path == "/api/pick-folder":
                    return self._json({"path": pick_folder(str(body.get("title") or "Scegli una cartella"))})
                if path == "/api/project":
                    studio.open_project(str(body.get("path") or ""))
                elif path == "/api/videos":
                    studio.set_videos(body)
                elif path == "/api/options":
                    studio.set_options(body)
                elif path == "/api/run":
                    studio.start([str(s) for s in body.get("steps", [])])
                elif path == "/api/cancel":
                    studio.cancel()
                elif path == "/api/open":
                    target = {"project": studio.project, "result": studio.result}.get(body.get("what"))
                    if studio.project is None or target is None:
                        raise ValueError("Nessun progetto aperto.")
                    target.mkdir(parents=True, exist_ok=True)
                    open_in_file_manager(target)
                elif path == "/api/quit":
                    studio.cancel()
                    threading.Thread(target=self.server.shutdown, daemon=True).start()
                else:
                    return self._json({"error": "richiesta sconosciuta"}, 404)
                self._json(studio.state())
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)
            except OSError as exc:
                self._json({"error": f"Operazione non riuscita: {exc}"}, 500)

    return Handler


def serve(port: int = 8420, open_browser: bool = True, project: str | None = None) -> None:
    studio = Studio()
    if project:
        studio.open_project(str(Path(project).resolve()))
    token = secrets.token_urlsafe(24)
    server = None
    for candidate in range(port, port + 20):  # the next free port if the usual one is taken
        try:
            server = ThreadingHTTPServer(("127.0.0.1", candidate), make_handler(studio, token))
            break
        except OSError:
            continue
    if server is None:
        raise RuntimeError(f"no free port between {port} and {port + 19}")
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"MISTER 4D Studio: {url}")
    print("Lascia aperta questa finestra mentre lavori; per uscire usa il pulsante Esci nella pagina.")
    if open_browser:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        studio.cancel()
        server.server_close()


if __name__ == "__main__":
    serve()
