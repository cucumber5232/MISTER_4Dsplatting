"""The test scene, whole and trained long: from nothing to a player, a video and a report to send back.

Run it on the workstation with the Python of .venv, from the folder of the project
(on Windows a double click on "Scena di prova su GPU.bat" does it):

    .venv\\Scripts\\python scripts\\scena_di_prova.py              every scene not yet done
    .venv\\Scripts\\python scripts\\scena_di_prova.py --only rig4  only these
    .venv\\Scripts\\python scripts\\scena_di_prova.py --redo       again, from the training on
    .venv\\Scripts\\python scripts\\scena_di_prova.py --smoke      everything tiny, also without a GPU

For every scene: synthetic takes of the test person on the studio set (scripts/make_benchmark.py, with
verification cameras that are never trained on), the whole scene reconstructed without masks
(mister4d train --full-scene), the measures (scripts/quality_check.py, scripts/contact_sheet.py), the
player and a video.

    bench/<scene>      the takes                     runs/<scene>       the training
    consegna/<scene>   player.html and video.mp4: what can be shown
    risultati/<scene>  measures, pictures and logs   risultati.zip      all of risultati/, to send back

The first scene is a quick one: it only checks that everything runs on this machine. If it fails
the long ones are not started. A scene already measured is skipped; a training that was
interrupted continues from what it had finished.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


@dataclass
class Scene:
    name: str
    what: str
    cameras: int
    arc: float
    width: int
    height: int
    frames: int
    scenery_iterations: int
    iterations: int
    between: int | None = None  # verification cameras between the training ones (None = one per gap)


SCENES = [
    Scene("prova_rapida", "controllo veloce: tutta la catena in piccolo", 4, 150.0, 480, 270, 8, 1500, 1500),
    Scene("rig4", "4 camere su un arco di 150 gradi", 4, 150.0, 1280, 720, 60, 20000, 25000),
    Scene("rig12", "12 camere su un arco di 220 gradi", 12, 220.0, 1280, 720, 60, 20000, 25000, between=3),
]
SMOKE = dict(width=160, height=90, frames=3, scenery_iterations=40, iterations=30)


def say(text: str) -> None:
    print(f"[scena] {text}", flush=True)


def run(cmd: list, log: Path) -> tuple[int, float]:
    """Run a command of the project, its output to the console and to ``log``; returns exit code and seconds."""
    cmd = [str(c) for c in cmd]
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    start = time.time()
    with open(log, "w", encoding="utf-8", errors="replace") as out:
        out.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace")
        for line in proc.stdout:
            out.write(line)
            out.flush()
            if not line.startswith("[train] step=") or " psnr_" in line:
                print("    " + line.rstrip(), flush=True)
        code = proc.wait()
    return code, time.time() - start


def tail(path: Path, lines: int = 60) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def small_copy(src: Path, dst: Path, side: int = 1800) -> None:
    """A JPEG copy of a sheet of pictures, small enough to send."""
    import cv2

    img = cv2.imread(str(src))
    if img is None:
        return
    scale = min(1.0, side / max(img.shape[:2]))
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 88])


class Failed(RuntimeError):
    pass


def environment(results: Path, smoke: bool) -> bool:
    """AMBIENTE.md: the machine, the GPUs, what 'mister4d doctor' says. True if a GPU can be used."""
    import torch

    results.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.is_available()
    lines = ["# Ambiente", "", f"- data: {time.strftime('%Y-%m-%d %H:%M')}", f"- sistema: {platform.platform()}",
             f"- Python {platform.python_version()}, PyTorch {torch.__version__}",
             f"- GPU utilizzabili: {torch.cuda.device_count() if gpu else 0}"]
    if gpu:
        lines += [f"  - {i}: {torch.cuda.get_device_name(i)}, "
                  f"{torch.cuda.get_device_properties(i).total_memory / 2 ** 30:.0f} GB"
                  for i in range(torch.cuda.device_count())]
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True)
    lines.append(f"- versione del codice: {head.stdout.strip() or 'non è un clone git'}")
    code, _ = run([sys.executable, "-m", "mister4d.cli", "doctor"], results / "doctor.txt")
    lines += ["", f"`mister4d doctor` è uscito con codice {code} (0 = nessun problema): vedi doctor.txt.", ""]
    (results / "AMBIENTE.md").write_text("\n".join(lines), encoding="utf-8")
    if not gpu and not smoke:
        say("ERRORE: PyTorch non vede nessuna GPU su questo computer. Vedi risultati/doctor.txt.")
    return gpu


def one(scene: Scene, args, results: Path) -> None:
    py, cli = sys.executable, [sys.executable, "-m", "mister4d.cli"]
    data, run_dir = REPO / "bench" / scene.name, REPO / "runs" / scene.name
    out, show = results / scene.name, REPO / "consegna" / scene.name
    out.mkdir(parents=True, exist_ok=True)
    show.mkdir(parents=True, exist_ok=True)
    note = [f"# {scene.name}", "", scene.what + ".", "",
            f"{scene.cameras} camere di training, {scene.width}x{scene.height}, {scene.frames} frame; "
            f"{scene.scenery_iterations} passi per il set, {scene.iterations} per la ripresa.", ""]
    seconds = {}

    def step(name: str, cmd: list, needed: bool = True) -> bool:
        say(f"{scene.name}: {name}")
        code, took = run(cmd, run_dir.parent / f"{scene.name}_log_{name}.txt")
        seconds[name] = took
        if code != 0:
            log = run_dir.parent / f"{scene.name}_log_{name}.txt"
            (out / f"errore_{name}.txt").write_text(tail(log, 120), encoding="utf-8")
            note.append(f"**{name} non riuscito** (codice {code}): vedi errore_{name}.txt.")
            if needed:
                raise Failed(name)
        return code == 0

    try:
        # 1. The takes, with the truth, prepared for the whole scene.
        wanted = dict(cameras=scene.cameras, arc=scene.arc, width=scene.width, height=scene.height, frames=scene.frames,
                      between=scene.between)
        made = data / "scena.json"
        if not (made.exists() and json.loads(made.read_text()) == wanted and (data / "scenery" / "meta.json").exists()):
            shutil.rmtree(data, ignore_errors=True)
            cmd = [py, "scripts/make_benchmark.py", "dataset", data, "--width", scene.width, "--height", scene.height,
                   "--frames", scene.frames, "--first", 0, "--train-cameras", scene.cameras, "--arc", scene.arc,
                   "--full-scene", "--jobs", max(1, min(args.jobs, (os.cpu_count() or 2) - 1))]
            if scene.between is not None:
                cmd += ["--verify-between", scene.between]
            if args.smoke:
                cmd += ["--points", 3000]
            step("riprese", cmd)
            made.write_text(json.dumps(wanted))
        verify = json.loads((data / "truth" / "cameras.json").read_text())["verify"]

        # 2. The whole scene: the empty set first, then the take on top of it.
        if args.redo:
            shutil.rmtree(run_dir, ignore_errors=True)
        cmd = [*cli, "train", data, "--out", run_dir, "--full-scene", "--holdout", *verify,
               "--set", f"scenery_iterations={scene.scenery_iterations}", f"iterations={scene.iterations}",
               *args.set]
        if args.devices:
            cmd += ["--devices", args.devices]
        step("training", cmd)

        # 3. Measures and pictures on the cameras that were never used.
        middle = scene.frames // 2
        step("misure", [py, "scripts/quality_check.py", run_dir], needed=False)
        step("viste", [py, "scripts/contact_sheet.py", run_dir, run_dir / "viste.png", "--scale", 1, "--zoom", 2.0,
                       "--width", 480, "--steps", 6, "--frames", f"0,{middle},{scene.frames - 1}"], needed=False)

        # 4. What can be shown: the player and a video along the arc of the cameras.
        step("player", [*cli, "export", run_dir, "--out", show / "player.html"], needed=False)
        step("video", [*cli, "render", run_dir, "--out", show / "video.mp4", "--seconds", 0.3 if args.smoke else 12,
                       "--width", min(1920, 2 * scene.width), "--height", min(1080, 2 * scene.height)], needed=False)
    except Failed as failed:
        say(f"{scene.name}: FERMATA al passo «{failed}»")
        raise
    finally:
        for src, dst in ((run_dir / "evalq" / "metrics.json", "metrics.json"),
                         (run_dir / "train_summary.json", "train_summary.json"),
                         (run_dir / "scenery" / "train_summary.json", "scenery_summary.json"),
                         (run_dir / "config.yaml", "config.yaml")):
            if src.exists():
                shutil.copyfile(src, out / dst)
        for src, dst in ((run_dir / "evalq" / "sheet.png", "sheet.jpg"), (run_dir / "viste.png", "viste.jpg")):
            if src.exists():
                small_copy(src, out / dst)
        for log in sorted(run_dir.parent.glob(f"{scene.name}_log_*.txt")):
            (out / log.name.replace(f"{scene.name}_", "")).write_text(tail(log, 80), encoding="utf-8")
        metrics = json.loads((out / "metrics.json").read_text()) if (out / "metrics.json").exists() else {}
        note += ["| Passo | Minuti |", "|---|---|", *[f"| {k} | {v / 60:.1f} |" for k, v in seconds.items()], ""]
        for group in ("verify", "train"):
            if group in metrics:
                note.append(f"- camere {'di verifica' if group == 'verify' else 'di training'}: "
                            + ", ".join(f"{k} {v:.2f}" for k, v in metrics[group].items()))
        if metrics:
            note.append(f"- Gaussiane della ripresa {metrics.get('gaussians')}, del set {metrics.get('scenery')}; "
                        f"opache {100 * metrics.get('opaque_share', 0):.0f}%, ad ago {100 * metrics.get('needle_share', 0):.0f}%")
        sizes = [f"{f.name} {f.stat().st_size / 1e6:.0f} MB" for f in sorted(show.glob("*")) if f.is_file()]
        note += ["", "Da mostrare (in consegna/" + scene.name + "): " + (", ".join(sizes) or "niente"), ""]
        (out / "NOTE.md").write_text("\n".join(note), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="only these scenes")
    ap.add_argument("--redo", action="store_true", help="train again scenes that are already there")
    ap.add_argument("--smoke", action="store_true", help="everything tiny: checks this script, says nothing on quality")
    ap.add_argument("--devices", default="", help='e.g. "cuda:0" (default: every GPU)')
    ap.add_argument("--jobs", type=int, default=8, help="cameras rendered at the same time when making the takes")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="more training options, for every scene")
    args = ap.parse_args()

    results = REPO / ("risultati_prova" if args.smoke else "risultati")
    scenes = [s for s in SCENES if not args.only or s.name in args.only]
    if args.smoke:
        scenes = [Scene(**{**s.__dict__, **SMOKE}) for s in scenes[:1]]
        args.devices = args.devices or "cpu"
        args.set = [*args.set, "views_per_step=2", "densify_start=10", "densify_every=10", "log_every=10"]
    gpu = environment(results, args.smoke)
    code = 0
    if not gpu and not args.smoke:
        code = 2
    else:
        for scene in scenes:
            if (results / scene.name / "metrics.json").exists() and not args.redo:
                say(f"{scene.name}: già fatta (--redo per rifarla)")
                continue
            try:
                one(scene, args, results)
            except Failed:
                code = 1
                break
            except Exception:  # noqa: BLE001 - whatever went wrong must reach the report
                (results / scene.name).mkdir(parents=True, exist_ok=True)
                (results / scene.name / "errore.txt").write_text(traceback.format_exc(), encoding="utf-8")
                say(f"{scene.name}: errore inatteso, vedi {results / scene.name / 'errore.txt'}")
                code = 1
                break
    archive = shutil.make_archive(str(REPO / results.name), "zip", root_dir=REPO, base_dir=results.name)
    say(f"Da mandare: {archive}")
    if code == 0:
        say(f"Da mostrare: la cartella {REPO / 'consegna'}")
    return code


if __name__ == "__main__":
    sys.exit(main())
