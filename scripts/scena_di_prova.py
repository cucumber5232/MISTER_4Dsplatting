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

The first scene is a quick one: it only checks that everything runs on this machine, every time.
If it fails the long ones are not started. A scene already measured is skipped; a training that
was interrupted continues from what it had finished. risultati.zip is written again after every
scene, so it can be sent before the last one is done.

A scene can use the takes already made for another one (same cameras, same frames). Scenes of
earlier versions of this script are not done again: their measures stay in risultati/.
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
    high: int = 0  # more training cameras, 2.6 m above the floor
    takes: str = ""  # the takes of this other scene: same cameras, same frames
    scenery: str = ""  # the set trained for this other scene, when its takes and options are the same
    options: tuple = ()  # training options of this scene only
    always: bool = False  # done again at every start


SCENES = [
    Scene("prova_rapida", "controllo veloce: tutta la catena in piccolo", 4, 150.0, 480, 270, 8, 1500, 1500,
          high=1, always=True),
    Scene("rig12_d", "12 camere su 220 gradi, 1280x720: tra le camere resta solo cio' su cui tre concordano", 12,
          220.0, 1280, 720, 60, 20000, 25000, between=3, takes="rig12"),
    Scene("rig12_k", "come rig12_d, con i fotogrammi chiave (uno ogni 3, vita dalla velocita')", 12, 220.0, 1280,
          720, 60, 20000, 25000, between=3, takes="rig12", scenery="rig12_d", options=("init_keyframes=3",)),
    Scene("sala16_c", "la scena da mostrare: 12 camere su 220 gradi e 4 in alto, 1920x1080, set nuovo", 12, 220.0,
          1920, 1080, 60, 20000, 30000, between=3, high=4, takes="sala16"),
]
SMOKE = dict(width=160, height=90, frames=3, scenery_iterations=40, iterations=30)
ROOTS = {"bench": "bench", "runs": "runs", "show": "consegna", "results": "risultati"}


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
    data, run_dir = REPO / ROOTS["bench"] / (scene.takes or scene.name), REPO / ROOTS["runs"] / scene.name
    out, show = results / scene.name, REPO / ROOTS["show"] / scene.name
    out.mkdir(parents=True, exist_ok=True)
    show.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").unlink(missing_ok=True)  # until it is measured again the scene is not done
    options = list(scene.options)
    note = [f"# {scene.name}", "", scene.what + ".", "",
            f"{scene.cameras + scene.high} camere di training, {scene.width}x{scene.height}, {scene.frames} frame; "
            f"{scene.scenery_iterations} passi per il set, {scene.iterations} per la ripresa.", ""]
    if options:
        note += ["Opzioni di training di questa scena: " + " ".join(options) + ".", ""]
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
        if scene.high:
            wanted["high"] = scene.high
        made = data / "scena.json"
        if not (made.exists() and json.loads(made.read_text()) == wanted and (data / "scenery" / "meta.json").exists()):
            shutil.rmtree(data, ignore_errors=True)
            cmd = [py, "scripts/make_benchmark.py", "dataset", data, "--width", scene.width, "--height", scene.height,
                   "--frames", scene.frames, "--first", 0, "--train-cameras", scene.cameras, "--arc", scene.arc,
                   "--full-scene", "--jobs", max(1, min(args.jobs, (os.cpu_count() or 2) - 1))]
            if scene.between is not None:
                cmd += ["--verify-between", scene.between]
            if scene.high:
                cmd += ["--high", scene.high]
            if args.smoke:
                cmd += ["--points", 3000]
            step("riprese", cmd)
            made.write_text(json.dumps(wanted))
        verify = json.loads((data / "truth" / "cameras.json").read_text())["verify"]
        if not (data / "scenery" / "rig_support.npz").exists():
            # Takes prepared by an earlier version: the starting points of the set are made again.
            step("punti", [*cli, "init", data, "--full-scene", "--exclude", *verify])

        # 2. The whole scene: the empty set first, then the take on top of it.
        if args.redo:
            shutil.rmtree(run_dir, ignore_errors=True)
        for stale in ("metrics.json", "set.json", "set_points.npz"):  # measures of an earlier training
            (run_dir / "evalq" / stale).unlink(missing_ok=True)
        ready = REPO / ROOTS["runs"] / scene.scenery / "scenery" if scene.scenery else None
        if ready and (ready / "ckpt.pt").exists() and not (run_dir / "scenery" / "ckpt.pt").exists():
            shutil.rmtree(run_dir / "scenery", ignore_errors=True)
            shutil.copytree(ready, run_dir / "scenery")  # kept by the training if data and options are the same
        cmd = [*cli, "train", data, "--out", run_dir, "--full-scene", "--holdout", *verify,
               "--set", f"scenery_iterations={scene.scenery_iterations}", f"iterations={scene.iterations}",
               *options, *args.set]
        if args.devices:
            cmd += ["--devices", args.devices]
        step("training", cmd)

        # 3. Measures and pictures on the cameras that were never used.
        middle = scene.frames // 2
        step("misure", [py, "scripts/quality_check.py", run_dir], needed=False)
        if not scene.scenery:  # the set against the studio, where it was trained
            step("set", [py, "scripts/make_benchmark.py", "set", run_dir], needed=False)
        step("viste", [py, "scripts/contact_sheet.py", run_dir, run_dir / "viste.png", "--scale", 1, "--zoom", 2.0,
                       "--width", 480, "--steps", 6, "--frames", f"0,{middle},{scene.frames - 1}"], needed=False)

        # 4. What can be shown: the player and a video along the arc of the cameras.
        step("player", [*cli, "export", run_dir, "--out", show / "player.html"], needed=False)
        step("video", [*cli, "render", run_dir, "--out", show / "video.mp4", "--seconds", 0.3 if args.smoke else 12,
                       "--width", min(1920, 2 * scene.width), "--height", min(1080, 2 * scene.height),
                       "--margin", -5], needed=False)  # inside the arc of the cameras: nothing is guessed
    except Failed as failed:
        say(f"{scene.name}: FERMATA al passo «{failed}»")
        raise
    finally:
        for src, dst in ((run_dir / "evalq" / "metrics.json", "metrics.json"),
                         (run_dir / "evalq" / "set.json", "set.json"),
                         (run_dir / "evalq" / "set_points.npz", "set_points.npz"),
                         (run_dir / "train_log.jsonl", "train_log.jsonl"),
                         (run_dir / "scenery" / "train_log.jsonl", "scenery_log.jsonl"),
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
        for cam, row in metrics.get("verify_per_camera", {}).items():
            note.append(f"  - {cam}: " + ", ".join(f"{k} {v:.2f}" for k, v in row.items()
                                                   if k in ("psnr", "psnr_subject", "depth_cm")))
        if metrics:
            note.append(f"- Gaussiane della ripresa {metrics.get('gaussians')}, del set {metrics.get('scenery')}; "
                        f"opache {100 * metrics.get('opaque_share', 0):.0f}%, ad ago {100 * metrics.get('needle_share', 0):.0f}%")
        if (out / "set.json").exists():
            the_set = json.loads((out / "set.json").read_text())
            note.append(f"- set: {the_set['in_air']} Gaussiane a mezz'aria nella stanza "
                        f"({100 * the_set['in_air_opacity_share']:.1f}% dell'opacità); dalle camere di verifica "
                        f"il {100 * the_set['verify']['in_front_share']:.1f}% dell'immagine è disegnato davanti al set")
        sizes = [f"{f.name} {f.stat().st_size / 1e6:.0f} MB" for f in sorted(show.glob("*")) if f.is_file()]
        note += ["", f"Da mostrare (in {ROOTS['show']}/{scene.name}): " + (", ".join(sizes) or "niente"), ""]
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

    if args.smoke:  # its own folders: a check of this script must not touch the real scenes
        ROOTS.update({key: name + "_prova" for key, name in ROOTS.items()})
    results = REPO / ROOTS["results"]
    scenes = [s for s in SCENES if not args.only or s.name in args.only]
    if args.smoke:
        scenes = [Scene(**{**s.__dict__, **SMOKE, "cameras": min(s.cameras, 4), "high": min(s.high, 1)})
                  for s in (scenes if args.only else scenes[:1])]
        args.devices = args.devices or "cpu"
        args.set = [*args.set, "views_per_step=2", "densify_start=10", "densify_every=10", "log_every=10"]
    gpu = environment(results, args.smoke)
    code = 0

    def archive() -> str:
        return shutil.make_archive(str(REPO / results.name), "zip", root_dir=REPO, base_dir=results.name)

    if not gpu and not args.smoke:
        code = 2
    else:
        for scene in scenes:
            if (results / scene.name / "metrics.json").exists() and not (args.redo or scene.always):
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
            say(f"{scene.name}: fatta. {archive()} è aggiornato e si può già mandare")
    say(f"Da mandare: {archive()}")
    if code == 0:
        say(f"Da mostrare: la cartella {REPO / ROOTS['show']}")
    return code


if __name__ == "__main__":
    sys.exit(main())
