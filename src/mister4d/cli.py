"""Command line interface: ``mister4d <command> --help``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _board_spec(args):
    from .charuco import BoardSpec

    return BoardSpec(args.squares_x, args.squares_y, args.square, args.marker, args.dictionary)


def _add_board_args(p):
    p.add_argument("--squares-x", type=int, default=5)
    p.add_argument("--squares-y", type=int, default=7)
    p.add_argument("--square", type=float, default=0.12, help="side of a square in metres, measured on the print")
    p.add_argument("--marker", type=float, default=0.09, help="side of a marker in metres")
    p.add_argument("--dictionary", default="DICT_4X4_50")


def cmd_doctor(args):
    from . import doctor

    return doctor.run()


def cmd_board(args):
    import cv2

    spec = _board_spec(args)
    img = spec.image(pixels_per_metre=args.dpi / 0.0254)
    cv2.imwrite(args.out, img)
    w, h = spec.size_m
    print(f"[board] {args.out}: {img.shape[1]}x{img.shape[0]} px at {args.dpi} dpi = "
          f"{w * 100:.1f} x {h * 100:.1f} cm. Print at 100% on a rigid matte panel, then measure "
          "one square and pass the real value with --square when calibrating.")


def cmd_ingest(args):
    from .ingest import ingest_clip

    ingest_clip(args.videos, args.out, sync=args.sync, manual_offsets=args.offsets,
                start=args.start, end=args.end, step=args.step, ext=args.ext,
                checksums=not args.no_checksum)


def cmd_calibrate(args):
    from .camera import Rig
    from .charuco import calibrate_rig

    frames = Path(args.clip) / "frames"
    intr = Rig.load(args.intrinsics_from) if args.intrinsics_from else None
    rig = calibrate_rig(frames, _board_spec(args), reference=args.reference,
                        distortion=args.distortion, floor_frame=args.floor_frame,
                        intrinsics_from=intr)
    rig.save(args.out)
    print(f"[calibrate] RMS reprojection error {rig.report['rms_px']:.3f} px, "
          f"world frame: {rig.report['world_mode']} -> {args.out}")
    if rig.report["rms_px"] > 1.0:
        print("[calibrate] WARNING: RMS above 1 px. Check focus, motion blur, the measured "
              "square size and that the cameras were synchronized while the board moved.")


def cmd_prepare(args):
    from .prepare import prepare_dataset

    prepare_dataset(args.clip, args.calibration, args.out, empty_dir=args.empty,
                    downscale=args.downscale, max_frames=args.max_frames, ext=args.ext)


def cmd_masks(args):
    from .masks import compute_masks

    compute_masks(args.data, threshold=args.threshold, suppress_shadows=not args.keep_shadows)


def cmd_init(args):
    from .hull import visual_hull

    visual_hull(args.data, voxel=args.voxel, frame_step=args.frame_step, exclude=args.exclude,
                total_points=args.points, center=args.center, size=args.size)


def cmd_train(args):
    from .trainer import load_config, train

    overrides = list(args.set or [])
    if args.holdout:
        overrides.append(f"holdout=[{','.join(args.holdout)}]")
    if args.iterations:
        overrides.append(f"iterations={args.iterations}")
    if args.devices:
        overrides.append(f"devices={args.devices}")
    train(args.data, args.out, load_config(args.config, overrides), resume=not args.fresh)


def cmd_lighten(args):
    from .lighten import lighten

    lighten(args.run, args.out, args.budget, data_dir=args.data, iterations=args.iterations or -1,
            devices=args.devices or "auto")


def cmd_eval(args):
    from .evaluate import evaluate_run

    evaluate_run(args.run, data_dir=args.data, frame_step=args.frame_step)


def cmd_render(args):
    from .render import render_orbit

    render_orbit(args.run, args.out, seconds=args.seconds, fps=args.fps, scale=args.scale,
                 fixed_camera=args.camera, take=args.take, speed=args.speed, transparent=args.transparent,
                 width=args.width, height=args.height, path=args.path)


def cmd_export(args):
    from .export import export_engine, export_html, export_ply, export_web

    writer = {"html": export_html, "web": export_web, "engine": export_engine, "ply": export_ply}[args.format]
    writer(args.run, args.out, frame_step=args.frame_step)


def cmd_view(args):
    from .export import serve

    serve(args.dir, port=args.port, host=args.host)


def cmd_app(args):
    from .app import serve

    serve(port=args.port, open_browser=not args.no_browser, project=args.project)


def cmd_demo(args):
    from .synth import SynthConfig, make_prepared_dataset, make_raw_session

    cfg = SynthConfig(num_cameras=args.cameras, width=args.width, height=args.height,
                      take_frames=args.frames, seed=args.seed, subject=args.subject)
    if args.run:
        from .demo import run_demo

        run_demo(args.out, cfg)
    elif args.prepared:
        make_prepared_dataset(args.out, cfg)
        print(f"[demo] ready-to-train dataset in {args.out}")
    else:
        make_raw_session(args.out, cfg)
        print(f"[demo] raw session in {args.out} (videos in raw/, ground truth in gt/)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mister4d",
        description="4D Gaussian Splatting from 3+ synchronized fixed cameras (POC).")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="check tools and GPUs; self-test the GPU rasterizer")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("board", help="write a printable ChArUco board image")
    _add_board_args(p)
    p.add_argument("--dpi", type=float, default=300.0)
    p.add_argument("--out", default="charuco_board.png")
    p.set_defaults(func=cmd_board)

    p = sub.add_parser("ingest", help="check the videos of one clip, align them in time, extract frames")
    p.add_argument("videos", nargs="+",
                   help="the folder of the clip, or its video files (one per camera; "
                        "the file name is the camera name)")
    p.add_argument("--out", required=True, help="clip folder (frames/ and ingest.json are written here)")
    p.add_argument("--sync", choices=["none", "audio", "flash"], default="none",
                   help="none = hardware-synced cameras started together; audio = clap; flash = light pulse")
    p.add_argument("--offsets", help="manual start offsets in frames, e.g. cam02=3,cam03=-1")
    p.add_argument("--start", type=int, default=0, help="first frame to keep on the common timeline")
    p.add_argument("--end", type=int, default=None, help="last frame to keep on the common timeline")
    p.add_argument("--step", type=int, default=1, help="keep one frame every N")
    p.add_argument("--ext", choices=["jpg", "png"], default="jpg")
    p.add_argument("--no-checksum", action="store_true", help="skip the SHA-256 of the source files")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("calibrate", help="intrinsics + extrinsics from a ChArUco clip")
    p.add_argument("clip", help="clip folder produced by 'ingest' with the calibration videos")
    p.add_argument("--out", required=True, help="calibration .json to write")
    _add_board_args(p)
    p.add_argument("--reference", help="reference camera (default: first)")
    p.add_argument("--distortion", choices=["none", "radial1", "radial2", "full"], default="radial2")
    p.add_argument("--floor-frame", type=int, default=None,
                   help="frame in which the board lies on the floor: fixes origin and floor plane")
    p.add_argument("--intrinsics-from", help="calibration .json to take the intrinsics from")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("prepare", help="undistort / resize a take into a training dataset")
    p.add_argument("clip", help="clip folder produced by 'ingest' with the take")
    p.add_argument("--calibration", required=True)
    p.add_argument("--out", required=True, help="dataset folder")
    p.add_argument("--empty", help="clip folder of the empty set (enables 'masks')")
    p.add_argument("--downscale", type=float, default=1.0)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--ext", choices=["jpg", "png"], default="jpg")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("masks", help="foreground masks by comparison with the empty set")
    p.add_argument("data", help="dataset folder")
    p.add_argument("--threshold", type=float, default=24.0, help="difference from the clean plate (0-255)")
    p.add_argument("--keep-shadows", action="store_true", help="do not try to remove cast shadows")
    p.set_defaults(func=cmd_masks)

    p = sub.add_parser("init", help="initial points from the visual hull of the masks, frame by frame")
    p.add_argument("data", help="dataset folder")
    p.add_argument("--voxel", type=float, default=0.015, help="voxel size in metres")
    p.add_argument("--frame-step", type=int, default=1, help="initialize one frame every N")
    p.add_argument("--exclude", nargs="*", default=[], help="cameras to ignore (the held-out ones)")
    p.add_argument("--points", type=int, default=600000, help="total number of initial points")
    p.add_argument("--center", type=float, nargs=3, help="centre of the capture volume (m)")
    p.add_argument("--size", type=float, nargs=3, help="size of the capture volume (m)")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("train", help="optimize the dynamic Gaussian scene (uses every GPU by default)")
    p.add_argument("data", help="dataset folder")
    p.add_argument("--out", required=True, help="run folder")
    p.add_argument("--config", help="YAML file with training options (see configs/)")
    p.add_argument("--holdout", nargs="*", help="cameras excluded from training, used for evaluation")
    p.add_argument("--iterations", type=int)
    p.add_argument("--devices", help='e.g. "cuda:0,cuda:1" (default: all GPUs)')
    p.add_argument("--set", nargs="*", metavar="KEY=VALUE", help="override any config option")
    p.add_argument("--fresh", action="store_true",
                   help="train every segment of a long take again, even those already there")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("lighten", help="a lighter copy of a trained take, for stand-alone headsets")
    p.add_argument("run", help="run folder of the full-quality take")
    p.add_argument("--out", required=True, help="run folder of the lighter copy")
    p.add_argument("--budget", type=int, default=80000, help="Gaussians visible at once, at most")
    p.add_argument("--data", help="dataset folder (default: the one the take was trained on)")
    p.add_argument("--iterations", type=int, help="refinement steps per segment (default: 15%% of the training)")
    p.add_argument("--devices", help='e.g. "cuda:0" (default: all GPUs)')
    p.set_defaults(func=cmd_lighten)

    p = sub.add_parser("eval", help="metrics and comparison images, held-out cameras first")
    p.add_argument("run", help="run folder")
    p.add_argument("--data", help="dataset folder (default: the one used for training)")
    p.add_argument("--frame-step", type=int, default=1)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("render", help="video of the result, for viewing or for editing")
    p.add_argument("run", help="run folder")
    p.add_argument("--out", required=True,
                   help="an .mp4 (H.264), a .mov (ProRes), or a folder for a sequence of PNG images")
    p.add_argument("--seconds", type=float, default=8.0, help="length, when the take is played back and forth")
    p.add_argument("--take", action="store_true", help="play the take once, from start to end")
    p.add_argument("--speed", type=float, default=1.0, help="with --take: 0.5 = half speed")
    p.add_argument("--fps", type=float, default=None, help="frame rate of the video (default: that of the take)")
    p.add_argument("--scale", type=float, default=1.0, help="resolution relative to the training images")
    p.add_argument("--width", type=int, help="width in pixels (instead of --scale)")
    p.add_argument("--height", type=int, help="height in pixels (instead of --scale)")
    p.add_argument("--camera", help="stand where this camera of the rig stood instead of orbiting")
    p.add_argument("--path", help="JSON file with the key views of a camera move (see the player: Copia la vista)")
    p.add_argument("--transparent", action="store_true",
                   help="transparent background: needs a .mov (ProRes 4444) or a folder of PNG images")
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("export", help="bake the scene: single-file player, player folder or .ply sequence")
    p.add_argument("run", help="run folder")
    p.add_argument("--out", required=True, help="an .html file (html) or a folder (web, engine, ply)")
    p.add_argument("--format", choices=["html", "web", "engine", "ply"], default="html",
                   help="html = one file that opens with a double click (default); "
                        "web = folder that loads the take piece by piece, for long takes; "
                        "engine = folder for a game engine (see unity/Mister4D); "
                        "ply = one 3DGS .ply per frame")
    p.add_argument("--frame-step", type=int, default=1)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("view", help="serve a web export on a local port")
    p.add_argument("dir", help="folder written by 'export --format web'")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--host", default="127.0.0.1")
    p.set_defaults(func=cmd_view)

    p = sub.add_parser("app", help="MISTER 4D Studio: the guided interface in the browser")
    p.add_argument("--project", help="project folder to open")
    p.add_argument("--port", type=int, default=8420)
    p.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    p.set_defaults(func=cmd_app)

    p = sub.add_parser("demo", help="synthetic capture session; with --run, the whole pipeline on it")
    p.add_argument("--out", required=True)
    p.add_argument("--cameras", type=int, default=4)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subject", choices=["mannequin", "human"], default="mannequin",
                   help="mannequin = quick figure made of ellipsoids; human = clothed person on a studio "
                        "set with a cast shadow, to judge how real the result looks")
    p.add_argument("--run", action="store_true",
                   help="also run every step, from the videos to the player")
    p.add_argument("--prepared", action="store_true",
                   help="write a ready-to-train dataset instead of raw videos")
    p.set_defaults(func=cmd_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
