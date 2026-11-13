"""MISTER 4D Studio: what the page asks the pipeline to do, and who may ask."""

import json
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from mister4d import app
from mister4d.cli import build_parser

TOKEN = "segreto"


WORKSTATION = {"checked": True, "ffmpeg": True, "error": "",
               "gpus": [{"name": "NVIDIA RTX 6000 Ada Generation", "memory_gb": 48}] * 2}


@pytest.fixture()
def studio(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "MEMORY", tmp_path / "memoria" / "studio.json")
    monkeypatch.setattr(app.Studio, "_check_system", lambda self: setattr(self, "system", dict(WORKSTATION)))
    studio = app.Studio()
    studio.open_project(str(tmp_path / "progetto"))
    studio.system = dict(WORKSTATION)  # the tests do not depend on the machine they run on
    return studio


@pytest.fixture()
def server(studio):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.make_handler(studio, TOKEN))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def make_clips(root, names=("cam01", "cam02", "cam03")):
    folders = {}
    for clip in ("calib", "empty", "take"):
        (root / clip).mkdir(parents=True)
        for name in names:
            (root / clip / f"{name}.mp4").write_bytes(b"")
        folders[clip] = str(root / clip)
    return folders


def fetch(url, token=None, body=None, headers=None):
    request = urllib.request.Request(url, data=None if body is None else json.dumps(body).encode())
    if token:
        request.add_header("X-Studio-Token", token)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never through a proxy
    try:
        with opener.open(request, timeout=10) as response:
            return response.status, response.read(), response.headers
    except urllib.error.HTTPError as error:
        return error.code, error.read(), error.headers


def wait_idle(studio, timeout=30.0):
    deadline = time.time() + timeout
    while studio.job is not None:
        assert time.time() < deadline, "the job did not end"
        time.sleep(0.02)


def cli_args(studio, step):
    """The command lines of a step, as the ``mister4d`` parser receives them."""
    commands = [argv[4:] for _, argv in studio.commands(step)]
    for argv in commands:
        build_parser().parse_args(argv)  # the Studio only asks for what the command line accepts
    return commands


# ------------------------------------------------------------------ videos ---

def test_the_three_folders_must_hold_the_same_cameras(studio, tmp_path):
    folders = make_clips(tmp_path / "riprese")
    studio.set_videos({"calib": folders["calib"]})
    assert not studio.video_report()["ready"]
    with pytest.raises(ValueError):
        studio.commands("video")

    studio.set_videos(folders)
    report = studio.video_report()
    assert report["ready"] and report["cameras"]["take"] == ["cam01", "cam02", "cam03"]

    (tmp_path / "riprese" / "empty" / "cam02.mp4").unlink()
    report = studio.video_report()
    assert not report["ready"] and "cam02" in report["problems"][0]


def test_the_checking_camera_must_exist_in_the_take(studio, tmp_path):
    studio.set_videos(make_clips(tmp_path / "a"))
    studio.set_options({"holdout": "cam03"})
    studio.set_videos(make_clips(tmp_path / "b", names=("sx", "centro", "dx")))
    assert studio.options["holdout"] == ""


# ---------------------------------------------------------------- commands ---

def test_commands_follow_the_options(studio, tmp_path):
    studio.set_videos(make_clips(tmp_path / "riprese"))
    studio.set_options({"sync": "audio", "skip_frames": 8, "square_cm": 10.0, "marker_cm": 7.5,
                        "floor_frame": 12, "downscale": 2, "keep_shadows": True, "holdout": "cam02",
                        "iterations": 500, "devices": "cuda:0", "learn_time_offsets": True})

    calib, empty, take = cli_args(studio, "video")
    assert calib[0] == empty[0] == take[0] == "ingest"
    assert calib[-4:] == take[-4:] == ["--sync", "audio", "--start", "8"]
    assert "--sync" not in empty  # the empty set is still: nothing to align

    (calibrate,) = cli_args(studio, "calibrazione")
    assert calibrate[calibrate.index("--square") + 1] == "0.1"  # the page asks centimetres
    assert calibrate[calibrate.index("--marker") + 1] == "0.075"
    assert calibrate[calibrate.index("--floor-frame") + 1] == "12"

    prepare, masks = cli_args(studio, "preparazione")
    assert prepare[prepare.index("--downscale") + 1] == "2"
    assert "--keep-shadows" in masks

    init, train = cli_args(studio, "ricostruzione")
    assert init[-2:] == ["--exclude", "cam02"]
    for flag, value in (("--holdout", "cam02"), ("--iterations", "500"), ("--devices", "cuda:0"),
                        ("--set", "learn_time_offsets=true")):
        assert train[train.index(flag) + 1] == value


def test_defaults_ask_for_nothing_optional(studio, tmp_path):
    studio.set_videos(make_clips(tmp_path / "riprese"))
    (calibrate,) = cli_args(studio, "calibrazione")
    assert "--floor-frame" not in calibrate
    init, train = cli_args(studio, "ricostruzione")
    assert "--exclude" not in init
    assert not {"--holdout", "--iterations", "--devices", "--set"} & set(train)


def test_without_a_gpu_the_reconstruction_is_reduced(studio, tmp_path):
    studio.set_videos(make_clips(tmp_path / "riprese"))
    studio.set_options({"learn_time_offsets": True})
    studio.system = {"checked": True, "ffmpeg": True, "gpus": [], "error": ""}
    assert studio.state()["system"]["reduced"]
    init, train = cli_args(studio, "ricostruzione")
    assert init[init.index("--points") + 1] == "60000"
    assert train[train.index("--devices") + 1] == "cpu"
    assert train[train.index("--iterations") + 1] == "1500"
    overrides = train[train.index("--set") + 1:]  # one --set: a second one would replace the first
    assert train.count("--set") == 1
    assert {"learn_time_offsets=true", "max_gaussians=80000"} <= set(overrides)
    assert "--scale" in cli_args(studio, "risultato")[0]

    studio.set_options({"iterations": 300})  # what the operator asks still wins
    _, train = cli_args(studio, "ricostruzione")
    assert train[train.index("--iterations") + 1] == "300"


def test_quality_is_measured_only_with_a_checking_camera(studio):
    assert [argv[0] for argv in cli_args(studio, "risultato")] == ["render", "export"]
    run = studio.result / "run"
    run.mkdir(parents=True)
    (run / "train_summary.json").write_text(json.dumps({"holdout_cameras": ["cam02"]}))
    assert [argv[0] for argv in cli_args(studio, "risultato")] == ["eval", "render", "export"]


def test_the_page_and_the_server_know_the_same_options():
    page = (app.ASSETS / "app.html").read_text(encoding="utf-8")
    asked = set(re.findall(r'data-option="(\w+)"', page)) | set(re.findall(r'type="radio" name="(\w+)"', page))
    assert asked == set(app.DEFAULT_OPTIONS) == set(app.Studio.OPTION_STEP)
    assert set(re.findall(r'data-run="(\w+)"', page)) == set(app.STEPS)


# ------------------------------------------------------------------- state ---

def test_changing_an_option_reopens_what_depends_on_it(studio):
    studio.status = {step: "done" for step in app.STEPS}
    studio.set_options({"downscale": 2})
    assert studio.status == {"video": "done", "calibrazione": "done", "preparazione": "todo",
                             "ricostruzione": "todo", "risultato": "todo"}
    studio.status["preparazione"] = "done"
    studio.set_options({"downscale": 2})  # the same value again: nothing to redo
    assert studio.status["preparazione"] == "done"


def test_a_project_comes_back_as_it_was_left(studio):
    studio.set_options({"square_cm": 8.0})
    studio.status.update(video="done", calibrazione="running")
    studio.save()
    again = app.Studio()  # opens the last project by itself
    assert again.project == studio.project
    assert again.options["square_cm"] == 8.0
    assert again.status["video"] == "done"
    assert again.status["calibrazione"] == "todo"  # it was interrupted by closing the program


def test_training_log_becomes_progress():
    job = app.Job(["ricostruzione"])
    job.read("[train] frames=30 size=960x540 train_cameras=3 iterations=7000 images/step=4")
    job.read("[train] step=3499  loss=0.02  l1=0.01  gaussians=1000  seconds=350.0")
    assert job.progress == pytest.approx(0.5)
    assert job.detail == "Passo 3500 di 7000"
    assert job.eta == pytest.approx(350.0, rel=0.01)


# -------------------------------------------------------------------- jobs ---

def test_a_queue_stops_at_the_failure_and_explains_it(studio, monkeypatch):
    code = {"video": "print('fatto')",
            "calibrazione": "import sys; print('error: only 2 frames with the board'); sys.exit(1)"}
    monkeypatch.setattr(app.Studio, "commands",
                        lambda self, step: [("prova", [sys.executable, "-c", code[step]])])
    studio.start(["video", "calibrazione", "preparazione"])
    wait_idle(studio)
    assert studio.status["video"] == "done"
    assert studio.status["calibrazione"] == "error"
    assert studio.status["preparazione"] == "todo"
    assert "pannello" in studio.problems["calibrazione"]  # said in the operator's words
    assert "only 2 frames" in studio.log("calibrazione")  # and kept as it was for whoever maintains it

    with pytest.raises(ValueError):
        studio.start(["ricostruzione"])  # what comes before is not done
    with pytest.raises(ValueError):
        studio.start([])
    with pytest.raises(ValueError):
        studio.start(["formatta-il-disco"])


def test_a_running_step_can_be_interrupted(studio, monkeypatch):
    slow = "import time; print('partito', flush=True); time.sleep(120)"
    monkeypatch.setattr(app.Studio, "commands", lambda self, step: [("prova", [sys.executable, "-c", slow])])
    studio.start(["video"])
    with pytest.raises(ValueError):
        studio.set_options({"downscale": 4})  # nothing changes under a running step
    deadline = time.time() + 30
    while not (studio.job and "partito" in studio.job.lines):
        assert time.time() < deadline
        time.sleep(0.02)
    assert studio.state()["steps"][0]["status"] == "running"
    studio.cancel()
    wait_idle(studio)
    assert studio.status["video"] == "todo" and "video" not in studio.problems


# -------------------------------------------------------------------- http ---

def test_only_the_page_itself_may_use_the_api(server, studio):
    status, page, _ = fetch(server + "/")
    assert status == 200 and TOKEN.encode() in page and b"{{TOKEN}}" not in page
    for asset in ("/app.js", "/app.css", "/theme.css"):
        assert fetch(server + asset)[0] == 200

    assert fetch(server + "/api/state")[0] == 403
    assert fetch(server + "/api/state", token="sbagliato")[0] == 403
    status, body, _ = fetch(server + "/api/state", token=TOKEN)
    assert status == 200 and json.loads(body)["project"]["name"] == "progetto"

    assert fetch(server + "/api/options", body={"downscale": 4})[0] == 403
    assert studio.options["downscale"] == 1
    assert fetch(server + "/api/options", token=TOKEN, body={"downscale": 4})[0] == 200
    assert studio.options["downscale"] == 4

    # A page of another site that reaches this server under its own name is refused.
    assert fetch(server + "/api/state", token=TOKEN, headers={"Host": "altro.example"})[0] == 403
    assert fetch(server + "/", headers={"Host": "altro.example"})[0] == 403

    status, body, _ = fetch(server + "/api/run", token=TOKEN, body={"steps": ["calibrazione"]})
    assert status == 400 and "precedenti" in json.loads(body)["error"]


def test_project_files_are_served_without_leaving_the_folder(server, studio, tmp_path):
    (studio.project / "dati").mkdir()
    (studio.project / "dati" / "nota.txt").write_text("dentro")
    (tmp_path / "fuori.txt").write_text("fuori")

    status, body, headers = fetch(server + "/project/dati/nota.txt")
    assert status == 200 and body == b"dentro"
    assert "sandbox" in headers["Content-Security-Policy"]  # a file of the project never acts as the Studio

    status, body, headers = fetch(server + "/project/dati/nota.txt", headers={"Range": "bytes=1-3"})
    assert status == 206 and body == b"ent" and headers["Content-Range"] == "bytes 1-3/6"

    for sneaky in ("/project/../fuori.txt", "/project/%2e%2e/fuori.txt", "/project/dati/../../fuori.txt"):
        assert fetch(server + sneaky)[0] == 404


def test_numbers_that_could_not_be_measured_reach_the_page_as_null(server, studio):
    folder = studio.result / "run" / "eval"
    folder.mkdir(parents=True)
    (folder / "metrics.json").write_text('{"holdout_cameras": ["cam02"], "holdout": {"psnr": 20.0, "iou": NaN}}')
    studio.status = {step: "done" for step in app.STEPS}
    status, body, _ = fetch(server + "/api/state", token=TOKEN)
    assert status == 200 and b"NaN" not in body  # a browser refuses NaN in JSON
    quality = json.loads(body)["steps"][-1]["summary"]["quality"]
    assert quality == {"cameras": ["cam02"], "psnr": 20.0, "iou": None}
