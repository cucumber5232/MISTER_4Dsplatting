"""The launcher: libraries, instance, and page and engine of the same version."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from importlib import metadata
from pathlib import Path

import pytest

from mister4d import app, launch

ROOT = Path(__file__).parent.parent


def test_the_requirements_of_the_project_are_read_and_checked():
    wanted = launch.read_requirements(ROOT / "pyproject.toml")
    assert any(w.startswith("torch") for w in wanted) and any(w.startswith("gsplat") for w in wanted)
    assert launch.unmet(wanted) == []  # this very environment runs the tests

    have = {"numpy": "1.26.4", "gsplat": "1.5.3+pt24cu124", "torch": "2.4.1+cu124"}

    def version_of(name):
        if name not in have:
            raise metadata.PackageNotFoundError(name)
        return have[name]

    asked = ["numpy", "gsplat>=1.5,<2", "torch", "scipy", "opencv-python-headless>=4.8"]
    assert launch.unmet(asked, version_of) == ["scipy", "opencv-python-headless>=4.8"]
    assert launch.unmet(["gsplat>=1.6"], version_of) == ["gsplat>=1.6"]  # an update asks for a newer one
    assert launch.unmet(["gsplat>=1.5.3", "numpy<2", "torch==2.4.1"], version_of) == []
    assert launch.unmet(["numpy>=2"], version_of) == ["numpy>=2"]
    assert launch.read_requirements(ROOT / "no-such-file.toml") == []


def test_the_build_changes_when_any_file_of_the_program_does(tmp_path):
    (tmp_path / "studio").mkdir()
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "studio" / "app.js").write_text("let a;\n")
    (tmp_path / "notes.txt").write_text("not code")
    first = launch.build_id(tmp_path)
    assert first == launch.build_id(tmp_path) and len(first) == 10
    (tmp_path / "notes.txt").write_text("still not code")
    assert launch.build_id(tmp_path) == first
    (tmp_path / "studio" / "app.js").write_text("let b;\n")  # the page...
    second = launch.build_id(tmp_path)
    assert second != first
    (tmp_path / "a.py").write_text("x = 2\n")  # ...or the engine
    assert launch.build_id(tmp_path) != second


@pytest.fixture()
def studio(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "MEMORY", tmp_path / "memoria" / "studio.json")
    monkeypatch.setattr(app.Studio, "_check_system", lambda self: None)
    studio = app.Studio()
    studio.open_project(str(tmp_path / "progetto"))
    return studio


@pytest.fixture()
def server(studio):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.make_handler(studio, "segreto"))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def _get(port, path, token=None):
    request = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    if token:
        request.add_header("X-Studio-Token", token)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=10) as response:
        return response.read()


def test_a_studio_already_open_is_found_and_tells_its_version(server, studio):
    assert launch.find_running(range(server, server + 1)) == (f"http://127.0.0.1:{server}/", studio.build)
    assert studio.build == launch.build_id()
    assert launch.find_running(range(1, 2)) is None  # nothing listens there


def test_the_page_belongs_to_the_program_that_serves_it(server, studio, monkeypatch):
    page = _get(server, "/").decode()
    assert f'name="studio-build" content="{studio.build}"' in page and "{{BUILD}}" not in page
    state = json.loads(_get(server, "/api/state", "segreto"))
    assert state["build"] == studio.build and state["update"] is False

    # The files change under a running program (an update): the page is told, and no step starts
    # until the program is opened again. What is served stays the page of the running version.
    monkeypatch.setattr(app, "build_id", lambda: "altra")
    studio._looked = 0.0
    state = json.loads(_get(server, "/api/state", "segreto"))
    assert state["update"] is True and state["build"] == studio.build
    with pytest.raises(ValueError, match="aggiornato"):
        studio.start(["video"])
    assert f'content="{studio.build}"' in _get(server, "/").decode()


def test_the_launchers_load_the_code_next_to_them():
    bat = (ROOT / "MISTER 4D Studio.bat").read_bytes()
    assert b"\r\n" in bat and b'set "PYTHONPATH=%~dp0src"' in bat and b"-m mister4d.launch %*" in bat
    shell = (ROOT / "scripts" / "studio.sh").read_text()
    assert 'PYTHONPATH="$PWD/src"' in shell and "-m mister4d.launch" in shell
