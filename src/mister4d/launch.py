"""What the double click on "MISTER 4D Studio" runs.

Before the page opens, three things are put in line:

- the libraries: if an update of the program asks for one that is not there,
  or for a newer one, it is installed, without touching the others (PyTorch
  and gsplat above all, which are tied to the graphics driver);
- the instance: if the Studio is already open, the browser goes to it instead
  of a second copy being started;
- interface and engine: the launcher makes Python load this very folder, and
  the server serves the page it was started with (see ``app.py``), so the two
  can never come from different versions.

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import urllib.request
import webbrowser
from importlib import metadata
from pathlib import Path

PACKAGE = Path(__file__).parent
PORTS = range(8420, 8440)


def build_id(root: Path = PACKAGE) -> str:
    """Identity of the code in ``root``: it changes whenever a file of the program does."""
    digest = hashlib.sha1()
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix in (".py", ".html", ".js", ".css"):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:10]


# ------------------------------------------------------------------ libraries ---

def read_requirements(pyproject: Path) -> list[str]:
    """The ``dependencies`` of the project, read without a TOML library (Python 3.10 has none)."""
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return []
    found = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.S | re.M)
    return re.findall(r'"([^"]+)"', found.group(1)) if found else []


def _numbers(version: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", version.split("+")[0])[:4])


def _satisfied(installed: str, spec: str) -> bool:
    for clause in filter(None, (c.strip() for c in spec.split(","))):
        found = re.match(r"(>=|<=|==|!=|>|<|~=)\s*([\w.]+)", clause)
        if not found:
            continue  # nothing this simple check understands: taken as fine
        have, want = _numbers(installed), _numbers(found.group(2))
        width = max(len(have), len(want))
        have, want = have + (0,) * (width - len(have)), want + (0,) * (width - len(want))
        ok = {">=": have >= want, "<=": have <= want, "==": have == want, "!=": have != want,
              ">": have > want, "<": have < want, "~=": have >= want}[found.group(1)]
        if not ok:
            return False
    return True


def unmet(requirements: list[str], version_of=metadata.version) -> list[str]:
    """The requirements that the installed libraries do not satisfy."""
    missing = []
    for requirement in requirements:
        plain = requirement.split(";")[0].strip()
        found = re.match(r"[A-Za-z0-9_.-]+", plain)
        if not found:
            continue
        try:
            installed = version_of(found.group(0))
        except metadata.PackageNotFoundError:
            missing.append(plain)
            continue
        if not _satisfied(installed, plain[found.end():]):
            missing.append(plain)
    return missing


def align_libraries(pyproject: Path = PACKAGE.parents[1] / "pyproject.toml", say=print) -> bool:
    """Install what the program asks for and is not there. True when everything is in place."""
    wanted = unmet(read_requirements(pyproject))
    if not wanted:
        return True
    say("Un aggiornamento del programma richiede: " + ", ".join(wanted))
    say("Lo installo (serve la connessione a internet)...")
    names = {re.match(r"[A-Za-z0-9_.-]+", w).group(0).lower().replace("_", "-") for w in wanted}
    # Everything already installed stays exactly as it is.
    frozen = subprocess.run([sys.executable, "-m", "pip", "freeze", "--exclude-editable"],
                            capture_output=True, text=True).stdout.splitlines()
    keep = [line for line in frozen if "==" in line and " @ " not in line
            and line.split("==")[0].lower().replace("_", "-") not in names]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as constraints:
        constraints.write("\n".join(keep) + "\n")
    done = subprocess.run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
                           "-c", constraints.name, *wanted])
    Path(constraints.name).unlink(missing_ok=True)
    if done.returncode != 0 or unmet(read_requirements(pyproject)):
        say("Non sono riuscito a installarlo. Controlla la connessione e riapri il programma, "
            "oppure rilancia l'installazione (scripts\\install.bat).")
        return False
    say("Fatto.")
    return True


# ------------------------------------------------------------------- instance ---

def find_running(ports=PORTS) -> tuple[str, str] | None:
    """Address and build of a Studio already open on this computer, if there is one."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never through a proxy
    for port in ports:
        url = f"http://127.0.0.1:{port}/"
        try:
            with opener.open(url + "api/ping", timeout=0.4) as response:
                answer = json.loads(response.read())
        except (OSError, ValueError):
            continue
        if isinstance(answer, dict) and answer.get("app") == "mister4d-studio":
            return url, str(answer.get("build", ""))
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="MISTER 4D Studio")
    parser.add_argument("--project", help="project folder to open")
    parser.add_argument("--port", type=int, default=PORTS[0])
    parser.add_argument("--no-browser", action="store_true", help="do not open the page automatically")
    args = parser.parse_args(argv)

    if not align_libraries():
        return 1
    mine = build_id()
    running = find_running()
    if running is not None:
        url, build = running
        if build == mine:
            print(f"MISTER 4D Studio è già aperto: {url}")
            if not args.no_browser:
                webbrowser.open(url)
            return 0
        print("È ancora aperta una versione precedente di MISTER 4D Studio: chiudila con il pulsante "
              "Esci della sua pagina. Intanto apro quella nuova.")

    from .app import serve

    serve(port=args.port, open_browser=not args.no_browser, project=args.project)
    return 0


if __name__ == "__main__":
    sys.exit(main())
