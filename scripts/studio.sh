#!/usr/bin/env bash
# MISTER 4D Studio: the guided interface. Run scripts/install.sh once before the first use.
# Like the Windows launcher: Python loads the code of this very folder, and the program itself
# (src/mister4d/launch.py) keeps libraries, page and engine in line.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  echo "MISTER 4D Studio non risulta installato: esegui prima scripts/install.sh" >&2
  exit 1
fi
export PYTHONPATH="$PWD/src"
exec .venv/bin/python -m mister4d.launch "$@"
