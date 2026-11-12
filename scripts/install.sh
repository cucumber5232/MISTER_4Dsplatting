#!/usr/bin/env bash
# Linux: create the virtual environment next to the code, install the project
# with its dependencies and check the machine (tools, GPUs, GPU rasterizer).
#
#   scripts/install.sh            uses python3
#   PYTHON=python3.12 scripts/install.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "error: Python 3.10 or newer is required ($PYTHON is $("$PYTHON" --version 2>&1))" >&2
  exit 1
fi

"$PYTHON" -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
# On Linux the default PyTorch wheel is the CUDA build.
python -m pip install -e ".[dev]"

mister4d doctor || true
echo
echo "Installed. In every new terminal run:  source .venv/bin/activate"
