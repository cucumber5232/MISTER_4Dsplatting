#!/usr/bin/env bash
# MISTER 4D Studio: the guided interface. Run scripts/install.sh once before the first use.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  echo "MISTER 4D Studio non risulta installato: esegui prima scripts/install.sh" >&2
  exit 1
fi
exec .venv/bin/python -m mister4d.app "$@"
