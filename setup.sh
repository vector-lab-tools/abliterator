#!/bin/bash
# Make the virtual environment inside this folder and install what the backend needs.
# ./setup.sh --mrt2 also makes .venv-mrt2 for the Magenta RT 2 engine (Apple silicon).
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
if command -v uv >/dev/null; then
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
else
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
if [ "$1" = "--mrt2" ]; then
  # The Magenta RT 2 engine runs in its own environment (MLX). Apple silicon only.
  if command -v uv >/dev/null; then
    uv venv --python 3.12 .venv-mrt2
    uv pip install --python .venv-mrt2/bin/python -r requirements-mrt2.txt
  else
    python3 -m venv .venv-mrt2
    .venv-mrt2/bin/pip install -r requirements-mrt2.txt
  fi
fi
echo "Done. Start with ./launch.sh"
