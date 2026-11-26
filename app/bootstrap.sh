#!/bin/bash
# The app's first run, and the run after an update: install Python 3.12 and the two environments from the lock
# files with uv, then download the Magenta RT 2 weights. Each step is skipped when it is already done.
#
#   bootstrap.sh <source folder> <support folder> <uv>
#
# The source folder is the app's copy of the code (read-only); the support folder is
# ~/Library/Application Support/Abliterator. Progress goes to stdout as JSON lines ({"step": ..., "state": ...},
# and the model fetcher's {"have": ..., "of": ...}); everything else is log text.
set -e
SRC="$1"; SUP="$2"; UV="$3"
export UV_PYTHON_INSTALL_DIR="$SUP/python" UV_CACHE_DIR="$SUP/cache" UV_NO_CONFIG=1 UV_PYTHON_PREFERENCE=only-managed
mkdir -p "$SUP/env" "$SUP/logs"
step() { echo "{\"step\": \"$1\", \"state\": \"$2\"}"; }
fail() { echo "{\"step\": \"$1\", \"state\": \"failed\"}"; exit 1; }

env_for() {                                                 # env_for <name> <lock> [extra uv args]
  local name="$1" lock="$2"; shift 2
  local want; want=$(shasum -a 256 "$lock" | cut -c1-64)
  if [ -x "$SUP/env/$name/bin/python" ] && [ "$(cat "$SUP/env/$name.lock-sha" 2>/dev/null)" = "$want" ]; then
    step "$name" done; return
  fi
  step "$name" start
  rm -rf "$SUP/env/$name" "$SUP/env/$name.lock-sha"
  "$UV" venv -q --python 3.12 "$SUP/env/$name" || fail "$name"
  "$UV" pip install --python "$SUP/env/$name/bin/python" --require-hashes --no-deps -r "$lock" "$@" 2>&1 || fail "$name"
  echo "$want" > "$SUP/env/$name.lock-sha"
  step "$name" done
}

step python start
"$UV" python install --no-bin 3.12 2>&1 || fail python
step python done
env_for server "$SRC/app/requirements.lock"
env_for engine "$SRC/app/requirements-mrt2.lock" --find-links "$SRC/app/wheels"
"$UV" cache prune -q 2>/dev/null || true

step models start
"$SUP/env/server/bin/python" -I "$SRC/app/fetch_models.py" || fail models
step models done
