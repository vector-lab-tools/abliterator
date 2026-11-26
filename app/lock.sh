#!/bin/bash
# Make app/requirements.lock and app/requirements-mrt2.lock from the requirements files, pinned with hashes to
# the versions installed in .venv and .venv-mrt2 (the environments the tests ran in). VAU is not on PyPI, so it
# is pinned to the wheel in app/wheels, built from the commit that requirements-mrt2.txt names.
set -e
cd "$(dirname "$0")/.."
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
export MACOSX_DEPLOYMENT_TARGET=14.0                       # mlx has no wheels for macOS 13
lock() {                                                    # lock <requirements> <venv> <out>
  uv pip freeze --python "$2/bin/python" | grep -v " @ " > "$T/tested.txt"
  uv pip compile "$1" -c "$T/tested.txt" --generate-hashes --python-version 3.12 \
    --python-platform aarch64-apple-darwin --no-header -q -o "$3"
  python3 - "$3" <<'EOF'
import re, sys, hashlib, pathlib
p = pathlib.Path(sys.argv[1]); s = p.read_text().replace(str(pathlib.Path.cwd()), ".")
s = re.sub(r"-c \S*tested\.txt", "-c tested.txt", s)
for w in pathlib.Path("app/wheels").glob("vector_abliteration_units-*.whl"):
    v, h = w.name.split("-")[1], hashlib.sha256(w.read_bytes()).hexdigest()
    s = re.sub(r"vector-abliteration-units @ git\+\S+(?: \\)?\n", f"vector-abliteration-units=={v} \\\\\n    --hash=sha256:{h}\n", s)
head = ("# Pinned with hashes for the app (Apple silicon, Python 3.12). Made by app/lock.sh: uv pip compile on the\n"
        "# requirements file, constrained (tested.txt) to the versions in the tested environment.\n")
p.write_text(head + s)
EOF
}
lock requirements.txt .venv app/requirements.lock
lock requirements-mrt2.txt .venv-mrt2 app/requirements-mrt2.lock
grep -q "git+" app/requirements-mrt2.lock && { echo "VAU wheel missing from app/wheels" >&2; exit 1; }
echo "Locked."
