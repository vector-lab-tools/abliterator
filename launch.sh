#!/bin/bash
# Start the Abliterator server if it is not already running, then open the page.
# The server listens on 127.0.0.1 only. Logs go to _tmp/server.log.
PORT="${ABLITERATOR_AUDIO_PORT:-8775}"
URL="http://127.0.0.1:$PORT/"
DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "No virtual environment yet. Run ./setup.sh first." >&2
  exit 1
fi
mkdir -p "$DIR/_tmp"
if ! /usr/bin/curl -fs -o /dev/null "${URL}api/status"; then
  nohup "$PY" "$DIR/backend/server.py" </dev/null >>"$DIR/_tmp/server.log" 2>&1 &
  for _ in $(seq 1 100); do
    /usr/bin/curl -fs -o /dev/null "${URL}api/status" && break
    sleep 0.2
  done
fi
if /usr/bin/curl -fs -o /dev/null "${URL}api/status"; then
  open "$URL" 2>/dev/null || echo "$URL"
else
  echo "The server did not start; see $DIR/_tmp/server.log" >&2
  exit 1
fi
