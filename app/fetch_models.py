# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""Download the Magenta RT 2 weights listed in models.json from Hugging Face, at the pinned revision, into
MAGENTA_HOME/magenta-rt-v2 (magenta-rt's own layout; the default is ~/Documents/Magenta). A file already there
at the right size is kept. A new download goes to a .part file, resumes if interrupted, and is checked against
the manifest's SHA-256 before it is moved into place. Standard library only.

Progress is printed as JSON lines: {"have": bytes, "of": bytes, "file": path}, then {"done": true} or
{"error": text}.

    python fetch_models.py [--check]     # --check: report what is missing, download nothing
"""
import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHUNK = 1 << 20


def say(**kw):
    print(json.dumps(kw), flush=True)


def home(man):
    return Path(os.environ.get("MAGENTA_HOME") or Path.home() / "Documents" / "Magenta") / man["home"]


def missing(man):
    root = home(man)
    return [f for f in man["files"] if not ((root / f["path"]).is_file() and (root / f["path"]).stat().st_size == f["size"])]


def fetch(man, f, have_before, total):
    dest = home(man) / f["path"]
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{man['repo']}/resolve/{man['revision']}/{f['path']}"
    for attempt in range(5):
        start = part.stat().st_size if part.exists() else 0
        if start > f["size"]:
            part.unlink()
            start = 0
        req = urllib.request.Request(url, headers={"User-Agent": "Abliterator", **({"Range": f"bytes={start}-"} if start else {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if start and r.status != 206:
                    start = 0
                with open(part, "ab" if start else "wb") as out:
                    got, last = start, 0.0
                    while chunk := r.read(CHUNK):
                        out.write(chunk)
                        got += len(chunk)
                        if time.time() - last > 0.25:
                            say(file=f["path"], have=have_before + got, of=total)
                            last = time.time()
            break
        except OSError as e:
            if attempt == 4:
                raise
            say(file=f["path"], retry=str(e))
            time.sleep(2 + 3 * attempt)
    h = hashlib.sha256()
    with open(part, "rb") as fh:
        while chunk := fh.read(CHUNK * 8):
            h.update(chunk)
    if h.hexdigest() != f["sha256"]:
        part.unlink()
        raise RuntimeError(f"{f['path']}: the download does not match the published checksum")
    part.replace(dest)


def main():
    man = json.loads((HERE / "models.json").read_text())
    todo = missing(man)
    total = sum(f["size"] for f in man["files"])
    if "--check" in sys.argv:
        say(missing=[f["path"] for f in todo], bytes=sum(f["size"] for f in todo), of=total, home=str(home(man)))
        return
    have = total - sum(f["size"] for f in todo)
    try:
        for f in todo:
            fetch(man, f, have, total)
            have += f["size"]
            say(file=f["path"], have=have, of=total)
    except Exception as e:
        say(error=str(e))
        sys.exit(1)
    say(done=True, home=str(home(man)))


if __name__ == "__main__":
    main()
