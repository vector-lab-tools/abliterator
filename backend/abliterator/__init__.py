# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VERSION = (ROOT / "VERSION").read_text().strip() if (ROOT / "VERSION").exists() else "0.0.0"
TOOL = "Abliterator"

# SHIPPED is data/ in the source tree: the effect and unit presets, read from there. DATA holds what the
# instrument writes (sounds, sessions, exports, the desk, directions, stream statistics, the calibration with
# your ears ranges). In the source tree they are the same folder; the app sets ABLITERATOR_DATA and
# ABLITERATOR_LOGS to folders in Application Support, since the app's own files cannot be written to.
SHIPPED = ROOT / "data"
DATA = Path(os.environ.get("ABLITERATOR_DATA") or SHIPPED)
LOGS = Path(os.environ.get("ABLITERATOR_LOGS") or ROOT / "_tmp")


def seed():
    """Copy the shipped calibration (and Tune's clips, where the tree has them) into DATA if DATA has none."""
    if DATA.resolve() == SHIPPED.resolve():
        return
    DATA.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    for f in (SHIPPED / "calibration").glob("*.json"):
        t = DATA / "calibration" / f.name
        if not t.exists():
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, t)
    clips = SHIPPED / "calibration" / "clips"
    if clips.is_dir() and not (DATA / "calibration" / "clips").exists():
        shutil.copytree(clips, DATA / "calibration" / "clips")
