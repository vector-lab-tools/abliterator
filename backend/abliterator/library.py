# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
The sound library: every sound saved from the instrument (a snapshot of what was played, or a
freeze rendered offline), as a WAV and its recipe in data/snapshots/. The tracks play sounds
from here.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from . import DATA

SOUNDS = DATA / "snapshots"
ID = re.compile(r"[0-9A-Za-z-]+")


def _recipe(sid: str) -> Dict[str, Any]:
    p = SOUNDS / f"{sid}.json"
    try:
        return json.loads(p.read_text()) if p.exists() else {}
    except ValueError:
        return {}


def default_name(r: Dict[str, Any]) -> str:
    pre = r.get("preset") or {}
    name = pre.get("name") or pre.get("id") or r.get("engine", "sound")
    strips = len((r.get("chain") or {}).get("strips") or r.get("start", {}).get("chain") or [])
    return f"{'frozen · ' if r.get('frozen') else ''}{name} · seed {r.get('seed', '?')}" + (f" · {strips} strip{'s' if strips != 1 else ''}" if strips else "")


def entry(sid: str, r: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = r if r is not None else _recipe(sid)
    pre = r.get("preset") or {}
    chain = (r.get("chain") or {}).get("strips") or (r.get("start") or {}).get("chain") or []
    return {"id": sid, "name": r.get("name") or default_name(r), "seconds": r.get("seconds"), "engine": r.get("engine"),
            "frozen": bool(r.get("frozen")), "preset": pre.get("name") or pre.get("id"), "seed": r.get("seed"),
            "strips": list(chain), "made": sid[:8] + " " + sid[9:11] + ":" + sid[11:13] if len(sid) >= 13 else sid,
            "sample_rate": r.get("sample_rate"), "channels": r.get("channels")}


def sounds() -> List[Dict[str, Any]]:
    if not SOUNDS.exists():
        return []
    return [entry(p.stem) for p in sorted(SOUNDS.glob("*.json"), reverse=True) if (SOUNDS / f"{p.stem}.wav").exists()]


def exists(sid: str) -> bool:
    return bool(ID.fullmatch(sid)) and (SOUNDS / f"{sid}.wav").exists()


def name(sid: str) -> str:
    return entry(sid)["name"]


def rename(sid: str, new: str) -> Dict[str, Any]:
    if not exists(sid):
        raise ValueError("no such sound")
    r = _recipe(sid)
    r["name"] = (new or "").strip()[:80] or default_name(r)
    (SOUNDS / f"{sid}.json").write_text(json.dumps(r, indent=1))
    return entry(sid, r)


def delete(sid: str):
    """Moves the sound's files to data/snapshots/trash/ (nothing is erased)."""
    if not exists(sid):
        raise ValueError("no such sound")
    trash = SOUNDS / "trash"
    trash.mkdir(exist_ok=True)
    for p in SOUNDS.glob(f"{sid}.*"):
        p.rename(trash / p.name)
