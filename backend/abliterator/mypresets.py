# Abliterator. Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
"""Presets of your own for units and effects: a card's settings saved under a name, kept in data/my_presets.json
and listed on every card of that unit or effect. Saving a name already kept for it replaces that preset."""
import json
import re
import time
from typing import Any, Dict, List, Optional

from . import ROOT

FILE = ROOT / "data" / "my_presets.json"
KINDS = ("units", "fx")
MAX_NAME = 40


def load() -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    try:
        d = json.loads(FILE.read_text())
    except (OSError, ValueError):
        d = {}
    return {k: d.get(k, {}) if isinstance(d.get(k), dict) else {} for k in KINDS}


def _save(d: Dict[str, Any]) -> None:
    FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    tmp.replace(FILE)


def _check(kind: str, owner: str) -> None:
    if kind not in KINDS:
        raise ValueError("kind is units or fx.")
    if not re.fullmatch(r"[A-Za-z0-9_\-]{1,40}", owner or ""):
        raise ValueError("no such unit or effect.")


def add(kind: str, owner: str, name: str, params: Dict[str, Any], block: Optional[Any] = None) -> Dict[str, Any]:
    _check(kind, owner)
    name = " ".join(str(name or "").split())[:MAX_NAME]
    if not name:
        raise ValueError("give the preset a name.")
    if not isinstance(params, dict) or not params:
        raise ValueError("nothing to save.")
    clean = {str(k): v for k, v in params.items() if isinstance(v, (int, float, str, bool)) and len(str(k)) <= 40}
    q = {"name": name, "params": clean, "made": time.strftime("%Y-%m-%d %H:%M")}
    if block is not None:
        q["block"] = block
    d = load()
    lst = [p for p in d[kind].get(owner, []) if p.get("name") != name] + [q]
    d[kind][owner] = lst
    _save(d)
    return q


def remove(kind: str, owner: str, name: str) -> bool:
    _check(kind, owner)
    d = load()
    lst = d[kind].get(owner, [])
    keep = [p for p in lst if p.get("name") != name]
    if len(keep) == len(lst):
        return False
    d[kind][owner] = keep
    _save(d)
    return True
