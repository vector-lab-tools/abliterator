# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""The presets, read from the YAML block in backend/config/presets.md, and the prompts typed on
the Engine panel, kept in data/prompts.json and listed after them under Your prompts."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, List, Optional

import yaml

from . import ROOT

FILE = ROOT / "backend" / "config" / "presets.md"
USER = ROOT / "data" / "prompts.json"
MAX_PROMPT = 240
GROUP = "Your prompts"


def builtin() -> List[Dict[str, Any]]:
    m = re.search(r"```yaml\n(.*?)```", FILE.read_text(), re.S)
    return yaml.safe_load(m.group(1))["presets"] if m else []


def user() -> List[Dict[str, Any]]:
    try:
        return json.loads(USER.read_text()) if USER.exists() else []
    except ValueError:
        return []


def presets() -> List[Dict[str, Any]]:
    return builtin() + user()


def clean_prompt(text: str) -> str:
    t = " ".join(str(text or "").split())
    if len(t) < 3:
        raise ValueError("type a style of at least three characters.")
    if len(t) > MAX_PROMPT:
        raise ValueError(f"a style prompt can be at most {MAX_PROMPT} characters.")
    return t


def add(text: str) -> Dict[str, Any]:
    """A typed prompt as a preset: the same prompt already listed is returned as it is; a new one is
    kept in data/prompts.json, named by its opening words."""
    t = clean_prompt(text)
    for p in presets():
        if p["prompt"].lower() == t.lower():
            return p
    name = t if len(t) <= 34 else t[:32].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    p = {"id": "my-" + hashlib.sha1(t.encode()).hexdigest()[:8], "group": GROUP, "name": name, "prompt": t,
         "cfg": 1.0, "temperature": 1.0, "top_k": 250, "seeds": [1, 2, 3], "heard": False, "x": "brightness", "y": "pulse", "user": True}
    us = user() + [p]
    USER.parent.mkdir(parents=True, exist_ok=True)
    USER.write_text(json.dumps(us, indent=1))
    return p


def remove(pid: str) -> bool:
    us = user()
    keep = [p for p in us if p["id"] != pid]
    if len(keep) == len(us):
        return False
    USER.write_text(json.dumps(keep, indent=1))
    return True


def find(pid: str) -> Optional[Dict[str, Any]]:
    return next((p for p in presets() if p["id"] == pid), None)
