# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
The server's side of the Magenta RT 2 engine: starts mrt2_proc.py in its own environment
(.venv-mrt2) and passes commands to it as JSON lines, one reply per command. The engine plays
through the sound card itself, so nothing here is on the audio path.
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import threading
from typing import Any, Dict, List, Optional

from . import ROOT

PY = ROOT / ".venv-mrt2" / "bin" / "python"
PROC = ROOT / "backend" / "abliterator" / "mrt2_proc.py"
LOG = ROOT / "_tmp" / "mrt2.log"


class MRT2Engine:
    name = "mrt2"
    label = "Magenta RT 2 small"

    def __init__(self):
        self.p: Optional[subprocess.Popen] = None
        self.status_text = "not started"
        self.error: Optional[str] = None
        self._ids = itertools.count(1)
        self._wait: Dict[int, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def available() -> bool:
        return PY.exists()

    def start(self, prompts: List[str], names: Optional[Dict[str, str]] = None):
        if not self.available():
            self.status_text = "not installed (no .venv-mrt2)"
            return
        LOG.parent.mkdir(exist_ok=True)
        env = dict(os.environ, ABLITERATOR_PROMPTS=json.dumps(prompts), ABLITERATOR_PRESET_NAMES=json.dumps(names or {}),
                   PYTHONUNBUFFERED="1")
        self.p = subprocess.Popen([str(PY), str(PROC)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=open(LOG, "a"), env=env, text=True, bufsize=1, cwd=str(ROOT))
        self.status_text = "loading Magenta RT 2"
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            try:
                m = json.loads(line)
            except ValueError:
                continue
            w = self._wait.pop(m.get("id"), None)
            if w is not None:
                w["reply"] = m
                w["ev"].set()
        self.status_text = "stopped"
        self.error = f"the Magenta RT 2 process ended (see {LOG.relative_to(ROOT)})"

    def call(self, cmd: str, timeout: float = 5.0, **kw) -> Dict[str, Any]:
        if self.p is None or self.p.poll() is not None:
            raise RuntimeError(self.error or f"Magenta RT 2 is {self.status_text}")
        rid = next(self._ids)
        w = {"ev": threading.Event(), "reply": None}
        self._wait[rid] = w
        with self._lock:
            self.p.stdin.write(json.dumps({"id": rid, "cmd": cmd, **kw}) + "\n")
            self.p.stdin.flush()
        if not w["ev"].wait(timeout):
            self._wait.pop(rid, None)
            raise TimeoutError(f"Magenta RT 2 did not answer '{cmd}' within {timeout:.0f} s")
        r = w["reply"]
        if not r.get("ok"):
            raise RuntimeError(r.get("error", "failed"))
        r.pop("id", None)
        r.pop("ok", None)
        return r

    # ------------------------------------------------------------------ the engine interface
    def status(self) -> Dict[str, Any]:
        try:
            s = self.call("status", timeout=2)
            self.status_text = s["state"]
            return s
        except Exception as e:
            return {"state": self.status_text, "error": self.error or (None if self.status_text == "loading Magenta RT 2" else str(e)),
                    "playing": False, "model": None, "lead": None, "latency": None, "directions": []}

    def ready(self) -> bool:
        return self.status_text == "ready"

    def play(self, preset: Dict[str, Any], seed: int, sampling: Optional[Dict[str, Any]] = None):
        return self.call("play", timeout=10, preset=preset, seed=seed, sampling=sampling)

    def stop(self) -> Optional[str]:
        return self.call("stop", timeout=20).get("saved")

    def explain_files(self) -> Dict[str, str]:
        return self.call("explain_files", timeout=10)["files"]

    def about(self) -> Dict[str, Any]:
        return self.call("about", timeout=10)

    def embed(self, prompt: str) -> Dict[str, Any]:
        """Makes a typed prompt's style tokens, so it can be switched to while playing."""
        return self.call("embed", timeout=60, prompt=prompt)

    def control(self, **kw) -> Dict[str, Any]:
        return self.call("control", **{k: v for k, v in kw.items() if v or v is False})

    def freeze(self, preset: Dict[str, Any], seed: int, seconds: float) -> Dict[str, Any]:
        return self.call("freeze", timeout=60 + seconds * 2, preset=preset, seed=seed, seconds=seconds)

    def set_chain(self, lines: List[str], on: Optional[bool] = None, anchor: Optional[List[Any]] = None,
                  snap: Optional[List[Any]] = None) -> Dict[str, Any]:
        return self.call("chain", timeout=60, lines=lines, **({} if on is None else {"on": on}),
                         **({"anchor": anchor} if anchor else {}), **({"snap": snap} if snap else {}))

    def set_scopes(self, layers: List[int]) -> Dict[str, Any]:
        return self.call("scopes", timeout=30, layers=layers)

    def set_mod(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        return self.call("mod", spec=spec)

    def units(self) -> Dict[str, Any]:
        return self.call("units", timeout=30)

    def snapshot(self, seconds: float) -> Dict[str, Any]:
        return self.call("snapshot", timeout=20, seconds=seconds)

    def set_lead(self, n: int) -> int:
        return self.call("lead", lead=n)["lead"]

    def quit(self):
        if self.p and self.p.poll() is None:
            try:
                self.call("quit", timeout=1)
            except Exception:
                pass
            self.p.terminate()
