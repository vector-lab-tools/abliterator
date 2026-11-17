# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
The scope: a meter on the model path. It reads the residual stream at a layer once a frame, after
the Steer and any units at that layer, and changes nothing. numpy only; the engine hands it the
stream (one vector of d numbers per frame) and the page draws what it keeps.

Each reading, against the stream's statistics over the reference set on the unchanged model
(units.StreamStats):

  trace    the stream less its mean, in coordinates along the layer's principal directions, each
           divided by the spread along it: the first TRACE of them, so 0 is the ordinary stream
           and ±1 a typical excursion. The scope's picture: the stream as a waveform.
  bands    the root mean square of those coordinates in BANDS log-spaced bands of directions,
           1 the typical level: a spectrum of the stream, low bands its largest movements.
  size     ‖h‖ against the stream's typical size at the layer.
  along    for each direction, (h − mean) · v / ‖h‖: on the scale of a Steer dose, which adds
           ‖h‖ · dose · v, so a dose of 0.1 raises it by about 0.1.
  motion   1 − cos(h, the previous frame's h): how far the stream turned in one frame.
"""
from __future__ import annotations

import collections
import threading
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

TRACE = 256
BANDS = 24
HISTORY = 125            # frames kept per layer: 5 s at 40 ms
SEND = 8                 # traces sent per reading of the status: the page plays them out at 40 ms each
MAX_SCOPES = 4


class Layer:
    """What a scope needs about one layer: mean, typical size, principal directions and spreads,
    and the directions to read along (unit vectors)."""

    def __init__(self, mean: np.ndarray, norm: float, vt: np.ndarray, sc: np.ndarray, dirs: Dict[str, np.ndarray]):
        self.mean = np.asarray(mean, np.float32)
        self.norm = float(norm)
        keep = np.asarray(sc) > 1e-6 * max(1e-12, float(np.max(sc)))
        self.vt = np.asarray(vt, np.float32)[keep]
        self.sc = np.asarray(sc, np.float32)[keep]
        n = len(self.sc)
        self.edges = np.unique(np.round(np.geomspace(1, n, BANDS + 1)).astype(int) - 1)
        self.edges[-1] = n
        self.dirs = {k: np.asarray(v, np.float32) / (np.linalg.norm(v) + 1e-12) for k, v in dirs.items()}


def read(L: Layer, h: np.ndarray, prev: Optional[np.ndarray] = None) -> Dict[str, Any]:
    h = np.asarray(h, np.float32).reshape(-1)
    x = h - L.mean
    nh = float(np.linalg.norm(h)) + 1e-12
    z = (L.vt @ x) / L.sc
    bands = [float(np.sqrt(np.mean(z[a:b] ** 2))) for a, b in zip(L.edges[:-1], L.edges[1:]) if b > a]
    r = {"trace": np.clip(z[:TRACE], -6, 6), "bands": bands, "size": nh / (L.norm + 1e-12),
         "along": {k: float(x @ v / nh) for k, v in L.dirs.items()}}
    r["motion"] = 0.0 if prev is None else float(1 - h @ prev / (nh * (np.linalg.norm(prev) + 1e-12)))
    return r


class Scope:
    """The scopes on the desk: their layers, and what each has read lately."""

    def __init__(self, layers_of=None):
        self.layers_of = layers_of            # layer -> Layer, made on demand (the engine's statistics)
        self.at: List[int] = []
        self._L: Dict[int, Layer] = {}
        self._prev: Dict[int, np.ndarray] = {}
        self._hist: Dict[int, "collections.deque"] = {}
        self._lock = threading.Lock()

    def set(self, layers: Sequence[int], n_layers: int = 12) -> List[int]:
        at = []
        for v in layers:
            v = int(v)
            if not 1 <= v <= n_layers:
                raise ValueError(f"a scope reads a layer from 1 to {n_layers}")
            if v not in at:
                at.append(v)
        at = at[:MAX_SCOPES]
        for v in at:
            if v not in self._L and self.layers_of is not None:
                self._L[v] = self.layers_of(v)
        with self._lock:
            self.at = at
            for v in list(self._hist):
                if v not in at:
                    del self._hist[v]
                    self._prev.pop(v, None)
        return list(at)

    def reset(self):
        with self._lock:
            self._hist.clear()
            self._prev.clear()

    def add(self, frame: int, streams: Dict[int, np.ndarray]):
        for v, h in streams.items():
            L = self._L.get(v)
            if L is None:
                continue
            h = np.asarray(h, np.float32).reshape(-1)
            r = read(L, h, self._prev.get(v))
            r["frame"] = int(frame)
            with self._lock:
                self._prev[v] = h
                self._hist.setdefault(v, collections.deque(maxlen=HISTORY)).append(r)

    def payload(self) -> Dict[str, Any]:
        """For the page: per layer, the last SEND traces, the last bands, and every kept frame's
        size, motion and readings along the directions."""
        out = {}
        with self._lock:
            hist = {v: list(d) for v, d in self._hist.items() if v in self.at}
        for v, rs in hist.items():
            if not rs:
                continue
            last = rs[-1]
            out[str(v)] = {
                "frame": last["frame"],
                "traces": [[round(float(t), 2) for t in r["trace"]] for r in rs[-SEND:]],
                "frames": [r["frame"] for r in rs[-SEND:]],
                "bands": [round(b, 3) for b in last["bands"]],
                "size": [round(r["size"], 3) for r in rs],
                "motion": [round(r["motion"], 4) for r in rs],
                "along": {k: [round(r["along"][k], 4) for r in rs] for k in last["along"]},
            }
        return out
