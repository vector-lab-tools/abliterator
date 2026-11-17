# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
The spectrogram of what is heard, for the page to draw while it plays. numpy only.

A sound-card callback hands each block it plays to `feed` (a copy goes on a queue, nothing else, so
the callback stays cheap). `payload`, called when the page asks for the status, turns what has
arrived into columns: a Hann window of N samples every HOP (20 ms at 48 kHz), each column the
largest power in ROWS log-spaced bands from 40 Hz to 20 kHz, in dB, scaled from TOP − RANGE to
TOP into 0 to 255. The calculation and the scale are the website's (docs/spectro.js), so the app's
pictures and the site's compare. The page asks again every quarter of a second and keeps the
columns it has not seen, by their index.
"""
from __future__ import annotations

import base64
import collections
import threading
from typing import Any, Dict

import numpy as np

N, HOP, ROWS, FLO, FHI = 2048, 960, 150, 40.0, 20000.0
TOP, RANGE = -22.0, 72.0
KEEP = 64                     # columns kept for the page between its requests (1.3 s)


class Spectro:
    def __init__(self, sr: int = 48000):
        self.sr = sr
        self.win = np.hanning(N).astype(np.float32)
        self.norm = 4.0 / float(self.win.sum()) ** 2
        f = np.fft.rfftfreq(N, 1 / sr)
        edges = FLO * (FHI / FLO) ** (np.arange(ROWS + 1) / ROWS)
        self.lo = np.maximum(1, np.floor(edges[:-1] / sr * N)).astype(int)
        self.hi = np.maximum(self.lo + 1, np.ceil(edges[1:] / sr * N)).astype(int)
        self.hi = np.minimum(self.hi, len(f))
        self.dead = self.lo >= len(f)                      # bands above the Nyquist frequency (MusicGen's 32 kHz stops at 16 kHz): drawn empty
        self.lo = np.minimum(self.lo, len(f) - 1)
        self.q: "collections.deque" = collections.deque()
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.q.clear()
            self.buf = np.zeros(0, np.float32)
            self.cols: "collections.deque" = collections.deque(maxlen=KEEP)
            self.end = 0                          # how many columns have been made since the last reset

    def feed(self, block: np.ndarray):
        """From the sound-card callback: what it played, (frames, channels)."""
        x = np.asarray(block, np.float32)
        self.q.append(x.mean(axis=1) if x.ndim == 2 else x.copy())

    def _update(self):
        parts = []
        while self.q:
            parts.append(self.q.popleft())
        if not parts:
            return
        self.buf = np.concatenate([self.buf] + parts)
        k = (len(self.buf) - N) // HOP + 1 if len(self.buf) >= N else 0
        if k <= 0:
            return
        idx = np.arange(N)[None, :] + HOP * np.arange(k)[:, None]
        X = np.fft.rfft(self.buf[idx] * self.win, axis=1)
        P = (X.real ** 2 + X.imag ** 2) * self.norm
        m = np.maximum.reduceat(P[:, : self.hi[-1]], self.lo, axis=1)       # each band's largest power
        m[:, self.dead] = 0.0
        db = 10 * np.log10(m + 1e-14)
        for c in np.clip((db - (TOP - RANGE)) / RANGE * 255, 0, 255).astype(np.uint8):
            self.cols.append(c)
        self.end += k
        self.buf = self.buf[k * HOP:]

    def payload(self) -> Dict[str, Any]:
        with self.lock:
            self._update()
            cols = list(self.cols)
            end = self.end
        data = np.stack(cols).tobytes() if cols else b""
        return {"end": end, "n": len(cols), "rows": ROWS, "hop_ms": HOP * 1000 / self.sr, "data": base64.b64encode(data).decode()}
