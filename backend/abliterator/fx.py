# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Track effects: conventional audio processing on a track's sound, after the model. They are not
strips (strips act inside the model). All run block by block on 48 kHz stereo, carry their state
between blocks so a block-by-block run equals a whole-file run, and add no latency beyond the
delays they are set to make.

Chain, in order: high-pass, low-pass (2nd-order Butterworth), delay (feedback), reverb (four
feedback combs and two allpasses per channel, after Schroeder and Moorer, without damping), gain,
pan (equal power). Every delay line is longer than CHUNK, so each chunk is computed as whole
arrays from the line's past.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

SR = 48000
CHUNK = 256
DEFAULTS: Dict[str, Any] = {"gain_db": 0.0, "pan": 0.0, "highpass_hz": 20.0, "lowpass_hz": 20000.0,
                            "delay_ms": 375.0, "delay_feedback": 0.35, "delay_mix": 0.0,
                            "reverb_size": 0.6, "reverb_mix": 0.0}
RANGES = {"gain_db": (-48.0, 12.0), "pan": (-1.0, 1.0), "highpass_hz": (20.0, 5000.0), "lowpass_hz": (500.0, 20000.0),
          "delay_ms": (20.0, 1500.0), "delay_feedback": (0.0, 0.9), "delay_mix": (0.0, 1.0),
          "reverb_size": (0.0, 1.0), "reverb_mix": (0.0, 1.0)}


def clean(p: Dict[str, Any]) -> Dict[str, float]:
    out = dict(DEFAULTS)
    for k, v in (p or {}).items():
        if k in RANGES:
            lo, hi = RANGES[k]
            out[k] = float(min(hi, max(lo, float(v))))
    return out


class Line:
    """A delay line per channel; read(d, n) gives the n samples written d samples before the next."""

    def __init__(self, size: int, ch: int = 2):
        self.buf = np.zeros((size, ch), np.float32)
        self.w = 0

    def read(self, d: int, n: int) -> np.ndarray:
        idx = (self.w - d + np.arange(n)) % len(self.buf)
        return self.buf[idx]

    def write(self, x: np.ndarray):
        idx = (self.w + np.arange(len(x))) % len(self.buf)
        self.buf[idx] = x
        self.w = (self.w + len(x)) % len(self.buf)


class Filter:
    def __init__(self, kind: str):
        self.kind, self.hz, self.sos, self.zi = kind, None, None, None

    def process(self, x: np.ndarray, hz: float) -> np.ndarray:
        if hz != self.hz:
            self.sos = butter(2, hz, btype=self.kind, fs=SR, output="sos")
            if self.zi is None:
                self.zi = np.repeat(sosfilt_zi(self.sos)[:, :, None], x.shape[1], axis=2) * 0.0
            self.hz = hz
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y.astype(np.float32)


COMBS = [1617, 1557, 1491, 1422]        # Moorer's / Freeverb's comb lengths at 44.1 kHz, scaled to 48 kHz below
ALLPASSES = [556, 441]
SPREAD = 23


class Reverb:
    def __init__(self):
        s = SR / 44100
        self.combs = [[int(c * s) + ch * SPREAD for ch in range(2)] for c in COMBS]
        self.aps = [[int(a * s) + ch * SPREAD for ch in range(2)] for a in ALLPASSES]
        self.cl = [[Line(4096, 1) for _ in range(2)] for _ in COMBS]
        self.ax = [[Line(2048, 1) for _ in range(2)] for _ in ALLPASSES]
        self.ay = [[Line(2048, 1) for _ in range(2)] for _ in ALLPASSES]

    def process(self, x: np.ndarray, size: float) -> np.ndarray:
        g = 0.70 + 0.28 * size
        n = len(x)
        out = np.zeros_like(x)
        for ch in range(2):
            xi = x[:, ch:ch + 1] * 0.25
            acc = np.zeros((n, 1), np.float32)
            for i, d in enumerate(c[ch] for c in self.combs):
                past = self.cl[i][ch].read(d, n)
                y = xi + g * past                            # y[n] = x[n] + g · y[n - d]
                self.cl[i][ch].write(y)
                acc += y
            for i, d in enumerate(a[ch] for a in self.aps):
                gx = 0.5
                xp, yp = self.ax[i][ch].read(d, n), self.ay[i][ch].read(d, n)
                y = -gx * acc + xp + gx * yp                 # Schroeder allpass
                self.ax[i][ch].write(acc)
                self.ay[i][ch].write(y)
                acc = y
            out[:, ch:ch + 1] = acc
        return out


class Chain:
    """One track's effects, with their state."""

    def __init__(self):
        self.hp, self.lp = Filter("highpass"), Filter("lowpass")
        self.dl = Line(SR * 2)
        self.rv = Reverb()

    def process(self, x: np.ndarray, p: Dict[str, float]) -> np.ndarray:
        out = []
        for i in range(0, len(x), CHUNK):
            out.append(self._chunk(x[i:i + CHUNK], p))
        return np.concatenate(out) if out else x

    def _chunk(self, x: np.ndarray, p: Dict[str, float]) -> np.ndarray:
        y = x
        if p["highpass_hz"] > 20.5:
            y = self.hp.process(y, p["highpass_hz"])
        if p["lowpass_hz"] < 19999:
            y = self.lp.process(y, p["lowpass_hz"])
        d = max(CHUNK, int(p["delay_ms"] * SR / 1000))
        past = self.dl.read(d, len(y))
        self.dl.write(y + p["delay_feedback"] * past)        # the line always runs, so a delay set later starts clean of clicks
        if p["delay_mix"] > 0:
            y = y + p["delay_mix"] * past
        if p["reverb_mix"] > 0:
            y = y + p["reverb_mix"] * self.rv.process(y, p["reverb_size"])
        g = 10 ** (p["gain_db"] / 20)
        a = (p["pan"] + 1) * np.pi / 4
        return (y * g * np.array([np.cos(a), np.sin(a)], np.float32) * np.sqrt(2)).astype(np.float32)
