# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
The effect slots: audio effects on the model's output, after auto level and before the output
section's low-pass, limiter and volume. Up to MAX_SLOTS slots, each holding one effect, worked in
order. numpy and scipy only.

Every effect gives the same output whatever block size it is fed (512 live, a 40 ms frame
offline, a whole file in a test). Loops run in chunks of at most CHUNK samples and every loop's
delay is at least CHUNK, so a chunk reads only samples already written; filters carry their
state; anything that varies over time (the chorus's sweep, the crusher's hold, the plate's and
hall's modulation) counts samples from the start. The plate's two short input allpasses (173 and
229 samples) are outside any loop and run in pieces no longer than their delay.

The plate and the hall have three colours (COLOURS). 2020s reads the modulated delays by cubic
interpolation. 1980s adds a high shelf in each loop, set from that loop's length so every loop's
high-frequency T60 is the same fraction of the Decay (after Jot and Chaigne 1991), and mixes random
segments into the modulation. 1970s does the same with more random modulation and runs the tank at
24 kHz, band-limited to 10 kHz by an elliptic low-pass on the way in and out. A plate or hall slot
saved before the colours existed (0.1.5) has no "colour" param; clean_slot leaves it out, and the
effect then runs as 0.1.5 did (linear reads, _flush), bit for bit.

  reverb   a feedback delay network of eight lines with a Householder mix, damping in the loop,
           four allpass diffusers before it and a pre-delay (after Signalsmith's "Let's write a
           reverb"; this code is ours)
  plate    Dattorro's plate (JAES 45(9), 1997): input diffusion, a figure-eight tank of two halves with
           modulated allpasses, damping and synthetic stereo from the paper's output taps (written from
           the paper's values; this code is ours)
  hall     the reverb's eight-line network, longer, with each line's delay varied by a smooth random
           line segment (after the random modulation in Csound's reverbsc; this code is ours)
  delay    a feedback delay with a low-pass in the loop, optionally ping-pong
  chorus   two modulated delays read by fractional lookup, one per channel, in quadrature
  drive    tanh saturation, gain-matched, then a tone low-pass
  eq       low shelf, mid peak and high shelf (RBJ biquads)
  filter   low-pass, high-pass or band-pass (RBJ biquad) with resonance
  crush    bit reduction and sample-rate reduction

eq8, tape, pan, abliterated, trim and declick are described in their classes; fx_mod, fx_tone and fx_time
register further types.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, NamedTuple, Optional

import numpy as np
from scipy.signal import ellip, lfilter, sosfilt

from . import ROOT

SR = 48000
CHUNK = 240                     # 5 ms: no loop here is shorter
MAX_SLOTS = 8                   # effects in the chain at most: eight reverbs cost about 3 ms of the 10.7 ms block
SMOOTH = math.exp(-1.0 / (0.010 * SR))   # the one-pole coefficient for knob moves smoothed over 10 ms

EQ8_BANDS = [("b60", "60 Hz"), ("b150", "150 Hz"), ("b400", "400 Hz"), ("b1k", "1 kHz"), ("b2k5", "2.5 kHz"), ("b6k", "6 kHz")]
EQ8_HZ = [60.0, 150.0, 400.0, 1000.0, 2500.0, 6000.0]
EQ8_Q = 1.2                                                          # each band about 1.2 octaves wide
# TYPES[name]["params"]: [(key, label, low, high, default, scale, unit)]; scale "log" knobs travel in ratios
TYPES: Dict[str, Dict[str, Any]] = {
    "reverb": {"label": "Reverb", "params": [("size", "Decay", 0.3, 20.0, 4.0, "log", "s"), ("damp", "Damp", 0.0, 1.0, 0.5, "lin", ""),
                                             ("predelay", "Pre-delay", 0.0, 200.0, 20.0, "lin", "ms"), ("mix", "Mix", 0.0, 1.0, 0.3, "lin", "")]},
    "plate": {"label": "Plate", "params": [("colour", "Colour", 0.0, 2.0, 0.0, "choice:2020s,1980s,1970s", ""),
                                           ("decay", "Decay", 0.2, 20.0, 2.5, "log", "s"), ("size", "Size", 0.5, 2.0, 1.0, "lin", ""),
                                           ("damp", "Damp", 0.0, 1.0, 0.3, "lin", ""), ("predelay", "Pre-delay", 0.0, 200.0, 10.0, "lin", "ms"),
                                           ("mod", "Mod", 0.0, 1.0, 0.5, "lin", ""), ("mix", "Mix", 0.0, 1.0, 0.3, "lin", "")],
              "added": ("colour",)},                               # params a slot saved before them leaves out
    "hall": {"label": "Hall", "params": [("colour", "Colour", 0.0, 2.0, 0.0, "choice:2020s,1980s,1970s", ""),
                                         ("size", "Decay", 0.3, 20.0, 3.0, "log", "s"), ("damp", "Damp", 0.0, 1.0, 0.5, "lin", ""),
                                         ("predelay", "Pre-delay", 0.0, 200.0, 20.0, "lin", "ms"), ("mod", "Mod", 0.0, 1.0, 0.5, "lin", ""),
                                         ("mix", "Mix", 0.0, 1.0, 0.3, "lin", "")],
             "added": ("colour",)},
    "delay": {"label": "Delay", "params": [("time", "Time", 20.0, 1500.0, 375.0, "log", "ms"), ("feedback", "Feedback", 0.0, 0.95, 0.4, "lin", ""),
                                           ("damp", "Damp", 0.0, 1.0, 0.4, "lin", ""), ("pingpong", "Ping-pong", 0.0, 1.0, 1.0, "toggle", ""),
                                           ("mix", "Mix", 0.0, 1.0, 0.3, "lin", "")]},
    "chorus": {"label": "Chorus", "params": [("rate", "Rate", 0.05, 5.0, 0.4, "log", "Hz"), ("depth", "Depth", 0.5, 12.0, 4.0, "lin", "ms"),
                                             ("mix", "Mix", 0.0, 1.0, 0.5, "lin", "")]},
    "drive": {"label": "Drive", "params": [("amount", "Drive", 1.0, 20.0, 3.0, "log", "×"), ("tone", "Tone", 1000.0, 20000.0, 9000.0, "log", "Hz"),
                                           ("mix", "Mix", 0.0, 1.0, 1.0, "lin", "")]},
    "eq": {"label": "EQ", "params": [("low", "Low", -12.0, 12.0, 0.0, "lin", "dB"), ("mid", "Mid", -12.0, 12.0, 0.0, "lin", "dB"),
                                     ("freq", "Mid freq", 200.0, 5000.0, 1000.0, "log", "Hz"), ("high", "High", -12.0, 12.0, 0.0, "lin", "dB")]},
    "eq8": {"label": "EQ 8", "params": [("lowcut", "Low cut", 20.0, 2000.0, 20.0, "log", "Hz")]
            + [(k, l, -15.0, 15.0, 0.0, "lin", "dB") for k, l in EQ8_BANDS]
            + [("highcut", "High cut", 1000.0, 20000.0, 20000.0, "log", "Hz"), ("slope", "Slope", 0.0, 1.0, 1.0, "choice:12 dB,24 dB", "")]},
    "filter": {"label": "Filter", "params": [("mode", "Mode", 0.0, 2.0, 0.0, "choice:LP,HP,BP", ""), ("cutoff", "Cutoff", 40.0, 18000.0, 2000.0, "log", "Hz"),
                                             ("res", "Resonance", 0.5, 12.0, 0.9, "log", "Q")]},
    "crush": {"label": "Crush", "params": [("bits", "Bits", 2.0, 16.0, 8.0, "int", ""), ("hold", "Rate ÷", 1.0, 32.0, 4.0, "int", ""),
                                           ("mix", "Mix", 0.0, 1.0, 1.0, "lin", "")]},
    "tape": {"label": "Tape", "params": [("drive", "Drive", 0.0, 1.0, 0.5, "lin", ""), ("wow", "Wow", 0.0, 1.0, 0.3, "lin", ""),
                                         ("flutter", "Flutter", 0.0, 1.0, 0.1, "lin", ""), ("noise", "Hiss", 0.0, 1.0, 0.0, "lin", ""),
                                         ("mix", "Mix", 0.0, 1.0, 1.0, "lin", ""), ("speed", "Speed", 0.0, 3.0, 1.0, "choice:3.75,7.5,15,30", "")]},
    "pan": {"label": "Pan", "params": [("pos", "Pan", -1.0, 1.0, 0.0, "lin", ""), ("rate", "Auto rate", 0.02, 8.0, 0.25, "log", "Hz"),
                                       ("depth", "Auto depth", 0.0, 1.0, 0.0, "lin", "")]},
    "abliterated": {"label": "Abliterated", "params": [("crunch", "Crunch", 0.0, 1.0, 0.5, "lin", ""), ("edge", "Edge", 0.0, 1.0, 0.2, "lin", ""),
                                                       ("steer", "Steer", 150.0, 5000.0, 900.0, "log", "Hz"), ("complicate", "Complicate", 0.0, 1.0, 0.0, "lin", ""),
                                                       ("tone", "Tone", 1500.0, 12000.0, 5500.0, "log", "Hz"), ("mix", "Mix", 0.0, 1.0, 1.0, "lin", ""),
                                                       ("level", "Level", -24.0, 6.0, 0.0, "lin", "dB")]},
    "trim": {"label": "Trim", "params": [("gain", "Gain", -36.0, 24.0, 0.0, "lin", "dB"), ("balance", "Balance", -1.0, 1.0, 0.0, "lin", ""),
                                       ("width", "Width", 0.0, 400.0, 100.0, "int", "%"), ("chans", "In", 0.0, 3.0, 0.0, "choice:Stereo,L,R,Swap", ""),
                                       ("phl", "Ø L", 0.0, 1.0, 0.0, "toggle", ""), ("phr", "Ø R", 0.0, 1.0, 0.0, "toggle", ""),
                                       ("mono", "Mono", 0.0, 1.0, 0.0, "toggle", ""), ("bass", "Bass mono", 0.0, 1.0, 0.0, "toggle", ""),
                                       ("bassf", "Bass below", 50.0, 500.0, 120.0, "log", "Hz"), ("solo", "Hear bass", 0.0, 1.0, 0.0, "toggle", ""),
                                       ("dc", "DC", 0.0, 1.0, 0.0, "toggle", ""), ("mute", "Mute", 0.0, 1.0, 0.0, "toggle", "")]},
    "declick": {"label": "De-click", "params": [("sens", "Sensitivity", 0.0, 1.0, 0.5, "lin", ""), ("longest", "Longest", 0.2, 4.0, 1.0, "log", "ms")]}
}
DEFAULT_SLOTS = [{"type": t, "on": False, "params": {}} for t in ("drive", "chorus", "delay", "reverb")]


def clean_slot(s: Optional[Dict[str, Any]], i: int) -> Dict[str, Any]:
    s = s or {}
    t = s.get("type") if s.get("type") in TYPES else DEFAULT_SLOTS[i % len(DEFAULT_SLOTS)]["type"]
    given = s.get("params") or {}
    ps = {}
    for key, _, lo, hi, d, scale, _u in TYPES[t]["params"]:
        if given and key not in given and key in TYPES[t].get("added", ()):
            continue                                                 # saved before this param existed: runs as it did then
        try:
            v = float(given.get(key, d))
        except (TypeError, ValueError):
            v = d
        v = min(hi, max(lo, v))
        ps[key] = round(v) if scale == "int" or scale.startswith("choice") or scale == "toggle" else v
    out = {"type": t, "on": bool(s.get("on", False)), "params": ps}
    if isinstance(s.get("id"), int) and not isinstance(s.get("id"), bool):
        out["id"] = s["id"]                                       # the page's name for the slot, so its state survives a removal before it
    return out


def clean(slots: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """The effect chain, in order: none to MAX_SLOTS effects."""
    slots = list(slots or [])[:MAX_SLOTS]
    return [clean_slot(s, i) for i, s in enumerate(slots)]


# Named presets for each type, measured offline; data/presets/fx.json holds how, and the figures each was kept on.
# Applying one merges its params into the slot's. A type has none if the file is missing.
PRESETS_FILE = ROOT / "data" / "presets" / "fx.json"


def _file_presets() -> Dict[str, List[Dict[str, Any]]]:
    try:
        types = json.loads(PRESETS_FILE.read_text())["types"]
    except (OSError, ValueError, KeyError):
        return {}
    return {t: [{k: q[k] for k in ("name", "params", "tip") if k in q} for q in qs] for t, qs in types.items() if t in TYPES}


PRESETS = _file_presets()


def catalogue() -> Dict[str, Any]:
    return {t: {"label": v["label"], "params": [{"key": k, "label": l, "min": lo, "max": hi, "default": d, "scale": sc, "unit": u}
                                                  for k, l, lo, hi, d, sc, u in v["params"]],
                "presets": PRESETS.get(t, [])} for t, v in TYPES.items()}


class Line:
    """A delay line per channel; read(d, n) gives the n samples written d samples before the next."""

    def __init__(self, size: int, ch: int):
        self.buf = np.zeros((size, ch))
        self.w = 0

    def read(self, d: int, n: int) -> np.ndarray:
        return self.buf[(self.w - d + np.arange(n)) % len(self.buf)]

    def read_frac(self, d: np.ndarray, ch: int) -> np.ndarray:
        """Per-sample fractional delays (≥ 1) for one channel, relative to the samples just written."""
        n = len(d)
        pos = (self.w - n + np.arange(n)) - d
        i0 = np.floor(pos).astype(int)
        f = pos - i0
        a, b = self.buf[i0 % len(self.buf), ch], self.buf[(i0 + 1) % len(self.buf), ch]
        return a + (b - a) * f

    def write(self, x: np.ndarray):
        self.buf[(self.w + np.arange(len(x))) % len(self.buf)] = x
        self.w = (self.w + len(x)) % len(self.buf)


def onepole(cut_hz: float) -> float:
    return _onepole_fs(cut_hz, SR)


def damp_hz(d: float) -> float:
    return 20000.0 * (1.5 / 20.0) ** d               # 0: 20 kHz, 1: 1.5 kHz


class Effect:
    def __init__(self, ch: int):
        self.ch, self.n = ch, 0
        self.p: Dict[str, float] = {}

    def set(self, p: Dict[str, float]):
        self.p = dict(p)

    def process(self, x: np.ndarray) -> np.ndarray:
        out = np.empty_like(x)
        i = 0
        while i < len(x):
            k = min(CHUNK, len(x) - i)
            out[i:i + k] = self.chunk(x[i:i + k])
            self.n += k
            i += k
        return out

    def chunk(self, x: np.ndarray) -> np.ndarray:
        return x


class Reverb(Effect):
    LENS = [1693, 1931, 2203, 2477, 2789, 3109, 3457, 3833]       # at size 1 s: about 35 to 80 ms, mutually prime
    DIFF = [347, 467, 593, 751]

    def __init__(self, ch):
        super().__init__(ch)
        self.lines = [Line(8192, 1) for _ in self.LENS]
        self.zs = [np.zeros(1) for _ in self.LENS]
        self.dx = [Line(1024, ch) for _ in self.DIFF]
        self.dy = [Line(1024, ch) for _ in self.DIFF]
        self.pre = Line(int(0.25 * SR) + CHUNK, ch)

    def front(self, x):
        """The pre-delay and the four allpass diffusers."""
        n = len(x)
        self.pre.write(x)
        d = max(CHUNK, int(self.p["predelay"] / 1000 * SR))
        v = self.pre.read(d + n, n)
        for i, dl in enumerate(self.DIFF):
            xp, yp = self.dx[i].read(dl, n), self.dy[i].read(dl, n)
            y = -0.6 * v + xp + 0.6 * yp
            self.dx[i].write(v)
            self.dy[i].write(y)
            v = y
        return v

    def chunk(self, x):
        size, mix = self.p["size"], self.p["mix"]
        n = len(x)
        inj = self.front(x).mean(axis=1) * 0.35
        outs = np.stack([ln.read(L, n)[:, 0] for ln, L in zip(self.lines, self.LENS)])          # (8, n)
        mixed = outs - (2.0 / len(self.LENS)) * outs.sum(0, keepdims=True)                       # Householder
        a = onepole(damp_hz(self.p["damp"]))
        for j, (ln, L) in enumerate(zip(self.lines, self.LENS)):
            g = 10 ** (-3 * L / (SR * size))
            s = (mixed[j] + inj) * g
            y, self.zs[j] = lfilter([a], [1, a - 1], s, zi=self.zs[j])
            ln.write(y[:, None])
        wet = np.zeros_like(x)
        sign = np.array([1, -1, 1, -1, 1, -1, 1, -1])[:, None]
        wet[:, 0] = (outs[0::2] * sign[0::2]).sum(0) * 0.5
        wet[:, -1] = (outs[1::2] * sign[1::2]).sum(0) * 0.5
        return x * (1 - mix) + wet * mix * 1.4


def _flush(v: np.ndarray) -> np.ndarray:
    """Sets values below about 1e-34 to exactly 0, sample by sample, so a decaying tail ends in zeros."""
    return (v + 1e-18) - 1e-18


def _snap(v: np.ndarray) -> np.ndarray:
    """Sets values below 1e-30 in magnitude to exactly 0 and leaves the rest unchanged. _flush rounds every value to
    a grid of about 2e-34; in the plate at Size 2 that rounding holds values of about 2e-34 going round the tank
    for good."""
    return np.where(np.abs(v) < 1e-30, 0.0, v)


def _cubic(xm1: np.ndarray, x0: np.ndarray, x1: np.ndarray, x2: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Four-point cubic Hermite (Catmull-Rom) interpolation between x0 and x1 at fraction f."""
    c1 = 0.5 * (x1 - xm1)
    c2 = xm1 - 2.5 * x0 + 2.0 * x1 - 0.5 * x2
    c3 = 0.5 * (x2 - xm1) + 1.5 * (x0 - x1)
    return x0 + f * (c1 + f * (c2 + f * c3))


def _onepole_fs(cut_hz: float, fs: int) -> float:
    return 1.0 - math.exp(-2 * math.pi * min(cut_hz, fs / 2 - 1) / fs)


class Ring:
    """A mono delay line of power-of-two length. past(d, n) gives the n samples written d samples before
    the next write; frac(d) does the same for per-sample fractional delays, by linear interpolation, and
    cubic(d) by cubic interpolation (which reads one sample further each way, so d must exceed len(d) + 1)."""

    def __init__(self, size: int):
        self.buf = np.zeros(1 << max(1, size - 1).bit_length())
        self.mask = len(self.buf) - 1
        self.w = 0

    def past(self, d: int, n: int) -> np.ndarray:
        i = (self.w - d) & self.mask
        if i + n <= len(self.buf):
            return self.buf[i:i + n].copy()
        return np.concatenate([self.buf[i:], self.buf[:i + n - len(self.buf)]])

    def frac(self, d: np.ndarray) -> np.ndarray:
        pos = self.w + np.arange(len(d)) - d
        i0 = np.floor(pos)
        f = pos - i0
        i0 = i0.astype(np.int64)
        a, b = self.buf[i0 & self.mask], self.buf[(i0 + 1) & self.mask]
        return a + (b - a) * f

    def cubic(self, d: np.ndarray) -> np.ndarray:
        pos = self.w + np.arange(len(d)) - d
        i0 = np.floor(pos)
        f = pos - i0
        i0 = i0.astype(np.int64)
        b, m = self.buf, self.mask
        return _cubic(b[(i0 - 1) & m], b[i0 & m], b[(i0 + 1) & m], b[(i0 + 2) & m], f)

    def write(self, x: np.ndarray):
        i, n = self.w & self.mask, len(x)
        if i + n <= len(self.buf):
            self.buf[i:i + n] = x
        else:
            k = len(self.buf) - i
            self.buf[i:] = x[:k]
            self.buf[:n - k] = x[k:]
        self.w = (self.w + n) & self.mask


def _allpass(line: Ring, v: np.ndarray, d: int, g: float, fl=_flush) -> np.ndarray:
    """s = v - g s[t-d], y = g s + s[t-d], in pieces of at most d samples so each piece reads only past samples."""
    out = np.empty_like(v)
    i = 0
    while i < len(v):
        k = min(d, len(v) - i)
        sd = line.past(d, k)
        s = fl(v[i:i + k] - g * sd)
        line.write(s)
        out[i:i + k] = g * s + sd
        i += k
    return out


def _rand(key: np.ndarray) -> np.ndarray:
    """A uniform value in [-1, 1) for each uint64 key (the splitmix64 finaliser), the same however calls are split."""
    z = key * np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * 2.0 ** -52 - 1.0


def _wander(t: np.ndarray, seg, key) -> np.ndarray:
    """Random values in [-1, 1), a new one every seg samples, joined by raised-cosine curves. t is the absolute
    sample index, rising along its last axis; the values come from a hash of key plus the segment number. seg and
    key broadcast against t. The hash runs once per segment the samples touch, not once per sample."""
    k, f = np.divmod(t, seg)
    k0 = k[..., :1]
    j = k - k0
    span = int(j.max()) + 2
    r = _rand(key + (k0 + np.arange(span)).astype(np.uint64))                # (..., span)
    if r.ndim > 1:                                                           # rows: index the flattened table
        j = j + np.arange(r.shape[0])[:, None] * span
        r = r.ravel()
    r0, r1 = r[j], r[j + 1]
    w = 0.5 - 0.5 * np.cos(np.pi * f / seg)
    return r0 + (r1 - r0) * w


class Colour(NamedTuple):
    down: int           # the tank's rate is SR / down
    cubic: bool         # cubic reads of the modulated delays; linear if False
    snap: bool          # tiny values set to 0 by _snap; by _flush if False
    hf: float           # the loops' high-frequency T60 as a fraction of the Decay; 0 for no shelf
    xover: float        # the shelf's crossover, Hz
    wander: float       # the share of the plate's sine LFO given to slow random segments
    flutter: float      # fast random segments added to the modulation, as a fraction of its depth


COLOURS = {
    None: Colour(1, False, False, 0.0, 0.0, 0.0, 0.0),     # a slot saved before the colours existed: as in 0.1.5
    0: Colour(1, True, True, 0.0, 0.0, 0.0, 0.0),          # 2020s
    1: Colour(1, True, True, 0.55, 4500.0, 0.4, 0.04),     # 1980s
    2: Colour(2, True, True, 0.7, 3500.0, 0.5, 0.07),      # 1970s
}
FLUTTER_HZ = 31.0           # the fast random segments' rate: x 1 and x 1.23 for the plate's halves, x (1 + 0.11 j) for hall line j


def colour_of(p: Dict[str, float]) -> Colour:
    c = p.get("colour")
    return COLOURS[None if c is None else min(2, max(0, int(round(c))))]


def _shelf(hf: float, length, decay: float, fs: int):
    """The high-frequency gain of a loop of `length` samples whose high frequencies should die hf times as fast as
    the Decay sets (Jot and Chaigne 1991: each loop's absorption follows its length)."""
    return 10 ** (-3 * length * (1.0 / hf - 1.0) / (decay * fs))


# the 1970s band limit, on the way into the tank and on the way out: 0.2 dB ripple to 10 kHz, 70 dB down from 14 kHz.
# As one transfer function (largest pole radius 0.955), which matches the second-order sections to 2e-14 and runs
# in a third of the time.
BAND_B, BAND_A = ellip(7, 0.2, 70, 10000.0, fs=SR)


class HalfRate:
    """Decimation by two before a tank and interpolation by two after it, both through the BAND filter. The kept
    samples are those at even absolute indices, so a chunk of odd length or odd start gives the same result; filter
    states carry over from chunk to chunk."""

    def __init__(self, ch: int):
        self.zd = np.zeros(len(BAND_A) - 1)
        self.zu = np.zeros((len(BAND_A) - 1, ch))

    def down(self, v: np.ndarray, n0: int) -> np.ndarray:
        """v at SR, its first sample at absolute index n0; gives the samples at even indices, band-limited."""
        y, zd = lfilter(BAND_B, BAND_A, v, zi=self.zd)
        self.zd = _snap(zd)
        return y[n0 & 1::2]

    def up(self, w: np.ndarray, n0: int, n: int) -> np.ndarray:
        """w (k, ch) at SR / 2 back to n samples at SR, by zero-stuffing and the BAND filter."""
        u = np.zeros((n, w.shape[1]))
        u[n0 & 1::2] = 2.0 * w
        y, zu = lfilter(BAND_B, BAND_A, u, axis=0, zi=self.zu)
        self.zu = _snap(zu)
        return _snap(y)


class Plate(Effect):
    """Dattorro's plate tank. Lengths are the paper's (at 29761 Hz) scaled to the tank's rate; the tank's lengths and
    output taps also scale with Size. Left half: modulated allpass, delay, damping, gain, allpass, delay, gain,
    into the right half; the right half is the same and feeds the left. The colours (COLOURS) set the tank's rate,
    the interpolation of the modulated reads, a high shelf after each half's damping and the modulation's shape."""

    R = SR / 29761.0
    IN = [(142, 0.75), (107, 0.75), (379, 0.625), (277, 0.625)]   # input diffusion 1 and 2
    AP1 = (672, 908)                                               # modulated, decay diffusion 1 (0.70)
    D1 = (4453, 4217)
    AP2 = (1800, 2656)                                             # decay diffusion 2
    D2 = (3720, 3163)
    # output taps (Table 2): (half, element, paper offset, sign); element 0 is D1, 1 is AP2, 2 is D2
    TAPS_L = [(1, 0, 266, 1), (1, 0, 2974, 1), (1, 1, 1913, -1), (1, 2, 1996, 1), (0, 0, 1990, -1), (0, 1, 187, -1), (0, 2, 1066, -1)]
    TAPS_R = [(0, 0, 353, 1), (0, 0, 3627, 1), (0, 1, 1228, -1), (0, 2, 2673, 1), (1, 0, 2111, -1), (1, 1, 335, -1), (1, 2, 121, -1)]
    SMAX = 2.0                                                     # the largest Size
    EXC = 26.0                                                     # peak excursion at Mod 1, samples at SR (the paper's 16 at 29761 Hz)
    RATE = 1.0                                                     # modulation rate, Hz; the two halves in quadrature
    WANDER_HZ = (1.3, 1.7)                                         # the slow random segments' rate, per half
    WET = 0.6                                                      # each tap's gain, as in the paper
    KEYS = (np.arange(2, 6, dtype=np.uint64) << np.uint64(40))[:, None]   # hash keys for the four rows of random segments

    def __init__(self, ch):
        super().__init__(ch)
        self.pre = Ring(int(0.2 * SR) + 2 * CHUNK)
        self.ins = [(Ring(round(L * self.R) + CHUNK), round(L * self.R), g) for L, g in self.IN]
        room = lambda L: int(L * self.R * self.SMAX + self.EXC) + 2 * CHUNK
        self.ap1 = [Ring(room(L)) for L in self.AP1]
        self.d1 = [Ring(room(L)) for L in self.D1]
        self.ap2 = [Ring(room(L)) for L in self.AP2]
        self.d2 = [Ring(room(L)) for L in self.D2]
        self.zb = np.zeros(1)
        self.zd = [np.zeros(1), np.zeros(1)]
        self.zs = [np.zeros(1), np.zeros(1)]                       # the shelves' low-passes
        self.half = HalfRate(2)
        self.down = 1
        self.key = None

    def clear(self, down: int):
        """Empties the tank when its rate changes."""
        for r in [r for r, _, _ in self.ins] + self.ap1 + self.d1 + self.ap2 + self.d2:
            r.buf[:] = 0.0
        self.zd = [np.zeros(1), np.zeros(1)]
        self.zs = [np.zeros(1), np.zeros(1)]
        self.half = HalfRate(2)
        self.down = down

    def lengths(self, c: Colour):
        """Tank lengths, taps and gains for the current params, computed when they change."""
        key = (self.p["decay"], self.p["size"], c)
        if key != self.key:
            fs = SR // c.down
            sc = self.R * self.p["size"] / c.down
            self.ind = [round(L * self.R / c.down) for L, _ in self.IN]
            self.L = {k: [int(round(L * sc)) for L in v] for k, v in (("ap1", self.AP1), ("d1", self.D1), ("ap2", self.AP2), ("d2", self.D2))}
            total = sum(sum(v) for v in self.L.values())
            self.gain = 10 ** (-3 * (total / 4) / (self.p["decay"] * fs))   # four gain stages per trip round the tank
            self.g2 = min(0.5, max(0.25, self.gain + 0.15))
            self.taps = [[(h, e, int(round(o * sc)), sg) for h, e, o, sg in t] for t in (self.TAPS_L, self.TAPS_R)]
            if c.hf:
                self.shelf = [_shelf(c.hf, sum(self.L[k][h] for k in self.L), self.p["decay"], fs) for h in (0, 1)]
                self.cx = _onepole_fs(c.xover, fs)
            # rows: slow segments for halves 0 and 1, then fast segments for halves 0 and 1
            self.segs = np.array([round(fs / r) for r in self.WANDER_HZ] + [round(fs / FLUTTER_HZ), round(fs / (FLUTTER_HZ * 1.23))])[:, None]
            self.key = key
        return self.L

    def chunk(self, x):
        n, p = len(x), self.p
        c = colour_of(p)
        if c.down != self.down:
            self.clear(c.down)
        self.lengths(c)
        m = (x[:, 0] + x[:, -1]) * 0.5
        self.pre.write(m)
        v = self.pre.past(int(p["predelay"] / 1000 * SR) + n, n)
        bw = onepole(damp_hz(0.5 * p["damp"]))
        v, self.zb = lfilter([bw], [1, bw - 1], v, zi=self.zb)
        if c.down == 1:
            wet = self.tank(v, self.n, c)
        else:
            v = self.half.down(v, self.n)
            wet = self.half.up(self.tank(v, (self.n + 1) // 2, c) if len(v) else np.zeros((0, 2)), self.n, n)
        if x.shape[1] == 1:
            wet = wet.mean(axis=1, keepdims=True)
        mix = p["mix"]
        return x * (1 - mix) + wet * mix

    def tank(self, v: np.ndarray, t0: int, c: Colour) -> np.ndarray:
        """Input diffusion and the tank at SR / c.down; v's first sample is at index t0 at that rate. Gives (len(v), 2)."""
        n, p, L = len(v), self.p, self.L
        fs = SR // c.down
        fl = _snap if c.snap else _flush
        for (line, _, g), d in zip(self.ins, self.ind):
            v = _allpass(line, v, d, g, fl)
        gain, g1, g2 = self.gain, 0.7, self.g2
        a = _onepole_fs(damp_hz(p["damp"]), fs)
        cross = [self.d2[1].past(L["d2"][1], n) * gain, self.d2[0].past(L["d2"][0], n) * gain]
        t = t0 + np.arange(n)
        ph = 2 * math.pi * self.RATE * t / fs
        exc = p["mod"] * self.EXC / c.down
        if c.wander or c.flutter:
            rnd = _wander(t[None, :], self.segs, self.KEYS)
        for h in (0, 1):
            lfo = np.sin(ph) if h == 0 else np.cos(ph)
            if c.wander or c.flutter:
                lfo = (1 - c.wander) * lfo + c.wander * rnd[h] + c.flutter * rnd[2 + h]
            u = fl(v + cross[h])
            dl = L["ap1"][h] + exc * lfo
            sd = self.ap1[h].cubic(dl) if c.cubic else self.ap1[h].frac(dl)
            s = fl(u + g1 * sd)                                      # decay diffusion 1, with the paper's sign
            self.ap1[h].write(s)
            u = sd - g1 * s
            y = self.d1[h].past(L["d1"][h], n)
            self.d1[h].write(fl(u))
            y, self.zd[h] = lfilter([a], [1, a - 1], y, zi=self.zd[h])
            if c.hf:
                lp, self.zs[h] = lfilter([self.cx], [1, self.cx - 1], y, zi=self.zs[h])
                y = lp + self.shelf[h] * (y - lp)
            y = y * gain
            sd = self.ap2[h].past(L["ap2"][h], n)
            s = fl(y - g2 * sd)
            self.ap2[h].write(s)
            self.d2[h].write(g2 * s + sd)
        lines = [(self.d1[0], self.ap2[0], self.d2[0]), (self.d1[1], self.ap2[1], self.d2[1])]
        wet = np.empty((n, 2))
        for ci, taps in enumerate(self.taps):
            acc = np.zeros(n)
            for h, e, o, sg in taps:
                acc += sg * lines[h][e].past(o + n, n)
            wet[:, ci] = acc
        wet *= self.WET
        return wet


class Hall(Reverb):
    """The reverb's network with longer lines, each line's delay varied by its own random line segments: a new
    random target every 1 / rate seconds, joined by a raised-cosine curve. Targets are drawn from a hash of the
    line number and the segment number, so the sweep depends only on the absolute sample index. The pre-delay and
    diffusers run at SR in every colour; the colours set the lines' rate, their reads, a high shelf after each
    line's damping and a fast random layer added to the sweep."""

    LENS = [2129, 2417, 2767, 3109, 3491, 3889, 4327, 4793]        # primes, about 1.25 x the reverb's
    RATES = [0.89, 1.37, 3.97, 1.06, 2.71, 1.73, 3.13, 2.29]       # Hz, one per line
    DEPTH = [1.7, 1.2, 0.6, 1.5, 0.8, 1.3, 0.7, 1.0]               # ms of peak variation at Mod 1

    def __init__(self, ch):
        super().__init__(ch)
        self.lines = []
        self.buf = np.zeros((len(self.LENS), 8192))
        self.mask = self.buf.shape[1] - 1
        self.w = 0
        self.zs = np.zeros((len(self.LENS), 1))
        self.seg = np.array([round(SR / r) for r in self.RATES])[:, None]
        self.depth = np.array(self.DEPTH)[:, None] / 1000 * SR
        self.base = np.array(self.LENS, dtype=float)[:, None]
        self.rows = np.arange(len(self.LENS))[:, None]
        self.keyline = (np.arange(len(self.LENS), dtype=np.uint64) << np.uint64(40))[:, None]
        self.keyfast = ((np.arange(len(self.LENS), dtype=np.uint64) + np.uint64(8)) << np.uint64(40))[:, None]
        self.zsh = np.zeros((len(self.LENS), 1))                   # the shelves' low-passes
        self.half = HalfRate(2)
        self.down = 1
        self.tables = {}

    def clear(self, down: int):
        """Empties the lines when their rate changes."""
        self.buf[:] = 0.0
        self.zs = np.zeros((len(self.LENS), 1))
        self.zsh = np.zeros((len(self.LENS), 1))
        self.half = HalfRate(2)
        self.down = down

    def rate(self, down: int):
        """Line lengths, segment lengths, depths and fast segment lengths, in samples at SR / down."""
        if down not in self.tables:
            fs = SR // down
            if down == 1:
                base, seg, depth = self.base, self.seg, self.depth
            else:
                base, depth = np.round(self.base / down), self.depth / down
                seg = np.array([round(fs / r) for r in self.RATES])[:, None]
            fseg = np.array([round(fs / (FLUTTER_HZ * (1 + 0.11 * j))) for j in range(len(self.LENS))])[:, None]
            self.tables[down] = (base, seg, depth, fseg)
        return self.tables[down]

    def sweep(self, t: np.ndarray, seg: np.ndarray, depth: np.ndarray) -> np.ndarray:
        return _wander(t[None, :], seg, self.keyline) * depth

    def chunk(self, x):
        mix, n = self.p["mix"], len(x)
        c = colour_of(self.p)
        if c.down != self.down:
            self.clear(c.down)
        inj = self.front(x).mean(axis=1) * 0.35
        if c.down == 1:
            wet = self.tank(inj, self.n, c)
        else:
            inj = self.half.down(inj, self.n)
            wet = self.half.up(self.tank(inj, (self.n + 1) // 2, c) if len(inj) else np.zeros((0, 2)), self.n, n)
        out = np.zeros_like(x)
        out[:, 0] = wet[:, 0]
        out[:, -1] = wet[:, 1]
        return x * (1 - mix) + out * mix * 1.4

    def tank(self, inj: np.ndarray, t0: int, c: Colour) -> np.ndarray:
        """The eight lines at SR / c.down; inj's first sample is at index t0 at that rate. Gives (len(inj), 2)."""
        size, n = self.p["size"], len(inj)
        fs = SR // c.down
        t = t0 + np.arange(n)
        base, seg, depth, fseg = self.rate(c.down)
        sw = self.sweep(t, seg, depth)
        if c.flutter:
            sw = sw + c.flutter * _wander(t[None, :], fseg, self.keyfast) * depth
        pos = self.w + np.arange(n)[None, :] - (base + self.p["mod"] * sw)
        i0 = np.floor(pos)
        f = pos - i0
        i0 = i0.astype(np.int64)
        b, m, r = self.buf.ravel(), self.mask, self.rows * self.buf.shape[1]     # each line's start in the flat buffer
        if c.cubic:
            outs = _cubic(b[r + ((i0 - 1) & m)], b[r + (i0 & m)], b[r + ((i0 + 1) & m)], b[r + ((i0 + 2) & m)], f)
        else:
            a0, a1 = b[r + (i0 & m)], b[r + ((i0 + 1) & m)]
            outs = a0 + (a1 - a0) * f                                # (8, n), read by linear interpolation
        mixed = outs - (2.0 / len(self.LENS)) * outs.sum(0, keepdims=True)
        g = 10 ** (-3 * base / (fs * size))
        ap = _onepole_fs(damp_hz(self.p["damp"]), fs)
        y, self.zs = lfilter([ap], [1, ap - 1], (mixed + inj) * g, axis=1, zi=self.zs)
        if c.hf:
            cx = _onepole_fs(c.xover, fs)
            lp, self.zsh = lfilter([cx], [1, cx - 1], y, axis=1, zi=self.zsh)
            y = lp + _shelf(c.hf, base, size, fs) * (y - lp)
        y = (_snap if c.snap else _flush)(y)
        i = self.w & self.mask
        if i + n <= self.buf.shape[1]:
            self.buf[:, i:i + n] = y
        else:
            k = self.buf.shape[1] - i
            self.buf[:, i:] = y[:, :k]
            self.buf[:, :n - k] = y[:, k:]
        self.w = (self.w + n) & self.mask
        wet = np.empty((n, 2))
        sign = np.array([1, -1, 1, -1, 1, -1, 1, -1])[:, None]
        wet[:, 0] = (outs[0::2] * sign[0::2]).sum(0) * 0.5
        wet[:, 1] = (outs[1::2] * sign[1::2]).sum(0) * 0.5
        return wet


class Delay(Effect):
    def __init__(self, ch):
        super().__init__(ch)
        self.line = Line(2 * SR + CHUNK, ch)
        self.z = np.zeros((1, ch))

    def chunk(self, x):
        d = max(CHUNK, int(self.p["time"] / 1000 * SR))
        past = self.line.read(d, len(x))
        a = onepole(damp_hz(self.p["damp"]))
        fb, self.z = lfilter([a], [1, a - 1], past * self.p["feedback"], axis=0, zi=self.z)
        if self.p["pingpong"] >= 0.5 and x.shape[1] == 2:
            w = np.stack([x.mean(1) + fb[:, 1], fb[:, 0]], 1)        # the input enters left; the repeats cross
        else:
            w = x + fb
        self.line.write(w)
        m = self.p["mix"]
        return x * (1 - 0.5 * m) + past * m


class Chorus(Effect):
    def __init__(self, ch):
        super().__init__(ch)
        self.line = Line(SR // 2, ch)

    def chunk(self, x):
        n = len(x)
        self.line.write(x)
        t = (self.n + np.arange(n)) / SR
        base, depth = 0.012 * SR, self.p["depth"] / 1000 * SR
        out = np.empty_like(x)
        for c in range(x.shape[1]):
            ph = 2 * math.pi * self.p["rate"] * t + c * math.pi / 2
            d = base + 0.5 * depth * (1 + np.sin(ph))
            out[:, c] = self.line.read_frac(d, c)
        m = self.p["mix"]
        return x * (1 - 0.5 * m) + out * m


class Tape(Effect):
    """A tape machine's colour, written for this instrument: wow and flutter from a delay line whose length wobbles
    (two slow sines at unrelated rates, and one near 7 Hz), a slightly lopsided tanh for saturation (the lopsidedness
    adds the even harmonics), gain-matched at -26 dBFS so Drive changes colour more than level (louder material is compressed, as on tape), and a DC blocker after it; then
    the speed's EQ, a head bump in the lows and a roll-off in the highs; then hiss, read from a fixed table by sample
    position. Every part depends on the absolute sample position, so the output is the same at any block size. The
    dry signal is delayed by the same base delay, so Mix does not comb-filter."""
    BASE = 240                                                     # 5 ms: room for the wobble either side
    WOW_MS, FLUTTER_MS = 4.0, 0.3                                  # at 1: about ±2% and ±1.3% pitch
    BUMP = (35.0, 50.0, 70.0, 100.0)                               # head bump, Hz, at 3.75, 7.5, 15, 30 ips
    BUMP_DB = (4.0, 3.5, 2.5, 1.5)
    ROLL = (4000.0, 7000.0, 12000.0, 18000.0)                      # treble roll-off, Hz (two poles, then two more)
    HISS = np.random.default_rng(7).standard_normal(1 << 16)

    def __init__(self, ch):
        super().__init__(ch)
        self.line = Line(4096, ch)
        self.sos, self.zi, self.key = None, None, None
        self.peak = 0.0                                            # the largest drive into the curve since last read

    def _design(self, speed: int) -> np.ndarray:
        return np.stack([_biquad("hp", 12.0, 0.707), _biquad("peak", self.BUMP[speed], 0.9, self.BUMP_DB[speed]),
                         _biquad("lp", self.ROLL[speed], 0.6), _biquad("lp", self.ROLL[speed] * 1.4, 0.7),
                         _biquad("highshelf", 3000.0, 0.707, -2.5 + 0.7 * speed)])

    def chunk(self, x):
        n = len(x)
        self.line.write(x)
        t = (self.n + np.arange(n)) / SR
        wob = (self.p["wow"] * self.WOW_MS * (0.6 * np.sin(2 * math.pi * 0.53 * t) + 0.4 * np.sin(2 * math.pi * 0.89 * t + 1.3))
               + self.p["flutter"] * self.FLUTTER_MS * np.sin(2 * math.pi * 7.1 * t)) / 1000 * SR
        d = self.BASE + wob
        w = np.stack([self.line.read_frac(d, c) for c in range(x.shape[1])], axis=1)
        dry = self.line.read(self.BASE + n, n)
        g = 1.0 + 15.0 * self.p["drive"] ** 1.5                      # into the curve: 1 to 16
        c = 0.12 * self.p["drive"]                                    # the lopsidedness
        self.peak = max(self.peak, float(np.abs(w).max()) * g)
        ref = 0.05                                                    # gain-matched at -26 dBFS, about the model's usual level
        y = (np.tanh(g * w + c) - math.tanh(c)) * (ref / (math.tanh(g * ref + c) - math.tanh(c)))
        speed = int(round(self.p.get("speed", 2)))
        if speed != self.key:
            self.sos, self.key = self._design(speed), speed
            if self.zi is None:
                self.zi = np.zeros((len(self.sos), 2, x.shape[1]))
        y, self.zi = sosfilt(self.sos, y, axis=0, zi=self.zi)
        if self.p["noise"] > 0:
            amp = 10 ** ((-84 + 34 * self.p["noise"]) / 20)          # -84 to -50 dBFS
            idx = (self.n + np.arange(n)) % len(self.HISS)
            y = y + amp * np.stack([self.HISS[(idx + 9973 * c) % len(self.HISS)] for c in range(x.shape[1])], axis=1)
        m = self.p["mix"]
        return dry * (1 - m) + y * m


class Pan(Effect):
    """Equal-power balance of a stereo signal: at Pan 0 both sides pass at unity, at ±1 one side is silent and the
    other 3 dB up. Auto rate and depth swing the position on a sine whose phase is carried from sample to sample, so
    the swing is the same at any block size. Moves are smoothed over 10 ms. A mono signal passes as it is."""

    def __init__(self, ch):
        super().__init__(ch)
        self.ph = 0.0
        self.z = None
        self.where = 0.0                                            # the position last played, for the page

    def chunk(self, x):
        if x.shape[1] < 2:
            return x
        n = len(x)
        w = 2 * math.pi * self.p["rate"] / SR
        ph = self.ph + w * np.arange(n)
        self.ph = float((self.ph + w * n) % (2 * math.pi))
        target = np.clip(self.p["pos"] + self.p["depth"] * np.sin(ph), -1.0, 1.0)
        a = SMOOTH
        if self.z is None:
            self.z = np.array([target[0] * a])
        pos, self.z = lfilter([1 - a], [1, -a], target, zi=self.z)
        self.where = float(pos[-1])
        ang = (pos + 1) * (math.pi / 4)
        out = x.copy()
        out[:, 0] *= np.cos(ang) * math.sqrt(2)
        out[:, 1] *= np.sin(ang) * math.sqrt(2)
        return out


class Drive(Effect):
    def __init__(self, ch):
        super().__init__(ch)
        self.z = np.zeros((1, ch))

    peak = 0.0                                                       # the largest drive into the tanh since last read

    def chunk(self, x):
        k = self.p["amount"]
        if len(x):
            self.peak = max(self.peak, float(np.abs(x).max()) * k)
        y = np.tanh(k * x) / math.tanh(k) * min(1.0, 1.0 / math.sqrt(k) * 1.6)
        a = onepole(self.p["tone"])
        y, self.z = lfilter([a], [1, a - 1], y, axis=0, zi=self.z)
        m = self.p["mix"]
        return x * (1 - m) + y * m


def _biquad(kind: str, f: float, q: float = 0.707, gain_db: float = 0.0) -> np.ndarray:
    """RBJ cookbook coefficients as one sos row."""
    A = 10 ** (gain_db / 40)
    w = 2 * math.pi * min(f, SR / 2 * 0.98) / SR
    cw, sw = math.cos(w), math.sin(w)
    al = sw / (2 * q)
    if kind == "lp":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]; a = [1 + al, -2 * cw, 1 - al]
    elif kind == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]; a = [1 + al, -2 * cw, 1 - al]
    elif kind == "bp":
        b = [al, 0, -al]; a = [1 + al, -2 * cw, 1 - al]
    elif kind == "peak":
        b = [1 + al * A, -2 * cw, 1 - al * A]; a = [1 + al / A, -2 * cw, 1 - al / A]
    elif kind == "lowshelf":
        s2 = 2 * math.sqrt(A) * al
        b = [A * ((A + 1) - (A - 1) * cw + s2), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - s2)]
        a = [(A + 1) + (A - 1) * cw + s2, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - s2]
    else:                                                            # highshelf
        s2 = 2 * math.sqrt(A) * al
        b = [A * ((A + 1) + (A - 1) * cw + s2), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - s2)]
        a = [(A + 1) - (A - 1) * cw + s2, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - s2]
    return np.array([b[0] / a[0], b[1] / a[0], b[2] / a[0], 1.0, a[1] / a[0], a[2] / a[0]])


class Biquads(Effect):
    def __init__(self, ch):
        super().__init__(ch)
        self.sos = None
        self.zi = None
        self.key = None

    def design(self) -> np.ndarray:
        raise NotImplementedError

    def chunk(self, x):
        key = tuple(sorted(self.p.items()))
        if key != self.key:
            self.sos, self.key = self.design(), key
            if self.zi is None or self.zi.shape[0] != len(self.sos):
                self.zi = np.zeros((len(self.sos), 2, x.shape[1]))
        y, self.zi = sosfilt(self.sos, x, axis=0, zi=self.zi)
        return y


class EQ(Biquads):
    def design(self):
        return np.stack([_biquad("lowshelf", 200, 0.707, self.p["low"]), _biquad("peak", self.p["freq"], 0.9, self.p["mid"]),
                         _biquad("highshelf", 5000, 0.707, self.p["high"])])


class EQ8(Biquads):
    """A low cut, six bells at fixed centres and a high cut. Each cut is Butterworth, 12 or 24 dB an octave; at
    12 dB its second section is a pass-through, so the slope switch keeps the number of sections (and the
    filter state) the same."""
    def design(self):
        p, flat = self.p, np.array([1.0, 0, 0, 1.0, 0, 0])
        qs = (0.5412, 1.3066) if int(p.get("slope", 1)) else (0.7071, None)
        cut = lambda kind, f: [_biquad(kind, f, q) if q else flat for q in qs]
        bells = [_biquad("peak", f, EQ8_Q, p[k]) for (k, _), f in zip(EQ8_BANDS, EQ8_HZ)]
        return np.stack(cut("hp", p["lowcut"]) + bells + cut("lp", p["highcut"]))


class Filter(Biquads):
    def design(self):
        return np.stack([_biquad(["lp", "hp", "bp"][int(self.p["mode"])], self.p["cutoff"], self.p["res"])])


class Crush(Effect):
    def __init__(self, ch):
        super().__init__(ch)
        self.held = np.zeros(ch)

    def chunk(self, x):
        h = max(1, int(self.p["hold"]))
        idx = self.n + np.arange(len(x))
        rel = idx - idx % h - idx[0]                                 # sample-and-hold, counted from the start
        out = np.where((rel >= 0)[:, None], x[np.maximum(rel, 0)], self.held[None, :])
        self.held = out[-1].copy()
        q = 2 ** (int(self.p["bits"]) - 1)
        out = np.round(out * q) / q
        m = self.p["mix"]
        return x * (1 - m) + out * m


class Abliterated(Effect):
    """Heavy distortion, written for this instrument. Steer sets where it bites: a resonant boost (+12 dB, Q 1.4) before
    the clipping, after a low cut at 90 Hz that keeps the lows tight. Crunch drives two lopsided tanh stages with a
    high-pass at 140 Hz between them, from about 2x to 300x (+50 dB) in all. Edge blends in a sine wavefolder, folding
    the clipped wave back on itself up to seven times, which rings like metal. Complicate adds an octave up (the wave
    rectified) over its first half and, over its second, a ring modulator at 1.5 times Steer, whose sum and difference
    tones are clangorous. Tone is a two-pole low-pass, standing in for a speaker cabinet, after a DC blocker. A make-up
    gain follows the input's power against the output's (300 ms, held between 0.001 and 10), so the controls change the
    character more than the level; Mix blends with the dry sound (parallel distortion) and Level trims the output. Every stage carries its state or reads the absolute
    sample position, so the output is the same at any block size."""
    REF = 0.05

    def __init__(self, ch):
        super().__init__(ch)
        self.sos, self.zi, self.key = None, None, None
        self.peak = 0.0
        self.env = np.array([[1e-6], [1e-6]])           # the input's and the output's power, followed

    FOLLOW = math.exp(-1.0 / (0.3 * SR))

    def _design(self, steer: float, tone: float):
        pre = np.stack([_biquad("hp", 90.0, 0.707), _biquad("peak", steer, 1.4, 12.0)])
        mid = np.stack([_biquad("hp", 140.0, 0.707)])
        post = np.stack([_biquad("hp", 20.0, 0.707), _biquad("lp", tone, 0.8), _biquad("lp", tone * 1.3, 0.6)])
        return pre, mid, post

    def chunk(self, x):
        p, n = self.p, len(x)
        key = (round(p["steer"], 1), round(p["tone"], 1))
        if key != self.key:
            self.sos, self.key = self._design(*key), key
            if self.zi is None:
                self.zi = [np.zeros((len(s), 2, x.shape[1])) for s in self.sos]
        y, self.zi[0] = sosfilt(self.sos[0], x / self.REF * 0.25, axis=0, zi=self.zi[0])   # -26 dBFS in reads as 0.25
        g = 2.0 + 298.0 * p["crunch"] ** 2
        g1, g2 = math.sqrt(g), math.sqrt(g)
        self.peak = max(self.peak, float(np.abs(y).max()) * g) if n else self.peak
        y = np.tanh(g1 * y + 0.2) - math.tanh(0.2)                                     # stage one, lopsided: even harmonics
        y, self.zi[1] = sosfilt(self.sos[1], y, axis=0, zi=self.zi[1])
        y = np.tanh(g2 * y - 0.1) + math.tanh(0.1)                                     # stage two, the other way
        e = p["edge"]
        if e > 0:
            y = (1 - e) * y + e * np.sin((1 + 6 * e) * (math.pi / 2) * y)
        c = p["complicate"]
        if c > 0:
            octave = np.abs(y) - 0.5                                                    # the rectified wave: an octave up
            y = y * (1 - 0.4 * min(1.0, 2 * c)) + 0.8 * min(1.0, 2 * c) * octave
            r = max(0.0, 2 * c - 1)
            if r > 0:
                t = (self.n + np.arange(n)) / SR
                y = y * (1 - 0.6 * r) + 0.9 * r * y * np.sin(2 * math.pi * 1.5 * p["steer"] * t)[:, None]
        y, self.zi[2] = sosfilt(self.sos[2], y, axis=0, zi=self.zi[2])
        a = self.FOLLOW
        pin, self.env[0] = lfilter([1 - a], [1, -a], (x.astype(np.float64) ** 2).mean(axis=1), zi=self.env[0])
        pout, self.env[1] = lfilter([1 - a], [1, -a], (y ** 2).mean(axis=1), zi=self.env[1])
        mk = np.clip(np.sqrt((pin + 1e-12) / (pout + 1e-12)), 0.001, 10.0)
        y = y * (mk * 10 ** (p["level"] / 20))[:, None]
        m = p["mix"]
        return x * (1 - m) + y * m


class Trim(Effect):
    """Gain staging and the stereo field, with no sound of its own. In order: In (Stereo, L or R on both
    sides, or Swap), a phase flip on each side, Width on the mid/side split (0% is mono, 100% as it came, up to 400%;
    Mono sets it to 0), Bass mono (the side below Bass below taken out with a 4th-order high-pass, so the low end is
    the same on both sides), Hear bass (only the mid below that frequency, to hear what Bass mono keeps), DC (a 5 Hz
    high-pass), then Gain, Balance (the far side turned down, the near side kept at unity) and Mute. Every switch and
    knob moves over 10 ms, so none of them clicks, and the output is the same at any block size. A mono signal takes
    the phase flip on L, DC, Gain and Mute."""
    MATS = {0: (1.0, 0.0, 0.0, 1.0), 1: (1.0, 0.0, 1.0, 0.0), 2: (0.0, 1.0, 0.0, 1.0), 3: (0.0, 1.0, 1.0, 0.0)}   # L' = a L + b R, R' = c L + d R

    def __init__(self, ch):
        super().__init__(ch)
        self.z: Dict[str, Any] = {}
        self.sf = None                                              # Bass below, as the filters were last made for
        self.zs = self.zl = None
        self.zdc = np.zeros((1, 2, ch))
        self.dc = np.array(_sos("hp", 5.0, 1))

    def _sm(self, key: str, target: float, n: int) -> np.ndarray:
        a = SMOOTH
        if key not in self.z:
            self.z[key] = np.array([target * a])
        y, self.z[key] = lfilter([1 - a], [1, -a], np.full(n, target), zi=self.z[key])
        return y

    def _filters(self):
        f = float(self.p["bassf"])
        if f != self.sf:
            self.hp, self.lp, self.sf = _sos("hp", f, 2), _sos("lp", f, 2), f
            if self.zs is None:
                self.zs, self.zl = np.zeros((2, 2)), np.zeros((2, 2))

    def chunk(self, x):
        p, n = self.p, len(x)
        x = x.astype(np.float64, copy=True)
        if x.shape[1] >= 2:
            self._filters()
            m = self.MATS[int(p["chans"])]
            a, b, c, d = (self._sm(k, m[j], n) for j, k in enumerate(("ma", "mb", "mc", "md")))
            L, R = x[:, 0].copy(), x[:, 1].copy()
            L, R = a * L + b * R, c * L + d * R
            L *= self._sm("pl", -1.0 if p["phl"] else 1.0, n)
            R *= self._sm("pr", -1.0 if p["phr"] else 1.0, n)
            mid, side = (L + R) * 0.5, (L - R) * 0.5
            side = side * self._sm("w", 0.0 if p["mono"] else p["width"] / 100.0, n)
            hs, self.zs = sosfilt(self.hp, side, zi=self.zs)
            kb = self._sm("bass", 1.0 if p["bass"] else 0.0, n)
            side = side * (1 - kb) + hs * kb
            lm, self.zl = sosfilt(self.lp, mid, zi=self.zl)
            ks = self._sm("solo", 1.0 if p["solo"] else 0.0, n)
            L = (mid + side) * (1 - ks) + lm * ks
            R = (mid - side) * (1 - ks) + lm * ks
            x[:, 0], x[:, 1] = L, R
        else:
            x[:, 0] *= self._sm("pl", -1.0 if p["phl"] else 1.0, n)
        hd, self.zdc = sosfilt(self.dc, x, axis=0, zi=self.zdc)
        kd = self._sm("dc", 1.0 if p["dc"] else 0.0, n)[:, None]
        x = x * (1 - kd) + hd * kd
        g = self._sm("g", 0.0 if p["mute"] else 10 ** (p["gain"] / 20.0), n)
        if x.shape[1] >= 2:
            bal = self._sm("bal", p["balance"], n)
            x[:, 0] *= g * np.minimum(1.0, 1.0 - bal)
            x[:, 1] *= g * np.minimum(1.0, 1.0 + bal)
        else:
            x[:, 0] *= g
        return x


def _sos(kind: str, f: float, order2: int) -> np.ndarray:
    """`order2` cascaded 2nd-order Butterworth sections at `f` (kind "hp" or "lp"), as sos rows."""
    from scipy.signal import butter
    return np.vstack([butter(2, f, btype="highpass" if kind == "hp" else "lowpass", fs=SR, output="sos")] * order2)


CLASSES = {"reverb": Reverb, "plate": Plate, "hall": Hall, "delay": Delay, "chorus": Chorus, "drive": Drive, "eq": EQ, "eq8": EQ8, "pan": Pan, "tape": Tape, "filter": Filter, "crush": Crush, "abliterated": Abliterated, "trim": Trim}


# Params whose change jumps a delay line's read point. With fades on, the effect's wet signal is
# taken out before such a change and brought back after it, so the jump is not heard as a click.
JUMPS = {"plate": {"size", "predelay", "colour"}, "reverb": {"predelay"}, "hall": {"predelay", "colour"}, "delay": {"time"}}
FADE = 960                      # 20 ms: an effect's wet signal in or out (on, off, or around a jump)
JUMP_HOLD = 2880                # 60 ms out after the last jump, so a turning knob is not heard in steps
DIP_UP = 7200                   # 150 ms: the chain's level back up after it changes
DIP_DOWN_STEP = 10              # down ten times faster (15 ms)
DIP_HOLD = 1920                 # 40 ms at the bottom
DIP_FLOOR = 0.16                # about -16 dB


def _ramp(k: int, up: bool, n: int, top: int, step: int = 1) -> np.ndarray:
    """k moved one step a sample toward top (up) or 0, for n samples; integers, so the result does
    not depend on how the samples are split into blocks."""
    r = np.arange(1, n + 1, dtype=np.int64) * step
    return np.minimum(k + r, top) if up else np.maximum(k - r, 0)


class Declicker(Effect):
    """dsp.Declick as an effect: the few-sample bursts a unit can make the model write, redrawn as a
    line between the good samples around them, anywhere in the chain. Sensitivity sets the threshold
    (0: a burst must stand 12 times above the local level, 0.5: 6 times, as Output's De-click,
    1: 3 times); Longest, the longest burst redrawn. 64 samples (1.3 ms) late, like Output's."""

    def __init__(self, ch):
        super().__init__(ch)
        from .dsp import Declick                                   # here, since dsp imports this module
        self.dc = Declick(ch)

    def process(self, x):
        self.dc.THRESH = 12.0 / 4.0 ** self.p["sens"]
        self.dc.MAX = max(4, int(round(self.p["longest"] / 1000 * SR)))
        y = self.dc.process(x).reshape(x.shape).astype(x.dtype)
        self.n += len(x)
        return y


CLASSES["declick"] = Declicker


class Rack:
    """The effect chain, with each effect's state. An effect keeps its state while it stays in the
    chain: matched by the slot's id where the page gives one, else by place; a new one, or one whose
    type changes, starts fresh.

    With fades on (a Master's fx_fade, set for live play and saved with the sound), a change
    to the chain itself (an effect added, removed, moved or retyped) dips the whole output and is
    made at the bottom of the dip; switching an effect on or off crossfades it; and a param in JUMPS
    is changed while that effect's wet signal is faded out. Every event falls on an exact sample, so
    the output is the same at any block size. With fades off every change is made at once, as sounds
    saved before fades existed expect."""

    def __init__(self, ch: int):
        self.ch = ch
        self.slots: List[Dict[str, Any]] = []
        self.fx: List[Optional[Effect]] = []
        self.smooth = False
        self.t = 0                                   # samples processed
        self.k: List[int] = []                       # each effect's wet level, 0..FADE
        self.hold: List[int] = []                    # each effect stays out until this sample
        self.pend: List[Optional[Dict[str, float]]] = []
        self.next: Optional[tuple] = None            # a chain waiting for the bottom of the dip
        self.g = DIP_UP                              # the chain's level, 0..DIP_UP
        self.ghold = 0
        self.on = True                               # the Effects panel's switch: off passes the sound by the chain
        self.kon = FADE                              # the chain's share of the output, 0..FADE

    def _match(self, new: List[Dict[str, Any]]) -> List[tuple]:
        """For each new slot, the effect it continues (or None) and that effect's old slot."""
        pool = list(zip(self.slots, self.fx, range(len(self.slots))))
        if self.next is not None:
            pool += [(s, f, None) for s, f in zip(self.next[0], self.next[1]) if not any(f is g for _, g, _ in pool)]
        by_id = {s["id"]: (s, f, i) for s, f, i in pool if "id" in s}
        out = []
        for i, s in enumerate(new):
            if "id" in s:
                m = by_id.get(s["id"])
            else:
                m = pool[i] if i < len(pool) and "id" not in pool[i][0] else None
            out.append(m if m is not None and m[0]["type"] == s["type"] else None)
        return out

    def update(self, slots: List[Dict[str, Any]]):
        new = clean(slots)
        match = self._match(new)
        if not self.smooth or self.t == 0:
            fx = []
            for s, m in zip(new, match):
                f = m[1] if m else CLASSES[s["type"]](self.ch)
                f.set(s["params"])
                fx.append(f)
            self.slots, self.fx, self.next = new, fx, None
            self.k = [FADE if s["on"] else 0 for s in new]
            self.hold, self.pend = [0] * len(new), [None] * len(new)
            return
        same = len(new) == len(self.slots) and all(m is not None and m[2] == i for i, m in enumerate(match))
        if not same:                                  # the chain itself changes: at the bottom of a dip
            fx = [m[1] if m else CLASSES[s["type"]](self.ch) for s, m in zip(new, match)]
            for f, s in zip(fx, new):
                f.set(s["params"])
            self.next = (new, fx)
            return
        self.next = None
        for i, (s, f) in enumerate(zip(new, self.fx)):
            old = self.slots[i]["params"]
            jump = {k for k in JUMPS.get(s["type"], ()) if s["params"].get(k) != old.get(k)}
            if jump and (self.k[i] > 0 or self.pend[i] is not None):
                self.pend[i] = dict(s["params"])
                held = {k: v for k, v in s["params"].items() if k not in jump or k in old}
                held.update({k: old[k] for k in jump if k in old})      # a key the old params lack (a 0.1.5 slot's colour) stays out
                f.set(held)
                s = {**s, "params": held}
            else:
                f.set(s["params"])
                if jump:
                    self.hold[i] = self.t + JUMP_HOLD
            new[i] = s
        self.slots = new

    def active(self) -> bool:
        return any(s["on"] for s in self.slots) or self.next is not None

    def readings(self) -> List[Dict[str, Any]]:
        """For the page, per effect in the chain: Drive's and Tape's peak drive into the curve since last
        read (above 1 it bends, above 3 it is flat at the top); Pan's position now, -1 left to 1 right; the
        Compressor's largest gain reduction since last read, in dB; De-click's bursts redrawn since it was
        added. Reading resets the peak and the reduction."""
        out = []
        for s, f in zip(self.slots, self.fx):
            r: Dict[str, Any] = {"id": s.get("id"), "type": s["type"]}
            if isinstance(f, (Drive, Tape)):
                r["drive"] = round(f.peak, 2)
                f.peak = 0.0
            elif isinstance(f, Pan):
                r["pos"] = round(f.where, 3)
            elif isinstance(f, fx_tone.Compressor):
                r["gr"] = round(f.gr, 1)
                f.gr = 0.0
            elif isinstance(f, Declicker):
                r["fixed"] = f.dc.repaired
            out.append(r)
        return out

    def wanted(self) -> List[Dict[str, Any]]:
        """The chain as last set: a change waiting for the bottom of a dip, or a param waiting for its
        effect's wet signal to fade, is reported as made. While nothing plays no audio reaches the
        rack, so the change waits; the page must not read the old chain back and undo it."""
        if self.next is not None:
            return [dict(s) for s in self.next[0]]
        return [dict(s, params=p) if p is not None else dict(s) for s, p in zip(self.slots, self.pend)]

    def _slot(self, i: int, x: np.ndarray, t0: int) -> np.ndarray:
        """Effect i on x (which starts at sample t0), with its wet level, its pending jump and its hold."""
        f, on = self.fx[i], self.slots[i]["on"]
        out, pos, n = np.empty_like(x), 0, len(x)
        while pos < n:
            if self.pend[i] is not None and self.k[i] == 0:
                f.set(self.pend[i])
                self.slots[i] = {**self.slots[i], "params": self.pend[i]}
                self.pend[i], self.hold[i] = None, t0 + pos + JUMP_HOLD
            t = t0 + pos
            up = on and self.pend[i] is None and t >= self.hold[i]
            if self.pend[i] is not None:
                L = self.k[i]
            elif on and t < self.hold[i]:
                L = self.hold[i] - t
            else:
                L = n - pos
            L = max(1, min(L, n - pos))
            xs = x[pos:pos + L]
            y = f.process(xs)
            ks = _ramp(self.k[i], up, L, FADE)
            if self.k[i] == FADE and ks[-1] == FADE:
                out[pos:pos + L] = y
            elif self.k[i] == 0 and ks[-1] == 0:
                out[pos:pos + L] = xs
            else:
                g = (ks / FADE).astype(x.dtype)[:, None]
                out[pos:pos + L] = xs + g * (y - xs)
            self.k[i] = int(ks[-1])
            pos += L
        return out

    def process(self, x: np.ndarray) -> np.ndarray:
        if not self.smooth:
            dry = x
            for s, f in zip(self.slots, self.fx):
                if f is None:
                    continue
                y = f.process(x)                                     # every slot runs, so turning one on starts with a settled tail
                if s["on"]:
                    x = y
            self.t += len(x)
            return x if self.on else dry
        out, pos, n = np.empty_like(x), 0, len(x)
        while pos < n:
            t = self.t + pos
            if self.next is not None and self.g == 0:
                new, fx = self.next
                old = {id(f): i for i, f in enumerate(self.fx)}
                self.k = [self.k[old[id(f)]] if id(f) in old else 0 for f in fx]
                self.hold = [self.hold[old[id(f)]] if id(f) in old else 0 for f in fx]
                self.pend = [None] * len(fx)
                self.slots, self.fx, self.next, self.ghold = new, fx, None, t + DIP_HOLD
            up = self.next is None and t >= self.ghold
            if self.next is not None:
                L = -(-self.g // DIP_DOWN_STEP)
            elif t < self.ghold:
                L = self.ghold - t
            else:
                L = n - pos
            L = max(1, min(L, n - pos))
            xin = y = x[pos:pos + L]
            for i in range(len(self.fx)):
                y = self._slot(i, y, t)
            ks = _ramp(self.kon, self.on, L, FADE)                  # the panel's switch, crossfaded
            if not (self.kon == FADE and ks[-1] == FADE):
                y = xin if self.kon == 0 and ks[-1] == 0 else xin + (ks / FADE).astype(x.dtype)[:, None] * (y - xin)
            self.kon = int(ks[-1])
            gs = _ramp(self.g, up, L, DIP_UP, 1 if up else DIP_DOWN_STEP)
            if not (self.g == DIP_UP and gs[-1] == DIP_UP):
                y = y * (DIP_FLOOR + (1 - DIP_FLOOR) * (gs / DIP_UP)).astype(x.dtype)[:, None]
            out[pos:pos + L] = y
            self.g = int(gs[-1])
            pos += L
        self.t += n
        return out


# The effects in their own modules: Phaser, Flanger, Tremolo (fx_mod); Compressor, Resonator, Freq Shift
# (fx_tone); Pitch, Shimmer, Freeze (fx_time). Each imports this module for its base classes and, once it has loaded,
# registers its types here with _register, so it works whichever is imported first.
def _register(m) -> None:
    TYPES.update(m.TYPES_NEW)
    CLASSES.update(m.CLASSES_NEW)
    JUMPS.update(getattr(m, "JUMPS_NEW", {}))
    PRESETS.update({t: v for t, v in _file_presets().items() if t in m.TYPES_NEW})


from . import fx_mod, fx_time, fx_tone  # noqa: E402,F401
