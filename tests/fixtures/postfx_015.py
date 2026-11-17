# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
The effect slots: audio effects on the model's output, after auto level and before the output
section's low-pass, limiter and volume. Four slots, each holding one effect, worked in order.
numpy and scipy only.

Every effect gives the same output whatever block size it is fed (512 live, a 40 ms frame
offline, a whole file in a test). Loops run in chunks of at most CHUNK samples and every loop's
delay is at least CHUNK, so a chunk reads only samples already written; filters carry their
state; anything that varies over time (the chorus's sweep, the crusher's hold, the plate's and
hall's modulation) counts samples from the start. The plate's two short input allpasses (173 and
229 samples) are outside any loop and run in pieces no longer than their delay.

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
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.signal import lfilter, sosfilt

SR = 48000
CHUNK = 240                     # 5 ms: no loop here is shorter
N_SLOTS = 4

# type: [(key, label, low, high, default, scale)]; scale "log" knobs travel in ratios
TYPES: Dict[str, Dict[str, Any]] = {
    "reverb": {"label": "Reverb", "params": [("size", "Decay", 0.3, 20.0, 4.0, "log", "s"), ("damp", "Damp", 0.0, 1.0, 0.5, "lin", ""),
                                             ("predelay", "Pre-delay", 0.0, 200.0, 20.0, "lin", "ms"), ("mix", "Mix", 0.0, 1.0, 0.3, "lin", "")]},
    "plate": {"label": "Plate", "params": [("decay", "Decay", 0.2, 20.0, 2.5, "log", "s"), ("size", "Size", 0.5, 2.0, 1.0, "lin", ""),
                                           ("damp", "Damp", 0.0, 1.0, 0.3, "lin", ""), ("predelay", "Pre-delay", 0.0, 200.0, 10.0, "lin", "ms"),
                                           ("mod", "Mod", 0.0, 1.0, 0.5, "lin", ""), ("mix", "Mix", 0.0, 1.0, 0.3, "lin", "")]},
    "hall": {"label": "Hall", "params": [("size", "Decay", 0.3, 20.0, 3.0, "log", "s"), ("damp", "Damp", 0.0, 1.0, 0.5, "lin", ""),
                                         ("predelay", "Pre-delay", 0.0, 200.0, 20.0, "lin", "ms"), ("mod", "Mod", 0.0, 1.0, 0.5, "lin", ""),
                                         ("mix", "Mix", 0.0, 1.0, 0.3, "lin", "")]},
    "delay": {"label": "Delay", "params": [("time", "Time", 20.0, 1500.0, 375.0, "log", "ms"), ("feedback", "Feedback", 0.0, 0.95, 0.4, "lin", ""),
                                           ("damp", "Damp", 0.0, 1.0, 0.4, "lin", ""), ("pingpong", "Ping-pong", 0.0, 1.0, 1.0, "toggle", ""),
                                           ("mix", "Mix", 0.0, 1.0, 0.3, "lin", "")]},
    "chorus": {"label": "Chorus", "params": [("rate", "Rate", 0.05, 5.0, 0.4, "log", "Hz"), ("depth", "Depth", 0.5, 12.0, 4.0, "lin", "ms"),
                                             ("mix", "Mix", 0.0, 1.0, 0.5, "lin", "")]},
    "drive": {"label": "Drive", "params": [("amount", "Drive", 1.0, 20.0, 3.0, "log", "×"), ("tone", "Tone", 1000.0, 20000.0, 9000.0, "log", "Hz"),
                                           ("mix", "Mix", 0.0, 1.0, 1.0, "lin", "")]},
    "eq": {"label": "EQ", "params": [("low", "Low", -12.0, 12.0, 0.0, "lin", "dB"), ("mid", "Mid", -12.0, 12.0, 0.0, "lin", "dB"),
                                     ("freq", "Mid freq", 200.0, 5000.0, 1000.0, "log", "Hz"), ("high", "High", -12.0, 12.0, 0.0, "lin", "dB")]},
    "filter": {"label": "Filter", "params": [("mode", "Mode", 0.0, 2.0, 0.0, "choice:LP,HP,BP", ""), ("cutoff", "Cutoff", 40.0, 18000.0, 2000.0, "log", "Hz"),
                                             ("res", "Resonance", 0.5, 12.0, 0.9, "log", "Q")]},
    "crush": {"label": "Crush", "params": [("bits", "Bits", 2.0, 16.0, 8.0, "int", ""), ("hold", "Rate ÷", 1.0, 32.0, 4.0, "int", ""),
                                           ("mix", "Mix", 0.0, 1.0, 1.0, "lin", "")]},
}
DEFAULT_SLOTS = [{"type": t, "on": False, "params": {}} for t in ("drive", "chorus", "delay", "reverb")]


def clean_slot(s: Optional[Dict[str, Any]], i: int) -> Dict[str, Any]:
    s = s or {}
    t = s.get("type") if s.get("type") in TYPES else DEFAULT_SLOTS[i]["type"]
    given = s.get("params") or {}
    ps = {}
    for key, _, lo, hi, d, scale, _u in TYPES[t]["params"]:
        try:
            v = float(given.get(key, d))
        except (TypeError, ValueError):
            v = d
        v = min(hi, max(lo, v))
        ps[key] = round(v) if scale == "int" or scale.startswith("choice") or scale == "toggle" else v
    return {"type": t, "on": bool(s.get("on", False)), "params": ps}


def clean(slots: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    slots = list(slots or [])
    return [clean_slot(slots[i] if i < len(slots) else None, i) for i in range(N_SLOTS)]


def catalogue() -> Dict[str, Any]:
    return {t: {"label": v["label"], "params": [{"key": k, "label": l, "min": lo, "max": hi, "default": d, "scale": sc, "unit": u}
                                                  for k, l, lo, hi, d, sc, u in v["params"]]} for t, v in TYPES.items()}


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
    return 1.0 - math.exp(-2 * math.pi * min(cut_hz, SR / 2 - 1) / SR)


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
        v = self.pre.read(d + n, n) if d else x                      # the pre-delayed input
        for i, dl in enumerate(self.DIFF):                           # allpass diffusers, g 0.6
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


class Ring:
    """A mono delay line of power-of-two length. past(d, n) gives the n samples written d samples before
    the next write; frac(d) does the same for per-sample fractional delays, by linear interpolation."""

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

    def write(self, x: np.ndarray):
        i, n = self.w & self.mask, len(x)
        if i + n <= len(self.buf):
            self.buf[i:i + n] = x
        else:
            k = len(self.buf) - i
            self.buf[i:] = x[:k]
            self.buf[:n - k] = x[k:]
        self.w = (self.w + n) & self.mask


def _allpass(line: Ring, v: np.ndarray, d: int, g: float) -> np.ndarray:
    """s = v - g s[t-d], y = g s + s[t-d], in pieces of at most d samples so each piece reads only past samples."""
    out = np.empty_like(v)
    i = 0
    while i < len(v):
        k = min(d, len(v) - i)
        sd = line.past(d, k)
        s = _flush(v[i:i + k] - g * sd)
        line.write(s)
        out[i:i + k] = g * s + sd
        i += k
    return out


class Plate(Effect):
    """Dattorro's plate tank. Lengths are the paper's (at 29761 Hz) scaled to 48 kHz; the tank's lengths and
    output taps also scale with Size. Left half: modulated allpass, delay, damping, gain, allpass, delay, gain,
    into the right half; the right half is the same and feeds the left."""

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
    EXC = 26.0                                                     # peak excursion at Mod 1, samples (the paper's 16 at 29761 Hz)
    RATE = 1.0                                                     # modulation rate, Hz; the two halves in quadrature
    WET = 0.6                                                      # each tap's gain, as in the paper

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
        self.key = None

    def lengths(self):
        """Tank lengths, taps and gains for the current params, computed when they change."""
        key = (self.p["decay"], self.p["size"])
        if key != self.key:
            sc = self.R * self.p["size"]
            self.L = {k: [int(round(L * sc)) for L in v] for k, v in (("ap1", self.AP1), ("d1", self.D1), ("ap2", self.AP2), ("d2", self.D2))}
            total = sum(sum(v) for v in self.L.values())
            self.gain = 10 ** (-3 * (total / 4) / (self.p["decay"] * SR))   # four gain stages per trip round the tank
            self.g2 = min(0.5, max(0.25, self.gain + 0.15))
            self.taps = [[(h, e, int(round(o * sc)), sg) for h, e, o, sg in t] for t in (self.TAPS_L, self.TAPS_R)]
            self.key = key
        return self.L

    def chunk(self, x):
        n, p = len(x), self.p
        L = self.lengths()
        m = (x[:, 0] + x[:, -1]) * 0.5
        self.pre.write(m)
        v = self.pre.past(int(p["predelay"] / 1000 * SR) + n, n)
        bw = onepole(damp_hz(0.5 * p["damp"]))
        v, self.zb = lfilter([bw], [1, bw - 1], v, zi=self.zb)
        for line, d, g in self.ins:
            v = _allpass(line, v, d, g)
        gain, g1, g2 = self.gain, 0.7, self.g2
        a = onepole(damp_hz(p["damp"]))
        cross = [self.d2[1].past(L["d2"][1], n) * gain, self.d2[0].past(L["d2"][0], n) * gain]
        ph = 2 * math.pi * self.RATE * (self.n + np.arange(n)) / SR
        exc = p["mod"] * self.EXC
        for h in (0, 1):
            u = _flush(v + cross[h])
            sd = self.ap1[h].frac(L["ap1"][h] + exc * (np.sin(ph) if h == 0 else np.cos(ph)))
            s = _flush(u + g1 * sd)                                  # decay diffusion 1, with the paper's sign
            self.ap1[h].write(s)
            u = sd - g1 * s
            y = self.d1[h].past(L["d1"][h], n)
            self.d1[h].write(_flush(u))
            y, self.zd[h] = lfilter([a], [1, a - 1], y, zi=self.zd[h])
            y = y * gain
            sd = self.ap2[h].past(L["ap2"][h], n)
            s = _flush(y - g2 * sd)
            self.ap2[h].write(s)
            self.d2[h].write(g2 * s + sd)
        lines = [(self.d1[0], self.ap2[0], self.d2[0]), (self.d1[1], self.ap2[1], self.d2[1])]
        wet = np.empty((n, 2))
        for c, taps in enumerate(self.taps):
            acc = np.zeros(n)
            for h, e, o, sg in taps:
                acc += sg * lines[h][e].past(o + n, n)
            wet[:, c] = acc
        wet *= self.WET
        if x.shape[1] == 1:
            wet = wet.mean(axis=1, keepdims=True)
        mix = p["mix"]
        return x * (1 - mix) + wet * mix


def _rand(key: np.ndarray) -> np.ndarray:
    """A uniform value in [-1, 1) for each uint64 key (the splitmix64 finaliser), the same however calls are split."""
    z = key * np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * 2.0 ** -52 - 1.0


class Hall(Reverb):
    """The reverb's network with longer lines, each line's delay varied by its own random line segments: a new
    random target every 1 / rate seconds, joined by a raised-cosine curve. Targets are drawn from a hash of the
    line number and the segment number, so the sweep depends only on the absolute sample index."""

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

    def sweep(self, n: int) -> np.ndarray:
        t = self.n + np.arange(n)
        k, f = np.divmod(t[None, :], self.seg)
        key = self.keyline + k.astype(np.uint64)
        r0, r1 = _rand(key), _rand(key + np.uint64(1))
        w = 0.5 - 0.5 * np.cos(np.pi * f / self.seg)
        return (r0 + (r1 - r0) * w) * self.depth

    def chunk(self, x):
        size, mix = self.p["size"], self.p["mix"]
        n = len(x)
        inj = self.front(x).mean(axis=1) * 0.35
        pos = self.w + np.arange(n)[None, :] - (self.base + self.p["mod"] * self.sweep(n))
        i0 = np.floor(pos)
        f = pos - i0
        i0 = i0.astype(np.int64)
        a, b = self.buf[self.rows, i0 & self.mask], self.buf[self.rows, (i0 + 1) & self.mask]
        outs = a + (b - a) * f                                       # (8, n), read by linear interpolation
        mixed = outs - (2.0 / len(self.LENS)) * outs.sum(0, keepdims=True)
        g = 10 ** (-3 * self.base / (SR * size))
        ap = onepole(damp_hz(self.p["damp"]))
        y, self.zs = lfilter([ap], [1, ap - 1], (mixed + inj) * g, axis=1, zi=self.zs)
        y = _flush(y)
        i = self.w & self.mask
        if i + n <= self.buf.shape[1]:
            self.buf[:, i:i + n] = y
        else:
            k = self.buf.shape[1] - i
            self.buf[:, i:] = y[:, :k]
            self.buf[:, :n - k] = y[:, k:]
        self.w = (self.w + n) & self.mask
        wet = np.zeros_like(x)
        sign = np.array([1, -1, 1, -1, 1, -1, 1, -1])[:, None]
        wet[:, 0] = (outs[0::2] * sign[0::2]).sum(0) * 0.5
        wet[:, -1] = (outs[1::2] * sign[1::2]).sum(0) * 0.5
        return x * (1 - mix) + wet * mix * 1.4


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


class Drive(Effect):
    def __init__(self, ch):
        super().__init__(ch)
        self.z = np.zeros((1, ch))

    def chunk(self, x):
        k = self.p["amount"]
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


CLASSES = {"reverb": Reverb, "plate": Plate, "hall": Hall, "delay": Delay, "chorus": Chorus, "drive": Drive, "eq": EQ, "filter": Filter, "crush": Crush}


class Rack:
    """The four slots, with their state; a slot whose type changes starts fresh."""

    def __init__(self, ch: int):
        self.ch = ch
        self.slots = clean(None)
        self.fx: List[Optional[Effect]] = [None] * N_SLOTS

    def update(self, slots: List[Dict[str, Any]]):
        new = clean(slots)
        for i, s in enumerate(new):
            if self.fx[i] is None or s["type"] != self.slots[i]["type"]:
                self.fx[i] = CLASSES[s["type"]](self.ch)
            self.fx[i].set(s["params"])
        self.slots = new

    def active(self) -> bool:
        return any(s["on"] for s in self.slots)

    def process(self, x: np.ndarray) -> np.ndarray:
        for s, f in zip(self.slots, self.fx):
            if f is None:
                continue
            y = f.process(x)                                         # every slot runs, so turning one on starts with a settled tail
            if s["on"]:
                x = y
        return x
