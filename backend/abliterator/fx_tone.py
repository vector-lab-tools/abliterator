# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Three chain effects for postfx: a compressor, a tuned resonator and a frequency shifter. numpy and scipy
only. Each gives the same output at any block size: every recursion carries its state across chunks, and
the shifter's oscillators carry their phase.

  compressor  feed-forward, stereo-linked, peak-sensing; a soft-knee gain computer in dB, then a
              decoupled peak detector on the gain reduction (after Giannoulis, Massberg and Reiss,
              JAES 60(6), 2012): an instant rise with an exponential release, then a one-pole attack
  resonator   up to four comb filters tuned to a root and a chord, each with a one-pole low-pass and a
              first-order allpass tuner in its loop
  freqshift   single-sideband shift: a two-path allpass network with a 90 degree phase difference
              (Niemitalo's coefficients) and a complex oscillator, with feedback through 5 ms
"""
from __future__ import annotations

import math
import sys
from typing import Any, Dict

import numpy as np
from scipy.signal import lfilter, sosfilt

from . import postfx as fx  # noqa: E402  (imported from the end of postfx.py, once its base classes exist)

SR, CHUNK = fx.SR, fx.CHUNK
SMOOTH = fx.SMOOTH
POW = SMOOTH ** np.arange(1, CHUNK + 1)
LN10_20 = math.log(10.0) / 20.0


def _col(v):
    return v[:, None] if isinstance(v, np.ndarray) else v


class Smoothed(fx.Effect):
    """An Effect whose knobs move over 10 ms: _sm gives the one-pole's path from its last value to the
    target, in closed form, or the target itself (a float) once it has arrived."""

    def __init__(self, ch):
        super().__init__(ch)
        self.z: Dict[str, float] = {}

    def _sm(self, key: str, target: float, n: int):
        s = self.z.get(key)
        if s is None or abs(s - target) <= 1e-12 * max(1.0, abs(target)):
            self.z[key] = target
            return target
        y = target + (s - target) * POW[:n]
        self.z[key] = float(y[-1])
        return y


# ---------------------------------------------------------------- compressor

class Compressor(Smoothed):
    """Feed-forward and stereo-linked: the louder side's peak sets one gain for both. The gain computer works
    in dB with a quadratic soft knee; its gain reduction goes through a decoupled peak detector, an instant
    rise with an exponential release (a max-times recursion solved for a whole chunk with a running maximum),
    then a one-pole for the attack. Both carry their state, so the result is the same at any block size.
    `gr` holds the largest reduction in dB since it was last read; the reader sets it back to 0."""

    def __init__(self, ch):
        super().__init__(ch)
        self.y1 = 0.0                                   # the release stage's last value, dB of reduction
        self.yl = 0.0                                   # the attack stage's last value
        self.gr = 0.0
        self.rkey = None

    def _release_pows(self, a: float):
        if a != self.rkey:
            k = np.arange(CHUNK)
            self.rp, self.rn, self.rkey = a ** k, a ** (-k.astype(float)), a
        return self.rp, self.rn

    def chunk(self, x):
        p, n = self.p, len(x)
        lv = np.abs(x).max(axis=1) if x.shape[1] > 1 else np.abs(x[:, 0])
        xg = 20.0 * np.log10(np.maximum(lv, 1e-10))
        t = self._sm("t", p["thresh"], n)
        s = self._sm("s", 1.0 - 1.0 / p["ratio"], n)
        w = np.maximum(self._sm("k", p["knee"], n), 1e-6)
        d = xg - t
        q = np.minimum(np.maximum(d + 0.5 * w, 0.0), w)
        gr = s * (q * q / (2.0 * w) + np.maximum(d - 0.5 * w, 0.0))
        ar = math.exp(-1.0 / (p["release"] / 1000.0 * SR))
        rp, rn = self._release_pows(ar)
        y1 = np.maximum(np.maximum.accumulate(gr * rn[:n]), ar * self.y1) * rp[:n]     # y1 = max(gr, ar * y1[-1])
        self.y1 = float(y1[-1])
        aa = math.exp(-1.0 / (p["attack"] / 1000.0 * SR))
        yl, _ = lfilter([1.0 - aa], [1.0, -aa], y1, zi=[aa * self.yl])
        self.yl = float(yl[-1])
        self.gr = max(self.gr, float(yl.max()))
        g = np.exp((self._sm("mk", p["makeup"], n) - yl) * LN10_20)
        m = self._sm("m", p["mix"], n)
        return x * _col(1.0 - m) + x * (g * m)[:, None]


# ---------------------------------------------------------------- resonator

CHORDS = {0: (0,), 1: (0, 7), 2: (0, 12), 3: (0, 4, 7, 12), 4: (0, 3, 7, 12), 5: (0, 5, 7, 12)}
G_MAX = 0.99995


def comb_design(f: float, decay: float, bright: float) -> Dict[str, Any]:
    """One comb at f Hz: a delay of N samples, a one-pole low-pass (DC gain 1) and an allpass tuner in the loop,
    and a loop gain g. The allpass makes up the fraction of a sample that N and the low-pass's phase delay at f
    leave, so the loop is exactly SR / f samples long at f. The low-pass is set so the highest frequencies die
    `r` times as fast as f (r = 100 at Brightness 0, 1 at Brightness 1), held back where it would take so much
    from f that g would reach 1 or the lows below f (which pass at g) would last more than twice the decay; g makes
    the loop lose 60 dB at f over the decay time."""
    P, w = SR / f, 2 * math.pi * f / SR
    cw, sw = math.cos(w), math.sin(w)
    ge = 10 ** (-3.0 * P / (SR * decay))                    # the loop's gain at f
    r = 10 ** (-2.0 + 2.0 * bright)                          # T60 at Nyquist over T60 at f
    rho = 10 ** (-3.0 * P / (SR * decay) * (1.0 / r - 1.0))  # gain at Nyquist over gain at DC
    a = (1.0 - rho) / (1.0 + rho)
    gam = max(ge / G_MAX, math.sqrt(ge))                     # the low-pass may take no more than this at f, so g < 1
                                                             # and the lows below f last no more than twice the decay
    if gam < 1.0 - 1e-12:
        u = 1.0 - gam * gam * cw
        a_max = (u - math.sqrt(max(u * u - (1.0 - gam * gam) ** 2, 0.0))) / (1.0 - gam * gam)
        a = min(a, a_max)
    else:
        a = 0.0
    hlp = (1.0 - a) / math.sqrt(1.0 - 2.0 * a * cw + a * a)
    tlp = math.atan2(a * sw, 1.0 - a * cw) / w
    R = P - tlp
    N = max(3, int(math.floor(R - 0.5)))
    th = w * (1.0 - (R - N)) / 2.0                           # allpass delay R - N, in 0.5 to 1.5
    tt = math.tan(th)
    c = tt / (sw - tt * cw)
    g = min(ge / hlp, G_MAX)
    b0 = 1.0 - a
    # loop filter F = b0 / (1 - a z^-1) * (c + z^-1) / (1 + c z^-1) = (be0 + be1 z^-1) / (1 + al1 z^-1 + al2 z^-2)
    return {"N": N, "g": g, "ge": ge, "be0": b0 * c, "be1": b0, "al1": c - a, "al2": -a * c, "in": math.sqrt(1.0 - ge * ge)}


class Resonator(Smoothed):
    """Comb filters tuned to the root and the chord above it, fed the input's mono sum (after a 20 Hz high-pass).
    Each comb is y = x + g v, where v is y delayed N samples and passed through the loop filter. A comb with a
    long delay runs in pieces of at most N samples against its ring buffer; a short one (N < SHORT) runs as one
    lfilter of order N + 1, its state rebuilt each chunk from the same ring buffer and the loop filter's last two
    outputs, so the two ways give the same samples and a comb can change between them. The combs' input is
    scaled by sqrt(1 - ge^2), so white noise comes out at about its own level, and they are spread across the
    stereo field at equal power."""
    SHORT = 90
    LEVEL = 1.0

    def __init__(self, ch):
        super().__init__(ch)
        self.rings = [fx.Ring(2048) for _ in range(4)]
        self.v = np.zeros((4, 2))                       # each comb's loop filter, last two outputs: v[n-1], v[n-2]
        self.key = None
        self.hp = np.array(fx._sos("hp", 20.0, 1))
        self.zhp = np.zeros((1, 2))

    def _design(self):
        p = self.p
        key = (int(p["root"]), int(p["chord"]), float(p["decay"]), float(p["bright"]))
        if key == self.key:
            return
        root, chord, decay, bright = key
        tones = CHORDS.get(chord, CHORDS[0])
        old = len(self.combs) if self.key is not None else 4
        self.combs = [comb_design(440.0 * 2 ** ((root + t - 69) / 12.0), decay, bright) for t in tones]
        for k in range(len(tones), old):                 # a comb the chord drops starts silent if it comes back
            self.rings[k] = fx.Ring(2048)
            self.v[k] = 0.0
            self.z.pop("in%d" % k, None)
        K = len(tones)
        pos = np.linspace(-0.6, 0.6, K) if K > 1 else np.zeros(1)
        if self.ch > 1:
            G = np.stack([np.cos((pos + 1) * math.pi / 4), np.sin((pos + 1) * math.pi / 4)], 1)
            G = G / np.sqrt((G ** 2).sum(0, keepdims=True))
        else:
            G = np.ones((K, 1)) / math.sqrt(K)
        self.G = G * self.LEVEL
        self.key = key

    def _ring_path(self, k, cb, xin):
        ring, N, g = self.rings[k], cb["N"], cb["g"]
        b, a = [cb["be0"], cb["be1"]], [1.0, cb["al1"], cb["al2"]]
        out = np.empty_like(xin)
        v1, v2 = self.v[k]
        i, n = 0, len(xin)
        while i < n:
            L = min(N, n - i)
            h = ring.past(N + 2, L + 2)                  # y[w-N-2 .. w-N+L-1]
            zi = [cb["be1"] * h[1] - cb["al1"] * v1 - cb["al2"] * v2, -cb["al2"] * v1]
            v, _ = lfilter(b, a, h[2:], zi=zi)
            y = xin[i:i + L] + g * v
            ring.write(y)
            out[i:i + L] = y
            v1, v2 = (v[-1], v[-2]) if L > 1 else (v[-1], v1)
            i += L
        self.v[k] = v1, v2
        return out

    def _filter_path(self, k, cb, xin):
        ring, N, g = self.rings[k], cb["N"], cb["g"]
        al1, al2 = cb["al1"], cb["al2"]
        den = np.zeros(N + 2)
        den[0], den[1], den[2] = 1.0, al1, al2
        den[N] -= g * cb["be0"]
        den[N + 1] -= g * cb["be1"]
        v1, v2 = self.v[k]
        h = ring.past(N + 1, N + 1)                      # y[n-N-1 .. n-1]
        zi = (g * cb["be1"]) * h
        zi[:N] += (g * cb["be0"]) * h[1:]
        zi[0] -= g * (al1 * v1 + al2 * v2)
        zi[1] -= g * al2 * v1
        y, _ = lfilter([1.0, al1, al2], den, xin, zi=zi)
        ring.write(y)
        n = len(y)
        nv1 = (y[-1] - xin[-1]) / g
        nv2 = (y[-2] - xin[-2]) / g if n > 1 else v1
        self.v[k] = nv1, nv2
        return y

    def chunk(self, x):
        p, n = self.p, len(x)
        self._design()
        u = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
        u, self.zhp = sosfilt(self.hp, u, zi=self.zhp)
        Y = np.empty((n, len(self.combs)))
        for k, cb in enumerate(self.combs):
            xin = u * self._sm("in%d" % k, cb["in"], n)
            Y[:, k] = self._filter_path(k, cb, xin) if cb["N"] < self.SHORT else self._ring_path(k, cb, xin)
        wet = Y @ self.G
        m = self._sm("m", p["mix"], n)
        return x * _col(1.0 - m) + wet * _col(m)


# ---------------------------------------------------------------- frequency shifter

HILBERT_A = (0.6923878, 0.9360654322959, 0.9882295226860, 0.9987488452737)
HILBERT_B = (0.4021921162426, 0.8561710882420, 0.9722909545651, 0.9952884791278)


def _hilbert_sos(cs) -> np.ndarray:
    """Four allpass sections in z^-2: y[t] = c^2 (x[t] + y[t-2]) - x[t-2]."""
    return np.array([[c * c, 0.0, -1.0, 1.0, 0.0, -c * c] for c in cs])


class FreqShift(Smoothed):
    """Single-sideband frequency shift. Two chains of four allpasses (Olli Niemitalo's coefficients) give two
    copies of the sound 90 degrees apart, within 0.7 degrees from 30 Hz to 23.5 kHz (measured); the first copy is one
    sample later, as his design has it. Multiplied by a cosine and a sine at the shift frequency and subtracted,
    they give the sound moved up (or down) by that many hertz, with its mirror image cancelled. Spread moves the
    right side's shift towards the opposite sign. Feedback sends the shifted sound back to the input through a
    5 ms delay, so each pass is shifted again. The oscillators' phases are carried from chunk to chunk."""
    FB = CHUNK                                          # 5 ms

    def __init__(self, ch):
        super().__init__(ch)
        self.sa, self.sb = _hilbert_sos(HILBERT_A), _hilbert_sos(HILBERT_B)
        self.za, self.zb = np.zeros((4, 2, ch)), np.zeros((4, 2, ch))
        self.last = np.zeros(ch)                       # the first path's last sample, for its one-sample delay
        self.line = fx.Line(1024, ch)
        self.ph = np.zeros(ch)

    def _phase(self, c: int, w, n: int) -> np.ndarray:
        if isinstance(w, np.ndarray):
            ph = self.ph[c] + np.cumsum(w)
        else:
            ph = self.ph[c] + w * np.arange(1, n + 1)
        self.ph[c] = float(ph[-1] % (2 * math.pi))
        return ph

    def chunk(self, x):
        p, n = self.p, len(x)
        fb = self._sm("fb", p["feedback"], n)
        u = x + _col(fb) * self.line.read(self.FB, n)
        a, self.za = sosfilt(self.sa, u, axis=0, zi=self.za)
        q, self.zb = sosfilt(self.sb, u, axis=0, zi=self.zb)
        i = np.empty_like(a)
        i[0], i[1:] = self.last, a[:-1]
        self.last = a[-1].copy()
        w = self._sm("f", p["shift"], n) * (2 * math.pi / SR)
        out = np.empty_like(x)
        for c in range(x.shape[1]):
            wc = w if c == 0 else w * (1.0 - 2.0 * self._sm("sp", p["spread"], n))
            ph = self._phase(c, wc, n)
            out[:, c] = i[:, c] * np.cos(ph) + q[:, c] * np.sin(ph)
        self.line.write(out)
        m = self._sm("m", p["mix"], n)
        return x * _col(1.0 - m) + out * _col(m)


# ---------------------------------------------------------------- the catalogue's entries

TYPES_NEW: Dict[str, Dict[str, Any]] = {
    "compressor": {"label": "Compressor", "params": [
        ("thresh", "Threshold", -50.0, 0.0, -20.0, "lin", "dB"), ("ratio", "Ratio", 1.0, 20.0, 4.0, "log", ":1"),
        ("attack", "Attack", 0.1, 100.0, 10.0, "log", "ms"), ("release", "Release", 10.0, 2000.0, 150.0, "log", "ms"),
        ("knee", "Knee", 0.0, 12.0, 6.0, "lin", "dB"), ("makeup", "Makeup", 0.0, 24.0, 0.0, "lin", "dB"),
        ("mix", "Mix", 0.0, 1.0, 1.0, "lin", "")]},
    "resonator": {"label": "Resonator", "params": [
        ("root", "Root", 24.0, 84.0, 48.0, "int", ""), ("chord", "Chord", 0.0, 5.0, 1.0, "choice:Uni,5th,Oct,Maj,Min,Sus4", ""),
        ("decay", "Decay", 0.1, 10.0, 2.0, "log", "s"), ("bright", "Brightness", 0.0, 1.0, 0.5, "lin", ""),
        ("mix", "Mix", 0.0, 1.0, 0.4, "lin", "")]},
    "freqshift": {"label": "Freq Shift", "params": [
        ("shift", "Shift", -1000.0, 1000.0, 25.0, "lin", "Hz"), ("spread", "Spread", 0.0, 1.0, 0.0, "lin", ""),
        ("feedback", "Feedback", 0.0, 0.8, 0.0, "lin", ""), ("mix", "Mix", 0.0, 1.0, 0.5, "lin", "")]},
}

CLASSES_NEW = {"compressor": Compressor, "resonator": Resonator, "freqshift": FreqShift}

JUMPS_NEW = {"resonator": {"root", "chord"}}    # a new root or chord changes the combs' delays at once

CANDIDATES = {
    "compressor": [
        ("Gentle Glue", {"thresh": -26.0, "ratio": 2.0, "attack": 30.0, "release": 300.0, "knee": 6.0, "makeup": 3.0, "mix": 1.0},
         "2:1 above -26 dB, slow enough to keep the attacks"),
        ("Vocal Level", {"thresh": -24.0, "ratio": 3.0, "attack": 5.0, "release": 150.0, "knee": 6.0, "makeup": 4.0, "mix": 1.0},
         "3:1 above -24 dB, evens out a line"),
        ("Punch", {"thresh": -20.0, "ratio": 4.0, "attack": 25.0, "release": 120.0, "knee": 3.0, "makeup": 3.0, "mix": 1.0},
         "Lets the first 25 ms of each hit through"),
        ("Brick Wall", {"thresh": -24.0, "ratio": 20.0, "attack": 0.1, "release": 50.0, "knee": 0.0, "makeup": 6.0, "mix": 1.0},
         "20:1 with a 0.1 ms attack above -24 dB, then 6 dB back up"),
        ("Parallel Smash", {"thresh": -40.0, "ratio": 10.0, "attack": 1.0, "release": 100.0, "knee": 0.0, "makeup": 12.0, "mix": 0.5},
         "Crushed hard, then half under the dry sound"),
        ("Slow Swell", {"thresh": -30.0, "ratio": 3.0, "attack": 80.0, "release": 1500.0, "knee": 9.0, "makeup": 3.0, "mix": 1.0},
         "Slow attack and 1.5 s release: the level breathes"),
        ("Pump", {"thresh": -38.0, "ratio": 8.0, "attack": 0.5, "release": 400.0, "knee": 0.0, "makeup": 8.0, "mix": 1.0},
         "Ducks on each loud note and climbs back over 400 ms"),
    ],
    "resonator": [
        ("Low Drone", {"root": 36, "chord": 2, "decay": 6.0, "bright": 0.3, "mix": 0.4}, "C2 and C3, dark, rings for 6 s"),
        ("Glass Fifth", {"root": 72, "chord": 1, "decay": 3.0, "bright": 0.8, "mix": 0.35}, "C5 and G5, bright"),
        ("Major Ring", {"root": 60, "chord": 3, "decay": 2.0, "bright": 0.6, "mix": 0.4}, "A C major chord from middle C"),
        ("Minor Haze", {"root": 57, "chord": 4, "decay": 4.0, "bright": 0.35, "mix": 0.45}, "A minor from A3, dark and long"),
        ("Sus Bell", {"root": 64, "chord": 5, "decay": 5.0, "bright": 0.9, "mix": 0.3}, "E sus4 from E4, bright, 5 s"),
        ("Short Pluck", {"root": 48, "chord": 0, "decay": 0.3, "bright": 0.5, "mix": 0.5}, "One pitch, C3, gone in 0.3 s"),
        ("Open Strings", {"root": 43, "chord": 1, "decay": 8.0, "bright": 1.0, "mix": 0.25}, "G2 and D3, every harmonic rings 8 s"),
    ],
    "freqshift": [
        ("Slow Phase", {"shift": 0.5, "spread": 0.0, "feedback": 0.0, "mix": 0.5}, "Half a hertz against the dry sound: a slow sweep"),
        ("Wide Drift", {"shift": 2.0, "spread": 1.0, "feedback": 0.0, "mix": 0.5}, "Left 2 Hz up, right 2 Hz down"),
        ("Barber Pole", {"shift": 6.0, "spread": 0.0, "feedback": 0.6, "mix": 0.6}, "Each pass round the 5 ms loop 6 Hz higher"),
        ("Metal Voice", {"shift": 80.0, "spread": 0.0, "feedback": 0.0, "mix": 1.0}, "Up 80 Hz: the harmonics no longer line up"),
        ("Sink", {"shift": -150.0, "spread": 0.0, "feedback": 0.0, "mix": 1.0}, "Down 150 Hz, the lows fold under"),
        ("Bell Spiral", {"shift": 150.0, "spread": 0.5, "feedback": 0.7, "mix": 0.5}, "Up 150 Hz and round again, the right side unshifted"),
        ("Ring Bright", {"shift": 600.0, "spread": 1.0, "feedback": 0.0, "mix": 0.3}, "600 Hz up on the left, down on the right, 30%"),
    ],
}

WHAT = {
    "compressor": "turns the loud parts down above a threshold, by a ratio",
    "resonator": "tuned comb filters that ring at a root note and a chord",
    "freqshift": "moves every frequency up or down by the same number of hertz",
}

HELP = {
    "compressor": {"how": "A feed-forward compressor, the same gain on both sides. It reads the louder side's peak in dB. "
                          "Above the Threshold the level rises by 1 dB for every Ratio dB in, and the Knee bends the curve over "
                          "that many dB around the threshold. The gain falls over the Attack time and recovers over the "
                          "Release time. The GR meter above the knobs shows the gain reduction: its cells light at 1, 2, 3, 4, 6, 8, 10, 12, 15 and 20 dB, and the figure is the reduction now.",
                   "p": {"thresh": "the level above which it acts, -50 to 0 dBFS",
                         "ratio": "how many dB in for each dB out above the threshold; 20:1 acts as a limiter",
                         "attack": "how fast the gain falls when the level rises",
                         "release": "how fast the gain comes back when the level falls",
                         "knee": "how many dB the curve takes to bend at the threshold; 0 is a sharp corner",
                         "makeup": "gain after the compressor, up to 24 dB",
                         "mix": "the compressed sound against the dry: below 1 it is parallel compression"}},
    "resonator": {"how": "Up to four comb filters, delay lines fed back into themselves, each one period of a note long, so "
                         "it rings at that note and its harmonics. The root sets the lowest note as a MIDI number (60 is middle C), "
                         "and the Chord adds the notes above it. A low-pass in each loop makes the high harmonics die sooner, and an "
                         "allpass sets the fraction of a sample that keeps each note in tune. The input is summed to mono; the "
                         "notes are spread across the stereo field.",
                  "p": {"root": "the lowest note, as a MIDI number from 24 (C1, 33 Hz) to 84 (C6, 1047 Hz)",
                        "chord": "Uni (the root alone), 5th, Oct (octave), Maj (major), Min (minor) or Sus4, above the root",
                        "decay": "how long each note takes to fall 60 dB, 0.1 to 10 s",
                        "bright": "how long the high harmonics last: at 0 they die 100 times as fast as the note, at 1 as slowly",
                        "mix": "the ringing against the dry sound"}},
    "freqshift": {"how": "Two chains of allpass filters make two copies of the sound 90 degrees apart. Multiplied by a sine and a "
                         "cosine at the Shift frequency and combined, they give the sound moved up or down by that many hertz, "
                         "with the mirror image cancelled. Every partial moves by the same amount, so harmonics stop lining up "
                         "and the sound turns metallic; a shift of a few hertz mixed with the dry sound sweeps like a phaser.",
                  "p": {"shift": "how many hertz every partial moves, -1000 to +1000",
                        "spread": "moves the right side's shift the other way: at 1 the right goes down as far as the left goes up",
                        "feedback": "sends the shifted sound back in through a 5 ms delay, so it is shifted again on each pass",
                        "mix": "the shifted sound against the dry"}},
}

ROWS = {
    "compressor": [["thresh", "ratio", "knee", "makeup"], ["attack", "release", "mix"]],
    "resonator": [["root", "decay"], ["bright", "mix"]],
    "freqshift": [["shift", "spread"], ["feedback", "mix"]],
}

fx._register(sys.modules[__name__])
