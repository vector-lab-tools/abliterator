# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Three effects that move sound in pitch or hold it: Pitch, Shimmer and Freeze. numpy and scipy only, in the same
form as postfx's effects (Effect.process splits any block into chunks of up to CHUNK samples).

  pitch    a delay-line pitch shifter: two read heads sweep their delay at the rate that gives the shift and
           crossfade (Hann) as each jumps back to the start of its window; feedback sends the shifted sound back
           into the line through a 30 Hz high-pass and a 10 kHz low-pass
  shimmer  the hall's eight lines (Householder mix, damping) with two of them read through a pitch shifter, so
           each trip round those lines moves part of the tail by the interval
  freeze   a spectral hold: 4,096-sample (85 ms) Hann frames every 1,024 samples; on Freeze, the average power
           of the last Blur frames is kept and resynthesised every 1,024 samples with phases from a generator
           seeded by the frame's number, faded in and out over 100 ms

Block-size independence: every grid (the shifter's head phase, Freeze's frames) counts from sample 0, knob moves
follow a closed-form 10 ms one-pole (_Glide), random numbers come from generators seeded by frame number, and every
loop's delay is longer than CHUNK, so a chunk reads only samples already written.
"""
from __future__ import annotations

import math
import sys
from typing import Dict

import numpy as np
from scipy.signal import lfilter

from . import postfx as fx  # noqa: E402  (imported from the end of postfx.py, once its base classes exist)

SR, CHUNK = fx.SR, fx.CHUNK
_A = fx.SMOOTH
_POW = _A ** np.arange(1, CHUNK + 1)
_K = np.arange(CHUNK)
_K1 = np.arange(1, CHUNK + 1, dtype=float)


class _Glide:
    """A knob's value moving to its target by a 10 ms one-pole, in closed form (target + (v - target) a^k), so any
    split of the samples gives the same values. A scalar once it has arrived, which costs nothing to apply."""

    def __init__(self):
        self.v = None

    def __call__(self, target: float, n: int):
        t = float(target)
        if self.v is None or self.v == t:
            self.v = t
            return t
        y = t + (self.v - t) * _POW[:n]
        v = float(y[-1])
        self.v = t if abs(v - t) <= 1e-15 * max(1.0, abs(t)) else v
        return y


def _col(v):
    """A glide's value as a column to multiply (n, ch) by: a scalar stays a scalar."""
    return v if np.ndim(v) == 0 else v[:, None]


def _phase(ph0: float, inc, n: int) -> np.ndarray:
    """A head phase advanced by inc a sample (scalar or per sample) from ph0, wrapped into [0, 1)."""
    ph = ph0 + (inc * _K1[:n] if np.ndim(inc) == 0 else np.cumsum(inc))
    return ph - np.floor(ph)


def _onepole_hp_lp(hp_hz: float, lp_hz: float):
    """A first-order high-pass and a first-order low-pass in series, as one second-order (b, a)."""
    a1 = math.exp(-2 * math.pi * hp_hz / SR)
    a2 = math.exp(-2 * math.pi * lp_hz / SR)
    bh, ah = np.array([(1 + a1) / 2, -(1 + a1) / 2]), np.array([1.0, -a1])
    bl, al = np.array([1 - a2]), np.array([1.0, -a2])
    return np.convolve(bh, bl), np.convolve(ah, al)


# --------------------------------------------------------------------------------------------------------- Pitch

class PitchShift(fx.Effect):
    """Two heads read a delay line at delays D0 + phase * window, the phases half a cycle apart; each phase moves
    by (1 - ratio) / window a sample, so the read runs at `ratio` times the write. A head's gain is sin²(pi phase):
    zero where it jumps, and the two gains sum to 1. The shortest delay is D0 (just over CHUNK), so the feedback loop
    reads only samples already written; the shifted sound is at least 5 ms late and the dry sound not at all."""

    D0 = CHUNK + 4
    SIZE = 16384                                                   # > D0 + 200 ms + 4
    FB_B, FB_A = _onepole_hp_lp(30.0, 10000.0)

    def __init__(self, ch):
        super().__init__(ch)
        self.buf = np.zeros((self.SIZE, ch))
        self.mask = self.SIZE - 1
        self.w = 0
        self.ph = 0.0
        self.zf = np.zeros((2, ch))
        self.g_st, self.g_win, self.g_fb, self.g_mix = _Glide(), _Glide(), _Glide(), _Glide()

    def chunk(self, x):
        n, p = len(x), self.p
        st = self.g_st(p["shift"] + p["fine"] / 100.0, n)
        W = self.g_win(p["window"] / 1000.0 * SR, n)
        r = 2.0 ** (st / 12.0)
        ph = _phase(self.ph, (1.0 - r) / W, n)
        self.ph = float(ph[-1])
        P = np.stack([ph, ph + 0.5])
        P -= np.floor(P)                                           # (2, n): the two heads
        gain = 0.5 - 0.5 * np.cos(2 * np.pi * P)
        pos = (self.w + _K[:n]) - (self.D0 + P * W)
        i0 = np.floor(pos)
        f = (pos - i0)[..., None]
        i0 = i0.astype(np.int64)
        b, m = self.buf, self.mask
        y = fx._cubic(b[(i0 - 1) & m], b[i0 & m], b[(i0 + 1) & m], b[(i0 + 2) & m], f)   # (2, n, ch)
        wet = (y * gain[..., None]).sum(0)
        loop, self.zf = lfilter(self.FB_B, self.FB_A, wet, axis=0, zi=self.zf)
        line_in = x + fx._snap(loop * _col(self.g_fb(p["feedback"], n)))
        i = self.w
        if i + n <= self.SIZE:
            b[i:i + n] = line_in
        else:
            k = self.SIZE - i
            b[i:] = line_in[:k]
            b[:n - k] = line_in[k:]
        self.w = (i + n) & m
        mix = _col(self.g_mix(p["mix"], n))
        return x * (1 - mix) + wet * mix


# --------------------------------------------------------------------------------------------------------- Shimmer

INTERVALS = [12, 7, 19, 24, -12]                                   # the Interval choice's semitones, in its order


class Shimmer(fx.Effect):
    """The hall's eight line lengths in a feedback delay network (Householder mix, one-pole damping in each line),
    fed through a 10 ms pre-delay and four allpass diffusers (g 0.6). Lines 0 and 1 are also read by a two-head
    pitch shifter (window 100 ms, delays from the line's length up to 100 ms more), and Shimmer crossfades those
    lines' outputs from the plain read to the shifted one. The shifted read passes a 40 Hz high-pass and a 12 kHz
    low-pass. Each line's gain sets its T60 to Decay, counting the shifter's mean extra delay on lines 0 and 1;
    the heads' gains sum to 1 and every filter's gain is at most 1, so every path in the loop has a gain below 1."""

    LENS = fx.Hall.LENS
    DIFF = fx.Reverb.DIFF
    WIN = 4800                                                     # the shifter's window, 100 ms
    PRE = 480                                                      # 10 ms pre-delay
    SIZE = 16384                                                   # > longest line + WIN + 4
    SH_B, SH_A = _onepole_hp_lp(40.0, 12000.0)

    def __init__(self, ch):
        super().__init__(ch)
        L = len(self.LENS)
        self.buf = np.zeros((L, self.SIZE))
        self.mask = self.SIZE - 1
        self.w = 0
        self.base = np.array(self.LENS, dtype=float)[:, None]
        self.basei = np.array(self.LENS, dtype=np.int64)[:, None]
        self.rows = (np.arange(L, dtype=np.int64) * self.SIZE)[:, None]
        self.extra = np.zeros((L, 1))
        self.extra[:2] = self.WIN / 2.0                            # the shifter's mean extra delay, at Shimmer 1
        self.pre = fx.Ring(self.PRE + 2 * CHUNK)
        self.diff = [fx.Ring(1024) for _ in self.DIFF]
        self.zs = np.zeros((L, 1))
        self.zb = np.zeros((2, 2))
        self.ph = 0.0
        self.g_dec, self.g_sh, self.g_st, self.g_mix = _Glide(), _Glide(), _Glide(), _Glide()

    def chunk(self, x):
        n, p = len(x), self.p
        K = _K[:n]
        v = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0].copy()
        self.pre.write(v)
        v = self.pre.past(self.PRE + n, n)
        for ring, d in zip(self.diff, self.DIFF):
            v = fx._allpass(ring, v, d, 0.6, fx._snap)
        inj = v * 0.35
        b, m, w = self.buf.ravel(), self.mask, self.w
        outs = b[self.rows + ((w - self.basei + K) & m)]           # (8, n): the plain reads
        # the shifter on lines 0 and 1
        st = self.g_st(INTERVALS[min(4, max(0, int(round(p["interval"]))))], n)
        r = 2.0 ** (st / 12.0)
        ph = _phase(self.ph, (1.0 - r) / self.WIN, n)
        self.ph = float(ph[-1])
        P = np.stack([ph, ph + 0.5])
        P -= np.floor(P)
        gain = 0.5 - 0.5 * np.cos(2 * np.pi * P)                   # (2 heads, n)
        pos = (w + K) - (self.base[:2, :, None] + P[None] * self.WIN)   # (2 lines, 2 heads, n)
        i0 = np.floor(pos)
        f = pos - i0
        i0 = i0.astype(np.int64)
        ro = self.rows[:2, :, None]
        y = fx._cubic(b[ro + ((i0 - 1) & m)], b[ro + (i0 & m)], b[ro + ((i0 + 1) & m)], b[ro + ((i0 + 2) & m)], f)
        sh = (y * gain[None]).sum(1)
        sh, self.zb = lfilter(self.SH_B, self.SH_A, sh, axis=1, zi=self.zb)
        s = self.g_sh(p["shimmer"], n)
        outs[:2] += s * (sh - outs[:2])
        mixed = outs - (2.0 / len(self.LENS)) * outs.sum(0, keepdims=True)
        T = self.g_dec(p["decay"], n)
        g = 10.0 ** (-3.0 * (self.base + self.extra * s) / (SR * T))
        a = fx.onepole(fx.damp_hz(p["damp"]))
        yl, self.zs = lfilter([a], [1, a - 1], (mixed + inj) * g, axis=1, zi=self.zs)
        yl = fx._snap(yl)
        i = w
        if i + n <= self.SIZE:
            self.buf[:, i:i + n] = yl
        else:
            k = self.SIZE - i
            self.buf[:, i:] = yl[:, :k]
            self.buf[:, :n - k] = yl[:, k:]
        self.w = (i + n) & m
        L = (outs[0] - outs[2] + outs[4] - outs[6]) * 0.7          # 0.5 each, times the reverb's 1.4
        R = (outs[1] - outs[3] + outs[5] - outs[7]) * 0.7
        wet = np.stack([L, R], 1) if x.shape[1] > 1 else ((L + R) * 0.5)[:, None]
        mix = _col(self.g_mix(p["mix"], n))
        return x * (1 - mix) + wet * mix


# --------------------------------------------------------------------------------------------------------- Freeze

class Freeze(fx.Effect):
    """Analysis frames of N samples (Hann) end at every multiple of H from sample 0; the last BMAX frames' power
    spectra are kept. When Freeze goes on at sample t, the RMS of the last Blur frames ending at or before t is
    captured. Synthesis frames start at every multiple of H: each is the captured magnitudes (times Drift's band
    gains) with phases drawn from a generator seeded by the frame's number, inverse transformed and windowed. The
    frames that started up to N samples before t are made at t too (their parts from t on), so the hold is at full
    level from the moment it starts, and a 100 ms equal-power crossfade takes the dry sound out and the hold in.
    Drift is an Ornstein-Uhlenbeck walk (time constant 2 s) on 24 bands, 40 Hz to 16 kHz, applied as gains whose
    total power is 1; it starts from 0 at each capture."""

    N, H, BMAX, FADE, NB = 4096, 1024, 32, 4800, 24
    SEED = 7340033
    RHO = math.exp(-1024 / (2.0 * SR))
    DEPTH = 1.2                                                    # the walk's spread at Drift 1, in nepers (about 10 dB)
    R = 16384
    PHASORS = np.exp(2j * np.pi * np.arange(4096) / 4096)      # the phases are drawn from 4,096 steps round the circle

    def __init__(self, ch):
        super().__init__(ch)
        N = self.N
        self.win = np.hanning(N + 1)[:-1]
        norm = float((self.win ** 2).sum() / self.H)               # Hann² at a quarter overlap sums to 1.5
        self.wsyn = (self.win / math.sqrt(norm * float((self.win ** 2).mean())))[:, None]
        self.wan = self.win[:, None]
        self.mask = self.R - 1
        self.inp = np.zeros((self.R, ch))
        self.acc = np.zeros((self.R, ch))
        self.pw = np.zeros((self.BMAX, N // 2 + 1, ch))
        self.na = 0                                                # the last analysis frame made (frame j ends at j * H)
        self.M = None
        self.P0 = None
        self.syn = False                                           # synthesis frames are being made
        self.k = 0                                                 # the hold's level, 0..FADE
        self.was = False
        self.wk = np.zeros(self.NB)
        fr = np.fft.rfftfreq(N, 1 / SR)
        xb = np.log(np.maximum(fr, 40.0) / 40.0) / math.log(16000.0 / 40.0) * (self.NB - 1)
        xb = np.clip(xb, 0, self.NB - 1)
        self.lo = np.minimum(np.floor(xb).astype(int), self.NB - 2)
        self.fr = xb - self.lo
        self.g_mix = _Glide()

    def _ring_get(self, buf, a: int, n: int) -> np.ndarray:
        i = a & self.mask
        if i + n <= self.R:
            return buf[i:i + n]
        return np.concatenate([buf[i:], buf[:i + n - self.R]])

    def _ring_put(self, buf, a: int, v: np.ndarray, add: bool):
        i, n = a & self.mask, len(v)
        parts = [(slice(i, i + n), slice(0, n))] if i + n <= self.R else [(slice(i, self.R), slice(0, self.R - i)), (slice(0, i + n - self.R), slice(self.R - i, n))]
        for d, s in parts:
            if add:
                buf[d] += v[s]
            else:
                buf[d] = v[s]

    def _analyse(self, j: int):
        X = np.fft.rfft(self._ring_get(self.inp, j * self.H - self.N, self.N) * self.wan, axis=0)
        self.pw[j % self.BMAX] = X.real ** 2 + X.imag ** 2
        self.na = j

    def _capture(self, t: int):
        B = min(self.BMAX, max(1, int(round(self.p["blur"]))))
        js = [j for j in range(self.na - B + 1, self.na + 1) if j >= 1]
        if js:
            self.M = np.sqrt(self.pw[[j % self.BMAX for j in js]].mean(axis=0))
        else:
            self.M = np.zeros_like(self.pw[0])
        self.P0 = (self.M ** 2).sum(axis=0)
        self.wk = np.zeros(self.NB)
        if not self.syn:                                           # the frames already under way at t, from t on
            self.syn = True
            k0 = (t - self.N) // self.H + 1
            for k in range(k0, -(-t // self.H)):
                self._frame(k, t)

    def _frame(self, k: int, t: int):
        """Synthesis frame k (it starts at k * H), added to the output ring from sample t on."""
        rng = np.random.default_rng(self.SEED + k)
        self.wk = self.RHO * self.wk + math.sqrt(1 - self.RHO ** 2) * rng.standard_normal(self.NB)
        M = self.M
        depth = self.DEPTH * float(self.p["drift"])
        if depth > 0:
            gb = np.exp(depth * self.wk)
            gbin = (gb[self.lo] * (1 - self.fr) + gb[self.lo + 1] * self.fr)[:, None]
            M = M * gbin
            M = M * np.sqrt(self.P0 / np.maximum((M ** 2).sum(axis=0), 1e-300))
        y = np.fft.irfft(M * self.PHASORS[rng.integers(0, 4096, M.shape)], n=self.N, axis=0) * self.wsyn
        s = k * self.H
        a = max(s, t)
        if a < s + self.N:
            self._ring_put(self.acc, a, y[a - s:], add=True)

    def chunk(self, x):
        n0, n, p = self.n, len(x), self.p
        self._ring_put(self.inp, n0, x, add=False)
        want = p["freeze"] >= 0.5
        if want and not self.was:
            self._capture(n0)
        self.was = want
        wet = None
        if self.syn:
            k = -(-n0 // self.H)
            while k * self.H < n0 + n:
                self._frame(k, n0)
                k += 1
            wet = self._ring_get(self.acc, n0, n).copy()
            self._ring_put(self.acc, n0, np.zeros((n, self.ch)), add=False)
        j = n0 // self.H + 1
        while j * self.H <= n0 + n:
            self._analyse(j)
            j += 1
        r = np.arange(1, n + 1)
        ks = np.minimum(self.k + r, self.FADE) if want else np.maximum(self.k - r, 0)
        mix = self.g_mix(p["mix"], n)
        if self.k == 0 and ks[-1] == 0:
            out = x
        else:
            g = ks * (0.5 * np.pi / self.FADE)
            out = x * _col(1 - mix * (1 - np.cos(g))) + wet * _col(mix * np.sin(g))
        self.k = int(ks[-1])
        if self.k == 0 and not want and self.syn:                  # faded out: stop making frames
            self.syn = False
            self.acc[:] = 0.0
            self.M = None
        return out


# --------------------------------------------------------------------------------------------------------- tables

TYPES_NEW: Dict[str, Dict] = {
    "pitch": {"label": "Pitch", "params": [("shift", "Shift", -24.0, 24.0, 12.0, "int", "st"),
                                           ("fine", "Fine", -50.0, 50.0, 0.0, "lin", "ct"),
                                           ("window", "Window", 20.0, 200.0, 50.0, "log", "ms"),
                                           ("feedback", "Feedback", 0.0, 0.9, 0.0, "lin", ""),
                                           ("mix", "Mix", 0.0, 1.0, 0.5, "lin", "")]},
    "shimmer": {"label": "Shimmer", "params": [("decay", "Decay", 0.5, 20.0, 6.0, "log", "s"),
                                               ("shimmer", "Shimmer", 0.0, 1.0, 0.5, "lin", ""),
                                               ("interval", "Interval", 0.0, 4.0, 0.0, "choice:+12,+7,+19,+24,-12", ""),
                                               ("damp", "Damp", 0.0, 1.0, 0.4, "lin", ""),
                                               ("mix", "Mix", 0.0, 1.0, 0.35, "lin", "")]},
    "freeze": {"label": "Freeze", "params": [("freeze", "Freeze", 0.0, 1.0, 0.0, "toggle", ""),
                                             ("blur", "Blur", 1.0, 32.0, 8.0, "int", ""),
                                             ("drift", "Drift", 0.0, 1.0, 0.2, "lin", ""),
                                             ("mix", "Mix", 0.0, 1.0, 0.6, "lin", "")]},
}

CANDIDATES = {
    "pitch": [
        ("Octave up", {"shift": 12, "fine": 0.0, "window": 50.0, "feedback": 0.0, "mix": 0.5}, "One octave up, half and half with the dry sound."),
        ("Octave down", {"shift": -12, "fine": 0.0, "window": 90.0, "feedback": 0.0, "mix": 0.5}, "One octave down under the dry sound. The 90 ms window suits low notes."),
        ("Fifth above", {"shift": 7, "fine": 0.0, "window": 50.0, "feedback": 0.0, "mix": 0.4}, "A fifth (7 semitones) above, at 40% mix."),
        ("Doubler", {"shift": 0, "fine": 9.0, "window": 30.0, "feedback": 0.0, "mix": 0.45}, "A second voice 9 cents sharp, read through a 30 ms window."),
        ("Rising spiral", {"shift": 5, "fine": 0.0, "window": 40.0, "feedback": 0.7, "mix": 0.5}, "A fourth up, fed back at 0.7: each repeat climbs another 5 semitones."),
        ("Falling spiral", {"shift": -7, "fine": 0.0, "window": 70.0, "feedback": 0.65, "mix": 0.5}, "A fifth down, fed back at 0.65: each repeat drops another 7 semitones."),
        ("Two down", {"shift": -24, "fine": 0.0, "window": 140.0, "feedback": 0.0, "mix": 0.6}, "Two octaves down at 60% mix, with a 140 ms window."),
    ],
    "shimmer": [
        ("Octave tail", {"decay": 6.0, "shimmer": 0.5, "interval": 0, "damp": 0.4, "mix": 0.35}, "A 6 s tail that climbs in octaves."),
        ("Fifth tail", {"decay": 5.0, "shimmer": 0.45, "interval": 1, "damp": 0.45, "mix": 0.35}, "A 5 s tail that climbs in fifths."),
        ("Long climb", {"decay": 14.0, "shimmer": 0.7, "interval": 0, "damp": 0.3, "mix": 0.4}, "A 14 s tail with most of each pass shifted up an octave."),
        ("Two octaves", {"decay": 8.0, "shimmer": 0.35, "interval": 3, "damp": 0.5, "mix": 0.3}, "An 8 s tail that jumps two octaves at a time."),
        ("Twelfth", {"decay": 7.0, "shimmer": 0.4, "interval": 2, "damp": 0.45, "mix": 0.3}, "A 7 s tail that climbs by an octave and a fifth."),
        ("Sinking", {"decay": 6.0, "shimmer": 0.5, "interval": 4, "damp": 0.2, "mix": 0.35}, "A 6 s tail that falls an octave at a time."),
        ("Short octave", {"decay": 1.8, "shimmer": 0.4, "interval": 0, "damp": 0.5, "mix": 0.3}, "A 1.8 s room with an octave above in its tail."),
    ],
    "freeze": [
        ("Plain hold", {"blur": 1, "drift": 0.0, "mix": 1.0}, "Holds one 85 ms window as it was and replaces the live sound."),
        ("Soft hold", {"blur": 12, "drift": 0.1, "mix": 0.7}, "Holds the average of the last 0.3 s, at 70%."),
        ("Under the live", {"blur": 8, "drift": 0.2, "mix": 0.4}, "Holds the last 0.23 s under the live sound, at 40%."),
        ("Moving hold", {"blur": 8, "drift": 0.6, "mix": 0.6}, "A held spectrum whose bands rise and fall slowly by a few dB."),
        ("Wide average", {"blur": 32, "drift": 0.15, "mix": 0.6}, "Holds the average of the last 0.73 s, at 60%."),
        ("Restless", {"blur": 16, "drift": 1.0, "mix": 0.6}, "Drift at full: the held bands move up and down by about 10 dB."),
        ("Faint bed", {"blur": 24, "drift": 0.3, "mix": 0.25}, "A quiet hold of the last 0.56 s, at 25% under the live sound."),
    ],
}

WHAT = {
    "pitch": "a delay-line pitch shifter with two crossfaded read heads",
    "shimmer": "a reverb whose tail is pitch-shifted on each pass",
    "freeze": "a spectral hold of the last moment of sound",
}

HELP = {
    "pitch": {
        "how": ("Pitch reads the sound back from a delay line through two heads. Each head sweeps its delay at the rate "
                "that gives the shift, then jumps back to the start of the window; the two heads are half a window apart "
                "and crossfade, so a head is silent when it jumps. Feedback sends the shifted sound back into the line, so "
                "each repeat is shifted again. The shifted sound is at least 5 ms late; the dry sound is not delayed."),
        "p": {"shift": "the shift in semitones, from two octaves down to two octaves up",
              "fine": "a further shift of up to 50 cents either way",
              "window": "how far each head sweeps, 20 to 200 ms: long windows suit low notes, short ones fast attacks",
              "feedback": "how much of the shifted sound goes back into the line, up to 0.9",
              "mix": "0 is the dry sound only, 1 the shifted sound only"},
    },
    "shimmer": {
        "how": ("Shimmer is a reverb of eight delay lines, 44 to 100 ms long, mixed into each other on every pass. Two of the "
                "lines are read through a pitch shifter, so on each pass part of the tail moves by the interval and the "
                "tail climbs (or falls) as it decays. Every path round the loop has a gain below 1, so the tail always "
                "dies away."),
        "p": {"decay": "the time the tail takes to fall by 60 dB, 0.5 to 20 s",
              "shimmer": "how much of the two shifted lines is the shifted read: 0 is a plain reverb",
              "interval": "the shift on each pass, in semitones",
              "damp": "the high-frequency loss in each line: 0 keeps the top, 1 rolls off above 1.5 kHz",
              "mix": "0 is the dry sound only, 1 the reverb only"},
    },
    "freeze": {
        "how": ("Freeze measures the sound's spectrum every 21 ms over 85 ms windows. When you switch it on it keeps the "
                "average of the last few windows and plays that spectrum without end, with new random phases every 21 ms. "
                "It fades in and out over 100 ms. When it is off the sound passes unchanged."
                "Its phases are drawn at random, so a held pure tone wanders in level by up to about 4 dB over a fifth of a second; noise-like sound holds within 0.5 dB."),
        "p": {"freeze": "on holds the sound; off lets it go",
              "blur": "how many windows, 21 ms apart, are averaged into the hold: 1 is a single 85 ms window, 32 covers 0.73 s",
              "drift": "slow random rises and falls of 24 frequency bands in the hold, up to about 10 dB, at the same total level",
              "mix": "the hold's share of the output while it is on"},
    },
}

ROWS = {
    "pitch": [["shift", "fine", "window"], ["feedback", "mix"]],
    "shimmer": [["decay", "shimmer"], ["damp", "mix"]],
    "freeze": [["blur", "drift", "mix"]],
}

CLASSES_NEW = {"pitch": PitchShift, "shimmer": Shimmer, "freeze": Freeze}

JUMPS_NEW = {"pitch": {"window"}}                                  # a new window moves both heads' read points

fx._register(sys.modules[__name__])
