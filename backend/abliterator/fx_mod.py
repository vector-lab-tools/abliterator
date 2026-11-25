# Abliterator. Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
"""Phaser, Flanger and Tremolo for the effect chain.

Each gives the same output at any block size: LFOs are phase accumulators carried from sample to sample (the
phaser's at control rate, on sub-blocks tied to the absolute sample count), and every filter and delay line carries
its state across chunks. Knob moves are smoothed over about 10 ms."""
import math
import sys

import numpy as np
from scipy.signal import lfilter

from . import postfx as fx  # noqa: E402  (imported from the end of postfx.py, once its base classes exist)

SR = fx.SR
SMOOTH = fx.SMOOTH


class _Mod(fx.Effect):
    """An effect with Trim's 10 ms knob smoothing. A smoothed value within 1e-12 of its target is set to it, so a
    knob at 0 gives exactly 0 once it has settled."""

    def __init__(self, ch):
        super().__init__(ch)
        self.z = {}

    def _sm(self, key, target, n):
        """The knob's value at each sample of the chunk, or the target itself (a float) once it has settled."""
        a, target = SMOOTH, float(target)
        z = self.z.get(key)
        if z is None or abs(z[0] - target * a) < 1e-13:
            self.z[key] = np.array([target * a])
            return target
        y, self.z[key] = lfilter([1 - a], [1, -a], np.full(n, target), zi=z)
        return np.where(np.abs(y - target) < 1e-12, target, y)

    def _phase(self, rate, n):
        """The LFO's phase in cycles at each sample of the chunk, carried across chunks."""
        if np.ndim(rate) == 0:
            inc = rate / SR
            ph = self.ph + inc * np.arange(n)
            self.ph = float((self.ph + inc * n) % 1.0)
            return ph
        inc = rate / SR
        cs = np.cumsum(inc)
        ph = self.ph + cs - inc
        self.ph = float((self.ph + cs[-1]) % 1.0)
        return ph


def _col(v):
    """A per-sample array as a column, so it scales each channel; a float as it is."""
    return v[:, None] if np.ndim(v) else v


# ---------------------------------------------------------------------------------------------------------------
# Phaser

class _PhaserTables:
    """For a cascade of identical first-order allpasses y = a u + z, z' = u - a y (scipy's DF2T form, one state per
    stage), on a grid of J break frequencies, the responses over a sub-block of L samples that the phaser needs:
      H[j, t, m]   the output of stage m at time t for an impulse into stage 0 (zero state);
      Y0[j, t, m]  the output of stage m at time t for a unit state on stage 0 (zero input);
      S[j, t, m]   the state of stage m after sample t, for an impulse into stage 0 at time 0;
      P[j, k, m]   the state of stage m after k samples, for a unit state on stage 0 (k = 0..L).
    Every stage is the same, so a unit state on stage l acts on stage m as a unit state on stage 0 acts on stage
    m - l: the cascade of N stages reads these as Toeplitz matrices, and one set (12 stages) serves 4, 8 and 12."""
    J, L, NMAX = 1024, 48, 12
    FLO, FHI = 20.0, 20000.0
    _built = None

    @classmethod
    def get(cls):
        if cls._built is None:
            cls._built = cls()
        return cls._built

    def __init__(self):
        J, L, M = self.J, self.L, self.NMAX
        self.f = self.FLO * (self.FHI / self.FLO) ** (np.arange(J) / (J - 1))
        tn = np.tan(np.pi * np.minimum(self.f, 0.49 * SR) / SR)
        a = (tn - 1) / (tn + 1)
        self.a = a
        H = np.zeros((J, L, M)); Y0 = np.zeros((J, L, M)); S = np.zeros((J, L, M)); P = np.zeros((J, L + 1, M))
        zi = np.zeros((J, M)); z0 = np.zeros((J, M)); z0[:, 0] = 1.0
        P[:, 0] = z0
        for t in range(L):
            ui = np.full(J, 1.0 if t == 0 else 0.0)
            u0 = np.zeros(J)
            for m in range(M):
                yi = a * ui + zi[:, m]; zi[:, m] = ui - a * yi; ui = yi; H[:, t, m] = yi
                y0 = a * u0 + z0[:, m]; z0[:, m] = u0 - a * y0; u0 = y0; Y0[:, t, m] = y0
            S[:, t] = zi
            P[:, t + 1] = z0
        self.S, self.P = S, P
        self.Hn = {}                                           # per N: the cascade's impulse response
        self.Ob = {}                                           # per N: Ob[j, t, l] the output for a unit state on stage l
        for N in (4, 8, 12):
            self.Hn[N] = np.ascontiguousarray(H[:, :, N - 1])
            self.Ob[N] = np.ascontiguousarray(Y0[:, :, N - 1::-1])
        self.lo = np.tril(np.ones((M, M)))                     # Toeplitz index for P: m - l on and below the diagonal
        r = np.arange(M)
        self.pix = np.clip(r[:, None] - r[None, :], 0, None)

    def index(self, f):
        q = np.log(np.clip(f, self.FLO, self.FHI) / self.FLO) / math.log(self.FHI / self.FLO) * (self.J - 1)
        return np.rint(q).astype(np.int64)


class Phaser(_Mod):
    """A chain of 4, 8 or 12 first-order allpasses, all at one break frequency, which an LFO sweeps on a log scale up
    to two octaves either side of Centre. Feedback returns the chain's output to its input with no delay (as in an
    analogue phaser), so the loop is H = A^N / (1 - fb A^N); the wet signal is scaled by sqrt(1 - fb^2), which
    keeps its level on noise as without feedback. Mix 1 sums dry and wet at half each: notches where the chain's
    phase is an odd multiple of 180°, N/2 of them.

    The break frequency is held for 48 samples (1 ms) at a time, on sub-blocks counted from sample 0, and taken from
    a grid of 1024 steps from 20 Hz to 20 kHz (about 0.12 semitone apart). Within a sub-block the system is fixed,
    so it is solved as matrices: precomputed responses (_PhaserTables) give the open-loop output and the state, the
    delay-free loop is a short triangular solve (lfilter with the impulse response as its denominator). The state
    carries across chunks, so the output is the same at any block size."""
    STAGES = (4, 8, 12)
    SWEEP_OCT = 2.0
    LS = _PhaserTables.L
    CSM = math.exp(-_PhaserTables.L / (0.010 * SR))          # the control-rate smoothing, 10 ms

    def __init__(self, ch):
        super().__init__(ch)
        self.tb = _PhaserTables.get()
        self.N = None
        self.s = None
        self.ph = 0.0
        self.c = None                                           # control-rate smoothed knobs
        self.j = np.zeros(ch, dtype=np.int64)
        self.fb = 0.0
        self.trace = None                                       # tests: a list to record (t, j, fb) at each sub-block

    def _control(self, t):
        p, c = self.p, self.c
        tgt = (p["rate"], p["depth"], math.log2(p["centre"]), p["feedback"], math.radians(p["spread"]))
        if c is None:
            c = list(tgt)
        else:
            q = self.CSM
            c = [v + (o - v) * q for o, v in zip(c, tgt)]
        self.c = c
        rate, depth, lc, fb, spread = c
        tb, ph = self.tb, 2 * math.pi * self.ph
        k = (tb.J - 1) / math.log2(tb.FHI / tb.FLO)
        lo, hi = math.log2(tb.FLO), math.log2(tb.FHI)
        js = []
        for ch in range(self.ch):
            lf = lc + self.SWEEP_OCT * depth * math.sin(ph + ch * spread)
            js.append(round((min(hi, max(lo, lf)) - lo) * k))
        self.j = np.array(js, dtype=np.int64)
        self.fb = fb
        self.ph = (self.ph + rate * self.LS / SR) % 1.0
        if self.trace is not None:
            self.trace.append((t, self.j.copy(), self.fb))

    def chunk(self, x):
        n, C = x.shape
        tb, Ls = self.tb, self.LS
        N = self.STAGES[int(round(self.p["stages"]))]
        if N != self.N:
            s = np.zeros((C, N))
            if self.s is not None:
                k = min(N, self.N)
                s[:, :k] = self.s[:, :k]                         # the stages both chains share keep their state
            self.s, self.N = s, N
        Hn, Ob = tb.Hn[N], tb.Ob[N]
        S, P = tb.S[:, :, :N], tb.P[:, :, :N]
        pix, lo = tb.pix[:N, :N], tb.lo[:N, :N]
        xt = x.T
        w = np.empty((C, n))
        s = self.s
        i = 0
        while i < n:
            t = self.n + i
            r = t % Ls
            if r == 0:
                self._control(t)
            k = min(Ls - r, n - i)
            j, fb = self.j, self.fb
            xs = xt[:, i:i + k]
            hj = Hn[j, :k]
            zir = (Ob[j, :k] @ s[:, :, None])[:, :, 0]              # the stored state's response, open loop
            g = np.zeros((C, k + 1))
            g[:, :k] = -fb * hj
            g[:, 0] += 1.0
            zir /= g[:, :1]
            v = np.empty((C, k))
            for c in range(C):                                      # (I - fb T) w = T x + zir, T the chain's Toeplitz
                v[c] = lfilter(hj[c], g[c], xs[c], zi=zir[c])[0]     # matrix: the ZIR enters as lfilter's zi
            u = xs + fb * v
            w[:, i:i + k] = v
            Pk = P[j, k][:, pix] * lo                               # (C, N, N): the state carried k samples
            s = (Pk @ s[:, :, None])[:, :, 0] + (u[:, None, ::-1] @ S[j, :k])[:, 0, :]
            i += k
        self.s = s
        m = self._sm("mix", self.p["mix"], n)
        fbs = self._sm("fb", self.p["feedback"], n)
        return x * _col(1.0 - 0.5 * m) + _col(0.5 * m * np.sqrt(1.0 - fbs * fbs)) * w.T


# ---------------------------------------------------------------------------------------------------------------
# Flanger

class Flanger(_Mod):
    """A delay line read at Delay plus a swept part of up to Depth, the sweep a raised cosine from an LFO (the right
    side Spread degrees behind the left), read by linear interpolation. Feedback writes the read back into the line
    with the input; negative feedback moves the peaks to where the notches were. With feedback the line is written
    in pieces no longer than the shortest delay in the chunk, so each read finds only samples already written. The
    wet signal is scaled by sqrt(1 - fb^2), which keeps its level on noise as without feedback. Mix 1 sums dry and
    wet at half each: notches at odd multiples of 1 / (2 x delay)."""
    SIZE = 2048                                                # > 20 ms (960 samples) + a chunk

    def __init__(self, ch):
        super().__init__(ch)
        self.buf = np.zeros((self.SIZE, ch))
        self.ph = 0.0

    def chunk(self, x):
        n, C = x.shape
        p = self.p
        rate = self._sm("rate", p["rate"], n)
        base = self._sm("delay", p["delay"], n)
        depth = self._sm("depth", p["depth"], n)
        spread = self._sm("spread", p["spread"], n)
        fbv = self._sm("fb", p["feedback"], n)
        m = self._sm("mix", p["mix"], n)
        ph = self._phase(rate, n)
        ang = 2 * math.pi * (ph[:, None] + np.arange(C)[None, :] * (_col(spread) / 360.0))
        d = (_col(base) + _col(depth) * 0.5 * (1.0 - np.cos(ang))) * (SR / 1000.0)        # (n, C) samples
        t = self.n + np.arange(n)
        pos = t[:, None] - d
        i0 = np.floor(pos)
        f = pos - i0
        i0 = i0.astype(np.int64)
        M = self.SIZE - 1
        i0m, i1m, wm = i0 & M, (i0 + 1) & M, t & M
        cidx = np.arange(C)
        buf = self.buf
        if not np.any(fbv):
            buf[wm] = x
            a, b = buf[i0m, cidx], buf[i1m, cidx]
            r = a + (b - a) * f
        else:
            r = np.empty((n, C))
            k = max(1, int(math.ceil(float(d.min()))) - 1)        # reads stay behind the samples written
            fb2 = _col(fbv) if np.ndim(fbv) else np.full((n, 1), fbv)
            for s in range(0, n, k):
                e = min(n, s + k)
                a, b = buf[i0m[s:e], cidx], buf[i1m[s:e], cidx]
                rs = a + (b - a) * f[s:e]
                buf[wm[s:e]] = x[s:e] + fb2[s:e] * rs
                r[s:e] = rs
        return x * _col(1.0 - 0.5 * m) + _col(0.5 * m * np.sqrt(1.0 - fbv * fbv)) * r


# ---------------------------------------------------------------------------------------------------------------
# Tremolo

class Tremolo(_Mod):
    """The level swung down from unity by an LFO: gain = 1 - depth (1 - s) / 2, s from 1 to -1, so at Depth 1 the
    level falls to silence once a cycle. Shapes: Sine; Triangle; Square, a triangle clipped and rounded by a
    quarter sine so each edge takes 5 ms (less at rates above 10 Hz, where an edge takes a fifth of a cycle). The
    right side runs Phase degrees behind the left; at 180 the sides alternate. A change of shape crossfades over
    10 ms."""
    EDGE = 0.005

    def __init__(self, ch):
        super().__init__(ch)
        self.ph = 0.0

    def chunk(self, x):
        n, C = x.shape
        p = self.p
        rate = self._sm("rate", p["rate"], n)
        depth = self._sm("depth", p["depth"], n)
        off = self._sm("phase", p["phase"], n) / 360.0
        shape = int(round(p["shape"]))
        ks = [self._sm(f"k{i}", 1.0 if shape == i else 0.0, n) for i in range(3)]
        ph = self._phase(rate, n)
        if np.ndim(depth) == 0 and depth == 0.0:
            return x
        q = ph[:, None] + np.arange(C)[None, :] * _col(off)
        q -= np.floor(q)
        tri = None
        s = 0.0
        for i, k in enumerate(ks):
            if not np.any(k):
                continue
            if i == 0:
                v = np.cos(2 * math.pi * q)
            else:
                if tri is None:
                    tri = 4.0 * np.abs(q - 0.5) - 1.0
                v = tri if i == 1 else np.sin(0.5 * math.pi * np.clip(tri / _col(np.minimum(1.0, 2.0 * rate * self.EDGE)), -1.0, 1.0))
            s = s + (v if np.ndim(k) == 0 and k == 1.0 else _col(k) * v)
        return x * (1.0 - _col(depth) * 0.5 * (1.0 - s))


# ---------------------------------------------------------------------------------------------------------------
# What the page needs

TYPES_NEW = {
    "phaser": {"label": "Phaser", "params": [
        ("rate", "Rate", 0.02, 5.0, 0.3, "log", "Hz"), ("depth", "Depth", 0.0, 1.0, 0.6, "lin", ""),
        ("centre", "Centre", 100.0, 4000.0, 700.0, "log", "Hz"), ("feedback", "Feedback", 0.0, 0.9, 0.5, "lin", ""),
        ("stages", "Stages", 0.0, 2.0, 0.0, "choice:4,8,12", ""), ("spread", "Spread", 0.0, 180.0, 90.0, "int", "°"),
        ("mix", "Mix", 0.0, 1.0, 1.0, "lin", "")]},
    "flanger": {"label": "Flanger", "params": [
        ("rate", "Rate", 0.02, 5.0, 0.2, "log", "Hz"), ("depth", "Depth", 0.0, 10.0, 2.0, "lin", "ms"),
        ("delay", "Delay", 0.3, 10.0, 1.0, "log", "ms"), ("feedback", "Feedback", -0.95, 0.95, 0.5, "lin", ""),
        ("spread", "Spread", 0.0, 180.0, 90.0, "int", "°"), ("mix", "Mix", 0.0, 1.0, 1.0, "lin", "")]},
    "tremolo": {"label": "Tremolo", "params": [
        ("rate", "Rate", 0.1, 20.0, 4.0, "log", "Hz"), ("depth", "Depth", 0.0, 1.0, 0.5, "lin", ""),
        ("shape", "Shape", 0.0, 2.0, 0.0, "choice:Sine,Triangle,Square", ""), ("phase", "Phase", 0.0, 180.0, 0.0, "int", "°")]},
}

CLASSES_NEW = {"phaser": Phaser, "flanger": Flanger, "tremolo": Tremolo}

JUMPS_NEW = {"phaser": {"stages"}}                             # a new stage count changes the chain's response at once

ROWS = {
    "phaser": [["rate", "depth", "centre"], ["feedback", "spread", "mix"]],
    "flanger": [["rate", "depth", "delay"], ["feedback", "spread", "mix"]],
    "tremolo": [["rate", "depth", "phase"]],
}

WHAT = {
    "phaser": "a chain of allpass filters swept by an LFO, with feedback",
    "flanger": "a short delay swept by an LFO, with feedback",
    "tremolo": "the level swung by an LFO",
}

HELP = {
    "phaser": {"how": "A chain of 4, 8 or 12 allpass filters, all at one frequency, which an LFO sweeps up to two "
                      "octaves either side of Centre. Each filter shifts the phase and leaves the level alone. Added to "
                      "the dry sound, the shifted copy cancels it where the chain's phase is an odd multiple of 180°: "
                      "2, 4 or 6 notches, which move with the sweep. Feedback sends the chain's output back to its "
                      "input and raises peaks between the notches. The right side's LFO runs Spread degrees behind "
                      "the left.",
               "p": {"rate": "how fast the sweep goes, 0.02 to 5 Hz",
                     "depth": "how far the sweep goes: at 1, two octaves either side of Centre",
                     "centre": "the middle of the sweep, 100 to 4000 Hz",
                     "feedback": "how much of the chain's output goes back to its input, 0 to 0.9: higher gives sharper peaks",
                     "stages": "4, 8 or 12 filters: 2, 4 or 6 notches",
                     "spread": "how far the right side's sweep runs behind the left, 0 to 180°",
                     "mix": "the swept copy against the dry sound; at 1 they are equal, for the deepest notches"}},
    "flanger": {"how": "The sound delayed by 0.3 to 20 ms, the delay swept by an LFO, and added to the dry sound. "
                       "The copy cancels the dry sound at every frequency where it arrives half a cycle late, so "
                       "the notches sit at odd multiples of 1 / (2 × delay), evenly spaced, and move as the delay "
                       "changes. Feedback sends the delayed sound back into the line and raises peaks between the "
                       "notches; negative feedback puts the peaks where the notches were. The right side's LFO runs "
                       "Spread degrees behind the left.",
                "p": {"rate": "how fast the sweep goes, 0.02 to 5 Hz",
                      "depth": "how far above Delay the sweep goes, 0 to 10 ms",
                      "delay": "the shortest delay, where the sweep starts, 0.3 to 10 ms",
                      "feedback": "how much of the delayed sound goes back into the line, -0.95 to 0.95",
                      "spread": "how far the right side's sweep runs behind the left, 0 to 180°",
                      "mix": "the delayed copy against the dry sound; at 1 they are equal, for the deepest notches"}},
    "tremolo": {"how": "The level swung by an LFO. At its top the sound passes at unity; at its bottom it is "
                       "turned down by Depth, to silence at 1. Square switches between the two with 5 ms edges, so "
                       "it does not click. The right side runs Phase degrees behind the left; at 180° the sides "
                       "take turns.",
                "p": {"rate": "how many times a second the level swings, 0.1 to 20 Hz",
                      "depth": "how far the level falls: 0 none, 0.5 to half, 1 to silence",
                      "shape": "Sine, Triangle or Square",
                      "phase": "how far the right side runs behind the left, 0 to 180°"}},
}

CANDIDATES = {
    "phaser": [
        ("Slow Sweep", {"rate": 0.1, "depth": 0.7, "centre": 800.0, "feedback": 0.3, "stages": 0, "spread": 90.0, "mix": 1.0},
         "Four stages, one sweep every 10 s"),
        ("Classic Four", {"rate": 0.5, "depth": 0.6, "centre": 700.0, "feedback": 0.5, "stages": 0, "spread": 0.0, "mix": 1.0},
         "Four stages at 0.5 Hz, the same on both sides"),
        ("Deep Eight", {"rate": 0.25, "depth": 0.8, "centre": 1000.0, "feedback": 0.6, "stages": 1, "spread": 90.0, "mix": 1.0},
         "Eight stages, four notches, a slow wide sweep"),
        ("Twelve Stage", {"rate": 0.15, "depth": 0.7, "centre": 1200.0, "feedback": 0.4, "stages": 2, "spread": 120.0, "mix": 1.0},
         "Six notches, the sides 120° apart"),
        ("Resonant", {"rate": 0.3, "depth": 0.6, "centre": 900.0, "feedback": 0.85, "stages": 1, "spread": 90.0, "mix": 1.0},
         "High feedback for sharp peaks"),
        ("Fast Warble", {"rate": 3.0, "depth": 0.3, "centre": 1500.0, "feedback": 0.4, "stages": 0, "spread": 180.0, "mix": 0.8},
         "A quick shallow sweep, the sides opposite"),
        ("Low Phase", {"rate": 0.2, "depth": 0.5, "centre": 250.0, "feedback": 0.5, "stages": 0, "spread": 90.0, "mix": 1.0},
         "The sweep held below 1 kHz"),
    ],
    "flanger": [
        ("Jet", {"rate": 0.1, "depth": 4.0, "delay": 0.5, "feedback": 0.7, "spread": 0.0, "mix": 1.0},
         "A slow sweep from 0.5 to 4.5 ms, strong feedback"),
        ("Gentle Flange", {"rate": 0.25, "depth": 2.0, "delay": 1.0, "feedback": 0.3, "spread": 90.0, "mix": 0.7},
         "A light sweep from 1 to 3 ms"),
        ("Negative", {"rate": 0.2, "depth": 3.0, "delay": 0.8, "feedback": -0.7, "spread": 90.0, "mix": 1.0},
         "Negative feedback: peaks where the notches were"),
        ("Wide Stereo", {"rate": 0.15, "depth": 3.0, "delay": 1.0, "feedback": 0.5, "spread": 180.0, "mix": 1.0},
         "The two sides sweep in opposite directions"),
        ("Metallic", {"rate": 0.05, "depth": 0.5, "delay": 2.0, "feedback": 0.9, "spread": 30.0, "mix": 1.0},
         "A near-fixed 2 ms comb with high feedback"),
        ("Fast Flutter", {"rate": 4.0, "depth": 1.0, "delay": 0.6, "feedback": 0.4, "spread": 90.0, "mix": 0.8},
         "A quick, shallow sweep"),
        ("Slow Chorus", {"rate": 0.3, "depth": 5.0, "delay": 6.0, "feedback": 0.1, "spread": 90.0, "mix": 0.6},
         "Longer delays, from 6 to 11 ms, little feedback"),
    ],
    "tremolo": [
        ("Gentle Sine", {"rate": 3.0, "depth": 0.3, "shape": 0, "phase": 0.0}, "A slow sine, down to 70%"),
        ("Classic Amp", {"rate": 5.5, "depth": 0.6, "shape": 0, "phase": 0.0}, "A sine at 5.5 Hz, down to 40%"),
        ("Triangle", {"rate": 4.0, "depth": 0.5, "shape": 1, "phase": 0.0}, "A triangle at 4 Hz, down to half"),
        ("Hard Chop", {"rate": 8.0, "depth": 1.0, "shape": 2, "phase": 0.0}, "Square at 8 Hz, on and off"),
        ("Stereo Swap", {"rate": 2.0, "depth": 0.8, "shape": 0, "phase": 180.0}, "The sides take turns, twice a second"),
        ("Slow Pulse", {"rate": 0.5, "depth": 0.7, "shape": 2, "phase": 0.0}, "Square, one pulse every 2 s"),
        ("Fast Shimmer", {"rate": 14.0, "depth": 0.4, "shape": 1, "phase": 90.0}, "A fast triangle, the sides 90° apart"),
    ],
}

fx._register(sys.modules[__name__])
