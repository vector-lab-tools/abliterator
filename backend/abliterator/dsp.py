# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""Conventional signal processing applied after the model (not strips). Imports only numpy and scipy,
so that both engines' environments can use it."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from scipy.signal import lfilter

from . import postfx


LEVEL_DB = -20.0          # auto level's target (RMS, dBFS)
LEVEL_TAU_S = 1.5         # how slowly it follows a fall
LEVEL_ATTACK_S = 0.3      # and a rise
LEVEL_RANGE_DB = (-12.0, 18.0)
DYN_STEP = 48             # dynamics move in 1 ms steps at 48 kHz, counted from the start
GATE_RANGE_DB = 40.0      # how far the gate closes
DYN_RANGES = {"dyn_ratio": (1.0, 20.0), "dyn_thresh": (-40.0, 0.0), "dyn_release": (0.05, 4.0), "gate_thresh": (-80.0, -20.0)}
WIDEN_MS = 12.0          # the stereo spread's delay: inside the Haas window, so it widens without an echo
LEVEL_GATE_DB = -60.0     # below this it holds its gain, so silence is not raised
STEP = 480                # auto level moves in 10 ms steps at 48 kHz


class Master:
    """The master chain: auto level, the effect chain (postfx.py, up to eight), one-pole low-pass, tanh
    soft limiter, volume. Zero latency
    (IIR, sample by sample). Takes (n,) or (n, channels) blocks.

    Auto level keeps the loudness steady while the controls move, so that a move is heard as a
    change of sound more than a change of level: it follows the block's power with a 1.5 s time
    constant (0.3 s to a rise) and sets a gain toward −20 dBFS RMS (−12 to +18 dB), ramped across the block, held
    while the sound is near silence. It changes the music's own slow swells as well."""

    def __init__(self, sr: int, channels: int = 1):
        self.sr, self.channels = sr, channels
        self.on = True
        self.level = True
        self.lowpass_hz = 16000.0
        self.limit = True
        self.volume = 0.5
        self.y = np.zeros(channels)
        self.p: float = -1.0                 # the followed power (−1: not yet)
        self.g0 = self.g1 = 1.0              # the gain ramp across the current step
        self.acc, self.pos = 0.0, 0
        self.rack = postfx.Rack(channels)
        self.width = 1.0                     # the side's gain: 0 mono, 1 as the model wrote it, 2 twice as wide
        self.spread = 0.0                    # side made from the mid, delayed WIDEN_MS: width from a mono sound
        self.whist = np.zeros(int(sr * WIDEN_MS / 1000))
        # dynamics, as on a console's channel strip: a compressor and a gate, off by default
        self.dyn_on, self.dyn_fast, self.dyn_peak = False, False, False
        self.dyn_ratio, self.dyn_thresh, self.dyn_release, self.gate_thresh = 4.0, -18.0, 0.3, -80.0
        self.dacc, self.dpos, self.dg0, self.dg1, self.cgr, self.ggr = 0.0, 0, 1.0, 1.0, 0.0, 0.0
        # De-click (Declick): off in a new Master, so a sound saved before it replays as it was; the
        # engine turns it on for live play. It always runs, so that turning it on has a past to start from.
        self.declick = False
        self.dc = Declick(channels)
        # Fades in the effect chain (postfx.Rack): off in a new Master for the same reason, on for live play.
        self.fx_fade = False
        self.fx_on = True                    # the Effects panel's switch; a sound saved before it has none, and played its effects
        self.speed = 1.0                     # playback speed of the model's sound (speed.py; the engine runs it, not this class)
        self.speed_mode = "tape"

    def settings(self) -> Dict[str, Any]:
        return {"on": self.on, "level": self.level, "lowpass_hz": self.lowpass_hz, "limit": self.limit, "volume": self.volume,
                "width": self.width, "spread": self.spread, "dyn_on": self.dyn_on, "dyn_fast": self.dyn_fast, "dyn_peak": self.dyn_peak, "declick": self.declick, "fx_fade": self.fx_fade, "fx_on": self.fx_on,
                "speed": self.speed, "speed_mode": self.speed_mode,
                "dyn_ratio": self.dyn_ratio, "dyn_thresh": self.dyn_thresh, "dyn_release": self.dyn_release, "gate_thresh": self.gate_thresh,
                "fx": self.rack.wanted()}

    def _dynamics(self, x: np.ndarray) -> np.ndarray:
        """A compressor and a gate in fixed steps of DYN_STEP samples counted from the start, so any block
        size gives the same result: each step's level (its peak, or its RMS) sets a target reduction,
        the compressor's above the threshold by the ratio, the gate's GATE_RANGE_DB below its threshold;
        each follows its target with its attack (3 ms fast, 30 ms otherwise) or the release, and the
        gain ramps across the next step."""
        out = np.empty_like(x)
        i = 0
        att = 0.003 if self.dyn_fast else 0.03
        while i < len(x):
            k = min(len(x) - i, DYN_STEP - self.dpos)
            seg = x[i:i + k]
            r = (self.dpos + np.arange(k)) / DYN_STEP
            out[i:i + k] = seg * (self.dg0 + (self.dg1 - self.dg0) * r)[:, None]
            if self.dyn_peak:
                self.dacc = max(self.dacc, float(np.abs(seg).max()))
            else:
                self.dacc += float(np.sum(seg ** 2)) / seg.shape[1]
            self.dpos += k
            i += k
            if self.dpos == DYN_STEP:
                lvl = self.dacc if self.dyn_peak else np.sqrt(self.dacc / DYN_STEP)
                db = 20 * np.log10(lvl + 1e-12)
                over = db - self.dyn_thresh
                ct = -over * (1 - 1 / self.dyn_ratio) if over > 0 else 0.0
                gt = -GATE_RANGE_DB if (self.gate_thresh > -79.5 and db < self.gate_thresh) else 0.0
                for name, target, a in (("cgr", ct, att), ("ggr", gt, 0.001)):
                    cur = getattr(self, name)
                    tau = a if target < cur else self.dyn_release
                    setattr(self, name, cur + (1 - np.exp(-DYN_STEP / (tau * self.sr))) * (target - cur))
                self.dg0, self.dg1 = self.dg1, 10 ** ((self.cgr + self.ggr) / 20)
                self.dacc, self.dpos = 0.0, 0
        return out

    def _widen(self, x: np.ndarray) -> np.ndarray:
        """Mid and side: the side scaled by width, plus spread times the mid delayed WIDEN_MS, added to
        the left and taken from the right, so the mono sum stays the mid. The delay line is carried
        from block to block, so any block size gives the same result."""
        m = (x[:, 0] + x[:, 1]) / 2
        buf = np.concatenate([self.whist, m])
        d, self.whist = buf[:len(m)], buf[len(m):]
        if self.width == 1.0 and self.spread == 0.0:
            return x
        s = (x[:, 0] - x[:, 1]) / 2 * self.width + d * self.spread
        return np.stack([m + s, m - s], axis=1)

    def _level(self, x: np.ndarray) -> np.ndarray:
        """Auto level in fixed steps of STEP samples counted from the start, so any block size gives
        the same result: the gain across a step ramps from the last step's to this step's, which
        depends only on the steps before it."""
        out = np.empty_like(x)
        i = 0
        while i < len(x):
            k = min(len(x) - i, STEP - self.pos)
            seg = x[i:i + k]
            r = (self.pos + np.arange(k)) / STEP
            out[i:i + k] = seg * (self.g0 + (self.g1 - self.g0) * r)[:, None]
            self.acc += float(np.sum(seg ** 2)) / seg.shape[1]
            self.pos += k
            i += k
            if self.pos == STEP:
                ms = self.acc / STEP
                if self.p < 0:
                    self.p = ms
                else:
                    tau = LEVEL_ATTACK_S if ms > self.p else LEVEL_TAU_S     # quicker to a rise than to a fall
                    self.p += (1 - np.exp(-STEP / (tau * self.sr))) * (ms - self.p)
                g = self.g1
                if 10 * np.log10(self.p + 1e-20) > LEVEL_GATE_DB:
                    db = LEVEL_DB - 10 * np.log10(self.p + 1e-20)
                    g = 10 ** (min(LEVEL_RANGE_DB[1], max(LEVEL_RANGE_DB[0], db)) / 20)
                self.g0, self.g1, self.acc, self.pos = self.g1, g, 0.0, 0
        return out

    def update(self, m: Dict[str, Any]):
        if "fx_fade" in m:                                       # before "fx", so the change it comes with is faded
            self.fx_fade = self.rack.smooth = bool(m["fx_fade"])
        if "fx_on" in m:
            self.fx_on = self.rack.on = bool(m["fx_on"])
        for k, v in m.items():
            if k == "fx":
                self.rack.update(v)
            elif k in ("on", "limit", "level"):
                setattr(self, k, bool(v))
            elif k in ("lowpass_hz", "volume"):
                setattr(self, k, float(v))
            elif k == "width":
                self.width = float(min(2.0, max(0.0, float(v))))
            elif k == "spread":
                self.spread = float(min(1.0, max(0.0, float(v))))
            elif k == "speed":
                self.speed = float(min(1.0, max(0.125, float(v))))
            elif k == "speed_mode":
                self.speed_mode = str(v) if str(v) in ("tape", "stretch", "smear") else "tape"
            elif k in ("dyn_on", "dyn_fast", "dyn_peak", "declick"):
                setattr(self, k, bool(v))
            elif k in DYN_RANGES:
                lo, hi = DYN_RANGES[k]
                setattr(self, k, float(min(hi, max(lo, float(v)))))

    def process(self, x: np.ndarray) -> np.ndarray:
        fixed = self.dc.process(x)                 # AHEAD samples late, bursts redrawn
        if self.declick:
            x = fixed.reshape(x.shape)
        if not self.on:
            return (x * self.volume).astype(np.float32)
        a = 1.0 - np.exp(-2 * np.pi * min(self.lowpass_hz, self.sr / 2 - 1) / self.sr)
        x2 = x.reshape(len(x), -1).astype(np.float64)
        if self.level:
            x2 = self._level(x2)
        x2 = self.rack.process(x2)
        if self.dyn_on:
            x2 = self._dynamics(x2)
        else:
            self.cgr = self.ggr = 0.0
            self.dg0 = self.dg1 = 1.0
        if x2.shape[1] >= 2:
            x2 = self._widen(x2)
        y, _ = lfilter([a], [1, a - 1], x2, axis=0, zi=(self.y * (1 - a))[None, :])
        self.y = y[-1].copy()
        if self.limit:
            pre = float(np.abs(y).max()) if len(y) else 0.0
            y = np.tanh(1.5 * y) / np.tanh(1.5)
            post = float(np.abs(y).max()) if len(y) else 0.0
            k = 1.5 / np.tanh(1.5)                                          # the curve's gain on quiet sound
            self.limit_db = min(0.0, 20 * np.log10(post / (pre * k))) if pre > 1e-6 and post > 0 else 0.0   # its pull on the peak, against that
        else:
            self.limit_db = 0.0
        return (y * self.volume).astype(np.float32).reshape(x.shape)

    def readings(self) -> Dict[str, float]:
        """What the automatic stages are doing: auto level's gain and the limiter's pull on the peak, in dB."""
        return {"auto_db": round(20 * float(np.log10(max(self.g1, 1e-6))), 1) if self.level else 0.0,
                "limit_db": round(float(getattr(self, "limit_db", 0.0)), 1),
                "comp_db": round(float(self.cgr), 1) if self.dyn_on else 0.0, "gate_db": round(float(self.ggr), 1) if self.dyn_on else 0.0,
                "clicks": self.dc.repaired, "fx": self.rack.readings()}


class Conceal:
    """Covers a dropout at the sound card without a click. When no audio is ready, the last block
    played goes on, mirrored so that it starts where the sound stopped (a block played backwards,
    then forwards, and so on), fading to silence over FADE_S; when audio comes back, it crossfades
    in over XFADE_S from whatever the cover is playing. Only what is heard is touched: recordings
    keep the model's audio as made."""

    FADE_S = 0.03
    XFADE_S = 0.005

    def __init__(self, sr: int):
        self.sr = sr
        self.src: Optional[np.ndarray] = None     # the block last played, unscaled: the cover plays it reversed next
        self.gain = 0.0                            # the cover's level at the end of the last block
        self.gaps = 0                              # dropouts in a row

    def good(self, y: np.ndarray) -> np.ndarray:
        """A block of real audio: crossfaded in from the cover after a dropout."""
        if self.gaps and len(y):
            n = min(len(y), max(1, int(self.XFADE_S * self.sr)))
            r = (np.arange(1, n + 1) / n)[:, None].astype(y.dtype)
            y = y.copy()
            if self.gain > 0 and self.src is not None:
                c = self._cover(n, y.shape)
                y[:n] = c * (self.gain * (1 - r)) + y[:n] * r
            else:
                y[:n] = y[:n] * r
        self.src, self.gain, self.gaps = y, 1.0, 0
        return y

    def gap(self, frames: int, shape) -> np.ndarray:
        """A block with no audio ready: the cover, fading."""
        self.gaps += 1
        if self.src is None or self.gain <= 0:
            self.gain = 0.0
            return np.zeros(shape, np.float32)
        c = self._cover(frames, shape)
        end = max(0.0, self.gain - frames / (self.FADE_S * self.sr))
        g = np.linspace(self.gain, end, frames, endpoint=False, dtype=np.float32)[:, None]
        self.gain = end
        return (c * g).astype(np.float32)

    def _cover(self, frames: int, shape) -> np.ndarray:
        """The next `frames` of the mirrored continuation; src becomes what was played."""
        out, src = [], self.src
        while sum(len(o) for o in out) < frames:
            src = src[::-1]
            out.append(src)
        c = np.concatenate(out)[:frames]
        self.src = c                               # its last sample is where the next block, reversed, starts
        return c


class Declick:
    """Repairs the few-sample bursts a model can write when a unit pushes it hard (Quantise at low bits
    puts spikes of up to full scale into the codec's output). A sample is a burst where its distance from
    the line through its neighbours is more than THRESH times the sound's mean step over the LOOK samples
    before it, and above FLOOR; the burst, widened by PAD samples each side, is redrawn as a straight
    line between the good samples around it. Bursts longer than MAX are left as written. Every decision
    reads the original samples only, from LOOK before to AHEAD after, and the output is AHEAD samples
    late, so any block size gives the same sound."""

    THRESH, FLOOR, PAD, MAX, LOOK, AHEAD = 6.0, 0.04, 3, 48, 256, 64

    def __init__(self, channels: int):
        self.hist = np.zeros((self.LOOK + 2 * self.AHEAD, channels), np.float64)
        self.repaired = 0                          # bursts redrawn, for the readout

    def process(self, x: np.ndarray) -> np.ndarray:
        n, H, A = len(x), len(self.hist), self.AHEAD
        z = np.concatenate([self.hist, x.reshape(n, -1).astype(np.float64)])
        self.hist = z[-H:]
        out = z[H - A: H - A + n].copy()           # the samples leaving now: AHEAD late
        e = np.zeros_like(z)
        e[1:-1] = np.abs(z[1:-1] - 0.5 * (z[:-2] + z[2:]))
        c = np.cumsum(e, axis=0)
        lvl = np.zeros_like(z)
        L = self.LOOK
        lvl[L + 1:] = (c[L:-1] - c[:-L - 1]) / L      # mean curvature over the LOOK samples before each one
        sg = np.zeros_like(z)
        sg[1:-1] = z[1:-1] - 0.5 * (z[:-2] + z[2:])
        zig = np.zeros(z.shape, bool)              # curvature flipping sign from one sample to the next
        zig[1:-1] = ((sg[1:-1] * sg[:-2] < 0) & (np.abs(sg[:-2]) > 0.5 * e[1:-1])) | ((sg[1:-1] * sg[2:] < 0) & (np.abs(sg[2:]) > 0.5 * e[1:-1]))
        hit = (e > self.FLOOR) & (e > self.THRESH * lvl) & zig
        hit[:L + 1] = False
        hit[-2:] = False                            # needs its right neighbour to judge
        if not hit[H - A - self.PAD - self.MAX: H - A + n + self.PAD].any():
            return out.astype(np.float32)
        for ch in range(z.shape[1]):
            m = hit[:, ch].copy()
            if not m.any():
                continue
            for k in range(1, self.PAD + 1):           # widen each burst
                m[k:] |= hit[:-k, ch]
                m[:-k] |= hit[k:, ch]
            idx = np.flatnonzero(m)
            runs = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            for r in runs:
                a, b = r[0] - 1, r[-1] + 1           # the good samples either side
                if len(r) > self.MAX or a < 0 or b >= len(z):
                    continue
                lo, hi = max(a + 1, H - A), min(b, H - A + n)
                if lo >= hi:
                    continue
                if r[0] >= H - A and r[0] < H - A + n:
                    self.repaired += 1
                t = (np.arange(lo, hi) - a) / (b - a)
                out[lo - (H - A): hi - (H - A), ch] = z[a, ch] + t * (z[b, ch] - z[a, ch])
        return out.astype(np.float32)
