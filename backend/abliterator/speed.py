# Abliterator: playback speed of the model's sound (Speed in the Engine panel).
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
The Speed stage sits between the model and the output section and reads the model's sound at a position that
advances by `ratio` model samples per output sample: at ½ the sound lasts twice as long. Only slowing down is
offered (the model writes about 1.6 times faster than real time, so faster playback would run it dry).

  Tape     pitch follows speed, as a slowed tape: a cubic (Catmull-Rom) read between samples.
  Stretch  pitch kept: a phase vocoder (2,048-sample Hann windows, a quarter overlap, identity phase locking
           around spectral peaks, after Laroche and Dolson). Each frame's frequencies come from two analyses a
           fixed hop apart, so a change of speed does not disturb them.
  Smear    pitch kept, for very slow ambient: after Paul Nasca's Paulstretch, 16,384-sample (0.34 s) windows
           whose magnitudes are kept and whose phases are drawn afresh for each frame, from a generator seeded
           by the frame's number.

It works in fixed chunks of CHUNK output samples, whatever block size asks for them; a new speed or mode is taken
at the start of a chunk (a speed glides there over a few chunks). Each change is reported with the output sample it
was taken at, so the same stage run offline over the same model audio, with the same changes at the same samples,
gives the same output to the bit (Speed.render). At speed 1 with nothing gliding or held, the stage passes samples
through unchanged.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import numpy as np

CHUNK = 256                     # output samples a chunk (5.3 ms at 48 kHz)
GLIDE = 0.35                    # of the way to a new speed each chunk: about 60 ms to settle
LO, HI = 0.125, 1.0
MODES = ("tape", "stretch", "smear")
HELD = 20000                    # model samples kept behind the read, for the longest window's first frame
FRAME = {"stretch": 2048, "smear": 16384}


def clamp(r: float) -> float:
    return min(HI, max(LO, float(r)))


class Spectral:
    """One windowed overlap-add process (Stretch or Smear) over the stage's input; hop = a quarter window."""

    def __init__(self, mode: str, ch: int):
        self.mode, self.ch = mode, ch
        self.N = FRAME[mode]
        self.H = self.N // 4
        self.w = np.hanning(self.N + 1)[:-1]
        self.norm = float((self.w ** 2).sum() / self.H)              # Hann² at a quarter overlap sums to 1.5
        self.acc = np.zeros((self.N, ch))
        self.ready = np.zeros((0, ch))
        self.prev = None                                             # the last frame's output phases (Stretch)
        self.k = 0                                                   # frames made, for Smear's seeds
        self.omega = 2 * np.pi * np.arange(self.N // 2 + 1) / self.N

    def frame(self, seg: callable, c: int):
        """One synthesis frame centred on model sample c; seg(a, n) gives n model samples from a (zeros before 0)."""
        N, H, w = self.N, self.H, self.w
        X = np.fft.rfft(seg(c - N // 2, N) * w[:, None], axis=0)
        mag = np.abs(X)
        if self.mode == "stretch":
            P = np.fft.rfft(seg(c - H - N // 2, N) * w[:, None], axis=0)
            ph, php = np.angle(X), np.angle(P)
            dev = ph - php - self.omega[:, None] * H
            dev -= 2 * np.pi * np.round(dev / (2 * np.pi))
            inst = self.omega[:, None] + dev / H                     # each bin's frequency, radians a sample
            out = np.empty_like(ph)
            for j in range(self.ch):
                m = mag[:, j]
                pk = np.flatnonzero((m[1:-1] > m[:-2]) & (m[1:-1] >= m[2:])) + 1
                if len(pk) == 0:
                    pk = np.array([int(np.argmax(m))])
                # each bin belongs to its nearest peak's region (boundaries halfway between peaks)
                edges = np.concatenate([[0], (pk[:-1] + pk[1:]) // 2 + 1, [len(m)]])
                owner = np.repeat(pk, np.diff(edges))
                base = (self.prev[:, j] if self.prev is not None else ph[:, j])
                peak_out = base[pk] + inst[pk, j] * H
                po = np.zeros(len(m))
                po[pk] = peak_out
                out[:, j] = po[owner] + (ph[:, j] - ph[owner, j])     # identity phase locking
            self.prev = out
            Y = mag * np.exp(1j * out)
        else:
            rng = np.random.default_rng(1000003 + self.k)
            Y = mag * np.exp(1j * rng.uniform(0, 2 * np.pi, mag.shape))
        y = np.fft.irfft(Y, n=N, axis=0) * w[:, None] / self.norm
        if self.mode == "smear":                                     # random phases add as power, not amplitude
            y *= math.sqrt(self.norm / float((w ** 2).mean()))
        self.acc += y
        self.ready = np.concatenate([self.ready, self.acc[:H]])
        self.acc = np.concatenate([self.acc[H:], np.zeros((H, self.ch))])
        self.k += 1


class Speed:
    def __init__(self, ch: int):
        self.ch = ch
        self.inp = np.zeros((0, ch))            # model audio from model sample `base` on
        self.base = 0
        self.pos = 0.0                          # the next read's place in the model's sound
        self.ratio = 1.0
        self.target = 1.0
        self.mode = "tape"
        self.out_n = 0                          # output samples made
        self.spare = np.zeros((0, ch))          # made but not yet asked for
        self.spec: Optional[Spectral] = None
        self.changes: List[Tuple[int, float, str]] = []   # (output sample, target, mode) as each was taken

    # -------------------------------------------------------------------- settings
    def set(self, target: float, mode: str = "tape"):
        self._want = (clamp(target), mode if mode in MODES else "tape")

    def _next(self) -> Tuple[float, str]:
        return getattr(self, "_want", (self.target, self.mode))

    def idle(self) -> bool:
        """At speed 1, nothing gliding and nothing held: the samples can pass straight through."""
        want = self._next()
        held = self.spec is not None and (len(self.spec.ready) or np.any(self.spec.acc))
        return (self.ratio == 1.0 and self.target == 1.0 and want[0] == 1.0 and len(self.spare) == 0
                and not held and self.pos == math.floor(self.pos))

    def skip(self, x: np.ndarray):
        """Passed straight through (idle): keep the position in step, and enough of what went by for the first
        read or window when slowing starts."""
        x = x.reshape(len(x), -1)
        n = len(x)
        held = np.concatenate([self.inp, x])[-HELD:] if len(self.inp) else x[-HELD:]
        self.pos += n
        self.out_n += n
        self.inp = held.copy()
        self.base = int(self.pos) - len(held)
        self.spec = None

    def ahead(self, mode: Optional[str] = None) -> int:
        """How far past the read position the stage reads (half a window for the spectral modes)."""
        m = mode or self.mode
        return FRAME[m] // 2 + 2 if m in FRAME else 3

    @property
    def consumed(self) -> int:
        """Model samples the stage has read up to: the engine keeps the model this far ahead and more."""
        return int(math.floor(self.pos)) + max(self.ahead(), self.ahead(self._next()[1]))   # the mode about to be taken too

    # -------------------------------------------------------------------- the stage
    def _seg(self, a: int, n: int) -> np.ndarray:
        """n model samples from model sample a; zeros before the start of the sound."""
        out = np.zeros((n, self.ch))
        lo = max(a, 0)
        if lo < a + n:
            s = lo - self.base
            out[lo - a:] = self.inp[s:s + (a + n - lo)]
        return out

    def _need(self, r: float, mode: str) -> int:
        """How many more model samples the next chunk needs."""
        if mode in FRAME:
            have = len(self.spec.ready) if (self.spec is not None and self.spec.mode == mode) else 0
            H = FRAME[mode] // 4
            k = max(0, math.ceil((CHUNK - have) / H))
            if k == 0:
                return 0                                          # enough made already: no new frame, nothing to read
            last = self.pos + (k - 1) * r * H
            top = int(round(last)) + FRAME[mode] // 2 + 1
        else:
            top = int(math.floor(self.pos + r * (CHUNK - 1))) + 3
        return top - (self.base + len(self.inp))

    def _chunk(self) -> np.ndarray:
        want = getattr(self, "_want", None)
        if want is not None and (want[0] != self.target or want[1] != self.mode):
            self.target, self.mode = want
            self.changes.append((self.out_n, self.target, self.mode))
        r = self.ratio + (self.target - self.ratio) * GLIDE
        if abs(r - self.target) < 1e-4:
            r = self.target
        self.ratio = r
        if self.mode in FRAME:
            if self.spec is None or self.spec.mode != self.mode:
                self.spec = Spectral(self.mode, self.ch)
            sp = self.spec
            while len(sp.ready) < CHUNK:
                sp.frame(self._seg, int(round(self.pos)))
                self.pos += r * sp.H
            out, sp.ready = sp.ready[:CHUNK], sp.ready[CHUNK:]
            back = sp.N // 2 + sp.H + 2
        else:
            self.spec = None
            p = self.pos + r * np.arange(CHUNK)
            i = np.floor(p).astype(np.int64) - self.base
            f = (p - np.floor(p))[:, None]
            n = len(self.inp)
            get = lambda k: self.inp[np.clip(k, 0, n - 1)]            # noqa: E731
            y0, y1, y2, y3 = get(i - 1), get(i), get(i + 1), get(i + 2)
            out = y1 + 0.5 * f * (y2 - y0 + f * (2 * y0 - 5 * y1 + 4 * y2 - y3 + f * (3 * (y1 - y2) + y3 - y0)))
            self.pos += r * CHUNK
            back = 2
        self.out_n += CHUNK
        keep = int(math.floor(self.pos)) - max(back, HELD) - self.base
        if keep > 0:
            self.inp = self.inp[keep:]
            self.base += keep
        return out

    def pull(self, n: int, supply: Callable[[int], Optional[np.ndarray]]) -> Optional[np.ndarray]:
        """n output samples, asking `supply(k)` for k more model samples when it needs them; None (and nothing
        used) if the model has not written enough yet."""
        while len(self.spare) < n:
            t, m = self._next()
            r_next = self.ratio + (t - self.ratio) * GLIDE
            k = self._need(max(r_next, self.ratio), m)
            defer = False
            if k > 0:
                x = supply(k)
                if x is None and m != self.mode:
                    # a mode that reads further ahead (Smear) waits until the model has written that far; meanwhile
                    # the current mode plays on, so the switch makes no gap (it is recorded where it is taken)
                    k = self._need(self.ratio, self.mode)
                    x = supply(k) if k > 0 else np.zeros((0, self.ch))
                    defer = True
                if x is None:
                    return None
                if len(x):
                    self.inp = np.concatenate([self.inp, x.reshape(len(x), -1)]) if len(self.inp) else x.reshape(len(x), -1).copy()
            if defer:
                want = self._want
                del self._want
                self.spare = np.concatenate([self.spare, self._chunk()])
                self._want = want
            else:
                self.spare = np.concatenate([self.spare, self._chunk()])
        y, self.spare = self.spare[:n], self.spare[n:]
        return y

    # -------------------------------------------------------------------- offline
    @staticmethod
    def render(pre: np.ndarray, start: float, mode: str, changes: List[Tuple[int, float, str]],
               marks: Optional[List[int]] = None) -> Tuple[np.ndarray, List[int]]:
        """The stage over recorded model audio, with each change taken at the output sample it was taken at
        live. Returns the output, and for each model sample index in `marks` the output sample at which the
        stage first read past it (to cut a snapshot's frames out of the output)."""
        x = pre.reshape(len(pre), -1)
        sp = Speed(x.shape[1])
        sp.set(start, mode)
        sp.target, sp.mode, sp.ratio = clamp(start), mode, clamp(start)
        sp.inp, sp.base = x, 0
        todo = sorted(changes)
        out, where, marks = [], [], sorted(marks or [])
        mi = 0
        while True:
            while todo and todo[0][0] <= sp.out_n:
                _, t, m = todo.pop(0)
                sp.set(t, m)
            while mi < len(marks) and sp.pos >= marks[mi]:
                where.append(sp.out_n)
                mi += 1
            if sp.idle():                                         # as live: straight through at 1, up to the next change
                upto = todo[0][0] if todo else sp.out_n + (len(x) - int(sp.pos))
                k = min(upto - sp.out_n, len(x) - int(sp.pos))
                if k <= 0:
                    break
                i0 = int(sp.pos)
                while mi < len(marks) and marks[mi] < i0 + k:
                    where.append(sp.out_n + max(0, marks[mi] - i0))
                    mi += 1
                out.append(x[i0:i0 + k])
                sp.pos += k
                sp.out_n += k
                sp.spec = None                                    # as live (skip): the spectral process starts afresh
                continue
            t, m = sp._next()
            if sp._need(max(sp.ratio + (t - sp.ratio) * GLIDE, sp.ratio), m) > 0:
                break                                             # the recording ends: as live would wait for the model
            out.append(sp._chunk())
        while mi < len(marks):
            where.append(sp.out_n)
            mi += 1
        return (np.concatenate(out) if out else x[:0]), where
