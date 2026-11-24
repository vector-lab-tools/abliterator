# Abliterator. Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
"""Phaser, Flanger and Tremolo: the same output at any block size, Mix or Depth 0 is the input, each does what its
card says, mono works, and each costs under 0.4 ms a 512-sample stereo block.
Run: cd ~/Projects/abliterator && .venv/bin/python -m pytest -q tests/test_fx_mod.py"""
import math
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from abliterator import fx_mod as m  # noqa: E402
from abliterator import postfx as fx  # noqa: E402

SR = fx.SR


def params(t, **kw):
    d = {q[0]: q[4] for q in m.TYPES_NEW[t]["params"]}
    d.update(kw)
    return d


def run(t, x, block=None, **kw):
    e = m.CLASSES_NEW[t](x.shape[1])
    e.set(params(t, **kw))
    block = block or len(x)
    return np.concatenate([e.process(x[i:i + block]) for i in range(0, len(x), block)])


def noise(s=1.0, seed=0, ch=2):
    return np.random.default_rng(seed).standard_normal((int(SR * s), ch)) * 0.1


def st(seconds=1.0, seed=4):
    t = np.arange(int(SR * seconds)) / SR
    r = np.random.default_rng(seed)
    return np.stack([0.3 * np.sin(2 * np.pi * 60 * t) + 0.1 * r.standard_normal(len(t)),
                     0.2 * np.sin(2 * np.pi * 60 * t + 1.0) + 0.1 * r.standard_normal(len(t))], 1)


ODD = {
    "phaser": dict(rate=2.7, depth=0.8, centre=600.0, feedback=0.75, stages=2, spread=130.0, mix=0.8),
    "flanger": dict(rate=1.9, depth=3.0, delay=0.4, feedback=-0.8, spread=70.0, mix=0.9),
    "tremolo": dict(rate=7.3, depth=0.9, shape=2, phase=120.0),
}


@pytest.mark.parametrize("t", list(m.CLASSES_NEW))
def test_blocks_equal_whole(t):
    x = noise(1.0)
    whole = run(t, x, None, **ODD[t])
    for b in (512, 97, 333):
        assert np.max(np.abs(run(t, x, b, **ODD[t]) - whole)) < 1e-9, (t, b)
    assert not np.allclose(whole, x)


@pytest.mark.parametrize("t", list(m.CLASSES_NEW))
def test_blocks_equal_while_knobs_move(t):
    """Knob changes made at the same samples give the same output at any block size."""
    x = noise(0.6)
    def go(block):
        e = m.CLASSES_NEW[t](2)
        out, pos = [], 0
        for cut, kw in ((9600, {}), (19200, ODD[t]), (len(x), {})):
            e.set(params(t, **kw))
            while pos < cut:
                k = min(block, cut - pos)
                out.append(e.process(x[pos:pos + k]))
                pos += k
        return np.concatenate(out)
    ref = go(len(x))
    for b in (512, 97, 333):
        assert np.max(np.abs(go(b) - ref)) < 1e-9, (t, b)


@pytest.mark.parametrize("t,kw", [("phaser", dict(mix=0.0, feedback=0.8)), ("flanger", dict(mix=0.0, feedback=0.9)),
                                  ("tremolo", dict(depth=0.0, rate=12.0))])
def test_zero_is_input(t, kw):
    x = noise(0.5)
    assert np.array_equal(run(t, x, 333, **kw), x)


def test_phaser_matches_per_sample_reference():
    """The matrix solution equals a plain per-sample loop over the same stages, coefficients and feedback."""
    x = noise(0.06, seed=3)
    for st_, N in ((0, 4), (2, 12)):
        e = m.Phaser(2)
        e.set(params("phaser", stages=st_, feedback=0.7, rate=3.0, depth=0.8, mix=1.0))
        e.trace = []
        y = np.concatenate([e.process(x[i:i + 333]) for i in range(0, len(x), 333)])
        tr = {t: (j, fb) for t, j, fb in e.trace}
        z = np.zeros((2, N))
        w = np.zeros_like(x)
        for t in range(len(x)):
            if t % m.Phaser.LS == 0:
                j, fb = tr[t]
            for c in range(2):
                a = e.tb.a[j[c]]
                u0, zz = 0.0, z[c].copy()
                for k in range(N):
                    u0 = a * u0 + zz[k]
                h0 = a ** N
                wv = (h0 * x[t, c] + u0) / (1 - fb * h0)
                u = x[t, c] + fb * wv
                for k in range(N):
                    yy = a * u + z[c, k]
                    z[c, k] = u - a * yy
                    u = yy
                w[t, c] = u
        ref = 0.5 * x + 0.5 * math.sqrt(1 - 0.7 ** 2) * w
        assert np.max(np.abs(y - ref)) < 1e-12, N


def _notch_freqs(y, x, lo=50, hi=10000):
    f = np.fft.rfftfreq(len(y), 1 / SR)
    Y, X = np.abs(np.fft.rfft(y * np.hanning(len(y)))), np.abs(np.fft.rfft(x * np.hanning(len(x))))
    k = np.ones(31) / 31
    r = np.convolve(Y ** 2, k, "same") / np.convolve(X ** 2, k, "same")
    band = (f > lo) & (f < hi)
    return f, r, band


def test_phaser_static_notches():
    """Depth 0, no feedback, 4 stages: notches where each stage turns 45° and 135°, at tan(pi f/SR) =
    tan(pi fb/SR) tan(pi/8) and tan(3 pi/8)."""
    x = noise(2.0, ch=1)
    y = run("phaser", x, 512, depth=0.0, feedback=0.0, centre=1000.0, stages=0, mix=1.0)[SR // 2:, 0]
    f, r, band = _notch_freqs(y, x[SR // 2:, 0])
    fb = m._PhaserTables.get().f[m._PhaserTables.get().index(np.array([1000.0]))[0]]
    for q in (math.pi / 8, 3 * math.pi / 8):
        want = SR / math.pi * math.atan(math.tan(math.pi * fb / SR) * math.tan(q))
        near = (f > want / 1.3) & (f < want * 1.3)
        got = f[near][np.argmin(r[near])]
        assert abs(got / want - 1) < 0.05, (want, got)
        assert r[near].min() < 0.01 * np.median(r[band])


def test_phaser_notches_move():
    x = noise(4.0, ch=1)
    y = run("phaser", x, 512, rate=0.25, depth=1.0, feedback=0.0, centre=800.0, stages=0, mix=1.0)[:, 0]
    seg = lambda a: (y[int(a * SR):int(a * SR) + SR // 4], x[int(a * SR):int(a * SR) + SR // 4, 0])
    lows = []
    for a in (0.9, 2.9):                                           # near the top and the bottom of the sweep
        f, r, band = _notch_freqs(*seg(a), lo=30, hi=3000)
        lows.append(f[band][np.argmin(r[band])])
    assert lows[0] > 2.5 * lows[1], lows


def test_flanger_comb_spacing():
    """Depth 0, no feedback, Mix 1 at 1 ms: notches at 500, 1500, 2500 Hz ..., 1 kHz apart."""
    x = noise(2.0, ch=1)
    y = run("flanger", x, 333, depth=0.0, delay=1.0, feedback=0.0, mix=1.0)[SR // 2:, 0]
    f, r, band = _notch_freqs(y, x[SR // 2:, 0])
    for want in (500.0, 1500.0, 2500.0, 3500.0):
        near = (f > want - 200) & (f < want + 200)
        got = f[near][np.argmin(r[near])]
        assert abs(got - want) < 20, (want, got)
        assert r[near].min() < 0.01


def test_flanger_feedback_peaks():
    """Feedback 0.8 at a fixed 1 ms: peaks at multiples of 1 kHz; at -0.8 they move to 500, 1500 Hz ..."""
    x = noise(2.0, ch=1)
    for fbk, first in ((0.8, 1000.0), (-0.8, 500.0)):
        y = run("flanger", x, 512, depth=0.0, delay=1.0, feedback=fbk, mix=1.0)[SR // 2:, 0]
        f, r, band = _notch_freqs(y, x[SR // 2:, 0])
        near = (f > first - 200) & (f < first + 200)
        assert abs(f[near][np.argmax(r[near])] - first) < 20


def test_tremolo_rate_and_depth():
    x = np.ones((2 * SR, 2))
    for shape in (0, 1, 2):
        y = run("tremolo", x, 333, rate=5.0, depth=0.6, shape=shape, phase=180.0)
        g = y[SR // 2:SR // 2 + SR, 0]                              # 1 s: FFT bins 1 Hz apart
        assert abs(g.max() - 1.0) < 1e-3 and abs(g.min() - 0.4) < 1e-3, shape
        spec = np.abs(np.fft.rfft(g - g.mean()))
        assert abs(np.fft.rfftfreq(len(g), 1 / SR)[np.argmax(spec)] - 5.0) < 0.2
        assert np.allclose(y[SR // 2:, 0] + y[SR // 2:, 1], 1.4, atol=0.05) if shape == 1 else True


def test_tremolo_square_does_not_click():
    x = st()
    for rate in (2.0, 20.0):
        y = run("tremolo", x, 512, rate=rate, depth=1.0, shape=2)
        assert np.max(np.abs(np.diff(y, axis=0))) < 1.5 * np.max(np.abs(np.diff(x, axis=0)))


def test_knob_moves_do_not_click():
    """A large knob move while playing: no sample-to-sample jump larger than the signal's own."""
    x = st()
    for t, a, b in (("phaser", dict(centre=200.0, feedback=0.0), dict(centre=3000.0, feedback=0.85)),
                    ("flanger", dict(delay=0.5, feedback=0.0), dict(delay=8.0, feedback=0.9)),
                    ("tremolo", dict(shape=0, depth=0.2), dict(shape=2, depth=1.0))):
        e = m.CLASSES_NEW[t](2)
        e.set(params(t, **a)); y1 = e.process(x[:SR // 2])
        e.set(params(t, **b)); y2 = e.process(x[SR // 2:])
        y = np.concatenate([y1, y2])
        assert np.max(np.abs(np.diff(y, axis=0))) < 1.5 * np.max(np.abs(np.diff(x, axis=0))), t


@pytest.mark.parametrize("t", list(m.CLASSES_NEW))
def test_mono_is_the_left_side(t):
    x = noise(0.5)
    kw = {**ODD[t]}
    y2 = run(t, x, 512, **kw)
    y1 = run(t, x[:, :1], 512, **kw)
    assert y1.shape == (len(x), 1)
    assert np.max(np.abs(y1[:, 0] - y2[:, 0])) < 1e-12


def test_candidates_and_texts():
    for t, spec in m.TYPES_NEW.items():
        rng = {k: (lo, hi) for k, _, lo, hi, _, _, _ in spec["params"]}
        cs = m.CANDIDATES[t]
        assert len(cs) == 7 and len({c[0] for c in cs}) == 7
        for name, p, tip in cs:
            assert 0 < len(name) <= 16 and "—" not in name and tip and "—" not in tip
            for k, v in p.items():
                assert rng[k][0] <= v <= rng[k][1], (t, name, k)
        assert set(m.HELP[t]["p"]) == set(rng)
        knobs = {k for k, *_r in spec["params"] if not (_r[4].startswith("choice") or _r[4] == "toggle")}
        assert {k for row in m.ROWS[t] for k in row} == knobs
        for s in [m.WHAT[t], m.HELP[t]["how"], *m.HELP[t]["p"].values(), *(c[2] for c in cs)]:
            assert "—" not in s


@pytest.mark.parametrize("t,kw", [("phaser", ODD["phaser"]), ("phaser", dict(stages=0)), ("flanger", ODD["flanger"]),
                                  ("flanger", dict(delay=0.3, depth=0.0, feedback=0.9)), ("tremolo", ODD["tremolo"])])
def test_cost(t, kw):
    x = noise(2.0)
    e = m.CLASSES_NEW[t](2)
    e.set(params(t, **kw))
    e.process(x[:4800])
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        nb = 0
        for i in range(0, len(x) - 512, 512):
            e.process(x[i:i + 512]); nb += 1
        best = min(best, (time.perf_counter() - t0) / nb * 1e3)
    assert best < 0.4, (t, kw, best)
