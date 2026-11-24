# Abliterator. Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
"""The compressor, resonator and frequency shifter: same output at any block size, mix 0 passes the input,
each does what its card says, mono works, and each fits the 0.4 ms budget for a 512-sample stereo block."""
import math
import os
import sys
import time

import numpy as np
import pytest

from abliterator import postfx as fx            # noqa: E402
from abliterator import fx_tone as T  # noqa: E402

SR = fx.SR


def params(kind, **kw):
    p = {k: d for k, _, _, _, d, _, _ in T.TYPES_NEW[kind]["params"]}
    p.update(kw)
    return p


def run(kind, x, block=512, **kw):
    e = T.CLASSES_NEW[kind](x.shape[1])
    e.set(params(kind, **kw))
    return np.concatenate([e.process(x[i:i + block]) for i in range(0, len(x), block)]), e


def noise(s=1.5, ch=2, seed=0):
    r = np.random.default_rng(seed)
    x = r.standard_normal((int(SR * s), ch)) * 0.1
    x[int(SR * 0.5):int(SR * 0.8)] *= 4.0                       # a loud stretch, for the compressor
    x[int(SR * 1.0):] *= 0.1
    return x


def sine(hz, s=1.0, amp=1.0, ch=2):
    t = np.arange(int(SR * s)) / SR
    return np.repeat((amp * np.sin(2 * np.pi * hz * t))[:, None], ch, axis=1)


ODD = {
    "compressor": [dict(thresh=-30.0, ratio=6.0, attack=2.0, release=80.0, knee=4.0, makeup=6.0, mix=0.7)],
    "resonator": [dict(root=30, chord=3, decay=3.0, bright=0.4, mix=0.7),          # long combs: the ring path
                  dict(root=69, chord=4, decay=1.5, bright=0.7, mix=0.8),          # both paths
                  dict(root=84, chord=5, decay=6.0, bright=0.1, mix=1.0)],         # short combs: the lfilter path
    "freqshift": [dict(shift=137.0, spread=0.6, feedback=0.5, mix=0.8), dict(shift=-3.5, spread=1.0, feedback=0.8, mix=0.5)],
}
CASES = [(k, p) for k, ps in ODD.items() for p in ps]


@pytest.mark.parametrize("kind,p", CASES)
def test_blocks_equal_whole(kind, p):
    x = noise()
    outs = [run(kind, x, b, **p)[0] for b in (len(x), 512, 97, 333)]
    for o in outs[1:]:
        assert np.max(np.abs(o - outs[0])) < 1e-9
    assert not np.allclose(outs[0], x)
    assert np.all(np.isfinite(outs[0]))


@pytest.mark.parametrize("kind,p", CASES)
def test_mix_zero_is_the_input(kind, p):
    x = noise()
    y, _ = run(kind, x, 333, **{**p, "mix": 0.0})
    assert np.array_equal(y, x)


@pytest.mark.parametrize("kind,p", CASES)
def test_mono_input(kind, p):
    x = noise(ch=1)
    y, _ = run(kind, x, 512, **p)
    assert y.shape == x.shape and np.all(np.isfinite(y))
    z, _ = run(kind, x, 97, **p)
    assert np.max(np.abs(y - z)) < 1e-9


# ---------------------------------------------------------------- compressor

@pytest.mark.parametrize("knee", [0.0, 6.0, 12.0])
def test_compressor_static_curve(knee):
    """A steady sine 20 dB over the threshold at 4:1 comes out 15 dB lower; one 20 dB under it is untouched."""
    for over, want in ((20.0, -15.0), (-20.0, 0.0)):
        amp = 10 ** ((-30.0 + over) / 20)
        x = sine(1000, 1.5, amp)
        y, e = run("compressor", x, 512, thresh=-30.0, ratio=4.0, attack=1.0, release=500.0, knee=knee)
        got = 20 * np.log10(np.abs(y[SR // 2:]).max() / amp)
        assert abs(got - want) < 0.3, (over, got)
    y, e = run("compressor", sine(1000, 1.5, 10 ** (-10 / 20)), 512, thresh=-30.0, ratio=4.0, attack=1.0, release=500.0)
    assert 14.7 < e.gr < 15.2


def test_compressor_makeup_attack_release_and_link():
    amp = 10 ** (-10 / 20)
    y, _ = run("compressor", sine(1000, 1.5, amp), 512, thresh=-30.0, ratio=4.0, attack=1.0, release=500.0, makeup=9.0)
    assert abs(20 * np.log10(np.abs(y[SR // 2:]).max() / amp) - (-6.0)) < 0.3
    # stereo link: a loud left and a quiet right are turned down by the same gain
    x = sine(1000, 1.0, amp); x[:, 1] *= 0.05
    y, _ = run("compressor", x, 512, thresh=-30.0, ratio=4.0, attack=1.0, release=500.0)
    g = y[SR // 2:] / np.where(np.abs(x[SR // 2:]) > 1e-6, x[SR // 2:], np.nan)
    assert np.nanmax(np.abs(g[:, 0] - g[:, 1])) < 1e-9
    # attack and release: a step up is pulled down within a few attack times; a step down recovers over the release
    x = np.concatenate([sine(1000, 0.5, 0.01), sine(1000, 0.5, 0.5), sine(1000, 1.0, 0.01)])
    y, _ = run("compressor", x, 512, thresh=-30.0, ratio=10.0, attack=5.0, release=200.0, knee=0.0)
    gain = lambda a, b: 20 * np.log10(np.abs(y[int(a * SR):int(b * SR)]).max() / np.abs(x[int(a * SR):int(b * SR)]).max())
    assert gain(0.500, 0.502) > -6.0                              # the first 2 ms of the step get through
    assert gain(0.6, 0.99) < -20.0                                # then it is held down
    assert gain(1.01, 1.03) < -10.0                               # just after the drop it is still down
    assert gain(1.9, 2.0) > -0.5                                  # 900 ms later it is back


# ---------------------------------------------------------------- resonator

def peak_near(y, hz, pad=16):
    Y = np.abs(np.fft.rfft(y, pad * len(y)))
    f = np.fft.rfftfreq(pad * len(y), 1 / SR)
    w = (f > 0.9 * hz) & (f < 1.1 * hz)
    return f[w][np.argmax(Y[w])]


@pytest.mark.parametrize("root", [24, 36, 48, 57, 60, 69, 72, 84])
@pytest.mark.parametrize("bright", [0.0, 0.5, 1.0])
def test_resonator_rings_in_tune(root, bright):
    """A click rings at the root's frequency within 1% (in fact within 0.1%), and with Octave at twice it."""
    imp = np.zeros((SR * 2, 2)); imp[100] = 1.0
    y, _ = run("resonator", imp, 512, root=root, chord=2, decay=3.0, bright=bright, mix=1.0)
    f0 = 440 * 2 ** ((root - 69) / 12)
    for h in (1, 2):
        got = peak_near(y.sum(1), h * f0)
        assert abs(got / (h * f0) - 1) < 0.01, (h, got)
        assert abs(got / (h * f0) - 1) < 0.001, (h, got)


def test_resonator_chord_notes():
    imp = np.zeros((SR * 2, 2)); imp[100] = 1.0
    for chord, steps in ((3, (0, 4, 7, 12)), (4, (0, 3, 7, 12)), (5, (0, 5, 7, 12)), (1, (0, 7))):
        y, _ = run("resonator", imp, 512, root=60, chord=chord, decay=3.0, bright=0.5, mix=1.0)
        for s in steps:
            f = 440 * 2 ** ((60 + s - 69) / 12)
            assert abs(peak_near(y.sum(1), f) / f - 1) < 0.003, (chord, s)


@pytest.mark.parametrize("root", [36, 60, 84])
def test_resonator_decay_time(root):
    """With Brightness 1 every harmonic falls 60 dB over the Decay time."""
    imp = np.zeros((int(SR * 2.5), 1)); imp[100] = 1.0
    y, _ = run("resonator", imp, 512, root=root, chord=0, decay=1.0, bright=1.0, mix=1.0)
    rms = lambda a, b: math.sqrt(float(np.mean(y[int(a * SR):int(b * SR)] ** 2)))
    drop = 20 * math.log10(rms(0.2, 0.4) / rms(1.2, 1.4))
    assert abs(drop - 60.0) < 3.0, drop


def test_resonator_paths_agree():
    """The ring path and the lfilter path are the same filter: forced one way or the other they give the same samples,
    and a comb can move from one to the other (a Root change) without a jump."""
    x = noise(1.0)
    p = dict(root=48, chord=3, decay=3.0, bright=0.4, mix=1.0)
    old = T.Resonator.SHORT
    try:
        T.Resonator.SHORT = 0
        a, _ = run("resonator", x, 512, **p)
        T.Resonator.SHORT = 10 ** 6
        b, _ = run("resonator", x, 333, **p)
    finally:
        T.Resonator.SHORT = old
    assert np.max(np.abs(a - b)) < 1e-9


def test_resonator_level():
    """White noise comes out of the wet side at about its own level."""
    x = np.random.default_rng(3).standard_normal((SR * 2, 2)) * 0.1
    for root, chord in ((36, 0), (60, 3), (84, 2)):
        y, _ = run("resonator", x, 512, root=root, chord=chord, decay=2.0, bright=0.5, mix=1.0)
        db = 20 * math.log10(np.std(y[SR:]) / np.std(x[SR:]))
        assert -9.0 < db < 3.0, (root, chord, db)


# ---------------------------------------------------------------- frequency shifter

def spectrum_db(y):
    Y = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return lambda hz: 20 * np.log10(Y[np.argmin(np.abs(f - hz))] + 1e-30)


def test_freqshift_moves_a_sine():
    """1 kHz shifted by +100 Hz comes out at 1100 Hz, with the image at 900 Hz more than 30 dB down; with Spread 1
    the right side goes the other way; a negative shift goes down."""
    x = sine(1000, 1.0)
    y, _ = run("freqshift", x, 512, shift=100.0, spread=1.0, feedback=0.0, mix=1.0)
    L, R = spectrum_db(y[SR // 4:, 0]), spectrum_db(y[SR // 4:, 1])
    assert L(1100) - L(900) > 30 and L(1100) - L(1000) > 30
    assert R(900) - R(1100) > 30
    y, _ = run("freqshift", x, 512, shift=-250.0, spread=0.0, feedback=0.0, mix=1.0)
    L = spectrum_db(y[SR // 4:, 0])
    assert L(750) - L(1250) > 30
    for hz in (100, 300, 3000, 10000):                              # across the band
        y, _ = run("freqshift", sine(hz, 1.0), 512, shift=37.0, mix=1.0)
        L = spectrum_db(y[SR // 4:, 0])
        assert L(hz + 37) - L(hz - 37) > 30, hz


def test_freqshift_keeps_level_and_feedback_is_stable():
    x = noise(2.0)
    y, _ = run("freqshift", x, 512, shift=200.0, mix=1.0)
    assert abs(20 * np.log10(np.std(y[SR // 4:SR // 2]) / np.std(x[SR // 4:SR // 2]))) < 0.5
    y, _ = run("freqshift", x, 512, shift=5.0, feedback=0.8, mix=1.0)
    assert np.all(np.isfinite(y)) and np.abs(y[-SR // 4:]).max() < 0.1


# ---------------------------------------------------------------- knob moves

@pytest.mark.parametrize("kind,a,b", [
    ("compressor", dict(makeup=0.0, mix=0.3), dict(makeup=18.0, mix=1.0)),
    ("compressor", dict(thresh=-10.0, ratio=2.0, knee=0.0), dict(thresh=-40.0, ratio=10.0, knee=12.0)),
    ("resonator", dict(mix=0.0), dict(mix=1.0)),
    ("freqshift", dict(mix=0.0, feedback=0.0, shift=20.0), dict(mix=1.0, feedback=0.6, shift=300.0, spread=1.0)),
])
def test_knob_moves_do_not_click(kind, a, b):
    """A knob turned while it plays moves over about 10 ms: no sample-to-sample step larger than the steady sound's."""
    x = sine(220, 1.0, 0.3)
    e = T.CLASSES_NEW[kind](2)
    e.set(params(kind, **a)); y1 = e.process(x[:SR // 2])
    e.set(params(kind, **{**a, **b})); y2 = e.process(x[SR // 2:])
    y = np.concatenate([y1, y2])
    d = np.abs(np.diff(y, axis=0)).max(axis=1)
    steady = max(d[SR // 4:SR // 2 - 2].max(), d[SR // 2 + SR // 4:].max())
    assert d[SR // 2 - 2:SR // 2 + 960].max() < 1.5 * steady


# ---------------------------------------------------------------- cost

HEAVY = [("compressor", {}), ("freqshift", dict(shift=137.0, spread=0.6, feedback=0.5)),
         ("resonator", dict(root=69, chord=3)), ("resonator", dict(root=72, chord=4)), ("resonator", dict(root=63, chord=5)),
         ("resonator", dict(root=24, chord=3)), ("resonator", dict(root=84, chord=3))]


@pytest.mark.parametrize("kind,p", HEAVY)
def test_cost_under_budget(kind, p):
    x = np.random.default_rng(1).standard_normal((SR * 2, 2)) * 0.1
    e = T.CLASSES_NEW[kind](2)
    e.set(params(kind, **p))
    best = 1e9
    for _ in range(3):
        ts = []
        for i in range(0, len(x) - 512, 512):
            t0 = time.perf_counter(); e.process(x[i:i + 512]); ts.append(time.perf_counter() - t0)
        best = min(best, float(np.median(ts)) * 1000)
    print(f"{kind} {p}: {best:.3f} ms per 512-sample block")
    assert best < 0.4


# ---------------------------------------------------------------- the catalogue's entries

def test_catalogue_entries():
    for t, spec in T.TYPES_NEW.items():
        keys = [k for k, *_ in spec["params"]]
        rng = {k: (lo, hi) for k, _, lo, hi, _, _, _ in spec["params"]}
        for k, _, lo, hi, d, sc, _u in spec["params"]:
            assert lo <= d <= hi and (sc in ("lin", "log", "int", "toggle") or sc.startswith("choice:"))
        cands = T.CANDIDATES[t]
        assert len(cands) == 7
        names = [n for n, _, _ in cands]
        assert len(set(names)) == len(names)
        for name, ps, tip in cands:
            assert 0 < len(name) <= 16 and tip
            for k, v in ps.items():
                assert k in rng and rng[k][0] <= v <= rng[k][1], (t, name, k)
        knobs = [k for k, _, _, _, _, sc, _ in spec["params"] if not (sc == "toggle" or sc.startswith("choice"))]
        assert sorted(sum(T.ROWS[t], [])) == sorted(knobs)
        assert set(T.HELP[t]["p"]) == set(keys) and T.HELP[t]["how"] and T.WHAT[t][0].islower()
        assert T.CLASSES_NEW[t] and set(T.TYPES_NEW) == set(T.CLASSES_NEW)
    text = repr((T.CANDIDATES, T.WHAT, T.HELP))
    assert "—" not in text and " rather than " not in text
