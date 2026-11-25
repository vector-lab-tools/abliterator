# Abliterator. Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
"""Pitch, Shimmer and Freeze: the same output at any block size, mix 0 is the input, mono works, each does what it
says, and each fits the time budget. Run: .venv/bin/python -m pytest -q tests/test_fx_time.py"""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from abliterator import fx_time as T  # noqa: E402

SR = T.SR


def params(kind, **p):
    return {**{q[0]: q[4] for q in T.TYPES_NEW[kind]["params"]}, **p}


def make(kind, ch=2, **p):
    e = T.CLASSES_NEW[kind](ch)
    e.set(params(kind, **p))
    return e


def run(e, x, block=512):
    return np.concatenate([e.process(x[i:i + block]) for i in range(0, len(x), block)])


def noise(s=2.0, ch=2, seed=0):
    x = np.random.default_rng(seed).standard_normal((int(SR * s), ch)) * 0.1
    x[len(x) // 2:] *= 0.3
    return x


def sine(hz, s=2.0, a=0.3, ch=2):
    t = np.arange(int(SR * s)) / SR
    return (a * np.sin(2 * np.pi * hz * t))[:, None].repeat(ch, 1)


def rms_db(a):
    return 20 * np.log10(np.sqrt(np.mean(a ** 2)) + 1e-30)


ODD = {"pitch": {"shift": -7, "fine": 23.0, "window": 83.0, "feedback": 0.6, "mix": 0.8},
       "shimmer": {"decay": 9.0, "shimmer": 0.8, "interval": 2, "damp": 0.25, "mix": 0.7},
       "freeze": {"freeze": 1, "blur": 13, "drift": 0.7, "mix": 0.9}}


def run_odd(kind, x, block, ch=2, **over):
    """The effect at its ODD settings over x in blocks. Freeze starts off and goes on at 0.5 s: the first 0.5 s runs
    as one block, so the switch falls between blocks at the same sample for every block size."""
    p = {**ODD[kind], **over}
    if kind != "freeze":
        return run(make(kind, ch, **p), x, block)
    e = make(kind, ch, **{**p, "freeze": 0})
    a = e.process(x[:SR // 2])
    e.set({**e.p, "freeze": 1})
    return np.concatenate([a, run(e, x[SR // 2:], block)])


# ------------------------------------------------------------------------------------------------ all three

@pytest.mark.parametrize("kind", list(T.CLASSES_NEW))
def test_blocks_equal_whole(kind):
    x = noise()
    outs = [run_odd(kind, x, b) for b in (len(x), 512, 97, 333)]
    for o in outs[1:]:
        assert np.max(np.abs(outs[0] - o)) < 1e-9
    assert not np.allclose(outs[0], x)


@pytest.mark.parametrize("kind", list(T.CLASSES_NEW))
def test_blocks_equal_whole_with_knob_moves(kind):
    """Params change at 0.5 s and 1 s, between blocks, so the block sizes all divide 0.5 s (97 and 333 do not)."""
    x = noise()
    moves = {"pitch": [{"shift": 5, "window": 120.0}, {"feedback": 0.2, "fine": -40.0, "mix": 0.3}],
             "shimmer": [{"interval": 4, "decay": 3.0}, {"shimmer": 0.2, "damp": 0.8, "mix": 0.4}],
             "freeze": [{"freeze": 1}, {"freeze": 0, "mix": 0.4}]}[kind]
    outs = []
    for b in (480, 960, 96, 2400):
        e = make(kind, **ODD[kind]) if kind != "freeze" else make(kind, blur=3, drift=0.5)
        ys = []
        for i in range(0, len(x), b):
            if i in (SR // 2, SR):
                e.set({**e.p, **moves[0 if i == SR // 2 else 1]})
            ys.append(e.process(x[i:i + b]))
        outs.append(np.concatenate(ys))
    for o in outs[1:]:
        assert np.max(np.abs(outs[0] - o)) < 1e-9


@pytest.mark.parametrize("kind", list(T.CLASSES_NEW))
def test_mix_zero_is_the_input(kind):
    x = noise()
    assert np.array_equal(run_odd(kind, x, 512, mix=0.0), x)


@pytest.mark.parametrize("kind", list(T.CLASSES_NEW))
def test_mono_input(kind):
    x = noise(ch=1)
    y = run_odd(kind, x, 333, ch=1)
    assert y.shape == x.shape and np.all(np.isfinite(y)) and not np.allclose(y, x)
    assert np.max(np.abs(y - run_odd(kind, x, len(x), ch=1))) < 1e-9


@pytest.mark.parametrize("kind", list(T.CLASSES_NEW))
def test_tables_fit(kind):
    spec = T.TYPES_NEW[kind]
    keys = {q[0]: q for q in spec["params"]}
    for q in spec["params"]:
        assert q[2] <= q[4] <= q[3] and (q[5] in ("lin", "log", "int", "toggle") or q[5].startswith("choice:"))
    c = T.CANDIDATES[kind]
    assert len(c) == 7 and len({n for n, _, _ in c}) == 7
    for name, p, tip in c:
        assert 0 < len(name) <= 16 and "—" not in name + tip and tip
        for k, v in p.items():
            assert k in keys and keys[k][2] <= v <= keys[k][3], (name, k)
        if kind == "freeze":
            assert "freeze" not in p or p["freeze"] == 0
    text = T.WHAT[kind] + T.HELP[kind]["how"] + "".join(T.HELP[kind]["p"].values()) + "".join(t for _, _, t in c)
    assert "—" not in text and "rather than" not in text
    assert T.WHAT[kind][0].islower()
    assert set(T.HELP[kind]["p"]) == set(keys)
    rows = [k for r in T.ROWS[kind] for k in r]
    knobs = {k for k, q in keys.items() if q[5] in ("lin", "log", "int")}
    assert set(rows) == knobs and len(rows) == len(knobs)
    for k in T.JUMPS_NEW.get(kind, ()):
        assert k in keys


@pytest.mark.parametrize("kind", list(T.CLASSES_NEW))
def test_cost(kind):
    """Under 0.5 ms a 512-sample stereo block on average (Freeze switched on halfway, so both states count)."""
    x = noise(4.0)
    e = make(kind, **({**ODD[kind], "freeze": 0} if kind == "freeze" else ODD[kind]))
    run(e, x[:SR // 2])
    ts = []
    for i in range(SR // 2, len(x), 512):
        if kind == "freeze" and i >= 2 * SR and not e.p["freeze"]:
            e.set({**e.p, "freeze": 1})
        t = time.perf_counter()
        e.process(x[i:i + 512])
        ts.append(time.perf_counter() - t)
    ts = np.array(ts) * 1000
    print(f"{kind}: mean {ts.mean():.3f} ms, worst {ts.max():.3f} ms")
    assert ts.mean() < 0.5


# ------------------------------------------------------------------------------------------------ Pitch

def peak_hz(y):
    S = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    return np.fft.rfftfreq(len(y), 1 / SR)[np.argmax(S)]


def test_pitch_octave_up():
    y = run(make("pitch", shift=12, mix=1.0), sine(440))[SR:, 0]
    assert abs(peak_hz(y) - 880) < 8.8
    y = run(make("pitch", shift=-12, mix=1.0, window=100.0), sine(440))[SR:, 0]
    assert abs(peak_hz(y) - 220) < 2.2
    y = run(make("pitch", shift=7, fine=0.0, mix=1.0), sine(440))[SR:, 0]
    assert abs(peak_hz(y) / (440 * 2 ** (7 / 12)) - 1) < 0.01


def test_pitch_level_and_feedback_stable():
    x = noise(6.0)
    y = run(make("pitch", shift=12, mix=1.0), x)
    assert abs(rms_db(y[SR // 2:SR]) - rms_db(x[SR // 2:SR])) < 2.0
    for sh in (12, -12, 1, 0):
        y = run(make("pitch", shift=sh, fine=3.0, feedback=0.9, mix=1.0, window=20.0), x)
        assert np.all(np.isfinite(y)) and rms_db(y) < rms_db(x) + 12
        assert rms_db(y[-SR // 2:]) < rms_db(x[-SR // 2:]) + 12


def test_pitch_knob_moves_do_not_click():
    x = sine(220, 1.0, 0.5)
    e = make("pitch", shift=12, mix=0.2)
    a = e.process(x[:SR // 2])
    e.set({**e.p, "shift": 5, "fine": 30.0, "mix": 1.0, "feedback": 0.5})
    b = e.process(x[SR // 2:])
    y = np.concatenate([a, b])
    d = np.abs(np.diff(y, axis=0)).max(1)
    step = d[SR // 2 - 4800:SR // 2 + 4800].max()
    steady = max(d[SR // 4:SR // 2].max(), d[SR // 2 + 9600:].max())   # before the move, and once it has settled
    assert step < 1.5 * steady


# ------------------------------------------------------------------------------------------------ Shimmer

def band(y, lo, hi, a, b):
    s = y[int(a * SR):int(b * SR)].sum(1)
    S = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
    f = np.fft.rfftfreq(len(s), 1 / SR)
    return S[(f > lo) & (f < hi)].sum()


def test_shimmer_tail_climbs():
    """A 440 Hz burst: with Shimmer the tail at 2 to 3 s holds much more energy around 880 and 1760 Hz, relative to
    440 Hz, than the same reverb with Shimmer at 0."""
    x = sine(440, 4.0)
    x[SR // 2:] = 0
    ratio = []
    for s in (0.0, 0.6):
        y = run(make("shimmer", shimmer=s, decay=8.0, mix=1.0, damp=0.3), x)
        ratio.append((band(y, 850, 910, 2, 3) + band(y, 1700, 1820, 2, 3)) / band(y, 420, 460, 2, 3))
    assert ratio[1] > 30 * ratio[0]


def test_shimmer_decays_to_silence():
    x = noise(14.0)
    x[SR // 2:] = 0
    y = run(make("shimmer", decay=2.0, shimmer=1.0, mix=1.0), x)
    assert rms_db(y[12 * SR:]) < rms_db(y[SR // 2:SR]) - 60


@pytest.mark.parametrize("interval", range(5))
def test_shimmer_stable_at_the_top(interval):
    """Noise for 20 s at Decay 20, Shimmer 1, Damp 0: the output never rises 12 dB above the input's level."""
    x = np.random.default_rng(3).standard_normal((20 * SR, 2)) * 0.1
    y = run(make("shimmer", decay=20.0, shimmer=1.0, interval=interval, damp=0.0, mix=1.0), x, 4800)
    assert np.all(np.isfinite(y))
    ref = rms_db(x)
    win = [rms_db(y[i:i + SR // 2]) for i in range(0, len(y), SR // 2)]
    print(interval, "max %.1f dB over input" % (max(win) - ref), "last %.1f" % (win[-1] - ref))
    assert max(win) < ref + 12


# ------------------------------------------------------------------------------------------------ Freeze

def chordnoise(s, seed=5):
    t = np.arange(int(SR * s)) / SR
    r = np.random.default_rng(seed)
    x = sum(0.1 * np.sin(2 * np.pi * f * t + r.uniform(0, 6)) for f in (220, 330, 554, 880, 1320))
    x = x + 0.02 * r.standard_normal(len(t))
    return np.stack([x, 0.8 * x], 1)


OCT = [(125, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]


def bands_db(y):
    """Each octave band's power, per channel, in dB; the same for a stationary sound whatever the length of y."""
    w = np.hanning(len(y))
    S = np.abs(np.fft.rfft(y * w[:, None], axis=0)) ** 2 / (len(y) * (w ** 2).sum())
    f = np.fft.rfftfreq(len(y), 1 / SR)
    return np.array([10 * np.log10(S[(f >= a) & (f < b)].sum(0) + 1e-30) for a, b in OCT])


def test_freeze_holds_the_spectrum():
    """Freeze on at 1 s, the input silent from then on: in each second of the 5 s that follow, each octave band from
    125 Hz to 8 kHz of the output is within 3 dB of the input's over the half second before the freeze. (Random
    phases make a held sine's level wander by a few dB from one 0.2 s to the next, so the bands are taken a second
    at a time.)"""
    x = chordnoise(7.0)
    x[SR:] = 0
    e = make("freeze", blur=8, drift=0.0, mix=1.0)
    a = run(e, x[:SR], 480)
    e.set({**e.p, "freeze": 1})
    b = run(e, x[SR:], 480)
    ref = bands_db(x[SR // 2:SR])
    for i in range(5):
        seg = b[int((0.2 + i) * SR):int((1.2 + i) * SR)]
        assert np.max(np.abs(bands_db(seg) - ref)) < 3.0, (i, bands_db(seg) - ref)
    assert np.array_equal(a, x[:SR])                               # off: the input unchanged


def test_freeze_drift_moves_the_bands_and_keeps_the_level():
    x = chordnoise(7.0)
    e = make("freeze", blur=8, drift=1.0, mix=1.0)
    run(e, x[:SR], 480)
    e.set({**e.p, "freeze": 1})
    b = run(e, np.zeros((6 * SR, 2)), 480)
    ref = bands_db(x[SR // 2:SR])
    d = [bands_db(b[int((0.2 + i) * SR):int((1.2 + i) * SR)]) - ref for i in range(5)]
    assert max(np.max(np.abs(q)) for q in d) > 3.0
    assert all(abs(rms_db(b[int((0.5 + i) * SR):int((1.0 + i) * SR)]) - rms_db(x[SR // 2:SR])) < 2.5 for i in range(5))


def test_freeze_blocks_equal_whole_with_the_switch():
    """Freeze switched on at 1.0 s and off at 2.0 s, between blocks, for block sizes that divide 48000."""
    x = chordnoise(3.0) + noise(3.0, seed=2) * 0.3
    outs = []
    for b in (480, 960, 1200, 2400, 96):
        e = make("freeze", blur=12, drift=0.6, mix=0.8)
        ys = []
        for i in range(0, len(x), b):
            if i == SR:
                e.set({**e.p, "freeze": 1})
            if i == 2 * SR:
                e.set({**e.p, "freeze": 0})
            ys.append(e.process(x[i:i + b]))
        outs.append(np.concatenate(ys))
    for o in outs[1:]:
        assert np.max(np.abs(outs[0] - o)) < 1e-9
    y = outs[0]
    assert np.array_equal(y[:SR], x[:SR])
    assert np.array_equal(y[2 * SR + T.Freeze.FADE:], x[2 * SR + T.Freeze.FADE:])   # off and faded out: dry again
    assert not np.allclose(y[SR:2 * SR], x[SR:2 * SR])


def test_freeze_switch_fades_without_a_click():
    assert T.Freeze.FADE >= 0.05 * SR
    x = chordnoise(3.0)
    e = make("freeze", blur=2, drift=0.0, mix=1.0)
    a = run(e, x[:SR], 480)
    e.set({**e.p, "freeze": 1})
    b = run(e, x[SR:2 * SR], 480)
    e.set({**e.p, "freeze": 0})
    c = run(e, x[2 * SR:], 480)
    y = np.concatenate([a, b, c])
    lim = 1.5 * np.max(np.abs(np.diff(x, axis=0)))
    for t in (SR, 2 * SR):
        assert np.max(np.abs(np.diff(y[t - 4800:t + 9600], axis=0))) < lim
    # 25 ms after the switch the dry sound is still at least half its level (a fade of 50 ms or more)
    k = int(0.025 * SR)
    g = 0.5 * np.pi * k / T.Freeze.FADE
    assert np.cos(g) > 0.5
