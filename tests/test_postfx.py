# Abliterator. Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
"""The effect slots: each gives the same output whatever block size it is fed, and does what it says."""
import numpy as np
import pytest

from abliterator import postfx as fx


def noise(s=2.0):
    x = np.random.default_rng(0).standard_normal((int(48000 * s), 2)) * 0.1
    x[24000:] *= 0.2
    return x


@pytest.mark.parametrize("kind", list(fx.TYPES))
def test_blocks_equal_whole(kind):
    x = noise()
    on = [{"type": kind, "on": True, "params": {"mix": 0.7, "feedback": 0.6, "size": 3.0, "amount": 6, "low": 6, "high": -6, "mode": 1, "res": 4, "bits": 5, "hold": 7, "pos": 0.4, "depth": 0.5, "rate": 1.3, "width": 250, "gain": -3, "bass": 1, "freeze": 1}}]
    outs = []
    for block in (len(x), 512, 1920, 333):
        r = fx.Rack(2)
        r.update(on)
        outs.append(np.concatenate([r.process(x[i:i + block]) for i in range(0, len(x), block)]))
    for o in outs[1:]:
        assert np.allclose(outs[0], o, atol=1e-9)
    assert not np.allclose(outs[0], x)


def test_off_is_untouched():
    x = noise()
    r = fx.Rack(2)
    r.update([{"type": "reverb", "on": False, "params": {"mix": 1}}])
    assert np.array_equal(r.process(x), x)


def test_reverb_tail_and_delay_echo():
    imp = np.zeros((48000 * 3, 2)); imp[0] = 1.0
    r = fx.Rack(2); r.update([{"type": "reverb", "on": True, "params": {"size": 2.0, "mix": 1.0, "predelay": 0}}])
    y = r.process(imp)
    e = lambda a, b: float(np.sqrt(np.mean(y[int(a * 48000):int(b * 48000)] ** 2)))
    assert e(0.1, 0.3) > 10 * e(2.4, 2.6) > 0
    d = fx.Rack(2); d.update([{"type": "delay", "on": True, "params": {"time": 250, "feedback": 0.0, "mix": 1.0, "pingpong": 0}}])
    y = d.process(imp)
    assert int(np.argmax(np.abs(y[100:, 0]))) + 100 == 12000


def test_clean_fills_and_clamps():
    s = fx.clean([{"type": "delay", "params": {"feedback": 3}}, {"type": "nonsense"}])
    assert len(s) == 2 and s[0]["params"]["feedback"] == 0.95 and s[1]["type"] == fx.DEFAULT_SLOTS[1]["type"] and not s[0]["on"]
    assert fx.clean(None) == [] and len(fx.clean([{"type": "plate"}] * 12)) == fx.MAX_SLOTS
    assert fx.clean([{"type": "hall", "id": 7}])[0]["id"] == 7 and "id" not in fx.clean([{"type": "hall", "id": "x"}])[0]


def test_an_empty_chain_plays_as_four_slots_off():
    x = np.random.default_rng(2).standard_normal((9600, 2)).astype(np.float32) * 0.1
    old, new = fx.Rack(2), fx.Rack(2)
    old.update(fx.DEFAULT_SLOTS)                                        # how a sound saved before 0.1.6 holds its effects
    new.update([])
    assert np.array_equal(old.process(x), x) and np.array_equal(new.process(x), x)


def test_removing_an_effect_keeps_the_others_tails():
    imp = np.zeros((4800, 2), np.float32); imp[0] = 1.0
    sil = np.zeros((4800, 2), np.float32)
    a = {"type": "delay", "on": True, "id": 1, "params": {"time": 50, "feedback": 0.6, "mix": 1.0, "pingpong": 0}}
    b = {"type": "hall", "on": True, "id": 2, "params": {}}
    r, ref = fx.Rack(2), fx.Rack(2)
    r.update([b, a]); ref.update([a])
    r.process(imp); ref.process(imp)
    r.update([a])                                                       # the hall is removed; the delay keeps ringing
    y, want = r.process(sil), ref.process(sil)
    assert np.abs(y).max() > 0.01
    assert np.abs(y - want).max() > 0                                   # its input had passed through the hall, so it differs
    r3 = fx.Rack(2); r3.update([a]); r3.process(imp); r3.update([dict(a, id=9)])
    assert np.abs(r3.process(sil)).max() == 0                           # a new id is a new effect: no tail


REVERBS = ["plate", "hall"]
COLOURS = [None, 0, 1, 2]                                            # None: a slot saved by 0.1.5, with no colour param
KC = [(k, c) for k in REVERBS for c in COLOURS]
SR = 48000


def effect(kind, colour=None, **p):
    if colour is not None:
        p["colour"] = colour
    f = fx.CLASSES[kind](2)
    f.set(fx.clean([{"type": kind, "params": p or {"mix": 0.3}}])[0]["params"])
    return f


def t60(y):
    """Schroeder backward integration; the time from -5 to -25 dB, times three."""
    e = (y ** 2).sum(1)[::-1].cumsum()[::-1]
    db = 10 * np.log10(e / e[0] + 1e-300)
    return 3 * (int(np.argmax(db < -25)) - int(np.argmax(db < -5))) / 48000


def above(y, hz, a=0.3, b=2.0):
    """The share of the energy between a and b seconds that lies above hz, in dB."""
    s = y[int(a * SR):int(b * SR)].sum(1)
    S = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
    fr = np.fft.rfftfreq(len(s), 1 / SR)
    return float(10 * np.log10(S[fr > hz].sum() / S.sum() + 1e-30))


@pytest.mark.parametrize("kind,colour", KC)
def test_reverb_blocks_any_size(kind, colour):
    x = noise(0.6)
    x[9000:] = 0                                                     # the tail runs through the later blocks
    whole = effect(kind, colour, mix=0.8, predelay=0, mod=1).process(x)
    for block in (512, 4096, 333, 241, 239, 17):
        f = effect(kind, colour, mix=0.8, predelay=0, mod=1)
        assert np.allclose(np.concatenate([f.process(x[i:i + block]) for i in range(0, len(x), block)]), whole, atol=1e-9)
    f = effect(kind, colour, mix=0.8, predelay=0, mod=1)
    short = x[:4000]
    assert np.allclose(np.concatenate([f.process(short[i:i + 1]) for i in range(len(short))]), whole[:4000], atol=1e-9)


@pytest.mark.parametrize("kind,colour", KC)
def test_reverb_stable_at_extremes(kind, colour):
    x = np.random.default_rng(1).standard_normal((48000 * 6, 2)) * 0.3
    long = "decay" if kind == "plate" else "size"
    for size in ([0.5, 2.0] if kind == "plate" else [None]):
        for damp in (0.0, 1.0):
            for mod in (0.0, 1.0):
                p = {long: 20.0, "damp": damp, "mod": mod, "mix": 1.0, "predelay": 0.0}
                if size:
                    p["size"] = size
                y = effect(kind, colour, **p).process(x)
                r = lambda a: float(np.sqrt(np.mean(y[a * 48000:(a + 1) * 48000] ** 2)))
                assert np.isfinite(y).all() and r(5) < 3 * 0.3 and r(5) < 1.15 * r(3), (p, r(3), r(5))


@pytest.mark.parametrize("kind,colour", KC)
def test_reverb_silence_after_tail(kind, colour):
    assert not effect(kind, colour, mix=1).process(np.zeros((48000, 2))).any()
    long = "decay" if kind == "plate" else "size"
    # (Size, Damp, seconds). 0.1.5's plate is kept as it was, and at Size 2 its rounding flush (_flush) holds values
    # near 2e-34 going round for good, so it is checked at its defaults only. At Size 2 the plate's modulated
    # allpasses (feedback 0.7, about 2200 and 2900 samples) ring for about 1.2 s whatever the Decay, hence 16 s.
    if colour is None:
        corners = [(None, None, 12)]
    elif kind == "plate":
        corners = [(0.5, 0.0, 12), (2.0, 0.0, 16), (1.0, 1.0, 12)]
    else:
        corners = [(None, 0.0, 12), (None, 1.0, 12)]
    for size, damp, sec in corners:
        p = {long: 0.5, "mix": 1.0, "mod": 1.0}
        if size is not None:
            p["size"] = size
        if damp is not None:
            p["damp"] = damp
        imp = np.zeros((48000 * sec, 2)); imp[0] = 1.0
        y = effect(kind, colour, **p).process(imp)
        assert not y[-48000:].any(), p                               # exact zeros, no denormals left circulating


@pytest.mark.parametrize("kind,colour", KC)
def test_reverb_decay_follows_param(kind, colour):
    long = "decay" if kind == "plate" else "size"
    for d in (0.5, 2.0, 6.0):
        imp = np.zeros((int(48000 * max(3.0, 2 * d)), 2)); imp[0] = 1.0
        y = effect(kind, colour, **{long: d, "mix": 1.0, "predelay": 0.0, "damp": 0.0, "mod": 0.0}).process(imp)
        assert 0.7 * d < t60(y) < 1.25 * d, (d, t60(y))


@pytest.mark.parametrize("kind,colour", KC)
def test_reverb_level_and_balance(kind, colour):
    x = np.random.default_rng(2).standard_normal((48000 * 3, 2)) * 0.1
    y = effect(kind, colour, mix=1.0).process(x)[48000:]
    ratio = float(y.std() / 0.1)
    left, right = y.std(0)
    assert 0.1 < ratio < 2.0 and 0.8 < left / right < 1.25, (ratio, left, right)


@pytest.mark.parametrize("kind,colour", KC)
def test_reverb_mix_zero_is_identity(kind, colour):
    x = noise(1.0)
    assert np.array_equal(effect(kind, colour, mix=0.0).process(x), x)


@pytest.mark.parametrize("kind", REVERBS)
def test_reverb_colours_sound(kind):
    """Cubic reads keep more top end than 0.1.5's linear ones; 1980s is darker than Now; 1970s stops at about 10 kHz."""
    long = "decay" if kind == "plate" else "size"
    imp = np.zeros((48000 * 3, 2)); imp[0] = 1.0
    y = {c: effect(kind, c, **{long: 3.0, "mix": 1.0, "predelay": 0.0, "damp": 0.0, "mod": 1.0}).process(imp) for c in COLOURS}
    assert above(y[0], 8000) > above(y[None], 8000) + 0.8
    assert above(y[1], 8000) < above(y[0], 8000) - 2
    assert above(y[2], 13000) < -60 and above(y[0], 13000) > -30
    assert not np.allclose(y[0], y[None])


def test_reverb_colour_change_while_running():
    """Moving a slot between colours (and so between tank rates) mid-stream stays finite and keeps sounding."""
    x = noise(2.0)
    f = effect("plate", 0, mix=1.0)
    h = effect("hall", 0, mix=1.0)
    outs = []
    for i, c in enumerate([0, 2, 1, 2, 0]):
        for e in (f, h):
            e.set({**e.p, "colour": c})
        seg = x[i * 19200:(i + 1) * 19200]
        outs.append(np.concatenate([f.process(seg), h.process(seg)]))
    y = np.concatenate(outs)
    assert np.isfinite(y).all() and y.std() > 0.01


def test_reverb_types_round_trip():
    slots = [{"type": "plate", "on": True, "params": {"decay": 7, "size": 9}}, {"type": "hall", "on": True, "params": {"mod": 0.2, "colour": 2}}]
    r = fx.Rack(2)
    r.update(slots)
    assert isinstance(r.fx[0], fx.Plate) and isinstance(r.fx[1], fx.Hall)
    s = r.slots
    assert s[0]["params"]["decay"] == 7 and s[0]["params"]["size"] == 2.0 and s[0]["params"]["mix"] == 0.3
    assert "colour" not in s[0]["params"]                            # named params but no colour: saved by 0.1.5
    assert s[1]["params"]["mod"] == 0.2 and s[1]["params"]["size"] == 3.0 and s[1]["params"]["colour"] == 2
    assert fx.clean(s) == s
    r2 = fx.Rack(2)
    r2.update(s)
    x = noise(0.5)
    assert np.array_equal(r.process(x), r2.process(x))
    cat = fx.catalogue()
    assert [q["key"] for q in cat["plate"]["params"]] == ["colour", "decay", "size", "damp", "predelay", "mod", "mix"]
    for k in REVERBS:
        q = cat[k]["params"][0]
        assert q["scale"] == "choice:2020s,1980s,1970s" and q["default"] == 0.0
        new = {q["key"]: q["default"] for q in cat[k]["params"]}      # a slot made in the UI (fxDefaults in app.js)
        assert fx.clean([{"type": k, "params": new}])[0]["params"]["colour"] == 0
        assert fx.clean([{"type": k}])[0]["params"]["colour"] == 0
    r.update([{"type": "reverb", "on": True}] + s[1:])               # a slot whose type changes starts fresh
    assert type(r.fx[0]) is fx.Reverb


def _played(blocks, events, n=48000, smooth=True, seed=4):
    """A rack fed noise in blocks of the given sizes, with updates at sample positions (rounded down to a block edge)."""
    x = np.random.default_rng(seed).standard_normal((n, 2)) * 0.1
    r = fx.Rack(2); r.smooth = smooth
    out, pos, ev, b = [], 0, sorted(events, key=lambda e: e[0]), 0
    while pos < n:
        while ev and ev[0][0] <= pos:
            r.update(ev.pop(0)[1])
        k = min(blocks[b % len(blocks)], n - pos); b += 1
        out.append(r.process(x[pos:pos + k])); pos += k
    return np.concatenate(out), x


A = {"type": "delay", "on": True, "id": 1, "params": {"time": 120, "feedback": 0.4, "mix": 0.5, "pingpong": 0}}
P = {"type": "plate", "on": True, "id": 2, "params": {}}
EVENTS = [(0, [A]), (9600, [A, P]), (19200, [P, A]), (24000, [P, dict(A, on=False)]),
          (28800, [dict(P, params={"size": 1.6}), dict(A, on=False)]), (38400, [dict(P, params={"size": 1.6})])]


def test_fades_are_the_same_at_any_block_size():
    whole, _ = _played([4800], EVENTS)
    for sizes in ([480], [960, 480, 1920, 1440], [160, 320]):    # every event falls on an edge of these
        y, _ = _played(sizes, EVENTS)
        assert np.allclose(y, whole, rtol=0, atol=1e-12), sizes


def test_a_change_to_the_chain_dips_the_output():
    y, x = _played([480], [(0, [A]), (9600, [A, P])])
    lvl = lambda a, b: float(np.sqrt(np.mean(y[a:b] ** 2)))
    assert lvl(9600 + 720, 9600 + 720 + 1920) < 0.3 * lvl(4800, 9600)    # at the bottom: about -16 dB
    assert lvl(30000, 40000) > 0.5 * lvl(4800, 9600)
    jump = np.abs(np.diff(y[9000:12000], axis=0)).max()
    assert jump < 4 * np.abs(np.diff(x[9000:12000], axis=0)).max()


def test_a_plate_size_change_waits_until_its_wet_is_out():
    r = fx.Rack(2); r.smooth = True
    r.update([P]); r.process(np.zeros((4800, 2)))
    r.update([dict(P, params={"size": 1.6})])
    assert r.slots[0]["params"]["size"] == 1.0 and r.pend[0]["size"] == 1.6
    r.process(np.zeros((fx.FADE, 2)))
    r.process(np.zeros((1, 2)))
    assert r.slots[0]["params"]["size"] == 1.6 and r.pend[0] is None


def test_without_fades_the_rack_is_as_before():
    """Updates without ids, as a sound saved before 0.1.6 makes them: each place keeps its effect while
    its type stays, and an effect off is skipped at once."""
    ev = [(0, fx.DEFAULT_SLOTS), (9728, [dict(s, on=True) for s in fx.DEFAULT_SLOTS]), (19456, [dict(fx.DEFAULT_SLOTS[0], on=True)] + fx.DEFAULT_SLOTS[1:])]
    y, x = _played([512], ev, smooth=False)
    effs, ref, pos = [None] * 4, [], 0
    for at, sl in ev + [(48000, None)]:
        while pos < at:
            xs = x[pos:pos + 512]
            for (s, f) in zip(cur, effs):
                yy = f.process(xs)
                if s["on"]:
                    xs = yy
            ref.append(xs); pos += 512
        if sl:
            cur = fx.clean(sl)
            for i, s in enumerate(cur):
                if effs[i] is None:
                    effs[i] = fx.CLASSES[s["type"]](2)
                effs[i].set(s["params"])
    assert np.array_equal(y, np.concatenate(ref))


@pytest.fixture(scope="module")
def postfx_015():
    """postfx.py as released in 0.1.5, kept in tests/fixtures."""
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).resolve().parent / "fixtures" / "postfx_015.py"
    spec = importlib.util.spec_from_file_location("postfx_015", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("kind", ["reverb", "plate", "hall", "delay", "chorus", "drive", "eq", "filter", "crush"])
def test_replays_as_015(kind, postfx_015):
    """A slot as 0.1.5 saved it (every 0.1.5 param, no colour) renders bit for bit as 0.1.5 rendered it, for every
    type, in stereo and mono, with the slot passed back through the rack as a recipe would be."""
    old = postfx_015
    x = noise(1.5)
    x[40000:] = 0
    for p in ({}, {"mix": 0.8, "mod": 1.0, "damp": 0.0, "decay": 8.0, "size": 1.7, "feedback": 0.7, "amount": 6, "mode": 2, "bits": 5}):
        saved = old.clean([{"type": kind, "on": True, "params": p}])[:1]
        for ch in (2, 1):
            ro, rn = old.Rack(ch), fx.Rack(ch)
            ro.update(saved)
            rn.update(saved)
            assert rn.slots[0] == saved[0]                           # kept, and so saved again, as it came
            rn2 = fx.Rack(ch)
            rn2.update(rn.slots)
            xs = x[:, :ch]
            yo = np.concatenate([ro.process(xs[i:i + 512]) for i in range(0, len(xs), 512)])
            yn = np.concatenate([rn2.process(xs[i:i + 512]) for i in range(0, len(xs), 512)])
            assert np.array_equal(yo, yn), (kind, p, ch)


def test_a_saved_plate_takes_a_colour_with_its_wet_faded():
    r = fx.Rack(2); r.smooth = True
    old = {"type": "plate", "on": True, "id": 3, "params": {"decay": 2.5, "size": 1.0, "damp": 0.3, "predelay": 10, "mod": 0.5, "mix": 0.3}}
    r.update([old]); r.process(np.zeros((4800, 2)))
    assert "colour" not in r.slots[0]["params"]
    r.update([dict(old, params={**old["params"], "colour": 2})])
    assert "colour" not in r.slots[0]["params"] and r.pend[0]["colour"] == 2
    r.process(np.zeros((fx.FADE + 1, 2)))
    assert r.slots[0]["params"]["colour"] == 2


def test_a_change_made_while_nothing_plays_is_reported_as_made():
    r = fx.Rack(2); r.smooth = True
    a, b = {"type": "drive", "on": True, "id": 1, "params": {}}, {"type": "plate", "on": True, "id": 2, "params": {}}
    r.update([a, b]); r.process(np.zeros((512, 2)))
    r.update([a])                                                       # removed while stopped: no audio, no dip yet
    assert [s["type"] for s in r.wanted()] == ["drive"]
    r.update([a, dict(b, id=3, params={"size": 1.5})]); r.update([a, dict(b, id=3, params={"size": 1.5})])
    r.process(np.zeros((4800, 2)))
    r.update([a, dict(b, id=3, params={"size": 0.8})])                  # a jump waits for the wet to fade
    assert r.wanted()[1]["params"]["size"] == 0.8


def test_declick_effect_repairs_a_burst_at_any_block_size():
    t = np.arange(9600) / 48000
    x = np.stack([0.2 * np.sin(2 * np.pi * 220 * t)] * 2, 1)
    x[4800:4804, 0] += [0.9, -0.9, 0.9, -0.9]                         # a zig-zag burst, as a unit pushed hard writes
    def run(sizes):
        f = fx.Declicker(2); f.set(fx.clean([{"type": "declick"}])[0]["params"])
        out, pos, b = [], 0, 0
        while pos < len(x):
            k = sizes[b % len(sizes)]; out.append(f.process(x[pos:pos + k])); pos += k; b += 1
        return np.concatenate(out), f
    y, f = run([512])
    assert f.dc.repaired >= 1 and np.abs(np.diff(y[4800:4900, 0])).max() < 0.1
    for sizes in ([1], [37, 480], [9600]):
        assert np.array_equal(run(sizes)[0], y), sizes


def test_the_panel_switch_passes_the_sound_by_and_keeps_the_tails():
    x = np.random.default_rng(5).standard_normal((9600, 2)) * 0.1
    r = fx.Rack(2); r.smooth = True
    r.update([{"type": "hall", "on": True, "id": 1, "params": {"mix": 0.6}}])
    a = r.process(x[:4800])
    r.on = False
    b = r.process(x[4800:])
    assert np.array_equal(b[fx.FADE:], x[4800 + fx.FADE:])
    assert np.abs(b[:fx.FADE] - x[4800:4800 + fx.FADE]).max() > 0
    r2 = fx.Rack(2); r2.on = False; r2.update([{"type": "drive", "on": True, "params": {}}])
    assert np.array_equal(r2.process(x), x)


def test_eq8_cuts_and_bands():
    """EQ 8's low cut falls 12 or 24 dB an octave below its corner, a band lifts at its centre, and the slope
    switch keeps the number of sections."""
    from scipy.signal import sosfreqz
    e = fx.EQ8(2)
    e.p = {q[0]: q[4] for q in fx.TYPES["eq8"]["params"]}
    e.p.update(lowcut=200.0, b1k=6.0)
    _, h = sosfreqz(e.design(), [100.0, 1000.0, 15000.0], fs=fx.SR)
    db = 20 * np.log10(np.abs(h))
    assert abs(db[0] + 24) < 0.5 and abs(db[1] - 6) < 0.3 and abs(db[2]) < 0.3
    n = len(e.design())
    e.p["slope"] = 0
    _, h = sosfreqz(e.design(), [100.0], fs=fx.SR)
    assert abs(20 * np.log10(abs(h[0])) + 12) < 0.5 and len(e.design()) == n


def test_pan_balance_and_auto():
    """Pan is equal-power: unity at the centre, one side silent and the other 3 dB up at the ends; the auto-pan's
    swing is the same at any block size."""
    x = np.ones((24000, 2)) * 0.5
    for pos, want in ((0.0, (0.5, 0.5)), (-1.0, (0.5 * 2 ** 0.5, 0.0)), (1.0, (0.0, 0.5 * 2 ** 0.5))):
        e = fx.Pan(2)
        e.set({"pos": pos, "rate": 0.25, "depth": 0.0})
        assert np.allclose(e.process(x)[-1], want, atol=1e-6)
    outs = []
    for block in (len(x), 512, 333):
        e = fx.Pan(2)
        e.set({"pos": 0.2, "rate": 2.0, "depth": 0.8})
        outs.append(np.concatenate([e.process(x[i:i + block]) for i in range(0, len(x), block)]))
    assert all(np.allclose(outs[0], o, atol=1e-9) for o in outs[1:])


def _tape(x, **p):
    e = fx.Tape(2)
    e.set({**{q[0]: q[4] for q in fx.TYPES["tape"]["params"]}, **p})
    return e.process(x)


def test_tape_level_colour_and_hiss():
    """Tape keeps a -26 dBFS sine's level within about 1.5 dB at any Drive, adds even and odd harmonics as Drive
    rises, and its Hiss at full is about -50 dBFS on silence."""
    t = np.arange(fx.SR * 2) / fx.SR
    x = np.stack([0.05 * np.sin(2 * np.pi * 440 * t)] * 2, 1)
    rms = lambda a: 20 * np.log10(np.sqrt((a[fx.SR:] ** 2).mean()))  # noqa: E731
    h2 = []
    for dr in (0.0, 0.4, 1.0):
        y = _tape(x, drive=dr, wow=0.0, flutter=0.0, speed=3)
        assert abs(rms(y) - rms(x)) < 1.5, dr
        seg = y[fx.SR:, 0] * np.hanning(fx.SR)
        S, f = np.abs(np.fft.rfft(seg)), np.fft.rfftfreq(fx.SR, 1 / fx.SR)
        h2.append(S[np.argmin(abs(f - 880))] / S[np.argmin(abs(f - 440))])
    assert h2[0] < h2[1] < h2[2]
    s = _tape(np.zeros((2 * fx.SR, 2)), drive=0.0, noise=1.0)
    assert abs(rms(s) + 50) < 2


def test_tape_same_at_any_block_size():
    x = noise()
    outs = []
    for block in (len(x), 512, 333):
        e = fx.Tape(2)
        e.set({**{q[0]: q[4] for q in fx.TYPES["tape"]["params"]}, "noise": 0.5, "wow": 0.8, "flutter": 0.6})
        outs.append(np.concatenate([e.process(x[i:i + block]) for i in range(0, len(x), block)]))
    assert all(np.allclose(outs[0], o, atol=1e-9) for o in outs[1:])


@pytest.mark.parametrize("p", [{}, {"crunch": 1.0, "edge": 1.0, "complicate": 1.0, "steer": 3000.0}, {"crunch": 0.0, "edge": 0.0}])
def test_abliterated_keeps_the_level_and_any_block_size(p):
    """Abliterated's make-up gain holds the output near the input's level at any setting, and the output is the same
    at any block size (the ring modulator reads the absolute sample position)."""
    t = np.arange(48000 * 3) / 48000
    x = (0.05 * (np.sin(2 * np.pi * 110 * t) + 0.5 * np.sin(2 * np.pi * 165 * t)))[:, None].repeat(2, 1)
    outs = []
    for block in (len(x), 512, 333):
        e = fx.Abliterated(2)
        e.set({**{k: d for k, _, _, _, d, _, _ in fx.TYPES["abliterated"]["params"]}, **p})
        outs.append(np.concatenate([e.process(x[i:i + block]) for i in range(0, len(x), block)]))
    for o in outs[1:]:
        assert np.allclose(outs[0], o, atol=1e-9)
    db = lambda a: 20 * np.log10(np.sqrt((a[48000:] ** 2).mean()))
    assert abs(db(outs[0]) - db(x)) < 1.5
    assert not np.allclose(outs[0], x, atol=1e-3)


def test_presets_fit_their_types():
    assert fx.PRESETS_FILE.exists()
    cat = fx.catalogue()
    for t, spec in fx.TYPES.items():
        ps = fx.PRESETS.get(t, [])
        assert len(ps) >= 5, t
        names = [p["name"] for p in ps]
        assert len(set(names)) == len(names), t
        assert all(0 < len(n) <= 16 and "—" not in n for n in names), t
        rng = {k: (lo, hi) for k, _, lo, hi, _, _, _ in spec["params"]}
        for p in ps:
            assert p["params"] and p.get("tip"), (t, p["name"])
            for k, v in p["params"].items():
                assert k in rng, (t, p["name"], k)
                assert rng[k][0] <= v <= rng[k][1], (t, p["name"], k, v)
            assert fx.clean_slot({"type": t, "params": p["params"]}, 0)["params"] == {**fx.clean_slot({"type": t}, 0)["params"], **p["params"]}
        assert cat[t]["presets"] == ps


def _trim(params, x, block=512):
    e = fx.Trim(x.shape[1])
    e.set(fx.clean_slot({"type": "trim", "params": params}, 0)["params"])
    return np.concatenate([e.process(x[i:i + block]) for i in range(0, len(x), block)])


def _st(seconds=1.0, seed=4):
    t = np.arange(int(fx.SR * seconds)) / fx.SR
    r = np.random.default_rng(seed)
    return np.stack([0.3 * np.sin(2 * np.pi * 60 * t) + 0.1 * r.standard_normal(len(t)),
                     0.2 * np.sin(2 * np.pi * 60 * t + 1.0) + 0.1 * r.standard_normal(len(t))], 1)


def test_trim_neutral_and_block_size():
    x = _st()
    assert np.max(np.abs(_trim({}, x) - x)) < 1e-12
    p = {"width": 230, "bass": 1, "bassf": 180, "dc": 1, "gain": -6, "balance": 0.3, "chans": 3, "phr": 1}
    assert np.max(np.abs(_trim(p, x, 512) - _trim(p, x, 97))) < 1e-12


def test_trim_controls():
    """Each control does what its card says, once its 10 ms move has settled."""
    x = _st(); s = slice(fx.SR // 4, None); xs = x[s]
    y = _trim({"mono": 1}, x)[s]; assert np.max(np.abs(y[:, 0] - y[:, 1])) < 1e-12
    y = _trim({"chans": 3}, x)[s]; assert np.allclose(y[:, 0], xs[:, 1]) and np.allclose(y[:, 1], xs[:, 0])
    y = _trim({"chans": 1}, x)[s]; assert np.allclose(y[:, 0], xs[:, 0]) and np.allclose(y[:, 1], xs[:, 0])
    y = _trim({"phl": 1}, x)[s]; assert np.allclose(y[:, 0], -xs[:, 0]) and np.allclose(y[:, 1], xs[:, 1])
    y = _trim({"gain": 6}, x)[s]; assert abs(np.std(y) / np.std(xs) - 10 ** (6 / 20)) < 1e-6
    y = _trim({"balance": 1.0}, x)[s]; assert np.max(np.abs(y[:, 0])) < 1e-9 and np.allclose(y[:, 1], xs[:, 1])
    y = _trim({"mute": 1}, x)[s]; assert np.max(np.abs(y)) < 1e-9
    y = _trim({"width": 200}, x)[s]
    side, side0 = (y[:, 0] - y[:, 1]) / 2, (xs[:, 0] - xs[:, 1]) / 2
    assert np.allclose(side, 2 * side0) and np.allclose(y[:, 0] + y[:, 1], xs[:, 0] + xs[:, 1])
    y = _trim({"bass": 1, "bassf": 200}, x)[s]                     # the side's 60 Hz is taken out; the mid is untouched
    f = np.fft.rfftfreq(len(y), 1 / fx.SR); k = np.argmin(np.abs(f - 60))
    side, side0 = (y[:, 0] - y[:, 1]) / 2, (xs[:, 0] - xs[:, 1]) / 2
    assert np.abs(np.fft.rfft(side))[k] < 0.03 * np.abs(np.fft.rfft(side0))[k]
    assert np.allclose(y[:, 0] + y[:, 1], xs[:, 0] + xs[:, 1])
    d = _trim({"dc": 1}, np.full((fx.SR, 2), 0.05))[-fx.SR // 4:]; assert abs(float(d.mean())) < 1e-3


def test_trim_switches_do_not_click():
    x = _st(); e = fx.Trim(2); base = fx.clean_slot({"type": "trim"}, 0)["params"]
    e.set(base); a = e.process(x[:fx.SR // 2])
    e.set({**base, "phl": 1, "chans": 3, "mute": 0, "mono": 1}); b = e.process(x[fx.SR // 2:])
    y = np.concatenate([a, b])
    assert np.max(np.abs(np.diff(y, axis=0))) < 1.5 * np.max(np.abs(np.diff(x, axis=0)))


def test_trim_mono_input():
    x = _st()[:, :1]
    y = _trim({"phl": 1, "gain": -6, "width": 300, "chans": 3}, x)[fx.SR // 4:]
    assert np.allclose(y[:, 0], -x[fx.SR // 4:, 0] * 10 ** (-6 / 20))


def test_compressor_reports_gain_reduction():
    """The rack reports the Compressor's largest gain reduction since the last reading, in dB, and resets it; a quiet
    signal under the threshold reads 0."""
    r = fx.Rack(2)
    r.update([{"type": "compressor", "on": True, "id": 7, "params": {"thresh": -30.0, "ratio": 4.0, "attack": 1.0, "release": 100.0, "knee": 0.0}}])
    t = np.arange(48000) / 48000
    loud = np.stack([0.5 * np.sin(2 * np.pi * 220 * t)] * 2, 1)             # about −9 dBFS RMS: some 20 dB over
    r.process(loud)
    gr = next(x for x in r.readings() if x["id"] == 7)["gr"]
    assert 10.0 < gr < 20.0, gr
    assert next(x for x in r.readings() if x["id"] == 7)["gr"] == 0.0
    r.process(loud * 0.001)
    r.readings(); r.process(loud * 0.001)
    assert next(x for x in r.readings() if x["id"] == 7)["gr"] < 0.5
