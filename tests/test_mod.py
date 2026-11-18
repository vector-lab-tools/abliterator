# Abliterator. Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
"""Modulation: sources as functions of time, routes, and the controls they move."""
import math

import pytest

from abliterator import mod


def spec(**src):
    return mod.clean({"sources": src.get("sources", {}), "routes": src.get("routes", [])}, ["brightness", "pulse"])


def test_clean_fills_and_checks():
    s = mod.clean({"sources": {"lfo1": {"rate_hz": 99, "shape": "nonsense"}},
                   "routes": [{"src": "lfo1", "tgt": "dose:brightness", "depth": 3}, {"src": "lfo9", "tgt": "guidance", "depth": 1},
                              {"src": "env", "tgt": "dose:nothing", "depth": 1}, {"src": "m1", "tgt": "volume", "depth": 0}]}, ["brightness"])
    assert s["sources"]["lfo1"]["rate_hz"] == 4.0 and s["sources"]["lfo1"]["shape"] == "sine"
    assert set(s["sources"]) == set(mod.SOURCES)
    assert s["routes"] == [{"src": "lfo1", "tgt": "dose:brightness", "depth": 1.0}]


def test_lfo_shapes_and_period():
    for shape in mod.SHAPES:
        p = {**spec()["sources"]["lfo1"], "shape": shape, "rate_hz": 0.5}
        vals = [mod.lfo(p, k * 0.04) for k in range(200)]
        assert -1.0 <= min(vals) and max(vals) <= 1.0
        assert abs(mod.lfo(p, 0.3) - mod.lfo(p, 2.3)) < 1e-9          # period 2 s
    p = {**spec()["sources"]["lfo1"], "shape": "sine", "rate_hz": 1}
    assert abs(mod.lfo(p, 0.0) + 1) < 1e-9 and abs(mod.lfo(p, 0.5) - 1) < 1e-9


def test_random_is_seeded_and_bounded():
    p = spec()["sources"]["rand1"]
    a = [mod.rand(p, k * 0.04) for k in range(500)]
    b = [mod.rand(p, k * 0.04) for k in range(500)]
    c = [mod.rand({**p, "seed": 2}, k * 0.04) for k in range(500)]
    assert a == b and a != c and all(-1 <= x <= 1 for x in a)


def test_envelope_with_release_at_the_end():
    p = {"attack_s": 2.0, "decay_s": 2.0, "sustain": 0.5, "release_s": 2.0}
    assert mod.env(p, 0, None) == 0 and mod.env(p, 2, None) == 1 and mod.env(p, 10, None) == 0.5
    assert mod.env(p, 9, 10.0) == 0.25 and mod.env(p, 10, 10.0) == 0


def test_envelope_loop_holds_releases_and_repeats():
    p = {"attack_s": 2.0, "decay_s": 2.0, "sustain": 0.5, "release_s": 2.0, "loop": 1, "hold_s": 4.0}   # a 10 s cycle
    assert mod.env(p, 0, None) == 0 and mod.env(p, 2, None) == 1 and mod.env(p, 7, None) == 0.5
    assert mod.env(p, 9, None) == 0.25 and mod.env(p, 12, None) == 1          # half released, then round again
    assert mod.env({**p, "release_s": 0.0}, 7.9, None) == 0.5                  # no release: straight back to Attack
    assert mod.env({**p, "loop": 0}, 30, None) == 0.5                          # off: holds at Sustain
    assert spec(sources={"env": {"loop": 1, "hold_s": 99}})["sources"]["env"]["hold_s"] == 60


def test_routes_move_controls_and_clamp():
    s = spec(sources={"m1": {"value": 1.0}}, routes=[{"src": "m1", "tgt": "dose:brightness", "depth": 0.5}, {"src": "m1", "tgt": "guidance", "depth": -1},
                                                      {"src": "m1", "tgt": "lowpass_hz", "depth": -0.25}])
    d, smp = mod.apply_model(s, {"brightness": 0.1, "pulse": 0.0}, {"guidance": 3.0, "temperature": 1.3, "top_k": 40}, 0)
    assert math.isclose(d["brightness"], 0.5) and d["pulse"] == 0.0
    assert smp["guidance"] == 0.0 and smp["temperature"] == 1.3          # 3 − 8, clamped
    m = mod.apply_master(s, {"on": True, "lowpass_hz": 8000, "limit": True, "volume": 0.5}, 0)
    assert m["lowpass_hz"] == 4000 and m["volume"] == 0.5               # one octave down
    assert mod.active(s, model=True) and mod.active(s, model=False)
    assert not mod.active(spec())


def test_curved_target_moves_along_the_travel():
    mod.set_curves({"dose:brightness": [[0.0, -0.2], [0.5, 0.0], [0.75, 0.05], [1.0, 0.3]]})
    try:
        s = spec(sources={"m1": {"value": 1.0}}, routes=[{"src": "m1", "tgt": "dose:brightness", "depth": 0.25}])
        d, _ = mod.apply_model(s, {"brightness": 0.0, "pulse": 0.0}, {"guidance": 3, "temperature": 1.3, "top_k": 40}, 0)
        assert math.isclose(d["brightness"], 0.05)                  # a quarter of the travel up from the centre
        s = spec(sources={"m1": {"value": 1.0}}, routes=[{"src": "m1", "tgt": "dose:brightness", "depth": 1}])
        d, _ = mod.apply_model(s, {"brightness": 0.0, "pulse": 0.0}, {"guidance": 3, "temperature": 1.3, "top_k": 40}, 0)
        assert math.isclose(d["brightness"], 0.3)                   # held at the end of the musical range
        d, _ = mod.apply_model(s, {"brightness": 1.0, "pulse": 0.0}, {"guidance": 3, "temperature": 1.3, "top_k": 40}, 0)
        assert math.isclose(d["brightness"], 1.0 + 0.8)             # off the curve (full range): by value
    finally:
        mod.set_curves({})


def test_unit_and_effect_knobs_are_targets():
    s = mod.clean({"routes": [{"src": "lfo1", "tgt": "unit:0:mix", "depth": 0.5}, {"src": "m1", "tgt": "pfx:3:mix", "depth": 0.5},
                              {"src": "lfo1", "tgt": "unit:x:mix", "depth": 0.5}, {"src": "lfo1", "tgt": "pfx:8:mix", "depth": 0.5}]})
    assert [r["tgt"] for r in s["routes"]] == ["unit:0:mix", "pfx:3:mix"]
    assert mod.active(s, model=True) and mod.active(s, model=False)
    assert mod.offsets(s, 1.0) == {}                     # the plain targets are not touched


def test_apply_units_moves_a_strips_knob_along_its_travel():
    info = lambda kind, key: (0.0, 1.0, "lin", 0.5) if key == "mix" else (1.0, 32.0, "int", 4) if key == "time" else None
    s = mod.clean({"sources": {"m1": {"value": 1.0}}, "routes": [{"src": "m1", "tgt": "unit:1:mix", "depth": 0.25},
                                                                  {"src": "m1", "tgt": "unit:1:time", "depth": 1.0}]})
    lines = ["saturator layer=6 drive=2", "delay layer=6 time=4 feedback=0.3 mix=0.2"]
    out = mod.apply_units(s, lines, 0.0, None, info)
    assert out[0] == lines[0]
    kv = dict(b.split("=") for b in out[1].split()[1:])
    assert out[1].startswith("delay layer=6 time=32 feedback=0.3 mix=")
    assert float(kv["mix"]) == pytest.approx(0.45, abs=1 / mod.QUANT)
    s0 = mod.clean({"routes": [{"src": "m1", "tgt": "unit:1:mix", "depth": 0.25}]})      # the macro at 0: no change
    assert mod.apply_units(s0, lines, 0.0, None, info) == lines
    assert mod.apply_units(s, lines[:1], 0.0, None, info) == lines[:1]                   # a strip that is not there


def test_apply_master_moves_an_effect_knob_and_keeps_the_base():
    from abliterator import postfx
    base = {"lowpass_hz": 16000.0, "volume": 0.5, "fx": postfx.clean(postfx.DEFAULT_SLOTS)}
    s = mod.clean({"sources": {"m1": {"value": 1.0}}, "routes": [{"src": "m1", "tgt": "pfx:3:mix", "depth": 0.5},
                                                                  {"src": "m1", "tgt": "pfx:2:time", "depth": 1.0}]})
    m = mod.apply_master(s, base, 0.0)
    assert m["fx"][3]["params"]["mix"] == pytest.approx(0.8)                           # reverb mix 0.3 + half its travel
    assert m["fx"][2]["params"]["time"] == pytest.approx(1500.0)                        # a log knob to its top
    assert base["fx"][3]["params"]["mix"] == pytest.approx(0.3)


def test_move_knob_log_and_clamp():
    assert mod.move_knob(100.0, 10.0, 1000.0, "log", 0.5) == pytest.approx(1000.0)
    assert mod.move_knob(0.5, 0.0, 1.0, "lin", -2) == 0.0


def test_amount_scales_an_lfo_and_old_specs_keep_full_swing():
    full = mod.clean({"sources": {"lfo1": {"shape": "square"}}})
    half = mod.clean({"sources": {"lfo1": {"shape": "square", "amount": 0.5}}})
    assert full["sources"]["lfo1"]["amount"] == 1.0                     # a recipe from before Amount: unchanged
    assert mod.value(half, "lfo1", 0.1) == pytest.approx(0.5 * mod.value(full, "lfo1", 0.1))
    r = mod.clean({"sources": {"rand1": {"amount": 0}}})
    assert mod.value(r, "rand1", 3.3) == 0.0


def test_steps_drift_chaos_are_functions_of_time():
    s = spec()["sources"]
    st = {**s["steps"], "length": 4, "rate_hz": 1.0, "glide": 0.0}
    assert [mod.steps(st, t) for t in (0.5, 1.5, 2.5, 3.5, 4.5)] == [st["s0"], st["s1"], st["s2"], st["s3"], st["s0"]]
    g = {**st, "glide": 0.5}
    assert mod.steps(g, 0.4) == st["s0"] and abs(mod.steps(g, 0.999) - st["s1"]) < 0.01      # eased into the next
    d = [mod.drift(s["drift"], i * 0.5) for i in range(400)]
    assert d == [mod.drift(s["drift"], i * 0.5) for i in range(400)] and all(-1 <= x <= 1 for x in d) and max(d) - min(d) > 0.3
    c = [mod.chaos(s["chaos"], i * 0.1) for i in range(600)]
    assert all(-1 <= x <= 1 for x in c) and max(c) > 0.5 and min(c) < -0.5
    r = [mod.chaos({**s["chaos"], "system": "rossler"}, i * 0.1) for i in range(600)]
    assert r != c and all(-1 <= x <= 1 for x in r)
    assert mod.chaos({**s["chaos"], "seed": 2}, 30.0) != mod.chaos(s["chaos"], 30.0)


def test_follower_reads_only_frames_before():
    import numpy as np
    tr = mod.FollowTrack()
    for i in range(40):
        tr.push(np.full((1920, 2), 0.1 if 10 <= i < 20 else 0.0))
    f = spec()["sources"]["follow"]
    assert tr.at(f, 10) == 0.0 and tr.at(f, 12) > 0 and tr.at(f, 18) > 0.7       # frame 10 is loud; from frame 11 on it rises
    assert tr.at(f, 30) < tr.at(f, 21) and tr.at(f, 99) == tr.at(f, 40)          # falls after; past the end, the last
    assert mod.value(spec(), "follow", 0.6, None, None) == 0.0                    # no track: still
    inv = {**f, "invert": 1}
    assert tr.at(inv, 10) == 1.0 and abs(tr.at(inv, 18) - (1 - tr.at(f, 18))) < 1e-12   # inverted: quiet high, loud low
