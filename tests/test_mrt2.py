# Abliterator. Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
"""The Magenta RT 2 engine, offline (run in .venv-mrt2: .venv-mrt2/bin/python -m pytest tests).
Loads the model once (about 30 s with directions already made)."""
import numpy as np
import pytest

pytest.importorskip("mlx")
from abliterator import mrt2_proc as M  # noqa: E402

P = "slow ambient drone, warm analogue pads, no drums"
F = 40


@pytest.fixture(scope="module")
def p():
    e = M.Proc()
    e.load([P])
    return e


def test_seed_reproduces(p):
    a = p.render({"prompt": P, "seed": 3}, F)["pre"]
    b = p.render({"prompt": P, "seed": 3}, F)["pre"]
    c = p.render({"prompt": P, "seed": 4}, F)["pre"]
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)


def test_zero_dose_is_no_strip(p):
    a = p.render({"prompt": P, "seed": 3}, F)["pre"]
    b = p.render({"prompt": P, "seed": 3, "start": {"doses": {k: 0.0 for k in p.vectors}}}, F)["pre"]
    assert np.array_equal(a, b)


def test_move_heard_from_its_frame(p):
    name = sorted(p.vectors)[0]
    k = 20
    a = p.render({"prompt": P, "seed": 3}, F)["pre"]
    b = p.render({"prompt": P, "seed": 3, "moves": [{"frame": k, "doses": {name: 0.5}}]}, F)["pre"]
    d = np.abs(a - b).max(axis=1)
    assert not d[:k * M.HOP].any()
    first = int(np.argmax(d > 1e-6))
    assert k * M.HOP <= first < k * M.HOP + 48           # 48 samples: within 1 ms of the frame


def test_style_move_changes_sound(p):
    q = "soft glassy ambient chords, slow attack, no beat"
    p.tokens(q)
    a = p.render({"prompt": P, "seed": 3}, F)["pre"]
    b = p.render({"prompt": P, "seed": 3, "moves": [{"frame": 20, "prompt": q}]}, F)["pre"]
    assert np.array_equal(a[:20 * M.HOP], b[:20 * M.HOP]) and not np.allclose(a, b)


def test_master_post_differs_only_by_chain(p):
    out = p.render({"prompt": P, "seed": 3, "start": {"master": {"volume": 0.5, "on": False}}}, F)
    assert np.allclose(out["post"], out["pre"] * 0.5, atol=1e-6)


def test_neutral_strip_leaves_audio_alone(p):
    a = p.render({"prompt": P, "seed": 3}, F)["pre"]
    b = p.render({"prompt": P, "seed": 3, "start": {"chain": ["saturator layer=6 mix=0", "delay layer=4 mix=0"]}}, F)["pre"]
    assert np.array_equal(a, b)


def test_strip_move_heard_from_its_frame(p):
    k = 20
    a = p.render({"prompt": P, "seed": 3}, F)["pre"]
    b = p.render({"prompt": P, "seed": 3, "moves": [{"frame": k, "chain": ["saturator layer=6 drive=2"]}]}, F)["pre"]
    d = np.abs(a - b).max(axis=1)
    assert not d[:k * M.HOP].any()
    assert k * M.HOP <= int(np.argmax(d > 1e-6)) < k * M.HOP + 48


def test_bad_strip_refused(p):
    from vau import UnitError
    with pytest.raises(UnitError):
        p.attach_chain(["saturator layer=40"], strict=True)
    with pytest.raises(UnitError):
        p.attach_chain(["abliterate block=30 dir=brightness"], strict=True)       # no such block


def test_bad_strip_left_out_and_the_rest_attached(p):
    edits, report = p.attach_chain(["neuron block=3 op=scale value=2", "saturator layer=6 drive=2 mix=1"])
    assert "error" in report[0] and "error" not in report[1]
    assert 6 in edits["edits"] and not edits["outs"] and not edits["ins"]           # the neuron left nothing behind
    a = p.render({"prompt": P, "seed": 3, "start": {"chain": ["saturator layer=6 drive=2 mix=1"]}}, 20)["pre"]
    b = p.render({"prompt": P, "seed": 3, "start": {"chain": ["neuron block=3 op=scale value=2", "saturator layer=6 drive=2 mix=1"]}}, 20)["pre"]
    assert np.array_equal(a, b)


def test_freeze_equals_render(p):
    p.chain = ["delay layer=6 time=4 feedback=0.3 mix=0.5"]
    try:
        snap = p.freeze({"id": "t", "name": "T", "prompt": P}, 3, 1.0)
    finally:
        p.chain = []
    import soundfile as sf
    a, _ = sf.read(M.SNAPSHOTS / f"{snap['snapshot']}.wav", dtype="float32")
    b = p.render({"prompt": P, "seed": 3, "start": {"chain": ["delay layer=6 time=4 feedback=0.3 mix=0.5"]}}, 25)["pre"]
    (M.SNAPSHOTS / f"{snap['snapshot']}.wav").unlink()
    (M.SNAPSHOTS / f"{snap['snapshot']}.json").unlink()
    assert np.array_equal(a, b)


def test_abliterate_removes_the_direction(p):
    """At one block, flat kernel, no projection or size keeping: what attention writes has no
    component along the direction (measured on the hook's output, as the model's own precision rounds it)."""
    import torch
    bundle, _ = p.attach_chain(["abliterate block=6 dir=brightness shape=flat strength=1 projected=0 size=0 share=0"])
    from abliterator import directions as dr
    d = dr.load_all(M.DIRS)["brightness"]
    v = torch.from_numpy(np.asarray(d.layers[7], np.float32))
    v = v / v.norm()
    seen = []
    bundle["outs"][(6, "attn")].append(lambda m, a, o: seen.append(float((o.float() @ v).abs().max() / o.float().norm())))
    p._use(bundle)
    st = p.initial_state(3)
    cond = p.conditioning(P, M.DEFAULTS)
    for f in range(5):
        p.hooks.frame = f
        _, st = p.step(st, cond)
    p._use({"edits": {}, "outs": {}, "ins": {}})
    assert len(seen) == 5 and max(seen) < 1e-4       # the MLX hook (mlxunits) hands back the model's half precision: ~3e-5 left


def test_snap_back_returns_to_the_moment_the_strip_went_on(p):
    """On at frame 10, off with snap-back at 20: from 21 the model goes on as if the strip had never played,
    so the sound equals the plain render from frame 11; frame 20 is the crossfade into the plain frame 10."""
    H = M.HOP
    plain = p.render({"prompt": P, "seed": 3}, F)["pre"]
    on = "abliterate block=all dir=pulse shape=flat size=0 strength=2"
    moves = [{"frame": 10, "chain": [on], "anchor": ["7"]}, {"frame": 20, "chain": [], "snap": ["7"]}]
    b = p.render({"prompt": P, "seed": 3, "moves": moves}, F)["pre"]
    assert np.array_equal(b[:10 * H], plain[:10 * H])
    assert not np.allclose(b[10 * H:20 * H], plain[10 * H:20 * H])
    assert np.array_equal(b[21 * H:], plain[11 * H:(F - 10) * H])
    lingers = p.render({"prompt": P, "seed": 3, "moves": [dict(moves[0], anchor=[]), dict(moves[1], snap=[])]}, F)["pre"]
    assert not np.allclose(lingers[21 * H:], plain[11 * H:(F - 10) * H])     # without snap-back the strip's frames are still read


def test_neuron_changes_sound(p):
    a = p.render({"prompt": P, "seed": 3}, F)["pre"]
    b = p.render({"prompt": P, "seed": 3, "start": {"chain": ["neuron block=5 n=1,2,3,4,5,6,7,8 op=zero"]}}, F)["pre"]
    assert not np.array_equal(a, b)


def test_output_move_heard_from_its_frame(p):
    k = 20
    a = p.render({"prompt": P, "seed": 3}, F)["pre"]
    b = p.render({"prompt": P, "seed": 3, "moves": [{"frame": k, "chain": ["abliterate block=all dir=pulse strength=1"]}]}, F)["pre"]
    d = np.abs(a - b).max(axis=1)
    assert not d[:k * M.HOP].any()
    assert k * M.HOP <= int(np.argmax(d > 1e-6)) < k * M.HOP + 48


def _modspec(p, routes, sources=None):
    from abliterator import mod
    return mod.clean({"sources": sources or {}, "routes": routes}, list(p.vectors))


def test_mod_constant_macro_equals_the_dose(p):
    name = sorted(p.vectors)[0]
    spec = _modspec(p, [{"src": "m1", "tgt": f"dose:{name}", "depth": 0.5}], {"m1": {"value": 1.0}})
    a = p.render({"prompt": P, "seed": 3, "start": {"mod": spec}}, F)["pre"]
    b = p.render({"prompt": P, "seed": 3, "start": {"doses": {name: 0.4}}}, F)["pre"]
    assert np.array_equal(a, b)


def test_follower_moves_the_sound_and_its_post_replays(p):
    """The Follower reads the model's own output: a render with it routed to a dose and to the volume differs from
    the plain one, repeats exactly, and the output section's re-render from the model output (post_of, as a saved
    sound is made) matches the render's own output."""
    name = sorted(p.vectors)[0]
    spec = _modspec(p, [{"src": "follow", "tgt": f"dose:{name}", "depth": 0.5}, {"src": "follow", "tgt": "volume", "depth": -0.5}],
                    {"follow": {"floor_db": -60.0, "attack_s": 0.05, "release_s": 0.3}})
    plain = p.render({"prompt": P, "seed": 3}, F)["pre"]
    st = {"mod": spec, "master": {"volume": 0.8}}
    a = p.render({"prompt": P, "seed": 3, "start": st}, F)
    b = p.render({"prompt": P, "seed": 3, "start": st}, F)
    assert np.array_equal(a["pre"], b["pre"]) and not np.allclose(a["pre"], plain)
    post = p.post_of(a["pre"], {**st["master"]}, [], 0, spec)
    assert np.allclose(post, a["post"], atol=1e-6)


def test_mod_lfo_moves_the_sound_and_live_equals_render(p):
    name = sorted(p.vectors)[0]
    spec = _modspec(p, [{"src": "lfo1", "tgt": f"dose:{name}", "depth": 0.4}, {"src": "lfo2", "tgt": "temperature", "depth": 0.2}],
                    {"lfo1": {"rate_hz": 2.0}, "lfo2": {"rate_hz": 1.0, "shape": "square"}})
    plain = p.render({"prompt": P, "seed": 3}, F)["pre"]
    from abliterator import mod
    a = p.render({"prompt": P, "seed": 3, "start": {"mod": spec, "mod_curves": dict(mod.CURVES)}}, F)["pre"]   # as a recipe carries them
    assert not np.allclose(a, plain)
    # the live path, without the sound card: the same frames from _generate_one
    p._reset()
    p.preset, p.seed, p.prompt = {"id": "t", "prompt": P}, 3, P
    p.sampling, p.memo, p.mod_spec = dict(M.DEFAULTS), {}, spec
    p.state = p.initial_state(3)
    try:
        for _ in range(F):
            p._generate_one()
        live = np.concatenate(p.recorded)
    finally:
        p.mod_spec = _modspec(p, [])
        p.hooks.vec = None
        p._reset()
    assert np.array_equal(live, a)


def test_mod_master_volume_in_post(p):
    spec = _modspec(p, [{"src": "m1", "tgt": "volume", "depth": -0.5}], {"m1": {"value": 1.0}})
    out = p.render({"prompt": P, "seed": 3, "start": {"master": {"volume": 1.0, "on": False}, "mod": spec}}, F)
    assert np.allclose(out["post"], out["pre"] * 0.5, atol=1e-6)


def test_steer_switch(p):
    name = sorted(p.vectors)[0]
    plain = p.render({"prompt": P, "seed": 3}, F)["pre"]
    off = p.render({"prompt": P, "seed": 3, "start": {"doses": {name: 0.3}, "steer_on": False}}, F)["pre"]
    assert np.array_equal(plain, off)
    on = p.render({"prompt": P, "seed": 3, "start": {"doses": {name: 0.3}}}, F)["pre"]
    moved = p.render({"prompt": P, "seed": 3, "start": {"doses": {name: 0.3}}, "moves": [{"frame": 20, "steer_on": False}]}, F)["pre"]
    assert np.array_equal(on[:20 * M.HOP], moved[:20 * M.HOP]) and not np.array_equal(on, moved)


def test_layers_switch_keeps_the_strips(p):
    p.chain, p.layers_on = [], True
    r = p.set_chain(["saturator layer=6 drive=3 mix=1"], on=False)
    try:
        assert r["chain"] == ["saturator layer=6 drive=3 mix=1"] and p.heard_chain() == [] and not p.layers_on
        r = p.set_chain(p.chain, on=True)
        assert p.heard_chain() == ["saturator layer=6 drive=3 mix=1"]
    finally:
        p.set_chain([], on=True)
        p._use(p.chain_pending)
        p.chain_pending = None


def test_scope_reads_without_changing_the_sound(p):
    name = "brightness" if "brightness" in p.vectors else sorted(p.vectors)[0]

    def run(dose, read):
        p.meters.set([6] if read else [], p.n_layers)
        p.hooks.read_at = (6,) if read else ()
        p.meters.reset()
        memo, st, out = {}, p.initial_state(3), []
        for f in range(F):
            p.hooks.frame = f
            cond = p._apply({name: dose}, M.DEFAULTS, P, memo)
            a, st = p.step(st, cond)
            p._read_scopes(f)
            out.append(a)
        p.hooks.vec = None
        return np.concatenate(out), p.meters.payload()

    try:
        a, _ = run(0.0, False)
        b, m0 = run(0.0, True)
        assert np.array_equal(a, b)
        assert p.status()["meters"]["6"]["frame"] == F - 1 and p.status()["scopes"] == [6]
        assert m0["6"]["frame"] == F - 1 and len(m0["6"]["size"]) == F
        assert 0.3 < m0["6"]["size"][0] < 3                          # the stream near its typical size
        _, m1 = run(0.1, True)
        # the Steer at layer 6 adds ‖h‖ · 0.1 · v, so at the first frame (the same stream before it) the reading along v rises by about 0.1
        d = m1["6"]["along"][name][0] - m0["6"]["along"][name][0]
        assert 0.07 < d < 0.13, d
        assert p.render({"prompt": P, "seed": 3}, 4)["pre"] is not None and p.hooks.read_at == (6,)   # render reads nothing and leaves the scopes as they were
    finally:
        p.set_scopes([])
        p.hooks.read = {}


def test_unit_knob_modulated_renders_the_same_twice(p):
    import time
    ch = ["delay layer=6 time=4 feedback=0.3 mix=0.2"]
    m = {"sources": {"lfo1": {"rate_hz": 2.0}}, "routes": [{"src": "lfo1", "tgt": "unit:0:mix", "depth": 0.3}]}
    a = p.render({"prompt": P, "seed": 3, "start": {"chain": ch}}, F)["pre"]
    t0 = time.perf_counter()
    b = p.render({"prompt": P, "seed": 3, "start": {"chain": ch, "mod": m}}, F)["pre"]
    t1 = time.perf_counter()
    c = p.render({"prompt": P, "seed": 3, "start": {"chain": ch, "mod": m}}, F)["pre"]
    print(f"modulated render: {(t1 - t0) * 1000 / F:.1f} ms a frame")
    assert np.array_equal(b, c) and not np.array_equal(a, b)
    k = 3                                                   # before the LFO has moved the knob a step, nothing differs
    assert np.array_equal(a[:k * M.HOP], b[:k * M.HOP])


def test_speed_off_plays_at_one():
    """Off plays at 1 whatever the Speed knob says, and the knob keeps its value; the other modes play the knob."""
    assert M.heard_speed({"speed": 0.25, "speed_mode": "off"}) == (1.0, "tape")
    assert M.heard_speed({"speed": 0.25, "speed_mode": "smear"}) == (0.25, "smear")
    assert M.heard_speed({}) == (1.0, "tape")
    from abliterator.dsp import Master
    m = Master(48000, 2)
    m.update({"speed": 0.25, "speed_mode": "off"})
    assert (m.speed, m.speed_mode) == (0.25, "off")


def test_automatic_buffer(p):
    """A long step or a dry card adds frames in hand above the Buffer knob's, up to BOOST_MAX, and each take starts from the knob's."""
    p.lead, p.boost = 2, 0
    p._need_more(1)
    assert p.lead_eff() == 3
    p._need_more(9)
    assert p.boost == M.BOOST_MAX and p.lead_eff() == 2 + M.BOOST_MAX
    p._reset()
    assert p.boost == 0 and p.lead_eff() == 2
