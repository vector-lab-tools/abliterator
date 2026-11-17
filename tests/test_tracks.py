# Abliterator. Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
"""Track effects and the desk's exports, on synthetic snapshots (no model, no sound card)."""
import json

import numpy as np
import pytest
import soundfile as sf

from abliterator import desk as dk
from abliterator import fx


def test_effects_blocks_equal_whole():
    x = (np.random.default_rng(0).standard_normal((24000, 2)) * 0.1).astype(np.float32)
    p = fx.clean({"delay_mix": 0.4, "reverb_mix": 0.5, "highpass_hz": 120, "lowpass_hz": 8000, "pan": 0.3})
    a = fx.Chain().process(x, p)
    c = fx.Chain()
    b = np.concatenate([c.process(x[i:i + 512], p) for i in range(0, len(x), 512)])
    assert np.allclose(a, b, atol=1e-6)


def test_neutral_chain_is_identity():
    x = (np.random.default_rng(1).standard_normal((4800, 2)) * 0.1).astype(np.float32)
    assert np.allclose(fx.Chain().process(x, fx.clean({})), x, atol=1e-6)


def test_delay_lands_on_its_sample():
    imp = np.zeros((24000, 2), np.float32)
    imp[0] = 1
    y = fx.Chain().process(imp, fx.clean({"delay_mix": 1.0, "delay_feedback": 0.0, "delay_ms": 250}))
    assert int(np.argmax(np.abs(y[100:, 0]))) + 100 == 12000


def test_ranges_clamp():
    p = fx.clean({"reverb_mix": 9, "highpass_hz": -5, "nonsense": 1})
    assert p["reverb_mix"] == 1.0 and p["highpass_hz"] == 20.0 and "nonsense" not in p


@pytest.fixture
def desk(tmp_path, monkeypatch):
    monkeypatch.setattr(dk, "SNAPSHOTS", tmp_path / "snapshots")
    monkeypatch.setattr(dk, "EXPORTS", tmp_path / "exports")
    monkeypatch.setattr(dk, "STATE", tmp_path / "tracks" / "desk.json")
    (tmp_path / "snapshots").mkdir()
    t = np.arange(48000) / 48000
    sf.write(tmp_path / "snapshots" / "a.wav", np.stack([np.sin(2 * np.pi * 220 * t)] * 2, 1).astype(np.float32) * 0.2, 48000, subtype="FLOAT")
    t2 = np.arange(16000) / 32000                                       # a MusicGen-like snapshot: 32 kHz mono, 0.5 s
    sf.write(tmp_path / "snapshots" / "b.wav", (np.sin(2 * np.pi * 330 * t2) * 0.2).astype(np.float32), 32000, subtype="FLOAT")
    (tmp_path / "snapshots" / "a.json").write_text(json.dumps({"id": "a", "seed": 1}))
    d = dk.Desk()
    d.add_clip(0, {"snapshot": "a", "label": "A", "seconds": 1.0})
    d.add_clip(1, {"snapshot": "b", "label": "B", "seconds": 0.5})
    return d, tmp_path


def test_mono_32k_is_resampled(desk):
    d, _ = desk
    assert d.tracks[1].audio.shape == (24000, 2)


def test_exports(desk):
    d, tmp = desk
    d.update_track(0, {"fx": {"reverb_mix": 0.5}})
    r = d.export("clip_pre", 0)
    a, sr = sf.read(tmp / "exports" / f"{r['export']}.wav", dtype="float32")
    assert sr == 48000 and a.shape == (48000, 2)
    side = json.loads((tmp / "exports" / f"{r['export']}.json").read_text())
    assert side["snapshot"]["seed"] == 1 and side["effects"] is None
    r = d.export("clip_post", 0)
    assert r["seconds"] == 1.0 + dk.TAIL_S
    r = d.export("track_pre", 1)                                         # looped to the mix length (the longer clip)
    assert r["seconds"] == 1.0
    d.update_track(1, {"mute": True})
    r = d.export("mix")
    side = json.loads((tmp / "exports" / f"{r['export']}.json").read_text())
    assert [t["i"] for t in side["tracks"]] == [0]
    with pytest.raises(ValueError):
        d.export("clip_pre", 5)


def test_state_survives_restart(desk):
    d, _ = desk
    d.update_track(0, {"name": "Drone", "level_db": -6, "fx": {"delay_mix": 0.2}})
    e = dk.Desk()
    assert e.tracks[0].name == "Drone" and e.tracks[0].level_db == -6 and e.tracks[0].fx["delay_mix"] == 0.2
    assert e.tracks[0].audio is not None and len(e.tracks[1].clips) == 1


def test_trigger_once_and_loop(desk, monkeypatch):
    d, _ = desk
    monkeypatch.setattr(d, "_open", lambda: None)                       # no sound card
    t = d.tracks[0]                                                      # a 1 s clip
    d.trigger(0, 0)
    assert t.live and t.pos == 0
    d.update_track(0, {"loop": False})
    a = np.concatenate([t.block(512) for _ in range(100)])               # 1.07 s
    assert not t.live and np.abs(a[48000:]).max() == 0 and np.abs(a[:48000]).max() > 0.1
    d.update_track(0, {"loop": True})
    d.trigger(0, 0)
    a = np.concatenate([t.block(512) for _ in range(100)])
    assert t.live and np.allclose(a[48000:51200], a[:3200])
    d.stop_track(0)
    assert not t.live and np.abs(t.block(512)).max() == 0


def test_stopped_track_rings_on(desk, monkeypatch):
    d, _ = desk
    monkeypatch.setattr(d, "_open", lambda: None)
    d.update_track(0, {"fx": {"reverb_mix": 0.8}})
    d.trigger(0, 0)
    d._mix_block(4800, d.heard())
    d.stop_track(0)
    assert np.abs(d._mix_block(512, d.heard())).max() > 1e-4             # the reverb's tail, after the stop


def test_library_rename_and_delete(desk, monkeypatch):
    from abliterator import library
    d, tmp = desk
    monkeypatch.setattr(library, "SOUNDS", tmp / "snapshots")
    assert {s["id"] for s in library.sounds()} == {"a"}                  # b has no recipe
    assert library.rename("a", "Drone one")["name"] == "Drone one"
    d.rename_sound("a", "Drone one")
    assert d.tracks[0].clips[0]["label"] == "Drone one"
    library.delete("a")
    d.drop_sound("a")
    assert not library.exists("a") and (tmp / "snapshots" / "trash" / "a.wav").exists()
    assert d.tracks[0].clips == [] and d.tracks[0].audio is None
    with pytest.raises(ValueError):
        library.rename("../x", "y")


def test_meters_read_each_track_after_its_fader(desk):
    d, _ = desk
    d.limit = False
    for t in d.tracks[:2]:
        t.live = True
    d.update_track(1, {"level_db": -6})
    d._mix_block(512, d.heard())
    assert d.tracks[0].peak == pytest.approx(0.2, rel=0.02)              # the 0.2 sine at 0 dB
    assert d.tracks[1].peak == pytest.approx(0.2 * 10 ** (-6 / 20), rel=0.05)
    assert d.state()["tracks"][0]["peak"] > 0 and d.state()["peak"] > 0
    d.update_track(0, {"level_db": -60})                                    # the foot of the fader is silence
    for _ in range(200):
        d._mix_block(512, d.heard())
    assert d.tracks[0].peak < 1e-3
    assert dk.gain(-60) == 0.0 and dk.gain(0) == 1.0
