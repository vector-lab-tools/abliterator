# Abliterator. Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
import numpy as np

from abliterator.dsp import Master


def test_blocks_equal_whole():
    x = np.random.default_rng(0).standard_normal((48000, 2)).astype(np.float32) * 0.3
    x[20000:] *= 0.05                                      # a fall, for auto level to follow
    a = Master(48000, 2)
    a.lowpass_hz = 2000
    whole = a.process(x)
    b = Master(48000, 2)
    b.lowpass_hz = 2000
    parts = np.concatenate([b.process(x[i:i + 512]) for i in range(0, len(x), 512)])
    assert np.allclose(whole, parts, atol=1e-6)
    c = Master(48000, 2)
    c.lowpass_hz = 2000
    frames = np.concatenate([c.process(x[i:i + 1920]) for i in range(0, len(x), 1920)])   # as offline, a frame at a time
    assert np.allclose(whole, frames, atol=1e-6)


def test_stereo_is_two_monos():
    """Without auto level, which is linked across the channels."""
    x = np.random.default_rng(1).standard_normal((2000, 2)).astype(np.float32) * 0.3
    a = Master(48000, 2)
    a.level = False
    s = a.process(x)
    for ch in range(2):
        b = Master(48000, 1)
        b.level = False
        m = b.process(x[:, ch].copy())
        assert np.allclose(s[:, ch], m, atol=1e-6)


def test_off_is_volume_only():
    x = np.ones(100, np.float32)
    m = Master(48000, 1)
    m.on, m.volume = False, 0.25
    assert np.allclose(m.process(x), 0.25)


def test_auto_level_holds_the_level_and_leaves_silence_alone():
    t = np.arange(48000 * 8) / 48000
    x = (np.sin(2 * np.pi * 220 * t) * np.where(t < 4, 0.02, 0.4)).astype(np.float32)   # −37 dB, then −11 dB
    m = Master(48000, 1)
    m.limit, m.volume, m.lowpass_hz = False, 1.0, 20000
    y = np.concatenate([m.process(x[i:i + 512]) for i in range(0, len(x), 512)])
    db = lambda a: 20 * np.log10(np.sqrt(np.mean(a.astype(np.float64) ** 2)))
    assert abs(db(y[48000 * 3:48000 * 4]) + 20) < 0.5 and abs(db(y[48000 * 7:]) + 20) < 0.5
    z = Master(48000, 1)
    assert np.abs(z.process(np.zeros(48000, np.float32))).max() == 0


def test_widen_keeps_the_mono_sum_and_any_block_size():
    import numpy as np
    from abliterator.dsp import Master
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((9600, 2)) * 0.1).astype(np.float32)
    x[:, 1] = x[:, 0] * 0.9 + x[:, 1] * 0.1                     # close to mono, as the model writes
    def run(block, **kw):
        m = Master(48000, 2); m.update({"level": False, "limit": False, "lowpass_hz": 23999, "volume": 1.0, **kw})
        return np.concatenate([m.process(x[i:i + block]) for i in range(0, len(x), block)])
    a, b = run(512, width=1.6, spread=0.5), run(1920, width=1.6, spread=0.5)
    assert np.allclose(a, b, atol=1e-6)
    plain = run(512)
    assert np.allclose(a.sum(axis=1), plain.sum(axis=1), atol=1e-4)
    mono = run(512, width=0.0)
    assert np.allclose(mono[:, 0], mono[:, 1], atol=1e-6)
    side = lambda y: np.std(y[:, 0] - y[:, 1])
    assert side(a) > 2 * side(plain)


def test_dynamics_same_at_any_block_size_and_compresses():
    import numpy as np
    from abliterator.dsp import Master
    t = np.arange(48000) / 48000
    loud = (np.sin(2 * np.pi * 220 * t) * np.where(t < 0.5, 0.05, 0.8))[:, None].repeat(2, 1).astype(np.float32)
    def run(block, **kw):
        m = Master(48000, 2); m.update({"level": False, "limit": False, "lowpass_hz": 23999, "volume": 1.0, **kw})
        return np.concatenate([m.process(loud[i:i + block]) for i in range(0, len(loud), block)]), m
    on = {"dyn_on": True, "dyn_ratio": 8.0, "dyn_thresh": -20.0, "dyn_release": 0.2}
    a, m = run(512, **on); b, _ = run(1920, **on)
    assert np.allclose(a, b, atol=1e-6)
    plain, _ = run(512)
    late = slice(36000, 48000)
    assert np.abs(a[late]).max() < 0.5 * np.abs(plain[late]).max()
    assert m.readings()["comp_db"] < -6
    off, _ = run(512, dyn_on=False, dyn_ratio=8.0)
    assert np.allclose(off, plain)
    g, mg = run(512, dyn_on=True, dyn_ratio=1.0, gate_thresh=-20.0)
    assert np.abs(g[2000:20000]).max() < 0.1 * np.abs(plain[2000:20000]).max()   # the quiet half is gated


def test_conceal_covers_a_dropout_without_a_step():
    import numpy as np
    from abliterator.dsp import Conceal
    sr, n = 48000, 512
    t = np.arange(n * 40) / sr
    sig = (0.5 * np.sin(2 * np.pi * 330 * t))[:, None].repeat(2, 1).astype(np.float32)
    c, out = Conceal(sr), []
    for i in range(40):
        blk = sig[i * n:(i + 1) * n]
        out.append(c.gap(n, blk.shape) if 10 <= i < 16 else c.good(blk))
    y = np.concatenate(out)
    plain = np.abs(np.diff(sig[:, 0])).max()
    assert np.abs(np.diff(y[:, 0])).max() <= 1.6 * plain
    assert np.abs(y[13 * n: 16 * n, 0]).max() == 0                           # silent once the fade (30 ms, to the end of its block) has run
    assert np.allclose(y[17 * n:], sig[17 * n:])
    hard = np.concatenate([sig[:10 * n], np.zeros((6 * n, 2), np.float32), sig[16 * n:]])
    assert np.abs(np.diff(hard[:, 0])).max() > 5 * plain                      # where plain silence would click


def test_declick_redraws_bursts_and_leaves_attacks():
    import numpy as np
    from abliterator.dsp import Declick
    sr = 48000
    t = np.arange(sr) / sr
    x = (0.1 * np.sin(2 * np.pi * 220 * t))[:, None].repeat(2, 1)
    x[12000:12005, :] += np.array([0.6, -0.9, 0.8, -0.7, 0.5])[:, None]          # a zig-zag burst, as Quantise makes
    k = np.exp(-np.arange(400) / 60.0) * np.sin(np.arange(400) / 3.0)          # a smooth attack, as a drum makes
    x[30000:30400, :] += 0.5 * k[:, None]
    x = x.astype(np.float32)

    def run(block):
        d = Declick(2)
        return np.concatenate([d.process(x[i:i + block]) for i in range(0, len(x), block)]), d
    a, d = run(512)
    b, _ = run(1920)
    assert np.array_equal(a, b)
    A = Declick.AHEAD
    y = a[A:]
    assert d.repaired >= 1
    assert np.abs(np.diff(y[11990:12020, 0])).max() < 0.05
    assert np.allclose(y[29990:30500], x[29990:30500], atol=1e-6)
    assert np.allclose(y[:11990], x[:11990], atol=1e-6)
