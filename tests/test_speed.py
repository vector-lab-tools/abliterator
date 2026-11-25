"""The Speed stage: Tape mode's pitch and length, pass-through at 1, and live use (any block size, changes taken
while it runs) matching the offline render of the same changes to the bit."""
import numpy as np

from abliterator.speed import CHUNK, Speed

SR = 48000


def live(pre, blocks, plan, start=1.0):
    """As the engine uses it: the model audio arrives as it is written; the sound card asks in `blocks`;
    `plan` maps an output sample count to a new speed, set when that many have been asked for."""
    sp = Speed(pre.shape[1])
    sp.set(start)
    sp.target = sp.ratio = start
    have = [0]

    def supply(k):
        if have[0] + k > len(pre):
            return None
        x = pre[have[0]:have[0] + k]
        have[0] += k
        return x

    out, asked, bi = [], 0, 0
    while True:
        n = blocks[bi % len(blocks)]
        bi += 1
        for at, r in plan:
            if asked >= at and not getattr(sp, "_done_" + str(at), False):
                sp.set(r)
                setattr(sp, "_done_" + str(at), True)
        if sp.idle():
            if have[0] + n > len(pre):
                break
            x = pre[have[0]:have[0] + n]
            have[0] += n
            sp.skip(x)
            y = x
        else:
            y = sp.pull(n, supply)
            if y is None:
                break
        out.append(y)
        asked += n
    return np.concatenate(out), sp.changes


def test_tape_halves_pitch_and_doubles_length():
    t = np.arange(SR) / SR
    pre = np.stack([np.sin(2 * np.pi * 440 * t)] * 2, 1)
    y, _ = Speed.render(pre, 0.5, "tape", [])
    assert abs(len(y) - 2 * len(pre)) < 4 * CHUNK
    seg = y[SR // 2: SR // 2 + SR // 2, 0] * np.hanning(SR // 2)
    f = np.fft.rfftfreq(len(seg), 1 / SR)[np.argmax(np.abs(np.fft.rfft(seg)))]
    assert abs(f - 220) < 3


def test_one_passes_through():
    pre = np.random.default_rng(0).standard_normal((SR, 2)) * 0.1
    y, ch = live(pre, [512], [])
    assert ch == [] and np.array_equal(y, pre[:len(y)])


def test_live_matches_offline_render():
    pre = np.random.default_rng(1).standard_normal((SR * 6, 2)) * 0.1
    plan = [(20000, 0.5), (90000, 0.25), (160000, 0.8)]
    for blocks in ([512], [333, 700, 128]):
        y, changes = live(pre, blocks, plan)
        assert len(changes) == 3
        z, _ = Speed.render(pre, 1.0, "tape", changes)
        n = min(len(y), len(z))
        assert n > SR * 6
        assert np.array_equal(y[:n], z[:n]), blocks


def _tone(seconds=2.0, hz=440.0):
    t = np.arange(int(SR * seconds)) / SR
    return np.stack([0.3 * np.sin(2 * np.pi * hz * t)] * 2, 1)


def _peak_hz(y):
    seg = y[:, 0] * np.hanning(len(y))
    return np.fft.rfftfreq(len(y), 1 / SR)[np.argmax(np.abs(np.fft.rfft(seg)))]


def test_stretch_and_smear_keep_pitch_and_level():
    pre = _tone(2.0)
    rms = lambda a: np.sqrt((a ** 2).mean())  # noqa: E731
    for mode, ratio in (("stretch", 0.5), ("smear", 0.25)):
        y, _ = Speed.render(pre, ratio, mode, [])
        assert len(y) > 1.6 / ratio * SR * 0.8, mode               # about 1/ratio times as long (less the last window)
        mid = y[len(y) // 3: len(y) // 3 + SR // 2]
        assert abs(_peak_hz(mid) - 440) < 6, mode
        assert abs(20 * np.log10(rms(mid) / rms(pre))) < 2.5, mode


def test_spectral_live_matches_offline_render():
    pre = np.random.default_rng(3).standard_normal((SR * 8, 2)) * 0.1
    plan = [(10000, 0.5), (60000, 0.3), (140000, 0.7)]
    for mode in ("stretch", "smear"):
        sp_plan = [(at, r) for at, r in plan]
        y, changes = live_mode(pre, [512, 300], sp_plan, mode)
        assert len(changes) == 3
        z, _ = Speed.render(pre, 1.0, "tape", changes)
        n = min(len(y), len(z))
        assert n > SR * 4
        assert np.array_equal(y[:n], z[:n]), mode


def live_mode(pre, blocks, plan, mode):
    """As live() with every new speed set in `mode`."""
    sp = Speed(pre.shape[1])
    sp.set(1.0, "tape")
    have = [0]

    def supply(k):
        if have[0] + k > len(pre):
            return None
        x = pre[have[0]:have[0] + k]
        have[0] += k
        return x

    out, asked, bi, done = [], 0, 0, set()
    while True:
        n = blocks[bi % len(blocks)]
        bi += 1
        for at, r in plan:
            if asked >= at and at not in done:
                sp.set(r, mode)
                done.add(at)
        if sp.idle():
            if have[0] + n > len(pre):
                break
            x = pre[have[0]:have[0] + n]
            have[0] += n
            sp.skip(x)
            y = x
        else:
            y = sp.pull(n, supply)
            if y is None:
                break
        out.append(y)
        asked += n
    return np.concatenate(out), sp.changes


def test_engine_pacing_never_starves_a_mode_change():
    """As the engine paces the model (one frame of HOP samples, at most one every two 512-sample blocks, whenever
    it is at most LEAD frames ahead of what the stage has read, `consumed`), switching to Stretch and then to Smear,
    which reads half a 16,384 window ahead, makes no gap and is taken."""
    HOP, LEAD = 1920, 2
    src = np.random.default_rng(4).standard_normal((SR * 20, 2)) * 0.1
    sp = Speed(2)
    sp.set(1.0)
    made = [0]
    buf = [np.zeros((0, 2))]

    def supply(k):
        if len(buf[0]) < k:
            return None
        x, buf[0] = buf[0][:k], buf[0][k:]
        return x

    gaps = 0
    for b in range(1200):                                        # 512-sample blocks, about 13 s
        if b == 100:
            sp.set(0.5, "stretch")
        if b == 400:
            sp.set(0.25, "smear")
        if b % 2 == 0 and made[0] - sp.consumed // HOP <= LEAD and made[0] * HOP < len(src):
            buf[0] = np.concatenate([buf[0], src[made[0] * HOP:(made[0] + 1) * HOP]])
            made[0] += 1
        if sp.idle():
            x = supply(512)
            if x is None:
                gaps += 1
                continue
            sp.skip(x)
        elif sp.pull(512, supply) is None:
            gaps += 1
    assert [c[2] for c in sp.changes] == ["stretch", "smear"]
    assert gaps == 0
