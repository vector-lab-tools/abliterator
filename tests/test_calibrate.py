# Abliterator. Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
"""Musical ranges from the sweep's measures."""
from abliterator import calibrate as cb


def row(ctl, v, rms=-30.0, clip=0.0, silence=0.0, flat=0.01, shift=0.0, preset="a"):
    ltas = [-60.0 + shift * (i / 47) for i in range(48)]
    return {"preset": preset, "seed": 1, "control": ctl, "value": v, "rms_db": rms, "peak": 0.5, "clip": clip, "silence": silence,
            "centroid_hz": 1000.0, "flatness": flat, "ltas": ltas, "flux": 1.0}


def test_safe_range_stops_at_the_first_break_and_lays_out_by_timbre():
    rows = [row("temperature", 0.5, flat=0.6), row("temperature", 0.9, shift=2), row("temperature", 1.3), row("temperature", 1.7, shift=8),
            row("temperature", 2.0, clip=0.01)]
    c = cb.build(rows)["temperature"]
    assert c["safe"] == [0.9, 1.7]
    ts = [t for t, _ in c["curve"]]
    assert ts[0] == 0 and ts[-1] == 1 and ts[1] < ts[2] - ts[1]      # the bigger timbre step gets more travel


def test_dose_keeps_zero_at_the_centre_and_ears_narrow_it():
    rows = [row("dose:x", v, shift=abs(v) * 20) for v in (-0.2, -0.1, 0.0, 0.1, 0.2)] + [row("dose:x", 0.4, rms=-80)]
    c = cb.build(rows)["dose:x"]
    assert c["safe"] == [-0.2, 0.2] and [0.5, 0.0] in c["curve"]
    cal = {"controls": {"dose:x": c}, "ears": {"dose:x": {"lo": -0.1, "hi": 0.2}}}
    e = cb.effective(cal)["dose:x"]
    assert e["safe"] == [-0.1, 0.2] and e["curve"][0] == [0.0, -0.1] and [0.5, 0.0] in e["curve"] and e["curve"][-1][0] == 1.0
