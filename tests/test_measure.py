# Abliterator. Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
"""The measures used to find musical ranges."""
import numpy as np

from abliterator import measure as ms


def _sine(f=440, s=3.0, a=0.3):
    t = np.arange(int(48000 * s)) / 48000
    return np.stack([a * np.sin(2 * np.pi * f * t)] * 2, 1)


def test_tone_and_noise():
    tone = ms.measure(_sine())
    noise = ms.measure(np.random.default_rng(0).standard_normal((144000, 2)) * 0.1)
    assert tone["flatness"] < 0.05 < 0.5 < noise["flatness"]
    assert abs(tone["rms_db"] - 20 * np.log10(0.3 / np.sqrt(2))) < 0.1
    assert abs(tone["centroid_hz"] - 440) < 60 and noise["centroid_hz"] > 8000
    assert tone["clip"] == 0 and ms.measure(_sine(a=1.5))["clip"] > 0.1


def test_timbre_distance_ignores_level():
    a, b = ms.measure(_sine(a=0.3)), ms.measure(_sine(a=0.03))
    assert ms.timbre_distance(a, b) < 0.01 and abs(a["rms_db"] - b["rms_db"] - 20) < 0.1
    assert ms.timbre_distance(a, ms.measure(_sine(f=3000))) > 3


def test_silence():
    x = _sine()
    x[: len(x) // 2] = 0
    assert abs(ms.measure(x)["silence"] - 0.5) < 0.02
