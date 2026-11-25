# Abliterator. Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
"""The spectrogram of what is heard (spectro.py)."""
import base64

import numpy as np

from abliterator import spectro as S


def cols(p):
    return np.frombuffer(base64.b64decode(p["data"]), np.uint8).reshape(p["n"], p["rows"])


def test_a_tone_lights_its_own_row():
    sp = S.Spectro(48000)
    t = np.arange(48000) / 48000
    x = 0.25 * np.sin(2 * np.pi * 1000 * t)
    for i in range(0, len(x), 512):                       # as the sound card hands it over, block by block
        sp.feed(np.stack([x[i:i + 512]] * 2, axis=1))
    p = sp.payload()
    assert p["end"] == (48000 - S.N) // S.HOP + 1 and p["n"] == min(p["end"], S.KEEP)
    c = cols(p)[-1]
    row = int(np.argmax(c))
    f0 = S.FLO * (S.FHI / S.FLO) ** (row / S.ROWS)
    f1 = S.FLO * (S.FHI / S.FLO) ** ((row + 1) / S.ROWS)
    assert f0 <= 1000 <= f1 * 1.05
    assert c[row] > 200 and np.median(c) < 60


def test_columns_carry_on_across_requests_and_reset():
    sp = S.Spectro(48000)
    sp.feed(np.zeros((4096, 2), np.float32))
    a = sp.payload()["end"]
    sp.feed(np.zeros((1920, 2), np.float32))
    b = sp.payload()["end"]
    assert b == a + 2                                      # 1,920 more samples: two more columns of 960
    sp.reset()
    assert sp.payload()["end"] == 0 and sp.payload()["n"] == 0
