# Abliterator. Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
import hashlib
import time

import numpy as np

from abliterator import directions as dr


def test_record_difference_of_means():
    rng = np.random.default_rng(0)
    u = np.zeros(8)
    u[3] = 1.0
    P = rng.standard_normal((6, 3, 8)) * 0.1 + u
    N = rng.standard_normal((6, 3, 8)) * 0.1 - u
    c = {"name": "t", "meta": {"title": "T", "positive": "up", "negative": "down"}, "positive": ["a"] * 6, "negative": ["b"] * 6}
    d = dr.record(c, P, N, {"model": "m"}, 0, time.time(), "off")
    v = np.asarray(d["layers"], np.float32)
    assert v.shape == (3, 8)
    assert np.argmax(np.abs(v[1])) == 3
    assert d["sha256"] == hashlib.sha256(v.astype("<f4").tobytes()).hexdigest()
    assert min(d["separation"]) > 3
    assert dr.Direction(d).summary()["positive"] == "up"


def test_contrast_sets_parse():
    cs = dr.contrasts()
    assert {c["name"] for c in cs} >= {"brightness", "pulse"}
    for c in cs:
        assert len(c["positive"]) == len(c["negative"]) > 0
