# Abliterator. Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
"""The scope's readings (scope.py), on a made-up layer."""
import numpy as np
import pytest

from abliterator import scope as S

D = 64


def layer(mean=None):
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(D, D)))
    sc = np.geomspace(4, 0.1, D)
    v = np.zeros(D); v[3] = 1.0
    return S.Layer(np.zeros(D) if mean is None else mean, 10.0, q.T, sc, {"bright": v * 5})


def test_trace_is_the_stream_in_spread_units():
    L = layer()
    h = L.vt[0] * 4 + L.vt[1] * -0.25                 # one typical excursion along the first direction, a small one along the second
    r = S.read(L, h)
    assert r["trace"][0] == pytest.approx(1.0, abs=1e-5)
    assert r["trace"][1] == pytest.approx(-0.25 / L.sc[1], abs=1e-5)
    assert len(r["trace"]) == D                       # fewer directions than TRACE: all of them


def test_bands_cover_every_direction_once():
    L = layer()
    widths = np.diff(L.edges)
    assert L.edges[0] == 0 and L.edges[-1] == D and (widths > 0).all()
    h = (L.vt.T @ L.sc).astype(np.float32)           # one spread along every direction: every band at 1
    assert np.allclose(S.read(L, h)["bands"], 1.0, atol=1e-4)


def test_size_along_and_motion():
    L = layer(mean=np.full(D, 0.5))
    h = np.full(D, 0.5); h[3] += 2.0
    r = S.read(L, h)
    assert r["size"] == pytest.approx(np.linalg.norm(h) / 10.0, rel=1e-5)
    assert r["along"]["bright"] == pytest.approx(2.0 / np.linalg.norm(h), rel=1e-5)   # (h − mean) · v / ‖h‖
    assert r["motion"] == 0.0
    assert S.read(L, h, prev=h)["motion"] == pytest.approx(0.0, abs=1e-6)
    assert S.read(L, h, prev=-h)["motion"] == pytest.approx(2.0, abs=1e-6)


def test_scope_keeps_its_layers_and_history():
    sc = S.Scope(lambda v: layer())
    assert sc.set([6, 6, 3]) == [6, 3]
    with pytest.raises(ValueError):
        sc.set([0])
    with pytest.raises(ValueError):
        sc.set([13])
    assert len(sc.set([1, 2, 3, 4, 5])) == S.MAX_SCOPES
    sc.set([6])
    rng = np.random.default_rng(1)
    for f in range(S.HISTORY + 10):
        sc.add(f, {6: rng.normal(size=D), 9: rng.normal(size=D)})     # a layer without a scope is ignored
    p = sc.payload()
    assert list(p) == ["6"]
    m = p["6"]
    assert m["frame"] == S.HISTORY + 9 and len(m["size"]) == S.HISTORY and len(m["traces"]) == S.SEND
    assert m["frames"][-1] == m["frame"] and len(m["along"]["bright"]) == S.HISTORY
    sc.reset()
    assert sc.payload() == {}
