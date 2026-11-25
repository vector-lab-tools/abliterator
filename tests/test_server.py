# Abliterator. Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
"""The server's guard and routes that need no model (run in .venv: .venv/bin/python -m pytest tests)."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402

OK = {"host": f"127.0.0.1:{server.PORT}"}
W = {**OK, "X-Abliterator": "1"}


@pytest.fixture
def c():
    return TestClient(server.app, base_url=f"http://127.0.0.1:{server.PORT}")


def test_refuses_foreign_host(c):
    r = c.get("/api/presets", headers={"host": "evil.example:8775"})
    assert r.status_code == 403


def test_refuses_write_without_header(c):
    r = c.post("/api/stop", json={}, headers=OK)
    assert r.status_code == 403


def test_presets_read(c):
    r = c.get("/api/presets", headers=OK)
    assert r.status_code == 200
    ps = r.json()["presets"]
    assert ps and all({"id", "prompt", "seeds"} <= set(p) for p in ps)


def test_play_refused_until_ready(c):
    r = c.post("/api/play", json={"preset": "warm-drone"}, headers=W)
    assert r.status_code == 409


def test_unknown_engine(c):
    r = c.post("/api/engine", json={"engine": "nope"}, headers=W)
    assert r.status_code == 400


def test_typed_prompts_are_kept_and_removed(tmp_path, monkeypatch):
    from abliterator import presets as pr
    monkeypatch.setattr(pr, "USER", tmp_path / "prompts.json")
    p = pr.add("  dusty   jazz trio, brushed drums ")
    assert p["prompt"] == "dusty jazz trio, brushed drums" and p["user"] and p["group"] == pr.GROUP
    assert pr.add("Dusty jazz trio, brushed drums")["id"] == p["id"]                 # case and spacing ignored: same id
    assert pr.add(pr.builtin()[0]["prompt"])["id"] == pr.builtin()[0]["id"]         # a preset's own prompt is the preset
    assert [x["id"] for x in pr.presets()][-1] == p["id"]
    with pytest.raises(ValueError):
        pr.add("x")
    assert pr.remove(p["id"]) and not pr.remove(p["id"]) and not pr.user()
