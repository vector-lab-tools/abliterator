# Abliterator. Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
"""Presets of your own: saved under a name, replaced by the same name, deleted, kept on disk."""
import pytest

from abliterator import mypresets as mp


def test_save_replace_delete(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "FILE", tmp_path / "my_presets.json")
    mp.add("units", "delay", "Dub", {"time": 8, "mix": 0.4}, 6)
    mp.add("units", "delay", "Dub", {"time": 12, "mix": 0.4}, 6)
    mp.add("fx", "plate", "Big", {"size": 0.9})
    d = mp.load()
    assert [q["params"]["time"] for q in d["units"]["delay"]] == [12]
    assert d["fx"]["plate"][0]["name"] == "Big" and "block" not in d["fx"]["plate"][0]
    assert mp.remove("units", "delay", "Dub") and not mp.remove("units", "delay", "Dub")
    assert mp.load()["units"]["delay"] == []


def test_refuses_bad_input(tmp_path, monkeypatch):
    monkeypatch.setattr(mp, "FILE", tmp_path / "my_presets.json")
    for args in (("nope", "delay", "A", {"x": 1}), ("units", "../x", "A", {"x": 1}), ("units", "delay", "  ", {"x": 1}), ("units", "delay", "A", {})):
        with pytest.raises(ValueError):
            mp.add(*args)
