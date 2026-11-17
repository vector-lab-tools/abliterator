"""The units' names fit a strip: its top bar holds NAME_MAX characters.
Runs where VAU is installed: .venv-mrt2/bin/python -m pytest tests/test_units.py"""
import pytest

pytest.importorskip("vau")

from abliterator import units  # noqa: E402


def test_names_fit_a_strip():
    for u in units.registry():
        assert len(units.short(u)) <= units.NAME_MAX, (u.id, units.short(u))
    assert all(len(n) <= units.NAME_MAX for n in units.NAMES.values())

