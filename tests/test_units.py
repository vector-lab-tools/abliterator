"""The units' names fit a strip: its top bar holds NAME_MAX characters.
Runs where VAU is installed: .venv-mrt2/bin/python -m pytest tests/test_units.py"""
import pytest

pytest.importorskip("vau")

from abliterator import units  # noqa: E402


def test_names_fit_a_strip():
    for u in units.registry():
        assert len(units.short(u)) <= units.NAME_MAX, (u.id, units.short(u))
    assert all(len(n) <= units.NAME_MAX for n in units.NAMES.values())



HOST_KEYS = {"keepnorm", "wet", "env", "wfrom", "wto", "control"}


def test_presets_name_real_settings_in_range():
    """Every preset sets only settings its unit has (or the host's own), within their ranges, at a place the unit can act;
    the review's new defaults are presets of their unit."""
    cat = {u["id"]: u for u in units.catalogue()}
    for uid, ps in units.HOST_PRESETS.items():
        u = cat[uid]; prm = {p["key"]: p for p in u["params"]}
        assert len(ps) >= 5, uid
        assert len({q["name"] for q in ps}) == len(ps), uid
        for q in ps:
            assert q["block"] == "all" or (0 <= int(q["block"]) <= 12), (uid, q["name"])
            for k, v in q["params"].items():
                assert k in prm or k in HOST_KEYS, (uid, q["name"], k)
                p = prm.get(k)
                if p and isinstance(v, (int, float)) and "min" in p:
                    assert float(p["min"]) <= v <= float(p.get("max", v)), (uid, q["name"], k, v)
                if p and p.get("kind") == "choice":
                    assert v in p["choices"], (uid, q["name"], k, v)
    for uid, d in units.HOST_HEARD.items():
        assert d and any(q["params"] == d for q in units.HOST_PRESETS[uid]), uid
        for k, v in d.items():
            assert next(p for p in cat[uid]["params"] if p["key"] == k)["default"] == v, (uid, k)
    if "neuron" in units.HOST_HEARD:
        assert cat["neuron"]["place_default"] == 11
