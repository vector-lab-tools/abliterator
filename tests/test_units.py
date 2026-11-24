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
        assert cat["neuron"]["place_default"] == 10


def test_vau_presets_match_ours():
    """VAU 0.3 carries the presets measured here (src/vau/presets/magenta-rt-2-small.json). Ours stay the source for the
    cards (they carry params, block and the search figures); VAU's must keep every one of them, unchanged, and start
    each unit where this host does."""
    import json
    import vau
    theirs = vau.presets_for("mrt2_small")
    if theirs is None:
        pytest.skip("VAU before 0.3 has no presets")
    mine = json.loads((units.ROOT / "data" / "presets" / "mrt2_small.json").read_text())["units"]
    assert set(theirs["units"]) == set(mine)
    for uid, ps in mine.items():
        t = {q["name"]: q for q in theirs["units"][uid]["presets"]}
        assert set(t) == {q["name"] for q in ps}, uid
        for q in ps:
            for k in ("line", "tip", "held_out"):
                assert t[q["name"]][k] == q[k], (uid, q["name"], k)
        start = theirs["units"][uid].get("start")
        here = units.HOST_HEARD.get(uid)
        assert (start is None) == (here is None), (uid, start)
        if start:
            assert units._preset(uid, start) == here, (uid, start)


def test_no_sink_on_this_model():
    """Magenta RT 2 has no attention sink: the host says so, Sink too is hidden, and Abliterate's envelope settings
    are dimmed while its Shape is flat (the manifest's `when`, read as the card reads it)."""
    host = units.AudioHost.__new__(units.AudioHost)
    assert host.sink() is False
    reg = units.registry().units
    for uid in ("gate", "limiter"):
        q = {p["key"]: p for p in units.params(reg[uid])}["sink"]
        assert q.get("hidden"), uid
    a = {p["key"]: p for p in units.params(reg["abliterate"])}
    for k in ("peak", "width", "edge"):
        assert a[k]["when"] == {"key": "shape", "in": ["linear", "bell"], "why": "no effect while Shape is flat"}, a[k].get("when")
    assert a["bias"].get("hidden")
