# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Musical ranges for the knobs, from the sweep's measures (bench/sweep.py, measure.py).

For each control and preset, a step is marked broken when the output clips (more than 0.1% of
samples at full scale), falls toward silence (more than a quarter of it below −55 dBFS, or 24 dB
under the control's default), is overdriven (louder than −12 dBFS RMS: the model's output at the
defaults sits between −28 and −41), or turns to noise (spectral flatness above 0.3 and three
times the default's). The control's safe range is the run of unbroken steps around its default that every
preset shares. Within it, the knob is laid out by timbre: equal travel gives equal change in the
level-free spectrum (measure.timbre_distance between neighbouring steps, averaged over presets),
so the knob spends its travel where the sound changes and passes quickly where it does not. A
changed sampler takes a different path, so every step also differs by what seeds alone do (the
noise floor: the defaults on other seeds); that is subtracted first. 40% of the travel is spread
evenly over the steps, so every setting stays in reach, and a control whose changes stay within
the floor is laid out in even steps. A
dose knob is laid out on each side of 0 separately, so 0 stays at the centre.

These are measured bounds of where the sound breaks. Listening (the grid) can narrow a range
further (`ears` in the file).

The second paragraph of this docstring is written into the calibration file as its `method`.

    .venv/bin/python -m abliterator.calibrate      (from backend/; writes data/calibration/mrt2_small.json)
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from . import ROOT
from .measure import timbre_distance

RESULTS = ROOT / "_tmp" / "sweep" / "results.jsonl"
UNITS = ROOT / "_tmp" / "sweep" / "units.jsonl"
FILE = ROOT / "data" / "calibration" / "mrt2_small.json"
DEFAULTS = {"guidance": 3.0, "temperature": 1.3, "top_k": 40}
EVEN = 0.4            # the share of each knob's travel spread evenly over its steps; the rest follows the change
CLIP, SILENCE, DROP_DB, FLAT_ABS, FLAT_REL, LOUD_DB = 0.001, 0.25, 24.0, 0.3, 3.0, -12.0


def broken(r: Dict[str, Any], base: Dict[str, Any]) -> List[str]:
    if r.get("error"):
        return [f"cannot run ({r['error'][:80]})"]
    why = []
    if r["clip"] > CLIP:
        why.append(f"clips ({r['clip'] * 100:.2f}% of samples)")
    if r["silence"] > SILENCE or r["rms_db"] < base["rms_db"] - DROP_DB:
        why.append(f"falls silent ({r['rms_db']:.0f} dBFS)")
    if r["rms_db"] > LOUD_DB:
        why.append(f"overdriven ({r['rms_db']:.0f} dBFS RMS; the model's own level is −28 to −41)")
    if r["flatness"] > FLAT_ABS and r["flatness"] > FLAT_REL * base["flatness"]:
        why.append(f"turns to noise (flatness {r['flatness']:.2f})")
    return why


def default_of(control: str) -> float:
    return 0.0 if control.startswith("dose:") else DEFAULTS[control]


def curve(values: List[float], dist: List[float], centre: Optional[float]) -> List[List[float]]:
    """Knob positions (0 to 1) for the values, so that travel follows the timbre distance between
    neighbours. `centre` (for doses) is pinned at 0.5."""
    def lay(vs, ds, t0, t1):
        c = np.concatenate([[0.0], np.cumsum(np.maximum(ds, 1e-3))])
        c = c / c[-1] if c[-1] > 0 else np.linspace(0, 1, len(vs))
        return [[round(float(t0 + (t1 - t0) * x), 4), float(v)] for x, v in zip(c, vs)]
    if centre is None:
        return lay(values, dist, 0.0, 1.0)
    i = values.index(centre)
    lo = lay(values[:i + 1], dist[:i], 0.0, 0.5) if i > 0 else [[0.5, centre]]
    hi = lay(values[i:], dist[i:], 0.5, 1.0) if i < len(values) - 1 else [[0.5, centre]]
    return lo[:-1] + hi


def floor_of(by) -> Dict[str, Dict[str, float]]:
    """Per preset: how much the music varies by itself at the defaults, over seeds. The median
    timbre distance between seed pairs, and the spread of the level."""
    out = {}
    for pid, vals in by.get("floor", {}).items():
        base = by.get("guidance", {}).get(pid, {}).get(DEFAULTS["guidance"])
        rs = list(vals.values()) + ([base] if base else [])
        if len(rs) < 2:
            continue
        ds = [timbre_distance(a, b) for i, a in enumerate(rs) for b in rs[i + 1:]]
        out[pid] = {"timbre_db": round(float(np.median(ds)), 2), "level_sd_db": round(float(np.std([r["rms_db"] for r in rs])), 2),
                    "level_range_db": round(float(np.ptp([r["rms_db"] for r in rs])), 2), "seeds": len(rs)}
    return out


def build(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by: Dict[str, Dict[str, Dict[float, Dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        by[r["control"]][r["preset"]][float(r["value"])] = r
    fl = floor_of(by)
    noise = float(np.mean([f["timbre_db"] for f in fl.values()])) if fl else 0.0
    out: Dict[str, Any] = {"_floor": fl}
    for ctl, presets in sorted(by.items()):
        if ctl == "floor":
            continue
        d0 = default_of(ctl)
        values = sorted(set.intersection(*[set(v) for v in presets.values()]))
        if d0 not in values:
            continue
        per = {}
        ok_all = [True] * len(values)
        for pid, vals in presets.items():
            base = vals[d0]
            marks = []
            for j, v in enumerate(values):
                why = broken(vals[v], base)
                if why:
                    ok_all[j] = False
                marks.append({"value": v, "broken": why, "rms_db": vals[v]["rms_db"], "level_db": round(vals[v]["rms_db"] - base["rms_db"], 1),
                              "timbre_db": timbre_distance(vals[v], base), "centroid_hz": vals[v]["centroid_hz"], "flatness": vals[v]["flatness"]})
            per[pid] = marks
        i0 = values.index(d0)
        lo = i0
        while lo > 0 and ok_all[lo - 1]:
            lo -= 1
        hi = i0
        while hi < len(values) - 1 and ok_all[hi + 1]:
            hi += 1
        safe = values[lo:hi + 1]
        raw = [float(np.mean([timbre_distance(presets[p][a], presets[p][b]) for p in presets])) for a, b in zip(safe[:-1], safe[1:])]
        excess = [max(0.0, d - noise) for d in raw]
        n = max(1, len(raw))
        dist = [EVEN / n + (1 - EVEN) * (e / sum(excess)) if sum(excess) > 1e-6 else 1.0 / n for e in excess]
        level = [float(np.mean([presets[p][v]["rms_db"] - presets[p][d0]["rms_db"] for p in presets])) for v in safe]
        out[ctl] = {"default": d0, "safe": [safe[0], safe[-1]], "steps": safe,
                    "curve": curve(safe, dist, d0 if ctl.startswith("dose:") else None),
                    "level_db": [round(x, 1) for x in level], "step_timbre_db": [round(x, 2) for x in raw],
                    "beyond_seeds": round(float(np.mean([max(0.0, m["timbre_db"] - noise) for pm in per.values() for m in pm if not m["broken"]])), 2),
                    "swept": [values[0], values[-1]], "presets": per}
    return out


def build_units(rows: List[Dict[str, Any]], noise: float) -> Dict[str, Any]:
    """Each unit setting, judged against the same preset with no unit. A setting's safe range is
    the run of unbroken steps that holds its default, or, where the default itself breaks (some of
    VAU's defaults silence this model), the nearest unbroken run; the default here becomes the
    unbroken step nearest the old one. Placement (`at`) is a list of the places that hold up."""
    ref = {r["preset"]: r for r in rows if r["control"] == "unit:none"}
    by: Dict[str, Dict[str, Dict[float, Dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["control"] != "unit:none":
            by[r["control"]][r["preset"]][float(r["value"])] = r
    out: Dict[str, Any] = {}
    for ctl, presets in sorted(by.items()):
        presets = {k: v for k, v in presets.items() if k in ref}
        if not presets:
            continue
        values = sorted(set.intersection(*[set(v) for v in presets.values()]))
        d0 = next(iter(next(iter(presets.values())).values()))["default"]
        ok = [all(not broken(presets[p][v], ref[p]) for p in presets) for v in values]
        per = {pid: [{"value": v, "broken": broken(vals[v], ref[pid]), "rms_db": vals[v].get("rms_db"),
                      "level_db": round(vals[v]["rms_db"] - ref[pid]["rms_db"], 1) if vals[v].get("rms_db") is not None else None,
                      "timbre_db": timbre_distance(vals[v], ref[pid]) if vals[v].get("ltas") else None,
                      "centroid_hz": vals[v].get("centroid_hz"), "flatness": vals[v].get("flatness")} for v in values] for pid, vals in presets.items()}
        entry: Dict[str, Any] = {"unit": True, "vau_default": d0, "steps": values, "presets": per, "swept": [values[0], values[-1]]}
        if ctl.endswith(":at"):
            entry.update(kind="placement", safe_places=[v for v, k in zip(values, ok) if k], default=d0,
                         safe=[min(values), max(values)], curve=[])
            if d0 not in entry["safe_places"] and entry["safe_places"]:
                entry["default"] = min(entry["safe_places"], key=lambda v: (abs(v - d0), v))
            out[ctl] = entry
            continue
        runs, cur = [], []
        for v, k in zip(values, ok):
            if k:
                cur.append(v)
            elif cur:
                runs.append(cur)
                cur = []
        if cur:
            runs.append(cur)
        if not runs:
            entry.update(default=d0, safe=None, curve=[], broken_everywhere=True)
            out[ctl] = entry
            continue
        run = next((r for r in runs if r[0] <= d0 <= r[-1]), None) or min(runs, key=lambda r: min(abs(v - d0) for v in r))
        dm = d0 if run[0] <= d0 <= run[-1] else min(run, key=lambda v: abs(v - d0))
        if len(run) > 1:
            raw = [float(np.mean([timbre_distance(presets[p][a], presets[p][b]) for p in presets])) for a, b in zip(run[:-1], run[1:])]
            excess = [max(0.0, d - noise) for d in raw]
            n = len(raw)
            dist = [EVEN / n + (1 - EVEN) * (e / sum(excess)) if sum(excess) > 1e-6 else 1.0 / n for e in excess]
            cv = curve(run, dist, None)
        else:
            cv = []
        entry.update(default=dm, safe=[run[0], run[-1]], curve=cv, moved_default=dm != d0)
        out[ctl] = entry
    return out


def main():
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines()]
    cal = {"format": "abliterator.calibration.v1", "engine": "mrt2", "made": time.strftime("%Y-%m-%d %H:%M"),
           "method": __doc__.split("\n\n")[1].replace("\n", " ").strip(), "renders": len(rows), "controls": build(rows), "ears": {}}
    cal["floor"] = cal["controls"].pop("_floor")
    if UNITS.exists():
        noise = float(np.mean([f["timbre_db"] for f in cal["floor"].values()])) if cal["floor"] else 0.0
        rows = [json.loads(l) for l in UNITS.read_text().splitlines()]
        second = ROOT / "_tmp" / "sweep" / "units2.jsonl"           # the second pass replaces those units' rows
        if second.exists():
            rows2 = [json.loads(l) for l in second.read_text().splitlines()]
            again = {r["control"].split(":")[1] for r in rows2 if r["control"] != "unit:none"}
            rows = [r for r in rows if r["control"] == "unit:none" or r["control"].split(":")[1] not in again] + [r for r in rows2 if r["control"] != "unit:none"]
        cal["controls"].update(build_units(rows, noise))
    old = json.loads(FILE.read_text()) if FILE.exists() else {}
    cal["ears"] = old.get("ears", {})                     # your marks survive a re-measure
    FILE.parent.mkdir(parents=True, exist_ok=True)
    FILE.write_text(json.dumps(cal, indent=1))
    for k, c in cal["controls"].items():
        if c.get("kind") == "placement":
            print(f"{k:28s} holds up at {', '.join(f'{v:g}' for v in c['safe_places']) or 'none'}")
        elif c.get("unit"):
            print(f"{k:28s} " + ("breaks everywhere" if not c["safe"] else f"safe {c['safe'][0]:g} to {c['safe'][1]:g} (swept {c['swept'][0]:g} to {c['swept'][1]:g}); default {c['vau_default']:g}" + (f" → {c['default']:g}" if c.get('moved_default') else "")))
        else:
            print(f"{k:22s} safe {c['safe'][0]:g} to {c['safe'][1]:g} (swept {c['swept'][0]:g} to {c['swept'][1]:g}); level {c['level_db'][0]:+.0f} to {c['level_db'][-1]:+.0f} dB")


if __name__ == "__main__":
    main()


def _relay(curve: List[List[float]], lo: float, hi: float, centre: Optional[float]) -> List[List[float]]:
    """The curve cut to [lo, hi] and laid out again over the whole travel (a dose keeps 0 at the centre)."""
    ts, vs = [p[0] for p in curve], [p[1] for p in curve]

    def travel_at(v):
        for i in range(1, len(vs)):
            if v <= vs[i]:
                a = (v - vs[i - 1]) / (vs[i] - vs[i - 1]) if vs[i] > vs[i - 1] else 0.0
                return ts[i - 1] + a * (ts[i] - ts[i - 1])
        return ts[-1]
    pts = [[travel_at(lo), lo]] + [[t, v] for t, v in curve if lo < v < hi] + [[travel_at(hi), hi]]
    if centre is None:
        t0, t1 = pts[0][0], pts[-1][0]
        return [[round((t - t0) / (t1 - t0), 4) if t1 > t0 else 0.0, v] for t, v in pts]
    c = travel_at(centre)
    out = []
    for t, v in pts:
        if v < centre:
            out.append([round(0.5 * (t - pts[0][0]) / (c - pts[0][0]), 4) if c > pts[0][0] else 0.0, v])
        elif v > centre:
            out.append([round(0.5 + 0.5 * (t - c) / (pts[-1][0] - c), 4) if pts[-1][0] > c else 1.0, v])
        else:
            out.append([0.5, v])
    return out


def effective(cal: Dict[str, Any]) -> Dict[str, Any]:
    """The controls as the knobs use them: the measured curve, cut to the range your ears set."""
    out = {}
    for k, c in (cal.get("controls") or {}).items():
        c = dict(c)
        e = (cal.get("ears") or {}).get(k)
        if e and c.get("safe") and c.get("curve"):
            lo, hi = max(e["lo"], c["safe"][0]), min(e["hi"], c["safe"][1])
            d = c["default"]
            if lo <= d <= hi and lo < hi:
                c["curve"] = _relay(c["curve"], lo, hi, d if k.startswith("dose:") else None)
                c["safe"], c["ears"] = [lo, hi], e
        out[k] = c
    return out


def set_ears(control: str, lo: float, hi: float) -> Dict[str, Any]:
    cal = json.loads(FILE.read_text())
    c = cal["controls"].get(control)
    if c is None:
        raise ValueError(f"no control '{control}' in the calibration")
    if not (lo <= c["default"] <= hi):
        raise ValueError(f"the range has to hold the knob's default ({c['default']:g})")
    cal.setdefault("ears", {})[control] = {"lo": float(lo), "hi": float(hi), "set": time.strftime("%Y-%m-%d %H:%M")}
    FILE.write_text(json.dumps(cal, indent=1))
    return cal


def clear_ears(control: str) -> Dict[str, Any]:
    cal = json.loads(FILE.read_text())
    (cal.get("ears") or {}).pop(control, None)
    FILE.write_text(json.dumps(cal, indent=1))
    return cal
