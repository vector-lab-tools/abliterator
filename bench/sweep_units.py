# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Sweeps each model-layer unit's numeric settings, and its placement, on a few presets, with the
other settings at this host's defaults, and measures the output (measure.py), as bench/sweep.py
does for the controls. Each preset is also rendered with no unit (`unit:none`), the reference a
unit's setting is judged against. One JSON line per render to _tmp/sweep/units.jsonl (a rerun
skips what is there); a 3-second clip of each to data/calibration/clips/.

    .venv-mrt2/bin/python bench/sweep_units.py [--frames 150] [--presets warm-drone,shimmer] [--steps 6]

A second pass (--units delay,reverb --pass2) re-measures units whose own defaults broke the sound:
the base becomes VAU's defaults with the mix and the placement taken from the first pass's
calibration (the measured safe mix, the nearest placement that held), and every setting and
placement is swept again from there, into _tmp/sweep/units2.jsonl.

Controls are named `unit:<id>:<setting>`; placement is `unit:<id>:at` (the layer, 1 to 12, for a
stream unit; the block for Abliterate and Neuron, with −1 for all blocks). The live engine shares
the GPU while this runs.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time

sys.path.insert(0, "backend")
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from abliterator import ROOT  # noqa: E402
from abliterator import measure as ms  # noqa: E402
from abliterator import presets as pr  # noqa: E402

OUT = ROOT / "_tmp" / "sweep" / "units.jsonl"
CLIPS = ROOT / "data" / "calibration" / "clips"
SKIP_S, CLIP_FROM, CLIP_S = 2.0, 2.5, 3.0
LAYERS = [2, 4, 6, 8, 10, 12]
BLOCKS = [-1, 0, 3, 6, 9, 11]


def steps(q, n):
    lo, hi, d = float(q["min"]), float(q["max"]), q.get("default")
    vs = np.linspace(lo, hi, n)
    if q["kind"] == "int":
        vs = np.unique(np.round(vs).astype(int)).astype(float)
    out = sorted(set([round(float(v), 4) for v in vs] + ([round(float(d), 4)] if d is not None else [])))
    return out


def line_of(u, settings, at):
    place = (f"block={'all' if at == -1 else int(at)}" if u["site"] == "output" else f"layer={int(at)}")
    kv = [f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in settings.items() if v not in (None, "")]
    return " ".join([u["id"], place] + kv)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--frames", type=int, default=150)
    a.add_argument("--presets", default="warm-drone,shimmer")
    a.add_argument("--steps", type=int, default=6)
    a.add_argument("--units", default="")
    a.add_argument("--pass2", action="store_true")
    args = a.parse_args()
    global OUT
    if args.pass2:
        OUT = ROOT / "_tmp" / "sweep" / "units2.jsonl"
    cal = json.loads((ROOT / "data" / "calibration" / "mrt2_small.json").read_text())["controls"] if args.pass2 else {}
    from abliterator import mrt2_proc as M
    from abliterator import units as U
    p = M.Proc()
    with contextlib.redirect_stdout(sys.stderr):
        p.load([x["prompt"] for x in pr.presets()])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        for l in OUT.read_text().splitlines():
            r = json.loads(l)
            done.add((r["preset"], r["control"], r["value"]))
    jobs = []
    base_of, at0_of = {}, {}
    cat = [u for u in U.catalogue() if u["available"] and (not args.units or u["id"] in args.units.split(","))]
    vau_defaults = {u.id: {q["key"]: U.HOST_DEFAULTS.get(u.id, {}).get(q["key"], q.get("default")) for q in u.m["params"]} for u in U.registry()}
    for pid in args.presets.split(","):
        pre = pr.find(pid)
        jobs.append((pre, None, "unit:none", 0.0, None, None))
        for u in cat:
            base = {q["key"]: q.get("default") for q in u["params"]}
            at0 = -1 if u["site"] == "output" else 6
            if args.pass2:                                   # VAU's defaults, with the measured mix and placement
                base = dict(vau_defaults[u["id"]])
                c = cal.get(f"unit:{u['id']}:mix")
                if c and c.get("safe"):
                    base["mix"] = c["default"]
                c = cal.get(f"unit:{u['id']}:at")
                if c and c.get("safe_places"):
                    at0 = c["default"]
            for q in u["params"]:
                if q["kind"] not in ("knob", "fader", "int"):
                    continue
                for v in steps({**q, "default": base.get(q["key"], q.get("default"))}, args.steps):      # always the base's own value
                    jobs.append((pre, u, f"unit:{u['id']}:{q['key']}", v, {**base, q["key"]: v}, at0))
            for at in (BLOCKS if u["site"] == "output" else LAYERS):
                jobs.append((pre, u, f"unit:{u['id']}:at", float(at), base, at))
            base_of[u["id"]], at0_of[u["id"]] = base, at0
    todo = [j for j in jobs if (j[0]["id"], j[2], j[3]) not in done]
    print(f"{len(jobs)} renders, {len(todo)} to do", file=sys.stderr)
    t0 = time.time()
    for i, (pre, u, ctl, v, settings, at) in enumerate(todo):
        seed = pre["seeds"][0]
        chain = [line_of(u, settings, at)] if u else []
        default = None
        if u:
            key = ctl.split(":")[2]
            default = float(at0_of[u["id"]]) if key == "at" else float(base_of[u["id"]][key])
        try:
            out = p.render({"prompt": pre["prompt"], "seed": seed, "start": {"chain": chain}}, args.frames)["pre"]
            m = ms.measure(out[int(SKIP_S * M.SR):])
            err = None
        except Exception as e:
            out, m, err = None, {}, f"{type(e).__name__}: {e}"
        row = {"preset": pre["id"], "seed": seed, "control": ctl, "value": v, "default": default, "line": chain[0] if chain else "", "error": err, **m}
        with OUT.open("a") as f:
            f.write(json.dumps(row) + "\n")
        if out is not None:
            clip = CLIPS / ctl.replace(":", "-") / f"{pre['id']}-s{seed}-{v:g}.flac"
            clip.parent.mkdir(parents=True, exist_ok=True)
            sf.write(clip, np.clip(out[int(CLIP_FROM * M.SR):int((CLIP_FROM + CLIP_S) * M.SR)], -1, 1), M.SR, subtype="PCM_16")
        el = time.time() - t0
        print(f"{i + 1}/{len(todo)} {pre['id']} {ctl}={v:g}: {m.get('rms_db')} dB {err or ''} ({el / (i + 1):.1f} s each, "
              f"{el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min left)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
