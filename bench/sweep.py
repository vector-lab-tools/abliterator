# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Sweeps each control across its range on several presets and measures the model's output at each
step (measure.py), to find where the control stays musical. Writes one JSON line per render to
_tmp/sweep/results.jsonl as it goes (a rerun skips what is there), and a 3-second clip of each
step to data/calibration/clips/ for listening. A sampler that is changed at all takes a different
path from then on, so each preset is also rendered at the defaults on six more seeds (`floor`):
a step's change counts only beyond that seed-to-seed spread.

    .venv-mrt2/bin/python bench/sweep.py [--frames 175] [--presets warm-drone,shimmer,...]

The live engine shares the GPU while this runs, so live play will drop out meanwhile.
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

OUT = ROOT / "_tmp" / "sweep"
CLIPS = ROOT / "data" / "calibration" / "clips"
DOSES = [-1.0, -0.6, -0.4, -0.3, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0]
CONTROLS = {
    "guidance": [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0],
    "temperature": [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 2.0],
    "top_k": [1, 5, 10, 20, 40, 80, 160, 250],
}
FLOOR = 6             # seeds at the defaults, per preset
SKIP_S = 2.0          # the opening, before the model has settled, is not measured
CLIP_FROM, CLIP_S = 3.0, 3.0


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--frames", type=int, default=175)
    a.add_argument("--presets", default="warm-drone,shimmer,dark-field,glass")
    args = a.parse_args()
    from abliterator import mrt2_proc as M
    p = M.Proc()
    with contextlib.redirect_stdout(sys.stderr):
        p.load([x["prompt"] for x in pr.presets()])
    OUT.mkdir(parents=True, exist_ok=True)
    res = OUT / "results.jsonl"
    done = set()
    if res.exists():
        for line in res.read_text().splitlines():
            r = json.loads(line)
            done.add((r["preset"], r["seed"], r["control"], r["value"]))
    jobs = []
    for pid in args.presets.split(","):
        pre = pr.find(pid)
        seed = pre["seeds"][0]
        for d in sorted(p.vectors):
            jobs += [(pre, seed, f"dose:{d}", v) for v in DOSES]
        for c, vals in CONTROLS.items():
            jobs += [(pre, seed, c, v) for v in vals]
        # the noise floor: the defaults on other seeds, for how much the music varies by itself
        jobs += [(pre, seed + 100 * k, "floor", k) for k in range(1, FLOOR + 1)]
    todo = [j for j in jobs if (j[0]["id"], j[1], j[2], j[3]) not in done]
    print(f"{len(jobs)} renders, {len(todo)} to do", file=sys.stderr)
    t0 = time.time()
    for i, (pre, seed, ctl, v) in enumerate(todo):
        sampling = {**M.DEFAULTS, **(pre.get("mrt2") or {})}
        doses = {k: 0.0 for k in p.vectors}
        if ctl.startswith("dose:"):
            doses[ctl[5:]] = v
        elif ctl != "floor":
            sampling[ctl] = v
        out = p.render({"prompt": pre["prompt"], "seed": seed, "start": {"doses": doses, "sampling": sampling}}, args.frames)["pre"]
        m = ms.measure(out[int(SKIP_S * M.SR):])
        row = {"preset": pre["id"], "seed": seed, "control": ctl, "value": v, **m}
        with res.open("a") as f:
            f.write(json.dumps(row) + "\n")
        clip = CLIPS / ctl.replace(":", "-") / f"{pre['id']}-s{seed}-{v:g}.flac"
        clip.parent.mkdir(parents=True, exist_ok=True)
        sf.write(clip, out[int(CLIP_FROM * M.SR):int((CLIP_FROM + CLIP_S) * M.SR)], M.SR, subtype="PCM_16")
        el = time.time() - t0
        print(f"{i + 1}/{len(todo)} {pre['id']} {ctl}={v}: {m['rms_db']} dB, clip {m['clip']}, flat {m['flatness']} "
              f"({el / (i + 1):.1f} s each, {el / (i + 1) * (len(todo) - i - 1) / 60:.0f} min left)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
