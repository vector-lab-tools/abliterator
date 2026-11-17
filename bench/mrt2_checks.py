# Abliterator: offline checks of the Magenta RT 2 engine (no sound card).
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
1. Reproducibility: the same seed gives the same audio; another seed gives other audio.
2. The Steer hook acts: a dose changes the audio, from the frame it is first applied.
3. Codec delay: switch a dose on at frame K and find the first sample that differs from the run
   without it. If the sampler's declared output latency (0) is right, that sample is K * 1920.
Also loads the engine as the server does, which makes any missing direction.

    .venv-mrt2/bin/python bench/mrt2_checks.py
"""
import json
import sys
import time

sys.path.insert(0, "backend")
import numpy as np  # noqa: E402

import abliterator.mrt2_proc as M  # noqa: E402

P = "slow ambient drone, warm analogue pads, no drums"
p = M.Proc()
t0 = time.time()
p.load([P])
res = {"load_s": round(time.time() - t0, 1), "directions": p.summaries}
cond = p.conditioning(P, M.DEFAULTS)


def run(seed, frames, dose_at=None, name=None, dose=0.3):
    st = p.initial_state(seed)
    p.hooks.vec = None
    out = []
    for j in range(frames):
        if dose_at is not None and j == dose_at:
            p.hooks.vec = p.mx.array(p.vectors[name] * dose)
        a, st = p.step(st, cond)
        out.append(a)
    p.hooks.vec = None
    return np.concatenate(out)


F, K = 60, 30
a1, a2, b = run(1, F), run(1, F), run(2, F)
res["same_seed_identical"] = bool(np.array_equal(a1, a2))
res["same_seed_max_diff"] = float(np.abs(a1 - a2).max())
res["other_seed_differs"] = bool(not np.allclose(a1, b))
name = sorted(p.vectors)[0]
for dose in (0.3, 1.0):
    s = run(1, F, dose_at=K, name=name, dose=dose)
    d = np.abs(s - a1).max(axis=1)
    first = int(np.argmax(d > 1e-6)) if (d > 1e-6).any() else None
    res[f"steer_{name}_{dose}"] = {"changes_audio": first is not None, "first_diff_sample": first,
                                   "expected_sample": K * M.HOP,
                                   "codec_delay_ms": None if first is None else round((first - K * M.HOP) / M.SR * 1000, 2),
                                   "rms_diff_after": float(np.sqrt(((s - a1)[K * M.HOP:] ** 2).mean()))}
print(json.dumps(res, indent=1))
json.dump(res, open("_tmp/mrt2/checks.json", "w"), indent=1)
