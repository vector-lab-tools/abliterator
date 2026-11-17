# Abliterator: the cost of the Steer hook in Magenta RT 2 small, measured with the hook on and off
# in alternating blocks of 25 frames within one run, so that warm-up and heat fall on both alike.
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""    .venv-mrt2/bin/python bench/mrt2_hookcost.py [--bits 8]"""
import json
import sys
import time

sys.path.insert(0, "backend")
import numpy as np  # noqa: E402

import abliterator.mrt2_proc as M  # noqa: E402

bits = int(sys.argv[sys.argv.index("--bits") + 1]) if "--bits" in sys.argv else 8
M.BITS = bits if bits < 32 else None
p = M.Proc()
p.load(["slow ambient drone, warm analogue pads, no drums"])
cond = p.conditioning("slow ambient drone, warm analogue pads, no drums", M.DEFAULTS)
st = p.initial_state(1)
v = p.mx.array(p.vectors["brightness"] * 0.3)
on, off = [], []
for _ in range(10):
    p.step(st, cond)
for blk in range(16):
    hooked = blk % 2 == 1
    p.hooks.vec = v if hooked else None
    for _ in range(25):
        t0 = time.perf_counter()
        _, st = p.step(st, cond)
        (on if hooked else off).append((time.perf_counter() - t0) * 1000)
r = {"bits": bits, "frames_each": len(on),
     "off_p50": round(float(np.median(off)), 2), "off_p95": round(float(np.percentile(off, 95)), 2),
     "on_p50": round(float(np.median(on)), 2), "on_p95": round(float(np.percentile(on, 95)), 2)}
print(json.dumps(r))
json.dump(r, open(f"_tmp/mrt2/hookcost_bits{bits}.json", "w"))
