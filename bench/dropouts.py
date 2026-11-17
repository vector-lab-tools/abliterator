# Abliterator: where a frame's time goes with a chain of units, and which steps overrun the 40 ms frame.
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Steps Magenta RT 2 small in this process (as live play does, without the sound card), for each variant:
no units; the whole chain; each line alone; the whole chain with the strips' readings (every READ_EVERY
frames, as live); the whole chain with Python's garbage collector off. Reports median, p95, p99, max and
how many steps went over 40 ms. Writes _tmp/dropouts.json.

    .venv-mrt2/bin/python bench/dropouts.py 'line 1' 'line 2' ...
"""
import gc
import json
import sys
import time

sys.path.insert(0, "backend")
import numpy as np  # noqa: E402

from abliterator import mrt2_proc as M  # noqa: E402

P = "slow ambient drone, warm analogue pads, no drums"
N = int(__import__("os").environ.get("N", "250"))
p = M.Proc()
p.load([P])
cond = p.conditioning(P, M.DEFAULTS)
EMPTY = {"edits": {}, "outs": {}, "ins": {}}
lines = sys.argv[1:]


def run(bundle, reading=False, nogc=False, n=N):
    p._use(bundle)
    st = p.initial_state(1)
    for _ in range(15):                              # warm the graph for this bundle
        _, st = p.step(st, cond)
    ms = []
    if nogc:
        gc.disable()
    try:
        for f in range(n):
            t0 = time.perf_counter()
            r = reading and f % M.READ_EVERY == 0
            p.hooks.meas, p.hooks.probes, p.hooks.measuring = [], {}, r
            p.hooks.frame = f
            _, st = p.step(st, cond)
            p.hooks.measuring = False
            if r:
                p._read_activity()
            ms.append((time.perf_counter() - t0) * 1000)
    finally:
        gc.enable()
        p._use(EMPTY)
    a = np.array(ms)
    return {"p50": round(float(np.median(a)), 2), "p95": round(float(np.percentile(a, 95)), 2), "p99": round(float(np.percentile(a, 99)), 2),
            "max": round(float(a.max()), 2), "over40": int((a > 40).sum()), "over35": int((a > 35).sum()), "n": n}


res = {}
def note(k, v):
    res[k] = v
    print(k, v, flush=True)
    json.dump(res, open("_tmp/dropouts.json", "w"), indent=1)


note("none", run(EMPTY))
whole, rep = p.attach_chain(lines)
note("errors", [r.get("error") for r in rep if r.get("error")])
note("whole", run(whole))
note("whole+readings", run(whole, reading=True))
note("whole+readings, no gc", run(whole, reading=True, nogc=True))
for ln in lines:
    b, _ = p.attach_chain([ln])
    note(ln.split()[0] + " " + " ".join(x for x in ln.split()[1:] if x.startswith(("layer=", "block="))), run(b, n=120))
