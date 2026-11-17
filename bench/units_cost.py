# Abliterator: each VAU unit on Magenta RT 2 small, through the torch bridge.
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
For every unit this host can run, at layer 6 with its manifest's example settings:
  - step time with the strip and without, in alternating blocks of 20 frames in one run;
  - whether the unit at its neutral values leaves the audio exactly as without it (VAU section 3);
  - how much it changes the audio at the example settings (RMS of the difference, seed 1, 6 s).
Also the bridge alone (an edit that returns its input). Writes _tmp/units/cost.json.

    .venv-mrt2/bin/python bench/units_cost.py
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, "backend")
import numpy as np  # noqa: E402

from abliterator import mrt2_proc as M  # noqa: E402
from abliterator import units as U  # noqa: E402

P = "slow ambient drone, warm analogue pads, no drums"
os.makedirs("_tmp/units", exist_ok=True)
p = M.Proc()
p.load([P])
cond = p.conditioning(P, M.DEFAULTS)


def at6(line):
    line = re.sub(r"\blayer=\S+", "layer=6", line)
    line = re.sub(r"\bdir=\S+", "dir=brightness", line)
    return re.sub(r"\bdir2=\S+", "dir2=pulse", line)


EMPTY = {"edits": {}, "outs": {}, "ins": {}}


def timing(bundle, n=8, k=20):
    st = p.initial_state(1)
    on, off = [], []
    for b in range(n):
        p._use(bundle if b % 2 else EMPTY)
        for _ in range(k):
            t0 = time.perf_counter()
            _, st = p.step(st, cond)
            (on if b % 2 else off).append((time.perf_counter() - t0) * 1000)
    p._use(EMPTY)
    return round(float(np.median(on)), 2), round(float(np.percentile(on, 95)), 2), round(float(np.median(off)), 2)


F = 150
clean = p.render({"prompt": P, "seed": 1}, F)["pre"]
rms = lambda x: float(np.sqrt((x ** 2).mean()))  # noqa: E731
res = {"bridge_only": timing({"edits": {6: [lambda h, pos: h]}, "outs": {}, "ins": {}}),
       "bridge_output_only": timing({"edits": {}, "outs": {(5, "attn"): [lambda m, a, o: o]}, "ins": {}})}
res["bridge_only_identical"] = bool(np.array_equal(clean, p.render({"prompt": P, "seed": 1, "start": {}}, F)["pre"]))
p._use(EMPTY)
only = sys.argv[1:]
for u in U.catalogue():
    if only and u["id"] not in only:
        continue
    if not u["available"]:
        res[u["id"]] = {"available": False, "missing": u["missing"]}
        continue
    line = at6(u["example"]) if u.get("example") else f"{u['id']} layer=6"
    unit = U.registry().get(u["id"])
    neutral = " ".join(f"{k}={v}" for k, v in unit.neutral_params().items() if k not in ("layer",))
    target = "layer=6" if "layer" in u.get("example", "layer") or u["site"] == "stream" else "block=all"
    if u["site"] == "output":
        line = line.replace("block=all", "block=all")
    r = {"line": line}
    try:
        edits, _ = p.attach_chain([line])
        r["on_p50"], r["on_p95"], r["off_p50"] = timing(edits)
        a = p.render({"prompt": P, "seed": 1, "start": {"chain": [line]}}, F)["pre"]
        r["rms_change_db_vs_clean"] = round(20 * np.log10(rms(a - clean) / rms(clean) + 1e-12), 1)
        r["level_change_db"] = round(20 * np.log10(rms(a) / rms(clean)), 1)
        tgt = re.search(r"\b(layer|block)=\S+", line).group(0)
        rest = " ".join(x for x in line.split()[1:] if not re.match(r"(layer|block)=", x) and x.split("=")[0] not in unit.neutral_params())
        nline = f"{u['id']} {tgt} {rest} {neutral}".strip()
        b = p.render({"prompt": P, "seed": 1, "start": {"chain": [nline]}}, F)["pre"]
        r["neutral_line"] = nline
        r["neutral_identical"] = bool(np.array_equal(b, clean))
    except Exception as e:
        r["error"] = f"{type(e).__name__}: {e}"
    res[u["id"]] = r
    print(u["id"], r, file=sys.stderr, flush=True)
json.dump(res, open("_tmp/units/cost" + ("-" + "-".join(only) if only else "") + ".json", "w"), indent=1)
print(json.dumps(res, indent=1), file=sys.stderr)
