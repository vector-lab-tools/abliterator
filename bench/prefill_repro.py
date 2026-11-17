# Abliterator: the cost of re-priming MusicGen's KV cache, and run-to-run reproducibility.
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
1. Prefill: one forward over C frames of context (the delay-patterned columns), for C in
   250, 500, 750, 1000, with guidance off and on: what a sliding-window re-prime costs.
2. Reproducibility: the same seed and prompt twice, 300 frames, in bfloat16 and float32 on MPS,
   and float32 on CPU: are the codes identical?
"""
import json
import sys
import time

import torch

sys.path.insert(0, "bench")
from musicgen_bench import MODEL, Stepper, encode_text  # noqa: E402
from transformers import AutoProcessor, MusicgenForConditionalGeneration  # noqa: E402

proc = AutoProcessor.from_pretrained(MODEL)
res = {"prefill": {}, "repro": {}}


def load(dev, dt):
    m = MusicgenForConditionalGeneration.from_pretrained(MODEL, dtype=dt, attn_implementation="eager").to(dev).eval()
    for mod in m.decoder.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
            mod.config._attn_implementation = "sdpa"
    return m


m = load("mps", torch.bfloat16)
for cfg in (1.0, 3.0):
    enc, mask = encode_text(m, proc, "slow ambient drone", cfg, "mps", torch.bfloat16)
    B = 2 if cfg > 1 else 1
    for C in (250, 500, 750, 1000):
        ids = torch.randint(0, 2048, (B * 4, C), device="mps")
        with torch.no_grad():
            m.decoder(input_ids=ids[:, :50], encoder_hidden_states=enc, encoder_attention_mask=mask, use_cache=True)
            torch.mps.synchronize()
            t0 = time.perf_counter()
            m.decoder(input_ids=ids, encoder_hidden_states=enc, encoder_attention_mask=mask, use_cache=True)
            torch.mps.synchronize()
        res["prefill"][f"cfg{cfg}_C{C}"] = round((time.perf_counter() - t0) * 1000, 1)
        print("prefill", cfg, C, res["prefill"][f"cfg{cfg}_C{C}"], "ms", flush=True)
del m

for dev, dt in (("mps", torch.bfloat16), ("mps", torch.float32), ("cpu", torch.float32)):
    m = load(dev, dt)
    runs = []
    for _ in range(2):
        enc, mask = encode_text(m, proc, "slow ambient drone", 3.0, dev, dt)
        st = Stepper(m, enc, mask, 3.0, 11)
        for _ in range(304):
            st.step()
        runs.append(torch.stack(st.frames, 1))
    same = bool(torch.equal(runs[0], runs[1]))
    first = None if same else int((runs[0] != runs[1]).any(0).nonzero()[0])
    res["repro"][f"{dev}_{str(dt).split('.')[-1]}"] = {"identical": same, "first_differing_frame": first}
    print("repro", dev, dt, same, first, flush=True)
    del m
json.dump(res, open("_tmp/bench/prefill_repro.json", "w"), indent=1)
