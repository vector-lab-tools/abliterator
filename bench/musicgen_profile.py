# Abliterator: where one MusicGen step's time goes.
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""Times the decoder forward alone, the sampling, and the device, for eager and sdpa attention,
on mps and cpu. Usage: .venv/bin/python bench/musicgen_profile.py --device mps --attn sdpa --dtype float16"""
import argparse
import json
import time

import numpy as np
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="mps")
ap.add_argument("--attn", default="sdpa")
ap.add_argument("--dtype", default="float16")
ap.add_argument("--cfg", type=float, default=1.0)
ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--threads", type=int, default=0)
a = ap.parse_args()
if a.threads:
    torch.set_num_threads(a.threads)
dev, dt = a.device, getattr(torch, a.dtype)
sync = (lambda: torch.mps.synchronize()) if dev == "mps" else (lambda: None)
proc = AutoProcessor.from_pretrained("facebook/musicgen-small")
m = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", dtype=dt, attn_implementation="eager").to(dev).eval()
m.decoder.config._attn_implementation = a.attn
for mod in m.decoder.modules():
    if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
        mod.config._attn_implementation = a.attn
inp = proc(text=["slow ambient drone, warm pads"], return_tensors="pt").to(dev)
with torch.no_grad():
    h = m.text_encoder(**inp).last_hidden_state
    mask = inp["attention_mask"]
    if a.cfg > 1:
        h, mask = torch.cat([h, torch.zeros_like(h)]), torch.cat([mask, torch.zeros_like(mask)])
    enc = m.enc_to_dec_proj(h).to(dt)
B = 2 if a.cfg > 1 else 1
K = 4
cur = torch.full((B * K, 1), 2048, dtype=torch.long, device=dev)
past = None
fwd, smp = [], []
g = torch.Generator().manual_seed(0)
with torch.no_grad():
    for i in range(a.steps):
        t0 = time.perf_counter()
        out = m.decoder(input_ids=cur, encoder_hidden_states=enc, encoder_attention_mask=mask, past_key_values=past, use_cache=True)
        past = out.past_key_values
        lg = out.logits[:, -1]
        sync()
        t1 = time.perf_counter()
        l = lg.float().cpu()
        v, ix = torch.topk(l, 250, -1)
        tok = ix.gather(-1, torch.multinomial(torch.softmax(v, -1), 1, generator=g))
        cur = tok.to(dev)
        t2 = time.perf_counter()
        fwd.append(t1 - t0)
        smp.append(t2 - t1)
f, s = np.array(fwd[10:]) * 1000, np.array(smp[10:]) * 1000
print(json.dumps({"device": dev, "attn": a.attn, "dtype": a.dtype, "cfg": a.cfg, "threads": torch.get_num_threads(),
                  "forward_ms_p50": round(float(np.median(f)), 2), "forward_ms_mean": round(float(f.mean()), 2),
                  "sample_ms_p50": round(float(np.median(s)), 2), "total_p50": round(float(np.median(f + s)), 2)}))
