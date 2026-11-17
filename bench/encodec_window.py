# Abliterator: can MusicGen's EnCodec (32 kHz, non-causal) be decoded in short windows?
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Generates 600 frames once with the frame loop, then decodes them in windows of
[s - ctx, s + hop + la) frames, keeps [s, s + hop) and crossfades `xf` samples into the
previous piece. Reports, for each setting, the error against the one-go decode, the size of
the jumps at the joins against the signal's own sample steps, the time per window on mps and
cpu, and the latency the window adds (hop + la frames).
"""
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "bench")
from musicgen_bench import MODEL, Stepper, encode_text  # noqa: E402
from transformers import AutoProcessor, MusicgenForConditionalGeneration  # noqa: E402

proc = AutoProcessor.from_pretrained(MODEL)
m = MusicgenForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager").to("mps").eval()
for mod in m.decoder.modules():
    if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
        mod.config._attn_implementation = "sdpa"
enc, mask = encode_text(m, proc, "slow ambient drone, warm analogue pads, no drums", 3.0, "mps", torch.bfloat16)
st = Stepper(m, enc, mask, 3.0, 7)
for _ in range(604):
    st.step()
codes = torch.stack(st.frames, 1)
torch.save(codes, "_tmp/bench/codes_600.pt")
SR, HOP = 32000, 640

res = {}
for devname in ("mps", "cpu"):
    ae = m.audio_encoder.to(devname).to(torch.float32)

    def dec(c):
        with torch.no_grad():
            a = ae.decode(c.view(1, 1, *c.shape).to(devname), [None])[0]
        if devname == "mps":
            torch.mps.synchronize()
        return a[0, 0].float().cpu().numpy()
    ref = dec(codes)
    dec(codes[:, :20])
    for hop in (2, 5, 10):
        for la in (0, 2, 5, 10):
            for ctx in (10, 25):
                xf = 128
                out = np.zeros(0, dtype=np.float32)
                tms = []
                start = 50
                for s in range(start, start + 300, hop):
                    lo, hi = s - ctx, s + hop + la
                    t0 = time.perf_counter()
                    w = dec(codes[:, lo:hi])
                    tms.append(time.perf_counter() - t0)
                    a0 = (s - lo) * HOP
                    piece = w[a0 - xf: a0 + hop * HOP]          # xf samples of overlap before s
                    if len(out) == 0:
                        out = piece[xf:].copy()
                    else:
                        r = np.linspace(0, 1, xf, dtype=np.float32)
                        out[-xf:] = out[-xf:] * (1 - r) + piece[:xf] * r
                        out = np.concatenate([out, piece[xf:]])
                y = out
                rf = ref[start * HOP: start * HOP + len(y)]
                err = np.sqrt(((y - rf) ** 2).mean()) / np.sqrt((rf ** 2).mean())
                d = np.abs(np.diff(y))
                joins = [i * hop * HOP for i in range(1, len(y) // (hop * HOP))]
                jr = float(np.mean([d[j - 1] for j in joins]) / d.mean())
                dr = np.abs(np.diff(rf))
                jref = float(np.mean([dr[j - 1] for j in joins]) / dr.mean())
                res[f"{devname} hop{hop} la{la} ctx{ctx}"] = {
                    "ms_per_window": round(float(np.median(tms)) * 1000, 1),
                    "ms_per_frame": round(float(np.median(tms)) * 1000 / hop, 2),
                    "err_db": round(float(20 * np.log10(err)), 1), "join_ratio": round(jr, 2), "join_ratio_ref": round(jref, 2),
                    "added_latency_ms": (hop + la) * 20}
                print(f"{devname} hop{hop} la{la} ctx{ctx}", res[f"{devname} hop{hop} la{la} ctx{ctx}"], flush=True)
json.dump(res, open("_tmp/bench/encodec_window.json", "w"), indent=1)
