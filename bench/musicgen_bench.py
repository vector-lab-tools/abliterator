# Abliterator: benchmark for MusicGen small, generated frame by frame.
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Measures, for facebook/musicgen-small on this machine:
- load time and memory;
- the time of one decoder step (one 20 ms frame, all four codebooks) with the KV cache,
  with classifier-free guidance on (batch 2) and off (batch 1), in float32 and float16;
- the real-time factor (audio seconds per wall second) of token generation alone;
- the time to decode EnCodec in one go and in short windows, and how far windowed decoding
  departs from the one-go decode at each amount of left context.

The loop is written out rather than calling model.generate, so that a hook could change the
stream between any two frames. Sampling is on the CPU from a seeded torch.Generator.

    .venv/bin/python bench/musicgen_bench.py --dtype float32 --cfg 3 --frames 500 --prompt "..." --out _tmp/x.wav
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import psutil
import soundfile as sf
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration

MODEL = "facebook/musicgen-small"


def mem():
    out = {"rss_gb": psutil.Process().memory_info().rss / 2**30}
    if torch.backends.mps.is_available():
        out["mps_driver_gb"] = torch.mps.driver_allocated_memory() / 2**30
        out["mps_current_gb"] = torch.mps.current_allocated_memory() / 2**30
    return out


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


def sample(logits: torch.Tensor, gen: torch.Generator, temperature: float, top_k: int) -> torch.Tensor:
    """logits (K, V) float64 on CPU; one token per codebook."""
    l = logits / max(temperature, 1e-6)
    v, i = torch.topk(l, top_k, dim=-1)
    p = torch.softmax(v, -1)
    pick = torch.multinomial(p, 1, generator=gen)
    return i.gather(-1, pick).squeeze(-1)


class Stepper:
    """One frame per call. Holds the KV cache, the delay pattern and the codes so far."""

    def __init__(self, model, enc, enc_mask, cfg: float, seed: int, temperature=1.0, top_k=250):
        self.m, self.enc, self.mask, self.cfg = model, enc, enc_mask, cfg
        self.K = model.decoder.num_codebooks
        self.pad = model.generation_config.pad_token_id
        self.dev = enc.device
        self.gen = torch.Generator().manual_seed(seed)
        self.temperature, self.top_k = temperature, top_k
        self.past = None
        self.cols = [torch.full((self.K,), self.pad, dtype=torch.long)]   # column 0: all pad (start)
        self.frames = []                                                   # complete frames, (K,) each

    def step(self):
        c = len(self.cols) - 1
        cur = self.cols[-1].to(self.dev).view(self.K, 1)
        if self.cfg > 1:
            cur = cur.repeat(2, 1)
        with torch.no_grad():
            out = self.m.decoder(input_ids=cur, encoder_hidden_states=self.enc, encoder_attention_mask=self.mask,
                                 past_key_values=self.past, use_cache=True)
        self.past = out.past_key_values
        lg = out.logits[:, -1].float().cpu().double()                    # (B*K, V)
        if self.cfg > 1:
            cond, unc = lg[: self.K], lg[self.K:]
            lg = unc + self.cfg * (cond - unc)
        tok = sample(lg, self.gen, self.temperature, self.top_k)
        nc = c + 1                                                 # the column just predicted
        for k in range(self.K):                                    # delay pattern: codebook k starts at column k+1
            if nc <= k:
                tok[k] = self.pad
        self.cols.append(tok)
        f = nc - self.K                                            # frame f is complete once column f+K exists
        if f >= 0:
            self.frames.append(torch.stack([self.cols[f + 1 + k][k] for k in range(self.K)]))
            return self.frames[-1]
        return None


def encode_text(model, proc, prompt, cfg, dev, dtype):
    inp = proc(text=[prompt], padding=True, return_tensors="pt").to(dev)
    with torch.no_grad():
        h = model.text_encoder(**inp).last_hidden_state
        mask = inp["attention_mask"]
        if cfg > 1:
            h = torch.cat([h, torch.zeros_like(h)], 0)
            mask = torch.cat([mask, torch.zeros_like(mask)], 0)
        h = model.enc_to_dec_proj(h).to(dtype)
    return h, mask


def decode(model, codes: torch.Tensor) -> np.ndarray:
    """codes (K, T) → mono float32 audio."""
    with torch.no_grad():
        a = model.audio_encoder.decode(codes.view(1, 1, *codes.shape).to(model.audio_encoder.device), [None])[0]
    return a[0, 0].float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--prompt", default="slow ambient drone, warm analogue pads, no drums")
    ap.add_argument("--out", default="")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--decode-test", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--attn", default="sdpa")
    a = ap.parse_args()
    dev, dtype = a.device, getattr(torch, a.dtype)
    res = {"dtype": a.dtype, "cfg": a.cfg, "frames": a.frames, "device": dev, "prompt": a.prompt, "seed": a.seed}

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(MODEL)
    model = MusicgenForConditionalGeneration.from_pretrained(MODEL, dtype=dtype, attn_implementation="eager")
    model.to(dev).eval()
    for mod in model.decoder.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
            mod.config._attn_implementation = a.attn
    res["attn"] = a.attn
    # EnCodec stays in float32: it is small and its convolutions are sensitive to precision.
    model.audio_encoder.to(torch.float32)
    sync(dev)
    res["load_s"] = round(time.time() - t0, 2)
    res["mem_after_load"] = mem()
    cfgd = model.audio_encoder.config
    res["encodec"] = {"sampling_rate": cfgd.sampling_rate, "channels": cfgd.audio_channels,
                      "causal": cfgd.use_causal_conv, "frame_rate": cfgd.frame_rate,
                      "codebooks": model.decoder.num_codebooks}
    dc = model.decoder.config
    res["decoder"] = {"layers": dc.num_hidden_layers, "hidden": dc.hidden_size, "heads": dc.num_attention_heads,
                      "params_m": round(sum(p.numel() for p in model.decoder.parameters()) / 1e6, 1)}

    t0 = time.time()
    enc, mask = encode_text(model, proc, a.prompt, a.cfg, dev, dtype)
    sync(dev)
    res["text_encode_ms"] = round((time.time() - t0) * 1000, 1)

    st = Stepper(model, enc, mask, a.cfg, a.seed)
    times = []
    first = None
    t_start = time.time()
    for i in range(a.frames + st.K):
        t1 = time.perf_counter()
        f = st.step()
        sync(dev)
        times.append(time.perf_counter() - t1)
        if f is not None and first is None:
            first = time.time() - t_start
    gen_s = time.time() - t_start
    ts = np.array(times[5:]) * 1000
    fr = cfgd.frame_rate
    res["step_ms"] = {"mean": round(float(ts.mean()), 2), "p50": round(float(np.percentile(ts, 50)), 2),
                      "p95": round(float(np.percentile(ts, 95)), 2), "p99": round(float(np.percentile(ts, 99)), 2),
                      "max": round(float(ts.max()), 2), "first5": [round(x * 1000, 1) for x in times[:5]]}
    res["frame_ms_budget"] = round(1000 / fr, 2)
    res["rtf_tokens"] = round((len(st.frames) / fr) / gen_s, 3)
    res["first_frame_s"] = round(first, 3)
    # how the step time grows with the length of the cache
    q = len(ts) // 4
    res["step_ms_by_quarter"] = [round(float(ts[i * q:(i + 1) * q].mean()), 2) for i in range(4)]
    res["mem_after_gen"] = mem()

    codes = torch.stack(st.frames, 1)          # (K, T)
    t0 = time.time()
    audio = decode(model, codes)
    sync(dev)
    res["decode_full_ms"] = round((time.time() - t0) * 1000, 1)
    res["audio_s"] = round(len(audio) / cfgd.sampling_rate, 2)
    res["rtf_with_full_decode"] = round(res["audio_s"] / (gen_s + res["decode_full_ms"] / 1000), 3)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        sf.write(a.out, audio, cfgd.sampling_rate)
        res["wav"] = a.out
    res["rms_dbfs"] = round(float(20 * np.log10(np.sqrt((audio ** 2).mean()) + 1e-12)), 1)

    if a.decode_test:
        # Windowed decoding: decode frames [s - ctx, s + hop) and keep the last hop frames' samples.
        hop_samples = cfgd.sampling_rate // fr
        T = codes.shape[1]
        out = {}
        for hop in (1, 2, 5, 10):
            for ctx in (0, 5, 10, 25, 50):
                pieces, tms = [], []
                for s in range(50, T - hop, hop):
                    lo = max(0, s - ctx)
                    t1 = time.perf_counter()
                    w = decode(model, codes[:, lo: s + hop])
                    sync(dev)
                    tms.append(time.perf_counter() - t1)
                    pieces.append(w[(s - lo) * hop_samples:(s - lo + hop) * hop_samples])
                    if len(pieces) * hop >= 150:
                        break
                y = np.concatenate(pieces)
                ref = audio[50 * hop_samples: 50 * hop_samples + len(y)]
                err = float(np.sqrt(((y - ref) ** 2).mean()) / (np.sqrt((ref ** 2).mean()) + 1e-12))
                # a click shows as a jump at a window boundary larger than the signal's own steps
                b = [i * hop * hop_samples for i in range(1, len(pieces))]
                jumps = np.abs(np.diff(y))
                ratio = float(np.mean([jumps[i - 1] for i in b]) / (jumps.mean() + 1e-12))
                out[f"hop{hop}_ctx{ctx}"] = {"ms_per_window": round(float(np.mean(tms)) * 1000, 2),
                                             "ms_per_frame": round(float(np.mean(tms)) * 1000 / hop, 2),
                                             "rel_err_db": round(20 * np.log10(err + 1e-12), 1),
                                             "boundary_jump_ratio": round(ratio, 2)}
        res["windowed_decode"] = out
    print(json.dumps(res, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
