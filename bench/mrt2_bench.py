# Abliterator: benchmark for Magenta RealTime 2 small, Python MLX path, with and without a Steer hook.
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Runs in its own environment (.venv-mrt2, magenta-rt 2.0.3 with MLX). For bits 8 (the vendor's default)
and None (bfloat16), measures the step time of MagentaRT2System's own frame loop (one 40 ms frame of
48 kHz stereo audio per step, SpectroStream decoding included), unhooked and with a Steer on one block
of the temporal transformer: h' = h + dose · ‖h‖ · v, v a fixed random unit vector, the dose read at
every step. Writes four 20 s ambient clips (unhooked, bits 8) for listening.

    .venv-mrt2/bin/python bench/mrt2_bench.py
"""
import json
import os
import resource
import sys
import time

import mlx.core as mx
import numpy as np
import soundfile as sf

import magenta_rt.mlx.system as S
import sequence_layers.mlx as sl
from magenta_rt.config import MUSICCOCA

OUT = "_tmp/mrt2"
os.makedirs(OUT, exist_ok=True)
res = {}


def blocks(mrt):
    tb = mrt._sampler.depthformer.decoder.temporal_body.layers[0]
    return tb.layers


def hook(block, state):
    """Wrap one block's step so that its output stream is steered."""
    # The temporal transformer's Serial calls step_with_emits; wrapping `step` hooks nothing.
    orig = block.step_with_emits

    def step(x, st, *a, **k):
        y, st2, e = orig(x, st, *a, **k)
        d, v = state["dose"], state["v"]
        if d:
            y = y.apply_values(lambda h: h + d * mx.linalg.norm(h.astype(mx.float32), axis=-1, keepdims=True).astype(h.dtype) * v.astype(h.dtype))
        return y, st2, e
    block.step_with_emits = step
    return orig


def run(mrt, emb, frames, state=None):
    t = []
    tok = mrt._style_model.tokenize(emb).tolist()
    cfg = next(c for c in mrt._model.input_configs if c.key == MUSICCOCA.key)
    tok = (tok + [-1] * cfg.rvq_truncation_level)[: cfg.rvq_truncation_level]
    blk, constants = mrt._build_conditioning({MUSICCOCA.key: tok})
    if state is None:
        state = mrt._sampler.get_initial_state(1, sl.ChannelSpec(shape=(mrt._num_channels,), dtype=mx.int32),
                                               constants={}, training=False)
    outs = []
    for _ in range(frames):
        t0 = time.perf_counter()
        y, state, _ = mrt._sampler.step_with_emits(x=blk, state=state, constants=constants, training=False)
        mx.eval(y.values)
        t.append((time.perf_counter() - t0) * 1000)
        outs.append(np.array(y.values[0]))
    return np.array(t), np.concatenate(outs, 0), state


for bits in (8, None):
    mrt = S.MagentaRT2System(size="mrt2_small", bits=bits)
    emb = mrt.embed_style("slow ambient drone, warm analogue pads", use_mapper=True)
    bl = blocks(mrt)
    st = {"dose": 0.0, "v": None}
    run(mrt, emb, 3)
    t_un, _, _ = run(mrt, emb, 250)
    v = mx.random.normal((1024,), key=mx.random.key(0))
    st["v"] = v / mx.linalg.norm(v)
    orig = hook(bl[6], st)
    st["dose"] = 0.3
    try:
        t_h, audio_h, _ = run(mrt, emb, 250)
    except Exception as e:
        t_h, audio_h = None, None
        res[f"bits{bits}_hook_error"] = f"{type(e).__name__}: {e}"
    bl[6].step_with_emits = orig
    key = f"bits{bits}"
    res[key] = {"unhooked_ms_p50": round(float(np.median(t_un)), 2), "unhooked_ms_p95": round(float(np.percentile(t_un, 95)), 2),
                "budget_ms": 40.0, "rtf_unhooked": round(40.0 / float(np.mean(t_un)), 2)}
    if t_h is not None:
        res[key].update({"hooked_ms_p50": round(float(np.median(t_h)), 2), "hooked_ms_p95": round(float(np.percentile(t_h, 95)), 2),
                         "rtf_hooked": round(40.0 / float(np.mean(t_h)), 2)})
        sf.write(f"{OUT}/mrt2_small_bits{bits}_steered_random.wav", audio_h / 32768.0 if np.abs(audio_h).max() > 2 else audio_h, 48000)
    res[key]["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)
    res[key]["mlx_peak_gb"] = round(mx.get_peak_memory() / 2**30, 2) if hasattr(mx, "get_peak_memory") else None
    print(key, res[key], res.get(f"bits{bits}_hook_error", ""), flush=True)
    if bits == 8 and "--clips" in sys.argv:
        prompts = ["slow ambient drone, warm analogue pads, no drums",
                   "evolving ambient pad texture, shimmering, reverb, calm, no percussion",
                   "deep low drone, dark atmospheric soundscape, field recording textures",
                   "soft glassy ambient chords, slow attack, no beat"]
        for i, p in enumerate(prompts, 1):
            e = mrt.embed_style(p, use_mapper=True)
            _, a, _ = run(mrt, e, 500)
            a = a / 32768.0 if np.abs(a).max() > 2 else a
            sf.write(f"{OUT}/mrt2_small_{i}.wav", a, 48000)
            print("clip", i, p, round(float(20 * np.log10(np.sqrt((a ** 2).mean()) + 1e-12)), 1), "dBFS", flush=True)
    del mrt
json.dump(res, open(f"{OUT}/mrt2_bench.json", "w"), indent=1)
