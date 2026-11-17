# Abliterator: live control-to-sound latency for Magenta RT 2 small (Python MLX path), as live_proto.py does for MusicGen.
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
A generation thread runs the sampler's step (one 40 ms frame of 48 kHz stereo, decoded in the step;
the sampler declares output_latency 0) with a Steer on one temporal block whose dose is read at every
step, holding at most `lead` frames beyond what the sound card has played. Every `--move` seconds the
dose jumps; the latency is from the move to the DAC time of the first sample of the first frame
computed with the new dose.

    .venv-mrt2/bin/python bench/mrt2_live.py --lead 3 --seconds 30
"""
import argparse
import collections
import json
import sys
import threading
import time

import mlx.core as mx
import numpy as np
import sounddevice as sd

sys.path.insert(0, "bench")
import magenta_rt.mlx.system as S  # noqa: E402
import sequence_layers.mlx as sl  # noqa: E402
from magenta_rt.config import MUSICCOCA  # noqa: E402
from mrt2_bench import hook  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--lead", type=int, default=3)
ap.add_argument("--seconds", type=float, default=30)
ap.add_argument("--block", type=int, default=512)
ap.add_argument("--dose", type=float, default=0.3)
ap.add_argument("--move", type=float, default=4.0)
ap.add_argument("--gain", type=float, default=0.3)
ap.add_argument("--bits", type=int, default=8)
a = ap.parse_args()
SR, HOP = 48000, 1920

mrt = S.MagentaRT2System(size="mrt2_small", bits=a.bits)
emb = mrt.embed_style("slow ambient drone, warm analogue pads", use_mapper=True)
cfgc = next(c for c in mrt._model.input_configs if c.key == MUSICCOCA.key)
tok = (mrt._style_model.tokenize(emb).tolist() + [-1] * cfgc.rvq_truncation_level)[: cfgc.rvq_truncation_level]
blk, constants = mrt._build_conditioning({MUSICCOCA.key: tok})
state0 = mrt._sampler.get_initial_state(1, sl.ChannelSpec(shape=(mrt._num_channels,), dtype=mx.int32), constants={}, training=False)
v = mx.random.normal((1024,), key=mx.random.key(0))
steer = {"dose": 0.0, "v": v / mx.linalg.norm(v)}
hook(mrt._sampler.depthformer.decoder.temporal_body.layers[0].layers[6], steer)

ring = collections.deque()
cnt = {"gen": 0, "played": 0, "under": 0}
step_ms, marks, dac_log = [], [], []
stop = threading.Event()


def gen():
    st = state0
    last = steer["dose"]
    while not stop.is_set():
        while cnt["gen"] - cnt["played"] // HOP > a.lead and not stop.is_set():
            time.sleep(0.0005)
        d = steer["dose"]
        if d != last:
            marks[-1]["frame"] = cnt["gen"]
            last = d
        t0 = time.perf_counter()
        y, st, _ = mrt._sampler.step_with_emits(x=blk, state=st, constants=constants, training=False)
        mx.eval(y.values)
        step_ms.append((time.perf_counter() - t0) * 1000)
        w = np.array(y.values[0]).astype(np.float32)
        if np.abs(w).max() > 2:
            w = w / 32768.0
        ring.append(w)
        cnt["gen"] += 1


buf = np.zeros((0, 2), np.float32)


def cb(out, frames, t, status):
    global buf
    while len(buf) < frames and ring:
        buf = np.concatenate([buf, ring.popleft()])
    if len(buf) >= frames:
        out[:] = buf[:frames] * a.gain
        buf = buf[frames:]
        dac_log.append((cnt["played"], t.outputBufferDacTime))
        cnt["played"] += frames
    else:
        out.fill(0)
        if cnt["played"]:
            cnt["under"] += 1


with sd.OutputStream(samplerate=SR, channels=2, blocksize=a.block, latency="low", callback=cb) as s:
    def mover():                       # the control moves, on their own thread; generation stays on the main thread (MLX)
        t0 = time.time()
        nxt = a.move
        while time.time() - t0 < a.seconds:
            time.sleep(0.005)
            if time.time() - t0 >= nxt:
                marks.append({"t": s.time, "frame": None})
                steer["dose"] = a.dose if steer["dose"] == 0 else 0.0
                nxt += a.move
        stop.set()
    threading.Thread(target=mover, daemon=True).start()
    gen()
    dev = s.latency
starts = np.array([x[0] for x in dac_log])
dacs = np.array([x[1] for x in dac_log])
lat = []
for m in marks:
    if m["frame"] is None:
        continue
    smp = m["frame"] * HOP
    i = np.searchsorted(starts, smp, side="right") - 1
    if 0 <= i < len(starts) and smp < cnt["played"]:
        lat.append(round((dacs[i] + (smp - starts[i]) / SR - m["t"]) * 1000, 1))
sm = np.array(step_ms[10:])
res = {"lead": a.lead, "bits": a.bits, "device_latency_s": dev, "step_ms_p50": round(float(np.median(sm)), 2),
       "step_ms_p95": round(float(np.percentile(sm, 95)), 2), "underruns": cnt["under"], "seconds_played": round(cnt["played"] / SR, 1),
       "latency_ms": lat, "latency_ms_median": float(np.median(lat)) if lat else None}
print(json.dumps(res))
json.dump(res, open(f"_tmp/mrt2/live_lead{a.lead}.json", "w"))
