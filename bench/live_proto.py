# Abliterator: a throwaway live prototype, to measure control-to-sound latency.
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Three threads, as the plan proposes:
- generation (MPS): MusicGen small frame by frame, with a steering hook on the residual stream
  at one layer whose dose is read from shared state at every step;
- decoding (CPU): EnCodec with the LSTM state carried, 4 frames of lookahead, exact;
- playback: a sounddevice callback reading a ring of samples.

Every `--move` seconds the dose jumps between 0 and --dose. For each move the script records
the stream time of the move and the DAC time of the first sample of the first frame generated
with the new dose (codebook 0; the other codebooks follow over 3 more frames).

    .venv/bin/python bench/live_proto.py --seconds 60 --cfg 1 --lead 3 --block 512
"""
import argparse
import collections
import json
import sys
import threading
import time

import numpy as np
import psutil
import sounddevice as sd
import soundfile as sf
import torch

sys.path.insert(0, "bench")
from musicgen_bench import MODEL, Stepper, encode_text  # noqa: E402
from transformers import AutoProcessor, MusicgenForConditionalGeneration, EncodecModel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=60)
ap.add_argument("--cfg", type=float, default=1.0)
ap.add_argument("--lead", type=int, default=8, help="complete frames the generator may hold beyond what has been played")
ap.add_argument("--block", type=int, default=512)
ap.add_argument("--layer", type=int, default=12)
ap.add_argument("--dose", type=float, default=0.6)
ap.add_argument("--move", type=float, default=5.0)
ap.add_argument("--gain", type=float, default=0.05)
ap.add_argument("--prompt", default="slow ambient drone, warm analogue pads, no drums")
ap.add_argument("--out", default="_tmp/bench/live.wav")
ap.add_argument("--json", default="_tmp/bench/live.json")
a = ap.parse_args()
SR, HOP, LA = 32000, 640, 3          # LA: the first conv's right context, in frames

proc = AutoProcessor.from_pretrained(MODEL)
m = MusicgenForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="eager").to("mps").eval()
for mod in m.decoder.modules():
    if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
        mod.config._attn_implementation = "sdpa"
codec = EncodecModel.from_pretrained("facebook/encodec_32khz").eval()       # CPU, float32
enc, mask = encode_text(m, proc, a.prompt, a.cfg, "mps", torch.bfloat16)

# ---- the live strip: steer the residual stream at one layer by dose × ‖h‖ × v
g = torch.Generator().manual_seed(123)
v = torch.randn(1024, generator=g)
v = (v / v.norm()).to("mps", torch.bfloat16)
state = {"dose": 0.0}


def steer(module, args, output):
    d = state["dose"]
    if d == 0.0:
        return output
    h = output[0] if isinstance(output, tuple) else output
    h = h + d * h.norm(dim=-1, keepdim=True) * v
    return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h


m.decoder.model.decoder.layers[a.layer - 1].register_forward_hook(steer)

frames_q: "collections.deque" = collections.deque()
ring = collections.deque()           # numpy arrays of samples
ring_lock = threading.Lock()
cnt = {"gen": 0, "dec_frames": 0, "played": 0, "underruns": 0, "callbacks": 0}
step_ms, dec_ms = [], []
marks = []                           # {"t_move", "dose", "frame"}; filled later with dac time
stop = threading.Event()
recorded = []


def gen_loop():
    st = Stepper(m, enc, mask, a.cfg, 7)
    last_dose = state["dose"]
    while not stop.is_set():
        # throttle: the generator stays at most `lead` frames ahead of what the decoder has taken in
        while len(st.frames) - cnt["played"] // HOP > a.lead and not stop.is_set():
            time.sleep(0.001)
        d = state["dose"]
        t0 = time.perf_counter()
        nc = len(st.cols)                       # the column this step predicts
        f = st.step()
        torch.mps.synchronize()
        step_ms.append((time.perf_counter() - t0) * 1000)
        if d != last_dose:
            marks[-1]["frame"] = nc - 1         # codebook 0 of frame nc-1 is the first written with the new dose
            last_dose = d
        if f is not None:
            frames_q.append(f)


cnt["dec_in"] = 0


def dec_loop():
    q = codec.quantizer
    conv0 = codec.decoder.layers[0]
    lstm = codec.decoder.layers[1].lstm
    tail = codec.decoder.layers[2:]
    embs, lstm_out = [], []
    hc = None
    done_conv = 0                               # frames whose conv0 output has gone through the LSTM
    emitted = 0                                 # frames whose audio has been emitted
    with torch.no_grad():
        while not stop.is_set():
            if not frames_q:
                time.sleep(0.001)
                continue
            f = frames_q.popleft()
            cnt["dec_in"] += 1
            t0 = time.perf_counter()
            e = q.decode(f.view(1, 4, 1).transpose(0, 1))       # (1, D, 1)
            embs.append(e)
            n = len(embs)
            # conv0 output for frame p needs embeddings p-3 .. p+3. Frames 0..2 use the module's own
            # reflect padding on the left (the start of the stream).
            new = []
            while done_conv + LA < n:
                p = done_conv
                if p < LA:
                    y = conv0(torch.cat(embs[: p + LA + 1], -1))[..., p: p + 1] if n >= 2 * LA + 1 else None
                    if y is None:
                        break
                else:
                    y = conv0.conv(torch.cat(embs[p - LA: p + LA + 1], -1))
                new.append(y)
                done_conv += 1
            if new:
                x = torch.cat(new, -1).permute(2, 0, 1)
                out, hc = lstm(x, hc)
                lstm_out.append((out + x).permute(1, 2, 0))
            L = torch.cat(lstm_out, -1) if lstm_out else None
            # the tail needs 1 frame to the right and 2 to the left; emit what is final
            while L is not None and emitted + 1 < L.shape[-1]:
                lo = max(0, emitted - 2)
                z = L[..., lo: emitted + 2]
                for layer in tail:
                    z = layer(z)
                w = z[0, 0, (emitted - lo) * HOP: (emitted - lo + 1) * HOP].numpy().copy()
                with ring_lock:
                    ring.append(w)
                recorded.append(w)
                emitted += 1
                cnt["dec_frames"] = emitted
            if len(lstm_out) > 8:               # keep the window short
                L = torch.cat(lstm_out, -1)
                keep = L[..., max(0, emitted - 4):]
                lstm_out[:] = [keep]
                # re-base: frames before emitted-4 are dropped
                emitted_offset = max(0, emitted - 4)
                dec_loop.offset = getattr(dec_loop, "offset", 0) + emitted_offset
                emitted -= emitted_offset
            dec_ms.append((time.perf_counter() - t0) * 1000)


buf = np.zeros(0, np.float32)
dac_log = []                                    # (first sample index of block, dac time)


def callback(out, frames, t, status):
    global buf
    cnt["callbacks"] += 1
    with ring_lock:
        while len(buf) < frames and ring:
            buf = np.concatenate([buf, ring.popleft()])
    if len(buf) >= frames:
        out[:, 0] = buf[:frames] * a.gain
        buf = buf[frames:]
        dac_log.append((cnt["played"], t.outputBufferDacTime))
        cnt["played"] += frames
    else:
        out.fill(0)
        if cnt["played"] > 0:
            cnt["underruns"] += 1


proc_ps = psutil.Process()
mem_log = []
threading.Thread(target=gen_loop, daemon=True).start()
threading.Thread(target=dec_loop, daemon=True).start()
with sd.OutputStream(samplerate=SR, channels=1, blocksize=a.block, latency="low", callback=callback) as s:
    t_start = time.time()
    next_move = a.move
    while time.time() - t_start < a.seconds:
        time.sleep(0.01)
        el = time.time() - t_start
        if el >= next_move:
            nd = a.dose if state["dose"] == 0 else 0.0
            marks.append({"t_move": s.time, "dose": nd, "frame": None})
            state["dose"] = nd
            next_move += a.move
        if int(el * 10) % 50 == 0:
            mem_log.append({"t": round(el, 1), "rss_gb": proc_ps.memory_info().rss / 2**30,
                            "mps_gb": torch.mps.driver_allocated_memory() / 2**30})
    out_latency = s.latency
    stop.set()

# latency of each move: the DAC time of the first sample of the marked frame
starts = np.array([x[0] for x in dac_log])
dacs = np.array([x[1] for x in dac_log])
lat = []
for mk in marks:
    if mk["frame"] is None:
        continue
    sample = mk["frame"] * HOP
    i = np.searchsorted(starts, sample, side="right") - 1
    if 0 <= i < len(starts) and sample < cnt["played"]:
        t_dac = dacs[i] + (sample - starts[i]) / SR
        lat.append(round((t_dac - mk["t_move"]) * 1000, 1))
st_ = np.array(step_ms[10:])
res = {"cfg": a.cfg, "lead": a.lead, "block": a.block, "device_latency_s": out_latency,
       "step_ms_p50": round(float(np.median(st_)), 2), "step_ms_p95": round(float(np.percentile(st_, 95)), 2),
       "step_ms_p99": round(float(np.percentile(st_, 99)), 2), "step_ms_max": round(float(st_.max()), 1),
       "dec_ms_p50": round(float(np.median(dec_ms)), 2), "dec_ms_p95": round(float(np.percentile(dec_ms, 95)), 2),
       "underruns": cnt["underruns"], "callbacks": cnt["callbacks"], "seconds_played": round(cnt["played"] / SR, 1),
       "latency_ms_codebook0": lat, "latency_ms_median": float(np.median(lat)) if lat else None,
       "latency_ms_all_codebooks_median": float(np.median(lat)) + 60 if lat else None,
       "mem_first": mem_log[0] if mem_log else None, "mem_last": mem_log[-1] if mem_log else None,
       "mem_peak_rss_gb": max(x["rss_gb"] for x in mem_log) if mem_log else None,
       "mem_peak_mps_gb": max(x["mps_gb"] for x in mem_log) if mem_log else None}
sf.write(a.out, np.concatenate(recorded), SR)
print(json.dumps(res, indent=1))
json.dump(res, open(a.json, "w"), indent=1)
