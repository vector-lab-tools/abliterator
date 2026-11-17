# Abliterator: does carrying the EnCodec decoder's LSTM state make windowed decoding exact?
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
The decoder is: conv (k7, non-causal) -> 2-layer LSTM -> upsampling convs. The LSTM is causal,
so its state can be carried from frame to frame; the first conv needs 3 frames to the right.
Here the conv and LSTM run once over the whole sequence (what a carried state gives), and only
the upsampling stack (layers 2 to 15) runs in windows [s - ctx, s + hop + la) with a crossfade.
If the error falls to numerical noise for a small la, a streaming decoder with a declared
lookahead of 3 + la frames reproduces the one-go decode.
"""
import time

import numpy as np
import torch
from transformers import EncodecModel

m = EncodecModel.from_pretrained("facebook/encodec_32khz").eval()
codes = torch.load("_tmp/bench/codes_600.pt")            # (K, T)
HOP = 640
with torch.no_grad():
    emb = m.quantizer.decode(codes.view(1, *codes.shape).transpose(0, 1))   # (1, D, T)
    x = emb
    for layer in m.decoder.layers[:2]:
        x = layer(x)
    lstm_out = x
    ref = emb
    for layer in m.decoder.layers:
        ref = layer(ref)
    ref = ref[0, 0].numpy()
    full = m.decode(codes.view(1, 1, *codes.shape), [None])[0][0, 0].numpy()
print("own full decode vs model.decode, err dB:", round(float(20 * np.log10(np.sqrt(((ref - full) ** 2).mean()) / np.sqrt((full ** 2).mean()) + 1e-12)), 1))


def tail(z):
    with torch.no_grad():
        for layer in m.decoder.layers[2:]:
            z = layer(z)
    return z[0, 0].numpy()


for dev in ("cpu", "mps"):
    m.to(dev)
    lo_ = lstm_out.to(dev)
    for hop in (1, 2, 5):
        for la in (0, 1, 2):
            for ctx in (2, 5):
                xf = 64
                out, tms = np.zeros(0, np.float32), []
                for s in range(50, 350, hop):
                    lo, hi = s - ctx, s + hop + la
                    t0 = time.perf_counter()
                    w = tail(lo_[:, :, lo:hi])
                    if dev == "mps":
                        torch.mps.synchronize()
                    tms.append(time.perf_counter() - t0)
                    a0 = (s - lo) * HOP
                    piece = w[a0 - xf: a0 + hop * HOP]
                    if len(out) == 0:
                        out = piece[xf:].copy()
                    else:
                        r = np.linspace(0, 1, xf, dtype=np.float32)
                        out[-xf:] = out[-xf:] * (1 - r) + piece[:xf] * r
                        out = np.concatenate([out, piece[xf:]])
                rf = ref[50 * HOP: 50 * HOP + len(out)]
                err = np.sqrt(((out - rf) ** 2).mean()) / np.sqrt((rf ** 2).mean())
                print(dev, f"hop{hop} la{la} ctx{ctx}", "err_db", round(float(20 * np.log10(err + 1e-12)), 1),
                      "ms/window", round(float(np.median(tms)) * 1000, 2), "ms/frame", round(float(np.median(tms)) * 1000 / hop, 2),
                      "lookahead_ms", (3 + la) * 20, flush=True)
