# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
A streaming decoder for MusicGen's EnCodec (facebook/encodec_32khz), and the process that runs it.

The 32 kHz EnCodec is non-causal: its convolutions look both ways and its decoder holds a
two-layer LSTM. Decoding overlapping windows from scratch never matches the one-go decode
(the LSTM restarts each time). This decoder carries the LSTM state from frame to frame and gives
each convolution the context it needs:

    codes -> quantizer (per frame) -> conv0 (kernel 7: 3 frames each side)
          -> LSTM (causal, state carried) -> upsampling stack (2 frames left, 1 right) -> audio

A frame's audio is final once 3 + 1 = 4 later frames have arrived: a declared lookahead of 80 ms.
Its output equals EncodecModel.decode on the whole sequence to rounding (bench/encodec_stateful.py:
about -160 dB).

The decoder runs in its own process, so that its Python work never holds the generation thread's
interpreter lock (bench/live_proto.py found that a decoder thread pushed the step time past 20 ms).
"""
from __future__ import annotations

import queue
from typing import List, Tuple

import numpy as np

SR = 32000
HOP = 640               # samples per frame (50 frames a second)
LA0 = 3                 # conv0's right context, frames
LA_TAIL = 1             # the upsampling stack's right context, frames
CTX_TAIL = 2            # and its left context
LOOKAHEAD_FRAMES = LA0 + LA_TAIL
MODEL = "facebook/encodec_32khz"


class StreamingDecoder:
    def __init__(self, threads: int = 2):
        import torch
        from transformers import EncodecModel
        torch.set_num_threads(threads)
        self.torch = torch
        self.m = EncodecModel.from_pretrained(MODEL).eval()
        self.reset()

    def reset(self):
        self.embs: List = []          # quantizer outputs, (1, D, 1) each, from frame self.ebase
        self.ebase = 0
        self.done = 0                 # frames that have been through conv0 and the LSTM
        self.hc = None
        self.L = None                 # LSTM outputs (1, D, n), from frame self.lbase
        self.lbase = 0
        self.emitted = 0              # frames whose audio has been given out

    def push(self, codes) -> List[Tuple[int, np.ndarray]]:
        """codes: four ints, one frame. Returns [(frame index, 640 samples)] for every frame that
        has become final."""
        t = self.torch
        m = self.m
        conv0, lstm, tail = m.decoder.layers[0], m.decoder.layers[1].lstm, m.decoder.layers[2:]
        with t.no_grad():
            c = t.as_tensor(codes, dtype=t.long).view(4, 1, 1)          # (K, B, T)
            self.embs.append(m.quantizer.decode(c))                       # (1, D, 1)
            n = self.ebase + len(self.embs)                               # frames received
            new = []
            while self.done + LA0 < n:
                p = self.done
                if p < LA0:                                               # the stream's start: conv0's own left padding
                    if n < 2 * LA0 + 1:
                        break
                    y = conv0(t.cat(self.embs[: p + LA0 + 1], -1))[..., p: p + 1]
                else:
                    a = p - LA0 - self.ebase
                    y = conv0.conv(t.cat(self.embs[a: a + 2 * LA0 + 1], -1))
                new.append(y)
                self.done += 1
            if new:
                x = t.cat(new, -1).permute(2, 0, 1)
                out, self.hc = lstm(x, self.hc)
                y = (out + x).permute(1, 2, 0)
                self.L = y if self.L is None else t.cat([self.L, y], -1)
            got = []
            if self.L is not None:
                avail = self.lbase + self.L.shape[-1]                     # LSTM frames known
                k = avail - LA_TAIL - self.emitted                        # frames that can be made final now
                if k > 0:
                    e = self.emitted
                    lo = max(0, e - CTX_TAIL)
                    z = self.L[..., lo - self.lbase: e + k + LA_TAIL - self.lbase]
                    for layer in tail:
                        z = layer(z)
                    w = z[0, 0].numpy()
                    for j in range(k):
                        s = (e - lo + j) * HOP
                        got.append((e + j, w[s: s + HOP].copy()))
                    self.emitted += k
                # keep only what later frames need
                cut = (self.emitted - CTX_TAIL) - self.lbase
                if cut > 8:
                    self.L = self.L[..., cut:]
                    self.lbase += cut
            cut = (self.done - LA0) - self.ebase
            if cut > 8:
                self.embs = self.embs[cut:]
                self.ebase += cut
        return got


def worker(inq, outq):
    """The decoder process. Messages in: ("frame", index, codes), ("reset",), None to stop.
    Messages out: ("ready",), ("audio", first frame index, samples), ("error", text)."""
    try:
        dec = StreamingDecoder()
        dec.push([0, 0, 0, 0])            # warm the kernels
        dec.reset()
        outq.put(("ready",))
    except Exception as e:                # pragma: no cover
        outq.put(("error", f"{type(e).__name__}: {e}"))
        return
    while True:
        msg = inq.get()
        batch = [msg]
        while True:                        # take whatever else is waiting, so frames decode together
            try:
                batch.append(inq.get_nowait())
            except queue.Empty:
                break
        for msg in batch:
            if msg is None:
                return
            if msg[0] == "reset":
                dec.reset()
                continue
            for idx, a in dec.push(msg[2]):
                outq.put(("audio", idx, a))
