# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
The live engine: generation, decoding and playback, none of which waits on the page.

    generation thread (MPS)  --frames-->  decoder process (CPU)  --samples-->  ring  -->  sound card
         ^ reads the Steer doses                                                   (callback, master chain)
           at every step

- The generation thread runs MusicGen one frame at a time and holds at most `lead` complete frames
  beyond what the sound card has played: that is the jitter buffer, and most of the latency.
- The decoder process (codec.py) carries EnCodec's state and gives exact audio 4 frames late.
- The callback plays from the ring, applies the master chain (conventional DSP after the model: a
  low-pass filter, a soft limiter, a volume; not a strip), and records the DAC time of each block, so
  that the latency of a control move is measured from the move to the moment its first frame is heard,
  and the level meter is reported for what is being heard now.
"""
from __future__ import annotations

import collections
import json
import multiprocessing as mp
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from . import ROOT, TOOL, VERSION
from .codec import HOP, LOOKAHEAD_FRAMES, SR, worker
from .dsp import Master

SESSIONS = ROOT / "data" / "sessions"
SNAPSHOTS = ROOT / "data" / "snapshots"
DELAY_PATTERN_FRAMES = 3


class Engine:
    name = "musicgen"
    label = "MusicGen small"

    def __init__(self):
        self.mg = None
        self.steer = None
        self.status_text = "not loaded"
        self.error: Optional[str] = None
        self.playing = False
        self.master = Master(SR, 1)
        self.lead = 8
        self.block = 512
        self._ctx = mp.get_context("spawn")
        self._lock = threading.RLock()
        self._reset_counters()

    # ------------------------------------------------------------------ loading
    def load(self, directions: Dict[str, Any], layer: int = 12):
        from .musicgen import MusicGen, Steer
        self.status_text = "starting the decoder"
        self.inq, self.outq = self._ctx.Queue(), self._ctx.Queue()
        self.dproc = self._ctx.Process(target=worker, args=(self.inq, self.outq), daemon=True)
        self.dproc.start()
        self.status_text = "loading MusicGen small"
        self.mg = MusicGen()
        self.status_text = "warming the kernels"
        self.mg.warm()
        msg = self.outq.get(timeout=120)
        if msg[0] != "ready":
            raise RuntimeError(f"the decoder did not start: {msg}")
        self.steer = Steer(self.mg, layer)
        for name, d in directions.items():
            self.steer.set_direction(name, d.unit(layer))
        threading.Thread(target=self._reader, daemon=True).start()
        self.status_text = "ready"

    # ------------------------------------------------------------------ play and stop
    def _reset_counters(self):
        self.ring: "collections.deque" = collections.deque()
        self.buf = np.zeros(0, np.float32)
        self.played = 0                    # samples played
        self.underruns = 0
        self.step_ms: "collections.deque" = collections.deque(maxlen=200)
        self.dac: "collections.deque" = collections.deque(maxlen=400)      # (first sample, dac time, rms)
        self.pending: List[Dict[str, Any]] = []      # control moves waiting for their first frame to be heard
        self.measured: "collections.deque" = collections.deque(maxlen=20)
        self.moves: List[Dict[str, Any]] = []        # the session's control moves, by frame
        self.recorded: List[np.ndarray] = []
        self.expect = 0                              # the next frame index the ring expects
        self.t_play = None

    def play(self, preset: Dict[str, Any], seed: int, cfg: Optional[float] = None):
        import sounddevice as sd
        from .musicgen import Stepper
        with self._lock:
            if self.playing:
                self.stop()
            self.preset, self.seed = preset, int(seed)
            self.cfg = float(preset.get("cfg", 3.0) if cfg is None else cfg)
            self._reset_counters()
            self.inq.put(("reset",))
            self.stepper = Stepper(self.mg, preset["prompt"], self.cfg, self.seed,
                                   preset.get("temperature", 1.0), preset.get("top_k", 250))
            self.start_doses = dict(self.steer.doses)
            self.start_master = self.master.settings()
            self._stop = threading.Event()
            self.playing = True
            self.gen_thread = threading.Thread(target=self._generate, daemon=True)
            self.gen_thread.start()
            self.stream = sd.OutputStream(samplerate=SR, channels=1, blocksize=self.block, latency="low",
                                          callback=self._callback)
            self.stream.start()
            self.t_play = time.time()

    def stop(self) -> Optional[str]:
        with self._lock:
            if not self.playing:
                return None
            self._stop.set()
            self.gen_thread.join(timeout=2)
            self.stream.stop()
            self.stream.close()
            self.playing = False
            return self._save()

    # ------------------------------------------------------------------ controls
    def control(self, doses: Optional[Dict[str, float]] = None, master: Optional[Dict[str, Any]] = None):
        t = self.stream.time if self.playing else None
        if doses:
            for k, v in doses.items():
                self.steer.set(k, float(v))
            if self.playing:
                # the generation thread marks the first frame written with these doses
                self.pending.append({"t": t, "frame": None, "doses": dict(self.steer.doses), "col": None})
        if master:
            self.master.update(master)
            if self.playing:
                self.moves.append({"frame": self.played // HOP, "master": self.master.settings()})

    # ------------------------------------------------------------------ the three loops
    def _generate(self):
        st = self.stepper
        try:
            while not self._stop.is_set():
                while st.n_frames - self.played // HOP > self.lead and not self._stop.is_set():
                    time.sleep(0.0005)
                marks = [p for p in self.pending if p["col"] is None]
                for p in marks:
                    p["col"] = st.next_col                 # this step writes codebook 0 of frame col - 1 with the new doses
                    p["frame"] = st.next_col - 1
                    self.moves.append({"frame": p["frame"], "doses": p["doses"]})
                t0 = time.perf_counter()
                f = st.step()
                self.mg.sync()
                self.step_ms.append((time.perf_counter() - t0) * 1000)
                if f is not None:
                    self.inq.put(("frame", st.n_frames - 1, f.tolist()))
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error = f"generation stopped: {type(e).__name__}: {e}"

    def _reader(self):
        while True:
            msg = self.outq.get()
            if msg[0] == "audio" and self.playing:
                _, idx, a = msg
                if idx == self.expect:
                    self.ring.append(a)
                    self.recorded.append(a)
                    self.expect += 1

    def _callback(self, out, frames, t, status):
        while len(self.buf) < frames and self.ring:
            self.buf = np.concatenate([self.buf, self.ring.popleft()])
        if len(self.buf) >= frames:
            x = self.buf[:frames]
            self.buf = self.buf[frames:]
            y = self.master.process(x)
            out[:, 0] = y
            start = self.played
            self.played += frames
            dac = t.outputBufferDacTime
            self.dac.append((start, dac, float(np.sqrt((y ** 2).mean()))))
            for p in list(self.pending):
                if p["frame"] is not None and p["t"] is not None:
                    s = p["frame"] * HOP
                    if start <= s < self.played:
                        self.measured.append(round((dac + (s - start) / SR - p["t"]) * 1000, 1))
                        self.pending.remove(p)
                    elif s < start:                        # already played (a move during the first frames)
                        self.pending.remove(p)
        else:
            out.fill(0)
            if self.played > 0:
                self.underruns += 1

    # ------------------------------------------------------------------ readings
    def latency(self) -> Dict[str, Any]:
        dev_ms = (self.stream.latency * 1000) if self.playing else 41.0
        parts = {"generation lead": self.lead * 20, "delay pattern (all four codebooks)": DELAY_PATTERN_FRAMES * 20,
                 "decoder lookahead": LOOKAHEAD_FRAMES * 20, "device": round(dev_ms, 1)}
        # the lead already holds the decoder's lookahead (the generator counts complete frames
        # beyond what has been played), so the first change is heard after lead + device; the
        # full change after the delay pattern as well
        first = self.lead * 20 + dev_ms
        m = list(self.measured)
        return {"declared_first_ms": round(first, 1), "declared_full_ms": round(first + DELAY_PATTERN_FRAMES * 20, 1),
                "parts": parts, "measured_ms": m[-5:], "measured_median_ms": float(np.median(m)) if m else None}

    def status(self) -> Dict[str, Any]:
        s = {"tool": TOOL, "version": VERSION, "state": self.status_text, "error": self.error, "playing": self.playing,
             "model": self.mg.info() if self.mg else None, "lead": self.lead, "lead_range": [4, 25], "frame_ms": 20,
             "block": self.block,
             "doses": dict(self.steer.doses) if self.steer else {}, "master": self.master.settings()}
        if self.playing:
            st = self.stepper
            sm = np.array(self.step_ms) if self.step_ms else np.zeros(1)
            now = self.stream.time
            heard = [d for d in self.dac if d[1] <= now]
            s.update({"preset": self.preset.get("id"), "seed": self.seed, "cfg": self.cfg,
                      "seconds": round(self.played / SR, 2), "frames_generated": st.n_frames, "reprimes": st.reprimes,
                      "step_ms_p50": round(float(np.median(sm)), 2), "step_ms_p95": round(float(np.percentile(sm, 95)), 2),
                      "underruns": self.underruns, "level": heard[-1][2] if heard else 0.0,
                      "latency": self.latency()})
        else:
            s["latency"] = self.latency()
        return s

    # ------------------------------------------------------------------ snapshots and the session's record
    def snapshot(self, seconds: float) -> Dict[str, Any]:
        """The last `seconds` of what the engine generated, with its recipe (MusicGen: not yet replayable)."""
        import secrets
        import soundfile as sf
        rec = list(self.recorded)
        if not rec:
            raise RuntimeError("nothing has been played yet")
        to = len(rec)
        n = max(1, min(to, int(round(seconds * SR / HOP))))
        fr = to - n
        SNAPSHOTS.mkdir(parents=True, exist_ok=True)
        sid = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        sf.write(SNAPSHOTS / f"{sid}.wav", np.concatenate(rec[fr:to]), SR, subtype="FLOAT")
        recipe = {"format": "abliterator.snapshot.v1", "tool": {"name": TOOL, "version": VERSION}, "id": sid,
                  "engine": "musicgen", "model": self.mg.info(), "preset": self.preset, "seed": self.seed, "cfg": self.cfg,
                  "start": {"doses": self.start_doses, "master": self.start_master},
                  "moves": [m for m in list(self.moves) if m["frame"] < to], "frames": [fr, to],
                  "seconds": round(n * HOP / SR, 2), "sample_rate": SR, "channels": 1, "audio": f"{sid}.wav",
                  "audio_is": "the model's output, before the master chain and any track effects; 32-bit float",
                  "chain": {"steer": {"layer": self.steer.layer, "directions": sorted(self.steer.vectors)}},
                  "replay": "not yet implemented for MusicGen"}
        (SNAPSHOTS / f"{sid}.json").write_text(json.dumps(recipe, indent=1))
        return {"snapshot": sid, "seconds": recipe["seconds"], "sample_rate": SR, "channels": 1,
                "label": f"{self.preset.get('name', self.preset.get('id'))} · seed {self.seed}"}

    def _save(self) -> Optional[str]:
        if not self.recorded:
            return None
        import soundfile as sf
        import torch
        import transformers
        SESSIONS.mkdir(parents=True, exist_ok=True)
        name = time.strftime("%Y%m%d-%H%M%S")
        audio = np.concatenate(self.recorded)
        sf.write(SESSIONS / f"{name}.wav", audio, SR, subtype="FLOAT")
        m = Master(SR, 1)
        m.update(self.start_master)
        by: Dict[int, List[Dict[str, Any]]] = {}
        for mv in self.moves:
            if "master" in mv:
                by.setdefault(mv["frame"], []).append(mv["master"])
        post = []
        for f in range(len(audio) // HOP):
            for upd in by.get(f, []):
                m.update(upd)
            post.append(m.process(audio[f * HOP:(f + 1) * HOP]))
        if post:
            sf.write(SESSIONS / f"{name}.post.wav", np.concatenate(post), SR, subtype="FLOAT")
        recipe = {
            "format": "abliterator.session.v1",
            "tool": {"name": TOOL, "version": VERSION},
            "model": self.mg.info(),
            "preset": self.preset, "seed": self.seed, "cfg": self.cfg, "apply_to": "both",
            "sampler": {"temperature": self.stepper.temperature, "top_k": self.stepper.top_k},
            "chain": {"steer": {"layer": self.steer.layer, "directions": sorted(self.steer.vectors)}},
            "start": {"doses": self.start_doses, "master": self.start_master},
            "moves": self.moves,
            "frames": len(self.recorded), "audio": f"{name}.wav", "audio_is": "the model's output, before the master chain",
            "audio_post": f"{name}.post.wav", "audio_post_is": "after the master chain, computed offline from the moves",
            "context": {"window": 1000, "keep": 500},
            "decoder": "streaming EnCodec, LSTM state carried, 4 frames lookahead (exact)",
            "underruns": self.underruns,
            "environment": {"torch": torch.__version__, "transformers": transformers.__version__},
        }
        (SESSIONS / f"{name}.json").write_text(json.dumps(recipe, indent=1))
        return name
