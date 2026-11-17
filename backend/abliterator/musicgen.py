# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
MusicGen small, generated one frame at a time.

The loop is written out, as Abliterton Lens writes out its sampling loop, so that the seed, the
sampler and the place where strips act are in one file:
- each step is one decoder pass over one column of the delay pattern, with the KV cache;
- codebook k is predicted k steps late (MusicGen's delay pattern), so a frame is complete three
  steps after its first code;
- with guidance (cfg > 1) the unconditional pass rides in the same batch, with zeroed encoder
  states, and the logits are mixed as u + cfg (c - u), as Transformers does;
- tokens are drawn on the CPU from a seeded torch.Generator (top-k), so a run repeats exactly;
- MusicGen's position table holds 2048 steps (41 s) and it was trained on 30 s excerpts, so the
  context slides: past WINDOW columns, the cache is rebuilt from the last KEEP (a re-prime).

Measured on an Apple M5 (bench/): bfloat16 with sdpa attention on MPS, 12.9 ms
a step without guidance and 17.3 ms with it, against 20 ms of audio a step.
"""
from __future__ import annotations

import os
import threading
from typing import Any, Dict, List, Optional

import torch

MODEL = "facebook/musicgen-small"
K = 4
PAD = 2048
WINDOW = 1000           # columns before a re-prime (20 s)
KEEP = 500              # columns kept as context (10 s)


def detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class MusicGen:
    """The loaded model, its text encodings, and the hook point for the stream."""

    def __init__(self, device: Optional[str] = None, dtype: str = "bfloat16"):
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        self.device = device or detect_device()
        self.dtype_name = dtype
        self.dtype = getattr(torch, dtype)
        self.proc = AutoProcessor.from_pretrained(MODEL)
        m = MusicgenForConditionalGeneration.from_pretrained(MODEL, dtype=self.dtype, attn_implementation="eager")
        # sdpa in the decoder only: Transformers 4.57 refuses it for the T5 encoder
        for mod in m.decoder.modules():
            if hasattr(mod, "config") and hasattr(mod.config, "_attn_implementation"):
                mod.config._attn_implementation = "sdpa"
        m.audio_encoder = None                       # EnCodec runs in the decoder process (codec.py)
        self.m = m.to(self.device).eval()
        self.layers = self.m.decoder.model.decoder.layers
        self.n_layers = len(self.layers)
        self.d_model = self.m.decoder.config.hidden_size
        self.revision = _revision()
        self._enc: Dict[Any, Any] = {}
        self.lock = threading.RLock()

    def info(self) -> Dict[str, Any]:
        return {"model_id": MODEL, "revision_hash": self.revision, "dtype": self.dtype_name, "device": self.device,
                "n_layers": self.n_layers, "d_model": self.d_model, "codebooks": K, "frame_rate": 50,
                "sample_rate": 32000, "channels": 1, "licence": "CC-BY-NC-4.0 (model weights, Meta); downloaded at first use, not redistributed"}

    def encode(self, prompt: str, cfg: float):
        """The projected text encoding for a prompt, with the unconditional half for guidance. Cached."""
        key = (prompt, cfg > 1)
        if key not in self._enc:
            inp = self.proc(text=[prompt], padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                h = self.m.text_encoder(**inp).last_hidden_state
                mask = inp["attention_mask"]
                if cfg > 1:
                    h = torch.cat([h, torch.zeros_like(h)], 0)
                    mask = torch.cat([mask, torch.zeros_like(mask)], 0)
                h = self.m.enc_to_dec_proj(h).to(self.dtype)
            self._enc[key] = (h, mask)
        return self._enc[key]

    def warm(self, cfg: float = 3.0):
        """Run the shapes the live loop will use (a step, a re-prime), so MPS builds their kernels before play."""
        for g in (1.0, cfg):
            st = Stepper(self, "warm", g, 0)
            for _ in range(8):
                st.step()
            st.cols = st.cols + [torch.randint(0, 2048, (K,)) for _ in range(KEEP)]
            st.reprime()
            st.step()

    def sync(self):
        if self.device == "mps":
            torch.mps.synchronize()


def _revision() -> Optional[str]:
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(MODEL, "config.json")
        if isinstance(hit, str):
            return os.path.basename(os.path.dirname(hit))
    except Exception:
        pass
    return None


def sample(logits: torch.Tensor, gen: torch.Generator, temperature: float, top_k: int) -> torch.Tensor:
    """logits (K, V), float64 on the CPU; one token per codebook."""
    if temperature <= 0:
        return logits.argmax(-1)
    v, i = torch.topk(logits / temperature, top_k, dim=-1)
    pick = torch.multinomial(torch.softmax(v, -1), 1, generator=gen)
    return i.gather(-1, pick).squeeze(-1)


class Stepper:
    """One column a call. Holds the KV cache, the columns so far and the completed frames."""

    def __init__(self, mg: MusicGen, prompt: str, cfg: float, seed: int, temperature: float = 1.0, top_k: int = 250):
        self.mg, self.cfg = mg, cfg
        self.enc, self.mask = mg.encode(prompt, cfg)
        self.gen = torch.Generator().manual_seed(int(seed))
        self.temperature, self.top_k = temperature, top_k
        self.past = None
        self.cols: List[torch.Tensor] = [torch.full((K,), PAD, dtype=torch.long)]
        self.cbase = 0                         # absolute index of self.cols[0]
        self.n_frames = 0                      # frames completed
        self.reprimes = 0

    @property
    def next_col(self) -> int:
        """Absolute index of the column the next step predicts."""
        return self.cbase + len(self.cols)

    def _decoder(self, ids, past):
        return self.mg.m.decoder(input_ids=ids, encoder_hidden_states=self.enc, encoder_attention_mask=self.mask,
                                 past_key_values=past, use_cache=True)

    def step(self) -> Optional[torch.Tensor]:
        """Predict the next column. Returns a completed frame's four codes, or None in the first steps."""
        cur = self.cols[-1].to(self.mg.device).view(K, 1)
        if self.cfg > 1:
            cur = cur.repeat(2, 1)
        with torch.no_grad():
            out = self._decoder(cur, self.past)
        self.past = out.past_key_values
        lg = out.logits[:, -1].float().cpu().double()
        if self.cfg > 1:
            c, u = lg[:K], lg[K:]
            lg = u + self.cfg * (c - u)
        tok = sample(lg, self.gen, self.temperature, self.top_k)
        nc = self.next_col
        for k in range(K):                     # the delay pattern: codebook k starts at column k + 1
            if nc <= k:
                tok[k] = PAD
        self.cols.append(tok)
        f = nc - K                             # frame f is complete once column f + K exists
        frame = None
        if f >= 0:
            frame = torch.stack([self.cols[f + 1 + k - self.cbase][k] for k in range(K)])
            self.n_frames = f + 1
        if len(self.cols) > WINDOW:
            self.reprime()
        return frame

    def reprime(self):
        """Rebuild the cache from the last KEEP columns, which start again at position 0."""
        self.cbase += len(self.cols) - KEEP
        self.cols = self.cols[-KEEP:]
        ids = torch.stack(self.cols[:-1], 1).to(self.mg.device)          # (K, KEEP - 1); the last is the next input
        if self.cfg > 1:
            ids = ids.repeat(2, 1)
        with torch.no_grad():
            self.past = self._decoder(ids, None).past_key_values
        self.reprimes += 1


class Steer:
    """The live Steer strips: at one layer of the stream, h' = h + ‖h‖ · Σ dose_i · v_i, the v_i unit
    directions. The doses are read at every step, so a move is heard from the next frame (latency
    class: live). Applied to both halves of a guidance batch (apply_to = both), so a dose means the same
    at any guidance scale."""

    def __init__(self, mg: MusicGen, layer: int):
        self.mg, self.layer = mg, layer
        self.vectors: Dict[str, torch.Tensor] = {}
        self.doses: Dict[str, float] = {}
        self.vec: Optional[torch.Tensor] = None
        self.handle = mg.layers[layer - 1].register_forward_hook(self._hook)    # the stream at layer l is block l-1's output

    def set_direction(self, name: str, unit: torch.Tensor):
        self.vectors[name] = unit.to(self.mg.device, self.mg.dtype)
        self.doses.setdefault(name, 0.0)
        self._sum()

    def set(self, name: str, dose: float):
        if name in self.vectors:
            self.doses[name] = float(dose)
            self._sum()

    def _sum(self):
        v = None
        for n, d in self.doses.items():
            if d != 0.0 and n in self.vectors:
                v = self.vectors[n] * d if v is None else v + self.vectors[n] * d
        self.vec = v                                     # one assignment: the hook sees the old sum or the new

    def _hook(self, module, args, output):
        v = self.vec
        if v is None:
            return output
        h = output[0] if isinstance(output, tuple) else output
        h = h + h.norm(dim=-1, keepdim=True) * v
        return (h,) + tuple(output[1:]) if isinstance(output, tuple) else h

    def remove(self):
        self.handle.remove()
