# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Directions in a model's residual stream, by difference of means over contrast prompt sets. `make`
runs MusicGen (without guidance); Magenta RT 2 averages its own stream (mrt2_proc.make_direction)
and calls `record`.

A contrast set (backend/config/contrasts/*.md) lists positive and negative prompts, as
Abliterton Lens's sets list sentences. For each prompt the model writes from a fixed seed, and the
stream at every layer is averaged over steps SKIP to FRAMES + 3. The direction at each layer is the
mean over positive prompts less the mean over negative prompts: the change the text conditioning
makes in the stream ("with drums" against "without drums") on this model.

The file follows Abliterton Lens's direction format (shared with Ablation Bench): a `layers` array
of shape (n_layers + 1, d), numbered as hidden states (layer l is block l-1's output), with a
sha256 over the float32 little-endian bytes of the vectors, and the source prompts.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from . import ROOT, DATA

CONTRASTS = ROOT / "backend" / "config" / "contrasts"
FOLDER = DATA / "directions"
FRAMES = 150
SKIP = 25


class Direction:
    def __init__(self, data: Dict[str, Any]):
        self.data = data
        self.name = data["name"]
        self.layers = np.asarray(data["layers"], dtype=np.float32)

    def unit(self, layer: int):
        import torch
        v = torch.from_numpy(self.layers[layer].copy())
        return v / v.norm().clamp_min(1e-12)

    def summary(self) -> Dict[str, Any]:
        s = self.data.get("source", {})
        return {"name": self.name, "title": s.get("title", self.name), "positive": s.get("positive_label"),
                "negative": s.get("negative_label"), "separation": self.data.get("separation")}


def read_contrast(path: Path) -> Dict[str, Any]:
    text = path.read_text()
    meta = {}
    m = re.match(r"---\n(.*?)\n---\n", text, re.S)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
    def items(head):
        sec = re.search(rf"## {head}\n(.*?)(\n## |\Z)", text, re.S)
        return [l[2:].strip() for l in sec.group(1).splitlines() if l.startswith("- ")] if sec else []
    return {"name": path.stem, "meta": meta, "positive": items("Positive"), "negative": items("Negative")}


def contrasts() -> List[Dict[str, Any]]:
    return [read_contrast(p) for p in sorted(CONTRASTS.glob("*.md"))]


def _stream_means(mg, prompt: str, seed: int) -> np.ndarray:
    """The stream at every layer, averaged over steps SKIP to FRAMES + 3: (n_layers + 1, d)."""
    import torch
    from .musicgen import Stepper
    sums = [None] * (mg.n_layers + 1)
    count = {"n": 0}
    hs = []

    def grab(i):
        def h(module, args, out):
            if count["on"]:
                y = (out[0] if isinstance(out, tuple) else out)[0, -1].float()
                sums[i + 1] = y if sums[i + 1] is None else sums[i + 1] + y
        return h

    def pre(module, args, kwargs):
        if count["on"]:
            x = (args[0] if args else kwargs["hidden_states"])[0, -1].float()
            sums[0] = x if sums[0] is None else sums[0] + x
    count["on"] = False
    hs.append(mg.layers[0].register_forward_pre_hook(pre, with_kwargs=True))
    for i, layer in enumerate(mg.layers):
        hs.append(layer.register_forward_hook(grab(i)))
    try:
        st = Stepper(mg, prompt, 1.0, seed)
        for j in range(FRAMES + 4):
            count["on"] = j >= SKIP
            st.step()
            count["n"] += int(count["on"])
    finally:
        for h in hs:
            h.remove()
    return torch.stack([s / count["n"] for s in sums]).cpu().numpy()


def make(mg, contrast: Dict[str, Any], seed: int = 0, emit=print) -> Dict[str, Any]:
    t0 = time.time()
    pos, neg = [], []
    for i, p in enumerate(contrast["positive"]):
        emit(f"  + {p}")
        pos.append(_stream_means(mg, p, seed + i))
    for i, p in enumerate(contrast["negative"]):
        emit(f"  - {p}")
        neg.append(_stream_means(mg, p, seed + i))
    return record(contrast, np.stack(pos), np.stack(neg), {"model": mg.info()["model_id"], "revision": mg.revision,
                  "dtype": mg.dtype_name}, seed, t0, "off")


def record(contrast, P, N, model: Dict[str, Any], seed: int, t0: float, guidance: str) -> Dict[str, Any]:
    """Difference of means and the direction file. P, N: (prompts, layers, d)."""
    diff = P.mean(0) - N.mean(0)
    # how far apart the two sets sit along the direction, in units of their spread, per layer
    sep = []
    for l in range(diff.shape[0]):
        u = diff[l] / (np.linalg.norm(diff[l]) + 1e-12)
        a, b = P[:, l] @ u, N[:, l] @ u
        sep.append(float((a.mean() - b.mean()) / (np.sqrt((a.var() + b.var()) / 2) + 1e-9)))
    vec = diff.astype("<f4")
    meta = contrast["meta"]
    return {
        "format": "abliterator.direction.v1",
        "name": contrast["name"],
        **model,
        "layer_convention": "hidden_states",
        "layers": vec.tolist(),
        "norm": [float(np.linalg.norm(v)) for v in vec],
        "sha256": hashlib.sha256(vec.tobytes()).hexdigest(),
        "separation": [round(x, 2) for x in sep],
        "source": {"kind": "contrast prompts", "title": meta.get("title", contrast["name"]),
                   "positive_label": meta.get("positive"), "negative_label": meta.get("negative"),
                   "positive": contrast["positive"], "negative": contrast["negative"],
                   "frames": FRAMES, "skip": SKIP, "seed": seed, "guidance": guidance},
        "made": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t0, 1),
    }


def save(d: Dict[str, Any], folder: Path = FOLDER) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{d['name']}.json"
    p.write_text(json.dumps(d))
    return p


def load_all(folder: Path = FOLDER) -> Dict[str, Direction]:
    out = {}
    for p in sorted(folder.glob("*.json")):
        d = Direction(json.loads(p.read_text()))
        out[d.name] = d
    return out


def ensure(mg, emit=print) -> Dict[str, Direction]:
    """Make any direction whose contrast set has no file yet (each takes about a minute)."""
    have = load_all()
    for c in contrasts():
        if c["name"] not in have:
            emit(f"making the direction '{c['name']}' from its contrast set")
            save(make(mg, c, emit=emit))
    return load_all()
