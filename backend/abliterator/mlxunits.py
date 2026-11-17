# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Native MLX versions of VAU units that act elementwise, so that their edits stay on the GPU.

The VAU units are written in torch, and the host (units.py, mrt2_proc.Hooks) runs each one by
copying the stream out of MLX, through torch on the CPU, and back. Each copy makes MLX finish the
frame up to that layer and wait while the CPU works: Neuron with block=all does it in all twelve
blocks of every frame. For the units here, the same arithmetic is written in MLX and replaces the
torch hook when the strip is attached; tests/test_mlxunits.py checks each against the VAU unit.

Ported: Neuron (the six operations on an MLP's hidden units) and Erosion (bits and hold), each
with the host features Wet, Envelope and Keep size. A strip with a window or a random control
keeps the torch path.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

PORTED = {"neuron", "erosion"}


def _mx():
    import mlx.core as mx
    return mx


def _settings(strip) -> Optional[Dict[str, Any]]:
    """The strip's host features, or None if one of them needs the torch path."""
    from vau import hostfx
    try:
        s = hostfx.settings(strip)
    except Exception:
        return None
    if s["control"] != "none" or hostfx.windowed(s):
        return None
    return s


def _factor(s: Dict[str, Any], index: Optional[int], n: Optional[int]) -> float:
    from vau import hostfx
    return hostfx.factor(s, index, n)


def _mix(h, out, f: float, keepnorm: bool):
    """hostfx.mix in MLX: h moved toward out by f, then, with keepnorm, scaled back to h's size."""
    mx = _mx()
    if f == 1.0 and not keepnorm:
        return out
    x = h + f * (out - h)
    if keepnorm:
        x = x * (mx.linalg.norm(h, axis=-1, keepdims=True) / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), 1e-9))
    return x


def mlx_fn(fn: Callable) -> Callable:
    fn.mlx = True
    return fn


def neuron_fn(ns: List[int], op: str, value: float, f: float, keepnorm: bool) -> Callable:
    """An MLP down-projection's input (one position, flattened) with the named neurons edited, as
    vau/units/neuron.py's neuron_hook does it."""
    masks: Dict[int, Any] = {}

    def edit(h):
        mx = _mx()
        shape, flat = h.shape, h.reshape(-1).astype(mx.float32)
        n = flat.shape[0]
        if n not in masks:
            m = np.zeros(n, np.float32)
            m[[i for i in ns if i < n]] = 1.0
            masks[n] = mx.array(m)
        m = masks[n]
        if op == "scale":
            a = flat * (1.0 + m * (value - 1.0))
        elif op == "add":
            a = flat + m * value
        elif op == "set":
            a = flat * (1.0 - m) + m * value
        elif op == "zero":
            a = flat * (1.0 - m)
        elif op == "flip":
            a = flat * (1.0 - 2.0 * m)
        else:                                                 # clamp
            v = abs(value)
            a = mx.where(m > 0, mx.clip(flat, -v, v), flat)
        out = _mix(flat.reshape(1, 1, -1), a.reshape(1, 1, -1), f, keepnorm)
        return out.reshape(shape).astype(h.dtype)
    return mlx_fn(edit)


def erosion_fn(step: Optional[np.ndarray], hold: int, held: Dict[int, Any], f: float, keepnorm: bool) -> Callable:
    """The stream at one position quantised to multiples of `step` and held over runs of `hold`
    frames, as vau/units/erosion.py's erosion_edit does it; frame 0 (the attention sink) untouched.
    Only the current run's first frame is kept in `held`."""
    st = None

    def edit(h, frame: int):
        nonlocal st
        mx = _mx()
        if frame < 1 or (step is None and hold <= 1):
            return h
        x = h.astype(mx.float32)
        if step is not None:
            if st is None:
                st = mx.array(step.astype(np.float32))
            x = mx.round(x / st) * st
        if hold > 1:
            first = 1 + ((frame - 1) // hold) * hold
            if frame == first:
                held.clear()
                held[first] = x
            elif first in held:
                x = held[first]
        return _mix(h.astype(mx.float32), x, f, keepnorm).astype(h.dtype)
    return mlx_fn(edit)


def _param(u, strip, key: str):
    p = u.param(key) or {}
    return strip.params.get(key, p.get("default"))


def port(u, strip, host, before: Dict[str, Dict], after: Dict[str, Dict]) -> Optional[Dict[str, Dict]]:
    """The edit tables with this strip's torch hooks (what `after` adds to `before`) replaced by
    MLX ones, or None to keep the torch path."""
    if u.id not in PORTED:
        return None
    s = _settings(strip)
    if s is None:
        return None
    out = {k: {kk: list(vv) for kk, vv in before[k].items()} for k in before}
    try:
        if u.id == "neuron":
            op = str(_param(u, strip, "op"))
            value = float(_param(u, strip, "value"))
            if (op == "scale" and value == 1) or (op == "add" and value == 0):
                return out                                    # neutral, as the unit leaves it
            ns = [int(x) for x in str(strip.params["n"]).replace(" ", "").split(",") if x != ""]
            n = host.depth() if s["env"] != "none" else None
            added = {k: v[len(before["ins"].get(k, [])):] for k, v in after["ins"].items() if len(v) > len(before["ins"].get(k, []))}
            for (block, kind), fns in added.items():
                f = _factor(s, block, n)
                out["ins"].setdefault((block, kind), []).extend(
                    [neuron_fn(ns, op, value, f, s["keepnorm"]) if f != 0.0 else mlx_fn(lambda h: h)] * len(fns))
            return out
        if u.id == "erosion":
            bits, hold = int(_param(u, strip, "bits")), max(1, int(_param(u, strip, "hold")))
            n = host.depth() + 1 if s["env"] != "none" else None
            added = {k: v[len(before["edits"].get(k, [])):] for k, v in after["edits"].items() if len(v) > len(before["edits"].get(k, []))}
            for layer, fns in added.items():
                step = None if bits >= 16 else np.asarray(host.stats()["std"][layer], np.float64) * (6.0 / 2 ** max(1, bits))
                f = _factor(s, layer, n)
                fn = erosion_fn(step, hold, host.memory(strip, ("mlx", layer)), f, s["keepnorm"]) if f != 0.0 else mlx_fn(lambda h, frame: h)
                out["edits"].setdefault(layer, []).extend([fn] * len(fns))
            return out
    except Exception:
        return None
    return None
