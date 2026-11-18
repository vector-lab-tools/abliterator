# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Modulation: sources that move the instrument's controls over time, and the routes that join a
source to a control with a depth.

Sources: three LFOs, three envelopes, two random sources, Steps, the Follower, Drift and Chaos, and four
macros. Every source but the Follower is a function of time from the start of play (frame f is at
f × 40 ms) alone, so live play and a replay of a recipe compute the same values at the same frame;
Chaos integrates its attractor once and reads it back (_chaos_track). The Follower follows the
loudness of the model's own output, frame by frame, from the frames written before the one it moves
(FollowTrack): live play, a render and the output section's re-render each build the track from the
same model output, so it replays too (exactly at Speed 1; a Follower moving an output or effect knob
while Speed is slowed is read in output frames and replays close to, not exactly as, it was heard).
LFOs, random sources, Steps, Drift and Chaos are bipolar (−1 to 1, times their Amount, which scales
every route from them at once); the envelopes, the Follower and the macros are unipolar (0 to 1).

Targets: the Steer doses (`dose:<direction>`), guidance and temperature (the model's
conditioning), the master's low-pass and volume (after the model), the knobs of the Model layers
units (`unit:<strip>:<key>`, the strip counted among those playing, from 0) and of the effect
slots (`pfx:<slot>:<key>`). A unit or effect knob is moved along its travel: depth is a share of
the knob's travel (log knobs travel in ratios), clamped to its ends. A route's depth (−1 to 1)
is a share of the target's span, so depth 0.5 on a bipolar source swings the control by a quarter
of its span either side of where its knob is set; the low-pass is moved in octaves (depth 1 is
four octaves). The result is clamped to the control's range (for doses, the full range).

The model makes a frame every 40 ms and a move is heard about 100 ms later, so useful rates are
slow: LFO rates run from 0.01 to 4 Hz.

A control with a calibrated curve (calibrate.py: knob travel laid out over its musical range) is
moved along the knob's travel, not along its value: the offset is a share of the travel, and the
result is read back through the curve, so modulation stays in the musical range and covers it as
evenly as the knob does. A setting outside the curve (the full range unlocked) is moved by value.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

FRAME_S = 0.04
SHAPES = ["sine", "triangle", "square", "saw", "ramp"]
LFOS = ["lfo1", "lfo2", "lfo3"]
RANDS = ["rand1", "rand2"]
MACROS = ["m1", "m2", "m3", "m4"]
ENVS = ["env", "env2", "env3"]                # "env" is envelope 1 (its id from before there were three)
STEPS, FOLLOWS, WILDS = ["steps"], ["follow"], ["drift", "chaos"]
SOURCES = LFOS + ENVS + RANDS + STEPS + FOLLOWS + WILDS + MACROS
BIPOLAR = set(LFOS + RANDS + STEPS + WILDS)
STEP_N = 16
STEP_DEFAULT = [1.0, -1.0, 0.5, -0.5, 0.75, -0.25, 0.25, -0.75, 1.0, 0.0, -0.5, 0.5, -1.0, 0.25, 0.75, -0.25]
CHAOS_SYSTEMS = ["lorenz", "rossler"]

# each source's settings: (default, low, high); a string default is a choice
PARAMS: Dict[str, Dict[str, Tuple]] = {
    "lfo": {"shape": ("sine", SHAPES), "rate_hz": (0.1, 0.01, 4.0), "amount": (1.0, 0.0, 1.0), "phase": (0.0, 0.0, 1.0),
            "symmetry": (0.5, 0.05, 0.95), "smooth": (0.0, 0.0, 1.0)},
    "env": {"attack_s": (4.0, 0.0, 60.0), "decay_s": (4.0, 0.0, 60.0), "sustain": (0.6, 0.0, 1.0), "release_s": (4.0, 0.0, 60.0),
            "loop": (0, 0, 1), "hold_s": (4.0, 0.0, 60.0)},
    "rand": {"rate_hz": (0.25, 0.01, 4.0), "amount": (1.0, 0.0, 1.0), "smooth": (0.5, 0.0, 1.0), "seed": (1, 0, 9999)},
    "macro": {"value": (0.0, 0.0, 1.0), "name": ("", None)},
    "steps": {"rate_hz": (2.0, 0.1, 16.0), "length": (8, 2, STEP_N), "glide": (0.0, 0.0, 1.0), "amount": (1.0, 0.0, 1.0),
              **{f"s{i}": (STEP_DEFAULT[i], -1.0, 1.0) for i in range(STEP_N)}},
    "follow": {"attack_s": (0.05, 0.01, 2.0), "release_s": (0.5, 0.04, 5.0), "floor_db": (-45.0, -70.0, -10.0), "amount": (1.0, 0.0, 1.0)},
    "drift": {"rate_hz": (0.05, 0.005, 1.0), "rough": (0.4, 0.0, 1.0), "amount": (1.0, 0.0, 1.0), "seed": (1, 0, 9999)},
    "chaos": {"system": ("lorenz", CHAOS_SYSTEMS), "rate_hz": (0.2, 0.01, 4.0), "amount": (1.0, 0.0, 1.0), "seed": (1, 0, 9999)},
}
TARGETS = {"guidance": (0.0, 8.0), "temperature": (0.5, 2.0), "lowpass_hz": (1000.0, 16000.0), "volume": (0.0, 1.0), "width": (0.0, 2.0), "spread": (0.0, 1.0)}
DOSE_SPAN = (-0.4, 0.4)
LP_OCTAVES = 4.0
PARAM_TGT = re.compile(r"(unit:\d{1,2}|pfx:[0-7]):[a-z][a-z0-9_]{0,23}")
QUANT = 512                                  # unit knobs move in 1/512ths of their travel: a re-attach only on a real change


def is_param(tgt: str) -> bool:
    return tgt.startswith(("unit:", "pfx:"))


CURVES: Dict[str, List[List[float]]] = {}    # target → [[travel 0..1, value], ...], from the calibration


def set_curves(curves: Dict[str, List[List[float]]]):
    CURVES.clear()
    CURVES.update({k: v for k, v in curves.items() if v and len(v) >= 2})


def _interp(x: float, xs: List[float], ys: List[float]) -> float:
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            a = (x - xs[i - 1]) / (xs[i] - xs[i - 1]) if xs[i] > xs[i - 1] else 0.0
            return ys[i - 1] + a * (ys[i] - ys[i - 1])
    return ys[-1]


def travel(tgt: str, value: float) -> Optional[float]:
    """The knob position of a value on the target's curve; None if it has none or the value is off it."""
    c = CURVES.get(tgt)
    if not c:
        return None
    ts, vs = [p[0] for p in c], [p[1] for p in c]
    if value < vs[0] - 1e-9 or value > vs[-1] + 1e-9:
        return None
    return _interp(value, vs, ts)


def along(tgt: str, t: float) -> float:
    c = CURVES[tgt]
    return _interp(min(1.0, max(0.0, t)), [p[0] for p in c], [p[1] for p in c])


def kind(src: str) -> str:
    return ("lfo" if src in LFOS else "rand" if src in RANDS else "macro" if src in MACROS else "steps" if src in STEPS
            else "follow" if src in FOLLOWS else src if src in WILDS else "env")


def target_range(tgt: str) -> Tuple[float, float]:
    return DOSE_SPAN if tgt.startswith("dose:") else TARGETS[tgt]


def clean(spec: Optional[Dict[str, Any]], directions: Optional[List[str]] = None) -> Dict[str, Any]:
    """A complete, checked spec: every source with every setting, and routes to known targets."""
    spec = spec or {}
    srcs: Dict[str, Dict[str, Any]] = {}
    for s in SOURCES:
        given = (spec.get("sources") or {}).get(s) or {}
        out = {}
        for k, d in PARAMS[kind(s)].items():
            v = given.get(k, d[0])
            if isinstance(d[0], str):
                out[k] = v if d[1] is None or v in d[1] else d[0]
                if d[1] is None:
                    out[k] = str(v)[:24]
            else:
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = d[0]
                v = min(d[2], max(d[1], v))
                out[k] = int(v) if isinstance(d[0], int) else v
        srcs[s] = out
    routes = []
    for r in spec.get("routes") or []:
        src, tgt = r.get("src"), str(r.get("tgt", ""))
        if src not in SOURCES:
            continue
        if not (tgt in TARGETS or (tgt.startswith("dose:") and (directions is None or tgt[5:] in directions)) or PARAM_TGT.fullmatch(tgt)):
            continue
        try:
            depth = max(-1.0, min(1.0, float(r.get("depth", 0))))
        except (TypeError, ValueError):
            continue
        routes = [x for x in routes if not (x["src"] == src and x["tgt"] == tgt)]
        if depth:
            routes.append({"src": src, "tgt": tgt, "depth": round(depth, 4)})
    return {"sources": srcs, "routes": routes}


def _hash01(seed: int, k: int) -> float:
    """A uniform value in [0, 1) from (seed, k), the same on every machine."""
    x = (seed * 0x9E3779B1 + k * 0x85EBCA77 + 0x165667B1) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x2C1B3C6D) & 0xFFFFFFFF
    x ^= x >> 12
    x = (x * 0x297A2D39) & 0xFFFFFFFF
    x ^= x >> 15
    return x / 4294967296.0


def lfo(p: Dict[str, Any], t: float) -> float:
    ph = (t * p["rate_hz"] + p["phase"]) % 1.0
    s = p["symmetry"]
    shape = p["shape"]
    if shape == "sine":                                   # a sine whose rise takes `symmetry` of the cycle
        u = ph / s * 0.5 if ph < s else 0.5 + (ph - s) / (1 - s) * 0.5
        v = -math.cos(2 * math.pi * u)
    elif shape == "triangle":
        v = -1 + 2 * ph / s if ph < s else 1 - 2 * (ph - s) / (1 - s)
    elif shape == "square":
        v = 1.0 if ph < s else -1.0
    elif shape == "saw":
        v = -1 + 2 * ph
    else:                                                 # ramp: falling
        v = 1 - 2 * ph
    if p["smooth"] > 0:                                   # toward a sine of the same rate and phase
        v = (1 - p["smooth"]) * v + p["smooth"] * -math.cos(2 * math.pi * ph)
    return v * p.get("amount", 1.0)                       # the amplitude: every route from this source scaled


def rand(p: Dict[str, Any], t: float) -> float:
    x = t * p["rate_hz"]
    k = int(math.floor(x))
    a, b = _hash01(int(p["seed"]), k) * 2 - 1, _hash01(int(p["seed"]), k + 1) * 2 - 1
    f = x - k
    e = 0.5 - 0.5 * math.cos(math.pi * f)                 # an eased glide to the next value
    glide = a + (b - a) * e
    return ((1 - p["smooth"]) * a + p["smooth"] * glide) * p.get("amount", 1.0)


def env(p: Dict[str, Any], t: float, length: Optional[float]) -> float:
    """Attack, Decay, then Sustain from ▶. With Loop on: Sustain for Hold, Release to zero, and round again."""
    a, d, s, r = p["attack_s"], p["decay_s"], p["sustain"], p["release_s"]
    if p.get("loop"):
        hold = p.get("hold_s", 4.0)
        cycle = a + d + hold + r
        if cycle <= 0:
            return s
        t = t % cycle
        if t >= a + d + hold:
            return s * (1 - (t - a - d - hold) / r)
        length = None
    if t < a:
        v = t / a if a > 0 else 1.0
    elif t < a + d:
        v = 1 - (1 - s) * ((t - a) / d if d > 0 else 1.0)
    else:
        v = s
    if length is not None and r > 0 and t > length - r:   # the release ends the sound (freeze only)
        v *= max(0.0, (length - t) / r)
    return v


def steps(p: Dict[str, Any], t: float) -> float:
    """A sequence of `length` values, one a step at `rate_hz`; Glide eases into the next value over that share of
    the step's end."""
    n = int(p["length"])
    x = t * p["rate_hz"]
    k = int(math.floor(x))
    a, b = p[f"s{k % n}"], p[f"s{(k + 1) % n}"]
    g, f = p["glide"], x - k
    if g > 0 and f > 1 - g:
        e = (f - (1 - g)) / g
        a = a + (b - a) * (0.5 - 0.5 * math.cos(math.pi * e))
    return a * p.get("amount", 1.0)


def _vnoise(seed: int, x: float) -> float:
    k = int(math.floor(x))
    f = x - k
    a, b = _hash01(seed, k) * 2 - 1, _hash01(seed, k + 1) * 2 - 1
    return a + (b - a) * (f * f * (3 - 2 * f))


def drift(p: Dict[str, Any], t: float) -> float:
    """A slow wander: smooth noise at the rate, with three faster layers at double the rate each, each weighted
    by Roughness to the power of its layer."""
    seed, x, r = int(p["seed"]), t * p["rate_hz"], p["rough"]
    tot = wsum = 0.0
    for o in range(4):
        w = r ** o
        tot += w * _vnoise(seed + 7919 * o, x * 2 ** o)
        wsum += w
    return max(-1.0, min(1.0, tot / wsum * 1.4)) * p.get("amount", 1.0)


import threading as _threading  # noqa: E402
_CH: Dict[Tuple[str, int], List[float]] = {}
_CH_LOCK = _threading.Lock()
CH_DT = 0.01                      # attractor time between stored samples; 1 Hz reads 1 unit of attractor time a second


def _chaos_track(system: str, seed: int, upto: int) -> List[float]:
    """The attractor's first coordinate, normalised to about ±1, integrated (RK4, two steps a sample) from a
    starting point set by the seed, past its first 2,000 samples, and kept: extended as play needs more."""
    with _CH_LOCK:
        tr = _CH.get((system, seed))
        if tr is None:
            tr = _CH[(system, seed)] = []
            st = [1.0 + 0.001 * seed, 1.0, 1.0 + 0.0007 * (seed % 97)]
            tr.append(st)                                   # the integrator's state rides at the front
            need = 2000
        else:
            need = 0
        st = tr[0]
        if system == "lorenz":
            f = lambda x, y, z: (10.0 * (y - x), x * (28.0 - z) - y, x * y - (8.0 / 3.0) * z)
            scale = 20.0
        else:
            f = lambda x, y, z: (-y - z, x + 0.2 * y, 0.2 + z * (x - 5.7))
            scale = 11.0
        h = CH_DT / 2
        while len(tr) - 1 < upto + 1 or need:
            for _ in range(2):
                x, y, z = st
                k1 = f(x, y, z)
                k2 = f(x + h / 2 * k1[0], y + h / 2 * k1[1], z + h / 2 * k1[2])
                k3 = f(x + h / 2 * k2[0], y + h / 2 * k2[1], z + h / 2 * k2[2])
                k4 = f(x + h * k3[0], y + h * k3[1], z + h * k3[2])
                st = [x + h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), y + h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
                      z + h / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])]
            if need:
                need -= 1
                continue
            tr.append(max(-1.0, min(1.0, st[0] / scale)))
        tr[0] = st
        return tr


def chaos(p: Dict[str, Any], t: float) -> float:
    """A chaotic attractor's path, read at the rate: smooth, never repeating, the same for the same seed."""
    x = t * p["rate_hz"] / CH_DT
    i = int(x)
    tr = _chaos_track(p["system"], int(p["seed"]), i + 2)
    a, b = tr[1 + i], tr[2 + i]
    return (a + (b - a) * (x - i)) * p.get("amount", 1.0)


class FollowTrack:
    """The Follower's input: the level of each frame of the model's output, in order (push), and, for each
    setting of the Follower, its value at each frame, worked out from the frames before it (at)."""

    def __init__(self):
        self.db: List[float] = []
        self.env: Dict[Tuple[float, float], List[float]] = {}
        self.lock = _threading.Lock()

    def push(self, frame) -> None:
        import numpy as np
        a = np.asarray(frame, np.float64)
        with self.lock:
            self.db.append(max(-100.0, 20 * math.log10(math.sqrt(float((a * a).mean())) + 1e-9)) if a.size else -100.0)

    def at(self, p: Dict[str, Any], f: int) -> float:
        key = (round(p["attack_s"], 4), round(p["release_s"], 4))
        up, down = math.exp(-FRAME_S / max(1e-3, key[0])), math.exp(-FRAME_S / max(1e-3, key[1]))
        with self.lock:
            e = self.env.get(key)
            if e is None:
                if len(self.env) >= 4:
                    self.env.pop(next(iter(self.env)))
                e = self.env[key] = [-100.0]                    # the follower before any frame: silence
            f = max(0, min(int(f), len(self.db)))
            while len(e) <= f:
                d, cur = self.db[len(e) - 1], e[-1]
                a = up if d > cur else down
                e.append(cur + (1 - a) * (d - cur))
            lvl = e[f]
        return max(0.0, min(1.0, (lvl - p["floor_db"]) / 30.0)) * p.get("amount", 1.0)


def value(spec: Dict[str, Any], src: str, t: float, length: Optional[float] = None, track: Optional[FollowTrack] = None) -> float:
    p = spec["sources"][src]
    k = kind(src)
    if k == "lfo":
        return lfo(p, t)
    if k == "rand":
        return rand(p, t)
    if k == "env":
        return env(p, t, length)
    if k == "steps":
        return steps(p, t)
    if k == "drift":
        return drift(p, t)
    if k == "chaos":
        return chaos(p, t)
    if k == "follow":
        return track.at(p, int(round(t / FRAME_S))) if track is not None else 0.0
    return float(p["value"])


def values(spec: Dict[str, Any], t: float, length: Optional[float] = None, track: Optional[FollowTrack] = None) -> Dict[str, float]:
    return {s: round(value(spec, s, t, length, track), 4) for s in SOURCES}


def active(spec: Dict[str, Any], model: Optional[bool] = None) -> bool:
    """Whether any route moves a model control (model=True), a master control (False), or any (None)."""
    for r in spec["routes"]:
        m = not (r["tgt"] in ("lowpass_hz", "volume", "width", "spread") or r["tgt"].startswith("pfx:"))
        if model is None or m == model:
            return True
    return False


def offsets(spec: Dict[str, Any], t: float, length: Optional[float] = None, curved: bool = False,
            track: Optional[FollowTrack] = None) -> Dict[str, float]:
    """Each routed target's offset, summed over its routes: in the target's units, except the
    low-pass, whose offset is in octaves; with `curved`, a target with a curve gets its offset as
    a share of the knob's travel."""
    out: Dict[str, float] = {}
    cache: Dict[str, float] = {}
    for r in spec["routes"]:
        tgt = r["tgt"]
        if is_param(tgt):
            continue
        v = cache.setdefault(r["src"], value(spec, r["src"], t, length, track))
        if tgt == "lowpass_hz":
            out[tgt] = out.get(tgt, 0.0) + r["depth"] * v * LP_OCTAVES
        elif curved and tgt in CURVES:
            out[tgt] = out.get(tgt, 0.0) + r["depth"] * v
        else:
            lo, hi = target_range(tgt)
            out[tgt] = out.get(tgt, 0.0) + r["depth"] * v * (hi - lo)
    return out


def _clamp(tgt: str, x: float) -> float:
    lo, hi = (-3.0, 3.0) if tgt.startswith("dose:") else target_range(tgt)     # doses: the full range
    return min(hi, max(lo, x))


def apply_model(spec: Dict[str, Any], doses: Dict[str, float], sampling: Dict[str, Any], t: float,
                length: Optional[float] = None, track: Optional[FollowTrack] = None) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """The doses and sampling at time t: the knobs' settings plus the routed sources."""
    off = offsets(spec, t, length, curved=True, track=track)

    def moved(tgt: str, base: float, digits: int) -> float:
        tr = travel(tgt, base)
        if tr is not None:
            return round(along(tgt, tr + off[tgt]), digits)
        plain = offsets({"routes": [r for r in spec["routes"] if r["tgt"] == tgt], "sources": spec["sources"]}, t, length, track=track)
        return round(_clamp(tgt, base + plain[tgt]), digits)
    d = dict(doses)
    for k in list(d):
        if f"dose:{k}" in off:
            d[k] = moved(f"dose:{k}", d[k], 5)
    s = dict(sampling)
    for k in ("guidance", "temperature"):
        if k in off:
            s[k] = moved(k, float(s[k]), 4)
    return d, s


def param_offsets(spec: Dict[str, Any], t: float, length: Optional[float] = None, prefix: str = "",
                  track: Optional[FollowTrack] = None) -> Dict[str, float]:
    """Each routed unit or effect knob's offset, as a share of its travel, summed over its routes."""
    out: Dict[str, float] = {}
    for r in spec["routes"]:
        if is_param(r["tgt"]) and r["tgt"].startswith(prefix):
            out[r["tgt"]] = out.get(r["tgt"], 0.0) + r["depth"] * value(spec, r["src"], t, length, track)
    return out


def move_knob(base: float, lo: float, hi: float, scale: str, off: float, quant: int = 0) -> float:
    """A knob's value moved by `off` of its travel, held to its ends; log knobs travel in ratios."""
    if hi <= lo:
        return base
    if quant:
        off = round(off * quant) / quant             # the offset in steps, so the knob's own setting is kept exactly
    if not off:
        return base
    log = scale == "log" and lo > 0
    pos = (math.log(max(base, lo) / lo) / math.log(hi / lo)) if log else (base - lo) / (hi - lo)
    pos = min(1.0, max(0.0, pos + off))
    v = lo * (hi / lo) ** pos if log else lo + pos * (hi - lo)
    return float(round(v)) if scale == "int" else v


def has_units(spec: Dict[str, Any]) -> bool:
    return any(r["tgt"].startswith("unit:") for r in spec["routes"])


def apply_units(spec: Dict[str, Any], lines: List[str], t: float, length: Optional[float], info,
                track: Optional[FollowTrack] = None) -> List[str]:
    """The strips with their routed knobs moved. info(kind, key) gives (low, high, scale, default) for a
    numeric knob, or None. Values move in 1/QUANT of the travel, so the lines change only when a move
    is large enough to matter."""
    off = param_offsets(spec, t, length, "unit:", track)
    if not off:
        return list(lines)
    out = list(lines)
    for tgt, o in off.items():
        _, i, key = tgt.split(":")
        i = int(i)
        if i >= len(out):
            continue
        bits = out[i].split()
        if not bits:
            continue
        q = info(bits[0], key)
        if q is None:
            continue
        lo, hi, scale, default = q
        params = dict(b.partition("=")[::2] for b in bits[1:])
        try:
            base = float(params.get(key, default))
        except (TypeError, ValueError):
            continue
        v = move_knob(base, float(lo), float(hi), scale, o, QUANT)
        params[key] = str(int(v)) if scale == "int" else f"{v:.5g}"
        out[i] = " ".join([bits[0]] + [f"{k}={x}" for k, x in params.items()])
    return out


def apply_master(spec: Dict[str, Any], master: Dict[str, Any], t: float, length: Optional[float] = None,
                 track: Optional[FollowTrack] = None) -> Dict[str, Any]:
    off = offsets(spec, t, length, track=track)
    m = dict(master)
    poff = param_offsets(spec, t, length, "pfx:", track)
    if poff and m.get("fx"):
        from abliterator import postfx
        fxs = [{**sl, "params": dict(sl.get("params") or {})} for sl in m["fx"]]
        for tgt, o in poff.items():
            _, i, key = tgt.split(":")
            i = int(i)
            if i >= len(fxs):
                continue
            q = next((p for p in postfx.TYPES.get(fxs[i].get("type"), {}).get("params", []) if p[0] == key), None)
            if q is None or q[5] not in ("lin", "log", "int"):
                continue
            _, _, lo, hi, d, scale, _u = q
            fxs[i]["params"][key] = move_knob(float(fxs[i]["params"].get(key, d)), lo, hi, scale, o)
        m["fx"] = fxs
    if "lowpass_hz" in off:
        m["lowpass_hz"] = round(_clamp("lowpass_hz", float(m["lowpass_hz"]) * 2 ** off["lowpass_hz"]), 1)
    for k in ("volume", "width", "spread"):
        if k in off:
            m[k] = round(_clamp(k, float(m.get(k, 1.0 if k == "width" else 0.0)) + off[k]), 4)
    return m
