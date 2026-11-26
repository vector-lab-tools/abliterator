# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Magenta RT 2 as a host for Vector Abliteration Units (VAU 1, the package `vau`, from
https://github.com/vector-lab-tools/vector-abliteration-units, pinned in requirements-mrt2.txt to the
commit Abliterton Lens pins).

The units are VAU's own files, run unchanged. Their edits are written in torch; the model runs
in MLX. Where mlxunits has no MLX port of a strip's edit, the stream for the frame being made (one
position of 1024 values) is copied out of MLX at that layer, passed through the edit in torch on
the CPU, and copied back, which forces MLX to compute the frame up to that layer first (cost
measured in bench/units_cost.py).

What this host offers (VAU capabilities): `stream` (the output of each of the temporal
transformer's 12 blocks, numbered as hidden states 1 to 12), `output` (what each block's
self-attention and MLP add to the stream: the output of the post-norm that ends each branch,
g ⊙ (W x + b) / rms; the component's weight is W with g folded into its rows, so a unit that
projects W, as Abliterate's Keep size does, matches the hook; the bias stays inside the norm, so
Abliterate's "Bias too" has no separate effect here; the cross-attention to the style is not
offered), `ordered` (positions are 40 ms frames from 0 at the start of play), `stats` and
`pca` (of the stream over the reference set, the presets' and contrast sets' prompts, measured once
on the unchanged model and cached in data/stats/), `directions` (data/directions/mrt2_small/) and
`contrasts`. Not offered: `weights`, `logits`, `texts`, so units that need them (Morph) are listed
as unavailable.

Latency: every stream unit is applied from the frame after the move (class live). A strip set by
hand is attached again and its memory (Delay's line, Erosion's held values) starts again; a move by
modulation keeps the memory (attach's `store`).
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import vau
from vau import collection
from vau.manifest import space as vau_space

from . import DATA, SHIPPED
from . import directions as dr

CAPABILITIES = {"stream", "output", "ordered", "stats", "pca", "directions", "contrasts"}
STATS = DATA / "stats"
LATENCY = {"class": "live", "ms": 0, "invalidates_cache": False,
           "note": "heard from the next frame; a move re-attaches the strip and resets its memory"}

_registry: Optional[vau.Registry] = None

# This host's titles in place of VAU's (SPEC 5d): here positions are 40 ms frames. Strip lines keep VAU's ids.
NAMES = {"beatrepeat": "Frame Repeat"}
# characters a strip's top bar holds; VAU caps its short names at 14 too (SPEC 2)
NAME_MAX = 14


# Defaults for units that do nothing or cannot run at VAU's own: Abliterate names no direction (and
# then does nothing, silently), Neuron no neuron, Rotate needs two different directions, Principal
# Bands and Quantise start neutral.
HOST_DEFAULTS = {
    "abliterate": {"dir": "brightness", "shape": "flat", "size": 0},
    "neuron": {"n": "373", "op": "scale", "value": 6},
    "shifter": {"dir": "brightness", "dir2": "pulse", "angle": 45},
    "eq": {"low": 1.3, "high": 0.7},
    "erosion": {"bits": 6},
}


# Ranges this host opens beyond VAU's own. Abliterate's Strength: VAU stops at 1.5 but its unit reads the value
# unclamped; strengths 1 to 2 and 4 to 6 held on Deep techno and the Warm drone (2026-11-18).
HOST_RANGES = {
    "abliterate": {"strength": {"max": 6.0, "taper": [[0.0, 0.0], [0.15, 0.5], [0.35, 1.0], [0.5, 2.0], [0.75, 4.0], [1.0, 6.0]]}},
}


# Presets on a unit's strip: each sets the settings it names and leaves the rest, the direction included.
# `block` is the placement ("all" or a block).
HOST_PRESETS = {
    "abliterate": [
        {"name": "Remove · 1", "block": "all", "params": {"shape": "flat", "size": 0, "strength": 1.0},
         "tip": "Takes the direction out at every block: the sound moves to the middle of that axis"},
        {"name": "Mirror · 2", "block": "all", "params": {"shape": "flat", "size": 0, "strength": 2.0},
         "tip": "Turns the direction round at every block: the sound goes as far the other way as it was"},
        {"name": "Beyond · 4", "block": "all", "params": {"shape": "flat", "size": 0, "strength": 4.0},
         "tip": "Three times as far the other way; on Deep techno, noisier and sharper in its attacks"},
        {"name": "Far side · 6", "block": "all", "params": {"shape": "flat", "size": 0, "strength": 6.0},
         "tip": "Five times as far the other way, the most this panel allows"},
        {"name": "Late blocks · VAU", "block": "all", "params": {"shape": "linear", "peak": 0.7, "width": 0.4, "edge": 0.3, "share": 0.5,
                                                                  "projected": 1, "size": 1, "bias": 1, "strength": 1.0},
         "tip": "VAU's own settings, as language-model abliteration uses them: strongest around block 8, sizes kept; gentle here"},
    ],
}


# The other units' presets, from the unit review (2026-11-23: searched on six seeds, checked on ten others), with the
# figures each was chosen on.
PRESETS_FILE = SHIPPED / "presets" / "mrt2_small.json"


def _file_presets() -> Dict[str, List[Dict[str, Any]]]:
    try:
        import json
        units = json.loads(PRESETS_FILE.read_text())["units"]
    except (OSError, ValueError, KeyError):
        return {}
    return {u: [{k: q[k] for k in ("name", "block", "params", "tip") if k in q} for q in qs] for u, qs in units.items()}


HOST_PRESETS = {**_file_presets(), **HOST_PRESETS}


def _preset(uid: str, name: str) -> Dict[str, Any]:
    return next((q["params"] for q in HOST_PRESETS.get(uid, []) if q["name"] == name), {})


# Defaults from the unit review, applied over the calibration's: at their old defaults these units changed no measure of
# the sound consistently across twenty seeds. Each is one of the unit's presets; HOST_PLACE is its placement.
HOST_HEARD = {k: v for k, v in {
    "delay": _preset("delay", "Feedback Loop"),
    "reverb": _preset("reverb", "Long Tail"),
    "limiter": _preset("limiter", "Lid"),
    "neuron": _preset("neuron", "Ten Set"),
    "erosion": _preset("erosion", "Bitcrush"),
    "multiband": _preset("multiband", "Low Squeeze"),
}.items() if v}                                       # a preset that failed the held-out check is not in the file
HOST_PLACE = {k: v for k, v in {"neuron": "10", "limiter": 10}.items() if k in HOST_HEARD}

# VAU host settings shown on a unit's card. Keep size on Lookback and Running Average: with it off, both fell near silent
# on some held-out seeds.
KEEP_SIZE = {"key": "keepnorm", "label": "Keep size", "kind": "toggle", "default": 0, "host": True,
             "tip": "h' ← h' · ‖h‖ / ‖h'‖ at each position: after the change, the stream is scaled back to the size it had"}
HOST_EXTRA = {"delay": [KEEP_SIZE], "reverb": [KEEP_SIZE]}


# What a unit does differently on this instrument, for its help box beside VAU's
HOST_NOTES = {
    "abliterate": "Strength runs to 6 here; VAU stops at 1.5. Up to 1 the direction is taken out, from 1 to 2 it is turned round, "
                  "and past 2 the sound is pushed further to the far side. Taking a direction out pulls a sound toward the middle of "
                  "that direction, so a sound far along it changes most: on Deep techno, Pulse at strength 1 took the centroid from 69 Hz "
                  "to 902 Hz, while on the Warm drone, which sits a little to the dark side of Brightness, Brightness at 1 raised it from "
                  "138 to 191 Hz. Almost all of the writing along Brightness and Pulse is done in block 0 and blocks 9 to 11, so on a "
                  "single block in between the unit changes little; it starts on all blocks, flat, with Keep size off. It runs in MLX, on "
                  "the GPU, so all twelve blocks cost no more than none.",
}


CALIBRATION = DATA / "calibration" / "mrt2_small.json"
_cal: Dict[str, Any] = {"mtime": None, "controls": {}}


def calibrated() -> Dict[str, Any]:
    """The units' entries in the calibration (calibrate.py), reread when the file changes."""
    try:
        mt = CALIBRATION.stat().st_mtime
    except OSError:
        return {}
    if mt != _cal["mtime"]:
        import json
        c = json.loads(CALIBRATION.read_text()).get("controls", {})
        _cal.update(mtime=mt, controls={k: v for k, v in c.items() if k.startswith("unit:")})
    return _cal["controls"]


# Settings the card hides because they cannot change the sound here (Bias too: see the module docstring); the line
# keeps their value.
HOST_HIDDEN = {"abliterate": {"bias"}}
# Host flags a manifest's `when: {"host": [...]}` can name; a setting that needs one this host lacks is hidden.
HOST_FLAGS = {"sink": False}


def _when(q: Dict[str, Any], qs: List[Dict[str, Any]]):
    """The manifest's `when` as the card reads it, {key, in, why} (the card dims the setting while the other setting is
    outside `in`), or None; and `hide`, True where it names a host flag this host lacks."""
    w = q.get("when")
    if not isinstance(w, dict):
        return w, False
    hide = any(not HOST_FLAGS.get(f, False) for f in w.get("host", []))
    for k, vals in w.items():
        if k == "host":
            continue
        other = next((x for x in qs if x.get("key") == k), {})
        rest = [c for c in (other.get("choices") or []) if c not in vals]
        why = f"no effect while {other.get('label', k)} is {' or '.join(map(str, rest))}" if rest else f"acts only while {other.get('label', k)} is {' or '.join(map(str, vals))}"
        return {"key": k, "in": list(vals), "why": why}, hide
    return None, hide


def params(u) -> List[Dict[str, Any]]:
    """The unit's settings with this host's defaults: measured ones where the calibration has them
    (a setting whose own default breaks the sound gets the nearest that holds), else HOST_DEFAULTS."""
    over = HOST_DEFAULTS.get(u.id, {})
    cal = calibrated()
    out = []
    qs = list(u.m["params"]) + HOST_EXTRA.get(u.id, [])
    for q in qs:
        q = {**q, "default": over[q["key"]]} if q["key"] in over else dict(q)
        q.update(HOST_RANGES.get(u.id, {}).get(q["key"], {}))
        c = cal.get(f"unit:{u.id}:{q['key']}")
        if "taper" in q:
            pass                                       # this host's own range and travel replace the calibration's
        elif c and c.get("safe"):
            q.update(default=c["default"], safe=c["safe"], measured=True, fragile=c["safe"][0] == c["safe"][1])
        elif c and c.get("broken_everywhere"):
            q.update(fragile=True)
        if q["key"] in HOST_HEARD.get(u.id, {}):
            q["default"] = HOST_HEARD[u.id][q["key"]]
            sf = q.get("safe")                         # measured at the old default, so it may not hold the new one
            if sf and isinstance(q["default"], (int, float)) and not sf[0] <= q["default"] <= sf[1]:
                q.pop("safe"); q.pop("measured", None); q["fragile"] = False
        w, hide = _when(q, qs)
        if w is None:
            q.pop("when", None)
        else:
            q["when"] = w
        if hide or q["key"] in HOST_HIDDEN.get(u.id, ()):
            q["hidden"] = True
        out.append(q)
    return out


def placement(u) -> Dict[str, Any]:
    c = calibrated().get(f"unit:{u.id}:at")
    if not c:
        return {}
    pd = HOST_PLACE.get(u.id, c["default"])
    return {"places": c["safe_places"], "place_default": int(pd) if isinstance(pd, str) and pd.isdigit() else pd}


def title(u) -> str:
    return NAMES.get(u.id, u.m["title"])


def short(u) -> str:
    """The name for a strip's top bar (at most NAME_MAX characters)."""
    return NAMES.get(u.id) or u.m.get("short") or u.m["title"]


def registry() -> vau.Registry:
    global _registry
    if _registry is None:
        _registry = vau.Registry().load()
    return _registry


ELEMENTWISE = {"saturator", "erosion", "gate", "limiter"}
ELEMENTWISE_NOTE = ("Acts on each of the stream's 1,024 coordinates separately, so what it does depends on the "
                    "coordinates the stream is written in. In the measurements these units broke the sound no more "
                    "often than the others.")


_VG: Dict[str, Dict[str, Any]] = {}


def _vau_groups() -> Dict[str, Dict[str, Any]]:
    """Unit id -> its group in VAU's Core Collection look (vau/look/core.json), or {} if it cannot be read."""
    if not _VG:
        try:
            import json
            from pathlib import Path as _P
            d = json.loads((_P(vau.__file__).parent / "look" / "core.json").read_text())
            _VG.update({uid: {"id": g["id"], "title": g["title"]} for g in d.get("groups", []) for uid in g.get("units", [])})
        except Exception:
            _VG["_none"] = {}
    return {k: v for k, v in _VG.items() if k != "_none"}


def space_of(u) -> Dict[str, Any]:
    """`space` (the manifest's own, or its site's), VAU's `axis`, and the unit's group in VAU's look (None if
    VAU gives it none; the page then files it under Other)."""
    space = vau_space(u.m)
    g = _vau_groups().get(u.id, {})
    return {"space": space, "axis": u.m.get("axis", "any"), "group": g.get("id"), "group_title": g.get("title"),
            "note": ELEMENTWISE_NOTE if u.id in ELEMENTWISE and space == "stream" else None}


def vau_name(u) -> str:
    """VAU's own name for a unit: its collection's prefix before the manifest's title (vLookback)."""
    return collection.name(u)


def catalogue() -> List[Dict[str, Any]]:
    out = []
    for u in registry():
        missing = u.missing(CAPABILITIES)
        if u.m["category"] == "meter":
            continue                                            # meters need a host view this engine has not got yet
        out.append({"id": u.id, "title": title(u), "short": short(u), "vau_title": vau_name(u), "renamed": u.id in NAMES, "what": u.m["what"], "category": u.m["category"],
                    "site": u.m["site"], **space_of(u), "params": params(u), **placement(u), "presets": HOST_PRESETS.get(u.id, []), "host_note": HOST_NOTES.get(u.id), "example": u.m.get("example"),
                    "available": not missing, "missing": missing, "latency": LATENCY if not missing else None,
                    "help": u.m.get("help"), "inspired_by": u.m.get("inspired_by"), "version": u.m.get("version"), "author": u.m.get("author"),
                    "file": u.path.name if getattr(u, "path", None) else None, "explain": _has_explainer(u)})
    return out


def _has_explainer(u) -> bool:
    try:
        from vau import explain as vx
        return vx.explainer(u) is not None
    except Exception:
        return False


def explain_files() -> Dict[str, str]:
    """VAU's explainer files a page may load (its runtime, styles and each unit's figure), by name."""
    from vau import explain as vx
    return {k: str(v) for k, v in vx.files(registry()).items()}


class Direction:
    """A direction file as VAU units read it: a vector per layer (hidden-state numbering)."""

    def __init__(self, d: dr.Direction):
        self.d = d
        self.data = d.data

    def raw(self, layer: int) -> np.ndarray:
        return self.d.layers[layer]

    def vector(self, layer: int, device) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.d.layers[layer], np.float32).copy()).to(device)

    def unit(self, layer: int, device) -> torch.Tensor:
        v = self.vector(layer, device)
        return v / v.norm().clamp_min(1e-12)


class Line:
    """A strip as written: `kind target=value key=value ...`."""

    def __init__(self, text: str, lineno: int):
        bits = text.split()
        self.text = text.strip()
        self.kind = bits[0] if bits else ""
        self.params: Dict[str, str] = {}
        for b in bits[1:]:
            k, _, v = b.partition("=")
            self.params[k] = v
        self.lineno = lineno

    def target(self) -> Tuple[str, Any]:
        for t in ("head", "attn", "mlp", "block", "layer"):
            if t in self.params:
                return t, self.params[t]
        return "", None


class StreamStats:
    """Mean, spread and size of the stream at every layer, and its principal directions, over the
    reference set, on the unchanged model. Made by Proc.measure_stream; read from data/stats/."""

    def __init__(self, size: str):
        self.size = size
        self.file = STATS / f"{size}.npz"
        self._d = None
        self._pca: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    def exists(self) -> bool:
        return self.file.exists()

    def data(self):
        if self._d is None:
            self._d = dict(np.load(self.file))
        return self._d

    def stats(self) -> Dict[str, np.ndarray]:
        d = self.data()
        return {"mean": d["mean"], "std": d["std"], "norm": d["norm"], "n": d["n"]}

    def pca(self, layer: int) -> Tuple[np.ndarray, np.ndarray]:
        if layer not in self._pca:
            xs = torch.from_numpy(self.data()["samples"][:, layer].astype(np.float32))
            xs = xs - xs.mean(0, keepdim=True)
            _, s, vt = torch.linalg.svd(xs, full_matrices=True)
            sc = np.zeros(vt.shape[0], np.float32)
            sc[:len(s)] = (s / np.sqrt(xs.shape[0])).numpy()
            self._pca[layer] = (vt.numpy(), sc)
        return self._pca[layer]

    @staticmethod
    def save(size: str, samples: np.ndarray, prompts: List[str], frames: int, skip: int):
        """samples: (n, layers, d) float32."""
        STATS.mkdir(parents=True, exist_ok=True)
        x = samples.astype(np.float64)
        np.savez(STATS / f"{size}.npz", mean=x.mean(0).astype(np.float32), std=x.std(0).astype(np.float32),
                 norm=np.linalg.norm(x, axis=-1).mean(0).astype(np.float32), n=np.array(len(samples)),
                 samples=samples.astype(np.float16), prompts=np.array(prompts), frames=np.array(frames), skip=np.array(skip),
                 made=np.array(time.strftime("%Y-%m-%d %H:%M")))


class AudioHost(vau.Host):
    name = "Abliterator (Magenta RT 2)"

    def __init__(self, n_layers: int, stats: StreamStats, directions: Dict[str, dr.Direction], frame: Callable[[], int],
                 components: Dict[Tuple[int, str], Any]):
        self.n_layers, self.st, self.dirs, self.frame, self.comps = n_layers, stats, directions, frame, components
        self.edits: Dict[int, List[Callable]] = {}
        self.outs: Dict[Tuple[int, str], List[Callable]] = {}
        self.ins: Dict[Tuple[int, str], List[Callable]] = {}
        self.store: Dict[Any, Dict] = {}

    def capabilities(self):
        return set(CAPABILITIES)

    def sink(self) -> bool:
        """Magenta RT 2 has no attention sink: its first position is the take's first 40 ms frame, which units bend
        like any other."""
        return HOST_FLAGS["sink"]

    def error(self, lineno, msg):
        return vau.UnitError(f"Strip {lineno}: {msg}")

    @property
    def device(self):
        return torch.device("cpu")

    def layers(self, strip) -> List[int]:
        t, v = strip.target()
        if t != "layer" or not v:
            raise self.error(strip.lineno, f"give a layer from 1 to {self.n_layers}, as layer=6 or layer=4-8.")
        out = []
        for part in str(v).split(","):
            a, _, b = part.partition("-")
            try:
                lo, hi = int(a), int(b or a)
            except ValueError:
                raise self.error(strip.lineno, f"'{v}' is not a layer.")
            out.extend(range(lo, hi + 1))
        if not out or min(out) < 1 or max(out) > self.n_layers:
            raise self.error(strip.lineno, f"layers run from 1 to {self.n_layers} on this model.")
        return out

    def depth(self) -> int:
        return self.n_layers

    def blocks(self, strip):
        """Blocks from 0, as `block=3`, `block=2-5`, `block=all`."""
        t, v = strip.target()
        if t != "block" or not v:
            raise self.error(strip.lineno, f"give a block from 0 to {self.n_layers - 1}, as block=3, block=2-5 or block=all.")
        if str(v) == "all":
            return list(range(self.n_layers))
        out = []
        for part in str(v).split(","):
            a, _, b = part.partition("-")
            try:
                lo, hi = int(a), int(b or a)
            except ValueError:
                raise self.error(strip.lineno, f"'{v}' is not a block.")
            out.extend(range(lo, hi + 1))
        if not out or min(out) < 0 or max(out) >= self.n_layers:
            raise self.error(strip.lineno, f"blocks run from 0 to {self.n_layers - 1} on this model.")
        return out

    def component(self, kind, layer):
        if kind not in ("attn", "mlp"):
            raise ValueError(kind)
        return self.comps[(layer, kind)]

    def component_path(self, kind, layer):
        return self.comps[(layer, kind)].name

    def _key(self, module):
        for k, c in self.comps.items():
            if c is module:
                return k
        raise ValueError("not a component of this model")

    def hook_output(self, strip, module, fn):
        self.outs.setdefault(self._key(module), []).append(fn)

    def hook_input(self, strip, module, fn):
        self.ins.setdefault(self._key(module), []).append(fn)

    def on_restore(self, fn):
        pass                        # nothing is changed in place: every hook goes with the chain

    def stats(self):
        return self.st.stats()

    def pca(self, layer):
        return self.st.pca(layer)[0]

    def pca_scales(self, layer):
        return self.st.pca(layer)[1]

    def direction(self, name, lineno):
        if name not in self.dirs:
            raise self.error(lineno, f"no direction '{name}' on this engine; there are: {', '.join(self.dirs)}.")
        return Direction(self.dirs[name])

    def direction_texts(self, name, lineno):
        src = self.direction(name, lineno).data.get("source", {})
        return {"positive": list(src.get("positive", [])), "negative": list(src.get("negative", []))}

    def memory(self, strip, layer):
        return self.store.setdefault((strip.lineno, layer), {})

    def positions(self, n, device):
        return torch.arange(self.frame() - n + 1, self.frame() + 1, device=device)

    def prompt_length(self):
        return 1                    # so that the window's step 1 is frame 0

    def hook_stream(self, strip, make):
        for layer in self.layers(strip):
            self.edits.setdefault(layer, []).append(make(layer))


def param_info(kind: str, key: str):
    """A numeric knob of a unit, for modulation: (low, high, scale, default), or None."""
    u = registry().get(kind)
    p = u.param(key) if u else None
    if p:
        p = {**p, **HOST_RANGES.get(kind, {}).get(key, {})}
    if not p or p.get("kind") not in ("knob", "fader", "int") or "min" not in p or "max" not in p:
        return None
    return float(p["min"]), float(p["max"]), "int" if p.get("kind") == "int" else "lin", p.get("default", p["min"])


class Tagged:
    """A strip's hook or edit with the strip's line number, so the engine can say how much each strip changes
    the stream (mrt2_proc.Hooks.act)."""
    def __init__(self, fn: Callable, strip: int, unit: str = ""):
        self.fn, self.strip, self.unit, self.mlx = fn, strip, unit, getattr(fn, "mlx", False)

    def __call__(self, *a, **k):
        return self.fn(*a, **k)


def _tag(host, mark, lineno: int, unit: str = ""):
    """Wrap the hooks and edits this strip added (those not in the tables before it) with its line number."""
    for table, before in ((host.edits, mark[0]), (host.outs, mark[1]), (host.ins, mark[2])):
        for key, fns in table.items():
            old = {id(f) for f in before.get(key, [])}
            table[key] = [Tagged(f, lineno, unit) if id(f) not in old and not isinstance(f, Tagged) else f for f in fns]


def attach(lines: List[str], host_args, strict: bool = True, store: Optional[Dict[Any, Dict]] = None,
           native: bool = False) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Read, check and attach a chain. Returns the edits by layer and a report per strip.
    strict: raises vau.UnitError with the strip's number for a line that cannot work. Otherwise a
    line that cannot work is left out, with its error in its report, and the rest is attached
    (live play, so that one bad strip does not silence the others' moves)."""
    host = AudioHost(*host_args)
    if store is not None:
        host.store = store                 # modulation moves a knob without clearing the units' memory
    report = []
    for i, text in enumerate(l for l in lines if l.strip() and not l.strip().startswith("#")):
        s = Line(text, i + 1)
        u = registry().get(s.kind)
        try:
            if u is None:
                raise host.error(s.lineno, f"no unit '{s.kind}'.")
            missing = u.missing(CAPABILITIES)
            if missing:
                raise host.error(s.lineno, f"{title(u)} needs {', '.join(missing)}, which this engine does not offer yet.")
            mark = (dict(host.edits), dict(host.outs), dict(host.ins))
            mark = ({k: list(v) for k, v in mark[0].items()}, {k: list(v) for k, v in mark[1].items()}, {k: list(v) for k, v in mark[2].items()})
            try:
                vau.validate(u, s)
                vau.prepare(u, s, host)
                vau.attach(u, s, host)
            except Exception:
                host.edits, host.outs, host.ins = mark
                raise
            if native:                                            # the strip's edits in MLX where mlxunits has a port
                from . import mlxunits
                p = mlxunits.port(u, s, host, {"edits": mark[0], "outs": mark[1], "ins": mark[2]},
                                  {"edits": host.edits, "outs": host.outs, "ins": host.ins})
                if p is not None:
                    host.edits, host.outs, host.ins = p["edits"], p["outs"], p["ins"]
        except Exception as e:
            if strict:
                raise
            report.append({"lineno": s.lineno, "line": s.text, "unit": s.kind, "title": title(u) if u else s.kind, "error": str(e)})
            continue
        _tag(host, mark, s.lineno, u.id)
        where = host.layers(s) if s.target()[0] == "layer" else [f"block {b}" for b in host.blocks(s)]
        report.append({"lineno": s.lineno, "line": s.text, "unit": u.id, "title": title(u), "short": short(u), "vau_title": vau_name(u), "renamed": u.id in NAMES, "at": where, "latency": LATENCY})
    return {"edits": host.edits, "outs": host.outs, "ins": host.ins, "store": host.store}, report
