# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Magenta RT 2 as a host for Vector Abliteration Units (VAU 1, the package `vau`, from
https://github.com/vector-lab-tools/vector-abliteration-units, pinned as Abliterton Lens pins it).

The units are VAU's own files, run unchanged. Their edits are written in torch; the model runs
in MLX. At a layer that carries a strip, the stream for the frame being made (one position of
1024 values) is copied out of MLX, passed through the strips' edits in torch on the CPU, and
copied back. That forces MLX to compute the frame up to that layer before going on, which is
the bridge's cost, measured in bench/units_cost.py.

What this host offers (VAU capabilities): `stream` (the output of each of the temporal
transformer's 12 blocks, numbered as hidden states 1 to 12), `output` (what each block's
self-attention and MLP add to the stream: the output of the post-norm that ends each branch,
g ⊙ (W x + b) / rms; the component's weight is W with g folded into its rows, so a unit that
projects W, as Abliterate's Keep size does, matches the hook; the bias stays inside the norm, so
Abliterate's "Bias too" has no separate effect here; the cross-attention to the style is not
offered), `ordered` (positions are frames,
40 ms apart, from 0 at the start of play), `stats` and `pca` (of the stream over the reference
set: the presets' and contrast sets' prompts, measured once on the unchanged model and cached
in data/stats/), `directions` (this engine's directions, data/directions/mrt2_small/) and
`contrasts`. Not offered yet: `weights`, `logits`, `texts` (so Morph, which fits on captured
inputs, is unavailable). Units that need them are listed as unavailable.

Latency: every stream unit here is applied from the frame after the move (class live); a move
re-attaches the strip, so a unit's memory (Delay's line, Erosion's held values) starts again.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import vau

from . import ROOT
from . import directions as dr

CAPABILITIES = {"stream", "output", "ordered", "stats", "pca", "directions", "contrasts"}
STATS = ROOT / "data" / "stats"
LATENCY = {"class": "live", "ms": 0, "invalidates_cache": False,
           "note": "heard from the next frame; a move re-attaches the strip and resets its memory"}

_registry: Optional[vau.Registry] = None

# A unit is shown by VAU's own title (VAU 0.2.3 names the Core units for what they do to the
# vectors) except where this host gives it a display title of its own (SPEC 5d): VAU's Position
# Repeat is Frame Repeat here, since its positions are 40 ms frames of sound. Strip lines keep
# VAU's ids (`beatrepeat layer=6 ...`), so recipes are unchanged.
NAMES = {"beatrepeat": "Frame Repeat"}
# A strip's top bar holds about this many characters of a name; VAU caps its short names at 14
# too (SPEC 2). VAU's own name for a unit is given in its help, not on the strip.
NAME_MAX = 14


# Defaults on this host for units that do nothing (or cannot run) at VAU's own: Abliterate names
# no direction (and then does nothing, silently), Neuron names no
# neuron, Rotate needs two different directions, Principal Bands and Quantise are neutral. These
# are starting points to be calibrated by ear like the rest.
HOST_DEFAULTS = {
    "abliterate": {"dir": "brightness"},
    "neuron": {"n": "373", "op": "scale", "value": 6},
    "shifter": {"dir": "brightness", "dir2": "pulse", "angle": 45},
    "eq": {"low": 1.3, "high": 0.7},
    "erosion": {"bits": 6},
}


CALIBRATION = ROOT / "data" / "calibration" / "mrt2_small.json"
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


def params(u) -> List[Dict[str, Any]]:
    """The unit's settings with this host's defaults: measured ones where the calibration has them
    (a setting whose own default breaks the sound gets the nearest that holds), else HOST_DEFAULTS."""
    over = HOST_DEFAULTS.get(u.id, {})
    cal = calibrated()
    out = []
    for q in u.m["params"]:
        q = {**q, "default": over[q["key"]]} if q["key"] in over else dict(q)
        c = cal.get(f"unit:{u.id}:{q['key']}")
        if c and c.get("safe"):
            q.update(default=c["default"], safe=c["safe"], measured=True, fragile=c["safe"][0] == c["safe"][1])
        elif c and c.get("broken_everywhere"):
            q.update(fragile=True)
        out.append(q)
    return out


def placement(u) -> Dict[str, Any]:
    c = calibrated().get(f"unit:{u.id}:at")
    if not c:
        return {}
    return {"places": c["safe_places"], "place_default": c["default"]}


def title(u) -> str:
    return NAMES.get(u.id, u.m["title"])


def short(u) -> str:
    """The name for tight spaces (a strip's top bar): this host's own title, else VAU's short one
    (Comp Clamp for Component Clamp, VAU 0.2.6), else the title."""
    return NAMES.get(u.id) or u.m.get("short") or u.m["title"]


def registry() -> vau.Registry:
    global _registry
    if _registry is None:
        _registry = vau.Registry().load()
    return _registry


# The space each unit acts on (the proposal now with VAU: a manifest may declare `space`; until it
# does, this host's own reading of each unit), and which units act along time. VAU's own `axis`
# (any | ordered) says whether a unit needs positions in order, and is reported as it is. Groups
# are what the add menu and the units page show.
SPACE = {"abliterate": "component", "neuron": "hidden"}             # the rest act on the stream at a layer
ALONG_TIME = {"delay", "reverb", "beatrepeat"}         # their edit draws on earlier frames (Erosion is ordered only for its hold)
ELEMENTWISE = {"saturator", "erosion", "gate", "limiter"}
SUBSPACE = {"eq", "multiband", "shifter"}
GROUP_TITLES = {"time": "Along time", "size": "On the stream's size", "element": "Coordinate by coordinate",
                "subspace": "Along directions and principal bands", "block": "Inside a block", "other": "Other"}
ELEMENTWISE_NOTE = ("Acts on each of the stream's 1,024 coordinates separately, so what it does depends on the "
                    "coordinates the stream is written in. In the measurements these units broke the sound no more "
                    "often than the others.")


def space_of(u) -> Dict[str, Any]:
    """`space` (the manifest's own, where it declares one), VAU's `axis`, and the unit's group here."""
    try:
        from vau.manifest import space as vau_space       # VAU 0.2.1 and after: the manifest's space, or its site's
        space = vau_space(u.m)
    except ImportError:
        space = u.m.get("space") or SPACE.get(u.id) or ("component" if u.m["site"] == "output" else "stream")
    axis = u.m.get("axis", "any")
    if u.id in ALONG_TIME:
        group = "time"
    elif space in ("component", "hidden"):
        group = "block"
    elif u.id in ELEMENTWISE:
        group = "element"
    elif u.id in SUBSPACE:
        group = "subspace"
    elif space == "stream":
        group = "size"
    else:
        group = "other"
    return {"space": space, "axis": axis, "group": group, "group_title": GROUP_TITLES[group],
            "note": ELEMENTWISE_NOTE if u.id in ELEMENTWISE and space == "stream" else None}


def vau_name(u) -> str:
    """VAU's own name for a unit, as VAU shows it: its collection's prefix before the manifest's
    title (vLookback, vNorm Shrink for the Core Collection)."""
    try:
        from vau import collection                         # VAU 0.2 and after
        return collection.name(u)
    except (ImportError, AttributeError):
        return "v" + u.m["title"]


def catalogue() -> List[Dict[str, Any]]:
    out = []
    for u in registry():
        missing = u.missing(CAPABILITIES)
        if u.m["category"] == "meter":
            continue                                            # meters need a host view this engine has not got yet
        out.append({"id": u.id, "title": title(u), "short": short(u), "vau_title": vau_name(u), "renamed": u.id in NAMES, "what": u.m["what"], "category": u.m["category"],
                    "site": u.m["site"], **space_of(u), "params": params(u), **placement(u), "example": u.m.get("example"),
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
        host.store = store                 # the units' memory carried over: modulation moves a knob without clearing it
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
                host.edits, host.outs, host.ins = mark            # nothing of a failed strip stays attached
                raise
            if native:                                            # the strip's edits in MLX, on the GPU, where mlxunits has them
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
