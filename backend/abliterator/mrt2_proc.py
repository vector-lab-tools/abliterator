# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
The Magenta RealTime 2 engine, in its own process and its own environment (.venv-mrt2: magenta-rt with
MLX, no torch). The server (mrt2.py) starts it and talks to it in JSON lines over stdin and a private
copy of stdout; anything else written to stdout, by this program or the libraries, goes to stderr.

    main thread (MLX)   one 40 ms frame of 48 kHz stereo per step, SpectroStream decoding included;
                        reads the Steer sum and the conditioning at every step; runs the tasks that
                        need MLX (play, stop, making directions)
    reader thread       commands from the server; doses, master, sampling and lead are set here
                        by single assignments, so a step sees the old value or the new
    callback            the sound card: plays the ring through the master chain, logs DAC times

MLX evaluates on the thread that made its streams, so generation stays on the main thread. The
sampler holds at most `lead` frames beyond what has been played, as in the MusicGen engine.

The Steer strip adds ‖h‖ · Σ dose_i · v_i to the stream after block `layer` of the temporal
transformer (12 blocks, d = 1024), the doses read at every step: latency class live. Magenta RT 2
passes guidance as conditioning tokens, not a second batch, so guidance and temperature are live
controls here and cost nothing.
"""
from __future__ import annotations

import os
import sys

_proto = None                                         # set by main(): the protocol's own stdout

import collections  # noqa: E402
import json  # noqa: E402
import queue  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Tuple  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.setswitchinterval(0.001)          # the callback should not wait long for the interpreter

import numpy as np  # noqa: E402

from abliterator import ROOT, TOOL, VERSION  # noqa: E402
from abliterator import directions as dr  # noqa: E402
from abliterator import mod  # noqa: E402
from abliterator import scope as scp  # noqa: E402
from abliterator.spectro import Spectro  # noqa: E402
from abliterator.dsp import Conceal, Master  # noqa: E402

SR, HOP, FRAME_MS, CHANNELS = 48000, 1920, 40, 2
DEVICE_S = 0.040                       # the sound card's queue (see play)
_TS = os.environ.get("ABL_TEST_SRC")   # diagnostic: a raw float32 stereo file played in place of the model, to test the rest of the path
TEST_SRC = np.fromfile(_TS, dtype=np.float32).reshape(-1, CHANNELS) if _TS else None
SIZE, BITS, LAYER = "mrt2_small", 8, 6
DIRS = ROOT / "data" / "directions" / SIZE
SESSIONS = ROOT / "data" / "sessions"
SNAPSHOTS = ROOT / "data" / "snapshots"
DEFAULTS = {"guidance": 3.0, "temperature": 1.3, "top_k": 40}     # the vendor's defaults
LICENCE = "CC-BY 4.0 (weights), Apache-2.0 (code)"
HEARD_IS = ("the output section applied (auto level, low-pass, limiter, and their modulation) at full level, the volume "
            "knob divided out; what the tracks play; 32-bit float")

_out_lock = threading.Lock()


def send(obj: Dict[str, Any]):
    with _out_lock:
        _proto.write(json.dumps(obj) + "\n")


READ_EVERY = 3                            # frames between the strips' readings: they cost GPU time, the page polls slower
PROBE_N = 32                              # each strip reports this many of the stream's values in and out (status.probe)


BOOST_MAX = 4                 # the automatic buffer: at most this many frames above the knob's
BOOST_HOLD_S = 60.0           # and each kept this long after the last sign of trouble
SCOPE_S = 3.2                 # the scope's history of the sound as played, in seconds


def heard_speed(master: Dict[str, Any]):
    """The speed and mode the Speed stage plays from a master's settings: Off plays at 1, keeping the knob's value."""
    if master.get("speed_mode") == "off":
        return 1.0, "tape"
    return float(master.get("speed", 1.0)), master.get("speed_mode", "tape")


def settle_heap():
    """Collect, then freeze what is left (the model, its weights' wrappers, the strips) into Python's permanent
    generation, which the garbage collector does not walk. Without it a full collection walks the whole model:
    measured live, they grew from 17 to 99 ms over a few minutes and, run in the status thread, held up the step
    (a 125 ms step, a dropout at a two-frame buffer)."""
    import gc
    gc.collect()
    gc.freeze()


def interactive():
    """Ask macOS to run this thread as user-interactive work, so it is kept on the performance cores and woken on
    time when other programs load the machine. Measured over 1,500 steps of a busy chain: steps over 55 ms (the
    two-frame buffer's headroom) fell from 5 to 3, the worst from 85 to 63 ms. Does nothing elsewhere."""
    if sys.platform != "darwin":
        return
    try:
        import ctypes
        ctypes.CDLL("/usr/lib/libSystem.B.dylib").pthread_set_qos_class_self_np(ctypes.c_uint(0x21), ctypes.c_int(0))   # QOS_CLASS_USER_INTERACTIVE
    except Exception:
        pass


class Hooks:
    """Wraps every block of the temporal transformer. One block carries the Steer sum; while `tap` is
    set, the stream at every layer (layer 0 the input to block 1, layer l the output of block l) is
    added into it, for making directions."""

    def __init__(self, blocks, layer: int):
        self.layer = layer
        self.vec = None                    # Σ dose_i · v_i, or None
        self.tap: Optional[List[Any]] = None
        self.edits: Dict[int, List[Any]] = {}     # VAU strips' edits by layer (torch, on the CPU)
        self.outs: Dict[Tuple[int, str], List[Any]] = {}    # VAU output hooks by (block, "attn" | "mlp")
        self.ins: Dict[Tuple[int, str], List[Any]] = {}     # VAU input (pre-) hooks on the output projections
        self.frame = 0                     # the frame being made: the strips' position
        self.read_at: Tuple[int, ...] = ()       # the scopes' layers: the stream there is kept, after the Steer and the strips
        self.read: Dict[int, Any] = {}
        self.meas: List[Tuple[int, Any]] = []   # (strip, |after - before| / |before|) for each edit run this frame
        self.measuring = False                   # live play only: offline renders do not measure
        self.probes: Dict[int, Tuple[Any, Any, Any]] = {}   # each strip's sample of values in and out, and |out| / |in|
        for i, b in enumerate(blocks):
            self._wrap(i, b)
            att, mlp = b.layers[0].body.layers, b.layers[2].body.layers
            self._wrap_part(att[4], (i, "attn"), "out")      # attention's post-norm: what it adds to the stream
            self._wrap_part(mlp[5], (i, "mlp"), "out")       # the MLP's post-norm
            self._wrap_part(att[2], (i, "attn"), "in")       # attention's output projection (heads in)
            self._wrap_part(mlp[3], (i, "mlp"), "in")        # the MLP's down-projection (hidden units in)

    def _wrap_part(self, layer, key, where: str):
        orig = layer.step_with_emits

        def step(x, st, *a, **k):
            fns = (self.outs if where == "out" else self.ins).get(key)      # looked up now: a chain swaps the tables
            if fns and where == "in":
                x = x.apply_values(lambda h: self._bridge_part(h, fns, "in"))
            y, st2, e = orig(x, st, *a, **k)
            if fns and where == "out":
                y = y.apply_values(lambda h: self._bridge_part(h, fns, "out"))
            return y, st2, e
        layer.step_with_emits = step

    def _mlx_change(self, fn, before, after):
        """How much a strip's edit changed the stream, as a share of its size; lazy, read after the step."""
        import mlx.core as mx
        s = getattr(fn, "strip", None)
        if s is not None and self.measuring:
            b = before.astype(mx.float32)
            self.meas.append((s, mx.linalg.norm(after.astype(mx.float32) - b) / (mx.linalg.norm(b) + 1e-6)))
            if s not in self.probes:                                      # the first place the strip acts, this frame
                d = before.shape[-1]
                bi, ai = before.reshape(-1, d)[-1].astype(mx.float32), after.reshape(-1, d)[-1].astype(mx.float32)
                g = mx.linalg.norm(ai) / (mx.linalg.norm(bi) + 1e-6)
                if getattr(fn, "unit", "") == "neuron":                   # the neuron changed most: [index, in], [index, out]
                    i = mx.argmax(mx.abs(ai - bi))
                    self.probes[s] = (mx.stack([i.astype(mx.float32), bi[i]]), mx.stack([i.astype(mx.float32), ai[i]]), g)
                else:
                    ix = mx.array([int(j * d / PROBE_N) + 7 for j in range(PROBE_N)])
                    self.probes[s] = (bi[ix], ai[ix], g)

    def _torch_change(self, fn, before, after):
        s = getattr(fn, "strip", None)
        if s is not None and after is not None and self.measuring:
            if s not in self.probes:
                d = before.shape[-1]
                bi, ai = before.reshape(-1, d)[-1].float().numpy(), after.reshape(-1, d)[-1].float().numpy()
                g = np.array([np.linalg.norm(ai) / (np.linalg.norm(bi) + 1e-6)])
                if getattr(fn, "unit", "") == "neuron":
                    i = int(np.argmax(np.abs(ai - bi)))
                    self.probes[s] = (np.array([i, bi[i]]), np.array([i, ai[i]]), g)
                else:
                    ix = [int(j * d / PROBE_N) + 7 for j in range(PROBE_N)]
                    self.probes[s] = (bi[ix], ai[ix], g)
            b = before.float()
            self.meas.append((s, float((after.float() - b).norm() / (b.norm() + 1e-6))))

    def _bridge_part(self, h, fns, where: str):
        """A component's output (or its projection's input) through VAU forward hooks (or pre-hooks).
        Hooks marked mlx (mlxunits.py) take the MLX array as it is; a run of torch hooks goes through the CPU."""
        i = 0
        while i < len(fns):
            if getattr(fns[i], "mlx", False):
                h0, h = h, fns[i](h)
                self._mlx_change(fns[i], h0, h)
                i += 1
                continue
            j = i
            while j < len(fns) and not getattr(fns[j], "mlx", False):
                j += 1
            h = self._torch_part(h, fns[i:j], where)
            i = j
        return h

    def _torch_part(self, h, fns, where: str):
        import mlx.core as mx
        import torch
        shape = h.shape
        t = torch.from_numpy(np.array(h.astype(mx.float32))).reshape(1, 1, -1)    # heads (n, h) flattened, as Lens has them
        for fn in fns:
            t0 = t
            if where == "out":
                r = fn(None, (None,), t)
                if r is not None:
                    t = r[0] if isinstance(r, tuple) else r
            else:
                r = fn(None, (t,))
                if r is not None:
                    t = r[0] if isinstance(r, tuple) else r
            self._torch_change(fn, t0, t)
        return mx.array(t.float().numpy().reshape(shape)).astype(h.dtype)

    def _wrap(self, i: int, block):
        import mlx.core as mx
        orig = block.step_with_emits           # what the temporal transformer's Serial calls

        def step(x, st, *a, **k):
            tap = self.tap
            if i == 0 and tap is not None:
                tap[0] = tap[0] + x.values[0, -1].astype(mx.float32)
            y, st2, emits = orig(x, st, *a, **k)
            if i + 1 == self.layer:
                v = self.vec
                if v is not None:
                    y = y.apply_values(lambda h: h + mx.linalg.norm(h.astype(mx.float32), axis=-1, keepdims=True).astype(h.dtype)
                                       * v.astype(h.dtype))
            edits = self.edits.get(i + 1)
            if edits:
                y = y.apply_values(lambda h: self._bridge(h, edits))
            if i + 1 in self.read_at:
                self.read[i + 1] = y.values[0, -1]
            if tap is not None:
                tap[i + 1] = tap[i + 1] + y.values[0, -1].astype(mx.float32)
            return y, st2, emits
        block.step_with_emits = step

    def _bridge(self, h, edits):
        """The stream for this frame through the strips' edits: those marked mlx (mlxunits.py) in MLX,
        each run of torch edits out of MLX, through torch on the CPU, and back."""
        import mlx.core as mx
        import torch
        i = 0
        while i < len(edits):
            if getattr(edits[i], "mlx", False):
                h0, h = h, edits[i](h, self.frame)
                self._mlx_change(edits[i], h0, h)
                i += 1
                continue
            j = i
            while j < len(edits) and not getattr(edits[j], "mlx", False):
                j += 1
            t = torch.from_numpy(np.array(h.astype(mx.float32)))
            pos = torch.tensor([self.frame])
            for fn in edits[i:j]:
                t0, t = t, fn(t, pos)
                self._torch_change(fn, t0, t)
            h = mx.array(t.float().numpy()).astype(h.dtype)
            i = j
        return h


class Component:
    """A module as VAU units see it: a weight (out, in) as a torch tensor, a bias, and a name."""

    def __init__(self, w: np.ndarray, name: str):
        import torch
        self.weight = torch.from_numpy(w.astype(np.float16))
        self.bias = None
        self.name = name


class Proc:
    def __init__(self):
        self.tasks: "queue.Queue" = queue.Queue()
        self.status_text = "loading Magenta RT 2"
        self.error: Optional[str] = None
        self.playing = False
        self.master = Master(SR, CHANNELS)
        self.master.update({"declick": True, "fx_fade": True})            # live play repairs the bursts a unit can make the model write
        self.lead = 2
        # The automatic buffer: extra frames held in hand, on top of the Buffer knob's, after a step runs long or the card
        # runs dry; given back one at a time after BOOST_HOLD_S without trouble. Another program loading the CPU or GPU
        # stretches a few steps past what two frames cover, and the 95th-percentile load figure does not show it.
        self.boost = 0
        self.boost_t = 0.0
        self.block = 512
        self.sampling = dict(DEFAULTS)
        self.vectors: Dict[str, np.ndarray] = {}         # unit directions at LAYER
        self.mod_spec = mod.clean({})                     # modulation (mod.py): sources and routes
        self.master_base = self.master.settings()         # the master's knobs; modulation moves the Master from these
        self.memo: Dict[str, Any] = {}                    # what the steer vector and conditioning were last made from
        self.scope_last: Optional[List[List[float]]] = None
        self.lock = threading.Lock()                      # a move and the frame it is first heard in
        self.doses: Dict[str, float] = {}
        self.summaries: List[Dict[str, Any]] = []
        self.tokens_cache: Dict[str, List[int]] = {}
        self.chain: List[str] = []                        # the VAU strips, as written
        self.layers_on = True                             # the Model layers panel's switch: off plays the model unbent, keeping the strips
        self.steer_on = True                              # the Steer panel's switch: off sets every dose to 0, keeping the knobs
        self.chain_report: List[Dict[str, Any]] = []
        self.in_status = False                                # a status reply is being built (reader thread)
        self.phase = "idle"                                   # what the main thread is doing: idle, prep, step, read, task (for the callback trace)
        self.gc_on = 0                                        # collections running now, any thread
        self.cbt: "collections.deque" = collections.deque(maxlen=6000)   # the callback trace: one entry a callback (cb_trace)
        self.ptr: "collections.deque" = collections.deque(maxlen=12000)  # the main thread's phases, timed
        self.str_: "collections.deque" = collections.deque(maxlen=2000)  # status replies, timed (start, end)
        self.status_ms: "collections.deque" = collections.deque(maxlen=200)
        self.slow: "collections.deque" = collections.deque(maxlen=40)
        self.gcs: "collections.deque" = collections.deque(maxlen=40)    # garbage collections over 2 ms: (generation, ms, thread)
        import gc
        self._gc0 = {}
        def _gc(phase, info):
            import threading as th
            k = th.get_ident()
            if phase == "start":
                self._gc0[k] = time.perf_counter(); self.gc_on += 1
            elif k in self._gc0:
                self.gc_on = max(0, self.gc_on - 1)
                ms = (time.perf_counter() - self._gc0.pop(k)) * 1000
                if ms > 2:
                    self.gcs.append({"t": time.perf_counter(), "gen": info["generation"], "ms": round(ms, 1), "thread": "main" if th.current_thread() is th.main_thread() else th.current_thread().name, "playing": self.playing})
        gc.callbacks.append(_gc)
        self.activity: Dict[int, float] = {}                 # each strip's change to the stream, by line (_read_activity)
        self.activity_now: Dict[int, float] = {}
        self.probe: Dict[int, List[List[float]]] = {}       # Quantise strips' values in and out, the last frame
        self.chain_pending = None                         # edits made on the reader thread, swapped in at the next frame
        self.unit_now: Dict[str, Any] = {"store": None, "lines": None}    # the units' memory, and the lines modulation last attached
        self.stream_stats = None
        self.freezing = None
        self.stream = None
        self.spectro = Spectro(SR)                        # the spectrogram of what is heard, for the page
        self.meters = scp.Scope(self._scope_layer)        # the scopes on the model path: meters, recorded nowhere
        self._reset()

    # ------------------------------------------------------------------ loading (main thread)
    def load(self, prompts: List[str]):
        import mlx.core as mx
        import magenta_rt.mlx.system as S
        import sequence_layers.mlx as sl
        from magenta_rt.config import MUSICCOCA
        self.mx, self.sl, self.key = mx, sl, MUSICCOCA.key
        t0 = time.time()
        self.mrt = S.MagentaRT2System(size=SIZE, bits=BITS)
        self.cfgc = next(c for c in self.mrt._model.input_configs if c.key == self.key)
        self.spec = sl.ChannelSpec(shape=(self.mrt._num_channels,), dtype=mx.int32)
        blocks = self.mrt._sampler.depthformer.decoder.temporal_body.layers[0].layers
        self.n_layers = len(blocks)
        self.hooks = Hooks(blocks, LAYER)
        self.components = self._components(blocks)
        self.load_s = round(time.time() - t0, 1)
        self.status_text = "embedding the preset prompts"
        for p in prompts:
            self.tokens(p)
        have = dr.load_all(DIRS)
        for c in dr.contrasts():
            if c["name"] not in have:
                self.status_text = f"making the direction '{c['name']}' (about a minute)"
                dr.save(self.make_direction(c), DIRS)
        self._set_directions(dr.load_all(DIRS))
        self.load_calibration()
        from abliterator import units
        self.stream_stats = units.StreamStats(SIZE)
        if not self.stream_stats.exists():
            self.status_text = "measuring the stream for the units (once, about two minutes)"
            refs = list(dict.fromkeys(prompts + [q for c in dr.contrasts() for q in c["positive"] + c["negative"]]))
            self.measure_stream(refs)
        self.status_text = "ready"
        settle_heap()

    def load_calibration(self):
        """The knobs' musical curves (calibrate.py), for modulation to move along."""
        f = ROOT / "data" / "calibration" / f"{SIZE}.json"
        cal = json.loads(f.read_text()) if f.exists() else {}
        from abliterator import calibrate
        mod.set_curves({k: c["curve"] for k, c in calibrate.effective(cal).items() if k in ("guidance", "temperature") or k.startswith("dose:")})
        return cal

    def _components(self, blocks) -> Dict[Tuple[int, str], Any]:
        """Each block's attention and MLP output projection as VAU's Abliterate and Neuron read them:
        a weight (out, in), with the post-norm's scale folded into its rows, since the hook acts on
        what the post-norm writes, g ⊙ (W x + b) / rms; the bias is left inside (bias None), and the
        per-position 1/rms is a scalar that projection and Keep size pass through. float16, to save
        memory; units take .float()."""
        import torch
        mx = self.mx
        out = {}
        for i, b in enumerate(blocks):
            att, mlp = b.layers[0].body.layers, b.layers[2].body.layers
            p = att[2]
            W = mx.dequantize(p.q_weight, p.q_scales, p.q_biases, group_size=p._group_size, bits=p._bits)
            g = att[4]._rms_norm.weight
            out[(i, "attn")] = Component(np.array((W * g[:, None]).astype(mx.float32)), f"block {i}: attention output projection")
            L = mlp[3].inner._linear
            W = mx.dequantize(L.weight, L.scales, L.biases, group_size=L.group_size, bits=L.bits)
            g = mlp[5]._rms_norm.weight
            out[(i, "mlp")] = Component(np.array((W * g[:, None]).astype(mx.float32)), f"block {i}: MLP down-projection")
        return out

    def _set_directions(self, ds):
        self.summaries = [d.summary() for d in ds.values()]
        for name, d in ds.items():
            v = np.asarray(d.layers[LAYER], np.float32)
            self.vectors[name] = v / (np.linalg.norm(v) + 1e-12)
            self.doses.setdefault(name, 0.0)
        self.mod_spec = mod.clean(self.mod_spec, list(self.vectors))

    def measure_stream(self, prompts: List[str], seed: int = 0):
        """The stream at every layer, frame by frame, over the reference prompts, on the unchanged
        model: frames SKIP to FRAMES of each, as for directions."""
        from abliterator import units
        mx = self.mx
        rows = []
        for i, q in enumerate(prompts):
            cond = self.conditioning(q, DEFAULTS)
            st = self.initial_state(seed + i)
            for j in range(dr.FRAMES):
                self.hooks.tap = [mx.zeros((1024,), mx.float32) for _ in range(self.n_layers + 1)] if j >= dr.SKIP else None
                _, st = self.step(st, cond)
                if self.hooks.tap is not None:
                    mx.eval(*self.hooks.tap)
                    rows.append(np.stack([np.array(t) for t in self.hooks.tap]))
            self.hooks.tap = None
        units.StreamStats.save(SIZE, np.stack(rows), prompts, dr.FRAMES, dr.SKIP)

    def _scope_layer(self, layer: int) -> scp.Layer:
        """What a scope at `layer` reads against: the stream's statistics there, its principal
        directions, and the directions (the stream after block `layer`, as the Steer counts)."""
        st = self.stream_stats.stats()
        vt, sc = self.stream_stats.pca(layer)
        dirs = {n: d.layers[layer] for n, d in dr.load_all(DIRS).items()}
        return scp.Layer(st["mean"][layer], float(st["norm"][layer]), vt, sc, dirs)

    def set_scopes(self, layers: List[int]) -> Dict[str, Any]:
        at = self.meters.set(layers, self.n_layers)
        self.hooks.read_at = tuple(at)
        return {"scopes": at}

    def _read_scopes(self, frame: int):
        rd, self.hooks.read = self.hooks.read, {}
        if rd:
            self.mx.eval(*rd.values())
            self.meters.add(frame, {v: np.array(a.astype(self.mx.float32)) for v, a in rd.items()})

    def _read_activity(self):
        """Each strip's change to the stream at a read frame (the largest over its layers and parts), kept with a
        fall of about 150 ms; a strip that changed nothing reads 0. Read every READ_EVERY frames."""
        ms, self.hooks.meas = self.hooks.meas, []
        pr, self.hooks.probes = self.hooks.probes, {}
        self.probe = {}
        if pr:
            flat = [a for pair in pr.values() for a in pair if not isinstance(a, np.ndarray)]
            if flat:
                self.mx.eval(*flat)
            self.probe = {k: [np.round(np.array(a, dtype=np.float64).reshape(-1), 4).tolist() for a in trio] for k, trio in pr.items()}
        lazy = [v for _, v in ms if not isinstance(v, float)]
        if lazy:
            self.mx.eval(*lazy)
        now: Dict[int, float] = {}
        for k, v in ms:
            now[k] = max(now.get(k, 0.0), float(v) if isinstance(v, float) else v.item())
        fall = 0.77 ** READ_EVERY                                     # 0.77 a 40 ms frame
        self.activity = {k: max(now.get(k, 0.0), self.activity.get(k, 0.0) * fall)
                         for k in set(now) | set(self.activity) if k in now or self.activity.get(k, 0.0) > 1e-6}
        self.activity_now = now

    def _use(self, bundle):
        self.hooks.edits, self.hooks.outs, self.hooks.ins = bundle["edits"], bundle["outs"], bundle["ins"]

    def attach_chain(self, lines: List[str], strict: bool = False, store: Optional[Dict[Any, Dict]] = None, dirs=None):
        """Read, check and attach VAU strips (on the calling thread; torch only). A line that cannot
        work is left out, with its error in the report (strict: raises instead). Render leaves it
        out too, so a recipe replays what was heard. With `store`, the units keep the memory they
        had (modulation moving a knob); without it they start fresh (a strip set by hand)."""
        from abliterator import units
        dirs = dirs if dirs is not None else dr.load_all(DIRS)
        return units.attach(lines, (self.n_layers, self.stream_stats, dirs, lambda: self.hooks.frame, self.components), strict, store, native=True)

    def _mod_units(self, spec: Dict[str, Any], base: List[str], t: float, length: Optional[float], now: Dict[str, Any],
                   track: Optional["mod.FollowTrack"] = None):
        """Modulation of the units' knobs at one frame, the same live and offline: the strips with their
        routed knobs moved, attached again with the units' memory kept, only when the lines change.
        `now` holds the store and the lines last attached (None: the strips as written)."""
        from abliterator import units
        lines = mod.apply_units(spec, base, t, length, units.param_info, track) if base and mod.has_units(spec) else None
        if lines == base:
            lines = None
        if lines != now.get("lines"):
            if now.get("dirs") is None:
                now["dirs"] = dr.load_all(DIRS)              # read once per chain, not at every move
            bundle = self.attach_chain(lines if lines is not None else base, store=now.get("store"), dirs=now["dirs"])[0]
            self._use(bundle)
            now["lines"] = lines

    def tokens(self, prompt: str) -> List[int]:
        if prompt not in self.tokens_cache:
            emb = self.mrt.embed_style(prompt, use_mapper=True)
            tok = self.mrt._style_model.tokenize(emb).tolist()
            n = self.cfgc.rvq_truncation_level
            self.tokens_cache[prompt] = (tok + [-1] * n)[:n]
        return self.tokens_cache[prompt]

    def conditioning(self, prompt: str, sampling: Dict[str, Any]):
        return self.mrt._build_conditioning({self.key: self.tokens(prompt)}, cfg_scales={"musiccoca": float(sampling["guidance"])},
                                            temperature=float(sampling["temperature"]), top_k=int(sampling["top_k"]))

    def initial_state(self, seed: int):
        """A fresh sampler state whose sampling key comes from the seed (the vendor's is fixed at 42)."""
        st = self.mrt._sampler.get_initial_state(1, self.spec, constants={}, training=False)
        s0 = list(st[0])
        dec = list(s0[2])
        dec[0] = self.mx.stack([self.mx.random.key(int(seed))])
        s0[2] = tuple(dec)
        return (tuple(s0),) + tuple(st[1:])

    def step(self, state, cond):
        blk, constants = cond
        y, state, _ = self.mrt._sampler.step_with_emits(x=blk, state=state, constants=constants, training=False)
        self.mx.eval(y.values)
        return (np.array(y.values[0]).astype(np.float32) / 32768.0), state

    def make_direction(self, contrast: Dict[str, Any], seed: int = 0) -> Dict[str, Any]:
        mx = self.mx
        t0 = time.time()

        def means(prompt, s):
            cond = self.conditioning(prompt, DEFAULTS)
            st = self.initial_state(s)
            zero = [mx.zeros((1024,), mx.float32) for _ in range(self.n_layers + 1)]
            n = 0
            self.hooks.tap = None
            for j in range(dr.FRAMES + 4):
                if j == dr.SKIP:
                    self.hooks.tap = list(zero)
                _, st = self.step(st, cond)
                if self.hooks.tap is not None:
                    mx.eval(*self.hooks.tap)
                    n += 1
            tap, self.hooks.tap = self.hooks.tap, None
            return np.stack([np.array(t) / n for t in tap])
        P = np.stack([means(p, seed + i) for i, p in enumerate(contrast["positive"])])
        N = np.stack([means(p, seed + i) for i, p in enumerate(contrast["negative"])])
        return dr.record(contrast, P, N, {"model": f"magenta-rt-2/{SIZE}", "revision": "magenta-rt 2.0.3",
                                          "dtype": f"{BITS}-bit"}, seed, t0, f"musiccoca {DEFAULTS['guidance']}")

    def steer_sum(self, doses: Dict[str, float]) -> Optional[np.ndarray]:
        """Σ dose_i · v_i over the doses in their order (live and offline alike, so the sums are equal)."""
        v = None
        for n, d in doses.items():
            if d and n in self.vectors:
                v = self.vectors[n] * d if v is None else v + self.vectors[n] * d
        return v

    def _apply(self, doses: Dict[str, float], sampling: Dict[str, Any], prompt: str, memo: Dict[str, Any]):
        """Make the steer vector and the conditioning for a frame from the doses, sampling and
        prompt in force at it, remaking each only when what it is made from has changed. Live play
        and render both go through here, so they make the same arrays at the same frames."""
        dk = tuple(doses.items())
        if memo.get("d") != dk:
            v = self.steer_sum(doses)
            self.hooks.vec = None if v is None else self.mx.array(v)
            memo["d"] = dk
        ck = (prompt, float(sampling["guidance"]), float(sampling["temperature"]), int(sampling["top_k"]))
        if memo.get("c") != ck:
            memo["cond"] = self.conditioning(prompt, sampling)
            memo["c"] = ck
        return memo["cond"]

    def _step_snapping(self, state, cond, anchors: Dict[str, Any], anchor, snap, frame: int):
        """One step, with snap-back. A strip switched on keeps the model's state at this frame; a strip switched
        off with snap-back returns the model to the state kept for it, so its effect does not linger in the
        frames the model reads back (about 20 s). The frame where it returns is a crossfade, equal-power over
        its 40 ms, from the step the model would have made to the step from the kept state. Live and offline
        alike, so a saved sound replays it."""
        for u in anchor or ():
            anchors[str(u)] = (state, frame)
        back = None
        for u in snap or ():
            got = anchors.pop(str(u), None)
            if got is not None and (back is None or got[1] < back[1]):
                back = got                                   # several at once: the earliest moment
        if back is None:
            return self.step(state, cond)
        cont, _ = self.step(state, cond)                     # the sound going on, faded from
        for k, (s, f) in list(anchors.items()):
            if f > back[1]:
                anchors[k] = back                            # a strip switched on since keeps the moment returned to
        a, state = self.step(back[0], cond)
        w = (0.5 * np.pi * (np.arange(len(a)) + 0.5) / len(a))[:, None]
        return (cont * np.cos(w) + a * np.sin(w)).astype(np.float32), state

    def render(self, spec: Dict[str, Any], frames: int, progress=None) -> Dict[str, np.ndarray]:
        """Offline: the same steps as live play, with each move applied before the step of its frame.
        spec holds `prompt` and `seed`, and optionally `start` {doses, sampling, master, prompt, chain,
        mod, mod_length_s} and `moves` (as in a session record). Modulation is evaluated at each
        frame, as in live play. Returns the model's output ("pre") and the master chain's ("post")."""
        st0 = spec.get("start") or {}
        doses = {**{k: 0.0 for k in self.vectors}, **(st0.get("doses") or {})}
        sampling = {**DEFAULTS, **(st0.get("sampling") or {})}
        prompt = st0.get("prompt") or spec["prompt"]
        master = Master(SR, CHANNELS)
        mbase = {**master.settings(), "level": False, **(st0.get("master") or {})}
        master.update(mbase)
        mspec = mod.clean(st0.get("mod"), list(self.vectors))
        steer_on = st0.get("steer_on", True)
        length = st0.get("mod_length_s")
        by_frame: Dict[int, List[Dict[str, Any]]] = {}
        for m in spec.get("moves") or []:
            by_frame.setdefault(int(m["frame"]), []).append(m)
        memo: Dict[str, Any] = {}
        state = self.initial_state(int(spec["seed"]))
        curves = dict(mod.CURVES)
        mod.set_curves(st0.get("mod_curves") or {})        # the curves the recipe was made with (none before calibration)
        live = {"edits": self.hooks.edits, "outs": self.hooks.outs, "ins": self.hooks.ins}
        empty = {"edits": {}, "outs": {}, "ins": {}}
        read_at, self.hooks.read_at = self.hooks.read_at, ()      # the scopes read live play only
        first = self.attach_chain(st0["chain"])[0] if st0.get("chain") else empty
        self._use(first)
        chain_now = list(st0.get("chain") or [])
        unow = {"store": first.get("store"), "lines": None}
        pre, post = [], []
        anchors: Dict[str, Any] = {}
        follow = mod.FollowTrack()                       # the Follower's input: the frames written so far
        try:
            for f in range(frames):
                self.hooks.frame = f
                anc, snp = [], []
                for m in by_frame.get(f, []):
                    anc += m.get("anchor") or []
                    snp += m.get("snap") or []
                    if "chain" in m:
                        b = self.attach_chain(m["chain"])[0] if m["chain"] else empty
                        self._use(b)
                        chain_now = list(m["chain"] or [])
                        unow = {"store": b.get("store"), "lines": None}
                    if "doses" in m:
                        doses = {**doses, **m["doses"]}
                    if "sampling" in m:
                        sampling = dict(m["sampling"])
                    if "prompt" in m:
                        prompt = m["prompt"]
                    if "master" in m:
                        mbase.update(m["master"])
                    if "mod" in m:
                        mspec = mod.clean(m["mod"], list(self.vectors))
                    if "steer_on" in m:
                        steer_on = bool(m["steer_on"])
                t = f * mod.FRAME_S
                self._mod_units(mspec, chain_now, t, length, unow, follow)
                d, smp = mod.apply_model(mspec, doses, sampling, t, length, follow) if mod.active(mspec, model=True) else (doses, sampling)
                if not steer_on:
                    d = {k: 0.0 for k in d}
                cond = self._apply(d, smp, prompt, memo)
                a, state = self._step_snapping(state, cond, anchors, anc, snp, f)
                follow.push(a)
                pre.append(a)
                master.update(mod.apply_master(mspec, mbase, t, length, follow) if mod.active(mspec, model=False) else mbase)
                post.append(master.process(a))
                if progress and f % 250 == 0:
                    progress(f, frames)
        finally:
            self.hooks.vec = None
            self.hooks.read_at, self.hooks.read = read_at, {}
            self._use(live)
            mod.set_curves(curves)
        pre = np.concatenate(pre)
        if heard_speed(mbase)[0] != 1.0 or any("speed_at" in m for m in spec.get("moves") or []):
            post = [self.post_of(pre, {**(st0.get("master") or {})}, spec.get("moves") or [], 0, st0.get("mod"), length=length)]
        return {"pre": pre, "post": np.concatenate(post)}

    def info(self) -> Dict[str, Any]:
        return {"model_id": f"Magenta RT 2 small ({SIZE})", "package": "magenta-rt 2.0.3", "dtype": f"{BITS}-bit",
                "device": "MLX (Metal)", "sample_rate": SR, "channels": CHANNELS, "frame_ms": FRAME_MS,
                "licence": LICENCE, "steer_layer": LAYER, "n_layers": getattr(self, "n_layers", None), "codec": "SpectroStream"}

    # ------------------------------------------------------------------ play and stop (main thread)
    def _field(self, y: np.ndarray):
        """The stereo field of what is heard: the last blocks' samples, and the left-right correlation
        over the last quarter second, for the Engine panel's goniometer."""
        y = y.reshape(len(y), -1)
        if y.shape[1] < 2:
            return
        self.field_blocks.append(y[:, :2].copy())
        self.field_sums.append((float((y[:, 0] * y[:, 1]).sum()), float((y[:, 0] ** 2).sum()), float((y[:, 1] ** 2).sum())))

    def field(self) -> Optional[Dict[str, Any]]:
        if not self.field_blocks:
            return None
        y = np.concatenate(list(self.field_blocks))[-768:]
        lr, ll, rr = (sum(v) for v in zip(*self.field_sums))
        corr = lr / np.sqrt(ll * rr) if ll > 1e-12 and rr > 1e-12 else None
        return {"lr": np.round(y, 3).tolist(), "corr": None if corr is None else round(float(corr), 3)}

    def _reset(self):
        self.anchors: Dict[str, Any] = {}              # snap-back: the model's state kept for each strip switched on
        self.follow_track = mod.FollowTrack()           # the Follower's input: this take's frames, as written
        self.ring: "collections.deque" = collections.deque()
        self.buf = np.zeros((0, CHANNELS), np.float32)
        self.conceal = Conceal(SR)                     # dropouts covered without a click (dsp.Conceal)
        self.xruns = 0                                 # the card's own underflows: the callback ran late
        self.cb_ms: "collections.deque" = collections.deque(maxlen=500)
        self.boost, self.boost_t = 0, 0.0                 # each take starts from the Buffer knob's frames
        if hasattr(self, "cbt"):
            self.cbt.clear(); self.ptr.clear(); self.str_.clear()
        self.clicks: "collections.deque" = collections.deque(maxlen=30)
        self.last_sample = None
        self.played = 0
        self.gen = 0
        self.underruns = 0
        self.step_ms: "collections.deque" = collections.deque(maxlen=250)
        self.dac: "collections.deque" = collections.deque(maxlen=400)
        self.pending: List[Dict[str, Any]] = []
        self.measured: "collections.deque" = collections.deque(maxlen=20)
        self.moves: List[Dict[str, Any]] = []
        self.recorded: List[np.ndarray] = []
        self.scope_hist: collections.deque = collections.deque()   # the sound as played, after Speed and before the output section: the scope's
        self.scope_n = 0                                             # last seconds, which scroll at the speed heard
        self.field_blocks: "collections.deque" = collections.deque(maxlen=2)
        self.field_sums: "collections.deque" = collections.deque(maxlen=24)

    def play(self, preset: Dict[str, Any], seed: int, sampling: Optional[Dict[str, Any]] = None):
        import sounddevice as sd
        if self.playing:
            self.stop()
        self._reset()
        self.preset, self.seed = preset, int(seed)
        self.sampling = {**DEFAULTS, **(preset.get("mrt2") or {}), **(sampling or {})}
        self.prompt = preset["prompt"]
        self.state = self.initial_state(self.seed)
        self.start = {"doses": dict(self.doses), "master": dict(self.master_base), "sampling": dict(self.sampling),
                      "prompt": self.prompt, "chain": self.heard_chain(), "mod": self.mod_spec, "mod_curves": dict(mod.CURVES),
                      "steer_on": self.steer_on}
        self.memo = {}
        self.meters.reset()
        self.spectro.reset()
        from abliterator.speed import Speed, clamp
        self.speed = Speed(CHANNELS)                     # playback speed: read the model's sound slower than it is made
        r0, m0 = heard_speed(self.master_base)
        r0 = clamp(r0)
        self.speed.set(r0, m0)
        self.speed.target = self.speed.ratio = r0
        self.speed.mode = m0
        self.speed_logged = 0
        fresh = Master(SR, CHANNELS)                     # the output section from silence: no tails, levels or envelopes from the last take,
        fresh.update({**self.master.settings(), **self.master_base})    # as an offline render of the recipe starts
        self.master = fresh
        self.master_modded = False
        if self.heard_chain():
            self.chain_pending, _ = self.attach_chain(self.heard_chain())      # fresh memory for the strips
        settle_heap()                                    # the strips just attached join the model in the frozen generation
        # 40 ms queued at the card, not "low" (about 12 ms): while a model step runs, the callback can take longer than
        # its block (Smear's FFT, waiting for the interpreter), and CoreAudio plays silence for the shortfall without
        # reporting an underflow. 40 ms covers the longest gaps between callbacks measured under load (27 ms).
        self.stream = sd.OutputStream(samplerate=SR, channels=CHANNELS, blocksize=self.block, latency=DEVICE_S,
                                      callback=self._callback)
        self.playing = True
        self.stream.start()

    def stop(self) -> Optional[str]:
        if not self.playing:
            return None
        self.playing = False
        self.boost = 0                                   # the automatic buffer's frames end with the take
        self.stream.stop()
        self.stream.close()
        return self._save()

    def run(self):
        """The main loop: tasks between steps, and one step whenever the lead allows."""
        interactive()
        # The sound card's callback is Python and needs the interpreter lock; a thread running Python holds it for up to the
        # switch interval (5 ms by default) before handing it over. At 1 ms the callback starts on time while the model's
        # step runs its Python between MLX calls (measured with Smear: the longest wait between callbacks fell from 47 to 24 ms).
        sys.setswitchinterval(0.001)
        while True:
            try:
                while True:
                    fn, rid = self.tasks.get_nowait()
                    self._do(fn, rid)
            except queue.Empty:
                pass
            if not self.playing:
                try:
                    fn, rid = self.tasks.get(timeout=0.02)
                    self._do(fn, rid)
                except queue.Empty:
                    pass
                continue
            if self.boost and time.perf_counter() - self.boost_t > BOOST_HOLD_S:   # quiet long enough: give a frame back
                self.boost -= 1
                self.boost_t = time.perf_counter()
            if self.gen - self.speed.consumed // HOP > self.lead_eff():   # ahead of the model sound used (slower speed: fewer steps)
                time.sleep(0.0005)
                continue
            try:
                self._generate_one()
            except Exception as e:
                traceback.print_exc()
                self.error = f"generation stopped: {type(e).__name__}: {e}"
                self.playing = False
                try:
                    self.stream.stop()
                except Exception:
                    pass

    def lead_eff(self) -> int:
        """Frames kept ahead of the sound card: the Buffer knob's, plus the automatic buffer's."""
        return min(12, self.lead + self.boost)

    def _need_more(self, frames: int):
        """Hold `frames` more in hand than now, up to BOOST_MAX above the knob's (any thread)."""
        self.boost = max(self.boost, min(BOOST_MAX, frames))
        self.boost_t = time.perf_counter()

    def _ph(self, name: str):
        self.phase = name
        self.ptr.append((time.perf_counter(), name))

    def cb_trace(self) -> Dict[str, Any]:
        """The callback trace, the main thread's phases and the status replies, timed, for _tmp diagnostics."""
        return {"cb": list(self.cbt), "phases": list(self.ptr), "status": list(self.str_), "gc": list(self.gcs), "block": self.block,
                "now": time.perf_counter(), "slow": list(self.slow)}

    def _do(self, fn, rid):
        self._ph("task")
        try:
            r = fn()
            send({"id": rid, "ok": True, **(r or {})})
        except Exception as e:
            traceback.print_exc()
            send({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"})
        finally:
            self._ph("idle")

    def _generate_one(self):
        ta = time.perf_counter(); st_seen = self.in_status; self._ph("prep")
        with self.lock:                                        # MLX arrays are made here, on the main thread
            if self.chain_pending is not None:
                self._use(self.chain_pending)
                self.unit_now = {"store": self.chain_pending.get("store"), "lines": None}
                self.chain_pending = None
            self.hooks.frame = self.gen
            anc, snp = [], []
            for p in self.pending:
                if p["frame"] is None:
                    p["frame"] = self.gen                      # the first frame computed with the move
                    self.moves.append({"frame": self.gen, **p["move"]})
                    anc += p["move"].get("anchor") or []
                    snp += p["move"].get("snap") or []
            t = self.gen * mod.FRAME_S
            self._mod_units(self.mod_spec, self.heard_chain(), t, None, self.unit_now, self.follow_track)
            if mod.active(self.mod_spec, model=True):
                d, smp = mod.apply_model(self.mod_spec, self.doses, self.sampling, t, None, self.follow_track)
            else:
                d, smp = self.doses, self.sampling
            if not self.steer_on:
                d = {k: 0.0 for k in d}
            self.eff = {"doses": dict(d), "sampling": dict(smp)}
            self.cond = self._apply(d, smp, self.prompt, self.memo)
        t0 = time.perf_counter(); self._ph("step")
        reading = self.gen % READ_EVERY == 0                    # the strips' readings, every third frame (120 ms)
        self.hooks.meas, self.hooks.probes, self.hooks.measuring = [], {}, reading
        try:
            if TEST_SRC is not None:
                i = (self.gen * HOP) % (len(TEST_SRC) - HOP)
                a = TEST_SRC[i:i + HOP].copy()
            else:
                a, self.state = self._step_snapping(self.state, self.cond, self.anchors, anc, snp, self.gen)
        finally:
            self.hooks.measuring = False
        self.step_ms.append((time.perf_counter() - t0) * 1000); self._ph("read")
        self.follow_track.push(a)
        r = max(0.125, float(getattr(getattr(self, "speed", None), "ratio", 1.0) or 1.0))   # slower playback: each frame in hand lasts longer
        cushion = self.lead * FRAME_MS / r - 20                       # what the knob's frames cover, less the card's own margin
        if self.step_ms[-1] > 0.85 * cushion:                         # this step would have used most of it: hold enough for one like it
            self._need_more(int(np.ceil((self.step_ms[-1] * 1.3 + 20) * r / FRAME_MS)) - self.lead)
        if TEST_SRC is None:
            self._read_scopes(self.gen)
        tc = time.perf_counter()
        if reading and TEST_SRC is None:
            self._read_activity()
        td = time.perf_counter(); st_seen = st_seen or self.in_status
        if (td - ta) * 1000 > 45:                                   # a slow step and where its time went (status.slow_steps)
            self.slow.append({"frame": self.gen, "total": round((td - ta) * 1000, 1), "prep": round((t0 - ta) * 1000, 1),
                              "step": round(self.step_ms[-1], 1), "after": round((td - tc) * 1000 + (tc - t0) * 1000 - self.step_ms[-1], 1),
                              "status_overlap": st_seen, "ahead_ms": round((len(self.ring) * HOP + len(self.buf)) / SR * 1000)})
        self.recorded.append(a)
        self.ring.append(a)
        self.gen += 1
        self._ph("idle")

    def _callback(self, out, frames, t, status):
        """The sound card's callback, timed: the card's own underflows (the callback came too late, which
        the cover cannot catch), how long each callback takes, and any large jump in the sound at a
        block's edge, with what the block was, are kept for status's `clicks`."""
        t0 = time.perf_counter()
        ent = (t0, (t.outputBufferDacTime - t.currentTime) * 1000, self.phase, self.in_status, self.gc_on, bool(status.output_underflow))
        if status.output_underflow:
            self.xruns += 1
            self._click("card underflow", 0.0)
        kind = self._fill(out, frames, t)
        y0 = out[0].astype(np.float64)
        if self.last_sample is not None and self.played > 0:
            jump = float(np.abs(y0 - self.last_sample).max())
            inner = float(np.abs(np.diff(out[: min(len(out), 64)], axis=0)).max()) if len(out) > 1 else 0.0
            if jump > 0.2 and jump > 4 * inner:
                self._click(kind, jump)
        self.last_sample = out[-1].astype(np.float64)
        self.cb_ms.append((time.perf_counter() - t0) * 1000)
        self.cbt.append(ent + (self.cb_ms[-1], kind))

    def _click(self, kind: str, jump: float):
        self.clicks.append({"t": round(self.played / SR, 2), "kind": kind, "jump": round(jump, 3),
                            "cb_ms_max": round(max(self.cb_ms), 2) if self.cb_ms else None})

    def _take(self, k: int) -> Optional[np.ndarray]:
        """k samples of the model's sound, or None (and nothing taken) if it has not written them yet."""
        while len(self.buf) < k and self.ring:
            self.buf = np.concatenate([self.buf, self.ring.popleft()])
        if len(self.buf) < k:
            return None
        x, self.buf = self.buf[:k], self.buf[k:]
        return x

    def _speed_read(self, frames: int) -> Optional[np.ndarray]:
        """The next block of sound through the Speed stage: straight through at 1, read slower otherwise; each new
        speed is recorded with the output sample it was taken at, so a take replays it exactly (post_of)."""
        sp = self.speed
        sp.set(*heard_speed(self.master_base))
        if sp.idle():
            x = self._take(frames)
            if x is not None:
                sp.skip(x)
            return x
        x = sp.pull(frames, self._take)
        for at, r, m in sp.changes[self.speed_logged:]:
            self.moves.append({"frame": self.gen, "speed_at": at, "speed": r, "speed_mode": m})
        self.speed_logged = len(sp.changes)
        return None if x is None else x.astype(np.float32)

    def _fill(self, out, frames, t) -> str:
        x = self._speed_read(frames)
        if x is not None:
            self.scope_hist.append(x)
            self.scope_n += len(x)
            while self.scope_hist and self.scope_n - len(self.scope_hist[0]) >= SCOPE_S * SR:
                self.scope_n -= len(self.scope_hist.popleft())
            spec = self.mod_spec
            if mod.active(spec, model=False):                   # the master's controls, moved at the frame being heard
                self.master.update(mod.apply_master(spec, self.master_base, (self.played // HOP) * mod.FRAME_S, None, self.follow_track))
                self.master_modded = True
            elif self.master_modded:
                self.master.update(self.master_base)
                self.master_modded = False
            y = self.master.process(x)
            resumed = self.conceal.gaps > 0
            out[:] = self.conceal.good(y)
            self.spectro.feed(y)
            self._field(y)
            start = self.played
            self.played += frames
            dac = t.outputBufferDacTime
            lr = np.sqrt((y.reshape(len(y), -1) ** 2).mean(axis=0))      # each channel's level, for the OUT meter's two bars
            pk = np.abs(y.reshape(len(y), -1)).max(axis=0)
            self.dac.append((start, dac, float(np.sqrt((y ** 2).mean())), [round(float(v), 4) for v in lr], [round(float(v), 4) for v in pk]))
            for p in list(self.pending):
                if p["frame"] is not None:
                    s = p["frame"] * HOP
                    if start <= s < self.played:
                        if p["t"] is not None:
                            self.measured.append(round((dac + (s - start) / SR - p["t"]) * 1000, 1))
                        self.pending.remove(p)
                    elif s < start:
                        self.pending.remove(p)
            return "resume" if resumed else "audio"
        else:
            out[:] = self.conceal.gap(frames, out.shape)              # the last block, mirrored and fading, not a click
            if self.played > 0:
                self.underruns += 1
                self._need_more(self.boost + 1)                       # it ran dry: one frame more in hand
            return "gap"

    # ------------------------------------------------------------------ controls (reader thread)
    def control(self, doses=None, master=None, sampling=None, prompt=None, steer_on=None) -> Dict[str, Any]:
        with self.lock:
            return self._control(doses, master, sampling, prompt, steer_on)

    def heard_chain(self) -> List[str]:
        """The strips the model plays with: all of them, or none while the Model layers switch is off."""
        return list(self.chain) if self.layers_on else []

    def set_chain(self, lines: List[str], on: Optional[bool] = None, anchor: Optional[List[Any]] = None,
                  snap: Optional[List[Any]] = None) -> Dict[str, Any]:
        """New strips, and/or the Model layers switch. A move records the strips heard (none while
        off), so a replay hears what was heard. `anchor` names strips just switched on (the model's
        state is kept at the frame they start acting); `snap` names strips just switched off whose
        kept state the model returns to (snap-back, _step_snapping)."""
        lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
        live = on if on is not None else self.layers_on
        edits, report = self.attach_chain(lines if live else [])          # a strip that cannot work is reported, the rest attached
        if not live:
            _, report = self.attach_chain(lines)                         # still checked, so the desk shows what will not play
        with self.lock:
            t = self.stream.time if self.playing else None
            self.chain, self.chain_report, self.layers_on = lines, report, live
            self.chain_pending = edits
            if self.playing:
                move: Dict[str, Any] = {"chain": self.heard_chain()}
                if anchor:
                    move["anchor"] = [str(u) for u in anchor]
                if snap:
                    move["snap"] = [str(u) for u in snap]
                self.pending.append({"t": t, "frame": None, "move": move})
        return {"chain": self.chain, "report": report, "layers_on": self.layers_on}

    def _control(self, doses, master, sampling, prompt, steer_on=None) -> Dict[str, Any]:
        t = self.stream.time if self.playing else None
        move: Dict[str, Any] = {}
        if steer_on is not None:
            self.steer_on = bool(steer_on)
            move["steer_on"] = self.steer_on
        if doses:
            for k, v in doses.items():
                if k in self.vectors:
                    self.doses[k] = float(v)
            move["doses"] = dict(self.doses)
        if sampling:
            s = dict(self.sampling)
            for k in ("guidance", "temperature"):
                if k in sampling:
                    s[k] = float(sampling[k])
            if "top_k" in sampling:
                s["top_k"] = int(sampling["top_k"])
            self.sampling = s
            move["sampling"] = dict(s)
        if prompt and self.playing:
            if prompt not in self.tokens_cache:
                raise ValueError("that prompt has not been embedded; choose a preset")
            self.prompt = prompt
            move["prompt"] = prompt
        if move and self.playing:
            self.pending.append({"t": t, "frame": None, "move": move})
        if master:
            self.master.update(master)
            self.master_base.update({k: v for k, v in self.master.settings().items() if k in master})
            if self.playing:
                self.moves.append({"frame": self.played // HOP, "master": dict(self.master_base)})
        return {"doses": dict(self.doses), "master": dict(self.master_base), "sampling": dict(self.sampling), "steer_on": self.steer_on}

    def set_mod(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """New modulation: heard from the next frame, and recorded as a move so replays follow it."""
        s = mod.clean(spec, list(self.vectors))
        with self.lock:
            t = self.stream.time if self.playing else None
            self.mod_spec = s
            if self.playing:
                self.pending.append({"t": t, "frame": None, "move": {"mod": s}})
        return {"mod": s}

    def scope(self, seconds: float = 3.0, points: int = 300) -> Optional[List[List[float]]]:
        """The last seconds of the model's sound as played (after Speed, before the output section), as (min, max)
        pairs of the two channels' mean. Read from what reached the sound card, so at a slower speed it scrolls at the
        speed heard rather than jumping with the frames the model writes."""
        rec = list(self.scope_hist) or self.recorded[-int(seconds * 1000 / FRAME_MS):]
        if rec and self.scope_hist:
            y0 = np.concatenate(rec)
            rec = [y0[-int(seconds * SR):]]
        if not rec:
            return self.scope_last
        y = np.concatenate(rec)
        n = len(y) // points
        if n < 1:
            return self.scope_last
        pairs = lambda x: np.round(np.stack([x.min(1), x.max(1)], 1), 4).tolist()
        y = y[-n * points:]
        self.scope_last = pairs(y.mean(axis=1).reshape(points, n))
        self.scope_lr = [pairs(y[:, c].reshape(points, n)) for c in range(min(2, y.shape[1]))]   # each channel, for the L and R view
        return self.scope_last

    # ------------------------------------------------------------------ readings
    def latency(self) -> Dict[str, Any]:
        dev_ms = (self.stream.latency * 1000) if (self.playing and self.stream) else DEVICE_S * 1000 + 10.0
        parts = {"generation lead": self.lead_eff() * FRAME_MS, "codec (declared by the sampler; not yet checked by ear)": 0,
                 "device": round(dev_ms, 1)}
        first = self.lead_eff() * FRAME_MS + dev_ms
        m = list(self.measured)
        return {"declared_first_ms": round(first, 1), "declared_full_ms": round(first, 1), "parts": parts,
                "measured_ms": m[-5:], "measured_median_ms": float(np.median(m)) if m else None}

    def status(self) -> Dict[str, Any]:
        s = {"tool": TOOL, "version": VERSION, "state": self.status_text, "error": self.error, "playing": self.playing,
             "model": self.info(), "lead": self.lead, "lead_boost": self.boost if self.playing else 0, "lead_range": [1, 12], "frame_ms": FRAME_MS, "block": self.block,
             "doses": dict(self.doses), "master": dict(self.master_base), "sampling": dict(self.sampling),
             "mod": self.mod_spec, "master_now": self.master.settings(), "scope": self.scope(), "scope_lr": getattr(self, "scope_lr", None), "field": self.field() if self.playing else None, "stages": self.master.readings() if self.playing else None,
             "steer_on": self.steer_on, "layers_on": self.layers_on, "scopes": list(self.meters.at), "meters": self.meters.payload(), "spec": self.spectro.payload(),
             "directions": self.summaries, "latency": self.latency(), "chain": self.chain, "chain_report": self.chain_report,
             "activity": {str(k): round(v, 5) for k, v in self.activity.items()} if self.playing else {},
             "probe": {str(k): v for k, v in self.probe.items()} if self.playing else {},
             "diag": {"status_ms_p50": round(float(np.median(self.status_ms)), 1) if self.status_ms else None, "status_ms_max": round(max(self.status_ms), 1) if self.status_ms else None,
                      "slow_steps": list(self.slow)[-12:], "gc": list(self.gcs)[-20:]},
             "mlp_width": int(self.components[(0, "mlp")].weight.shape[1]) if getattr(self, "components", None) else None,
             "freezing": self.freezing, "cpu_s": time.process_time()}   # this process's CPU seconds, all threads
        if self.playing:
            sm = np.array(self.step_ms) if self.step_ms else np.zeros(1)
            now = self.stream.time
            heard = [d for d in self.dac if d[1] <= now]
            s.update({"preset": self.preset.get("id"), "seed": self.seed, "prompt": self.prompt,
                      "seconds": round(self.played / SR, 2), "frames_generated": self.gen, "reprimes": None,
                      "step_ms_p50": round(float(np.median(sm)), 2), "step_ms_p95": round(float(np.percentile(sm, 95)), 2), "step_ms_peak": round(float(max(list(self.step_ms)[-50:])), 2) if self.step_ms else None,
                      "underruns": self.underruns, "xruns": self.xruns, "clicks": list(self.clicks),
                      "cb_ms_p50": round(float(np.median(self.cb_ms)), 2) if self.cb_ms else None,
                      "cb_ms_max": round(float(max(self.cb_ms)), 2) if self.cb_ms else None, "level": heard[-1][2] if heard else 0.0, "level_lr": heard[-1][3] if heard else [0.0, 0.0], "peak_lr": [round(max(d[4][c] for d in heard[-12:]), 4) for c in range(len(heard[-1][4]))] if heard else [0.0, 0.0],
                      "mod_values": mod.values(self.mod_spec, (self.played // HOP) * mod.FRAME_S, None, self.follow_track) if self.playing else None, "eff": getattr(self, "eff", None)})
        return s

    def about(self) -> Dict[str, Any]:
        """What the settings panel shows about the model and its runtime (read-only, any thread)."""
        import importlib.metadata as md
        import platform
        from abliterator import units
        vers = {}
        for k in ("magenta-rt", "mlx", "torch", "numpy", "scipy", "sounddevice"):
            try:
                vers[k] = md.version(k)
            except Exception:
                pass
        try:
            import vau
            vers["vau"] = getattr(vau, "__version__", "?")
        except Exception:
            pass
        st = None
        if self.stream_stats is not None and self.stream_stats.exists():
            d = self.stream_stats.data()
            st = {"made": str(d.get("made", "")), "prompts": int(len(d["prompts"])), "frames": int(d["frames"]), "skip": int(d["skip"]),
                  "samples": int(d["samples"].shape[0]), "layers": int(d["mean"].shape[0]), "width": int(d["mean"].shape[1]),
                  "norm_by_layer": [round(float(x), 1) for x in d["norm"]]}
        cal = None
        try:
            c = json.loads((ROOT / "data" / "calibration" / f"{SIZE}.json").read_text())
            cal = {"made": c.get("made"), "renders": c.get("renders"), "controls": len(c.get("controls", {})), "presets": sorted(c.get("floor", {}).keys())}
        except Exception:
            pass
        dirs = []
        for n, dd in dr.load_all(DIRS).items():
            sep = dd.data.get("separation") or []
            src = dd.data.get("source", {})
            dirs.append({"name": n, "made": dd.data.get("made"), "pairs": len(src.get("positive", [])),
                         "separation_at_steer": round(float(sep[LAYER]), 2) if len(sep) > LAYER else None,
                         "best_layer": int(np.argmax(sep)) if len(sep) else None, "seconds": dd.data.get("seconds")})
        try:
            mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30
        except Exception:
            mem = None
        sm = np.array(self.step_ms) if self.step_ms else None
        return {"model": self.info(), "load_s": getattr(self, "load_s", None), "versions": vers, "python": platform.python_version(),
                "machine": {"platform": platform.platform(), "chip": platform.processor() or platform.machine(), "memory_gb": round(mem, 1) if mem else None},
                "stream_stats": st, "calibration": cal, "directions": dirs,
                "units": {"available": len(units.catalogue()), "capabilities": sorted(units.CAPABILITIES)},
                "timing": {"lead_frames": self.lead, "frame_ms": FRAME_MS, "block": self.block, "latency": self.latency(),
                           "step_ms_p50": round(float(np.median(sm)), 2) if sm is not None and len(sm) else None,
                           "step_ms_p95": round(float(np.percentile(sm, 95)), 2) if sm is not None and len(sm) else None,
                           "underruns": self.underruns if self.playing else None},
                "playing": self.playing, "chain": self.heard_chain(), "routes": len(self.mod_spec["routes"])}

    def post_of(self, pre: np.ndarray, start_master: Dict[str, Any], moves: List[Dict[str, Any]], frame0: int = 0,
                start_mod: Optional[Dict[str, Any]] = None, unity: bool = False, length: Optional[float] = None) -> np.ndarray:
        """The master chain's output for recorded audio, with its moves and modulation at their frames
        (offline). With unity, divided by the volume knob's setting: the sound as the output section
        shapes it, at full level (modulation of the volume is kept). A recipe from before auto level
        has no `level` in its master, and is heard without it."""
        m = Master(SR, CHANNELS)
        base = {**m.settings(), "level": False, **start_master}
        m.update(base)
        spec = mod.clean(start_mod, list(self.vectors))
        self.last_where = None
        changes = [(int(mv["speed_at"]), float(mv["speed"]), mv.get("speed_mode", "tape")) for mv in moves if "speed_at" in mv]
        follow = mod.FollowTrack()                       # the Follower's input, from the same model output as live
        if frame0:
            follow.db = [-100.0] * frame0
        for f in range(len(pre) // HOP):
            follow.push(pre[f * HOP:(f + 1) * HOP])
        if heard_speed(base)[0] != 1.0 or changes:
            return self._post_slowed(pre, m, base, spec, moves, changes, frame0, unity, length, follow)
        by: Dict[int, List[Dict[str, Any]]] = {}
        for mv in moves:
            if "master" in mv or "mod" in mv:
                by.setdefault(mv["frame"], []).append(mv)
        out = []
        for f in range(len(pre) // HOP):
            for mv in by.get(frame0 + f, []):
                if "master" in mv:
                    base.update(mv["master"])
                if "mod" in mv:
                    spec = mod.clean(mv["mod"], list(self.vectors))
            m.update(mod.apply_master(spec, base, (frame0 + f) * mod.FRAME_S, length, follow) if mod.active(spec, model=False) else base)
            y = m.process(pre[f * HOP:(f + 1) * HOP])
            out.append(y / max(float(base["volume"]), 1e-6) if unity else y)
        return np.concatenate(out) if out else pre

    def _post_slowed(self, pre, m, base, spec, moves, changes, frame0, unity, length, follow=None) -> np.ndarray:
        """post_of with the Speed stage: the model's sound read slower, each change at the output sample it was
        taken at live, then the output section in output time (as live: its modulation at the frame being heard,
        its moves where their model frame came out). Keeps, for each model frame, where it came out (last_where)."""
        from abliterator.speed import Speed
        if frame0:
            raise ValueError("a slowed take is rendered from its start")
        nfr = len(pre) // HOP
        y, where = Speed.render(pre, *heard_speed(base), changes,
                                marks=[f * HOP for f in range(nfr + 1)])
        self.last_where = where
        at = sorted((where[min(mv["frame"], nfr)], i, mv) for i, mv in enumerate(moves) if ("master" in mv or "mod" in mv))
        out, j = [], 0
        for b in range(0, len(y), HOP):
            while j < len(at) and at[j][0] <= b:
                mv = at[j][2]
                if "master" in mv:
                    base.update({k: v for k, v in mv["master"].items() if k not in ("speed", "speed_mode")})
                if "mod" in mv:
                    spec = mod.clean(mv["mod"], list(self.vectors))
                j += 1
            m.update(mod.apply_master(spec, base, (b // HOP) * mod.FRAME_S, length, follow) if mod.active(spec, model=False) else base)
            z = m.process(y[b:b + HOP].astype(np.float32))
            out.append(z / max(float(base["volume"]), 1e-6) if unity else z)
        return np.concatenate(out) if out else y

    def snapshot(self, seconds: float) -> Dict[str, Any]:
        """The last `seconds` of what the engine generated (not what the sound card played, so
        dropouts are not in it), with a recipe that regenerates it: the session's start, the moves
        before its end, and its frame range."""
        import secrets
        import soundfile as sf
        rec = list(self.recorded)
        if not rec:
            raise RuntimeError("nothing has been played yet")
        to = len(rec)
        n = max(1, min(to, int(round(seconds * 1000 / FRAME_MS))))
        fr = to - n
        pre = np.concatenate(rec[fr:to])
        moves = [m for m in list(self.moves) if m["frame"] < to]
        SNAPSHOTS.mkdir(parents=True, exist_ok=True)
        sid = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        sf.write(SNAPSHOTS / f"{sid}.wav", pre, SR, subtype="FLOAT")
        heard = self.post_of(np.concatenate(rec[:to]), self.start["master"], moves, 0, self.start.get("mod"), unity=True)
        w = self.last_where                              # slowed: where the snapshot's frames came out
        heard = heard[w[fr]:w[to]] if w else heard[fr * HOP:to * HOP]
        sf.write(SNAPSHOTS / f"{sid}.heard.wav", heard, SR, subtype="FLOAT")
        recipe = {
            "format": "abliterator.snapshot.v1", "tool": {"name": TOOL, "version": VERSION},
            "id": sid, "engine": "mrt2", "model": self.info(), "preset": self.preset, "seed": self.seed,
            "start": self.start, "moves": moves, "frames": [fr, to], "seconds": round(n * FRAME_MS / 1000, 2),
            "sample_rate": SR, "channels": CHANNELS, "audio": f"{sid}.wav",
            "audio_is": "the model's output, before the master chain and any track effects; 32-bit float",
            "heard": f"{sid}.heard.wav", "heard_is": HEARD_IS,
            "chain": {"steer": {"layer": LAYER, "directions": sorted(self.vectors), "folder": f"data/directions/{SIZE}"}},
            "replay": "./abliterator replay data/snapshots/<id>.json --compare: frames 0 to the end are generated "
                      "with the start and the moves, and the frame range is kept",
        }
        (SNAPSHOTS / f"{sid}.json").write_text(json.dumps(recipe, indent=1))
        return {"snapshot": sid, "seconds": recipe["seconds"], "sample_rate": SR, "channels": CHANNELS,
                "label": f"{self._style_name()} · seed {self.seed}"}

    def _style_name(self) -> str:
        """The preset's name while its prompt is playing; after a live style change, that style's preset."""
        if self.prompt == self.preset.get("prompt"):
            return self.preset.get("name", self.preset.get("id"))
        for q in json.loads(os.environ.get("ABLITERATOR_PRESET_NAMES", "{}")).items():
            if q[0] == self.prompt:
                return q[1]
        return self.prompt[:40]

    def freeze(self, preset: Dict[str, Any], seed: int, seconds: float) -> Dict[str, Any]:
        """Render the current settings (style, doses, sampling, strips, master) offline from frame 0
        for `seconds`, as a snapshot: what the chain plays, made without the real-time limit. Runs on
        the main thread, so not while playing."""
        import secrets
        import soundfile as sf
        if self.playing:
            raise RuntimeError("stop playing first (or take a snapshot instead): freezing renders on the same GPU")
        frames = max(1, int(round(seconds * 1000 / FRAME_MS)))
        start = {"doses": dict(self.doses), "master": dict(self.master_base),
                 "sampling": {**DEFAULTS, **(preset.get("mrt2") or {}), **self.sampling}, "prompt": preset["prompt"],
                 "chain": self.heard_chain(), "steer_on": self.steer_on, "mod": self.mod_spec, "mod_length_s": round(frames * mod.FRAME_S, 3),
                 "mod_curves": dict(mod.CURVES)}
        t0 = time.time()
        self.freezing = (0, frames)

        def prog(f, n):
            self.freezing = (f, n)
        try:
            out = self.render({"prompt": preset["prompt"], "seed": seed, "start": start}, frames, prog)
        finally:
            self.freezing = None
        SNAPSHOTS.mkdir(parents=True, exist_ok=True)
        sid = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
        sf.write(SNAPSHOTS / f"{sid}.wav", out["pre"], SR, subtype="FLOAT")
        heard = self.post_of(out["pre"], start["master"], [], 0, start["mod"], unity=True, length=start["mod_length_s"])
        sf.write(SNAPSHOTS / f"{sid}.heard.wav", heard, SR, subtype="FLOAT")
        recipe = {
            "format": "abliterator.snapshot.v1", "tool": {"name": TOOL, "version": VERSION}, "id": sid,
            "engine": "mrt2", "frozen": True, "model": self.info(), "preset": preset, "seed": seed, "start": start,
            "moves": [], "frames": [0, frames], "seconds": round(frames * FRAME_MS / 1000, 2), "sample_rate": SR,
            "channels": CHANNELS, "audio": f"{sid}.wav", "seconds_to_render": round(time.time() - t0, 1),
            "audio_is": "the model's output, before the master chain and any track effects; rendered offline; 32-bit float",
            "heard": f"{sid}.heard.wav", "heard_is": HEARD_IS,
            "chain": {"strips": self.heard_chain(), "steer": {"layer": LAYER, "directions": sorted(self.vectors), "on": self.steer_on}},
        }
        (SNAPSHOTS / f"{sid}.json").write_text(json.dumps(recipe, indent=1))
        label = ("frozen · " + preset.get("name", preset.get("id", "")) + f" · seed {seed}"
                 + (f" · {len(self.chain)} strip{'s' if len(self.chain) != 1 else ''}" if self.chain else ""))
        return {"snapshot": sid, "seconds": recipe["seconds"], "sample_rate": SR, "channels": CHANNELS, "label": label,
                "seconds_to_render": recipe["seconds_to_render"]}

    def _save(self) -> Optional[str]:
        if not self.recorded:
            return None
        import soundfile as sf
        import mlx.core as mx
        SESSIONS.mkdir(parents=True, exist_ok=True)
        name = time.strftime("%Y%m%d-%H%M%S")
        pre = np.concatenate(self.recorded)
        sf.write(SESSIONS / f"{name}.wav", pre, SR, subtype="FLOAT")
        sf.write(SESSIONS / f"{name}.post.wav", self.post_of(pre, self.start["master"], self.moves, 0, self.start.get("mod")), SR, subtype="FLOAT")
        recipe = {
            "format": "abliterator.session.v1",
            "tool": {"name": TOOL, "version": VERSION},
            "engine": "mrt2",
            "model": self.info(),
            "preset": self.preset, "seed": self.seed,
            "chain": {"steer": {"layer": LAYER, "directions": sorted(self.vectors), "folder": f"data/directions/{SIZE}"}},
            "start": self.start,
            "moves": self.moves,
            "frames": len(self.recorded), "audio": f"{name}.wav", "audio_is": "the model's output, before the master chain",
            "audio_post": f"{name}.post.wav", "audio_post_is": "after the master chain, computed offline from the moves",
            "underruns": self.underruns,
            "environment": {"mlx": mx.__version__},
        }
        (SESSIONS / f"{name}.json").write_text(json.dumps(recipe, indent=1))
        return name


def reader(p: Proc):
    for line in sys.stdin:
        try:
            m = json.loads(line)
        except ValueError:
            continue
        rid, cmd = m.get("id"), m.get("cmd")
        try:
            if cmd == "status":
                t0 = time.perf_counter(); p.in_status = True
                try:
                    send({"id": rid, "ok": True, **p.status()})
                finally:
                    t1 = time.perf_counter(); p.in_status = False; p.status_ms.append((t1 - t0) * 1000); p.str_.append((t0, t1))
            elif cmd == "cbtrace":
                send({"id": rid, "ok": True, **p.cb_trace()})
            elif cmd == "control":
                if p.status_text != "ready":
                    raise RuntimeError("not ready")
                send({"id": rid, "ok": True, **p.control(m.get("doses"), m.get("master"), m.get("sampling"), m.get("prompt"), m.get("steer_on"))})
            elif cmd == "chain":
                if p.status_text != "ready":
                    raise RuntimeError("not ready")
                send({"id": rid, "ok": True, **p.set_chain(m.get("lines") or [], m.get("on"), m.get("anchor"), m.get("snap"))})
            elif cmd == "scopes":
                if p.status_text != "ready":
                    raise RuntimeError("not ready")
                send({"id": rid, "ok": True, **p.set_scopes(m.get("layers") or [])})
            elif cmd == "mod":
                if p.status_text != "ready":
                    raise RuntimeError("not ready")
                send({"id": rid, "ok": True, **p.set_mod(m.get("spec") or {})})
            elif cmd == "calibration":
                with p.lock:
                    p.load_calibration()
                send({"id": rid, "ok": True})
            elif cmd == "units":
                from abliterator import units
                send({"id": rid, "ok": True, "units": units.catalogue(), "capabilities": sorted(units.CAPABILITIES)})
            elif cmd == "freeze":
                p.tasks.put((lambda m=m: p.freeze(m["preset"], int(m.get("seed", 1)), float(m.get("seconds", 20))), rid))
            elif cmd == "snapshot":
                send({"id": rid, "ok": True, **p.snapshot(float(m.get("seconds", 8)))})
            elif cmd == "lead":
                p.lead = max(1, min(12, int(m["lead"])))
                send({"id": rid, "ok": True, "lead": p.lead})
            elif cmd == "play":
                p.tasks.put((lambda m=m: (p.play(m["preset"], m.get("seed", 0), m.get("sampling")), p.status())[1], rid))
            elif cmd == "stop":
                p.tasks.put((lambda: {"saved": p.stop()}, rid))
            elif cmd == "embed":                         # on the main thread, between steps (MLX)
                def emb(m=m):
                    t0 = time.perf_counter()
                    known = m["prompt"] in p.tokens_cache
                    p.tokens(m["prompt"])
                    return {"prompt": m["prompt"], "ms": round((time.perf_counter() - t0) * 1000, 1), "cached": known}
                p.tasks.put((emb, rid))
            elif cmd == "explain_files":
                from abliterator import units
                send({"id": rid, "ok": True, "files": units.explain_files()})
            elif cmd == "about":
                send({"id": rid, "ok": True, **p.about()})
            elif cmd == "quit":
                os._exit(0)
            else:
                raise ValueError(f"unknown command {cmd!r}")
        except Exception as e:
            send({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"})
    os._exit(0)                     # the server has gone


def main():
    global _proto
    _proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    prompts = json.loads(os.environ.get("ABLITERATOR_PROMPTS", "[]"))
    p = Proc()
    threading.Thread(target=reader, args=(p,), daemon=True).start()
    try:
        p.load(prompts)
    except Exception as e:
        traceback.print_exc()
        p.error = f"{type(e).__name__}: {e}"
        p.status_text = "failed to load"
    p.run()


if __name__ == "__main__":
    main()
