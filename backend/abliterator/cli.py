# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
The command line, for the Magenta RT 2 engine (runs in .venv-mrt2; ./abliterator is the wrapper).

    ./abliterator render --preset warm-drone --seed 1 --seconds 20 --dose brightness=0.3
        A WAV pair from the same seed, clean (no strips) and steered, and a recipe that names both.
    ./abliterator replay data/sessions/NAME.json [--seconds 60] [--compare]
        Regenerates a session from its recipe: the moves applied at their frames. With --compare,
        reports whether the result matches the session's recorded audio sample for sample.

All audio written is the model's output, before the master chain, unless --post is given, which
writes the master chain's output beside it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import ROOT, TOOL, VERSION
from . import presets as pr

RENDERS = ROOT / "data" / "renders"


def _engine():
    from . import mrt2_proc as M
    import contextlib
    p = M.Proc()
    t0 = time.time()
    with contextlib.redirect_stdout(sys.stderr):        # the libraries print while loading
        p.load([x["prompt"] for x in pr.presets()])
    print(f"loaded Magenta RT 2 in {time.time() - t0:.1f} s", file=sys.stderr)
    return M, p


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(a.astype("<f4").tobytes()).hexdigest()


def _progress(f, n):
    print(f"  frame {f} of {n}", file=sys.stderr)


def render(a):
    import soundfile as sf
    preset = pr.find(a.preset)
    if preset is None:
        sys.exit(f"no preset '{a.preset}'; presets: {', '.join(x['id'] for x in pr.presets())}")
    doses = {}
    for d in a.dose:
        k, _, v = d.partition("=")
        doses[k] = float(v)
    M, p = _engine()
    unknown = [k for k in doses if k not in p.vectors]
    if unknown:
        sys.exit(f"no direction {unknown}; directions: {', '.join(p.vectors)}")
    sampling = {k: v for k, v in (("guidance", a.guidance), ("temperature", a.temperature)) if v is not None}
    frames = int(round(a.seconds * 1000 / M.FRAME_MS))
    base = {"prompt": preset["prompt"], "seed": a.seed,
            "start": {"sampling": {**M.DEFAULTS, **(preset.get("mrt2") or {}), **sampling}, "prompt": preset["prompt"]}}
    clean_spec = {**base, "start": {**base["start"], "doses": {}}}
    steer_spec = {**base, "start": {**base["start"], "doses": doses, "chain": list(a.strip)}}
    if a.strip:
        p.attach_chain(a.strip)                          # a bad strip fails here, before anything renders
    t0 = time.time()
    clean = p.render(clean_spec, frames, _progress)
    steered = p.render(steer_spec, frames, _progress)
    RENDERS.mkdir(parents=True, exist_ok=True)
    name = a.name or f"{time.strftime('%Y%m%d-%H%M%S')}-{a.preset}-s{a.seed}"
    files = {}
    for label, out in (("clean", clean), ("steered", steered)):
        sf.write(RENDERS / f"{name}.{label}.wav", out["pre"], M.SR, subtype="FLOAT")
        files[label] = {"file": f"{name}.{label}.wav", "sha256_f32": _sha(out["pre"])}
        if a.post:
            sf.write(RENDERS / f"{name}.{label}.post.wav", out["post"], M.SR, subtype="FLOAT")
            files[label]["post"] = f"{name}.{label}.post.wav"
    diff = steered["pre"] - clean["pre"]
    rms = lambda x: float(np.sqrt((x ** 2).mean()))  # noqa: E731
    recipe = {
        "format": "abliterator.render.v1", "tool": {"name": TOOL, "version": VERSION},
        "engine": "mrt2", "model": p.info(), "preset": preset, "seed": a.seed, "frames": frames,
        "chain": {"strips": list(a.strip), "vau": _vau_version(),
                  "steer": {"layer": M.LAYER, "doses": doses, "folder": f"data/directions/{M.SIZE}",
                            "directions": {k: p_sum["sha256"] for k, p_sum in _dir_shas(M).items() if k in doses}}},
        "clean": clean_spec, "steered": steer_spec, "files": files,
        "audio_is": "the model's output, before the master chain; 32-bit float WAV",
        "measures": {"rms_clean": rms(clean["pre"]), "rms_steered": rms(steered["pre"]), "rms_difference": rms(diff),
                     "identical": bool(np.array_equal(clean["pre"], steered["pre"]))},
        "seconds_to_render": round(time.time() - t0, 1),
    }
    (RENDERS / f"{name}.json").write_text(json.dumps(recipe, indent=1))
    print(json.dumps({"written": str((RENDERS / name).relative_to(ROOT)) + ".{clean,steered}.wav, .json", **recipe["measures"]}))


def _vau_version():
    try:
        import vau
        return {"vau": vau.VAU_VERSION, "package": vau.__version__}
    except ImportError:
        return None


def _dir_shas(M):
    from . import directions as dr
    return {k: d.data for k, d in dr.load_all(M.DIRS).items()}


def replay(a):
    import soundfile as sf
    path = Path(a.recipe)
    rec = json.loads(path.read_text())
    if rec.get("engine") != "mrt2":
        sys.exit("replay is written for Magenta RT 2 sessions; this one is not")
    M, p = _engine()
    snap = rec.get("format") == "abliterator.snapshot.v1"
    total = rec["frames"][1] if snap else rec["frames"]
    frames = total if (a.seconds is None or snap) else min(total, int(round(a.seconds * 1000 / M.FRAME_MS)))
    spec = {"prompt": rec["start"]["prompt"], "seed": rec["seed"], "start": rec["start"], "moves": rec["moves"]}
    t0 = time.time()
    out = p.render(spec, frames, _progress)
    if snap:
        out = {k: v[rec["frames"][0] * M.HOP:] for k, v in out.items()}
    RENDERS.mkdir(parents=True, exist_ok=True)
    name = f"{path.stem}.replay"
    sf.write(RENDERS / f"{name}.wav", out["pre"], M.SR, subtype="FLOAT")
    res = {"written": str((RENDERS / name).relative_to(ROOT)) + ".wav", "frames": frames,
           "seconds_to_render": round(time.time() - t0, 1)}
    if a.compare:
        live, sr = sf.read(path.with_suffix(".wav"), dtype="float32", always_2d=True)
        n = len(out["pre"])
        ref = live[:n]
        got = out["pre"][:len(ref)]
        # the live session is written as 16-bit PCM by default, so compare at that resolution
        q = lambda x: np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)  # noqa: E731
        d = np.abs(q(got).astype(np.int32) - q(ref).astype(np.int32))
        first = int(np.argmax(d.max(1) > 1)) if (d > 1).any() else None
        res.update({"compared_samples": len(ref), "max_difference_lsb": int(d.max()),
                    "matches": first is None, "first_difference_frame": None if first is None else first // M.HOP})
    print(json.dumps(res))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="abliterator", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="a clean and steered WAV pair, and a recipe")
    r.add_argument("--preset", required=True)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--seconds", type=float, default=20)
    r.add_argument("--dose", action="append", default=[], help="direction=dose, as a share of the stream's size")
    r.add_argument("--strip", action="append", default=[], help='a VAU strip, as "saturator layer=6 drive=2 mix=0.5"')
    r.add_argument("--guidance", type=float)
    r.add_argument("--temperature", type=float)
    r.add_argument("--post", action="store_true", help="also write the master chain's output")
    r.add_argument("--name")
    r.set_defaults(fn=render)
    p = sub.add_parser("replay", help="regenerate a session from its recipe")
    p.add_argument("recipe")
    p.add_argument("--seconds", type=float)
    p.add_argument("--compare", action="store_true")
    p.set_defaults(fn=replay)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
