# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
The local server: serves the page on http://127.0.0.1:8775/ and controls the engine.

It listens on 127.0.0.1 only, refuses a request whose Host header is not this machine, and
refuses a write (POST, PUT, DELETE) without the header `X-Abliterator: 1`. It runs no jobs: the
engines play on their own threads and processes, and the server only reads their state and passes
control moves to them, so nothing the page does can hold up the sound.

Two engines, one playing at a time: Magenta RT 2 small (mrt2.py, in its own process and
environment) and MusicGen small (engine.py). Both are loaded at start.
"""
from __future__ import annotations

import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from abliterator import ROOT, TOOL, VERSION
from abliterator import directions as dr
import atexit

from abliterator import library
from abliterator.desk import Desk
from abliterator.engine import Engine
from abliterator.mrt2 import MRT2Engine

PORT = int(os.environ.get("ABLITERATOR_AUDIO_PORT", "8775"))
HOST = "127.0.0.1"
WEB = ROOT / "web"
RANGES = {"musical": [-0.4, 0.4], "full": [-3.0, 3.0]}     # Steer doses, as shares of the stream's size

app = FastAPI(title=TOOL, version=VERSION, docs_url=None, redoc_url=None)
musicgen = Engine()
mrt2 = MRT2Engine()
ENGINES = {"mrt2": mrt2, "musicgen": musicgen}
state: Dict[str, Any] = {"directions": {}, "active": "mrt2" if MRT2Engine.available() else "musicgen"}
atexit.register(mrt2.quit)
desk = Desk()


def active():
    return ENGINES[state["active"]]


def engines_list():
    return [{"id": k, "label": e.label, "state": e.status_text} for k, e in ENGINES.items()]


@app.middleware("http")
async def guard(request: Request, call_next):
    host = request.headers.get("host", "")
    if host not in (f"127.0.0.1:{PORT}", f"localhost:{PORT}"):
        return PlainTextResponse("Refused: this server only answers this machine.", status_code=403)
    if request.method in ("POST", "DELETE", "PUT") and request.headers.get("x-abliterator") != "1":
        return PlainTextResponse("Refused: a write needs the X-Abliterator header.", status_code=403)
    return await call_next(request)


def err(msg: str, code: int = 400):
    return JSONResponse({"error": msg}, status_code=code)


from abliterator.presets import presets  # noqa: E402


def _load():
    try:
        musicgen.status_text = "loading"
        # directions need the model: the engine loads it, then any missing direction is made
        have = dr.load_all()
        musicgen.load(have)
        missing = [c for c in dr.contrasts() if c["name"] not in have]
        for c in missing:
            musicgen.status_text = f"making the direction '{c['name']}' (about a minute)"
            d = dr.Direction(dr.make(musicgen.mg, c, emit=lambda s: None))
            dr.save(d.data)
            musicgen.steer.set_direction(d.name, d.unit(musicgen.steer.layer))
        state["directions"] = dr.load_all()
        musicgen.status_text = "ready"
    except Exception as e:
        traceback.print_exc()
        musicgen.error = f"{type(e).__name__}: {e}"
        musicgen.status_text = "failed to load"


def mg_directions():
    return [d.summary() for d in state["directions"].values()]


SOFTWARE = {
    "name": TOOL, "author": "David M. Berry", "institution": "University of Sussex", "orcid": "0000-0002-7737-5586",
    "licence": "GPL-3.0-or-later", "repository": "https://github.com/vector-lab-tools/abliterator",
    "site": "https://vector-lab-tools.github.io/abliterator/", "lab": "https://vector-lab-tools.github.io",
    "year": 2026, "doi": "10.5281/zenodo.24517766",      # Zenodo concept DOI: every version, resolving to the latest
}


_EXPLAIN: Dict[str, str] = {}


@app.get("/vau/explain/{name}")
def vau_explain(name: str):
    """A unit's explainer, the runtime that draws it, or its styles: only the files the engine's vau
    names (vau.explain.files), so nothing else on disk can be asked for."""
    if not _EXPLAIN:
        try:
            _EXPLAIN.update(mrt2.explain_files())
        except Exception as e:
            return err(f"{type(e).__name__}: {e}", 503)
    path = _EXPLAIN.get(name)
    if not path:
        return PlainTextResponse("No such explainer file.", status_code=404)
    kind = "text/css" if name.endswith(".css") else "application/json" if name.endswith(".json") else "text/javascript"
    return FileResponse(path, media_type=kind)


@app.get("/api/about")
def about():
    """The settings panel: the software, how to cite it, and what the engine reports about its model."""
    import platform
    out = {"software": {**SOFTWARE, "version": VERSION}, "server_python": platform.python_version(), "engine": state["active"],
           "presets": len(presets())}
    if state["active"] == "mrt2":
        try:
            out["model"] = mrt2.about()
        except Exception as e:
            out["model_error"] = f"{type(e).__name__}: {e}"
    else:
        out["model"] = {"model": {"model_id": "MusicGen small", "licence": "CC-BY-NC 4.0 (weights)"}}
    return out


@app.get("/api/diag/cbtrace")
def diag_cbtrace():
    """The engine's callback trace (each sound-card callback's lateness, length and what the other threads were doing),
    for diagnosing dropouts."""
    return mrt2.call("cbtrace", timeout=10)


@app.get("/api/status")
def status():
    s = active().status()
    if state["active"] == "musicgen":
        s["directions"] = mg_directions()
    if state["active"] != "mrt2" and mrt2.status_text != "ready":
        mrt2.status()                                      # refreshes its loading state for the picker
    s.update(engine=state["active"], engines=engines_list(), cpu_pct=cpu_load(s.get("cpu_s")))
    return s


CPU_SAMPLES: list = []


def cpu_load(engine_cpu_s):
    """Abliterator's share of the machine's CPU, as a percentage of all its cores: the CPU seconds the
    server and the engine process used over about the last second, over the wall time and the core count."""
    now, used = time.monotonic(), time.process_time() + (engine_cpu_s or 0.0)
    CPU_SAMPLES.append((now, used))
    while len(CPU_SAMPLES) > 2 and now - CPU_SAMPLES[1][0] >= 1.0:
        CPU_SAMPLES.pop(0)
    (t0, u0) = CPU_SAMPLES[0]
    if now - t0 < 0.2 or used < u0:                      # too soon, or the engine restarted
        if used < u0:
            CPU_SAMPLES[:] = [(now, used)]
        return None
    return round(100 * (used - u0) / (now - t0) / (os.cpu_count() or 1), 1)


@app.post("/api/engine")
async def set_engine(request: Request):
    b = await request.json()
    name = b.get("engine")
    if name not in ENGINES:
        return err("No such engine.")
    saved = None
    if name != state["active"]:
        try:
            if active().status().get("playing"):
                saved = active().stop()
        except Exception as e:
            traceback.print_exc()
            return err(f"{type(e).__name__}: {e}", 500)
        state["active"] = name
    return {"engine": name, "saved": saved}


@app.get("/api/presets")
def get_presets():
    dirs = mg_directions() if state["active"] == "musicgen" else (mrt2.status().get("directions") or [])
    return {"presets": presets(), "ranges": RANGES, "directions": dirs, "engine": state["active"]}


@app.post("/api/prompts")
async def add_prompt(request: Request):
    """A typed style prompt: kept under Your prompts, and embedded now so it can be switched to live."""
    from abliterator import presets as pr
    b = await request.json()
    try:
        p = pr.add(b.get("prompt", ""))
    except ValueError as e:
        return err(str(e))
    embedded = None
    if state["active"] == "mrt2" and mrt2.ready():
        try:
            embedded = mrt2.embed(p["prompt"])
        except Exception as e:
            return err(f"{type(e).__name__}: {e}", 500)
    return {"preset": p, "embedded": embedded}


@app.get("/api/mypresets")
def get_mypresets():
    from abliterator import mypresets as mp
    return mp.load()


@app.post("/api/mypresets")
async def add_mypreset(request: Request):
    """A unit's or an effect's settings kept under a name of yours (data/my_presets.json)."""
    from abliterator import mypresets as mp
    b = await request.json()
    try:
        q = mp.add(b.get("kind", ""), b.get("id", ""), b.get("name", ""), b.get("params") or {}, b.get("block"))
    except ValueError as e:
        return err(str(e))
    return {"preset": q, "all": mp.load()}


@app.post("/api/mypresets/delete")
async def delete_mypreset(request: Request):
    from abliterator import mypresets as mp
    b = await request.json()
    try:
        ok = mp.remove(b.get("kind", ""), b.get("id", ""), b.get("name", ""))
    except ValueError as e:
        return err(str(e))
    if not ok:
        return err("No such preset of yours.", 404)
    return {"all": mp.load()}


@app.post("/api/prompts/{pid}/delete")
def delete_prompt(pid: str):
    from abliterator import presets as pr
    if not pr.remove(pid):
        return err("No such prompt of yours.", 404)
    return {"deleted": pid}


def ready_or_409():
    e = active()
    if e.status_text != "ready":
        return err(f"{e.label} is not ready: {e.status_text}.", 409)
    return None


@app.post("/api/play")
async def play(request: Request):
    b = await request.json()
    if (r := ready_or_409()) is not None:
        return r
    p = next((x for x in presets() if x["id"] == b.get("preset")), None)
    if p is None:
        return err("No such preset.")
    seed = int(b.get("seed", p["seeds"][0]))
    try:
        if state["active"] == "mrt2":
            mrt2.play(p, seed, b.get("sampling"))
        else:
            musicgen.play(p, seed, b.get("cfg"))
    except Exception as e:
        traceback.print_exc()
        return err(f"{type(e).__name__}: {e}", 500)
    return status()


@app.post("/api/stop")
def stop():
    try:
        return {"saved": active().stop()}
    except Exception as e:
        return err(f"{type(e).__name__}: {e}", 500)


@app.post("/api/control")
async def control(request: Request):
    b = await request.json()
    if (r := ready_or_409()) is not None:
        return r
    lo, hi = RANGES["full" if b.get("full") else "musical"]
    doses = {k: min(hi, max(lo, float(v))) for k, v in (b.get("doses") or {}).items()}
    if state["active"] == "mrt2":
        prompt = None
        if b.get("preset"):
            p = next((x for x in presets() if x["id"] == b["preset"]), None)
            prompt = p["prompt"] if p else None
        sampling = {k: float(v) for k, v in (b.get("sampling") or {}).items() if k in ("guidance", "temperature", "top_k")}
        if "top_k" in sampling:
            sampling["top_k"] = int(min(250, max(1, round(sampling["top_k"]))))
        if "guidance" in sampling:
            sampling["guidance"] = min(10.0, max(0.0, sampling["guidance"]))
        if "temperature" in sampling:
            sampling["temperature"] = min(3.0, max(0.1, sampling["temperature"]))
        try:
            return mrt2.control(doses=doses, master=b.get("master"), sampling=sampling, prompt=prompt,
                                steer_on=(bool(b["steer_on"]) if "steer_on" in b else None))
        except Exception as e:
            return err(f"{type(e).__name__}: {e}", 500)
    musicgen.control(doses=doses or None, master=b.get("master"))
    return {"doses": musicgen.steer.doses, "master": musicgen.master.settings()}


@app.post("/api/lead")
async def lead(request: Request):
    b = await request.json()
    if state["active"] == "mrt2":
        try:
            return {"lead": mrt2.set_lead(int(b.get("lead", 2)))}
        except Exception as e:
            return err(f"{type(e).__name__}: {e}", 500)
    musicgen.lead = max(4, min(25, int(b.get("lead", musicgen.lead))))
    return {"lead": musicgen.lead}


@app.get("/api/sessions")
def sessions():
    folder = ROOT / "data" / "sessions"
    return {"sessions": sorted((p.stem for p in folder.glob("*.json")), reverse=True) if folder.exists() else []}


@app.get("/api/desk")
def desk_state():
    return desk.state()


@app.get("/api/units")
def get_units():
    if state["active"] != "mrt2":
        return {"units": [], "note": "VAU strips run on Magenta RT 2; MusicGen has the Steer strip only, for now."}
    try:
        return mrt2.units()
    except Exception as e:
        return err(str(e), 409)


@app.post("/api/chain")
async def set_chain(request: Request):
    b = await request.json()
    if state["active"] != "mrt2":
        return err("VAU strips run on Magenta RT 2 for now.", 409)
    if (r := ready_or_409()) is not None:
        return r
    lines = b.get("lines")
    if not isinstance(lines, list) or not all(isinstance(x, str) for x in lines):
        return err("Send the strips as a list of lines.")
    try:
        ids = lambda k: [str(x)[:16] for x in (b.get(k) or [])][:16] if isinstance(b.get(k), list) else None   # snap-back: strips' ids
        return mrt2.set_chain(lines[:16], on=(bool(b["on"]) if "on" in b else None), anchor=ids("anchor"), snap=ids("snap"))
    except Exception as e:
        return err(str(e))


@app.get("/api/fxtypes")
def fxtypes():
    from abliterator import postfx
    return {"types": postfx.catalogue(), "max": postfx.MAX_SLOTS}


@app.get("/api/calibration")
def calibration():
    """The knobs' musical ranges and curves (calibrate.py, cut to the ranges your ears set)."""
    from abliterator import calibrate
    if not calibrate.FILE.exists():
        return {"controls": {}, "ears": {}}
    import json as _j
    cal = _j.loads(calibrate.FILE.read_text())
    return {"made": cal.get("made"), "floor": cal.get("floor"), "ears": cal.get("ears", {}),
            "controls": {k: {x: c.get(x) for x in ("default", "safe", "curve", "level_db", "steps", "swept", "ears", "kind", "safe_places")}
                         for k, c in calibrate.effective(cal).items()}}


@app.get("/api/calibration/grid")
def calibration_grid():
    """Everything the Tune view shows: each control's steps on each preset, with the measures and clips."""
    from abliterator import calibrate
    if not calibrate.FILE.exists():
        return {"controls": {}}
    import json as _j
    cal = _j.loads(calibrate.FILE.read_text())
    eff = calibrate.effective(cal)
    out = {}
    for k, c in cal.get("controls", {}).items():
        out[k] = {"default": c["default"], "safe": c["safe"], "range": eff[k]["safe"] or [c["default"], c["default"]], "ears": cal.get("ears", {}).get(k),
                  "kind": c.get("kind"), "vau_default": c.get("vau_default"), "moved_default": c.get("moved_default"),
                  "values": sorted({m["value"] for ms in c["presets"].values() for m in ms}), "presets": c["presets"]}
    return {"controls": out, "floor": cal.get("floor"), "made": cal.get("made")}


@app.get("/api/calibration/clip/{folder}/{name}.flac")
def calibration_clip(folder: str, name: str):
    if not re.fullmatch(r"[a-z_]+(-[a-z_]+){0,2}", folder) or not re.fullmatch(r"[a-z-]+-s\d+--?[0-9.]+", name):
        return err("No such clip.", 404)
    p = ROOT / "data" / "calibration" / "clips" / folder / f"{name}.flac"
    if not p.exists():
        return err("No such clip.", 404)
    return FileResponse(p, media_type="audio/flac")


@app.post("/api/calibration/ears")
async def calibration_ears(request: Request):
    """Set (or with clear, remove) the range you hear as musical for a control; the knob is laid
    out over it from then on, and the engine's modulation follows."""
    from abliterator import calibrate
    b = await request.json()
    try:
        if b.get("clear"):
            calibrate.clear_ears(str(b.get("control")))
        else:
            calibrate.set_ears(str(b.get("control")), float(b["lo"]), float(b["hi"]))
    except (ValueError, KeyError, TypeError) as e:
        return err(str(e))
    if state["active"] == "mrt2" and mrt2.ready():
        try:
            mrt2.call("calibration", timeout=5)
        except Exception:
            pass
    return calibration()


@app.post("/api/scopes")
async def set_scopes(request: Request):
    """The scopes on the model path: {"layers": [1 to 12, at most 4]}. Meters: they change nothing
    and are not recorded in sounds. Magenta RT 2 only."""
    b = await request.json()
    if state["active"] != "mrt2":
        return err("The scope reads Magenta RT 2's stream.", 409)
    if (r := ready_or_409()) is not None:
        return r
    layers = b.get("layers") if isinstance(b, dict) else None
    if not isinstance(layers, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in layers):
        return err("Send the scopes' layers as a list of whole numbers.")
    try:
        return mrt2.set_scopes(layers)
    except Exception as e:
        return err(str(e))


@app.post("/api/mod")
async def set_mod(request: Request):
    """Modulation (mod.py): {"sources": {...}, "routes": [{"src", "tgt", "depth"}]}. Magenta RT 2 only."""
    b = await request.json()
    if state["active"] != "mrt2":
        return err("Modulation is written for Magenta RT 2.", 409)
    if (r := ready_or_409()) is not None:
        return r
    if not isinstance(b, dict):
        return err("Send the modulation as an object.")
    try:
        return mrt2.set_mod(b)
    except Exception as e:
        return err(str(e))


@app.post("/api/desk/play")
def desk_play():
    try:
        desk.play()
    except Exception as e:
        traceback.print_exc()
        return err(f"{type(e).__name__}: {e}", 500)
    return desk.state()


@app.post("/api/desk/stop")
def desk_stop():
    desk.stop()
    return desk.state()


@app.post("/api/desk/snapshot")
async def desk_snapshot(request: Request):
    b = await request.json()
    ti = int(b.get("track", 0))
    if not 0 <= ti < len(desk.tracks):
        return err("No such track.")
    seconds = max(0.5, min(60.0, float(b.get("seconds", 8))))
    try:
        snap = active().snapshot(seconds)
    except Exception as e:
        return err(f"No snapshot: {e}", 409)
    desk.add_clip(ti, snap)
    return {"snapshot": snap, "desk": desk.state()}


@app.get("/api/sounds")
def sounds():
    return {"sounds": library.sounds()}


@app.post("/api/sounds/snapshot")
async def sound_snapshot(request: Request):
    """Save the last few seconds of what the engine generated as a sound in the library."""
    b = await request.json()
    seconds = max(0.5, min(60.0, float(b.get("seconds", 8))))
    try:
        snap = active().snapshot(seconds)
    except Exception as e:
        return err(f"Nothing saved: {e}", 409)
    snap["label"] = library.rename(snap["snapshot"], str(b.get("name") or snap.get("label", "")))["name"]
    return {"snapshot": snap, "sounds": library.sounds()}


@app.get("/api/sounds/{sid}.wav")
def sound_wav(sid: str):
    if not library.exists(sid):
        return err("No such sound.", 404)
    heard = library.SOUNDS / f"{sid}.heard.wav"
    return FileResponse(heard if heard.exists() else library.SOUNDS / f"{sid}.wav", media_type="audio/wav")


@app.post("/api/sounds/{sid}")
async def sound_rename(sid: str, request: Request):
    b = await request.json()
    try:
        e = library.rename(sid, str(b.get("name", "")))
    except ValueError as ex:
        return err(str(ex), 404)
    desk.rename_sound(sid, e["name"])
    return {"sound": e, "sounds": library.sounds()}


@app.post("/api/sounds/{sid}/delete")
def sound_delete(sid: str):
    try:
        library.delete(sid)
    except ValueError as ex:
        return err(str(ex), 404)
    desk.drop_sound(sid)
    return {"sounds": library.sounds(), "desk": desk.state()}


@app.post("/api/desk/track/{ti}/add")
async def desk_add(ti: int, request: Request):
    b = await request.json()
    sid = str(b.get("sound", ""))
    if not 0 <= ti < len(desk.tracks) or not library.exists(sid):
        return err("Name a track and a sound.", 404)
    e = library.entry(sid)
    desk.add_clip(ti, {"snapshot": sid, "label": e["name"], "seconds": e["seconds"]})
    return desk.state()


@app.post("/api/desk/track/{ti}/trigger")
async def desk_trigger(ti: int, request: Request):
    b = await request.json()
    if not 0 <= ti < len(desk.tracks):
        return err("No such track.", 404)
    try:
        desk.trigger(ti, int(b.get("clip", -1)))
    except (ValueError, TypeError) as e:
        return err(str(e))
    except Exception as e:
        traceback.print_exc()
        return err(f"{type(e).__name__}: {e}", 500)
    return desk.state()


@app.post("/api/desk/track/{ti}/stop")
def desk_track_stop(ti: int):
    if not 0 <= ti < len(desk.tracks):
        return err("No such track.", 404)
    desk.stop_track(ti)
    return desk.state()


@app.post("/api/desk/track/{ti}")
async def desk_track(ti: int, request: Request):
    if not 0 <= ti < len(desk.tracks):
        return err("No such track.", 404)
    try:
        desk.update_track(ti, await request.json())
    except Exception as e:
        return err(f"{type(e).__name__}: {e}")
    return desk.state()


@app.post("/api/desk/mix")
async def desk_mix(request: Request):
    desk.update_mix(await request.json())
    return desk.state()


@app.post("/api/desk/export")
async def desk_export(request: Request):
    b = await request.json()
    try:
        return desk.export(b.get("what", ""), b.get("track"), b.get("clip"))
    except (ValueError, IndexError, TypeError) as e:
        return err(str(e))


NAME = re.compile(r"[A-Za-z0-9-]+")


@app.get("/api/exports")
def exports():
    folder = ROOT / "data" / "exports"
    return {"exports": sorted((p.stem for p in folder.glob("*.json")), reverse=True) if folder.exists() else []}


@app.get("/api/exports/{name}.{ext}")
def export_file(name: str, ext: str):
    p = ROOT / "data" / "exports" / f"{name}.{ext}"
    if ext not in ("wav", "json") or not NAME.fullmatch(name) or not p.exists():
        return err("No such export.", 404)
    return FileResponse(p, media_type="audio/wav" if ext == "wav" else "application/json", filename=p.name)


@app.get("/api/sessions/{name}.post.wav")
def session_post_wav(name: str):
    p = ROOT / "data" / "sessions" / f"{name}.post.wav"
    if not re.fullmatch(r"[0-9-]+", name) or not p.exists():
        return err("No such session.", 404)
    return FileResponse(p, media_type="audio/wav", filename=p.name)


@app.get("/api/sessions/{name}.wav")
def session_wav(name: str):
    p = ROOT / "data" / "sessions" / f"{name}.wav"
    if not re.fullmatch(r"[0-9-]+", name) or not p.exists():
        return err("No such session.", 404)
    return FileResponse(p, media_type="audio/wav")


@app.get("/")
def index():
    stamp = str(int(max(f.stat().st_mtime for f in WEB.iterdir())))
    html = (WEB / "index.html").read_text().replace('src="app.js"', f'src="app.js?v={stamp}"')
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


class Page(StaticFiles):
    async def get_response(self, path, scope):
        r = await super().get_response(path, scope)
        r.headers["Cache-Control"] = "no-cache"
        return r


app.mount("/", Page(directory=WEB), name="web")


if __name__ == "__main__":
    import uvicorn
    mrt2.start([p["prompt"] for p in presets()], {p["prompt"]: p["name"] for p in presets()})
    threading.Thread(target=_load, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
