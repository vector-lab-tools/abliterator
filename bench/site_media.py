# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Makes the website's sound examples from this machine's own renders: the brightness sweep on the
Warm drone preset, the Lookback mix sweep and every model-layer unit's sweep (bench/sweep.py,
bench/sweep_units.py), and the sounds saved in the library. Writes MP3s to docs/audio/ and their
measures and waveform peaks to docs/media.js and docs/units-media.js, which the site's figures
read. The renders are raised by a fixed 12 dB and the saved sounds to about −20 dBFS, each through
a limiter, since the model writes quietly. Needs ffmpeg.

    .venv-mrt2/bin/python bench/site_media.py [--sounds id,id,...]
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "backend")
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from abliterator import ROOT  # noqa: E402

DOCS = ROOT / "docs"
AUDIO = DOCS / "audio"
CLIPS = ROOT / "data" / "calibration" / "clips"
SWEEP = ROOT / "_tmp" / "sweep"
FFMPEG = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"


SWEEP_GAIN_DB = 12.0     # every sweep render raised by the same amount, so their levels still compare
SOUND_RMS_DB = -20.0     # each saved sound raised to about this level


def rms_db(path):
    x, _ = sf.read(str(path), always_2d=True)
    return 20 * np.log10(np.sqrt(np.mean(x.mean(axis=1) ** 2)) + 1e-12)


def mp3(src, dst, gain_db=0.0, kbps=160):
    """MP3, raised by gain_db through a limiter at −0.3 dBFS: the model writes quietly."""
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-i", str(src),
           "-af", f"volume={gain_db:.2f}dB,alimiter=limit=0.966:attack=5:release=50:level=disabled",
           "-ac", "2", "-ar", "48000", "-b:a", f"{kbps}k", str(dst)]
    subprocess.run(cmd, check=True)


def peaks(path, n=240):
    x, _ = sf.read(str(path), always_2d=True)
    m = x.mean(axis=1)
    k = max(1, len(m) // n)
    m = m[: k * n].reshape(n, k) if len(m) >= n else np.pad(m, (0, n - len(m))).reshape(n, 1)
    return [round(float(v), 3) for v in np.abs(m).max(axis=1)]


def rows(path, control, preset):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r["control"] == control and r["preset"] == preset:
                out[r["value"]] = r          # a later row replaces an earlier one, as the clip did
    return out


def sweep(control, preset, seed, rowsrc):
    cal = json.loads((ROOT / "data" / "calibration" / "mrt2_small.json").read_text())["controls"].get(control, {})
    folder = control.replace(":", "-")
    found = {}
    for p in rowsrc:
        found.update(rows(p, control, preset))
    steps = []
    for v in sorted(found):
        r = found[v]
        clip = CLIPS / folder / f"{preset}-s{seed}-{v:g}.flac"
        if not clip.exists():
            continue
        name = f"{folder}-{preset}-{v:g}.mp3".replace("--", "-m")
        mp3(clip, AUDIO / name, SWEEP_GAIN_DB)
        steps.append({"v": v, "src": f"audio/{name}", "rms_db": r["rms_db"], "centroid_hz": r["centroid_hz"],
                      "flatness": r["flatness"], "silence": r["silence"], "clip": r["clip"],
                      "ltas": [round(x, 1) for x in r["ltas"]], "peaks": peaks(clip, 160)})
    broken = {}
    for p in (cal.get("presets") or {}).get(preset, []):
        broken[p["value"]] = p.get("broken", [])
    for s in steps:
        s["broken"] = broken.get(s["v"], [])
    return {"control": control, "preset": preset, "seed": seed, "default": cal.get("default"),
            "safe": cal.get("safe"), "curve": cal.get("curve"), "steps": steps}


# the units' groups on the page: by what they act on
GROUPS = {"delay": "time", "reverb": "time", "beatrepeat": "time", "compressor": "size",
          "saturator": "element", "erosion": "element", "gate": "element", "limiter": "element",
          "eq": "subspace", "multiband": "subspace", "shifter": "subspace", "abliterate": "block", "neuron": "block"}


def units_media(preset="warm-drone", seed=1):
    """Every model-layer unit's sweep on one preset (bench/sweep_units.py): each setting's level,
    silence and verdict, its chain line, and its clip, for the units page."""
    from abliterator import units as U
    cal = json.loads((ROOT / "data" / "calibration" / "mrt2_small.json").read_text())["controls"]
    found = {}
    for f in (SWEEP / "units.jsonl", SWEEP / "units2.jsonl"):      # the second pass replaced clips of the same name
        if f.exists():
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r["preset"] == preset:
                    found[(r["control"], r["value"])] = r
    out = {"preset": preset, "seed": seed, "units": []}
    none = found.get(("unit:none", 0.0))
    clip = CLIPS / "unit-none" / f"{preset}-s{seed}-0.flac"
    if none and clip.exists():
        mp3(clip, AUDIO / "u-none.mp3", SWEEP_GAIN_DB, 96)
        out["none"] = {"src": "audio/u-none.mp3", "rms_db": none["rms_db"]}
    for u in U.catalogue():
        if not u["available"]:
            continue
        ctls = {}
        for key in [q["key"] for q in u["params"] if q["kind"] in ("knob", "fader", "int")] + ["at"]:
            name = f"unit:{u['id']}:{key}"
            c = cal.get(name, {})
            verdict = {p["value"]: p.get("broken", []) for p in (c.get("presets") or {}).get(preset, [])}
            others = {}                                          # what the same setting did on the other presets swept
            for pid, rows in (c.get("presets") or {}).items():
                if pid != preset:
                    for p in rows:
                        others.setdefault(p["value"], []).append({"preset": pid, "rms_db": p.get("rms_db"), "broken": p.get("broken", [])})
            steps = []
            for (ctl, v), r in sorted(found.items(), key=lambda kv: kv[0][1]):
                if ctl != name:
                    continue
                src = CLIPS / name.replace(":", "-") / f"{preset}-s{seed}-{v:g}.flac"
                if not src.exists():
                    continue
                fn = f"u-{u['id']}-{key}-{v:g}.mp3".replace("--", "-m")
                mp3(src, AUDIO / fn, SWEEP_GAIN_DB, 96)
                steps.append({"v": v, "src": f"audio/{fn}", "rms_db": r["rms_db"], "silence": r["silence"], "line": r["line"],
                              "centroid_hz": r["centroid_hz"], "broken": verdict.get(v, []), "others": others.get(v, [])})
            if steps:
                ctls[key] = {"steps": steps, "default": c.get("default"), "safe": c.get("safe"), "vau_default": c.get("vau_default"),
                             "moved_default": c.get("moved_default"), "safe_places": c.get("safe_places")}
        out["units"].append({"id": u["id"], "title": u["title"], "vau_title": u["vau_title"], "what": u["what"], "site": u["site"],
                             "group": GROUPS.get(u["id"], "other"), "example": u.get("example"),
                             "places": u.get("places"), "place_default": u.get("place_default"),
                             "params": [{k: q.get(k) for k in ("key", "label", "tip", "kind", "min", "max", "default", "safe", "fragile", "choices")}
                                        for q in u["params"]],
                             "controls": ctls})
    (DOCS / "units-media.js").write_text("// Made by bench/site_media.py from this machine's unit sweep. Do not edit by hand.\n"
                                         "window.UNITS = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"{len(out['units'])} units, {sum(len(c['steps']) for x in out['units'] for c in x['controls'].values())} clips", file=sys.stderr)


def design_media(preset="warm-drone", seed=1):
    """For the abliteration page: the model bent from inside set beside an audio effect doing the
    nearest thing to the same clip. An EQ's high shelf on the unbent render against a Steer toward
    brightness; a delay on the audio against Lookback (VAU's Delay) on the stream at layer 6."""
    from abliterator import measure as ms
    from abliterator import postfx
    tmp = ROOT / "_tmp" / "design"
    tmp.mkdir(parents=True, exist_ok=True)
    out = {}

    def fx(src, slot, name):
        x, sr = sf.read(str(src), always_2d=True)
        rack = postfx.Rack(x.shape[1])
        rack.update([{**slot, "on": True}, *[{"type": t, "on": False, "params": {}} for t in ("chorus", "delay", "reverb")]][:4])
        y = rack.process(x.astype(np.float32))
        f = tmp / f"{name}.flac"
        sf.write(str(f), np.clip(y, -1, 1), sr, subtype="PCM_16")
        mp3(f, AUDIO / f"{name}.mp3", SWEEP_GAIN_DB, 128)
        return y

    def info(src, x=None):
        if x is None:
            x, _ = sf.read(str(src), always_2d=True)
        m = ms.measure(x)
        return {"rms_db": m["rms_db"], "centroid_hz": m["centroid_hz"]}

    base = CLIPS / "dose-brightness" / f"{preset}-s{seed}-0.flac"
    bright = CLIPS / "dose-brightness" / f"{preset}-s{seed}-0.2.flac"
    y = fx(base, {"type": "eq", "params": {"low": -6.0, "mid": 0.0, "freq": 1000.0, "high": 12.0}}, "d-eq-bright")
    out["eq"] = {"base": {"src": "audio/dose-brightness-warm-drone-0.mp3", **info(base)},
                 "fx": {"src": "audio/d-eq-bright.mp3", "line": "EQ: low −6 dB, high +12 dB", **info(None, y)},
                 "model": {"src": "audio/dose-brightness-warm-drone-0.2.mp3", "line": "Steer: brightness +0.2 at layer 6", **info(bright)}}
    none = CLIPS / "unit-none" / f"{preset}-s{seed}-0.flac"
    look = CLIPS / "unit-delay-mix" / f"{preset}-s{seed}-0.2.flac"
    y = fx(none, {"type": "delay", "params": {"time": 160.0, "feedback": 0.3, "damp": 0.0, "pingpong": 0.0, "mix": 0.2}}, "d-delay-audio")
    out["delay"] = {"base": {"src": "audio/u-none.mp3", **info(none)},
                    "fx": {"src": "audio/d-delay-audio.mp3", "line": "Delay on the audio: 160 ms, feedback 0.3, mix 0.2", **info(None, y)},
                    "model": {"src": "audio/u-delay-mix-0.2.mp3", "line": "Lookback on the stream at layer 6: 4 frames (160 ms), feedback 0.3, mix 0.2", **info(look)}}
    (DOCS / "design-media.js").write_text("// Made by bench/site_media.py. Do not edit by hand.\n"
                                          "window.DESIGN = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(json.dumps(out, indent=1), file=sys.stderr)


# the website's spectrogram, as docs/spectro.js computes it: the same window, hop, bands, scale, colours
SP_N, SP_HOP, SP_ROWS, SP_FLO, SP_FHI, SP_TOP, SP_RANGE = 2048, 512, 150, 40.0, 20000.0, -22.0, 72.0
SP_STOPS = [(0, (7, 9, 12)), (0.3, (32, 30, 92)), (0.55, (123, 47, 190)), (0.78, (227, 164, 60)), (1, (255, 248, 228))]


def _png(path, rgb):
    """An 8-bit RGB PNG, written with zlib (no image library in the environments)."""
    import struct
    import zlib
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[r].tobytes() for r in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def spectrograms():
    """Each clip's spectrogram drawn once, as the page would draw it, into docs/spec/ as WebP, with an
    index (docs/spec-index.js) the page reads, so it shows the picture without computing it."""
    import tempfile
    out_dir = DOCS / "spec"
    out_dir.mkdir(exist_ok=True)
    lut = np.zeros((256, 3), np.uint8)
    for v in range(256):
        t = v / 255
        i = next(k for k in range(1, len(SP_STOPS)) if t <= SP_STOPS[k][0] or k == len(SP_STOPS) - 1)
        (t0, c0), (t1, c1) = SP_STOPS[i - 1], SP_STOPS[i]
        k = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
        lut[v] = [round(c0[j] + k * (c1[j] - c0[j])) for j in range(3)]
    sr = 48000
    win = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(SP_N) / (SP_N - 1))      # as the page's window
    ws = win.sum()
    edges = SP_FLO * (SP_FHI / SP_FLO) ** (np.arange(SP_ROWS + 1) / SP_ROWS)
    lo = np.maximum(1, np.floor(edges[:-1] / sr * SP_N)).astype(int)
    hi = np.maximum(np.floor(edges[:-1] / sr * SP_N).astype(int) + 1, np.ceil(edges[1:] / sr * SP_N).astype(int))
    hi = np.minimum(hi, SP_N // 2)
    freqs = np.arange(SP_N // 2) * sr / SP_N
    index = {}
    for f in sorted(AUDIO.glob("*.mp3")):
        raw = subprocess.run([FFMPEG, "-loglevel", "error", "-i", str(f), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                             check=True, capture_output=True).stdout
        x = np.frombuffer(raw, np.float32).astype(np.float64)
        frames = max(1, (len(x) - SP_N) // SP_HOP + 1)
        idx = np.arange(SP_N)[None, :] + SP_HOP * np.arange(frames)[:, None]
        xx = np.pad(x, (0, max(0, SP_N - len(x))))[np.minimum(idx, len(x) - 1)] * win
        X = np.fft.rfft(xx, axis=1)[:, : SP_N // 2]
        P = (X.real ** 2 + X.imag ** 2) * 4 / ws ** 2
        cen = float((P * freqs).sum() / P.sum()) if P.sum() > 0 else 0.0
        m = np.stack([P[:, a:b].max(axis=1) for a, b in zip(lo, hi)], axis=1)
        v = np.clip((10 * np.log10(m + 1e-14) - (SP_TOP - SP_RANGE)) / SP_RANGE * 255, 0, 255).astype(np.uint8)
        img = lut[v.T[::-1]]                                          # rows top (20 kHz) to bottom, columns in time
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            _png(tmp.name, np.ascontiguousarray(img))
            subprocess.run(["cwebp", "-quiet", "-q", "90", tmp.name, "-o", str(out_dir / (f.stem + ".webp"))], check=True)
        index[f"audio/{f.name}"] = {"img": f"spec/{f.stem}.webp", "frames": frames, "seconds": round(len(x) / sr, 3), "centroid": round(cen, 1)}
    (DOCS / "spec-index.js").write_text("// Made by bench/site_media.py: each clip's spectrogram, drawn once. Do not edit by hand.\n"
                                        "window.SPECS = " + json.dumps(index, separators=(",", ":")) + ";\n")
    print(f"{len(index)} spectrograms", file=sys.stderr)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--sounds", default="")
    a.add_argument("--units-only", action="store_true")
    a.add_argument("--design-only", action="store_true")
    a.add_argument("--spectrograms-only", action="store_true")
    args = a.parse_args()
    AUDIO.mkdir(parents=True, exist_ok=True)
    if args.spectrograms_only:
        spectrograms()
        return
    if args.design_only:
        design_media()
        return
    units_media()
    design_media()
    if args.units_only:
        return
    AUDIO.mkdir(parents=True, exist_ok=True)
    media = {
        "brightness": sweep("dose:brightness", "warm-drone", 1, [SWEEP / "results.jsonl"]),
        "lookback": sweep("unit:delay:mix", "warm-drone", 1, [SWEEP / "units.jsonl", SWEEP / "units2.jsonl"]),
    }
    none = rows(SWEEP / "units.jsonl", "unit:none", "warm-drone").get(0.0)
    media["lookback"]["none_rms_db"] = none["rms_db"] if none else None
    from abliterator import library
    want = [s for s in args.sounds.split(",") if s]
    sounds = []
    for e in library.sounds():
        if want and e["id"] not in want:
            continue
        src = ROOT / "data" / "snapshots" / f"{e['id']}.heard.wav"
        if not src.exists():
            src = ROOT / "data" / "snapshots" / f"{e['id']}.wav"
        if not src.exists():
            continue
        mp3(src, AUDIO / f"sound-{e['id']}.mp3", min(30.0, SOUND_RMS_DB - rms_db(src)))
        rec = json.loads((ROOT / "data" / "snapshots" / f"{e['id']}.json").read_text())
        sounds.append({"id": e["id"], "label": e["name"], "seconds": e.get("seconds"),
                       "preset": (rec.get("preset") or {}).get("name"), "prompt": (rec.get("preset") or {}).get("prompt"),
                       "seed": rec.get("seed"), "frozen": bool(rec.get("frozen")),
                       "chain": e["strips"],
                       "src": f"audio/sound-{e['id']}.mp3", "peaks": peaks(src)})
    media["sounds"] = sounds
    spectrograms()
    (DOCS / "media.js").write_text("// Made by bench/site_media.py from this machine's renders. Do not edit by hand.\n"
                                   "window.MEDIA = " + json.dumps(media, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(f"{len(media['brightness']['steps'])} brightness steps, {len(media['lookback']['steps'])} Lookback steps, "
          f"{len(sounds)} sounds", file=sys.stderr)


if __name__ == "__main__":
    main()
