# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
The tracks: eight of them, each holding sounds from the library (library.py) as clips, with its
own audio effects (fx.py), a level, a pan, mute and solo, summed into a mix bus with a soft limiter.

A clip is triggered: it starts from its beginning on its track (stopping the clip that track was
playing) and loops, or plays once, as the track is set. A stopped track's effects run on, so a
delay or reverb tail dies away. The desk plays on its own output stream (48 kHz stereo) beside the
live engine's, so it works with either engine and its latency is the sound card's alone (one block,
about 11 ms, plus the device's). MusicGen sounds (32 kHz mono) are resampled to 48 kHz and set on
both channels when loaded.

Everything can be exported as a WAV with a JSON sidecar (exports in data/exports/):
  clip pre    the sound as its track loads it (load_clip)
  clip post   the same through its track's effects, level and pan, with the effects' tail
  track pre   the track's current clip, looped to the mix length, before its effects
  track post  the same after its effects, level and pan
  mix         all tracks that would be heard (mute and solo applied), through the mix bus
"""
from __future__ import annotations

import json
import threading
import time
from math import gcd
from typing import Any, Dict, List, Optional

import numpy as np

from . import ROOT, TOOL, VERSION
from . import fx

SR = fx.SR
N_TRACKS = 8
BLOCK = 512
LEVEL_FLOOR = -60.0          # the foot of a track fader: silence
RELEASE_DB = 24.0            # the meters' release, dB a second
TAIL_S = 3.0
SNAPSHOTS = ROOT / "data" / "snapshots"
EXPORTS = ROOT / "data" / "exports"
STATE = ROOT / "data" / "tracks" / "desk.json"


def load_clip(sid: str) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import resample_poly
    heard = SNAPSHOTS / f"{sid}.heard.wav"             # as heard, when the sound has it; else the model's output
    a, sr = sf.read(heard if heard.exists() else SNAPSHOTS / f"{sid}.wav", dtype="float32", always_2d=True)
    if sr != SR:
        g = gcd(SR, sr)
        a = resample_poly(a, SR // g, sr // g, axis=0).astype(np.float32)
    if a.shape[1] == 1:
        a = np.repeat(a, 2, axis=1)
    return a


def snapshot_recipe(sid: str) -> Dict[str, Any]:
    p = SNAPSHOTS / f"{sid}.json"
    return json.loads(p.read_text()) if p.exists() else {}


class Track:
    def __init__(self, i: int):
        self.i = i
        self.name = f"Track {i + 1}"
        self.clips: List[Dict[str, Any]] = []        # {"id", "label", "seconds"}
        self.current: Optional[int] = None           # the clip last triggered or chosen (exports use it)
        self.live = False                            # whether it is sounding
        self.loop = True
        self.level_db = 0.0
        self.peak = 0.0                              # the track's peak after its effects and level, with a release
        self.mute = False
        self.solo = False
        self.fx = fx.clean({})
        self.chain = fx.Chain()
        self.audio: Optional[np.ndarray] = None
        self.pos = 0

    def state(self) -> Dict[str, Any]:
        return {"i": self.i, "name": self.name, "clips": self.clips, "current": self.current, "level_db": self.level_db,
                "mute": self.mute, "solo": self.solo, "fx": self.fx, "loop": self.loop}

    def play_state(self) -> Dict[str, Any]:
        n = len(self.audio) if self.audio is not None else 0
        return {"live": self.live, "pos_s": round(self.pos / SR, 2), "len_s": round(n / SR, 2), "peak": round(self.peak, 4)}

    def select(self, k: Optional[int], audio: Optional[np.ndarray] = None):
        self.current = k
        ok = k is not None and 0 <= k < len(self.clips)
        self.audio = (audio if audio is not None else load_clip(self.clips[k]["id"])) if ok else None
        self.pos = 0
        if not ok:
            self.live = False

    def block(self, n: int) -> np.ndarray:
        a = self.audio
        if not self.live or a is None or not len(a):
            return np.zeros((n, 2), np.float32)
        if self.loop:
            idx = (self.pos + np.arange(n)) % len(a)
            self.pos = (self.pos + n) % len(a)
            return a[idx]
        y = np.zeros((n, 2), np.float32)
        k = min(n, len(a) - self.pos)
        y[:k] = a[self.pos:self.pos + k]
        self.pos += k
        if self.pos >= len(a):
            self.live, self.pos = False, 0
        return y


def gain(level_db: float) -> float:
    """A fader's gain: the foot of its travel (-60 dB, drawn as ∞) is silence."""
    return 0.0 if level_db <= LEVEL_FLOOR else 10 ** (level_db / 20)


def limiter(x: np.ndarray) -> np.ndarray:
    return (np.tanh(1.5 * x) / np.tanh(1.5)).astype(np.float32)


class Desk:
    def __init__(self):
        self.tracks = [Track(i) for i in range(N_TRACKS)]
        self.playing = False
        self.mix_volume = 0.8
        self.limit = True
        self.stream = None
        self.lock = threading.RLock()
        self.level = 0.0
        self.level_lr = [0.0, 0.0]                 # each channel's level, for the MIX meter's two bars
        self.peak = 0.0                            # the mix's peak, with the same release as the tracks'
        self.underruns = 0
        from abliterator.spectro import Spectro
        self.spectro = Spectro(SR)
        self._load()

    def state(self) -> Dict[str, Any]:
        return {"playing": self.playing, "mix_volume": self.mix_volume, "limit": self.limit, "level": round(self.level, 4), "level_lr": [round(v, 4) for v in self.level_lr], "peak": round(self.peak, 4),
                "tracks": [{**t.state(), **t.play_state()} for t in self.tracks], "fx_defaults": fx.DEFAULTS, "fx_ranges": fx.RANGES,
                "latency_ms": round(self.stream.latency * 1000, 1) if self.stream else None, "spec": self.spectro.payload()}

    def _save(self):
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps({"mix_volume": self.mix_volume, "limit": self.limit,
                                     "tracks": [{k: v for k, v in t.state().items()} for t in self.tracks]}, indent=1))

    def _load(self):
        if not STATE.exists():
            return
        try:
            d = json.loads(STATE.read_text())
            self.mix_volume, self.limit = float(d.get("mix_volume", 0.8)), bool(d.get("limit", True))
            for t, s in zip(self.tracks, d.get("tracks", [])):
                t.name = s.get("name", t.name)
                t.clips = [c for c in s.get("clips", []) if (SNAPSHOTS / f"{c['id']}.wav").exists()]
                t.level_db, t.mute, t.solo = float(s.get("level_db", 0)), bool(s.get("mute")), bool(s.get("solo"))
                t.loop = bool(s.get("loop", True))
                t.fx = fx.clean(s.get("fx", {}))
                cur = s.get("current")
                t.select(cur if cur is not None and cur < len(t.clips) else (len(t.clips) - 1 if t.clips else None))
        except Exception:
            import traceback
            traceback.print_exc()

    def add_clip(self, ti: int, snap: Dict[str, Any]):
        audio = load_clip(snap["snapshot"])
        with self.lock:
            t = self.tracks[ti]
            t.clips.append({"id": snap["snapshot"], "label": snap.get("label", snap["snapshot"]), "seconds": snap.get("seconds")})
            if not t.live:
                t.select(len(t.clips) - 1, audio)
            self._save()

    def trigger(self, ti: int, k: int):
        """Start clip k on track ti from its beginning (opening the desk's stream if it is shut)."""
        t = self.tracks[ti]
        if not 0 <= k < len(t.clips):
            raise ValueError("no such clip")
        audio = load_clip(t.clips[k]["id"])
        with self.lock:
            t.select(k, audio)
            t.live = True
        self._open()

    def stop_track(self, ti: int):
        with self.lock:
            self.tracks[ti].live = False
            self.tracks[ti].pos = 0

    def rename_sound(self, sid: str, label: str):
        with self.lock:
            for t in self.tracks:
                for c in t.clips:
                    if c["id"] == sid:
                        c["label"] = label
            self._save()

    def drop_sound(self, sid: str):
        with self.lock:
            for t in self.tracks:
                cur = t.clips[t.current]["id"] if t.current is not None and t.current < len(t.clips) else None
                t.clips = [c for c in t.clips if c["id"] != sid]
                if cur == sid or t.current is not None and t.current >= len(t.clips):
                    t.select(len(t.clips) - 1 if t.clips else None)
                elif cur is not None:
                    t.current = next(i for i, c in enumerate(t.clips) if c["id"] == cur)
            self._save()

    def update_track(self, ti: int, b: Dict[str, Any]):
        with self.lock:
            t = self.tracks[ti]
            if "name" in b:
                t.name = str(b["name"])[:40] or t.name
            if "level_db" in b:
                t.level_db = float(min(12, max(-60, float(b["level_db"]))))
            for k in ("mute", "solo", "loop"):
                if k in b:
                    setattr(t, k, bool(b[k]))
            if "fx" in b:
                t.fx = fx.clean({**t.fx, **b["fx"]})
            if "current" in b:
                k = b["current"]
                live = t.live
                t.select(None if k is None else int(k))
                t.live = live and t.audio is not None
            if b.get("remove_clip") is not None:
                k = int(b["remove_clip"])
                if 0 <= k < len(t.clips):
                    t.clips.pop(k)
                    if t.current == k:
                        t.select(min(k, len(t.clips) - 1) if t.clips else None)
                    elif t.current is not None and t.current > k:
                        t.current -= 1
            if b.get("clear"):
                t.clips = []
                t.select(None)
            self._save()

    def update_mix(self, b: Dict[str, Any]):
        if "mix_volume" in b:
            self.mix_volume = float(min(1.5, max(0, float(b["mix_volume"]))))
        if "limit" in b:
            self.limit = bool(b["limit"])
        self._save()

    def heard(self) -> List[Track]:
        """The tracks that would be heard: a clip chosen, not muted, soloed if any track is."""
        solo = any(t.solo for t in self.tracks)
        return [t for t in self.tracks if t.audio is not None and not t.mute and (t.solo or not solo)]

    def _mix_block(self, n: int, tracks: List[Track]) -> np.ndarray:
        fall = 10 ** (-RELEASE_DB * n / SR / 20)
        y = np.zeros((n, 2), np.float32)
        for t in self.tracks:
            t.peak *= fall
        for t in tracks:
            yt = t.chain.process(t.block(n), t.fx) * gain(t.level_db)
            t.peak = max(t.peak, float(np.abs(yt).max()) if len(yt) else 0.0)
            y += yt
        if self.limit:
            y = limiter(y)
        y = y * self.mix_volume
        self.peak = max(self.peak * fall, float(np.abs(y).max()) if len(y) else 0.0)
        return y

    def _callback(self, out, frames, ti, status):
        try:
            with self.lock:
                y = self._mix_block(frames, self.heard())
            out[:] = y
            self.spectro.feed(y)
            self.level = float(np.sqrt((y ** 2).mean()))
            self.level_lr = [float(v) for v in np.sqrt((y ** 2).mean(axis=0))]
        except Exception:
            out.fill(0)
            self.underruns += 1

    def _open(self):
        import sounddevice as sd
        with self.lock:
            if self.playing:
                return
            for t in self.tracks:
                t.chain = fx.Chain()
            self.spectro.reset()
            self.stream = sd.OutputStream(samplerate=SR, channels=2, blocksize=BLOCK, latency="low", callback=self._callback)
            self.playing = True
            self.stream.start()

    def play(self):
        """Every track that has a clip starts it from the beginning, together."""
        with self.lock:
            for t in self.tracks:
                t.pos = 0
                t.live = t.audio is not None
        self._open()

    def stop(self):
        """Stops every track and shuts the desk's stream."""
        with self.lock:
            for t in self.tracks:
                t.live, t.pos = False, 0
            if not self.playing:
                return
            self.playing = False
        self.stream.stop()
        self.stream.close()
        self.level = self.peak = 0.0
        self.level_lr = [0.0, 0.0]
        for t in self.tracks:
            t.peak = 0.0

    def mix_length(self) -> int:
        ls = [len(t.audio) for t in self.heard()]
        return max(ls) if ls else 0

    def _render(self, t: Track, audio: np.ndarray, effects: bool, tail: float = 0.0) -> np.ndarray:
        if not effects:
            return audio
        ch = fx.Chain()
        x = np.concatenate([audio, np.zeros((int(tail * SR), 2), np.float32)])
        g = gain(t.level_db)
        return np.concatenate([ch.process(x[i:i + BLOCK], t.fx) * g for i in range(0, len(x), BLOCK)])

    def _loop(self, t: Track, n: int) -> np.ndarray:
        return t.audio[np.arange(n) % len(t.audio)]

    def export(self, what: str, ti: Optional[int] = None, clip: Optional[int] = None) -> Dict[str, Any]:
        import soundfile as sf
        with self.lock:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            side: Dict[str, Any] = {"format": "abliterator.export.v1", "tool": {"name": TOOL, "version": VERSION},
                                    "what": what, "sample_rate": SR, "channels": 2, "made": time.strftime("%Y-%m-%d %H:%M:%S")}
            if what in ("clip_pre", "clip_post"):
                t = self.tracks[ti]
                k = t.current if clip is None else clip
                if k is None or not (0 <= k < len(t.clips)):
                    raise ValueError("that track has no clip")
                c = t.clips[k]
                audio = load_clip(c["id"])
                y = self._render(t, audio, what == "clip_post", TAIL_S)
                name = f"{stamp}-t{ti + 1}-{c['id']}-{'post' if what == 'clip_post' else 'pre'}"
                side.update(track=t.state(), clip=c, snapshot=snapshot_recipe(c["id"]),
                            effects=(t.fx if what == "clip_post" else None), level_db=(t.level_db if what == "clip_post" else None),
                            tail_seconds=(TAIL_S if what == "clip_post" else 0))
            elif what in ("track_pre", "track_post"):
                t = self.tracks[ti]
                if t.audio is None:
                    raise ValueError("that track has no clip")
                n = max(self.mix_length(), len(t.audio))
                y = self._render(t, self._loop(t, n), what == "track_post")
                name = f"{stamp}-t{ti + 1}-{'post' if what == 'track_post' else 'pre'}"
                side.update(track=t.state(), snapshot=snapshot_recipe(t.clips[t.current]["id"]), looped_to_seconds=round(n / SR, 2),
                            effects=(t.fx if what == "track_post" else None))
            elif what == "mix":
                tracks = self.heard()
                n = self.mix_length()
                if not n:
                    raise ValueError("no track would be heard")
                y = np.zeros((n, 2), np.float32)
                for t in tracks:
                    y += self._render(t, self._loop(t, n), True)
                if self.limit:
                    y = limiter(y)
                y = y * self.mix_volume
                name = f"{stamp}-mix"
                side.update(tracks=[t.state() for t in tracks], snapshots={t.clips[t.current]["id"]: snapshot_recipe(t.clips[t.current]["id"]) for t in tracks},
                            mix_volume=self.mix_volume, limit=self.limit, seconds=round(n / SR, 2))
            else:
                raise ValueError(f"unknown export '{what}'")
        EXPORTS.mkdir(parents=True, exist_ok=True)
        sf.write(EXPORTS / f"{name}.wav", y.astype(np.float32), SR, subtype="FLOAT")
        side["audio"] = f"{name}.wav"
        side["peak"] = float(np.abs(y).max()) if len(y) else 0.0
        (EXPORTS / f"{name}.json").write_text(json.dumps(side, indent=1))
        return {"export": name, "seconds": round(len(y) / SR, 2), "peak": side["peak"]}
