# Abliterator
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Measures of a stretch of the model's output, for finding where each control stays musical
(bench/sweep.py, calibrate.py). numpy and scipy only.

  rms_db       level, dB below full scale (mean of the channels)
  peak         largest sample
  clip         share of samples at or above 0.98
  silence      share of 40 ms frames quieter than −55 dBFS
  centroid_hz  spectral centroid (brightness)
  flatness     spectral flatness, mean over frames: 0 for a pure tone, toward 1 for noise
  ltas         long-term average spectrum in 48 log-spaced bands, 40 Hz to 20 kHz, dB
  flux         mean spectral change between frames (movement), level-normalised

`timbre_distance(a, b)` compares two measures' spectra with their levels taken out: the root
mean square difference, in dB, of the band levels after each spectrum's mean is subtracted, over
the bands above −100 dB in both.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

SR = 48000
N_FFT = 2048
HOP = 1024
BANDS = np.geomspace(40, 20000, 49)


def measure(x: np.ndarray, sr: int = SR) -> Dict[str, Any]:
    x = np.asarray(x, np.float64)
    if x.ndim == 2:
        mono = x.mean(axis=1)
        peak = float(np.abs(x).max()) if x.size else 0.0
        clip = float((np.abs(x) >= 0.98).mean()) if x.size else 0.0
    else:
        mono = x
        peak = float(np.abs(x).max()) if x.size else 0.0
        clip = float((np.abs(x) >= 0.98).mean()) if x.size else 0.0
    rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
    fr = int(0.04 * sr)
    n = len(mono) // fr
    frms = np.sqrt((mono[:n * fr].reshape(n, fr) ** 2).mean(1)) if n else np.zeros(1)
    silence = float((20 * np.log10(frms + 1e-12) < -55).mean())
    win = np.hanning(N_FFT)
    k = max(0, (len(mono) - N_FFT) // HOP + 1)
    if k == 0:
        return {"rms_db": 20 * np.log10(rms + 1e-12), "peak": peak, "clip": clip, "silence": silence,
                "centroid_hz": 0.0, "flatness": 0.0, "ltas": [-120.0] * (len(BANDS) - 1), "flux": 0.0}
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(k)[:, None]
    S = np.abs(np.fft.rfft(mono[idx] * win, axis=1)) ** 2 + 1e-18
    f = np.fft.rfftfreq(N_FFT, 1 / sr)
    P = S.mean(0)
    centroid = float((f * P).sum() / P.sum())
    flat = float(np.mean(np.exp(np.log(S[:, 1:]).mean(1)) / S[:, 1:].mean(1)))
    ltas = []
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        m = (f >= lo) & (f < hi)
        ltas.append(float(10 * np.log10(P[m].mean() if m.any() else 1e-18)))
    L = 10 * np.log10(S)
    L = L - L.mean(1, keepdims=True)
    flux = float(np.mean(np.sqrt(np.mean(np.diff(L, axis=0) ** 2, axis=1)))) if k > 1 else 0.0
    return {"rms_db": round(20 * np.log10(rms + 1e-12), 2), "peak": round(peak, 4), "clip": round(clip, 6),
            "silence": round(silence, 4), "centroid_hz": round(centroid, 1), "flatness": round(flat, 5),
            "ltas": [round(v, 2) for v in ltas], "flux": round(flux, 3)}


def timbre_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    la, lb = np.array(a["ltas"]), np.array(b["ltas"])
    ok = (la > -100) & (lb > -100)
    if not ok.any():
        return 0.0
    da, db = la[ok] - la[ok].mean(), lb[ok] - lb[ok].mean()
    return round(float(np.sqrt(np.mean((da - db) ** 2))), 3)
