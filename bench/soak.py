# Abliterator: a soak run through the server's API, as the page drives it.
# Copyright (C) 2026 David M. Berry
# SPDX-License-Identifier: GPL-3.0-or-later
"""
Plays for --seconds, moving the XY pad every 3 s, changing the preset every 60 s (on Magenta RT 2
a live style change) and the guidance at the half-way point, and samples the status every 5 s.

    .venv/bin/python bench/soak.py --engine mrt2 --seconds 600
"""
import argparse
import json
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--engine", default="mrt2")
ap.add_argument("--seconds", type=float, default=600)
ap.add_argument("--port", type=int, default=8775)
ap.add_argument("--lead", type=int, default=None)
a = ap.parse_args()
B = f"http://127.0.0.1:{a.port}"


def api(path, body=None):
    req = urllib.request.Request(B + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "X-Abliterator": "1"})
    return json.load(urllib.request.urlopen(req, timeout=30))


api("/api/engine", {"engine": a.engine})
if a.lead:
    api("/api/lead", {"lead": a.lead})
pre = api("/api/presets")["presets"]
api("/api/play", {"preset": pre[0]["id"], "seed": pre[0]["seeds"][0]})
pts = [(0.25, 0.0), (-0.25, 0.2), (0.0, -0.3), (0.35, 0.3), (-0.3, -0.2), (0.0, 0.0)]
t0 = time.time()
samples, lat, nxt_move, nxt_status, nxt_preset, k, pi, guided = [], [], 3, 5, 60, 0, 0, False
while (el := time.time() - t0) < a.seconds:
    time.sleep(0.05)
    if el >= nxt_move:
        x, y = pts[k % len(pts)]
        k += 1
        api("/api/control", {"doses": {pre[0]["x"]: x, pre[0]["y"]: y}})
        nxt_move += 3
    if el >= nxt_preset:
        pi += 1
        if a.engine == "mrt2":
            api("/api/control", {"preset": pre[pi % len(pre)]["id"]})
        nxt_preset += 60
    if a.engine == "mrt2" and not guided and el >= a.seconds / 2:
        api("/api/control", {"sampling": {"guidance": 5.0}})
        guided = True
    if el >= nxt_status:
        s = api("/api/status")
        samples.append({"t": round(el, 1), "seconds": s.get("seconds"), "underruns": s.get("underruns"),
                        "step_p50": s.get("step_ms_p50"), "step_p95": s.get("step_ms_p95")})
        for m in s["latency"]["measured_ms"]:
            lat.append(m)
        nxt_status += 5
s = api("/api/status")
saved = api("/api/stop", {})
lat = sorted(set(lat))
res = {"engine": a.engine, "seconds": a.seconds, "played_s": s.get("seconds"), "underruns": s.get("underruns"),
       "lead": s.get("lead"), "device_ms": s["latency"]["parts"].get("device"),
       "step_p50_last": s.get("step_ms_p50"), "step_p95_last": s.get("step_ms_p95"),
       "step_p95_max": max(x["step_p95"] or 0 for x in samples),
       "latency_ms_seen": lat, "latency_median": sorted(lat)[len(lat) // 2] if lat else None,
       "moves": k, "preset_changes": pi, "saved": saved, "samples": samples}
json.dump(res, open(f"_tmp/soak_{a.engine}_lead{s.get('lead')}.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "samples"}))
