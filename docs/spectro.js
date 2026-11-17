// Abliterator: spectrograms of the clips. Each is drawn once when the site is built
// (bench/site_media.py, docs/spec/, indexed in spec-index.js); a clip without one is computed here. A box
// `<div class="specbox" data-src="audio/….mp3" data-label="…" data-line="…" data-note="…">` becomes a
// play button, the clip's spectrogram (40 Hz to 20 kHz on a log axis, a fixed dB scale, so boxes
// compare), and a playhead while it plays. `Spectro.show(box, src, label)` changes a box's clip.
// Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
(function () {
  "use strict";
  var N = 2048, HOP = 512, ROWS = 150, FLO = 40, FHI = 20000, DB_TOP = -22, DB_RANGE = 72;
  var cache = {}, octx = null;
  function ctx() {
    if (!octx) { var O = window.OfflineAudioContext || window.webkitOfflineAudioContext; octx = new O(1, 48000, 48000); }
    return octx;
  }
  // The analysis runs in a worker, one clip at a time, so the page stays responsive while it scrolls
  // and plays; where workers are not available it runs here.
  var worker = null, jobs = {}, jobId = 0, queue = Promise.resolve();
  try {
    var code = "var N=" + N + ",HOP=" + HOP + ",ROWS=" + ROWS + ",FLO=" + FLO + ",FHI=" + FHI + ";" + fft.toString() + analyse.toString() +
      "onmessage=function(e){var A=analyse(e.data.x,e.data.sr);postMessage({id:e.data.id,img:A.img,frames:A.frames,seconds:A.seconds,centroid:A.centroid},[A.img.buffer]);};";
    worker = new Worker(URL.createObjectURL(new Blob([code], { type: "text/javascript" })));
    worker.onmessage = function (e) { var j = jobs[e.data.id]; delete jobs[e.data.id]; if (j) j(e.data); };
  } catch (e) { worker = null; }
  function analyseOff(x, sr) {
    if (!worker) return Promise.resolve(analyse(x, sr));
    return new Promise(function (ok) { var id = ++jobId; jobs[id] = ok; worker.postMessage({ id: id, x: x, sr: sr }, [x.buffer]); });
  }
  function decode(src) {
    if (!cache[src]) cache[src] = queue = queue.then(function () {
      return fetch(src).then(function (r) { return r.arrayBuffer(); }).then(function (buf) {
        return new Promise(function (ok, bad) { var p = ctx().decodeAudioData(buf, ok, bad); if (p && p.then) p.then(ok, bad); });
      }).then(function (ab) {
        var n = ab.length, x = new Float32Array(n);
        for (var c = 0; c < ab.numberOfChannels; c++) { var d = ab.getChannelData(c); for (var i = 0; i < n; i++) x[i] += d[i] / ab.numberOfChannels; }
        return analyseOff(x, ab.sampleRate);
      });
    }).catch(function (e) { delete cache[src]; throw e; });
    return cache[src];
  }
  // radix-2 FFT, in place
  function fft(re, im) {
    var n = re.length, i, j, k, l, t;
    for (i = 1, j = 0; i < n; i++) { var bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (l = 2; l <= n; l <<= 1) {
      var a = -2 * Math.PI / l, wr = Math.cos(a), wi = Math.sin(a);
      for (i = 0; i < n; i += l) {
        var cr = 1, ci = 0;
        for (k = 0; k < l / 2; k++) {
          var ur = re[i + k], ui = im[i + k], vr = re[i + k + l / 2] * cr - im[i + k + l / 2] * ci, vi = re[i + k + l / 2] * ci + im[i + k + l / 2] * cr;
          re[i + k] = ur + vr; im[i + k] = ui + vi; re[i + k + l / 2] = ur - vr; im[i + k + l / 2] = ui - vi;
          t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
        }
      }
    }
  }
  function analyse(x, sr) {
    var win = new Float32Array(N), ws = 0, i;
    for (i = 0; i < N; i++) { win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (N - 1)); ws += win[i]; }
    var frames = Math.max(1, Math.floor((x.length - N) / HOP) + 1), img = new Float32Array(frames * ROWS);
    var lo = [], hi = [];
    for (var r = 0; r < ROWS; r++) {
      var f0 = FLO * Math.pow(FHI / FLO, r / ROWS), f1 = FLO * Math.pow(FHI / FLO, (r + 1) / ROWS);
      lo.push(Math.max(1, Math.floor(f0 / sr * N))); hi.push(Math.max(Math.floor(f0 / sr * N) + 1, Math.ceil(f1 / sr * N)));
    }
    var re = new Float32Array(N), im = new Float32Array(N), P = new Float32Array(N / 2), cnum = 0, cden = 0;
    for (var f = 0; f < frames; f++) {
      for (i = 0; i < N; i++) { re[i] = (x[f * HOP + i] || 0) * win[i]; im[i] = 0; }
      fft(re, im);
      for (i = 0; i < N / 2; i++) { P[i] = (re[i] * re[i] + im[i] * im[i]) * 4 / (ws * ws); cnum += P[i] * i * sr / N; cden += P[i]; }
      for (r = 0; r < ROWS; r++) {
        var m = 0;
        for (var b = lo[r]; b < hi[r] && b < N / 2; b++) m = Math.max(m, P[b]);
        img[f * ROWS + r] = 10 * Math.log10(m + 1e-14);
      }
    }
    return { img: img, frames: frames, seconds: x.length / sr, centroid: cden ? cnum / cden : 0 };
  }
  // dark, violet, amber, white
  var STOPS = [[0, [7, 9, 12]], [0.3, [32, 30, 92]], [0.55, [123, 47, 190]], [0.78, [227, 164, 60]], [1, [255, 248, 228]]];
  function colour(t) {
    for (var i = 1; i < STOPS.length; i++) if (t <= STOPS[i][0]) {
      var a = STOPS[i - 1], b = STOPS[i], k = (t - a[0]) / (b[0] - a[0]);
      return [a[1][0] + k * (b[1][0] - a[1][0]), a[1][1] + k * (b[1][1] - a[1][1]), a[1][2] + k * (b[1][2] - a[1][2])];
    }
    return STOPS[STOPS.length - 1][1];
  }
  function paint(cv, A) {
    cv.width = A.frames; cv.height = ROWS;
    var g = cv.getContext("2d"), id = g.createImageData(A.frames, ROWS);
    for (var f = 0; f < A.frames; f++) for (var r = 0; r < ROWS; r++) {
      var t = Math.max(0, Math.min(1, (A.img[f * ROWS + r] - (DB_TOP - DB_RANGE)) / DB_RANGE)), c = colour(t), o = ((ROWS - 1 - r) * A.frames + f) * 4;
      id.data[o] = c[0]; id.data[o + 1] = c[1]; id.data[o + 2] = c[2]; id.data[o + 3] = 255;
    }
    g.putImageData(id, 0, 0);
  }
  function hz(f) { return f >= 1000 ? (f / 1000).toFixed(2).replace(/0$/, "") + " kHz" : Math.round(f) + " Hz"; }
  var boxes = [];
  function build(box) {
    if (box._built) return; box._built = true;
    var label = box.getAttribute("data-label") || "", line = box.getAttribute("data-line") || "", note = box.getAttribute("data-note") || "";
    box.innerHTML = '<div class="sbhead"><button type="button" class="sbplay" aria-label="Play"><span>▶</span></button><span class="sblabel"></span><span class="sbnum"></span></div>' +
      '<div class="sbview"><canvas class="sbcv" aria-hidden="true"></canvas><i class="sbhead-line"></i><span class="sbax"><b>20k</b><b>1k</b><b>100</b></span><span class="sbwait">computing the spectrogram</span></div>' +
      (line ? '<code class="sbline"></code>' : "") + (note ? '<p class="sbnote"></p>' : "");
    box.querySelector(".sblabel").textContent = label;
    if (line) box.querySelector(".sbline").textContent = line;
    if (note) box.querySelector(".sbnote").textContent = note;
    box.querySelector(".sbplay").addEventListener("click", function () { if (window.AA) window.AA.play(box.getAttribute("data-src"), box.getAttribute("data-label") || "clip", box, !!box.getAttribute("data-loop")); });
    boxes.push(box);
    load(box);
  }
  function load(box) {
    var src = box.getAttribute("data-src"), cv = box.querySelector(".sbcv"), wait = box.querySelector(".sbwait");
    wait.style.display = "";
    var pre = window.SPECS && window.SPECS[src];
    if (pre) {                                        // drawn once when the site was built (bench/site_media.py)
      var im = new Image();
      im.onload = function () {
        if (box.getAttribute("data-src") !== src) return;
        cv.width = im.naturalWidth; cv.height = im.naturalHeight;
        cv.getContext("2d").drawImage(im, 0, 0);
        wait.style.display = "none";
        var lv = box.getAttribute("data-level");
        box.querySelector(".sbnum").textContent = "centroid " + hz(pre.centroid) + (lv ? " · " + lv : "");
      };
      im.onerror = function () { delete window.SPECS[src]; load(box); };
      im.src = pre.img;
      return;
    }
    decode(src).then(function (A) {
      if (box.getAttribute("data-src") !== src) return;
      paint(cv, A); box._A = A; wait.style.display = "none";
      var lv = box.getAttribute("data-level");
      box.querySelector(".sbnum").textContent = "centroid " + hz(A.centroid) + (lv ? " · " + lv : "");
    }).catch(function () { wait.textContent = "could not decode the clip"; });
  }
  function show(box, src, label, level) {
    box.setAttribute("data-src", src);
    if (label != null) { box.setAttribute("data-label", label); var l = box.querySelector(".sblabel"); if (l) l.textContent = label; }
    if (level != null) box.setAttribute("data-level", level);
    if (!box._built) { build(box); return; }
    var playing = window.AA && window.AA.current().owner === box && window.AA.isPlaying();
    load(box);
    if (playing) window.AA.play(src, box.getAttribute("data-label"), box, !!box.getAttribute("data-loop"));
  }
  // playheads
  (window.AA_FRAME = window.AA_FRAME || []).push(function () {
    if (!window.AA) return;
    var cur = window.AA.current(), on = window.AA.isPlaying(), pl = window.AA.player;
    boxes.forEach(function (b) {
      var mine = on && cur.owner === b, line = b.querySelector(".sbhead-line");
      var label = mine ? (window.AA.isLoading && window.AA.isLoading() ? "…" : "❚❚") : "▶", span = b.querySelector(".sbplay span");
      if (b._on !== mine) { b._on = mine; b.classList.toggle("on", mine); line.style.display = mine ? "block" : "none"; }
      if (span.textContent !== label) span.textContent = label;           // no needless writes: they cost a layout each frame
      if (mine && pl.duration) line.style.left = (100 * pl.currentTime / pl.duration).toFixed(2) + "%";
    });
  });
  function init() {
    var els = [].slice.call(document.querySelectorAll(".specbox"));
    if ("IntersectionObserver" in window) {
      var io = new IntersectionObserver(function (es) { es.forEach(function (e) { if (e.isIntersecting) { build(e.target); io.unobserve(e.target); } }); }, { rootMargin: "300px" });
      els.forEach(function (b) { io.observe(b); });
    } else els.forEach(build);
  }
  window.Spectro = { show: show, build: build, decode: decode, init: init };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
