// Abliterator: the site's one script. The theme switch, the sections' fade-in, the glitches,
// the lightbox, the player with its scope and spectrum, and the explanatory figures, drawn in
// inline SVG and canvas with no library. The sounds and the knob's data are this machine's own
// renders (media.js, made by bench/site_media.py).
// Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
(function () {
  "use strict";
  var root = document.documentElement;
  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var M = window.MEDIA || { brightness: { steps: [], curve: [[0, -1], [1, 1]] }, lookback: { steps: [] }, sounds: [] };

  function css(name) { return getComputedStyle(root).getPropertyValue(name).trim(); }

  /* ---------- theme ---------- */
  function currentTheme() {
    var a = root.getAttribute("data-theme");
    if (a) return a;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
  }
  document.querySelectorAll(".theme-toggle").forEach(function (b) {
    b.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("abl-theme", next); } catch (e) {}
    });
  });

  /* ---------- reveal, with a glitch on each heading as it arrives ---------- */
  function burst(el) {
    if (reduced || !el) return;
    el.classList.remove("glitch-burst"); void el.offsetWidth; el.classList.add("glitch-burst");
    setTimeout(function () { el.classList.remove("glitch-burst"); }, 500);
  }
  var reveals = document.querySelectorAll(".reveal");
  if ("IntersectionObserver" in window) {
    var rio = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { e.target.classList.add("in"); burst(e.target.querySelector("h2")); rio.unobserve(e.target); } });
    }, { rootMargin: "0px 0px -8% 0px" });
    reveals.forEach(function (el) { rio.observe(el); });
  } else reveals.forEach(function (el) { el.classList.add("in"); });

  var navLinks = [].slice.call(document.querySelectorAll(".site-nav a[href^='#']"));
  if (navLinks.length && "IntersectionObserver" in window) {
    var nio = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) navLinks.forEach(function (a) { a.classList.toggle("on", a.getAttribute("href") === "#" + e.target.id); }); });
    }, { rootMargin: "-40% 0px -55% 0px" });
    navLinks.forEach(function (a) { var s = document.querySelector(a.getAttribute("href")); if (s) nio.observe(s); });
  }
  var title = document.getElementById("title");
  if (title && !reduced) setInterval(function () { burst(title); }, 14000);
  if (title) title.addEventListener("mouseenter", function () { burst(title); });

  /* ---------- the lightbox ---------- */
  var lb = document.getElementById("lightbox"), lbImg = lb && lb.querySelector("img"), lbCap = lb && lb.querySelector(".cap");
  var shots = [].slice.call(document.querySelectorAll("a.shot")), lbIndex = -1, lastFocus = null;
  shots.forEach(function (a, i) {
    var img = a.querySelector("img");
    if (img) a.style.setProperty("--src", "url(\"" + img.getAttribute("src") + "\")");
    a.addEventListener("click", function (e) { if (e.metaKey || e.ctrlKey || e.shiftKey) return; e.preventDefault(); openLb(i); });
  });
  function openLb(i) {
    if (!lb) return;
    lbIndex = (i + shots.length) % shots.length;
    var a = shots[lbIndex], img = a.querySelector("img");
    lastFocus = lastFocus || document.activeElement;
    lb.classList.remove("open"); void lb.offsetWidth;
    lbImg.src = a.getAttribute("href"); lbImg.alt = img ? img.alt : "";
    lbCap.textContent = a.getAttribute("data-cap") || (img ? img.alt : "");
    lb.hidden = false; lb.classList.add("open"); document.body.style.overflow = "hidden";
    lb.querySelector("button").focus();
  }
  function closeLb() {
    if (!lb || lb.hidden) return;
    lb.classList.remove("open"); lb.hidden = true; document.body.style.overflow = "";
    if (lastFocus && lastFocus.focus) lastFocus.focus(); lastFocus = null;
  }
  if (lb) {
    lb.addEventListener("click", function (e) { if (e.target !== lbImg) closeLb(); });
    document.addEventListener("keydown", function (e) {
      if (lb.hidden) return;
      if (e.key === "Escape") closeLb();
      else if (e.key === "ArrowRight") openLb(lbIndex + 1);
      else if (e.key === "ArrowLeft") openLb(lbIndex - 1);
      else if (e.key === "Tab") { e.preventDefault(); lb.querySelector("button").focus(); }
    });
  }

  /* ---------- helpers ---------- */
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }
  function T(x, y, s, cls, extra) { return '<text x="' + x + '" y="' + y + '" class="' + (cls || "f-ink") + '"' + (extra || "") + ">" + esc(s) + "</text>"; }
  function R(x, y, w, h, cls, extra) { return '<rect x="' + x + '" y="' + y + '" width="' + Math.max(0, w) + '" height="' + Math.max(0, h) + '" class="' + cls + '"' + (extra || "") + "/>"; }
  function L(x1, y1, x2, y2, cls, extra) { return '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '" class="' + cls + '"' + (extra || "") + "/>"; }
  function C(x, y, r, cls, extra) { return '<circle cx="' + x + '" cy="' + y + '" r="' + r + '" class="' + cls + '"' + (extra || "") + "/>"; }
  function P(d, cls, extra) { return '<path d="' + d + '" class="' + cls + '" fill="none"' + (extra || "") + "/>"; }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function ease(k) { k = clamp(k, 0, 1); return k * k * (3 - 2 * k); }
  function fmtS(v, d) { return (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v).toFixed(d); }
  function hz(f) { if (f < 1000) return Math.round(f) + " Hz"; var k = (f / 1000).toFixed(f >= 10000 ? 0 : 2); if (k.indexOf(".") >= 0) k = k.replace(/0+$/, "").replace(/\.$/, ""); return k + " kHz"; }
  function arcPath(cx, cy, r, a0, a1) {
    var x0 = cx + r * Math.cos(a0), y0 = cy + r * Math.sin(a0), x1 = cx + r * Math.cos(a1), y1 = cy + r * Math.sin(a1);
    return "M" + x0.toFixed(1) + " " + y0.toFixed(1) + " A" + r + " " + r + " 0 " + (a1 - a0 > Math.PI ? 1 : 0) + " 1 " + x1.toFixed(1) + " " + y1.toFixed(1);
  }
  // the calibrated travel: [[travel, value], ...], piecewise linear both ways
  var CURVE = (M.brightness && M.brightness.curve) || [[0, -1], [1, 1]];
  function along(t) {
    t = clamp(t, 0, 1);
    for (var i = 1; i < CURVE.length; i++) if (t <= CURVE[i][0]) { var a = CURVE[i - 1], b = CURVE[i], k = (t - a[0]) / ((b[0] - a[0]) || 1); return a[1] + k * (b[1] - a[1]); }
    return CURVE[CURVE.length - 1][1];
  }
  function travelOf(v) {
    for (var i = 1; i < CURVE.length; i++) if (v <= CURVE[i][1]) { var a = CURVE[i - 1], b = CURVE[i], k = (v - a[1]) / ((b[1] - a[1]) || 1); return clamp(a[0] + k * (b[0] - a[0]), 0, 1); }
    return 1;
  }

  /* ---------- the player: one audio element, a scope and a spectrum ---------- */
  var player = document.getElementById("player"), actx = null, analyser = null, tbuf = null, fbuf = null;
  var current = { src: null, label: "", owner: null };
  var listeners = [];
  // The audio element plays the clip itself: nothing is routed through Web Audio, which in some
  // browsers (Safari) stays silent or ignores a pause until its context wakes. Only the home page's
  // scope listens, to a copy of the element's output (captureStream), where the browser offers one;
  // elsewhere it shows the clip's spectrogram from the site's build with a playhead.
  var tapNode = null, tapTrack = null;
  function ensureCtx() {
    if (!player || !document.getElementById("live-scope")) return;
    var cap = player.captureStream || player.mozCaptureStream;
    if (!cap) return;
    try {
      if (!actx) {
        var AC = window.AudioContext || window.webkitAudioContext;
        actx = new AC();
        analyser = actx.createAnalyser(); analyser.fftSize = 4096; analyser.smoothingTimeConstant = 0.72;
        var mute = actx.createGain(); mute.gain.value = 0; analyser.connect(mute); mute.connect(actx.destination);
        tbuf = new Float32Array(analyser.fftSize); fbuf = new Float32Array(analyser.frequencyBinCount);
      }
      var tracks = cap.call(player).getAudioTracks();
      if (tracks.length && tracks[0] !== tapTrack) {
        if (tapNode) try { tapNode.disconnect(); } catch (e) {}
        tapTrack = tracks[0];
        tapNode = actx.createMediaStreamSource(new MediaStream([tapTrack]));
        tapNode.connect(analyser);
      }
    } catch (e) { analyser = null; }
  }
  function notify() { listeners.forEach(function (f) { f(); }); var np = document.getElementById("now-playing"); if (np) np.textContent = playing() ? current.label : "nothing playing"; }
  function playing() { return player && !player.paused && !player.ended; }
  // A click on the clip that is playing, or still loading (the MP3 is fetched on the first click,
  // and its button shows … meanwhile), stops it. Esc stops whatever is playing.
  // loading: asked to play, not paused, and not yet enough data to play on (HAVE_FUTURE_DATA)
  function isLoading() { return !!player && !player.paused && !player.error && player.readyState < 3; }
  function wake() {
    if (actx && actx.state !== "running") { try { var r = actx.resume(); if (r && r.catch) r.catch(function () {}); } catch (e) {} }
  }
  function start() {
    var p = player.play();
    if (p && p.then) p.then(function () { wake(); }, function (e) {
      if (e && e.name === "AbortError") return;           // replaced by a newer click
      notify();
      var np = document.getElementById("now-playing"); if (np) np.textContent = "could not play: " + (e && e.message ? e.message : "the browser refused");
    });
  }
  function stop() {
    if (!player) return;
    player.pause();
    try { player.currentTime = 0; } catch (e) {}
    notify();
  }
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && playing() && !(document.getElementById("lightbox") && !document.getElementById("lightbox").hidden)) stop();
  });
  function play(src, label, owner, loop) {
    if (!player) return;
    wake();
    if (current.src === src && current.owner === owner && playing()) { stop(); return; }
    current = { src: src, label: label, owner: owner };
    player.loop = !!loop;
    if (player.getAttribute("src") !== src) { player.preload = "auto"; player.src = src; }
    else { try { player.currentTime = 0; } catch (e) {} }
    start();
    notify();
  }
  function swapTo(src, label, owner) {           // the knob: change clip, keep playing
    if (!player) return;
    var at = playing() && player.duration ? player.currentTime : 0;
    current = { src: src, label: label, owner: owner };
    player.loop = true; player.src = src;
    player.addEventListener("loadedmetadata", function once() { player.removeEventListener("loadedmetadata", once); try { player.currentTime = at % (player.duration || 3); } catch (e) {} }, { once: true });
    wake(); start();
    notify();
  }
  if (player) {
    ["play", "pause", "ended"].forEach(function (ev) { player.addEventListener(ev, notify); });
    ["playing", "waiting", "canplay", "error"].forEach(function (ev) { player.addEventListener(ev, function () { if (ev === "playing") { ensureCtx(); wake(); } notify(); }); });
    player.addEventListener("error", function () { var np = document.getElementById("now-playing"); if (np) np.textContent = "could not load the clip"; });
  }

  /* ---------- canvases ---------- */
  function fitCanvas(cv) {
    var dpr = Math.min(window.devicePixelRatio || 1, 2), w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    var g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); return { g: g, w: w, h: h };
  }
  function visibleObs(el, onChange) {
    if (!("IntersectionObserver" in window)) { onChange(true); return; }
    new IntersectionObserver(function (es) { es.forEach(function (e) { onChange(e.isIntersecting); }); }, { rootMargin: "80px" }).observe(el);
  }

  // the hero: a phosphor scope of a slowly morphing tone, the dry trace splitting into the wet one,
  // over the bars of a spectrum
  var hero = document.getElementById("hero-scope"), heroVis = true;
  if (hero) visibleObs(hero, function (v) { heroVis = v; });
  function partials(t) {                         // a drone's partials, their levels drifting
    var ps = [];
    for (var k = 1; k <= 9; k++) ps.push([k, (1 / k) * (0.55 + 0.45 * Math.sin(t / (2300 + 410 * k) + k * 1.7))]);
    return ps;
  }
  function drawHero(t) {
    if (!hero || !heroVis) return;
    var c = fitCanvas(hero), g = c.g, W = c.w, H = c.h, acc = css("--accent"), blue = css("--blue"), dim = css("--fig-dim");
    g.clearRect(0, 0, W, H);
    var ps = partials(t), mid = H * 0.42, amp = Math.min(120, H * 0.24), part = W * 0.55, n = Math.min(900, Math.round(W / 1.6));
    // spectrum bars along the foot
    var nb = 64, bw = W / nb;
    g.globalAlpha = 0.16; g.fillStyle = acc;
    for (var b = 0; b < nb; b++) {
      var f = b / nb, lvl = 0;
      ps.forEach(function (p) { lvl += p[1] * Math.exp(-Math.pow((f - Math.log(p[0]) / Math.log(30)) * 9, 2)); });
      lvl += 0.04 + 0.03 * Math.sin(t / 300 + b);
      var hgt = Math.min(H * 0.4, lvl * H * 0.36);
      g.fillRect(b * bw + 1, H - hgt, bw - 2, hgt);
    }
    g.globalAlpha = 1;
    function trace(wet, colour, width, alpha) {
      g.beginPath();
      for (var i = 0; i <= n; i++) {
        var x = i / n * W, ph = x / 95 - t / 650, y = 0;
        ps.forEach(function (p) { y += p[1] * Math.sin(ph * p[0] + p[0] * 0.7); });
        if (wet) {
          var k = Math.max(0, (x - part) / (W - part));
          y += k * (0.5 * Math.sin(ph * 2.5 + t / 420) * Math.sin(ph * 0.31 - t / 900) + 0.35 * Math.sin(ph * 11.3 - t / 260) * k);
          y = Math.tanh(y * (1 + 1.4 * k));
        }
        var yy = mid - y * amp * 0.7;
        if (i) g.lineTo(x, yy); else g.moveTo(x, yy);
      }
      g.strokeStyle = colour; g.lineWidth = width; g.globalAlpha = alpha; g.stroke(); g.globalAlpha = 1;
    }
    trace(false, blue, 6, 0.08); trace(false, blue, 1.2, 0.55);
    trace(true, acc, 7, 0.1); trace(true, acc, 1.6, 0.9);
    // graticule
    g.strokeStyle = dim; g.globalAlpha = 0.18; g.lineWidth = 1;
    for (var gx = 0; gx < W; gx += 80) { g.beginPath(); g.moveTo(gx + 0.5, mid - amp); g.lineTo(gx + 0.5, mid + amp); g.stroke(); }
    g.beginPath(); g.moveTo(0, mid + 0.5); g.lineTo(W, mid + 0.5); g.stroke(); g.globalAlpha = 1;
  }

  // the live scope and spectrum over the sounds
  var scopeCv = document.getElementById("live-scope"), specCv = document.getElementById("live-spec"), deckVis = false, peaksHold = [];
  if (scopeCv) visibleObs(scopeCv, function (v) { deckVis = v; });
  // where the browser gives no copy of the sound (Safari), the deck shows the playing clip's
  // waveform and its spectrogram from the site's build, with a playhead
  var cannedImg = {};
  function drawCannedDeck() {
    var acc = css("--accent"), dim = css("--fig-dim"), prog = player.duration ? player.currentTime / player.duration : 0;
    var c = fitCanvas(scopeCv), g = c.g, W = c.w, H = c.h;
    g.clearRect(0, 0, W, H);
    var pk = current.owner && current.owner.s && current.owner.s.peaks;
    if (pk) {
      var mx = 0.001; pk.forEach(function (p) { mx = Math.max(mx, p); });
      for (var i = 0; i < pk.length; i++) { var h = pk[i] / mx * (H - 10), on = i / pk.length < prog; g.fillStyle = on ? acc : dim; g.globalAlpha = on ? 0.95 : 0.5; g.fillRect(i * W / pk.length, (H - h) / 2, Math.max(1, W / pk.length - 1), h); }
      g.globalAlpha = 1;
    } else { g.strokeStyle = acc; g.beginPath(); g.moveTo(0, H / 2); g.lineTo(W, H / 2); g.stroke(); }
    g.fillStyle = "#fff"; g.fillRect(prog * W, 0, 1.5, H);
    var s2 = fitCanvas(specCv), h2 = s2.g, SW = s2.w, SH = s2.h, rec = window.SPECS && window.SPECS[current.src];
    h2.fillStyle = "#07090c"; h2.fillRect(0, 0, SW, SH);
    if (rec) {
      var im = cannedImg[rec.img];
      if (!im) { im = cannedImg[rec.img] = new Image(); im.src = rec.img; }
      if (im.complete && im.naturalWidth) { h2.imageSmoothingEnabled = true; h2.drawImage(im, 0, 0, SW, SH); }
      h2.fillStyle = "#fff"; h2.fillRect(prog * SW, 0, 1.5, SH);
    }
  }
  function drawDeck(t) {
    if (!scopeCv || !deckVis) return;
    if (!analyser && playing()) { drawCannedDeck(); return; }
    var acc = css("--accent"), dim = css("--fig-dim"), line = css("--fig-line"), on = analyser && playing();
    var c = fitCanvas(scopeCv), g = c.g, W = c.w, H = c.h, mid = H / 2;
    g.clearRect(0, 0, W, H);
    g.strokeStyle = line; g.lineWidth = 1;
    for (var gx = 0; gx <= W; gx += W / 10) { g.beginPath(); g.moveTo(gx, 0); g.lineTo(gx, H); g.stroke(); }
    for (var gy = 0; gy <= H; gy += H / 6) { g.beginPath(); g.moveTo(0, gy); g.lineTo(W, gy); g.stroke(); }
    g.beginPath();
    if (on) {
      analyser.getFloatTimeDomainData(tbuf);
      var start = 0; for (var i = 1; i < tbuf.length / 2; i++) if (tbuf[i - 1] < 0 && tbuf[i] >= 0) { start = i; break; }   // trigger on a rising zero
      var span = Math.min(2048, tbuf.length - start), peak = 0.02;
      for (i = 0; i < span; i++) peak = Math.max(peak, Math.abs(tbuf[start + i]));
      var gain = Math.min(8, 0.8 / peak);
      for (i = 0; i < span; i++) { var x = i / (span - 1) * W, y = mid - tbuf[start + i] * gain * (H * 0.45); if (i) g.lineTo(x, y); else g.moveTo(x, y); }
    } else {
      for (var j = 0; j <= 200; j++) { var xx = j / 200 * W, yy = mid + (reduced ? 0 : Math.sin(j * 0.9 + t / 90) * 0.6 + Math.sin(j * 2.3 - t / 50) * 0.4); if (j) g.lineTo(xx, yy); else g.moveTo(xx, yy); }
    }
    g.strokeStyle = on ? acc : dim; g.lineWidth = on ? 1.6 : 1; g.stroke();
    // the spectrum, 40 Hz to 20 kHz on a log axis
    var s = fitCanvas(specCv), h = s.g, SW = s.w, SH = s.h, nb = 72, bw = SW / nb;
    h.clearRect(0, 0, SW, SH);
    var sr = actx ? actx.sampleRate : 48000, bins = analyser ? analyser.frequencyBinCount : 0;
    if (on) analyser.getFloatFrequencyData(fbuf);
    for (var b = 0; b < nb; b++) {
      var f0 = 40 * Math.pow(500, b / nb), f1 = 40 * Math.pow(500, (b + 1) / nb), v = -110;
      if (on) {
        var k0 = Math.floor(f0 / (sr / 2) * bins), k1 = Math.max(k0 + 1, Math.ceil(f1 / (sr / 2) * bins)), m = -200;
        for (var k = k0; k < k1 && k < bins; k++) m = Math.max(m, fbuf[k]);
        v = m;
      }
      var lv = clamp((v + 100) / 80, 0, 1);
      if (!on) lv = reduced ? 0.02 : 0.02 + 0.015 * (1 + Math.sin(b * 0.7 + t / 400));
      peaksHold[b] = Math.max(lv, (peaksHold[b] || 0) - 0.008);
      var bh = lv * (SH - 8);
      var grad = h.createLinearGradient(0, SH, 0, SH - bh);
      grad.addColorStop(0, css("--blue")); grad.addColorStop(1, acc);
      h.fillStyle = grad; h.globalAlpha = on ? 0.9 : 0.35; h.fillRect(b * bw + 1, SH - bh, bw - 2, bh);
      h.globalAlpha = 0.9; h.fillStyle = acc; h.fillRect(b * bw + 1, SH - peaksHold[b] * (SH - 8) - 2, bw - 2, 1.5);
    }
    h.globalAlpha = 1; h.fillStyle = dim; h.font = "9px " + css("--mono");
    [100, 1000, 10000].forEach(function (f) { var x = Math.log(f / 40) / Math.log(500) * SW; h.fillText(hz(f), x + 3, 11); h.fillRect(x, 0, 1, 14); });
  }

  /* ---------- the sounds ---------- */
  var soundCards = [];
  function drawPeaks(cv, peaks, prog) {
    var c = fitCanvas(cv), g = c.g, W = c.w, H = c.h, n = peaks.length, bw = W / n, mx = 0.001;
    peaks.forEach(function (p) { mx = Math.max(mx, p); });
    g.clearRect(0, 0, W, H);
    var acc = css("--accent"), dim = css("--fig-dim");
    for (var i = 0; i < n; i++) {
      var hgt = Math.max(1, peaks[i] / mx * (H - 4));
      g.fillStyle = (i / n) < prog ? acc : dim; g.globalAlpha = (i / n) < prog ? 0.95 : 0.55;
      g.fillRect(i * bw, (H - hgt) / 2, Math.max(1, bw - 0.6), hgt);
    }
    g.globalAlpha = 1;
    if (prog > 0 && prog < 1) { g.fillStyle = css("--bright"); g.fillRect(prog * W, 0, 1.5, H); }
  }
  var host = document.getElementById("sounds");
  if (host) M.sounds.forEach(function (s) {
    var b = document.createElement("button");
    b.type = "button"; b.className = "snd";
    var chain = (s.chain || []).filter(Boolean);
    b.innerHTML = '<span class="pb" aria-hidden="true">▶</span><span class="nm"></span><span class="meta"></span><canvas aria-hidden="true"></canvas>' +
      (chain.length ? '<span class="chainline"></span>' : "");
    b.querySelector(".nm").textContent = s.label;
    b.querySelector(".meta").textContent = (s.seconds ? s.seconds.toFixed(s.seconds % 1 ? 1 : 0) + " s · " : "") + (s.frozen ? "frozen · " : "") + (s.prompt || s.preset || "");
    if (chain.length) b.querySelector(".chainline").textContent = chain.join(" · ");
    b.setAttribute("aria-label", "Play " + s.label);
    host.appendChild(b);
    var card = { el: b, s: s, cv: b.querySelector("canvas") };
    soundCards.push(card);
    b.addEventListener("click", function () { play(s.src, s.label, card, false); });
  });
  function paintCards() {
    soundCards.forEach(function (c) {
      var on = current.owner === c && playing();
      if (c.el.classList.contains("on") !== on) c.el.classList.toggle("on", on);
      var pbt = on ? (isLoading() ? "…" : "❚❚") : "▶", pbe = c.el.querySelector(".pb");
      if (pbe.textContent !== pbt) pbe.textContent = pbt;
      var prog = current.owner === c && player.duration ? player.currentTime / player.duration : 0;
      if (on || c.lastProg !== prog || !c.painted) { drawPeaks(c.cv, c.s.peaks, prog); c.lastProg = prog; c.painted = true; }
    });
  }
  listeners.push(paintCards);
  window.addEventListener("resize", function () { soundCards.forEach(function (c) { c.painted = false; }); paintCards(); });

  // Lookback's mix, step by step
  var lbHost = document.getElementById("lookback"), lbBtns = [];
  if (lbHost && M.lookback) M.lookback.steps.forEach(function (st) {
    var b = document.createElement("button");
    b.type = "button";
    var lv = clamp((st.rms_db + 75) / 50, 0.02, 1), dead = st.silence > 0.5;
    b.innerHTML = '<span class="lv"><i class="' + (dead ? "dead" : "") + '" style="height:' + (lv * 100).toFixed(0) + '%"></i></span><span>' + st.v + "</span>";
    b.title = "mix " + st.v + ": " + st.rms_db.toFixed(1) + " dBFS, " + Math.round(st.silence * 100) + "% of frames silent. Click to listen.";
    b.setAttribute("aria-label", "Lookback mix " + st.v + ", " + st.rms_db.toFixed(0).replace("-", "−") + " dBFS. Play");
    b.addEventListener("click", function () { play(st.src, "Lookback mix " + st.v + " · Warm drone · " + st.rms_db.toFixed(1) + " dBFS", b, false); });
    lbHost.appendChild(b); lbBtns.push(b);
  });
  listeners.push(function () { lbBtns.forEach(function (b) { b.classList.toggle("on", current.owner === b && playing()); }); });

  /* ---------- the SVG figures ---------- */
  var figs = [];
  var fio = "IntersectionObserver" in window ? new IntersectionObserver(function (es) {
    es.forEach(function (e) { figs.forEach(function (f) { if (f.el === e.target) f.visible = e.isIntersecting; }); });
  }, { rootMargin: "60px" }) : null;
  function addFig(id, init) {
    var el = document.getElementById(id);
    if (!el) return;
    var f = { el: el, svg: el.querySelector("svg"), visible: !fio, t0: performance.now(), draw: null };
    init(f);
    figs.push(f);
    f.redraw = function () { f.draw(reduced ? (f.still || 0) : performance.now() - f.t0); };
    f.redraw();
    if (fio) fio.observe(el);
  }
  function pressToggle(f, onToggle) {
    f.el.querySelectorAll("button[data-k]").forEach(function (b) {
      b.addEventListener("click", function () { var on = b.getAttribute("aria-pressed") !== "true"; b.setAttribute("aria-pressed", on ? "true" : "false"); onToggle(b.getAttribute("data-k"), on); f.redraw(); });
    });
  }
  function pressGroup(f, attr, onPick) {
    var bs = f.el.querySelectorAll("button[" + attr + "]");
    bs.forEach(function (b) { b.addEventListener("click", function () { bs.forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); }); onPick(b.getAttribute(attr)); f.redraw(); }); });
  }
  function sliders(f, onInput) {
    f.el.querySelectorAll("input[type=range]").forEach(function (inp) {
      var out = inp.parentNode.querySelector("output");
      inp.addEventListener("input", function () { onInput(inp.getAttribute("data-k"), parseFloat(inp.value), out); f.redraw(); });
    });
  }

  /* 1. the signal path */
  addFig("viz-path", function (f) {
    var st = { steer: true, layers: true, fx: true }, ro = f.el.querySelector(".readout");
    f.still = 1600;
    var LX0 = 150, LW = 300, NL = 12, laneY = 255;
    function lx(l) { return LX0 + (l / NL) * LW; }
    // the model's output, as drawn: partials; Steer brightens it, the layers add a lookback shimmer
    function model(ph, t) {
      var y = 0.62 * Math.sin(ph) + 0.22 * Math.sin(2 * ph + 0.4) + 0.08 * Math.sin(3 * ph);
      if (st.steer) y += 0.2 * Math.sin(5 * ph + t / 700) + 0.12 * Math.sin(8 * ph);
      if (st.layers) y = y * (0.75 + 0.25 * Math.sin(ph / 3.7 - t / 500)) + 0.15 * Math.sin(ph * 1.5 - 2.2);
      return y;
    }
    f.draw = function (t) {
      var s = "", i;
      // the prompt
      s += R(14, 40, 110, 70, "f-panel", ' rx="6" style="stroke:var(--fig-line)"');
      s += T(24, 60, "prompt", "f-dim t-xs t-up"); s += T(24, 78, "“warm drone”", "f-ink t-s"); s += T(24, 96, "seed 1", "f-dim t-s");
      s += L(124, 75, LX0 - 8, 75, "s-dim", ' stroke-dasharray="3 3"');
      // the transformer: twelve layers, the stream through them
      s += T(LX0, 24, "Magenta RT 2 · 12 transformer layers · one 40 ms frame per step", "f-dim t-xs t-up");
      for (i = 0; i < NL; i++) {
        var x0 = lx(i) + 3, w = LW / NL - 6;
        s += R(x0, 40, w, 70, "f-panel", ' rx="3" style="stroke:var(--fig-line)"');
        s += T(x0 + w / 2 - (i > 8 ? 6 : 3), 104, String(i + 1), "f-dim t-xs");
      }
      s += L(LX0, 75, LX0 + LW, 75, "s-ink", ' stroke-width="1.5" opacity="0.4"');
      var steerX = lx(6), unitX = lx(6) + 6;
      // Steer and the unit, at layer 6
      s += L(steerX, 112, steerX, 140, st.steer ? "s-acc" : "s-dim", ' stroke-width="1.5"');
      s += R(steerX - 44, 140, 60, 24, "f-bg", ' rx="4" style="stroke:' + (st.steer ? "var(--accent)" : "var(--fig-line)") + '"');
      s += C(steerX - 34, 152, 4, st.steer ? "f-acc" : "f-dim"); s += T(steerX - 26, 156, "Steer", st.steer ? "f-ink t-s" : "f-dim t-s");
      s += L(unitX, 112, unitX + 30, 178, st.layers ? "s-blue" : "s-dim", ' stroke-width="1.2"');
      s += R(unitX + 18, 172, 84, 24, "f-bg", ' rx="4" style="stroke:' + (st.layers ? "var(--blue)" : "var(--fig-line)") + '"');
      s += C(unitX + 28, 184, 4, st.layers ? "f-blue" : "f-dim"); s += T(unitX + 36, 188, "Lookback", st.layers ? "f-ink t-s" : "f-dim t-s");
      // tokens through the layers
      for (var k = 0; k < 7; k++) {
        var ph = ((t / 2200 + k / 7) % 1), x = LX0 - 6 + ph * (LW + 12), bent = x > steerX && (st.steer || st.layers);
        s += R(x - 3, 72 + (bent ? Math.sin(x / 7 + k) * 3 : 0) - 3, 6, 6, bent ? (st.steer ? "f-acc" : "f-blue") : "f-ink", ' opacity="0.9"');
      }
      // the codec, then the output section
      var CX = 488;
      s += L(LX0 + LW, 75, CX, 75, "s-dim", ' stroke-dasharray="3 3"');
      s += R(CX, 40, 70, 70, "f-panel", ' rx="6" style="stroke:var(--fig-line)"');
      s += T(CX + 10, 62, "codec", "f-ink t-s"); s += T(CX + 10, 80, "tokens →", "f-dim t-xs"); s += T(CX + 10, 92, "1,920 × 2", "f-dim t-xs");
      var boxes = [[580, 56, "level", "auto"], [652, 150, "effects", "4 slots"], [818, 54, "LP", "16 kHz"], [888, 58, "limiter", "· vol"]];
      boxes.forEach(function (b, j) {
        var lit = j !== 1 || st.fx;
        s += R(b[0], 40, b[1], 70, "f-panel", ' rx="6" style="stroke:' + (j === 1 && st.fx ? "var(--green)" : "var(--fig-line)") + '"');
        s += T(b[0] + 8, 62, b[2], lit ? "f-ink t-s" : "f-dim t-s"); s += T(b[0] + 8, 80, b[3], "f-dim t-xs");
        if (j === 1) ["drive", "chorus", "delay", "reverb"].forEach(function (nm, q) { s += R(b[0] + 8 + q * 35, 88, 30, 12, q > 1 && st.fx ? "f-green" : "f-dim", ' rx="2" opacity="' + (q > 1 && st.fx ? 0.55 : 0.2) + '"'); s += T(b[0] + 10 + q * 35, 97, nm.slice(0, 5), "f-bg t-xs", ' style="font-size:7px"'); });
      });
      s += L(CX + 70, 75, 580, 75, "s-dim"); s += L(636, 75, 652, 75, "s-dim"); s += L(802, 75, 818, 75, "s-dim"); s += L(872, 75, 888, 75, "s-dim");
      // the audio lane, from the codec to the card
      s += T(CX, laneY - 70, "the audio, as drawn", "f-dim t-xs t-up");
      var d = "", dw = "", X0 = CX, X1 = 946, n = 220;
      for (i = 0; i <= n; i++) {
        var xx = X0 + (X1 - X0) * i / n, p = xx / 9 - t / 140, y = model(p, t);
        var fxK = st.fx ? clamp((xx - 652) / 60, 0, 1) : 0;
        if (fxK > 0) y = y + fxK * (0.35 * model(p - 3.1, t) + 0.22 * model(p - 7.3, t) + 0.09 * Math.sin(p * 13 + t / 90));
        if (xx > 888) y = Math.tanh(y * 1.3);
        var yy = laneY - y * 34;
        d += (i ? "L" : "M") + xx.toFixed(1) + " " + yy.toFixed(1);
      }
      s += P(d, "s-acc", ' stroke-width="1.6" style="filter:drop-shadow(0 0 3px var(--accent))"');
      [580, 652, 818, 888].forEach(function (bx) { s += L(bx, 112, bx, laneY + 44, "s-line", ' stroke-dasharray="2 4"'); });
      s += T(X1 - 30, laneY + 60, "out", "f-dim t-xs t-up");
      f.svg.innerHTML = s;
      var parts = [];
      parts.push(st.steer ? "Steer adds the brightness direction at layer 6" : "Steer off: the doses play at 0");
      parts.push(st.layers ? "Lookback at layer 6 changes what layers 7 to 12 read" : "Model layers off: the model plays without vector bending");
      parts.push(st.fx ? "Delay and Reverb are lit in slots 3 and 4" : "Effects bypassed");
      var msg = parts.join(" · ") + ".";
      if (ro.textContent !== msg) ro.textContent = msg;
    };
    pressToggle(f, function (k, on) { st[k] = on; });
  });

  /* 2. frames ahead of the card */
  addFig("viz-frames", function (f) {
    var lead = 2, ro = f.el.querySelector(".readout"), SLOW = 9, FW = 62, PX = 150, DEV = 22, move = null;
    f.still = 4000;
    function stepMs(k) { var r = Math.sin(k * 12.9898) * 43758.5453; r -= Math.floor(r); return 23.5 + 3 * r; }
    f.draw = function (now) {
      var t = now / SLOW, s = "", i;            // t in model milliseconds
      var playing = Math.floor(t / 40), inFrame = t - playing * 40;
      var written = playing + lead, prog = clamp(inFrame / stepMs(written + 1), 0, 1);
      s += T(16, 22, "the model writes ahead; the card plays one frame every 40 ms (slowed " + SLOW + "×)", "f-dim t-xs t-up");
      // the track
      s += L(10, 110, 510, 110, "s-line");
      for (var k = playing - 2; k <= written + 1; k++) {
        var x = PX + (k * 40 - t) * FW / 40;
        if (x > 510 || x + FW < 10) continue;
        var cls = "f-panel", op = 1, lab = "";
        if (k > written) { cls = "f-acc"; op = 0.25; }
        var moved = move && k >= move.k;
        s += R(x + 2, 72, FW - 4, 76, moved ? (k > written ? "f-acc" : "f-acc") : "f-panel", ' rx="4" style="stroke:' + (moved ? "var(--accent)" : "var(--fig-line)") + '" opacity="' + (k > written ? 0.35 : moved ? 0.55 : 1) + '"');
        if (k === written + 1) s += R(x + 2, 72 + 76 * (1 - prog), FW - 4, 76 * prog, "f-acc", ' rx="4" opacity="0.6"');
        // a little waveform inside each written frame
        if (k <= written) {
          var d = "";
          for (var j = 0; j <= 20; j++) { var px = x + 5 + j * (FW - 10) / 20, ph = (k * 20 + j) / 2.6; d += (j ? "L" : "M") + px.toFixed(1) + " " + (110 - (0.6 * Math.sin(ph) + 0.3 * Math.sin(ph * 2.7 + k) + (moved ? 0.35 * Math.sin(ph * 6) : 0)) * 22).toFixed(1); }
          s += P(d, moved ? "s-acc" : "s-blue", ' stroke-width="1.1" opacity="0.9"');
        }
        if (k >= playing && k <= written + 1) s += T(x + FW / 2 - 8, 164, k === written + 1 ? "next" : (k === playing ? "now" : "+" + (k - playing)), "f-dim t-xs");
      }
      // the playhead (the card) and the speaker
      s += L(PX, 60, PX, 160, "s-red", ' stroke-width="2"');
      s += T(PX - 26, 54, "the card", "f-red t-xs t-up");
      
      // step times against the budget
      var BY = 290, BH = 80;
      s += T(16, 196, "step time against the 40 ms budget (Magenta RT 2 small, Apple M5)", "f-dim t-xs t-up");
      s += L(16, BY - BH * 40 / 45, 504, BY - BH * 40 / 45, "s-red", ' stroke-dasharray="4 3"'); s += T(470, BY - BH * 40 / 45 - 4, "40 ms", "f-red t-xs");
      for (i = 0; i < 24; i++) {
        var kk = written + 1 - 23 + i, ms = stepMs(kk), hh = BH * ms / 45, cur = i === 23;
        s += R(16 + i * 20, BY - (cur ? hh * prog : hh), 14, cur ? hh * prog : hh, cur ? "f-acc" : "f-blue", ' opacity="' + (cur ? 0.9 : 0.55) + '"');
      }
      s += L(16, BY, 504, BY, "s-dim");
      f.svg.innerHTML = s;
      var msg;
      if (move) {
        if (t >= move.heard) msg = "That move was heard " + Math.round(move.heard - move.t) + " ms after it was made: " + Math.round(move.wait) + " ms waiting for the next step, then " + lead + " × 40 ms in the buffer and the card's " + DEV + " ms.";
        else msg = "Waiting: the move goes into frame +" + (move.k - playing) + ", heard when it reaches the card.";
      } else msg = "Buffer " + lead + " frames: about " + (lead * 40 + DEV) + " ms from a move to the sound, with " + (40 - 25) + " ms to spare at each step.";
      if (ro.textContent !== msg) ro.textContent = msg;
    };
    sliders(f, function (k, v, out) { lead = Math.round(v); out.textContent = lead; move = null; });
    f.el.querySelector("[data-act=move]").addEventListener("click", function () {
      // the move goes into the next step, which writes frame playing + lead + 1; that frame
      // reaches the card lead frames after the step starts, and the card adds its own delay
      var t = (performance.now() - f.t0) / SLOW, playing = Math.floor(t / 40), startNext = (playing + 1) * 40;
      move = { t: t, k: playing + lead + 1, wait: startNext - t, heard: startNext + lead * 40 + DEV };
    });
  });

  /* 3. the Brightness knob, on real renders */
  addFig("viz-knob", function (f) {
    var B = M.brightness || { steps: [] }, steps = B.steps || [], safe = B.safe || [-1, 1];
    // the lowest bands hold no FFT bin at this resolution (−180 dB): draw them as their neighbours
    steps.forEach(function (st) { st.ltas = st.ltas.map(function (v, i, a) { if (v > -100) return v; var l = i - 1, r = i + 1; while (l >= 0 && a[l] <= -100) l--; while (r < a.length && a[r] <= -100) r++; var vl = l >= 0 ? a[l] : null, vr = r < a.length ? a[r] : null; return vl == null ? (vr == null ? -120 : vr) : vr == null ? vl : (vl + vr) / 2; }); });
    var ro = f.el.querySelector(".readout"), full = false, listen = false, travel = travelOf(0), shown = null, lastNearest = null;
    var CX = 150, CY = 165, RK = 80, A0 = Math.PI * 0.75, A1 = Math.PI * 2.25;
    var lo = -110, hi = -100;
    steps.forEach(function (s) { s.ltas.forEach(function (v) { if (v > -100) { lo = Math.max(lo, Math.min(lo === -110 ? v : lo, v)); hi = Math.max(hi, v); } }); });
    lo = hi - 60;
    var def = steps.filter(function (s) { return s.v === 0; })[0] || steps[0];
    function dose() { return full ? -1 + 2 * travel : along(travel); }
    function travelFor(v) { return full ? (v + 1) / 2 : travelOf(v); }
    function nearest() { var v = dose(), best = steps[0]; steps.forEach(function (s) { if (Math.abs(s.v - v) < Math.abs(best.v - v)) best = s; }); return best; }
    function ang(tr) { return A0 + (A1 - A0) * tr; }
    var bars = [];
    f.still = 0;
    f.draw = function (t) {
      if (!steps.length) { f.svg.innerHTML = T(20, 40, "no renders", "f-dim t-s"); return; }
      var v = dose(), nb = nearest(), s = "", i;
      if (nb !== lastNearest) { lastNearest = nb; if (listen) swapTo(nb.src, "Brightness " + fmtS(nb.v, 2) + " · Warm drone · seed 1", f); }
      // the knob's arc: the travel, the measured range, the steps
      s += P(arcPath(CX, CY, RK + 14, A0, A1), "s-line", ' stroke-width="10" stroke-linecap="round"');
      if (full) {
        var a = ang(travelFor(safe[0])), b = ang(travelFor(safe[1]));
        s += P(arcPath(CX, CY, RK + 14, a, b), "s-acc", ' stroke-width="10" opacity="0.35"');
      } else s += P(arcPath(CX, CY, RK + 14, A0, A1), "s-acc", ' stroke-width="10" stroke-linecap="round" opacity="0.35"');
      steps.forEach(function (st) {
        if (!full && (st.v < safe[0] - 1e-9 || st.v > safe[1] + 1e-9)) return;
        var an = ang(travelFor(st.v)), r0 = RK + 6, r1 = RK + 24, br = st.broken && st.broken.length;
        s += L((CX + r0 * Math.cos(an)).toFixed(1), (CY + r0 * Math.sin(an)).toFixed(1), (CX + r1 * Math.cos(an)).toFixed(1), (CY + r1 * Math.sin(an)).toFixed(1), br ? "s-red" : (st === nb ? "s-acc" : "s-dim"), ' stroke-width="' + (st === nb ? 3 : 1.5) + '"');
        if ([-1, -0.2, 0, 0.15, 0.4, 1].indexOf(st.v) >= 0) { var r2 = RK + 38; s += T((CX + r2 * Math.cos(an) - 12).toFixed(1), (CY + r2 * Math.sin(an) + 4).toFixed(1), fmtS(st.v, st.v % 1 ? 2 : 0), br ? "f-red t-xs" : "f-dim t-xs"); }
      });
      s += C(CX, CY, RK - 4, "f-panel", ' style="stroke:var(--fig-line)"');
      s += C(CX, CY, RK - 22, "f-bg", ' style="stroke:var(--fig-line)"');
      var an2 = ang(travel);
      s += L(CX + (RK - 46) * Math.cos(an2), CY + (RK - 46) * Math.sin(an2), CX + (RK - 8) * Math.cos(an2), CY + (RK - 8) * Math.sin(an2), "s-acc", ' stroke-width="5" stroke-linecap="round" style="filter:drop-shadow(0 0 4px var(--accent))"');
      s += T(CX - 34, CY + 6, fmtS(v, 3), "f-ink t-l");
      s += T(CX - 33, CY + 24, "brightness", "f-dim t-xs t-up");
      s += C(CX, CY, RK + 30, "knob-hit", ' fill="transparent"');
      // the spectrum of the nearest render, eased
      var SX = 330, SW = 610, SY = 40, SH = 220, n = nb.ltas.length;
      s += T(SX, 28, "long-term spectrum of the render at " + fmtS(nb.v, 2) + " (bars) and at 0 (outline) · 40 Hz to 20 kHz", "f-dim t-xs t-up");
      for (i = 0; i < n; i++) {
        var target = clamp((nb.ltas[i] - lo) / (hi - lo), 0, 1);
        bars[i] = bars[i] == null ? target : bars[i] + (target - bars[i]) * 0.18;
        var jit = reduced ? 0 : 0.012 * Math.sin(t / 120 + i * 1.3);
        var hgt = clamp(bars[i] + jit, 0, 1) * SH, x = SX + i * SW / n;
        s += R(x + 1, SY + SH - hgt, SW / n - 2, hgt, "f-acc", ' opacity="' + (0.35 + 0.55 * bars[i]) + '"');
      }
      var od = "";
      def.ltas.forEach(function (vv, j) { var y = SY + SH - clamp((vv - lo) / (hi - lo), 0, 1) * SH, x0 = SX + j * SW / n, x1 = x0 + SW / n; od += (j ? "L" : "M") + x0.toFixed(1) + " " + y.toFixed(1) + " L" + x1.toFixed(1) + " " + y.toFixed(1); });
      s += P(od, "s-blue", ' stroke-width="1.2" opacity="0.9"');
      s += L(SX, SY + SH, SX + SW, SY + SH, "s-dim");
      function fx(fq) { return SX + SW * Math.log(fq / 40) / Math.log(500); }
      [100, 1000, 10000].forEach(function (fq) { s += L(fx(fq), SY + SH, fx(fq), SY + SH + 5, "s-dim"); s += T(fx(fq) - 12, SY + SH + 16, hz(fq), "f-dim t-xs"); });
      var cx = fx(Math.max(40, nb.centroid_hz));
      s += L(cx, SY, cx, SY + SH, "s-green", ' stroke-width="1.5" stroke-dasharray="4 3"');
      s += T(Math.min(cx + 5, SX + SW - 110), SY + 12, "centroid " + hz(nb.centroid_hz), "f-green t-xs");
      // the render's waveform
      var WY = 330, WH = 34, pk = nb.peaks, mx = 0.001;
      pk.forEach(function (p) { mx = Math.max(mx, p); });
      var prog = (listen && current.owner === f && playing() && player.duration) ? player.currentTime / player.duration : -1;
      for (i = 0; i < pk.length; i++) { var hh = pk[i] / mx * WH, xw = SX + i * SW / pk.length; s += R(xw, WY - hh / 2, Math.max(1, SW / pk.length - 1), hh, (i / pk.length) < prog ? "f-acc" : "f-dim", ' opacity="0.8"'); }
      s += T(SX, WY - WH / 2 - 8, "the render, 3 s · peak scaled · level " + nb.rms_db.toFixed(1) + " dBFS (" + fmtS(nb.rms_db - def.rms_db, 1) + " dB against 0)", "f-dim t-xs t-up");
      // travel to value, small
      var GX = 40, GY = 382, GW = 220, GH = 44;
      s += R(GX, GY - GH, GW, GH, "f-panel", ' rx="3"');
      var gd = "";
      for (i = 0; i <= 60; i++) { var tr = i / 60, vv2 = full ? -1 + 2 * tr : along(tr); gd += (i ? "L" : "M") + (GX + tr * GW).toFixed(1) + " " + (GY - GH / 2 - vv2 / 1 * GH / 2).toFixed(1); }
      s += P(gd, "s-acc", ' stroke-width="1.4"');
      s += C(GX + travel * GW, GY - GH / 2 - v * GH / 2, 3.5, "f-acc");
      s += T(GX, GY + 14, "travel →  (" + (full ? "straight, −1 to 1" : "calibrated") + ")", "f-dim t-xs");
      f.svg.innerHTML = s;
      var msg = "dose " + fmtS(v, 3) + " · nearest render " + fmtS(nb.v, 2) + ": level " + fmtS(nb.rms_db - def.rms_db, 1) + " dB against 0, centroid " + hz(nb.centroid_hz) + (nb.broken && nb.broken.length ? " · broken: " + nb.broken.join("; ") : "") + (v < safe[0] - 1e-9 || v > safe[1] + 1e-9 ? " · outside the measured range" : "");
      if (ro.textContent !== msg) ro.textContent = msg;
    };
    // turning: drag up and down, the wheel, the arrow keys
    var svg = f.svg, drag = null;
    svg.setAttribute("tabindex", "0");
    svg.setAttribute("aria-hidden", "false"); svg.setAttribute("role", "slider"); svg.setAttribute("aria-label", "Brightness knob");
    function setTravel(tr) { travel = clamp(tr, 0, 1); svg.setAttribute("aria-valuenow", dose().toFixed(3)); f.redraw(); }
    svg.addEventListener("pointerdown", function (e) { var r = svg.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * 960; if (x > 300) return; drag = { y: e.clientY, tr: travel }; svg.setPointerCapture(e.pointerId); e.preventDefault(); });
    svg.addEventListener("pointermove", function (e) { if (drag) setTravel(drag.tr + (drag.y - e.clientY) / 260); });
    svg.addEventListener("pointerup", function () { drag = null; });
    svg.addEventListener("wheel", function (e) { var r = svg.getBoundingClientRect(), x = (e.clientX - r.left) / r.width * 960; if (x > 300) return; e.preventDefault(); setTravel(travel - e.deltaY / 1200); }, { passive: false });
    svg.addEventListener("keydown", function (e) {
      if (e.key === "ArrowUp" || e.key === "ArrowRight") { setTravel(travel + 0.02); e.preventDefault(); }
      if (e.key === "ArrowDown" || e.key === "ArrowLeft") { setTravel(travel - 0.02); e.preventDefault(); }
    });
    f.el.querySelector("[data-act=listen]").addEventListener("click", function (e) {
      listen = !listen; e.currentTarget.setAttribute("aria-pressed", listen ? "true" : "false");
      if (listen) { var nb = nearest(); lastNearest = nb; play(nb.src, "Brightness " + fmtS(nb.v, 2) + " · Warm drone · seed 1", f, true); }
      else if (current.owner === f) player.pause();
    });
    f.el.querySelector("[data-act=full]").addEventListener("click", function (e) {
      var v = dose(); full = !full; e.currentTarget.setAttribute("aria-pressed", full ? "true" : "false");
      travel = travelFor(clamp(v, full ? -1 : (safe[0]), full ? 1 : safe[1])); f.redraw();
    });
    listeners.push(function () { if (current.owner !== f && listen) { listen = false; f.el.querySelector("[data-act=listen]").setAttribute("aria-pressed", "false"); } });
  });

  /* 4. an LFO on the Brightness knob, moving it along its travel */
  addFig("viz-lfo", function (f) {
    var shape = "sine", depth = 0.35, base = travelOf(0), ro = f.el.querySelector(".readout"), RATE = 4200;
    f.still = 1000;
    function lfo(ph) {
      ph = ph - Math.floor(ph);
      if (shape === "tri") return ph < 0.5 ? 4 * ph - 1 : 3 - 4 * ph;
      if (shape === "square") return ph < 0.5 ? 1 : -1;
      return Math.sin(ph * 2 * Math.PI);
    }
    f.draw = function (t) {
      var s = "", i, ph = t / RATE, tr = clamp(base + depth * 0.5 * lfo(ph), 0, 1), v = along(tr);
      var CX = 110, CY = 170, RK = 66, A0 = Math.PI * 0.75, A1 = Math.PI * 2.25;
      function ang(x) { return A0 + (A1 - A0) * x; }
      s += P(arcPath(CX, CY, RK + 12, A0, A1), "s-line", ' stroke-width="8" stroke-linecap="round"');
      var r0 = clamp(base - depth * 0.5, 0, 1), r1 = clamp(base + depth * 0.5, 0, 1);
      s += P(arcPath(CX, CY, RK + 12, ang(r0), ang(r1) + 0.001), "s-green", ' stroke-width="8" stroke-linecap="round" opacity="0.55"');
      s += C(CX, CY, RK - 4, "f-panel", ' style="stroke:var(--fig-line)"');
      var a = ang(tr);
      s += L(CX + (RK - 34) * Math.cos(a), CY + (RK - 34) * Math.sin(a), CX + (RK - 6) * Math.cos(a), CY + (RK - 6) * Math.sin(a), "s-acc", ' stroke-width="4" stroke-linecap="round"');
      var ab = ang(base);
      s += C(CX + (RK + 12) * Math.cos(ab), CY + (RK + 12) * Math.sin(ab), 3, "f-ink");
      s += T(CX - 28, CY + 5, fmtS(v, 3), "f-ink t-m");
      s += T(CX - 36, CY + 112, "the ring: the route's depth", "f-dim t-xs");
      // the LFO and what it does to the dose, over two cycles
      var GX = 230, GW = 270, G1 = 60, G2 = 190, GH = 46;
      s += T(GX, 30, "LFO 1, along the travel", "f-dim t-xs t-up");
      s += T(GX, G2 - GH - 18, "the dose it plays, through the curve", "f-dim t-xs t-up");
      var d1 = "", d2 = "";
      for (i = 0; i <= 160; i++) {
        var p2 = ph - 2 + i / 80, trr = clamp(base + depth * 0.5 * lfo(p2), 0, 1), vv = along(trr), x = GX + i / 160 * GW;
        d1 += (i ? "L" : "M") + x.toFixed(1) + " " + (G1 - (trr - base) / 0.5 * GH * 0.9).toFixed(1);
        d2 += (i ? "L" : "M") + x.toFixed(1) + " " + (G2 - clamp(vv / 0.3, -1.4, 1.4) * GH * 0.7).toFixed(1);
      }
      s += L(GX, G1, GX + GW, G1, "s-line"); s += L(GX, G2, GX + GW, G2, "s-line");
      s += P(d1, "s-green", ' stroke-width="1.5"'); s += P(d2, "s-acc", ' stroke-width="1.8" style="filter:drop-shadow(0 0 3px var(--accent))"');
      s += C(GX + GW, G1 - (tr - base) / 0.5 * GH * 0.9, 3.5, "f-green"); s += C(GX + GW, G2 - clamp(v / 0.3, -1.4, 1.4) * GH * 0.7, 3.5, "f-acc");
      s += T(GX + GW - 40, G2 + GH + 4, "0 = default", "f-dim t-xs");
      f.svg.innerHTML = s;
      var msg = "travel " + tr.toFixed(2) + " → dose " + fmtS(v, 3) + " · the swing spans " + fmtS(along(r0), 2) + " to " + fmtS(along(r1), 2);
      if (ro.textContent !== msg) ro.textContent = msg;
    };
    pressGroup(f, "data-shape", function (m) { shape = m; });
    sliders(f, function (k, v, out) { depth = v; out.textContent = v.toFixed(2); });
  });

  // for the units page's own script
  window.AA = { play: play, isPlaying: playing, isLoading: isLoading, stop: stop, player: player, onChange: function (f) { listeners.push(f); }, current: function () { return current; }, css: css };

  /* ---------- the frame loop ---------- */
  var lastCards = 0;
  var lastDraw = 0;
  function frame(now) {
    requestAnimationFrame(frame);
    if (now - lastDraw < 32) return;                 // at most about 30 redraws a second
    lastDraw = now;
    drawHero(now);
    drawDeck(now);
    figs.forEach(function (f) { if (f.visible) f.draw(now - f.t0); });
    (window.AA_FRAME || []).forEach(function (fn) { fn(now); });
    if (playing() && now - lastCards > 50) { paintCards(); lastCards = now; }
  }
  paintCards();
  if (!reduced) requestAnimationFrame(frame);
  else { drawHero(0); setInterval(function () { drawDeck(0); figs.forEach(function (f) { if (f.visible) f.redraw(); }); if (playing()) paintCards(); }, 200); }
})();
