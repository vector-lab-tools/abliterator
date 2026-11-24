// Abliterator: the figures on the signal chain's pages (Engine, Effects, Output), drawn in inline SVG
// with the home page's figure frame and helpers (site.js, window.AAFIG). Curves that stand for the
// instrument's own processing are computed with its formulas (postfx.py, dsp.py); the others are
// drawn, and their captions say so.
// Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
(function () {
  "use strict";
  var F = window.AAFIG;
  if (!F) return;
  var T = F.T, R = F.R, L = F.L, C = F.C, P = F.P, clamp = F.clamp;
  var SR = 48000;
  function db(v) { return 20 * Math.log10(Math.max(v, 1e-9)); }
  function pathOf(pts) { return pts.map(function (p, i) { return (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1); }).join(""); }
  function readout(f, html) { var ro = f.el.querySelector(".readout"); if (ro && ro.innerHTML !== html) ro.innerHTML = html; }
  function fmtHz(v) { return v >= 1000 ? (v / 1000).toFixed(v >= 10000 ? 0 : 1) + " kHz" : Math.round(v) + " Hz"; }

  /* ---------- biquads, as postfx._biquad (RBJ cookbook) ---------- */
  function biquad(kind, f, q, g) {
    var A = Math.pow(10, (g || 0) / 40), w = 2 * Math.PI * Math.min(f, SR / 2 * 0.98) / SR;
    var cw = Math.cos(w), sw = Math.sin(w), al = sw / (2 * (q || 0.707)), b, a, s2;
    if (kind === "lp") { b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]; a = [1 + al, -2 * cw, 1 - al]; }
    else if (kind === "hp") { b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]; a = [1 + al, -2 * cw, 1 - al]; }
    else if (kind === "peak") { b = [1 + al * A, -2 * cw, 1 - al * A]; a = [1 + al / A, -2 * cw, 1 - al / A]; }
    else if (kind === "lowshelf") {
      s2 = 2 * Math.sqrt(A) * al;
      b = [A * ((A + 1) - (A - 1) * cw + s2), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - s2)];
      a = [(A + 1) + (A - 1) * cw + s2, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - s2];
    } else {
      s2 = 2 * Math.sqrt(A) * al;
      b = [A * ((A + 1) + (A - 1) * cw + s2), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - s2)];
      a = [(A + 1) - (A - 1) * cw + s2, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - s2];
    }
    return [b[0] / a[0], b[1] / a[0], b[2] / a[0], a[1] / a[0], a[2] / a[0]];
  }
  function magDb(sos, f) {                       // the response of a cascade at frequency f, in dB
    var w = 2 * Math.PI * f / SR, c1 = Math.cos(w), s1 = Math.sin(w), c2 = Math.cos(2 * w), s2 = Math.sin(2 * w), m = 0;
    sos.forEach(function (q) {
      var nr = q[0] + q[1] * c1 + q[2] * c2, ni = -(q[1] * s1 + q[2] * s2);
      var dr = 1 + q[3] * c1 + q[4] * c2, di = -(q[3] * s1 + q[4] * s2);
      m += 10 * Math.log10((nr * nr + ni * ni) / (dr * dr + di * di));
    });
    return m;
  }
  // a response plot: log frequency from 20 Hz to 20 kHz across, ±range dB up and down
  function respFrame(x, y, w, h, range, label) {
    var s = R(x, y, w, h, "f-panel", ' style="stroke:var(--fig-line)"');
    [30, 100, 300, 1000, 3000, 10000].forEach(function (fq) {
      var px = x + w * Math.log(fq / 20) / Math.log(1000);
      s += L(px.toFixed(1), y, px.toFixed(1), y + h, "s-line") + T(px - 10, y + h + 13, fmtHz(fq).replace(" ", ""), "f-dim t-xs");
    });
    [-range, -range / 2, 0, range / 2, range].forEach(function (g) {
      var py = y + h / 2 - g / range * h / 2;
      s += L(x, py.toFixed(1), x + w, py.toFixed(1), g === 0 ? "s-dim" : "s-line") + T(x - 30, py + 3, (g > 0 ? "+" : g < 0 ? "−" : "") + Math.abs(g), "f-dim t-xs");
    });
    return s + T(x, y - 8, label, "f-dim t-xs t-up");
  }
  function respPath(sos, x, y, w, h, range) {
    var pts = [];
    for (var i = 0; i <= 240; i++) {
      var fq = 20 * Math.pow(1000, i / 240), g = clamp(magDb(sos, fq), -range * 1.4, range * 1.4);
      pts.push([x + w * i / 240, clamp(y + h / 2 - g / range * h / 2, y - 2, y + h + 2)]);
    }
    return pts;
  }

  /* ---------- Engine: the sampler (drawn) ---------- */
  F.add("viz-sampler", function (f) {
    var st = { guidance: 3, temperature: 1.3, topk: 8 };
    // twenty candidate tokens: how likely each is without the style, and with it (drawn, not the model's numbers)
    var base = [], style = [], i;
    for (i = 0; i < 20; i++) { base.push(1.6 * Math.cos(i * 1.7) + 0.4 * Math.sin(i * 0.6) - i * 0.05); style.push(base[i] + 2.4 * Math.exp(-Math.pow((i - 12) / 2.2, 2)) - 0.6 * Math.exp(-Math.pow((i - 3) / 2, 2))); }
    f.still = 0;
    f.draw = function () {
      var lg = base.map(function (b, k) { return (b + st.guidance / 3 * (style[k] - b)) / st.temperature; });
      var mx = Math.max.apply(null, lg), ex = lg.map(function (v) { return Math.exp(v - mx); }), sum = ex.reduce(function (a, b) { return a + b; }, 0);
      var p = ex.map(function (v) { return v / sum; });
      var order = p.map(function (v, k) { return k; }).sort(function (a, b) { return p[b] - p[a]; }), kept = {};
      order.slice(0, st.topk).forEach(function (k) { kept[k] = 1; });
      var ksum = 0; order.slice(0, st.topk).forEach(function (k) { ksum += p[k]; });
      var X = 70, Y = 40, W = 820, H = 200, bw = W / 20, s = "", top = Math.max.apply(null, p);
      s += T(X, 26, "candidate tokens for the next step, and the share of the choice each gets", "f-dim t-xs t-up");
      s += L(X, Y + H, X + W, Y + H, "s-line");
      for (i = 0; i < 20; i++) {
        var on = !!kept[i], q = on ? p[i] / ksum : 0, hh = (on ? q : p[i]) / Math.max(top / ksum, 0.0001) * (H - 10) * (on ? 1 : 1);
        hh = Math.min(H - 4, (on ? q : p[i]) / (top / ksum) * (H - 10));
        s += R((X + i * bw + 4).toFixed(1), (Y + H - hh).toFixed(1), (bw - 8).toFixed(1), hh.toFixed(1), on ? "f-acc" : "f-dim", on ? "" : ' opacity="0.35"');
        if (on && q > 0.04) s += T(X + i * bw + 5, Y + H - hh - 5, Math.round(q * 100) + "%", "f-ink t-xs");
      }
      s += T(X, Y + H + 18, "dim: cut by Top-k; the rest share the choice in proportion", "f-dim t-xs");
      f.svg.innerHTML = s;
      readout(f, "Strength <b>" + st.guidance.toFixed(1) + "</b> · Temperature <b>" + st.temperature.toFixed(2) + "</b> · Top-k <b>" + st.topk + "</b> of 20 shown: the likeliest token gets <b>" + Math.round(p[order[0]] / ksum * 100) + "%</b> of the choice");
    };
    F.sliders(f, function (k, v, out) { st[k] = k === "topk" ? Math.round(v) : v; out.textContent = k === "topk" ? String(Math.round(v)) : v.toFixed(k === "temperature" ? 2 : 1); });
  });

  /* ---------- Engine: Speed (drawn from its rules) ---------- */
  F.add("viz-speed", function (f) {
    var st = { speed: 0.5, mode: "tape" };
    function env(t) { return Math.exp(-Math.pow((t % 0.5) / 0.09, 1)) * 0.9 + 0.1; }         // two notes a second
    function src(t) { return env(t) * (Math.sin(2 * Math.PI * 9 * t) + 0.35 * Math.sin(2 * Math.PI * 18 * t + 0.6)); }
    function rnd(n) { var x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); }
    f.still = 900;
    f.draw = function (t) {
      var X = 60, W = 860, s = "", i, n = 700, sp = st.speed, ph = (t / 1000) % 2;
      var Y1 = 54, Y2 = 210, A = 42;
      s += T(X, 30, "the model's sound: one second", "f-dim t-xs t-up");
      s += T(X, 196, st.mode === "tape" ? "Tape: read slower, so every cycle is longer and the pitch falls" : st.mode === "stretch" ? "Stretch: the cycles keep their length; the notes last longer" : "Smear: the pitch is kept, the detail washed into a slow cloud", "f-dim t-xs t-up");
      var a = [], b = [];
      for (i = 0; i <= n; i++) { var u = i / n; a.push([X + u * W, Y1 + A - src(u) * A * 0.9]); }
      // the output shows the same width of time: 1 second of output holds sp seconds of the model's sound
      for (i = 0; i <= n; i++) {
        var v = i / n, y;
        if (st.mode === "tape") y = src(v * sp);
        else if (st.mode === "stretch") y = env(v * sp) * (Math.sin(2 * Math.PI * 9 * v) + 0.35 * Math.sin(2 * Math.PI * 18 * v + 0.6));
        else {
          var e = 0, k, wsum = 0;
          for (k = -6; k <= 6; k++) { var c = v * sp + k * 0.04, w = Math.exp(-Math.pow(k / 3, 2)); e += env(Math.max(0, c)) * w; wsum += w; }
          e /= wsum;
          var g = Math.floor(v * 24), fr = v * 24 - g, ph0 = rnd(g) * 6.283, ph1 = rnd(g + 1) * 6.283;
          y = e * ((1 - fr) * Math.sin(2 * Math.PI * 9 * v + ph0) + fr * Math.sin(2 * Math.PI * 9 * v + ph1) + 0.3 * Math.sin(2 * Math.PI * 18 * v + 2 * ph0));
        }
        b.push([X + v * W, Y2 + A - y * A * 0.9]);
      }
      s += R(X, Y1 - 4, W, 2 * A + 8, "f-panel");
      s += R(X, Y2 - 4, W, 2 * A + 8, "f-panel");
      // the part of the model's sound that the second of output reads
      s += R(X, Y1 - 4, (W * sp).toFixed(1), 2 * A + 8, "f-acc", ' opacity="0.08"');
      s += L((X + W * sp).toFixed(1), Y1 - 8, (X + W * sp).toFixed(1), Y1 + 2 * A + 8, "s-acc", ' stroke-dasharray="3 3"');
      s += P(pathOf(a), "s-dim", ' stroke-width="1.2"');
      s += P(pathOf(b), "s-acc", ' stroke-width="1.4"');
      var hx = X + W * (ph / 2);
      s += L(hx.toFixed(1), Y2 - 6, hx.toFixed(1), Y2 + 2 * A + 6, "s-ink", ' opacity="0.6"');
      s += L((X + W * sp * ph / 2).toFixed(1), Y1 - 6, (X + W * sp * ph / 2).toFixed(1), Y1 + 2 * A + 6, "s-ink", ' opacity="0.6"');
      s += T(X + W * sp + 6, Y1 + 10, "read in one second at " + sp.toFixed(2) + "×", "f-acc t-xs");
      s += T(X, Y2 + 2 * A + 22, "one second of what you hear", "f-dim t-xs t-up");
      f.svg.innerHTML = s;
      var semis = 12 * Math.log2(sp);
      readout(f, "One second of the model's sound lasts <b>" + (1 / sp).toFixed(2) + " s</b>; the model writes <b>" + (25 * sp).toFixed(1) + "</b> frames a second instead of 25" +
        (st.mode === "tape" ? "; the pitch falls by <b>" + Math.abs(semis).toFixed(1) + "</b> semitones" : "; the pitch stays where it was"));
    };
    F.pressGroup(f, "data-mode", function (m) { st.mode = m; });
    F.sliders(f, function (k, v, out) { st.speed = Math.pow(2, v); out.textContent = st.speed.toFixed(2) + "×"; });
  });

  /* ---------- Effects: EQ 8's response, computed as postfx.EQ8 designs it ---------- */
  F.add("viz-eq8", function (f) {
    var HZ = [60, 150, 400, 1000, 2500, 6000], st = { lowcut: 80, highcut: 14000, slope: 1, b0: 0, b1: 2, b2: -3, b3: 0, b4: 2.5, b5: -1.5 };
    f.still = 0;
    f.draw = function () {
      var qs = st.slope ? [0.5412, 1.3066] : [0.7071], sos = [];
      qs.forEach(function (q) { sos.push(biquad("hp", st.lowcut, q)); });
      HZ.forEach(function (hz, i) { if (st["b" + i]) sos.push(biquad("peak", hz, 1.2, st["b" + i])); });
      qs.forEach(function (q) { sos.push(biquad("lp", st.highcut, q)); });
      var X = 60, Y = 34, W = 860, H = 210, s = respFrame(X, Y, W, H, 18, "EQ 8's response, as the instrument designs it");
      var pts = respPath(sos, X, Y, W, H, 18);
      s += P(pathOf(pts) + "L" + (X + W) + " " + (Y + H / 2) + "L" + X + " " + (Y + H / 2) + "Z", "s-acc", ' stroke-width="0" style="fill:var(--accent);opacity:.12"');
      s += P(pathOf(pts), "s-acc", ' stroke-width="2"');
      HZ.forEach(function (hz, i) {
        var px = X + W * Math.log(hz / 20) / Math.log(1000), g = st["b" + i];
        s += C(px.toFixed(1), (Y + H / 2 - g / 18 * H / 2).toFixed(1), 4, g ? "f-acc" : "f-dim");
      });
      f.svg.innerHTML = s;
      readout(f, "Low cut <b>" + fmtHz(st.lowcut) + "</b> · High cut <b>" + fmtHz(st.highcut) + "</b> · cuts <b>" + (st.slope ? 24 : 12) + " dB</b> an octave · bells about 1.2 octaves wide (Q 1.2)");
    };
    F.sliders(f, function (k, v, out) {
      if (k === "lowcut" || k === "highcut") { st[k] = Math.round(Math.pow(10, v)); out.textContent = fmtHz(st[k]).replace(" ", ""); }
      else { st[k] = v; out.textContent = (v > 0 ? "+" : "") + v.toFixed(1); }
    });
    F.pressGroup(f, "data-slope", function (m) { st.slope = +m; });
  });

  /* ---------- Effects: Tape's curve and its speed EQ, computed as postfx.Tape ---------- */
  F.add("viz-tape", function (f) {
    var st = { drive: 0.6, level: -26, speed: 1 };
    var BUMP = [35, 50, 70, 100], BUMP_DB = [4, 3.5, 2.5, 1.5], ROLL = [4000, 7000, 12000, 18000], IPS = ["3.75", "7.5", "15", "30"];
    function curve(x) {
      var g = 1 + 15 * Math.pow(st.drive, 1.5), c = 0.12 * st.drive, ref = 0.05;
      return (Math.tanh(g * x + c) - Math.tanh(c)) * (ref / (Math.tanh(g * ref + c) - Math.tanh(c)));
    }
    f.still = 400;
    f.draw = function (t) {
      var s = "", i, amp = Math.pow(10, st.level / 20);
      // left: a sine at the level set, before (dim) and after (amber), scaled to the input's peak
      var X = 50, Y = 50, W = 400, H = 180, pk = 0, out = [];
      for (i = 0; i <= 300; i++) { var y = curve(amp * Math.sin(2 * Math.PI * 2 * i / 300 + t / 900)); out.push(y); pk = Math.max(pk, Math.abs(y)); }
      var sc = Math.max(amp, pk);
      s += T(X, 30, "a sine at " + st.level + " dBFS, in (grey) and out (amber)", "f-dim t-xs t-up");
      s += R(X, Y, W, H, "f-panel") + L(X, Y + H / 2, X + W, Y + H / 2, "s-line");
      var pin = [], pout = [];
      for (i = 0; i <= 300; i++) {
        pin.push([X + W * i / 300, Y + H / 2 - amp * Math.sin(2 * Math.PI * 2 * i / 300 + t / 900) / sc * (H / 2 - 8)]);
        pout.push([X + W * i / 300, Y + H / 2 - out[i] / sc * (H / 2 - 8)]);
      }
      s += P(pathOf(pin), "s-dim", ' stroke-width="1.2"') + P(pathOf(pout), "s-acc", ' stroke-width="1.8"');
      // right: the speed's EQ, the head bump and the roll-off
      var sos = [biquad("hp", 12, 0.707), biquad("peak", BUMP[st.speed], 0.9, BUMP_DB[st.speed]), biquad("lp", ROLL[st.speed], 0.6), biquad("lp", ROLL[st.speed] * 1.4, 0.7), biquad("highshelf", 3000, 0.707, -2.5 + 0.7 * st.speed)];
      var X2 = 530, W2 = 390;
      s += respFrame(X2, Y, W2, H, 12, "the EQ of tape at " + IPS[st.speed] + " inches a second");
      s += P(pathOf(respPath(sos, X2, Y, W2, H, 12)), "s-acc", ' stroke-width="2"');
      f.svg.innerHTML = s;
      readout(f, "Drive <b>" + st.drive.toFixed(2) + "</b>: the peak comes out at <b>" + db(pk).toFixed(1) + " dBFS</b> for a sine in at <b>" + st.level + " dBFS</b> · head bump <b>+" + BUMP_DB[st.speed] + " dB at " + BUMP[st.speed] + " Hz</b> · roll-off from <b>" + fmtHz(ROLL[st.speed]) + "</b>");
    };
    F.sliders(f, function (k, v, out) { st[k] = v; out.textContent = k === "level" ? v + " dB" : v.toFixed(2); });
    F.pressGroup(f, "data-ips", function (m) { st.speed = +m; });
  });

  /* ---------- Effects: Pan's equal-power law, as postfx.Pan ---------- */
  F.add("viz-pan", function (f) {
    var st = { pos: 0, rate: 0.25, depth: 0.5 };
    f.still = 0;
    f.draw = function (t) {
      var p = clamp(st.pos + st.depth * Math.sin(2 * Math.PI * st.rate * t / 1000), -1, 1), s = "", i;
      var X = 70, Y = 40, W = 520, H = 200;
      s += T(X, 26, "each side's gain against the position", "f-dim t-xs t-up");
      s += R(X, Y, W, H, "f-panel");
      [0, -3, -6, -12].forEach(function (g) { var y = Y + H * (1 - (Math.pow(10, g / 20) * Math.SQRT2) / 1.5); s += L(X, y.toFixed(1), X + W, y.toFixed(1), "s-line") + T(X - 34, y + 3, (g ? "−" : "") + Math.abs(g - 3 + 3) + "", "f-dim t-xs"); });
      var l = [], r = [];
      for (i = 0; i <= 100; i++) {
        var q = -1 + 2 * i / 100, ang = (q + 1) * Math.PI / 4;
        l.push([X + W * i / 100, Y + H * (1 - Math.cos(ang) * Math.SQRT2 / 1.5)]);
        r.push([X + W * i / 100, Y + H * (1 - Math.sin(ang) * Math.SQRT2 / 1.5)]);
      }
      s += P(pathOf(l), "s-blue", ' stroke-width="2"') + P(pathOf(r), "s-acc", ' stroke-width="2"');
      var ang = (p + 1) * Math.PI / 4, px = X + W * (p + 1) / 2, gl = Math.cos(ang) * Math.SQRT2, gr = Math.sin(ang) * Math.SQRT2;
      s += L(px.toFixed(1), Y, px.toFixed(1), Y + H, "s-ink", ' opacity="0.5"');
      s += C(px.toFixed(1), (Y + H * (1 - gl / 1.5)).toFixed(1), 5, "f-blue") + C(px.toFixed(1), (Y + H * (1 - gr / 1.5)).toFixed(1), 5, "f-acc");
      s += T(X, Y + H + 16, "L", "f-dim t-xs") + T(X + W / 2 - 3, Y + H + 16, "C", "f-dim t-xs") + T(X + W - 6, Y + H + 16, "R", "f-dim t-xs");
      s += T(X + W - 120, Y + 16, "left", "f-blue t-s") + T(X + W - 60, Y + 16, "right", "f-acc t-s");
      // the field: where the sound sits
      var CX = 790, CY = 140, RR = 92;
      s += C(CX, CY, RR, "f-panel", ' style="stroke:var(--fig-line)"');
      s += T(CX - 52, 26, "where it sits", "f-dim t-xs t-up");
      s += L(CX - RR, CY, CX + RR, CY, "s-line");
      s += C((CX + p * RR * 0.85).toFixed(1), CY, 9, "f-acc", ' style="filter:drop-shadow(0 0 6px var(--accent))"');
      f.svg.innerHTML = s;
      readout(f, "Position <b>" + (p >= 0 ? "+" : "−") + Math.abs(p).toFixed(2) + "</b> · left <b>" + db(gl).toFixed(1) + " dB</b> · right <b>" + db(gr).toFixed(1) + " dB</b>: at the centre both pass at unity, at a side one is silent and the other 3 dB up");
    };
    F.sliders(f, function (k, v, out) { st[k] = v; out.textContent = k === "rate" ? v.toFixed(2) + " Hz" : v.toFixed(2); });
  });

  /* ---------- Output: Dynamics' curve, as dsp.Master._dynamics sets its targets ---------- */
  F.add("viz-dyn", function (f) {
    var st = { ratio: 3, thresh: -24, gate: -60 };
    f.still = 2000;
    f.draw = function (t) {
      var X = 80, Y = 30, W = 330, H = 250, s = "", i;
      function out(d) {
        var o = d - (d > st.thresh ? (d - st.thresh) * (1 - 1 / st.ratio) : 0);
        if (st.gate > -79.5 && d < st.gate) o -= 40;
        return o;
      }
      function sx(d) { return X + W * (d + 80) / 80; }
      function sy(d) { return Y + H * (1 - (d + 80) / 80); }
      s += R(X, Y, W, H, "f-panel");
      [-80, -60, -40, -20, 0].forEach(function (d) { s += L(sx(d), Y, sx(d), Y + H, "s-line") + L(X, sy(d), X + W, sy(d), "s-line") + T(sx(d) - 8, Y + H + 14, String(d).replace("-", "−"), "f-dim t-xs") + T(X - 28, sy(d) + 3, String(d).replace("-", "−"), "f-dim t-xs"); });
      s += L(sx(-80), sy(-80), sx(0), sy(0), "s-dim", ' stroke-dasharray="3 4"');
      var pts = [];
      for (i = 0; i <= 160; i++) { var d = -80 + i * 0.5; pts.push([sx(d), clamp(sy(out(d)), Y, Y + H)]); }
      s += P(pathOf(pts), "s-acc", ' stroke-width="2"');
      s += T(X, 22, "level out against level in, dBFS", "f-dim t-xs t-up");
      // a level that rises and falls, and the reduction on LEDs as the panel shows them
      var lvl = -36 + 26 * Math.sin(t / 1400) + 6 * Math.sin(t / 310), red = out(lvl) - lvl;
      s += C(sx(lvl).toFixed(1), clamp(sy(out(lvl)), Y, Y + H).toFixed(1), 6, "f-acc", ' style="filter:drop-shadow(0 0 5px var(--accent))"');
      var X2 = 520, comp = Math.max(0, lvl - st.thresh) * (1 - 1 / st.ratio), gated = st.gate > -79.5 && lvl < st.gate;
      s += T(X2, 70, "COMP", "f-dim t-xs t-up") + T(X2, 120, "GATE", "f-dim t-xs t-up");
      // ten cells each, on the panel's scales: green to 4 dB, yellow to 10, red above
      var COMP = [1, 2, 3, 4, 6, 8, 10, 12, 15, 20], GATE = [2, 4, 6, 8, 10, 15, 20, 25, 30, 40], gdb = gated ? 40 : 0;
      for (i = 0; i < 10; i++) {
        var band = i < 4 ? "#4d9f5a" : i < 7 ? "#d4b13a" : "#c84a38";
        s += R(X2 + 60 + i * 26, 58, 22, 14, "f-panel", comp >= COMP[i] ? ' style="fill:' + band + ';stroke:var(--fig-line)"' : ' style="stroke:var(--fig-line)"');
        s += R(X2 + 60 + i * 26, 108, 22, 14, "f-panel", gdb >= GATE[i] ? ' style="fill:#4dff8a;stroke:var(--fig-line)"' : ' style="stroke:var(--fig-line)"');
      }
      s += T(X2 + 340, 70, comp >= 0.05 ? "−" + comp.toFixed(1) : "0.0", "f-ink t-s") + T(X2 + 340, 120, gated ? "−40" : "0.0", "f-ink t-s");
      s += T(X2, 180, "in " + lvl.toFixed(1) + " dBFS → out " + out(lvl).toFixed(1), "f-ink t-m");
      s += T(X2, 204, "COMP cells: 1, 2, 3, 4, 6, 8, 10, 12, 15, 20 dB of reduction", "f-dim t-xs");
      f.svg.innerHTML = s;
      readout(f, "Ratio <b>" + st.ratio.toFixed(1) + ":1</b> above <b>" + st.thresh + " dBFS</b> · gate " + (st.gate > -79.5 ? "closes <b>40 dB</b> below <b>" + st.gate + " dBFS</b>" : "<b>off</b>") + " · now <b>" + red.toFixed(1) + " dB</b>");
    };
    F.sliders(f, function (k, v, out) { st[k] = v; out.textContent = k === "ratio" ? v.toFixed(1) + ":1" : k === "gate" && v <= -79.5 ? "off" : v + " dB"; });
  });

  /* ---------- Output: De-click (drawn) ---------- */
  F.add("viz-declick", function (f) {
    var st = { on: true };
    var n = 240, x = [], i;
    for (i = 0; i < n; i++) x.push(0.45 * Math.sin(2 * Math.PI * i / 60) + 0.12 * Math.sin(2 * Math.PI * i / 13));
    var b0 = 128, burst = [0.95, -0.9, 0.85, -0.7, 0.5];
    burst.forEach(function (v, k) { x[b0 + k] = v; });
    f.still = 0;
    f.draw = function () {
      var X = 50, Y = 40, W = 860, H = 200, s = "", y = x.slice();
      if (st.on) { var a = y[b0 - 1], b = y[b0 + burst.length]; for (i = 0; i < burst.length; i++) y[b0 + i] = a + (b - a) * (i + 1) / (burst.length + 1); }
      s += R(X, Y, W, H, "f-panel") + L(X, Y + H / 2, X + W, Y + H / 2, "s-line");
      var pts = y.map(function (v, k) { return [X + W * k / (n - 1), Y + H / 2 - v * H / 2 * 0.95]; });
      if (st.on) s += P(pathOf(x.map(function (v, k) { return [X + W * k / (n - 1), Y + H / 2 - v * H / 2 * 0.95]; }).slice(b0 - 2, b0 + burst.length + 2)), "s-red", ' stroke-width="1" stroke-dasharray="2 3" opacity="0.7"');
      s += P(pathOf(pts), "s-acc", ' stroke-width="1.6"');
      pts.forEach(function (p) { s += C(p[0].toFixed(1), p[1].toFixed(1), 1.4, "f-acc"); });
      var ax = X + W * (b0 + 64) / (n - 1);
      s += L(ax.toFixed(1), Y, ax.toFixed(1), Y + H, "s-blue", ' stroke-dasharray="3 3"') + T(ax + 5, Y + 14, "64 samples ahead (1.3 ms)", "f-blue t-xs");
      s += T(X, 28, st.on ? "De-click on: the burst redrawn as a straight line between the samples either side" : "De-click off: five samples swinging up and down, heard as a click", "f-dim t-xs t-up");
      f.svg.innerHTML = s;
      readout(f, st.on ? "A burst is a run of samples that jump far above the local level and reverse from one sample to the next; it is <b>redrawn</b>, and everything else passes untouched" : "The model wrote this: a unit pushed hard (Quantise at 6 bits, Neuron at a large scale) can make it");
    };
    F.pressGroup(f, "data-dc", function (m) { st.on = m === "on"; });
  });

  /* ---------- Output: the dropout cover (drawn, with dsp.Conceal's times) ---------- */
  F.add("viz-cover", function (f) {
    var st = { on: true };
    f.still = 1000;
    f.draw = function (t) {
      var X = 50, Y = 50, W = 860, H = 170, s = "", i, n = 800;
      // 100 ms across: blocks arrive, a gap from 40 to 70 ms, then audio again
      function sig(ms) { return 0.6 * Math.sin(2 * Math.PI * ms / 4.3) * (0.8 + 0.2 * Math.sin(ms / 9)); }
      var g0 = 40, g1 = 70, pts = [];
      for (i = 0; i <= n; i++) {
        var ms = 100 * i / n, v;
        if (ms < g0) v = sig(ms);
        else if (ms < g1) {
          if (!st.on) v = 0;
          else { var k = ms - g0, blk = 10.7, ph = Math.floor(k / blk), r = k % blk, src = ph % 2 === 0 ? g0 - r : g0 - blk + r; v = sig(src) * Math.max(0, 1 - k / 30); }
        } else {
          var kk = ms - g1, xf = st.on ? Math.min(1, kk / 5) : 1, cov = st.on ? 0 : 0;
          v = sig(ms) * xf + cov * (1 - xf);
        }
        pts.push([X + W * ms / 100, Y + H / 2 - v * H / 2]);
      }
      s += R(X, Y, W, H, "f-panel") + L(X, Y + H / 2, X + W, Y + H / 2, "s-line");
      s += R((X + W * g0 / 100).toFixed(1), Y, (W * (g1 - g0) / 100).toFixed(1), H, "f-red", ' opacity="0.07"');
      s += T(X + W * g0 / 100 + 6, Y + 14, "no audio ready", "f-red t-xs t-up");
      s += P(pathOf(pts), "s-acc", ' stroke-width="1.6"');
      var hx = X + W * ((t / 30) % 100) / 100;
      s += L(hx.toFixed(1), Y - 4, hx.toFixed(1), Y + H + 4, "s-ink", ' opacity="0.5"');
      [0, 20, 40, 60, 80, 100].forEach(function (ms) { s += T(X + W * ms / 100 - 8, Y + H + 16, ms + " ms", "f-dim t-xs"); });
      s += T(X, 30, st.on ? "the cover: the last block played on, mirrored, fading over 30 ms; then a 5 ms crossfade back" : "without the cover: the sound stops dead and starts dead, a click at each edge", "f-dim t-xs t-up");
      f.svg.innerHTML = s;
      readout(f, st.on ? "Only what is heard is covered: a saved sound is taken from the frames the model wrote, so a dropout heard live is not in it" : "Each edge is a step in the waveform, which is heard as a click");
    };
    F.pressGroup(f, "data-cover", function (m) { st.on = m === "on"; });
  });
})();
