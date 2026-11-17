// Abliterator: the abliteration page's figures. "Add, remove, turn": one direction in the
// stream and the three things a unit can do along it, each step a real render with its
// spectrogram. "Depth": one unit moved through the layers. Data: media.js, units-media.js.
// Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
(function () {
  "use strict";
  var M = window.MEDIA || {}, U = window.UNITS || { units: [] };
  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var FRAME = (window.AA_FRAME = window.AA_FRAME || []);
  function esc(s) { return String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function T(x, y, s, cls, extra) { return '<text x="' + x + '" y="' + y + '" class="' + (cls || "f-ink") + '"' + (extra || "") + ">" + esc(s) + "</text>"; }
  function L(x1, y1, x2, y2, cls, extra) { return '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '" class="' + cls + '"' + (extra || "") + "/>"; }
  function C(x, y, r, cls, extra) { return '<circle cx="' + x + '" cy="' + y + '" r="' + r + '" class="' + cls + '"' + (extra || "") + "/>"; }
  function arrow(x1, y1, x2, y2, cls, w) {
    var a = Math.atan2(y2 - y1, x2 - x1), h = 11;
    return L(x1, y1, x2, y2, cls, ' stroke-width="' + (w || 2.5) + '" stroke-linecap="round"') +
      '<path d="M' + x2 + " " + y2 + " L" + (x2 - h * Math.cos(a - 0.4)) + " " + (y2 - h * Math.sin(a - 0.4)) + " L" + (x2 - h * Math.cos(a + 0.4)) + " " + (y2 - h * Math.sin(a + 0.4)) + 'Z" class="' + cls.replace("s-", "f-") + '"/>';
  }
  function unit(id) { return U.units.filter(function (u) { return u.id === id; })[0]; }
  function fmt(v) { return (v > 0 ? "+" : v < 0 ? "−" : "") + String(+Math.abs(v).toFixed(2)); }

  /* ---------- add, remove, turn ---------- */
  (function () {
    var el = document.getElementById("viz-art"); if (!el) return;
    var svg = el.querySelector("svg"), ro = el.querySelector(".readout"), box = document.getElementById("art-spec");
    var inp = el.querySelector("input[type=range]"), out = el.querySelector("output");
    var steer = ((M.brightness || {}).steps || []).filter(function (s) { return s.v >= -0.2 && s.v <= 0.6; }).map(function (s) { return { v: s.v, src: s.src, rms: s.rms_db, broken: s.broken }; });
    var ab = unit("abliterate"), sh = unit("shifter");
    function steps(u, key) { return u && u.controls[key] ? u.controls[key].steps.map(function (s) { return { v: s.v, src: s.src, rms: s.rms_db, broken: s.broken }; }) : []; }
    var MODES = {
      add: { steps: steer, title: "Add: Steer toward brightness", knob: "dose", def: 0, line: function (v) { return "Steer at layer 6: brightness " + fmt(v); } },
      remove: { steps: steps(ab, "strength"), title: "Remove: Abliterate brightness", knob: "strength", def: 0, line: function (v) { return "abliterate block=all dir=brightness strength=" + v; } },
      turn: { steps: steps(sh, "angle"), title: "Turn: Rotate from brightness toward pulse", knob: "angle", def: 0, line: function (v) { return "shifter layer=6 dir=brightness dir2=pulse angle=" + v; } },
    };
    var mode = "add", idx = 0, shown = { k: null, h: null };
    function cur() { return MODES[mode].steps[idx] || { v: 0 }; }
    function setMode(m) {
      mode = m; var st = MODES[m].steps;
      var j = 0; st.forEach(function (s, k) { if (Math.abs(s.v - MODES[m].def) < Math.abs(st[j].v - MODES[m].def)) j = k; });
      idx = j; inp.min = 0; inp.max = st.length - 1; inp.value = idx; update();
    }
    function update() {
      var s = cur();
      out.textContent = mode === "turn" ? s.v + "°" : fmt(s.v);
      var key = mode + ":" + s.v;
      if (shown.k !== key && box && window.Spectro) {
        shown.k = key;
        window.Spectro.show(box, s.src, MODES[mode].title + " · " + (mode === "turn" ? s.v + "°" : fmt(s.v)), (s.rms != null ? s.rms.toFixed(1) + " dBFS" : "") + (s.broken && s.broken.length ? " · broken: " + s.broken[0] : ""));
        var lineEl = box.querySelector(".sbline"); if (lineEl) lineEl.textContent = MODES[mode].line(s.v);
      }
      draw(performance.now());
    }
    function draw(t) {
      var s = cur(), O = [210, 175], R = 120, dAng = -0.18, d = [Math.cos(dAng), Math.sin(dAng)], e2 = [-d[1], d[0]];
      var th = -0.95 + (reduced ? 0 : 0.06 * Math.sin(t / 900));
      var h = [R * Math.cos(th), R * Math.sin(th)], c = h[0] * d[0] + h[1] * d[1], q = h[0] * e2[0] + h[1] * e2[1], h2;
      if (mode === "add") h2 = [h[0] + R * 2.2 * s.v * d[0], h[1] + R * 2.2 * s.v * d[1]];
      else if (mode === "remove") h2 = [h[0] - Math.min(1, s.v) * c * d[0], h[1] - Math.min(1, s.v) * c * d[1]];
      else { var a = -s.v * Math.PI / 180, c2 = c * Math.cos(a) - q * Math.sin(a), q2 = c * Math.sin(a) + q * Math.cos(a); h2 = [c2 * d[0] + q2 * e2[0], c2 * d[1] + q2 * e2[1]]; }
      var out2 = "";
      for (var gx = 30; gx <= 400; gx += 30) out2 += L(gx, 20, gx, 320, "s-line", ' stroke-width="0.5"');
      for (var gy = 20; gy <= 320; gy += 30) out2 += L(30, gy, 400, gy, "s-line", ' stroke-width="0.5"');
      out2 += L(O[0] - 180 * d[0], O[1] - 180 * d[1], O[0] + 190 * d[0], O[1] + 190 * d[1], "s-acc", ' stroke-dasharray="6 4" stroke-width="1.3"');
      out2 += T(O[0] + 150 * d[0] - 40, O[1] + 150 * d[1] + 24, "brightness", "f-acc t-s");
      if (mode === "turn") { out2 += L(O[0] - 140 * e2[0], O[1] - 140 * e2[1], O[0] + 140 * e2[0], O[1] + 140 * e2[1], "s-blue", ' stroke-dasharray="6 4" stroke-width="1"'); out2 += T(O[0] - 130 * e2[0] + 6, O[1] - 130 * e2[1], "pulse", "f-blue t-s"); }
      if (mode === "remove") out2 += L(O[0], O[1], O[0] + c * d[0], O[1] + c * d[1], "s-acc", ' stroke-width="7" stroke-linecap="round" opacity="' + (0.5 * Math.max(0, 1 - Math.min(1, s.v))) + '"');
      out2 += arrow(O[0], O[1], O[0] + h[0], O[1] + h[1], "s-dim", 1.6);
      out2 += T(O[0] + h[0] + 8, O[1] + h[1], "h", "f-dim t-m");
      out2 += arrow(O[0], O[1], O[0] + h2[0], O[1] + h2[1], "s-green", 3);
      out2 += T(O[0] + h2[0] + 8, O[1] + h2[1] + 14, "h′", "f-green t-m");
      out2 += C(O[0], O[1], 3, "f-ink");
      var form = mode === "add" ? "h′ = h + ‖h‖ · dose · d" : mode === "remove" ? "h′ = h − strength · (h · d) d" : "(h · d, h · p) turned by the angle";
      out2 += T(440, 60, MODES[mode].title, "f-ink t-l"); out2 += T(440, 86, form, "f-acc t-m");
      out2 += T(440, 120, mode === "add" ? "the stream is pushed along the direction:" : mode === "remove" ? "the stream's part along the direction is taken out" : "the stream's part in the plane is turned:", "f-dim t-s");
      out2 += T(440, 138, mode === "add" ? "a dose is a share of the stream's own size" : mode === "remove" ? "of what each block's attention and MLP write" : "brightness becomes pulse, and pulse brightness", "f-dim t-s");
      out2 += T(440, 172, "the stream is 1,024 numbers; two of its directions are drawn", "f-dim t-xs t-up");
      out2 += T(440, 190, "the sound below is a real render at this setting", "f-dim t-xs t-up");
      svg.innerHTML = out2;
      var msg = MODES[mode].line(s.v) + (s.rms != null ? " · " + s.rms.toFixed(1) + " dBFS" : "") + (s.broken && s.broken.length ? " · the measures found it broken" : "");
      if (ro.textContent !== msg) ro.textContent = msg;
    }
    el.querySelectorAll("button[data-mode]").forEach(function (b) {
      b.addEventListener("click", function () { el.querySelectorAll("button[data-mode]").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); }); setMode(b.getAttribute("data-mode")); });
    });
    inp.addEventListener("input", function () { idx = +inp.value; update(); });
    var vis = true;
    if ("IntersectionObserver" in window) new IntersectionObserver(function (es) { es.forEach(function (e) { vis = e.isIntersecting; }); }).observe(el);
    FRAME.push(function (now) { if (vis) draw(now); });
    setMode("add");
  })();

  /* ---------- depth: one unit through the layers ---------- */
  (function () {
    var el = document.getElementById("viz-depth"); if (!el) return;
    var u = unit("saturator"), svg = el.querySelector("svg"), bar = el.querySelector(".bar"), box = document.getElementById("depth-spec");
    if (!u || !u.controls.at) return;
    var st = u.controls.at.steps, good = u.places || [], pick = st.filter(function (s) { return s.v === 6; })[0] || st[0];
    st.forEach(function (s) {
      var b = document.createElement("button");
      b.type = "button"; b.textContent = "layer " + s.v; b.setAttribute("aria-pressed", s === pick ? "true" : "false");
      b.addEventListener("click", function () { pick = s; bar.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); }); show(); });
      bar.appendChild(b);
    });
    function show() {
      var ok = good.indexOf(pick.v) >= 0, s = "";
      for (var l = 1; l <= 12; l++) {
        var x = 40 + (l - 1) * 72, m = st.some(function (q) { return q.v === l; }), g = good.indexOf(l) >= 0;
        s += '<rect x="' + x + '" y="40" width="60" height="44" rx="5" class="f-panel" style="stroke:var(--fig-line)"/>';
        s += T(x + 8, 66, "block " + l, "f-dim t-xs");
        s += C(x + 66, 120, l === pick.v ? 10 : 6, m ? (g ? "f-green" : "f-red") : "f-panel", ' style="stroke:' + (l === pick.v ? "var(--accent)" : "var(--fig-line)") + '" stroke-width="' + (l === pick.v ? 3 : 1) + '" opacity="0.9"');
      }
      s += L(30, 120, 900, 120, "s-ink", ' stroke-width="1.5" opacity="0.4"');
      var NAMES = { "warm-drone": "Warm drone", "shimmer": "Shimmering pad", "dark-field": "Dark field", "glass": "Glass chords" };
      var here = "Warm drone (this clip): " + pick.rms_db.toFixed(1) + " dBFS, " + (pick.broken && pick.broken.length ? pick.broken[0] : "held");
      var there = (pick.others || []).map(function (o) { return (NAMES[o.preset] || o.preset) + ": " + (o.rms_db != null ? o.rms_db.toFixed(1) + " dBFS, " : "") + (o.broken && o.broken.length ? o.broken[0] : "held"); }).join(" · ");
      s += T(40, 152, "Tanh Squash after block " + pick.v + " · " + (ok ? "held on both presets" : "broke on at least one preset"), ok ? "f-ink t-s" : "f-red t-s");
      s += T(40, 172, here + (there ? " · " + there : ""), "f-dim t-xs");
      svg.innerHTML = s;
      if (box && window.Spectro) window.Spectro.show(box, pick.src, "Tanh Squash at layer " + pick.v, pick.rms_db.toFixed(1) + " dBFS");
    }
    show();
  })();
})();
