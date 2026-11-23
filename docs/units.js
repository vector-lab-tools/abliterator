// Abliterator: the units page. The two figures (where the units live, where they act), and a
// card for each unit with a small drawing of what it does and its measured sweep, playable.
// The data is this machine's unit sweep (units-media.js, made by bench/site_media.py).
// Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
(function () {
  "use strict";
  var D = window.UNITS || { units: [] };
  var reduced = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var FRAME = (window.AA_FRAME = window.AA_FRAME || []);
  // VAU's name, where this instrument calls the unit something else (Frame Repeat, VAU's vPosition Repeat)
const otherName = (u) => u.vau_title && u.vau_title.replace(/^v/, "") !== u.title;
function esc(s) { return String(s == null ? "" : s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }
  function T(x, y, s, cls, extra) { return '<text x="' + x + '" y="' + y + '" class="' + (cls || "f-ink") + '"' + (extra || "") + ">" + esc(s) + "</text>"; }
  function R(x, y, w, h, cls, extra) { return '<rect x="' + x + '" y="' + y + '" width="' + Math.max(0, w) + '" height="' + Math.max(0, h) + '" class="' + cls + '"' + (extra || "") + "/>"; }
  function L(x1, y1, x2, y2, cls, extra) { return '<line x1="' + x1 + '" y1="' + y1 + '" x2="' + x2 + '" y2="' + y2 + '" class="' + cls + '"' + (extra || "") + "/>"; }
  function C(x, y, r, cls, extra) { return '<circle cx="' + x + '" cy="' + y + '" r="' + r + '" class="' + cls + '"' + (extra || "") + "/>"; }
  function P(d, cls, extra) { return '<path d="' + d + '" class="' + cls + '" fill="none"' + (extra || "") + "/>"; }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function arrow(x1, y1, x2, y2, cls, w) {
    var a = Math.atan2(y2 - y1, x2 - x1), h = 7;
    return L(x1, y1, x2, y2, cls, ' stroke-width="' + (w || 2) + '" stroke-linecap="round"') +
      '<path d="M' + x2 + " " + y2 + " L" + (x2 - h * Math.cos(a - 0.45)) + " " + (y2 - h * Math.sin(a - 0.45)) + " L" + (x2 - h * Math.cos(a + 0.45)) + " " + (y2 - h * Math.sin(a + 0.45)) + 'Z" class="' + cls.replace("s-", "f-") + '"/>';
  }
  function visible(el, cb) {
    if (!("IntersectionObserver" in window)) { cb(true); return; }
    new IntersectionObserver(function (es) { es.forEach(function (e) { cb(e.isIntersecting); }); }, { rootMargin: "80px" }).observe(el);
  }
  function play(src, label, owner, loop) { if (window.AA) window.AA.play(src, label, owner, loop); }
  var owners = [];
  function paintOwners() {
    if (!window.AA) return;
    var cur = window.AA.current(), on = window.AA.isPlaying();
    owners.forEach(function (b) { b.classList.toggle("on", on && cur.owner === b); });
  }

  var BYID = {}; D.units.forEach(function (u) { BYID[u.id] = u; });

  /* ---------- 1. where the units live ---------- */
  (function () {
    var el = document.getElementById("viz-where"); if (!el) return;
    var svg = el.querySelector("svg"), ro = el.querySelector(".readout"), vis = true;
    visible(el, function (v) { vis = v; });
    var CW = 250;
    var cols = [[20, "VAU", "the package vau · own repository", ["delay.py", "reverb.py", "saturator.py", "erosion.py", "shifter.py", "abliterate.py", "neuron.py", "…"], "f-blue"],
                [355, "Abliterator", "the host · this repository", ["units.py: reads each line,", "checks it, names it,", "gives measured defaults", "mrt2_proc.py: wraps the blocks,", "bridges MLX ↔ torch", "scope.py: the meter"], "f-acc"],
                [690, "Magenta RT 2", "the model · MLX · weights not here", ["temporal transformer,", "12 blocks", "depth transformer,", "codebook tokens", "SpectroStream codec → audio"], "f-green"]];
    function draw(t) {
      var s = "";
      cols.forEach(function (c, i) {
        s += R(c[0], 30, CW, 220, "f-panel", ' rx="8" style="stroke:var(--fig-line)"');
        s += R(c[0], 30, CW, 4, c[4], ' rx="2"');
        s += T(c[0] + 14, 58, c[1], "f-ink t-l"); s += T(c[0] + 14, 76, c[2], "f-dim t-xs t-up");
        c[3].forEach(function (line, j) { s += T(c[0] + 14, 104 + j * 19, line, i === 0 ? "f-blue t-s" : "f-ink t-s"); });
      });
      s += arrow(276, 140, 349, 140, "s-dim", 1.5); s += T(283, 132, "installed", "f-dim t-xs");
      s += arrow(611, 140, 684, 140, "s-dim", 1.5); s += T(620, 132, "attaches", "f-dim t-xs");
      // one frame's stream, out of the model, through a unit, and back
      var k = reduced ? 0.5 : (t % 3200) / 3200, y0 = 300, xs = [805, 640, 480, 640, 805];
      s += T(20, 282, "each step, for the frame being written", "f-dim t-xs t-up");
      s += R(745, 286, 120, 30, "f-panel", ' rx="5" style="stroke:var(--green)"'); s += T(756, 305, "stream at layer 6", "f-ink t-xs");
      s += R(580, 286, 120, 30, "f-panel", ' rx="5" style="stroke:var(--fig-line)"'); s += T(592, 305, "MLX → torch", "f-ink t-xs");
      s += R(420, 286, 120, 30, "f-panel", ' rx="5" style="stroke:var(--blue)"'); s += T(432, 305, "the unit's edit", "f-ink t-xs");
      s += P("M745 301 L700 301 M580 301 L540 301", "s-dim", ' stroke-dasharray="3 3"');
      s += P("M480 316 Q480 350 640 350 Q805 350 805 316", "s-dim", ' stroke-dasharray="3 3"');
      s += T(560, 345, "torch → MLX, back into the block", "f-dim t-xs");
      var seg = Math.min(3, Math.floor(k * 4)), f = k * 4 - seg, x, y;
      if (seg < 2) { x = xs[seg] + (xs[seg + 1] - xs[seg]) * f; y = y0 + 1; }
      else { var q = (seg - 2 + f) / 2; x = 480 + (805 - 480) * q; y = 316 + Math.sin(q * Math.PI) * 34; }
      s += C(x, y, 5, seg >= 2 ? "f-acc" : "f-green", ' style="filter:drop-shadow(0 0 4px var(--accent))"');
      svg.innerHTML = s;
      var msg = "The units' code is in VAU; Abliterator hosts it; the model is Magenta RT 2. Each step the stream for one frame goes out of MLX, through the unit, and back.";
      if (ro.textContent !== msg) ro.textContent = msg;
    }
    FRAME.push(function (now) { if (vis) draw(now); });
    draw(0);
  })();

  /* ---------- 2. where they act inside the model ---------- */
  (function () {
    var el = document.getElementById("viz-inside"); if (!el) return;
    var svg = el.querySelector("svg"), ro = el.querySelector(".readout"), bar = document.getElementById("inside-pick"), vis = true;
    visible(el, function (v) { vis = v; });
    var pick = BYID.delay ? "delay" : (D.units[0] || {}).id;
    D.units.slice().sort(function (a, b) { return a.title.localeCompare(b.title); }).forEach(function (u) {   // the picker in alphabetical order
      var b = document.createElement("button");
      b.type = "button"; b.textContent = u.title; b.setAttribute("aria-pressed", u.id === pick ? "true" : "false");
      b.addEventListener("click", function () { pick = u.id; bar.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", x === b ? "true" : "false"); }); draw(performance.now()); });
      bar.appendChild(b);
    });
    var X0 = 70, BW = 56, GAP = 8;               // twelve blocks left to right; place l sits after block l
    function bx(i) { return X0 + i * (BW + GAP); }
    function lx(l) { return l === 0 ? X0 - GAP / 2 - 6 : bx(l - 1) + BW + GAP / 2; }
    var pimg = el.querySelector(".pickstrip");
    function draw(t) {
      var u = BYID[pick], s = "", i;
      if (pimg && u && pimg.dataset.id !== u.id) { pimg.dataset.id = u.id; pimg.src = "img/strips/" + u.id + ".webp"; }
      var at = u && u.controls.at, measured = at ? at.steps.map(function (st) { return st.v; }) : [];
      var good = (u && u.places) || [], def = u ? u.place_default : null, out = u && u.site === "output";
      s += T(X0, 24, "the temporal transformer · one frame per step · the stream is 1,024 numbers", "f-dim t-xs t-up");
      // the stream
      var sy = 160;
      s += L(30, sy, bx(11) + BW + 30, sy, "s-ink", ' stroke-width="2" opacity="0.45"');
      s += T(18, sy - 10, "in", "f-dim t-xs");
      for (i = 0; i < 12; i++) {
        var x = bx(i), inBlock = out && (def === -1 || def === i);
        var mB = out && measured.indexOf(i) >= 0, gB = out && good.indexOf(i) >= 0;
        s += R(x, 50, BW, 80, "f-panel", ' rx="5" style="stroke:' + (inBlock ? "var(--accent)" : mB ? (gB ? "var(--green)" : "var(--red)") : "var(--fig-line)") + '"');
        s += T(x + 6, 66, "block " + (i + 1), "f-dim t-xs");
        s += R(x + 6, 74, BW - 12, 20, "f-bg", ' rx="3" style="stroke:var(--fig-line)"'); s += T(x + 10, 88, "attention", "f-dim t-xs", ' style="font-size:8px"');
        s += R(x + 6, 100, BW - 12, 20, "f-bg", ' rx="3" style="stroke:var(--fig-line)"'); s += T(x + 10, 114, u && u.id === "neuron" ? "MLP · neurons" : "MLP", "f-dim t-xs", ' style="font-size:8px"');
        if (inBlock) {
          if (u.id === "neuron") s += R(x + 6, 100, BW - 12, 20, "f-acc", ' rx="3" opacity="0.35"');
          else { s += R(x + 6, 74, BW - 12, 20, "f-acc", ' rx="3" opacity="0.3"'); s += R(x + 6, 100, BW - 12, 20, "f-acc", ' rx="3" opacity="0.3"'); }
        }
        s += L(x + 18, sy, x + 18, 130, "s-dim", ' stroke-width="1"'); s += L(x + BW - 18, 130, x + BW - 18, sy, "s-dim", ' stroke-width="1" stroke-dasharray="2 2"');
      }
      // the places between blocks
      for (var l = 1; l <= 12; l++) {
        var px = lx(l), m = !out && measured.indexOf(l) >= 0, g = good.indexOf(l) >= 0, isDef = !out && def === l;
        s += C(px, sy, isDef ? 9 : 6, m ? (g ? "f-green" : "f-red") : "f-panel", ' style="stroke:' + (isDef ? "var(--accent)" : "var(--fig-line)") + '" stroke-width="' + (isDef ? 3 : 1) + '"' + (m ? ' opacity="0.85"' : ""));
        s += T(px - (l > 9 ? 6 : 3), sy + 26, String(l), "f-dim t-xs");
      }
      s += T(X0, sy + 44, "layer: the stream after that block", "f-dim t-xs t-up");
      // the Steer, fixed at layer 6
      s += L(lx(6), sy + 50, lx(6), sy + 10, "s-acc", ' stroke-dasharray="2 3"'); s += T(lx(6) - 14, sy + 62, "Steer", "f-acc t-xs");
      // a frame's tokens travelling through
      var k = reduced ? 0.4 : (t % 2600) / 2600, fx = 30 + k * (bx(11) + BW + 10);
      var bent = out ? fx > bx(Math.max(0, def === -1 ? 0 : def)) : fx > lx(def || 12);
      s += R(fx - 5, sy - 5, 10, 10, bent ? "f-acc" : "f-blue", ' rx="2" style="filter:drop-shadow(0 0 3px var(--accent))"');
      // after the stack
      var ax = bx(11) + BW + 20;
      s += T(ax - 40, sy - 10, "out →", "f-dim t-xs"); s += T(X0 + 430, sy + 44, "then: depth transformer → codec → 1,920 stereo samples", "f-dim t-xs");
      // the legend
      var ly = 270;
      s += C(X0, ly, 6, "f-green", ' opacity="0.85"'); s += T(X0 + 12, ly + 4, "measured: the sound held", "f-ink t-s");
      s += C(X0 + 230, ly, 6, "f-red", ' opacity="0.85"'); s += T(X0 + 242, ly + 4, "measured: it broke on a preset", "f-ink t-s");
      s += C(X0 + 500, ly, 8, "f-panel", ' style="stroke:var(--accent)" stroke-width="3"'); s += T(X0 + 514, ly + 4, "default placement here", "f-ink t-s");
      if (u) {
        s += T(X0, 318, u.title + (otherName(u) ? "  (VAU: " + u.vau_title + ")" : ""), "f-ink t-l");
        s += T(X0, 340, u.what, "f-dim t-s");
        s += T(X0, 362, (out ? "acts inside blocks: " : "acts on the stream at a layer: ") + (u.example || ""), "f-dim t-xs");
        if (u.id === "delay" || u.id === "reverb" || u.id === "beatrepeat") s += T(X0, 384, "also along time: it reads the stream of earlier frames at the same layer", "f-acc t-xs");
      }
      svg.innerHTML = s;
      var msg = !u ? "" : out
        ? u.title + " acts inside the blocks; its default here is " + (def === -1 ? "every block" : "block " + (def + 1)) + ". Measured placements: " + measured.map(function (v) { return (v === -1 ? "all" : "block " + (v + 1)) + (good.indexOf(v) >= 0 ? " held" : " broke"); }).join(", ") + "."
        : u.title + " at layer " + def + " by default. Measured at layers " + measured.join(", ") + "; it held at " + (good.length ? good.join(", ") : "none") + ".";
      if (ro.textContent !== msg) ro.textContent = msg;
    }
    FRAME.push(function (now) { if (vis) draw(now); });
    draw(0);
  })();

  /* ---------- 3. the units, by group, each with a drawing and its sweep ---------- */
  var GROUPS = [
    ["time", "Along time", "These act along the frames rather than within one: each frame's stream at their layer gets something of the frames before it. The space is the stream; the axis is time."],
    ["size", "On the stream's size", "Acts on the length of the stream's vector and leaves its direction alone, so it does not depend on the coordinates the stream is written in."],
    ["element", "Coordinate by coordinate", "Act on each of the 1,024 numbers separately: squash, round, zero or clamp them, so what they do depends on the coordinates the stream is written in. Those coordinates have a weaker claim to meaning one by one than an MLP's neurons, which the nonlinearity applied to each singles out; in Magenta RT 2 each block's RMSNorm, with a learned scale for every coordinate, gives them some. In the measurements these units broke the sound no more often than the others."],
    ["subspace", "Along directions and principal bands", "Act along chosen directions of the stream: two Steer directions (Rotate), or bands of its principal directions, measured over reference prompts on the unchanged model (Principal Bands, Band Shrink)."],
    ["block", "Inside a block", "Act inside the blocks rather than between them: on what attention and the MLP write (Abliterate), or on the MLP's hidden units, its neurons (Neuron)."],
  ];
  var gStrip = document.getElementById("group-strip"), host = document.getElementById("unit-groups"), cards = [];
  GROUPS.forEach(function (g) {
    var us = D.units.filter(function (u) { return u.group === g[0]; })
      .sort(function (a, b) { return a.title.localeCompare(b.title); });      // alphabetical within each group
    if (!us.length || !host) return;
    if (gStrip) { var a = document.createElement("a"); a.href = "#g-" + g[0]; a.textContent = g[1]; gStrip.appendChild(a); }
    var sec = document.createElement("div");
    sec.className = "ugroup"; sec.id = "g-" + g[0];
    sec.innerHTML = "<h3>" + esc(g[1]) + "</h3><p>" + esc(g[2]) + "</p>";
    us.forEach(function (u) { sec.appendChild(card(u)); });
    host.appendChild(sec);
  });

  function fmt(v) { return Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1).replace(/\.0$/, "") : String(+v.toFixed(3)); }
  function card(u) {
    var el = document.createElement("article");
    el.className = "ucard2"; el.id = "u-" + u.id;
    var site = u.site === "output" ? "inside a block" : "the stream at a layer";
    var rows = u.params.filter(function (q) { return q.kind === "knob" || q.kind === "fader" || q.kind === "int"; }).map(function (q) {
      var c = u.controls[q.key] || {}, safe = q.safe || c.safe;
      return "<tr><td>" + esc(q.label) + (q.fragile ? " ⚠" : "") + "</td><td>" + fmt(q.min) + " to " + fmt(q.max) + "</td><td>" + (safe ? fmt(safe[0]) + (safe[0] === safe[1] ? " only" : " to " + fmt(safe[1])) : "–") + "</td><td>" + (q.default == null ? "–" : fmt(+q.default)) + (c.moved_default ? ' <span class="moved" title="VAU\'s default broke the sound here">(VAU ' + fmt(c.vau_default) + ")</span>" : "") + '</td><td class="tip"><code>' + esc(q.tip || "") + "</code></td></tr>";
    }).join("");
    el.innerHTML = '<header><h4>' + esc(u.title) + "</h4>" + (otherName(u) ? '<span class="vaun">VAU: ' + esc(u.vau_title) + "</span>" : "") + '<span class="usite">' + site + "</span></header>" +
      '<div class="ubody"><a class="ustrip shot" href="img/strips/' + u.id + '.webp" data-cap="' + esc(u.title + " as a card on the Model layers row, at the defaults measured on this engine" + (otherName(u) ? " (VAU: " + u.vau_title + ")" : "")) + '"><img src="img/strips/' + u.id + '.webp" width="132" loading="lazy" alt="' + esc("The " + u.title + " strip: its power light, its place, and its knobs at their defaults") + '"></a>' +
      '<div class="umini"><svg viewBox="0 0 220 120" aria-hidden="true"></svg><span class="cap">what it does</span></div><div class="utext"><p>' + esc(u.what) + "</p>" +
      '<div class="table-wrap"><table><thead><tr><th>Knob</th><th>Range</th><th>Measured</th><th>Default</th><th>What it does</th></tr></thead><tbody>' + rows + "</tbody></table></div></div></div>" +
      '<div class="usweeps"></div>';
    var sw = el.querySelector(".usweeps");
    if (D.none) {
      var nb = document.createElement("button");
      nb.type = "button"; nb.className = "unone"; nb.textContent = "▶ without the unit";
      nb.title = "Warm drone, seed 1, no unit: " + D.none.rms_db.toFixed(1) + " dBFS";
      nb.addEventListener("click", function () { play(D.none.src, "Warm drone · seed 1 · no unit", nb, false); });
      owners.push(nb); sw.appendChild(nb);
    }
    Object.keys(u.controls).sort(function (a, b) { return (a === "at") - (b === "at"); }).forEach(function (key) {
      var c = u.controls[key], q = u.params.filter(function (x) { return x.key === key; })[0];
      var row = document.createElement("div"); row.className = "urow";
      var name = key === "at" ? (u.site === "output" ? "Block" : "Layer") : q ? q.label : key;
      row.innerHTML = '<span class="ulab">' + esc(name) + "</span>";
      var bars = document.createElement("div"); bars.className = "ubars";
      c.steps.forEach(function (st) {
        var b = document.createElement("button");
        b.type = "button";
        var lv = clamp((st.rms_db + 75) / 70, 0.02, 1), dead = st.silence > 0.5, br = st.broken && st.broken.length;
        var isDef = c.default != null && Math.abs(st.v - c.default) < 1e-6;
        var lab = key === "at" ? (st.v === -1 ? "all" : u.site === "output" ? String(st.v + 1) : String(st.v)) : fmt(st.v);
        var noneLv = D.none ? clamp((D.none.rms_db + 75) / 70, 0, 1) : null;
        b.className = (isDef ? "def " : "") + (br || dead ? "bad" : "");
        b.innerHTML = '<span class="lv">' + (noneLv != null ? '<s style="bottom:' + (noneLv * 100).toFixed(0) + '%"></s>' : "") + '<i style="height:' + (lv * 100).toFixed(0) + '%"></i></span><span>' + esc(lab) + "</span>";
        b.title = st.line + "\n" + st.rms_db.toFixed(1) + " dBFS, " + Math.round(st.silence * 100) + "% of frames silent" + (br ? "\nbroken: " + st.broken.join("; ") : "") + (isDef ? "\nthe default here" : "") + "\nClick to listen.";
        b.setAttribute("aria-label", u.title + " " + name + " " + lab + ", " + st.rms_db.toFixed(0).replace("-", "−") + " dBFS" + (br ? ", broken" : "") + ". Play");
        b.addEventListener("click", function () { play(st.src, u.title + " · " + name.toLowerCase() + " " + lab + " · " + st.rms_db.toFixed(1) + " dBFS", b, false); });
        owners.push(b); bars.appendChild(b);
      });
      row.appendChild(bars); sw.appendChild(row);
    });
    var cd = { el: el, u: u, svg: el.querySelector(".umini svg"), vis: false };
    visible(el, function (v) { cd.vis = v; });
    cards.push(cd);
    return el;
  }

  // small drawings of what each unit does: 220 × 120
  var rnd = function (i) { var r = Math.sin(i * 12.9898) * 43758.5453; return r - Math.floor(r); };
  var MINI = {
    delay: function (t) {
      var s = "", p = 4 + Math.floor(t / 600) % 8, i;
      for (i = 0; i < 12; i++) s += R(8 + i * 17, 70, 14, 22, i === p ? "f-acc" : "f-panel", ' rx="2" style="stroke:var(--fig-line)"');
      [[p - 4, 1], [p - 8, 0.4]].forEach(function (a) { if (a[0] >= 0) { var x0 = 15 + a[0] * 17, x1 = 15 + p * 17; s += P("M" + x0 + " 68 Q" + (x0 + x1) / 2 + " " + (68 - 40 * a[1]) + " " + x1 + " 68", "s-acc", ' stroke-width="' + (1 + a[1]) + '" opacity="' + a[1] + '"'); } });
      return s + T(8, 112, "frames → · time 4, feedback 0.3", "f-dim t-xs");
    },
    reverb: function (t) {
      var s = "", p = 11, i, ph = (t / 900) % 1;
      for (i = 0; i < 12; i++) { var w = Math.pow(0.8, p - 1 - i); s += R(8 + i * 17, 92 - (i < p ? 60 * w : 0), 14, i < p ? 60 * w : 0, "f-blue", ' opacity="' + (0.3 + 0.5 * w) + '"'); }
      s += R(8 + p * 17, 32, 14, 60, "f-acc", ' opacity="' + (0.5 + 0.4 * Math.sin(ph * 6.28)) + '"');
      return s + T(8, 112, "a decaying average of earlier frames", "f-dim t-xs");
    },
    beatrepeat: function (t) {
      var s = "", p = Math.floor(t / 500) % 12, i;
      for (i = 0; i < 12; i++) { var rep = i % 8 >= 6; s += R(8 + i * 17, 70, 14, 22, i === p ? "f-acc" : "f-panel", ' rx="2" style="stroke:' + (rep ? "var(--blue)" : "var(--fig-line)") + '"'); }
      if (p % 8 >= 6) { var x0 = 15 + (p - 2) * 17, x1 = 15 + p * 17; s += P("M" + x0 + " 68 Q" + (x0 + x1) / 2 + " 40 " + x1 + " 68", "s-blue", ' stroke-width="2"'); }
      return s + T(8, 112, "the last 2 of each 8 replay the 2 before", "f-dim t-xs");
    },
    compressor: function (t) {
      var s = "", cx = 70, cy = 58, th = 30, a = t / 1400, n = 20 + 26 * (0.5 + 0.5 * Math.sin(t / 700));
      var o = n > th ? th + (n - th) / 4 : n;
      s += C(cx, cy, th, "f-panel", ' style="stroke:var(--fig-dim)" stroke-dasharray="3 3"');
      s += arrow(cx, cy, cx + n * Math.cos(a), cy + n * Math.sin(a), "s-blue", 1.5);
      s += arrow(cx, cy, cx + o * Math.cos(a), cy + o * Math.sin(a), "s-acc", 2.5);
      s += T(130, 50, "‖h‖ " + (n / th).toFixed(2), "f-blue t-xs"); s += T(130, 66, "→ " + (o / th).toFixed(2), "f-acc t-xs");
      return s + T(8, 112, "above the threshold, the length ÷ ratio", "f-dim t-xs");
    },
    saturator: function (t) {
      var s = "", x0 = 20, y0 = 58, W = 120, i, d = "";
      s += L(x0, y0, x0 + W, y0, "s-line"); s += L(x0 + W / 2, 10, x0 + W / 2, 106, "s-line");
      for (i = 0; i <= 40; i++) { var u = (i / 40) * 2 - 1, y = Math.tanh(u * 2.5) / Math.tanh(2.5); d += (i ? "L" : "M") + (x0 + (u + 1) * W / 2).toFixed(1) + " " + (y0 - y * 40).toFixed(1); }
      s += P(d, "s-acc", ' stroke-width="2"');
      var v = Math.sin(t / 500), yv = Math.tanh(v * 2.5) / Math.tanh(2.5);
      s += C(x0 + (v + 1) * W / 2, y0 - yv * 40, 4, "f-acc");
      s += T(150, 40, "in " + v.toFixed(2), "f-blue t-xs"); s += T(150, 56, "out " + yv.toFixed(2), "f-acc t-xs");
      return s + T(8, 112, "each coordinate through tanh", "f-dim t-xs");
    },
    erosion: function (t) {
      var s = "", d = "", q = "", i, lv = 6, ph = t / 900;
      for (i = 0; i <= 100; i++) { var x = 8 + i * 2, y = Math.sin(i / 12 + ph) * 0.8 + 0.2 * Math.sin(i / 3.1 + ph * 2); d += (i ? "L" : "M") + x + " " + (58 - y * 40).toFixed(1); q += (i ? "L" : "M") + x + " " + (58 - Math.round(y * lv / 2) / (lv / 2) * 40).toFixed(1); }
      s += P(d, "s-blue", ' stroke-width="1" opacity="0.6"'); s += P(q, "s-acc", ' stroke-width="2"');
      return s + T(8, 112, "each coordinate rounded to few levels", "f-dim t-xs");
    },
    gate: function (t) {
      var s = "", i, hs = [];
      for (i = 0; i < 16; i++) hs.push(Math.abs(Math.sin(i * 1.7 + t / 800) * 0.7 + 0.3 * Math.sin(i * 0.6 - t / 500)));
      var cut = hs.slice().sort(function (a, b) { return b - a; })[3];
      hs.forEach(function (h, j) { var keep = h >= cut; s += R(10 + j * 12.5, 92 - h * 70, 9, h * 70, keep ? "f-acc" : "f-dim", ' opacity="' + (keep ? 0.9 : 0.15) + '"'); });
      return s + T(8, 112, "keeps the largest quarter, zeroes the rest", "f-dim t-xs");
    },
    limiter: function (t) {
      var s = "", i, ceil = 0.55;
      for (i = 0; i < 16; i++) { var h = Math.abs(Math.sin(i * 2.3 + t / 700)) * (0.4 + 0.6 * rnd(i)), o = Math.min(h, ceil); s += R(10 + i * 12.5, 92 - h * 70, 9, (h - o) * 70, "f-red", ' opacity="0.35"'); s += R(10 + i * 12.5, 92 - o * 70, 9, o * 70, "f-acc"); }
      s += L(6, 92 - ceil * 70, 214, 92 - ceil * 70, "s-ink", ' stroke-dasharray="3 3"');
      return s + T(8, 112, "the largest components clamped", "f-dim t-xs");
    },
    eq: function (t) {
      var s = "", i, g = [1.3, 1, 0.7], cls = ["f-acc", "f-blue", "f-green"];
      for (i = 0; i < 18; i++) { var b = i < 4 ? 0 : i < 10 ? 1 : 2, base = Math.pow(i + 1, -0.5) * (1 + 0.08 * Math.sin(t / 300 + i)); s += R(10 + i * 11, 92 - base * 70, 8, base * 70, "", ' fill="none" style="stroke:var(--fig-dim)"'); s += R(10 + i * 11, 92 - base * 70 * g[b], 8, base * 70 * g[b], cls[b], ' opacity="0.8"'); }
      return s + T(8, 112, "low ×1.3 · mid ×1 · high ×0.7, principal bands", "f-dim t-xs");
    },
    multiband: function (t) {
      var s = "", i, th = 0.45;
      for (i = 0; i < 18; i++) { var h = Math.pow(i + 1, -0.35) * (0.7 + 0.3 * Math.sin(t / 400 + i * 0.9)), o = h > th ? th + (h - th) / 4 : h; s += R(10 + i * 11, 92 - h * 70, 8, h * 70, "", ' fill="none" style="stroke:var(--fig-dim)"'); s += R(10 + i * 11, 92 - o * 70, 8, o * 70, i < 4 ? "f-acc" : i < 10 ? "f-blue" : "f-green", ' opacity="0.8"'); }
      s += L(6, 92 - th * 70, 214, 92 - th * 70, "s-ink", ' stroke-dasharray="3 3"');
      return s + T(8, 112, "each band compressed on its own", "f-dim t-xs");
    },
    shifter: function (t) {
      var s = "", cx = 70, cy = 58, a = t / 1600, r = 42, rot = Math.PI / 4;
      s += L(cx - 50, cy, cx + 50, cy, "s-acc", ' stroke-dasharray="3 3"'); s += L(cx, cy - 50, cx, cy + 50, "s-blue", ' stroke-dasharray="3 3"');
      s += T(cx + 40, cy - 4, "bright", "f-acc t-xs"); s += T(cx + 4, cy - 44, "pulse", "f-blue t-xs");
      s += arrow(cx, cy, cx + r * Math.cos(a), cy - r * Math.sin(a), "s-dim", 1.5);
      s += arrow(cx, cy, cx + r * Math.cos(a + rot), cy - r * Math.sin(a + rot), "s-acc", 2.5);
      s += P("M" + (cx + 22 * Math.cos(a)) + " " + (cy - 22 * Math.sin(a)) + " A22 22 0 0 0 " + (cx + 22 * Math.cos(a + rot)) + " " + (cy - 22 * Math.sin(a + rot)), "s-ink", ' opacity="0.6"');
      s += T(140, 58, "45° in their plane", "f-dim t-xs");
      return s + T(8, 112, "rotates the stream in two directions' plane", "f-dim t-xs");
    },
    abliterate: function (t) {
      var s = "", cx = 60, cy = 80, a = -0.9 + 0.35 * Math.sin(t / 1100), r = 60, d = -0.25;
      var hx = r * Math.cos(a), hy = r * Math.sin(a), c = hx * Math.cos(d) + hy * Math.sin(d), px = c * Math.cos(d), py = c * Math.sin(d);
      s += L(cx - 40 * Math.cos(d), cy - 40 * Math.sin(d), cx + 140 * Math.cos(d), cy + 140 * Math.sin(d), "s-acc", ' stroke-dasharray="4 3"');
      s += arrow(cx, cy, cx + hx, cy + hy, "s-dim", 1.5);
      s += arrow(cx, cy, cx + hx - px, cy + hy - py, "s-acc", 2.5);
      s += L(cx + hx, cy + hy, cx + hx - px, cy + hy - py, "s-dim", ' stroke-dasharray="2 2"');
      s += T(150, 28, "attention's and", "f-dim t-xs"); s += T(150, 40, "the MLP's output", "f-dim t-xs"); s += T(150, 52, "less its part", "f-dim t-xs"); s += T(150, 64, "along brightness", "f-acc t-xs");
      return s + T(8, 112, "removes a direction from what blocks write", "f-dim t-xs");
    },
    neuron: function (t) {
      var s = "", i;
      for (i = 0; i < 14; i++) { var a = 0.3 + 0.5 * Math.abs(Math.sin(i * 1.3 + t / 400)), hot = i % 3 === 1; s += C(16 + i * 14.5, 50, hot ? 6 : 4, hot ? "f-acc" : "f-blue", ' opacity="' + (hot ? 0.95 : a) + '"' + (hot ? ' style="filter:drop-shadow(0 0 5px var(--accent))"' : "")); }
      s += T(76, 30, "32 neurons set to 8", "f-acc t-xs");
      s += T(8, 82, "those block 10's MLP writes", "f-dim t-xs"); s += T(8, 94, "most along brightness", "f-dim t-xs");
      return s + T(8, 112, "edits named MLP neurons", "f-dim t-xs");
    },
  };
  var lastMini = 0;
  FRAME.push(function (now) {                         // the small drawings move at 15 frames a second
    if (now - lastMini < 66) return; lastMini = now;
    cards.forEach(function (c) { if (c.vis && MINI[c.u.id]) c.svg.innerHTML = MINI[c.u.id](reduced ? 2000 : now); });
  });
  cards.forEach(function (c) { if (MINI[c.u.id]) c.svg.innerHTML = MINI[c.u.id](2000); });

  // the play buttons light while their clip plays
  var tries = 0;
  (function hook() { if (window.AA) { window.AA.onChange(paintOwners); return; } if (tries++ < 50) setTimeout(hook, 50); })();
})();
