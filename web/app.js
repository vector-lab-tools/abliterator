// Abliterator. Copyright (C) 2026 David M. Berry. SPDX-License-Identifier: GPL-3.0-or-later
//
// The page: a control surface only. The sound is played by the backend through the sound card;
// this page sends moves and reads the engine's and the desk's state a few times a second.
// Two views: Synth (the model as a synth panel: engine, steer, output, strips, modulation; saving sounds) and Session (the saved sounds on eight
// tracks, triggered and layered). The browser on the left holds the saved sounds for both.
// Laid out as Abliterton Lens's page; the knobs and faders follow Lens's desk.js (after Video Kintsugi).
"use strict";
const $ = (s) => document.querySelector(s);
// a fault on the page is reported, not hidden: a short notice with its message, and the last twenty kept in this
// browser (localStorage "aa-errors") so they can be read after the fact
const PAGE_ERR = { last: 0 };
function pageError(msg) {
  try { const a = JSON.parse(localStorage.getItem("aa-errors") || "[]"); a.push({ t: new Date().toISOString(), msg: String(msg).slice(0, 500) }); localStorage.setItem("aa-errors", JSON.stringify(a.slice(-20))); } catch (e) { /* storage blocked */ }
  if (Date.now() - PAGE_ERR.last > 10000 && typeof info === "function") { PAGE_ERR.last = Date.now(); try { info(`The page hit an error: ${String(msg).slice(0, 140)}`, "bad", 8000); } catch (e) { /* the notice itself failed */ } }
}
window.addEventListener("error", (e) => pageError(`${e.message} (${(e.filename || "").split("/").pop()}:${e.lineno})`));
window.addEventListener("unhandledrejection", (e) => pageError(e.reason?.stack || e.reason));
const STILL = (() => { try { return matchMedia("(prefers-reduced-motion: reduce)").matches; } catch (e) { return false; } })();   // the system asks for less motion
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const h = (html) => { const t = document.createElement("template"); t.innerHTML = html.trim(); return t.content.firstElementChild; };

async function api(path, body) {
  const opt = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json", "X-Abliterator": "1" }, body: JSON.stringify(body) };
  const r = await fetch(path, opt);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || r.statusText);
  return j;
}

// Sending while a control moves: the first change goes at once, then at most one every `ms`
// while it keeps moving, and the last one always goes.
function throttle(fn, ms) {
  let last = 0, timer = null;
  return () => {
    const wait = ms - (performance.now() - last);
    if (wait <= 0) { clearTimeout(timer); timer = null; last = performance.now(); fn(); }
    else if (!timer) timer = setTimeout(() => { timer = null; last = performance.now(); fn(); }, wait);
  };
}

// ------------------------------------------------------------- the info bar
let infoUntil = 0;
function info(msg, kind = "", ms = 4000) { $("#infotext").textContent = msg; $("#infotext").className = kind; infoUntil = Date.now() + ms; }
const fail = (e) => info(e.message || String(e), "bad");

// ------------------------------------------------------------- theme
const THEMES = ["system", "light", "dark"];
function setTheme(t) {
  if (t === "system") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = t;
  $("#themebtn").title = `Theme: ${t} (click to change)`;
  try { localStorage.setItem("aa-theme", t); } catch (e) { /* private window */ }
}
let theme = "system";
try { theme = localStorage.getItem("aa-theme") || "system"; } catch (e) { /* private window */ }
setTheme(theme);
$("#themebtn").addEventListener("click", () => { theme = THEMES[(THEMES.indexOf(theme) + 1) % 3]; setTheme(theme); drawXY(); });

// ------------------------------------------------------------- controls: knob and fader
// p: {label, min, max, step, int, log, reset, get(), put(v), fmt(v), mod}. Each returns an element
// with .paint(), which redraws it from p.get() unless it is being moved. A knob with `mod` (a
// modulation target) draws a ring for the range its routes move it through and a dot for where
// it is now; while a source is being assigned, dragging it sets that route's depth instead.
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const rnd = (v, p) => (p.int ? Math.round(v) : p.step ? +(Math.round(v / p.step) * p.step).toFixed(4) : +v.toFixed(4));
let holding = 0;                                 // a control is being moved: polls do not repaint it
const MODKNOBS = new Set();
// the calibration (calibrate.py): each control's musical range, laid out so equal travel gives equal change
const CAL = { controls: {} };
function interp(x, xs, ys) {
  if (x <= xs[0]) return ys[0];
  if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
  for (let i = 1; i < xs.length; i++) if (x <= xs[i]) { const a = xs[i] > xs[i - 1] ? (x - xs[i - 1]) / (xs[i] - xs[i - 1]) : 0; return ys[i - 1] + a * (ys[i] - ys[i - 1]); }
  return ys[ys.length - 1];
}
function curveOf(tgt) {
  const c = CAL.controls[tgt]?.curve;
  return c && c.length > 1 && !(tgt.startsWith("dose:") && $("#full")?.checked) ? c : null;
}
function curveScale(c) {
  const ts = c.map((q) => q[0]), vs = c.map((q) => q[1]);
  return { pos: (v) => interp(v, vs, ts), val: (t) => interp(t, ts, vs), lo: vs[0], hi: vs[vs.length - 1] };
}
function scaleOf(p) {
  if (p.taper) return curveScale(p.taper);
  const c = p.curve ? curveOf(p.curve) : null;
  if (c) return curveScale(c);
  return p.log ? { pos: (v) => Math.log(Math.max(v, p.min) / p.min) / Math.log(p.max / p.min), val: (t) => p.min * Math.pow(p.max / p.min, t) }
    : { pos: (v) => (v - p.min) / (p.max - p.min), val: (t) => p.min + t * (p.max - p.min) };
}
const R = 19;
function arcPath(t0, t1, r = R) {
  const a0 = -135 + 270 * Math.min(t0, t1), a1 = -135 + 270 * Math.max(t0, t1);
  const pt = (d) => [24 + r * Math.sin(d * Math.PI / 180), 24 - r * Math.cos(d * Math.PI / 180)];
  const [x0, y0] = pt(a0), [x1, y1] = pt(a1 + (a1 - a0 < 0.5 ? 0.5 : 0));
  return `M${x0.toFixed(2)} ${y0.toFixed(2)} A${r} ${r} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}
function knob(p) {
  const pos = (v) => scaleOf(p).pos(v), val = (t) => scaleOf(p).val(t);
  const show = (v) => (p.fmt ? p.fmt(v) : String(rnd(v, p)));
  const el = h(`<div class="knob" tabindex="0" role="slider" aria-label="${esc(p.label)}"><svg viewBox="0 0 48 48" aria-hidden="true">
    <circle cx="24" cy="24" r="15" class="kbody"/><path class="ktrack"/><path class="kmod"/><path class="karc"/><line x1="24" y1="24" x2="24" y2="12" class="kline"/><circle class="kdot" r="2.3" cx="24" cy="5"/></svg><div class="kv"></div><div class="kl">${esc(p.label)}</div></div>`);
  el.title = `${p.label}${p.tip ? ": " + p.tip : ""}. Drag up or right (Shift for fine), scroll, or use the arrow keys; double-click resets.${p.mod ? " Modulation can move it: drop a source here, or choose one and drag here to set how far." : ""}`;
  if (p.mod) el.dataset.mod = p.mod;
  el.querySelector(".ktrack").setAttribute("d", arcPath(0, 1));
  let acc = null, mine = false;
  const draw = (v) => {
    const t = clamp01(pos(v));
    el.querySelector(".karc").setAttribute("d", arcPath(0, Math.max(t, 0.001)));
    el.querySelector(".kline").setAttribute("transform", `rotate(${-135 + 270 * t} 24 24)`);
    el.querySelector(".kv").textContent = show(v);
    el.setAttribute("aria-valuetext", show(v));
  };
  const put = (v) => { p.put(v); draw(v); el.paintMod(); };
  const nudge = (dt) => { if (acc === null) acc = clamp01(pos(p.get())); acc = clamp01(acc + dt); const v = rnd(val(acc), p); if (v !== p.get()) put(v); };
  const assigning = () => p.mod && MOD.assign && MOD.on;
  el.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    e.preventDefault(); el.focus({ preventScroll: true, focusVisible: false }); el.setPointerCapture(e.pointerId);
    mine = true; holding++;
    let x = e.clientX, y = e.clientY;
    const depthMode = assigning();
    let base = depthMode ? routeDepth(MOD.assign, p.mod) : 0, yb = e.clientY, fine = e.shiftKey;
    acc = clamp01(pos(p.get()));
    const move = (ev) => {
      if (depthMode) {                         // the send: up raises it, down lowers it, 150 px from 0 to full (Shift: 600)
        if (ev.shiftKey !== fine) { base = routeDepth(MOD.assign, p.mod); yb = ev.clientY; fine = ev.shiftKey; }
        const dep = Math.max(-1, Math.min(1, base + (yb - ev.clientY) / (fine ? 600 : 150)));
        setRoute(MOD.assign, p.mod, Math.abs(dep) < 0.005 ? 0 : +dep.toFixed(3), false);
        el.paintMod();
        info(`${SRC[MOD.assign].label} → ${p.label}: send ${dep >= 0 ? "+" : ""}${dep.toFixed(2)}`, "", 1500);
        return;
      }
      const d = ((y - ev.clientY) + (ev.clientX - x)) / (ev.shiftKey ? 800 : 160);
      x = ev.clientX; y = ev.clientY;
      nudge(d);
    };
    const x0 = e.clientX, y0 = e.clientY;
    const up = (ev) => {
      acc = null; mine = false; holding--;
      if (depthMode) {
        const moved = Math.hypot((ev?.clientX ?? x0) - x0, (ev?.clientY ?? y0) - y0) > 3;
        if (!moved && !routeDepth(MOD.assign, p.mod)) { setRoute(MOD.assign, p.mod, START_SEND, false); el.paintMod(); }   // a click links it at the starting send
        sendMod(); drawEditor();
      }
      el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); el.removeEventListener("pointercancel", up);
    };
    el.addEventListener("pointermove", move); el.addEventListener("pointerup", up); el.addEventListener("pointercancel", up);
  });
  let idle = null, restAt = Infinity;
  el.addEventListener("pointerenter", () => { restAt = performance.now() + 300; });
  el.addEventListener("pointerleave", () => { restAt = Infinity; });
  el.addEventListener("wheel", (e) => {
    if (performance.now() < restAt && document.activeElement !== el) { restAt = performance.now() + 300; return; }   // the page is scrolling past
    e.preventDefault();
    if (assigning()) { const dd = Math.max(-1, Math.min(1, routeDepth(MOD.assign, p.mod) - e.deltaY / (e.shiftKey ? 2400 : 600))); setRoute(MOD.assign, p.mod, Math.abs(dd) < 0.005 ? 0 : +dd.toFixed(3)); el.paintMod(); clearTimeout(idle); idle = setTimeout(drawEditor, 300); return; }
    mine = true; nudge(-e.deltaY / (e.shiftKey ? 3000 : 600)); clearTimeout(idle); idle = setTimeout(() => { acc = null; mine = false; }, 500);
  }, { passive: false });
  // on the knob itself: pointerdown captures the pointer, so the clicks go to the knob, not to its drawing
  el.addEventListener("dblclick", () => {
    acc = null;
    if (assigning()) {                         // a source chosen: its link to this knob goes
      setRoute(MOD.assign, p.mod, 0); drawEditor(); el.paintMod();
      info(`${SRC[MOD.assign].label} → ${p.label}: link removed`, "ok", 2500);
    } else put(p.reset);
  });
  el.addEventListener("keydown", (e) => {
    const d = { ArrowUp: 1, ArrowRight: 1, ArrowDown: -1, ArrowLeft: -1 }[e.key];
    if (!d) return;
    e.preventDefault(); e.stopPropagation(); acc = null;
    put(p.int ? Math.min(p.max, Math.max(p.min, Math.round(p.get()) + d)) : rnd(val(clamp01(pos(p.get()) + d * (e.shiftKey ? 0.005 : 0.025))), p));
  });
  // the modulation ring and the live dot
  el.paintMod = () => {
    if (!p.mod) return;
    const routes = MOD.spec.routes.filter((r) => r.tgt === p.mod);
    el.classList.toggle("target", !!assigning());
    el.classList.toggle("linked", !!assigning() && Math.abs(routeDepth(MOD.assign, p.mod)) >= 0.001);   // the source already moves it
    if (assigning()) el.style.setProperty("--assign", SRC[MOD.assign].color);
    const ring = el.querySelector(".kmod"), dot = el.querySelector(".kdot");
    if (assigning()) {                         // the knob is the selected source's send: its arc from 12 o'clock, its readout
      const d = routeDepth(MOD.assign, p.mod), arc = el.querySelector(".karc");
      arc.setAttribute("d", Math.abs(d) < 0.001 ? "" : arcPath(0.5 + Math.min(0, d) / 2, 0.5 + Math.max(0, d) / 2));
      arc.style.stroke = SRC[MOD.assign].color;
      el.querySelector(".kline").setAttribute("transform", `rotate(${135 * d} 24 24)`);
      el.querySelector(".kv").textContent = `send ${d >= 0 ? "+" : ""}${d.toFixed(2)}`;
      ring.setAttribute("d", ""); el.classList.add("sending");
      return;
    }
    if (el.classList.contains("sending")) { el.classList.remove("sending"); el.querySelector(".karc").style.stroke = ""; if (!mine) draw(p.get()); }
    if (!routes.length) { ring.setAttribute("d", ""); el.classList.remove("modded"); return; }
    const base = p.get(), [lo, hi] = modRange(p.mod, base, routes);
    ring.setAttribute("d", arcPath(clamp01(pos(lo)), clamp01(pos(hi))));
    const r0 = routes.find((r) => r.src === MOD.assign) || routes[0];
    ring.style.stroke = SRC[r0.src].color;
    const now = effOf(p.mod, base, modTime());
    const a = (-135 + 270 * clamp01(pos(now))) * Math.PI / 180;
    dot.setAttribute("cx", (24 + R * Math.sin(a)).toFixed(2)); dot.setAttribute("cy", (24 - R * Math.cos(a)).toFixed(2));
    el.classList.add("modded");
  };
  el.paint = () => { if (!mine) draw(p.get()); el.paintMod(); };
  if (p.mod) MODKNOBS.add(el);
  el.paint();
  return el;
}
// A fader as on a console channel strip: a ladder of LEDs with a hold button (p.meter, a peak
// reading), the slot and its cap, and a scale (p.ticks, [value, label] pairs) beside it.
const LEDS = [["OL", 0, "r", true], ["0", 0, "a"], ["-3", -3, "a"], ["-6", -6, "g"], ["-12", -12, "g"], ["-18", -18, "g"], ["-27", -27, "g"], ["-36", -36, "g"]];
function fader(p) {
  const pos = (v) => scaleOf(p).pos(v), val = (t) => scaleOf(p).val(t);
  const show = (v) => (p.fmt ? p.fmt(v) : String(rnd(v, p)));
  const leds = p.meter ? `<div class="leds"><button class="hold" type="button" aria-pressed="false" title="Hold: the LEDs keep the highest reading until Hold is pressed again"><i></i></button><span class="hl">HOLD</span>${LEDS.map(([l, , c]) => `<div class="lr"><span>${l}</span><i class="${c}"></i></div>`).join("")}</div>` : "";
  const ticks = p.ticks ? `<div class="fscale" aria-hidden="true">${p.ticks.map(([v, l]) => `<span style="--t:${clamp01(pos(v)).toFixed(4)}">${esc(l)}</span>`).join("")}</div>` : "";
  const el = h(`<div class="fader${p.meter ? " metered" : ""}"><div class="fv"></div><div class="fbody">${leds}<div class="fwell"><input type="range" min="0" max="1000" step="1" aria-label="${esc(p.label)}"></div>${ticks}</div><div class="fl">${esc(p.label)}</div></div>`);
  if (p.meter) {
    let held = -Infinity, hold = false;
    const hb = el.querySelector(".hold"), lit = [...el.querySelectorAll(".lr i")];
    hb.addEventListener("click", () => { hold = !hold; held = -Infinity; hb.setAttribute("aria-pressed", hold ? "true" : "false"); });
    el.meter = () => {
      const pk = p.meter() || 0, db = pk > 0 ? 20 * Math.log10(pk) : -Infinity;
      if (hold) held = Math.max(held, db);
      const v = hold ? held : db;
      LEDS.forEach(([, thr, , over], i) => lit[i].classList.toggle("on", over ? v > thr : v >= thr));
    };
  }
  const input = el.querySelector("input");
  el.title = `${p.label}${p.tip ? ": " + p.tip : ""}. Drag the cap or use the arrow keys; double-click resets.`;
  let mine = false;
  const draw = (v) => { input.value = Math.round(clamp01(pos(v)) * 1000); el.querySelector(".fv").textContent = show(v); };
  input.addEventListener("pointerdown", () => { mine = true; holding++; });
  const release = () => { if (mine) { mine = false; holding--; } };
  input.addEventListener("pointerup", release); input.addEventListener("pointercancel", release); input.addEventListener("blur", release);
  input.addEventListener("input", () => { const v = rnd(val(input.value / 1000), p); p.put(v); el.querySelector(".fv").textContent = show(v); });
  input.addEventListener("dblclick", () => { p.put(p.reset); draw(p.reset); });
  el.paint = () => { if (!mine) draw(p.get()); };
  el.paint();
  return el;
}
function toggle(label, get, put, cls = "") {
  const el = h(`<button class="tg ${cls}" type="button"><i></i>${esc(label)}</button>`);
  el.addEventListener("click", () => { put(!get()); el.paint(); });
  el.paint = () => el.setAttribute("aria-pressed", get() ? "true" : "false");
  el.paint();
  return el;
}
function meter(label, cells = false) {
  // a peak meter: each channel's bar follows its peak, a hold line keeps the highest for 1.5 s and then
  // falls 20 dB a second, and the figure is the held peak; given no peaks, it shows the level
  const el = h(`<div class="master" title="${esc(label)}: the peak heard, left and right, with a hold line"><div class="mh">${esc(label)}</div><div class="mm"><div class="mo cl"></div><div class="mo cr"></div><div class="mhold cl"></div><div class="mhold cr"></div></div><div class="mi">–</div></div>`);
  const dbOf = (v) => (v > 0 ? 20 * Math.log10(v) : -Infinity), pos = (db) => clamp01((db + 60) / 60) * 100;
  const hold = [{ db: -Infinity, at: 0 }, { db: -Infinity, at: 0 }];
  const bar = (mo, side, db) => {
    if (cells) {    // whole cells of 4 px light, each coloured by its place on the scale (style.css .cells)
      const n = Math.max(1, Math.floor(mo.parentNode.clientHeight / 4)), f = Math.round((pos(db) / 100) * n) / n;
      mo.style.clipPath = `inset(${(1 - f) * 100}% 0 0 0)`; return;
    }
    mo.style.height = `${pos(db)}%`; mo.className = `mo ${side}` + (db > -1 ? " r" : db > -6 ? " o" : ""); };
  el.set = (rms, lr, pk) => {
    const now = performance.now(), v = pk && pk.length >= 2 ? pk : lr && lr.length >= 2 ? lr : [rms, rms];
    const [ml, mr] = el.querySelectorAll(".mo"), hs = el.querySelectorAll(".mhold");
    [ml, mr].forEach((m, i) => {
      const d = dbOf(v[i]); bar(m, i ? "cr" : "cl", d);
      const H = hold[i], fall = H.db - Math.max(0, (now - H.at - 1500) / 1000) * 20;
      if (d >= fall) { H.db = d; H.at = now; H.zone = d > -1 ? "r" : d > -6 ? "o" : ""; } else { H.db = fall; H.at = now - 1500; }
      hs[i].style.bottom = `${pos(H.db)}%`; hs[i].style.display = isFinite(H.db) && H.db > -60 ? "block" : "none";
      hs[i].classList.toggle("r", H.zone === "r");        // within 1 dB of full scale: red; from −6 to −1 dB: amber; kept while held and falling
      hs[i].classList.toggle("o", H.zone === "o");
    });
    const top = Math.max(hold[0].db, hold[1].db);
    el.querySelector(".mi").textContent = isFinite(top) && top > -60 ? top.toFixed(1) : "–";
  };
  if (cells) el.classList.add("cells");
  return el;
}
// a menu by a button, as Lens's add menu
let openMenu = null;
function menu(anchor, items, cls = "") {
  if (openMenu && openMenu._anchor === anchor) { closeMenu(); return; }   // its button again: close it
  closeMenu();
  const m = h(`<div class="addmenu ${cls}" role="menu"></div>`);
  m._anchor = anchor;
  let list = null, focus = null;                                    // the scrolled list a Find box filters
  const put = (box, it) => {
    if (it.where) return box.appendChild(h(`<div class="am-where">${esc(it.where)}</div>`));
    if (it.head) return box.appendChild(h(`<div class="am-h">${esc(it.head)}</div>`));
    if (it.sep) return box.appendChild(h(`<div class="am-sep" role="separator"></div>`));
    if (it.scroll) {                                                // a long list scrolls on its own; what follows stays in view
      list = box.appendChild(h(`<div class="am-scroll"></div>`));
      it.scroll.forEach((x) => put(list, x)); return;
    }
    if (it.filter) {                                                // a Find box over the scrolled list
      const f = box.appendChild(h(`<input class="am-find" type="search" placeholder="${esc(it.filter)}" aria-label="${esc(it.filter)}">`));
      f.addEventListener("input", () => {
        const q = f.value.trim().toLowerCase(); let head = null, any = false;
        for (const el of list?.children || []) {
          if (el.classList.contains("am-h")) { if (head) head.hidden = !any; head = el; any = false; continue; }
          const show = !q || el.textContent.toLowerCase().includes(q); el.hidden = !show; any ||= show;
        }
        if (head) head.hidden = !any;
      });
      focus = f; return;
    }
    if (it.input) {                                                 // a name and a button: Enter or the button runs it
      const row = box.appendChild(h(`<div class="am-in"><input type="text" maxlength="40" placeholder="${esc(it.input)}" aria-label="${esc(it.input)}"><button type="button">${esc(it.button || "OK")}</button></div>`));
      const inp = row.querySelector("input"), go = () => { const v = inp.value.trim(); if (v) { closeMenu(); it.run(v); } };
      row.querySelector("button").addEventListener("click", go);
      inp.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); e.stopPropagation(); });
      return;
    }
    const ck = "check" in it;                                       // an on/off item: a tick when on
    const b = h(`<button role="${ck ? "menuitemcheckbox" : "menuitem"}"${ck ? ` aria-checked="${!!it.check}"` : ""}>${ck ? `<i class="am-ck">${it.check ? "✓" : ""}</i>` : ""}<b>${esc(it.label)}</b>${it.sub ? `<span>${esc(it.sub)}</span>` : ""}</button>`);
    if (it.title) b.title = it.title;
    if (it.danger) b.classList.add("am-danger");
    b.disabled = !!it.disabled;
    if (it.del) {                                                   // a small × deletes the item, and the menu stays open
      const x = h(`<i class="am-x" role="button" aria-label="Delete ${esc(it.label)}" title="Delete">×</i>`);
      x.addEventListener("click", async (e) => {
        e.stopPropagation();
        if (await it.del() === false) return;
        const head = b.previousElementSibling?.classList.contains("am-h") && !b.nextElementSibling?.matches("button") ? b.previousElementSibling : null;
        b.remove(); head?.remove();
      });
      b.appendChild(x); b.classList.add("am-has-x");
    }
    b.addEventListener("click", () => { closeMenu(); it.run(); });
    box.appendChild(b);
  };
  for (const it of items) put(m, it);
  document.body.appendChild(m);
  if (focus) setTimeout(() => focus.focus(), 0);
  const r = anchor.getBoundingClientRect(), mh = m.offsetHeight, mw = m.offsetWidth;
  m.style.left = `${Math.max(8, Math.min(window.innerWidth - mw - 8, r.left))}px`;
  m.style.top = `${r.bottom + mh + 8 < window.innerHeight ? r.bottom + 4 : Math.max(8, r.top - mh - 4)}px`;
  openMenu = m;
  setTimeout(() => document.addEventListener("pointerdown", outside), 0);
}
// A card clicked is selected (outlined); Delete or Backspace then removes it, unless a text field has the keys.
let SELECTED = null;                                            // {key: the unit's or effect's own object, el, remove}
function select(s) { SELECTED?.el?.classList.remove("sel"); SELECTED = s; s?.el?.classList.add("sel"); }
function selectable(el, key, remove) {
  el.classList.add("selectable");
  if (SELECTED?.key === key) { SELECTED = { key, el, remove }; el.classList.add("sel"); }   // kept across a redraw
  el.addEventListener("pointerdown", (e) => { if (!e.target.closest(".spre")) select({ key, el, remove }); });
}
document.addEventListener("pointerdown", (e) => { if (SELECTED && !e.target.closest(".selectable, .addmenu")) select(null); }, true);
document.addEventListener("keydown", (e) => {
  if ((e.key === "Delete" || e.key === "Backspace") && SELECTED && !e.target.closest?.("input, textarea, select, [contenteditable]")) {
    e.preventDefault(); const s = SELECTED; select(null); s.remove();
  }
});
function removeUnit(c) { const k = U.chain.indexOf(c); if (k >= 0) { U.chain.splice(k, 1); drawStrips(); sendChain(); } }
function removeFx(sl, i) { let k = PFX.slots.indexOf(sl); if (k < 0 && PFX.slots[i]?.type === sl.type) k = i; if (k >= 0) { PFX.slots.splice(k, 1); PFX.touched = Date.now(); flushFx(); drawFx(); } }
// Presets of your own (mypresets.py): a card's settings kept under a name, on every card of that unit or effect.
const MY = { units: {}, fx: {} };
async function loadMine() { try { Object.assign(MY, await api("/api/mypresets")); } catch (e) { /* tried again when one is saved */ } }
async function delMine(kind, id, q) {
  try { const j = await api("/api/mypresets/delete", { kind, id, name: q.name }); Object.assign(MY, j.all); info(`deleted ${q.name}`, "ok"); return true; }
  catch (e) { fail(e); return false; }
}
function presetItems(builtIn, mine, apply, del) {                    // the card's presets, then yours; long lists scroll, with a Find box
  const list = [];
  if (builtIn?.length) list.push({ head: "Presets" }, ...builtIn.map((q) => ({ label: q.name, title: q.tip || "", run: () => apply(q) })));
  if (mine?.length) list.push({ head: "User Presets" }, ...mine.map((q) => ({ label: q.name, title: `Saved ${q.made || ""}`, run: () => apply(q), del: () => del(q) })));
  if (!list.length) return [];
  return [...((builtIn?.length || 0) + (mine?.length || 0) > 10 ? [{ filter: "Find a preset" }] : []), { scroll: list }];
}
function mineItems(kind, id, params, block) {
  return [{ input: "Save preset…", button: "Save", run: async (name) => {
      try { const j = await api("/api/mypresets", { kind, id, name, params: params(), block: block() }); Object.assign(MY, j.all); info(`saved as ${name}`, "ok"); } catch (e) { fail(e); }
    } },
 ];
}
function outside(e) { if (openMenu && !openMenu.contains(e.target) && !openMenu._anchor?.contains(e.target)) closeMenu(); }
function closeMenu() { if (openMenu) { openMenu.remove(); openMenu = null; document.removeEventListener("pointerdown", outside); } }
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeMenu(); select(null); if (MOD.assign) assign(null); } });

// ------------------------------------------------------------- modulation (mod.py, drawn here)
// The engine computes modulation at each frame; this copy of the sources' shapes draws them,
// and the dots on the knobs, between the engine's readings.
const SRC = {};
[["lfo1", "LFO 1", "lfo"], ["lfo2", "LFO 2", "lfo"], ["lfo3", "LFO 3", "lfo"], ["env", "ENV 1", "env"], ["env2", "ENV 2", "env"], ["env3", "ENV 3", "env"], ["rand1", "RAND 1", "rand"], ["rand2", "RAND 2", "rand"], ["steps", "STEPS", "steps"], ["follow", "FOLLOW", "follow"], ["drift", "DRIFT", "drift"], ["chaos", "CHAOS", "chaos"],
 ["m1", "M 1", "macro"], ["m2", "M 2", "macro"], ["m3", "M 3", "macro"], ["m4", "M 4", "macro"]]
  .forEach(([id, label, kind]) => (SRC[id] = { id, label, kind, color: `var(--c-${kind})`, bipolar: ["lfo", "rand", "steps", "drift", "chaos"].includes(kind) }));
const tabOf = (k) => (k === "drift" || k === "chaos" ? "wild" : k);           // Drift and Chaos share the Wild tab
const CSSCOL = { lfo: "#d9aa3f", env: "#f08a4b", rand: "#a78bfa", macro: "#4ac6e8", steps: "#5fd38d", follow: "#ef6f8f", drift: "#7fb8ff", chaos: "#ff6a4c" };
const MOD = { spec: { sources: {}, routes: [] }, assign: null, on: false, touched: 0, t0: 0, tAt: performance.now(), playing: false, tab: "lfo", folHist: [] };
const TGT = { guidance: { label: "Strength", span: 8 }, temperature: { label: "Temperature", span: 1.5 }, lowpass_hz: { label: "Low-pass", span: 4, oct: true }, volume: { label: "Volume", span: 1 }, width: { label: "Width", span: 2 }, spread: { label: "Spread", span: 1 } };
// the knobs of the Model layers units and the effect slots, as modulation targets: {label, min, max, log}
const MODTGT = {};
const tgtOf = (k) => MODTGT[k] ? { label: MODTGT[k].label, span: 1 } : (k.startsWith("dose:") ? { label: S.dirs.find((d) => d.name === k.slice(5))?.title || k.slice(5), span: 0.8 } : TGT[k]);
function hash01(seed, k) {
  let x = (Math.imul(seed, 0x9E3779B1) + Math.imul(k, 0x85EBCA77) + 0x165667B1) >>> 0;
  x ^= x >>> 15; x = Math.imul(x, 0x2C1B3C6D) >>> 0; x ^= x >>> 12; x = Math.imul(x, 0x297A2D39) >>> 0; x ^= x >>> 15;
  return (x >>> 0) / 4294967296;
}
function lfoV(p, t) {
  const ph = (((t * p.rate_hz + p.phase) % 1) + 1) % 1, s = p.symmetry;
  let v;
  if (p.shape === "sine") { const u = ph < s ? ph / s * 0.5 : 0.5 + (ph - s) / (1 - s) * 0.5; v = -Math.cos(2 * Math.PI * u); }
  else if (p.shape === "triangle") v = ph < s ? -1 + 2 * ph / s : 1 - 2 * (ph - s) / (1 - s);
  else if (p.shape === "square") v = ph < s ? 1 : -1;
  else if (p.shape === "saw") v = -1 + 2 * ph;
  else v = 1 - 2 * ph;
  if (p.smooth > 0) v = (1 - p.smooth) * v + p.smooth * -Math.cos(2 * Math.PI * ph);
  return v * (p.amount ?? 1);
}
function randV(p, t) {
  const x = t * p.rate_hz, k = Math.floor(x), seed = Math.round(p.seed);
  const a = hash01(seed, k) * 2 - 1, b = hash01(seed, k + 1) * 2 - 1, f = x - k, e = 0.5 - 0.5 * Math.cos(Math.PI * f);
  return ((1 - p.smooth) * a + p.smooth * (a + (b - a) * e)) * (p.amount ?? 1);
}
function envV(p, t, len) {                 // as mod.env: with Loop on, Sustain for Hold, Release to zero, and round again
  const { attack_s: a, decay_s: d, sustain: s, release_s: r } = p;
  if (p.loop) {
    const hold = p.hold_s ?? 4, cyc = a + d + hold + r;
    if (cyc <= 0) return s;
    t = ((t % cyc) + cyc) % cyc; len = null;
    if (t >= a + d + hold) return s * (1 - (t - a - d - hold) / r);
  }
  let v = t < a ? (a > 0 ? t / a : 1) : t < a + d ? 1 - (1 - s) * (d > 0 ? (t - a) / d : 1) : s;
  if (len != null && r > 0 && t > len - r) v *= Math.max(0, (len - t) / r);
  return v;
}
function stepsV(p, t) {                    // as mod.steps
  const n = Math.round(p.length), x = t * p.rate_hz, k = Math.floor(x), m = (j) => ((j % n) + n) % n;
  let a = +p[`s${m(k)}`]; const b = +p[`s${m(k + 1)}`], g = p.glide, f = x - k;
  if (g > 0 && f > 1 - g) { const e = (f - (1 - g)) / g; a += (b - a) * (0.5 - 0.5 * Math.cos(Math.PI * e)); }
  return a * (p.amount ?? 1);
}
function vnoise(seed, x) { const k = Math.floor(x), f = x - k, a = hash01(seed, k) * 2 - 1, b = hash01(seed, k + 1) * 2 - 1; return a + (b - a) * (f * f * (3 - 2 * f)); }
function driftV(p, t) {                    // as mod.drift
  const seed = Math.round(p.seed), x = t * p.rate_hz, r = p.rough;
  let tot = 0, ws = 0;
  for (let o = 0; o < 4; o++) { const w = r ** o; tot += w * vnoise(seed + 7919 * o, x * 2 ** o); ws += w; }
  return Math.max(-1, Math.min(1, (tot / ws) * 1.4)) * (p.amount ?? 1);
}
const CHT = new Map();                      // as mod._chaos_track, for drawing: the attractor integrated once a seed
function chaosTrack(system, seed, upto) {
  const key = `${system}:${seed}`;
  let T = CHT.get(key);
  const f = system === "lorenz" ? (x, y, z) => [10 * (y - x), x * (28 - z) - y, x * y - (8 / 3) * z] : (x, y, z) => [-y - z, x + 0.2 * y, 0.2 + z * (x - 5.7)];
  const scale = system === "lorenz" ? 20 : 11, h = 0.005;
  if (!T) { T = { st: [1 + 0.001 * seed, 1, 1 + 0.0007 * (seed % 97)], v: [], need: 2000 }; CHT.set(key, T); }
  while (T.v.length < upto + 1 || T.need) {
    for (let r = 0; r < 2; r++) {
      const [x, y, z] = T.st, k1 = f(x, y, z), k2 = f(x + h / 2 * k1[0], y + h / 2 * k1[1], z + h / 2 * k1[2]),
        k3 = f(x + h / 2 * k2[0], y + h / 2 * k2[1], z + h / 2 * k2[2]), k4 = f(x + h * k3[0], y + h * k3[1], z + h * k3[2]);
      T.st = [0, 1, 2].map((j) => T.st[j] + h / 6 * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]));
    }
    if (T.need) { T.need--; continue; }
    T.v.push(Math.max(-1, Math.min(1, T.st[0] / scale)));
  }
  return T.v;
}
function chaosV(p, t) { const x = Math.max(0, t) * p.rate_hz / 0.01, i = Math.floor(x), v = chaosTrack(p.system, Math.round(p.seed), i + 1); return (v[i] + (v[i + 1] - v[i]) * (x - i)) * (p.amount ?? 1); }
function followV(t) {                       // the engine's Follower readings (status), the last at or before t
  const H = MOD.folHist; let v = 0;
  for (let i = H.length - 1; i >= 0; i--) if (H[i][0] <= t) { v = H[i][1]; break; }
  return v;
}
function srcV(id, t, len = null) {
  const p = MOD.spec.sources[id];
  if (!p) return 0;
  const k = SRC[id].kind;
  return k === "lfo" ? lfoV(p, t) : k === "rand" ? randV(p, t) : k === "env" ? envV(p, Math.max(0, t), len) : k === "steps" ? stepsV(p, t)
    : k === "drift" ? driftV(p, t) : k === "chaos" ? chaosV(p, t) : k === "follow" ? followV(t) : +p.value;
}
function modTime() { return MOD.playing ? MOD.t0 + (performance.now() - MOD.tAt) / 1000 : performance.now() / 1000; }
const START_SEND = 0.5;                     // a new link's send, however it is made: drag the knob for another
const routeDepth = (src, tgt) => MOD.spec.routes.find((r) => r.src === src && r.tgt === tgt)?.depth || 0;
function setRoute(src, tgt, depth, send = true) {
  const had = MOD.spec.routes.find((r) => r.src === src && r.tgt === tgt);
  if (had && depth) had.depth = depth;                                       // changed in place: the routes keep their order
  else {
    MOD.spec.routes = MOD.spec.routes.filter((r) => r !== had);
    if (depth) MOD.spec.routes.push({ src, tgt, depth });
  }
  MOD.touched = Date.now();
  if (send) sendMod();
  drawTiles();
}
// a target with a curve is moved along the knob's travel (as mod.py does), else by value
function onCurve(tgt, base) {
  const q = MODTGT[tgt];
  if (q) {                                             // a unit or effect knob: moved along its travel, as mod.py moves it
    const lg = q.log && q.min > 0;
    return { pos: (v) => (lg ? Math.log(Math.max(v, q.min) / q.min) / Math.log(q.max / q.min) : (v - q.min) / (q.max - q.min)),
      val: (t) => (lg ? q.min * Math.pow(q.max / q.min, t) : q.min + t * (q.max - q.min)), lo: q.min, hi: q.max };
  }
  const c = curveOf(tgt);
  if (!c) return null;
  const sc = curveScale(c);
  return base >= sc.lo - 1e-9 && base <= sc.hi + 1e-9 ? sc : null;
}
function modRange(tgt, base, routes) {
  const sc = onCurve(tgt, base);
  let lo = 0, hi = 0;
  for (const r of routes) {
    const amp = SRC[r.src].bipolar ? (MOD.spec.sources[r.src]?.amount ?? 1) : 1;            // an LFO's Amount narrows its ring
    const span = (sc ? 1 : tgtOf(r.tgt).span) * amp, a = r.depth * span * (SRC[r.src].bipolar ? -1 : 0), b = r.depth * span;
    lo += Math.min(a, b); hi += Math.max(a, b);
  }
  if (sc) { const t = sc.pos(base); return [sc.val(clamp01(t + lo)), sc.val(clamp01(t + hi))]; }
  return TGT[tgt]?.oct ? [base * Math.pow(2, lo), base * Math.pow(2, hi)] : [base + lo, base + hi];
}
function effOf(tgt, base, t) {
  const sc = onCurve(tgt, base);
  let off = 0;
  for (const r of MOD.spec.routes) if (r.tgt === tgt) off += r.depth * (sc ? 1 : tgtOf(tgt).span) * srcV(r.src, t);
  if (sc) return sc.val(clamp01(sc.pos(base) + off));
  return TGT[tgt]?.oct ? base * Math.pow(2, off) : base + off;
}
// the engine's answer is adopted only if nothing changed here after it was sent (else a newer
// change, still on its way, would be undone by an older answer)
const flushMod = throttle(async () => {
  const sent = MOD.touched;
  try { const r = await api("/api/mod", MOD.spec); if (!holding && MOD.touched === sent) { MOD.spec = r.mod; MOD.touched = Date.now(); } } catch (e) { fail(e); }
}, 80);
function sendMod() { MOD.touched = Date.now(); flushMod(); }
// the hint under the tabs shows while the pointer is over a source
function hintOn(el) {
  el.addEventListener("pointerenter", () => $("#assignnote").classList.add("hover"));
  el.addEventListener("pointerleave", () => $("#assignnote").classList.remove("hover"));
}
// while a source is chosen, the info bar at the foot says what the knobs now do (and comes back to it after other messages)
const assignText = () => `${SRC[MOD.assign].label}: each knob it can move is now its send. Click links it at ${START_SEND}; drag up for more, down for less; double-click removes. Esc to finish.`;
function assign(src) {
  MOD.assign = MOD.assign === src ? null : src;
  if (MOD.assign) info(assignText(), "", 0); else if (Date.now() > infoUntil) { $("#infotext").textContent = ""; $("#infotext").className = ""; }
  if (MOD.assign) { MOD.tab = tabOf(SRC[MOD.assign].kind); drawEditor(); }
  drawTiles(); MODKNOBS.forEach((k) => k.paint());          // every knob it can move turns into its send, or back
}

// ------------------------------------------------------------- the two views
let view = "make";
try { view = localStorage.getItem("aa-view") || "make"; } catch (e) { /* private window */ }
function setView(v) {
  view = v;
  $("#vw-make").setAttribute("aria-pressed", v === "make"); $("#vw-ses").setAttribute("aria-pressed", v === "session"); $("#vw-tune").setAttribute("aria-pressed", v === "tune"); $("#vw-listen").setAttribute("aria-pressed", v === "listen");
  $("#v-make").hidden = v !== "make"; $("#v-session").hidden = v !== "session"; $("#v-tune").hidden = v !== "tune"; $("#v-listen").hidden = v !== "listen";
  if (v === "tune") loadTune();
  $("#dock").hidden = v !== "session"; $("#dock-ses").hidden = v !== "session";
  $("#snapbtn").hidden = v !== "make";
  $("#play").title = v === "make" ? "Play the model" : "Start every track";
  $("#stop").title = v === "make" ? "Stop the model (the session is saved)" : "Stop every track";
  try { localStorage.setItem("aa-view", v); } catch (e) { /* private window */ }
  if (MOD.assign && v !== "make") assign(null);
  transport();
}
$("#vw-make").addEventListener("click", () => setView("make"));
$("#vw-ses").addEventListener("click", () => setView("session"));
$("#vw-tune").addEventListener("click", () => setView("tune"));
$("#vw-listen").addEventListener("click", () => setView("listen"));

// the dock's height, dragged; double-click folds it
(() => {
  const dock = $("#dock"), hd = $("#dockhandle");
  try { const v = localStorage.getItem("aa-dock"); if (v) dock.style.setProperty("--dock-h", v); } catch (e) { /* private window */ }
  hd.addEventListener("pointerdown", (e) => {
    hd.setPointerCapture(e.pointerId); hd.classList.add("held");
    const y0 = e.clientY, h0 = dock.getBoundingClientRect().height;
    const move = (ev) => { dock.classList.remove("folded"); dock.style.setProperty("--dock-h", `${Math.max(160, h0 + y0 - ev.clientY)}px`); };
    const up = () => { hd.classList.remove("held"); hd.removeEventListener("pointermove", move); hd.removeEventListener("pointerup", up);
      try { localStorage.setItem("aa-dock", dock.style.getPropertyValue("--dock-h")); } catch (e) { /* private window */ } };
    hd.addEventListener("pointermove", move); hd.addEventListener("pointerup", up);
  });
  hd.addEventListener("dblclick", () => dock.classList.toggle("folded"));
})();
$("#brfold").addEventListener("click", () => $("#browser").classList.add("folded"));
$("#brrail").addEventListener("click", () => $("#browser").classList.remove("folded"));

// ================================================================ MAKE
const S = { presets: [], dirs: [], ranges: {}, doses: {}, preset: null, playing: false, engine: null, frameMs: 40, status: {}, sampling: {}, master: {} };
function range() { return S.ranges[$("#full").checked ? "full" : "musical"] || [-0.4, 0.4]; }

// sending moves, at most every 30 ms
let pendingDoses = null, sendTimer = null, dosesTouched = 0;
function sendDoses(d) {
  Object.assign(S.doses, d); dosesTouched = Date.now();
  pendingDoses = Object.assign(pendingDoses || {}, d);
  if (sendTimer) return;
  sendTimer = setTimeout(async () => {
    const body = { doses: pendingDoses, full: $("#full").checked };
    pendingDoses = null; sendTimer = null;
    try { await api("/api/control", body); } catch (e) { fail(e); }
  }, 30);
}
let masterTouched = 0, samplingTouched = 0, masterPending = {}, sampPending = {};
const flushMaster = throttle(() => { const b = masterPending; masterPending = {}; api("/api/control", { master: b }).catch(fail); }, 40);
function sendMaster(m) { Object.assign(S.master, m); Object.assign(masterPending, m); masterTouched = Date.now(); flushMaster(); }
const flushSampling = throttle(() => { const b = sampPending; sampPending = {}; api("/api/control", { sampling: b }).catch(fail); }, 40);
function sendSampling(d) { Object.assign(S.sampling, d); Object.assign(sampPending, d); samplingTouched = Date.now(); flushSampling(); }

async function loadPresets() {
  const j = await api("/api/presets");
  const keep = $("#preset").value;
  S.presets = j.presets; S.dirs = j.directions; S.ranges = j.ranges;
  try { Object.assign(CAL, await api("/api/calibration")); } catch (e) { /* none yet */ }
  const groups = [];
  for (const p of S.presets) { const g = p.group || "Presets"; let e = groups.find((x) => x[0] === g); if (!e) groups.push(e = [g, []]); e[1].push(p); }
  $("#preset").innerHTML = groups.map(([g, ps]) => `<optgroup label="${esc(g)}">${ps.map((p) => `<option value="${esc(p.id)}">${esc(p.name)}</option>`).join("")}</optgroup>`).join("");
  if (keep && S.presets.some((p) => p.id === keep)) $("#preset").value = keep;
  pickPreset(true);
}
function pickPreset(keepSeed) {
  S.preset = S.presets.find((p) => p.id === $("#preset").value) || S.presets[0];
  if (!S.preset) return;
  if (!keepSeed) { $("#seed").value = S.preset.seeds[0]; seedKnob.paint?.(); }
  $("#prompt").textContent = `“${S.preset.prompt}”`; $("#prompt").title = S.preset.prompt;   // two lines at most; the whole prompt on hover
  $("#pdel").hidden = !S.preset.user;
  $("#cfg").checked = (S.preset.cfg || 1) > 1;
  buildDoses();
}
function presetMove(d) {
  const s = $("#preset");
  s.selectedIndex = (s.selectedIndex + d + s.options.length) % s.options.length;
  s.dispatchEvent(new Event("change"));
}
$("#presetprev").addEventListener("click", () => presetMove(-1));
$("#presetnext").addEventListener("click", () => presetMove(1));
const dirLabel = (name) => { const d = S.dirs.find((x) => x.name === name); return d ? `${d.negative || "−"} ← ${d.title} → ${d.positive || "+"}` : `${name || ""}`; };

// Steer: the pad and a knob per direction
let doseKnobs = [];
function buildDoses() {
  const [lo, hi] = range(), box = $("#doseknobs");
  doseKnobs.forEach((k) => MODKNOBS.delete(k));
  box.innerHTML = ""; doseKnobs = [];
  if (!S.dirs.length) { box.innerHTML = `<div class="pnote">Directions are being made…</div>`; return; }
  for (const d of S.dirs) {
    const k = knob({ label: d.title, min: lo, max: hi, step: 0.005, reset: 0, get: () => S.doses[d.name] || 0, mod: `dose:${d.name}`, curve: `dose:${d.name}`,
      put: (v) => { sendDoses({ [d.name]: v }); drawXY(); }, fmt: (v) => (v >= 0 ? "+" : "") + v.toFixed(3), tip: `Steer toward ${d.positive || "+"} (or ${d.negative || "−"}) at layer 6` });
    box.appendChild(k); doseKnobs.push(k);
  }
  $("#xlab").textContent = dirLabel(S.preset?.x); $("#ylab").textContent = dirLabel(S.preset?.y);
  drawXY();
}
const paintDoses = () => doseKnobs.forEach((f) => f.paint());
const cv = $("#xy"), cx = cv.getContext("2d");
function drawXY() {
  const w = cv.width, hh = cv.height, [lo, hi] = range();
  cx.clearRect(0, 0, w, hh);
  cx.strokeStyle = "#1f232b"; cx.lineWidth = 2;
  for (let k = 1; k < 8; k++) { cx.beginPath(); cx.moveTo((w * k) / 8, 0); cx.lineTo((w * k) / 8, hh); cx.moveTo(0, (hh * k) / 8); cx.lineTo(w, (hh * k) / 8); cx.stroke(); }
  cx.strokeStyle = "#343a45"; cx.beginPath(); cx.moveTo(w / 2, 0); cx.lineTo(w / 2, hh); cx.moveTo(0, hh / 2); cx.lineTo(w, hh / 2); cx.stroke();
  const xn = S.preset?.x, yn = S.preset?.y, bx = S.doses[xn] || 0, by = S.doses[yn] || 0;
  const P = (x, y) => [doseT(xn, x) * w, hh - doseT(yn, y) * hh];
  const t = modTime(), mx = effOf(`dose:${xn}`, bx, t), my = effOf(`dose:${yn}`, by, t);
  if (mx !== bx || my !== by) {                       // where modulation has moved it
    const [qx, qy] = P(mx, my);
    cx.fillStyle = "rgba(255,255,255,.85)"; cx.beginPath(); cx.arc(qx, qy, 6, 0, 2 * Math.PI); cx.fill();
  }
  const [px, py] = P(bx, by);
  cx.strokeStyle = "rgba(217,170,63,.35)"; cx.lineWidth = 2; cx.beginPath(); cx.moveTo(w / 2, hh / 2); cx.lineTo(px, py); cx.stroke();
  cx.fillStyle = "#d9aa3f"; cx.shadowColor = "rgba(217,170,63,.7)"; cx.shadowBlur = 16;
  cx.beginPath(); cx.arc(px, py, 14, 0, 2 * Math.PI); cx.fill(); cx.shadowBlur = 0;
  cx.fillStyle = "#0d0f13"; cx.beginPath(); cx.arc(px, py, 5, 0, 2 * Math.PI); cx.fill();
}
// a dose's place on the pad: along its curve when calibrated, else across the range
function doseT(name, v) { const c = curveOf(`dose:${name}`); if (c) return clamp01(curveScale(c).pos(v)); const [lo, hi] = range(); return clamp01((v - lo) / (hi - lo)); }
function doseV(name, t) { const c = curveOf(`dose:${name}`); if (c) return curveScale(c).val(t); const [lo, hi] = range(); return lo + t * (hi - lo); }
function padMove(e) {
  const r = cv.getBoundingClientRect();
  const fx = clamp01((e.clientX - r.left) / r.width), fy = clamp01(1 - (e.clientY - r.top) / r.height);
  const d = {};
  if (S.preset?.x) d[S.preset.x] = +doseV(S.preset.x, fx).toFixed(4);
  if (S.preset?.y) d[S.preset.y] = +doseV(S.preset.y, fy).toFixed(4);
  sendDoses(d); drawXY(); paintDoses();
}
let dragging = false;
cv.addEventListener("pointerdown", (e) => { dragging = true; cv.setPointerCapture(e.pointerId); padMove(e); });
cv.addEventListener("pointermove", (e) => { if (dragging) padMove(e); });
cv.addEventListener("pointerup", () => { dragging = false; });
$("#full").addEventListener("change", () => {
  if (!$("#full").checked) {
    const [lo, hi] = S.ranges.musical, d = {};
    for (const k in S.doses) d[k] = Math.min(hi, Math.max(lo, S.doses[k]));
    sendDoses(d);
  }
  buildDoses();
});
$("#centre").addEventListener("click", () => { const d = {}; S.dirs.forEach((x) => (d[x.name] = 0)); sendDoses(d); paintDoses(); drawXY(); });

// Engine
$("#preset").addEventListener("change", () => {
  pickPreset(false);
  if (S.playing && S.engine === "mrt2") api("/api/control", { preset: S.preset.id }).catch(fail);
});
// a style of your own: kept under Your prompts, embedded by the engine, then played (or switched to, if playing)
$("#pform").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = $("#ptype").value.trim();
  if (!text) return;
  $("#puse").disabled = true; info("making the style's tokens…", "", 8000);
  try {
    const r = await api("/api/prompts", { prompt: text });
    await loadPresets();
    $("#preset").value = r.preset.id; pickPreset(false);
    $("#ptype").value = "";
    if (S.playing && S.engine === "mrt2") await api("/api/control", { preset: r.preset.id });
    else await playModel();
    info(`${r.preset.name}${r.embedded && !r.embedded.cached ? ` (tokens made in ${Math.round(r.embedded.ms)} ms)` : ""}`, "ok");
  } catch (err) { fail(err); }
  finally { $("#puse").disabled = false; }
});
$("#pdel").addEventListener("click", async () => {
  if (!S.preset?.user) return;
  try { await api(`/api/prompts/${encodeURIComponent(S.preset.id)}/delete`, {}); await loadPresets(); info("removed from Your prompts", "ok"); } catch (err) { fail(err); }
});
// The model is chosen in Settings (⚙), Model: it is seldom changed, so the Engine panel only names it
async function switchEngine(id) {
  if (!id || id === S.engine) return;
  try { await api("/api/engine", { engine: id }); dirsKnown = -1; S.engine = null; listSessions(); } catch (e) { fail(e); }
}
function drawEngine(s) {
  const es = s.engines || [], cur = es.find((e) => e.id === s.engine);
  $("#engname").textContent = cur ? cur.label : (s.engine || "");
  const pick = $("#engpick");
  const html = es.map((e) => `<button type="button" data-e="${esc(e.id)}" aria-pressed="${e.id === s.engine}"><b>${esc(e.label)}</b><span>${e.id === s.engine ? "in use" : e.state === "ready" ? "ready" : esc(e.state)}</span></button>`).join("");
  if (pick && pick._html !== html) { pick.innerHTML = html; pick._html = html; }
}
$("#engname").addEventListener("click", () => openSettings("model"));
const engKnobs = [
  knob({ label: "Strength", min: 0, max: 8, step: 0.1, reset: 3, get: () => +(S.sampling.guidance ?? 3), put: (v) => sendSampling({ guidance: v }), fmt: (v) => v.toFixed(1), mod: "guidance", curve: "guidance", tip: "how strongly the prompt steers the music: the model's guidance, its weight on the style conditioning; live, at no cost" }),
  knob({ label: "Temperature", min: 0.5, max: 2, step: 0.05, reset: 1.3, get: () => +(S.sampling.temperature ?? 1.3), put: (v) => sendSampling({ temperature: v }), fmt: (v) => v.toFixed(2), mod: "temperature", curve: "temperature", tip: "the sampler's temperature" }),
  knob({ label: "Top-k", min: 1, max: 250, int: true, reset: 40, curve: "top_k", get: () => +(S.sampling.top_k ?? 40), put: (v) => sendSampling({ top_k: v }), tip: "how many of the likeliest tokens the sampler chooses among" }),
];
const speedKnob = knob({ label: "Speed", min: 0.125, max: 1, log: true, step: 0.005, reset: 1, get: () => +(S.master.speed ?? 1), put: (v) => sendMaster({ speed: +v.toFixed(3) }),
  fmt: (v) => ({ 1: "1×", 0.5: "½×", 0.25: "¼×", 0.125: "⅛×" }[+v.toFixed(3)] || `${v.toFixed(2)}×`),
  tip: "how fast the model's sound is played, from 1× down to ⅛×. The model writes only as fast as it is used. A saved sound replays its speed exactly" });
// how Speed slows the sound: Tape lets the pitch fall; Stretch keeps it (a phase vocoder); Smear keeps it and
// washes it out over long windows (after Paulstretch), for very slow ambient; Off plays at 1× and keeps the Speed knob's value
const SPEED_MODES = [["off", "Off", "plays at 1×; Speed keeps its setting"], ["tape", "Tape", "pitch falls with speed, as a slowed tape"],
  ["stretch", "Stretch", "pitch kept: a phase vocoder with locked phases"], ["smear", "Smear", "pitch kept, washed out over 0.34 s windows: for very slow ambient"]];
const modeAt = () => Math.max(0, SPEED_MODES.findIndex(([k]) => k === (S.master.speed_mode || "tape")));
const speedModes = knob({ label: "Mode", min: 0, max: SPEED_MODES.length - 1, step: 1, int: true, reset: 0, get: modeAt,
  put: (v) => { const k = SPEED_MODES[Math.round(v)][0]; if (k !== S.master.speed_mode) { S.master.speed_mode = k; sendMaster({ speed_mode: k }); paintSpeedModes(); } },
  fmt: (v) => SPEED_MODES[Math.round(v)][1],
  tip: "how Speed slows the sound. " + SPEED_MODES.map(([, l, t]) => `${l}: ${t}`).join(". ") + ". Double-click: Off" });
// while Mode is Off, Speed is dimmed and does not turn: its value is kept for when a mode is chosen again
// MusicGen's output is one channel and has no Speed stage, so there all four are dimmed too
const dimKnob = (k, off) => { k.classList.toggle("disabled", off); k.setAttribute("aria-disabled", String(off)); k.tabIndex = off ? -1 : 0; };
const paintSpeedModes = () => {
  speedModes.paint?.();
  const mg = S.engine === "musicgen";
  dimKnob(speedModes, mg);
  dimKnob(speedKnob, mg || (S.master.speed_mode || "tape") === "off");
  widenKnobs?.forEach((k) => dimKnob(k, mg));
};
const speedPair = h(`<div class="seedpair speedpair kgroup" data-label="SPEED CONTROL"></div>`);
speedPair.append(speedModes, speedKnob);
// the seed as a knob beside them (its value is #seed, read at the next ▶); the die picks one at random
const seedKnob = knob({ label: "Seed", min: 0, max: 999, int: true, reset: 1, get: () => +$("#seed").value || 0,
  put: (v) => { $("#seed").value = String(Math.round(v)); }, fmt: (v) => String(Math.round(v)),
  tip: "the seed the model starts from; it takes effect at the next ▶. Double-click: seed 1" });
const dice = h(`<button class="ib seeddie" type="button" title="A random seed, at the next ▶">⚄</button>`);
dice.addEventListener("click", () => { $("#seed").value = String(Math.floor(Math.random() * 1000)); seedKnob.paint?.(); });
const seedPair = h(`<div class="seedpair"></div>`);
seedPair.append(seedKnob, dice);
$("#engknobs").appendChild(seedPair);
// Strength (the model's guidance) in the row of the prompt it weighs; under them Seed, then Temperature and Top-k as a pair under SAMPLING
const [guidKnob, tempKnob, topkKnob] = engKnobs;
$("#pform").appendChild(guidKnob); guidKnob.classList.add("kmini");     // in the Prompt box's row, after ↵
const sampPair = h(`<div class="kgroup samppair" data-label="SAMPLING"></div>`);
sampPair.append(tempKnob, topkKnob);
$("#engknobs").appendChild(sampPair);

// Output: the master, after the model
let hdrVol = null;                                               // the header's Volume, the same as the Output panel's
const outKnobs = [
  knob({ label: "Low-pass", min: 500, max: 16000, log: true, step: 10, reset: 16000, get: () => +(S.master.lowpass_hz ?? 16000), put: (v) => sendMaster({ lowpass_hz: v }), fmt: (v) => `${(v / 1000).toFixed(1)}k`, mod: "lowpass_hz", tip: "a low-pass filter on the model's output" }),
  knob({ label: "Volume", min: 0, max: 1, step: 0.01, reset: 0.5, get: () => +(S.master.volume ?? 0.5), put: (v) => { sendMaster({ volume: v }); hdrVol?.paint(); }, fmt: (v) => v.toFixed(2), mod: "volume" }),
];
const outToggles = [toggle("Bypass", () => !S.master.on, (v) => sendMaster({ on: !v })), toggle("Level", () => !!S.master.level, (v) => sendMaster({ level: v })),
  toggle("Limiter", () => !!S.master.limit, (v) => sendMaster({ limit: v })), toggle("De-click", () => !!S.master.declick, (v) => sendMaster({ declick: v }))];
outToggles[3].title = "De-click: redraws the few-sample bursts a unit can make the model write (Quantise at low bits makes them), across 1.3 ms of look-ahead; drum attacks are left as they are";
outToggles[0].title = "Bypass: the model straight to Volume, skipping auto level, the effects, Dynamics, Width/Spread, the low-pass and the limiter";
outToggles[1].title = "Auto level: holds the loudness steady (−20 dBFS) while the controls move, so a move is heard as a change of sound more than of level. Saved sounds keep it.";
const leadKnob = knob({ label: "Buffer", min: 1, max: 12, int: true, reset: 2, get: () => +(S.status.lead || 2), put: (v) => { S.status.lead = v; api("/api/lead", { lead: v }).catch(fail); }, fmt: (v) => `${(v + (S.status.lead_boost || 0)) * S.frameMs} ms`, tip: "frames made ahead of the sound card: shorter answers sooner and risks dropouts. Shown in gold when the engine has added frames by itself after a step ran long (given back one at a time after a quiet minute); the readout is then the total" });
outKnobs.forEach((k) => $("#outknobs").appendChild(k));
// the stereo field's knobs, beside its scope at the foot of the Engine panel
const widenKnobs = [
  knob({ label: "Width", min: 0, max: 2, step: 0.01, reset: 1, get: () => +(S.master.width ?? 1), put: (v) => sendMaster({ width: v }), fmt: (v) => `${Math.round(v * 100)}%`, mod: "width",
    tip: "the side's gain: 0% mono, 100% as the model wrote it, 200% twice as wide" }),
  knob({ label: "Spread", min: 0, max: 1, step: 0.01, reset: 0, get: () => +(S.master.spread ?? 0), put: (v) => sendMaster({ spread: v }), fmt: (v) => `${Math.round(v * 100)}%`, mod: "spread",
    tip: "width from a mono sound: the mid, delayed 12 ms, added to the left and taken from the right; the mono sum is unchanged" }),
];
// Speed and its Mode, then Width and Spread: each pair under a bracket with its name; all four act on the sound the model has written
const stereoPair = h(`<div class="stereopair kgroup" data-label="STEREO"></div>`);
widenKnobs.forEach((k) => stereoPair.appendChild(k));
$("#widen").append(speedPair, stereoPair);
$("#outknobs").appendChild(leadKnob);
$("#outknobs").appendChild((() => { const r = h(`<div class="outtg"></div>`); outToggles.forEach((t) => r.appendChild(t)); return r; })());
// Dynamics: the master's compressor and gate (dsp.Master._dynamics), after the effects, with their reduction on LEDs
const dynKnobs = [
  knob({ label: "Ratio", min: 1, max: 20, log: true, step: 0.1, reset: 4, get: () => +(S.master.dyn_ratio ?? 4), put: (v) => sendMaster({ dyn_ratio: v }), fmt: (v) => `${v < 10 ? v.toFixed(1) : Math.round(v)}:1`, tip: "how hard the level above the threshold is pulled down: 4:1 lets 1 dB out for every 4 dB over" }),
  knob({ label: "Thresh", min: -40, max: 0, step: 0.5, reset: -18, get: () => +(S.master.dyn_thresh ?? -18), put: (v) => sendMaster({ dyn_thresh: v }), fmt: (v) => v.toFixed(0), tip: "the compressor's threshold, dBFS" }),
  knob({ label: "Release", min: 0.05, max: 4, log: true, step: 0.01, reset: 0.3, get: () => +(S.master.dyn_release ?? 0.3), put: (v) => sendMaster({ dyn_release: v }), fmt: (v) => (v < 1 ? `${Math.round(v * 1000)}ms` : `${v.toFixed(1)}s`), tip: "how fast the compressor and the gate let go" }),
  knob({ label: "Gate", min: -80, max: -20, step: 0.5, reset: -80, get: () => +(S.master.gate_thresh ?? -80), put: (v) => sendMaster({ gate_thresh: v }), fmt: (v) => (v <= -79.5 ? "off" : v.toFixed(0)), tip: "the gate's threshold, dBFS: below it the sound drops 40 dB; fully left is off" }),
];
const dynToggles = [toggle("Dyn", () => !!S.master.dyn_on, (v) => sendMaster({ dyn_on: v })), toggle("Fast", () => !!S.master.dyn_fast, (v) => sendMaster({ dyn_fast: v })), toggle("Peak", () => !!S.master.dyn_peak, (v) => sendMaster({ dyn_peak: v }))];
dynToggles[0].title = "Dynamics on: a compressor and a gate after the effects, before Width and the limiter";
dynToggles[1].title = "Fast attack: 3 ms instead of 30 ms";
dynToggles[2].title = "Peak: the compressor and gate follow the peak instead of the average (RMS) level";
const DYN = (() => {
  const el = $("#dyn"), COMP = [1, 3, 6, 10, 14, 20], GATE = [3, 10, 20, 40];
  el.innerHTML = `<div class="dyh"></div><div class="dyk"></div>`;
  dynToggles.forEach((t) => el.querySelector(".dyh").appendChild(t));
  dynKnobs.forEach((k) => el.querySelector(".dyk").appendChild(k));
  // the reduction LEDs sit in Output's stage rows, above Auto and Limit
  const rows = h(`<div class="dyrows"><div class="sg"><span>Comp</span><div class="dled c">${COMP.map((d) => `<i title="${d} dB"></i>`).join("")}</div><span class="sv"></span></div><div class="sg"><span>Gate</span><div class="dled g">${GATE.map((d) => `<i title="${d} dB"></i>`).join("")}</div><span class="sv"></span></div></div>`);
  let cl = null, gl = null;
  return { set(s) {
    if (!cl) { $("#stages").prepend(rows); [cl, gl] = [...rows.querySelectorAll(".dled")].map((x) => [...x.querySelectorAll("i")]); }
    const sv = rows.querySelectorAll(".sv");
    const st = s.playing && S.master.dyn_on ? s.stages || {} : {};
    const c = -(st.comp_db || 0), g = -(st.gate_db || 0);
    COMP.forEach((d, i) => cl[i].classList.toggle("on", c >= d));
    GATE.forEach((d, i) => gl[i].classList.toggle("on", g >= d));
    sv[0].textContent = S.master.dyn_on ? (st.comp_db != null ? (-c).toFixed(1) : "–") : "off";
    sv[1].textContent = S.master.dyn_on ? (st.gate_db != null ? (-g).toFixed(1) : "–") : "off";
    el.classList.toggle("off", !S.master.dyn_on); rows.classList.toggle("off", !S.master.dyn_on);
  } };
})();
const engMeter = meter("OUT", true);
$("#outmeter").appendChild(engMeter);
// the output level and Volume at the top right of the header, as on a synth's top bar: always in view
const hdrMeter = meter("OUT", true);
hdrMeter.classList.add("hdrmeter");
hdrMeter.title = "The output level: each channel's peak, with a hold line; the figure is the held peak (dBFS). The same as the Output panel's OUT meter";
hdrVol = knob({ label: "Volume", min: 0, max: 1, step: 0.01, reset: 0.5, get: () => +(S.master.volume ?? 0.5),
  put: (v) => { sendMaster({ volume: v }); outKnobs.find((k) => k.getAttribute("aria-label") === "Volume")?.paint(); }, fmt: (v) => v.toFixed(2), mod: "volume",
  tip: "the output's volume, the same as Volume in the Output panel" });
hdrVol.classList.add("kmini");
$("#hdrout").append(hdrMeter, hdrVol);

// Drag to reorder: an item is picked up by its handle and dropped before or after another item in
// the same row; move(from, to) gets the item's old place and its new one.
const DRAG = { row: null, from: -1 };
function sortable(item, handle, row, k, move) {
  handle.draggable = true;
  handle.classList.add("grab");
  handle.addEventListener("dragstart", (e) => {
    DRAG.row = row; DRAG.from = k;
    e.dataTransfer.effectAllowed = "move";
    e.dataTransfer.setData("text/plain", String(k));
    try { e.dataTransfer.setDragImage(item, 20, 14); } catch (err) { /* the handle alone is shown */ }
    item.classList.add("dragging");
  });
  handle.addEventListener("dragend", () => { item.classList.remove("dragging"); row.querySelectorAll(".dropl,.dropr").forEach((x) => x.classList.remove("dropl", "dropr")); DRAG.row = null; });
  const side = (e) => { const b = item.getBoundingClientRect(); return e.clientX < b.left + b.width / 2 ? 0 : 1; };
  item.addEventListener("dragover", (e) => {
    if (DRAG.row !== row) return;
    e.preventDefault(); e.dataTransfer.dropEffect = "move";
    const s = side(e);
    item.classList.toggle("dropl", s === 0); item.classList.toggle("dropr", s === 1);
  });
  item.addEventListener("dragleave", () => item.classList.remove("dropl", "dropr"));
  item.addEventListener("drop", (e) => {
    if (DRAG.row !== row) return;
    e.preventDefault();
    const from = DRAG.from, to0 = k + side(e), to = to0 > from ? to0 - 1 : to0;
    item.classList.remove("dropl", "dropr");
    if (to !== from) move(from, to);
  });
}
const moved = (list, from, to) => { const [x] = list.splice(from, 1); list.splice(to, 0, x); };
// A + in the gap before item `slot`, shown when the pointer is over the gap, as Abliterton Lens has it;
// open(button, slot) shows the menu of what can go there.
function gapAdd(slot, title, open) {
  const b = h(`<button type="button" class="addgap" aria-haspopup="menu" title="${esc(title)}" aria-label="${esc(title)}"><span class="plus">+</span></button>`);
  b.addEventListener("click", () => open(b, slot));
  return b;
}
const whereIn = (slot, n, what) => (slot >= n ? (n ? `Add after ${what} ${n}` : `Add the first ${what}`) : slot === 0 ? `Add before ${what} 1` : `Add between ${what}s ${slot} and ${slot + 1}`);

// the effect chain (postfx.py): none to eight effects, in order, on the model's output. Each has an
// id, so the engine keeps its tail when others are added, removed or moved around it.
const PFX = { types: {}, order: [], slots: [], ctl: [], touched: 0, max: 8, uid: 0, sent: null };
const fxId = (x) => (x.id ??= ++PFX.uid);
const flushFx = throttle(() => { followFx(); sendMaster({ fx: PFX.slots.map((x) => ({ id: fxId(x), type: x.type, on: x.on, params: { ...x.params } })) }); }, 60);
function fxSet(i, patch) { Object.assign(PFX.slots[i], patch); PFX.touched = Date.now(); flushFx(); }
async function loadFx() {
  try { const j = await api("/api/fxtypes"); PFX.types = j.types || {}; PFX.order = Object.keys(PFX.types); PFX.max = j.max || 8; }
  catch (e) { PFX.loading = false; return; }                     // tried again at the next status
  PFX.loading = false;
  if (!PFX.order.length) return;
  if (!PFX.slots.length) PFX.slots = (S.master.fx || []).map((x) => ({ ...x, params: { ...x.params } }));
  PFX.slots.forEach((x) => { if (typeof x.id === "number") PFX.uid = Math.max(PFX.uid, x.id); });
  PFX.sent = PFX.slots.map(fxId);
  drawFx();
}
// Modulation names an effect's knob by its place in the chain (pfx:<i>:<key>); when effects are
// added, removed or moved, the routes follow their effects, and a removed effect's routes go.
function followFx() {
  const now = PFX.slots.map(fxId), was = PFX.sent;
  PFX.sent = now;
  if (!was) return;
  let changed = false;
  const routes = [];
  for (const r of MOD.spec.routes) {
    const m = /^pfx:(\d+):(.+)$/.exec(r.tgt);
    if (!m) { routes.push(r); continue; }
    const k = now.indexOf(was[+m[1]]);
    if (k < 0) { changed = true; continue; }
    if (k !== +m[1]) changed = true;
    routes.push({ ...r, tgt: `pfx:${k}:${m[2]}` });
  }
  if (changed) { MOD.spec.routes = routes; sendMod(); drawTiles(); }
}
const FX_GROUPS = [["Space", ["reverb", "plate", "hall", "delay", "shimmer"]], ["Modulation", ["chorus", "phaser", "flanger", "tremolo", "pan"]], ["Tone", ["eq", "eq8", "filter", "tape", "resonator"]], ["Pitch and time", ["pitch", "freqshift", "freeze"]], ["Distortion", ["drive", "crush", "abliterated"]], ["Dynamics and stereo", ["compressor", "trim"]], ["Repair", ["declick"]]];
const FX_WHAT = { phaser: "a chain of allpass filters swept by an LFO, with feedback", flanger: "a short delay swept by an LFO, with feedback", tremolo: "the level swung by an LFO", compressor: "turns the loud parts down above a threshold, by a ratio", resonator: "tuned comb filters that ring at a root note and a chord", freqshift: "moves every frequency up or down by the same number of hertz", pitch: "a delay-line pitch shifter with two crossfaded read heads", shimmer: "a reverb whose tail is pitch-shifted on each pass", freeze: "a spectral hold of the last moment of sound", trim: "gain, balance and the stereo field: width, a phase flip on each side, channel swap, bass mono, a mono check and a DC filter", abliterated: "heavy distortion: two clipping stages (Crunch), a wavefolder (Edge), an octave and a ring modulator (Complicate), aimed by Steer", reverb: "an eight-line feedback delay network", plate: "Dattorro's plate: a dense, bright tail", hall: "a larger network whose delays drift slowly at random",
  delay: "repeats, with feedback and ping-pong", chorus: "a modulated short delay", drive: "saturation with a tone control", pan: "places the sound left or right, or swings it with an auto-pan", tape: "a tape machine's colour: saturation, wow and flutter, head bump and roll-off by speed (inches a second), hiss", eq: "three bands", eq8: "a low cut, six bands and a high cut", filter: "low-, high- or band-pass with resonance", crush: "fewer bits and a lower rate", declick: "redraws the few-sample bursts a unit can make the model write" };
function addFx(type, slot = PFX.slots.length) {
  if (PFX.slots.length >= PFX.max) return;
  PFX.slots.splice(slot, 0, { id: ++PFX.uid, type, on: true, params: fxDefaults(type) });
  PFX.touched = Date.now(); flushFx(); drawFx();
}
function fxDefaults(type) { const o = {}; for (const q of PFX.types[type].params) o[q.key] = q.default; return o; }
// How each effect's card is laid out: rows of knobs (Mix last), then a bottom row of its menus and
// switches. The display at the top draws what the settings do (fxGraph).
const FX_ROWS = {
  reverb: [["size", "predelay", "damp", "mix"]],
  plate: [["decay", "size", "predelay"], ["damp", "mod", "mix"]],
  hall: [["size", "predelay", "damp"], ["mod", "mix"]],
  delay: [["time", "feedback", "damp", "mix"]],
  chorus: [["rate", "depth", "mix"]],
  drive: [["amount"], ["tone"], ["mix"]],
  pan: [["pos"], ["rate"], ["depth"]],
  tape: [["drive", "wow", "flutter"], ["noise", "mix"]],
  eq: [["low", "mid", "freq", "high"]],
  eq8: [["lowcut", "b60", "b150", "b400"], ["b1k", "b2k5", "b6k", "highcut"]],
  filter: [["cutoff", "res"]],
  crush: [["bits", "hold", "mix"]],
  abliterated: [["crunch", "edge", "complicate", "steer"], ["tone", "level", "mix"]],
  declick: [["sens"], ["longest"]],
  trim: [["width", "gain", "balance"], ["bassf"]],
  phaser: [["rate", "depth", "centre"], ["feedback", "spread", "mix"]],
  flanger: [["rate", "depth", "delay"], ["feedback", "spread", "mix"]],
  tremolo: [["rate", "depth", "phase"]],
  compressor: [["thresh", "ratio", "knee", "makeup"], ["attack", "release", "mix"]],
  resonator: [["root", "decay"], ["bright", "mix"]],
  freqshift: [["shift", "spread"], ["feedback", "mix"]],
  pitch: [["shift", "fine", "window"], ["feedback", "mix"]],
  shimmer: [["decay", "shimmer"], ["damp", "mix"]],
  freeze: [["blur", "drift", "mix"]],
};
const FX_SPACE = new Set(["reverb", "plate", "hall", "delay", "shimmer"]);
const FX_NARROW = new Set(["drive", "pan", "declick"]);     // a knob a row, on a narrow card
const FX_ZIG = new Set(["drive", "pan"]);                    // its rows offset left and right
const FX_GRAPH = new Set(["filter", "eq", "eq8"]);          // a display only where a curve says more than the knobs
// RBJ biquads as postfx._biquad designs them, for drawing: the magnitude in dB at frequency f
function biquadDb(kind, f0, q, g, f) {
  const SR = 48000, A = 10 ** (g / 40), w = 2 * Math.PI * Math.min(f0, SR / 2 * 0.98) / SR, cw = Math.cos(w), sw = Math.sin(w), al = sw / (2 * q);
  let b, a;
  if (kind === "lp") { b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]; a = [1 + al, -2 * cw, 1 - al]; }
  else if (kind === "hp") { b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]; a = [1 + al, -2 * cw, 1 - al]; }
  else if (kind === "bp") { b = [al, 0, -al]; a = [1 + al, -2 * cw, 1 - al]; }
  else if (kind === "peak") { b = [1 + al * A, -2 * cw, 1 - al * A]; a = [1 + al / A, -2 * cw, 1 - al / A]; }
  else { const s2 = 2 * Math.sqrt(A) * al, lo = kind === "lowshelf", sg = lo ? 1 : -1;
    b = [A * ((A + 1) - sg * (A - 1) * cw + s2), sg * 2 * A * ((A - 1) - sg * (A + 1) * cw), A * ((A + 1) - sg * (A - 1) * cw - s2)];
    a = [(A + 1) + sg * (A - 1) * cw + s2, -sg * 2 * ((A - 1) + sg * (A + 1) * cw), (A + 1) + sg * (A - 1) * cw - s2]; }
  const x = 2 * Math.PI * f / SR, c1 = Math.cos(x), s1 = Math.sin(x), c2 = Math.cos(2 * x), s2_ = Math.sin(2 * x);
  const nr = b[0] + b[1] * c1 + b[2] * c2, ni = -(b[1] * s1 + b[2] * s2_), dr = a[0] + a[1] * c1 + a[2] * c2, di = -(a[1] * s1 + a[2] * s2_);
  return 10 * Math.log10((nr * nr + ni * ni) / Math.max(1e-30, dr * dr + di * di));
}
const GW = 200, GH = 44;                                         // the display's drawing units
const gpath = (pts) => pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`).join("");
const clampY = (y) => Math.max(1, Math.min(GH - 1, y));
// What the settings do, drawn: the filter's or EQ's response.
function fxGraph(type, p) {
  const fx = (i, n) => 20 * Math.pow(1000, i / n);                // 20 Hz to 20 kHz, log
  if (type === "eq8") {                                             // as postfx.EQ8: Butterworth cuts, bells of Q 1.2
    const qs = +p.slope === 0 ? [0.7071] : [0.5412, 1.3066], hz = [60, 150, 400, 1000, 2500, 6000], ks = ["b60", "b150", "b400", "b1k", "b2k5", "b6k"];
    const db = (f) => qs.reduce((a, q) => a + biquadDb("hp", p.lowcut, q, 0, f) + biquadDb("lp", p.highcut, q, 0, f), 0)
      + ks.reduce((a, k, j) => a + biquadDb("peak", hz[j], 1.2, +p[k] || 0, f), 0);
    const pts = []; for (let i = 0; i <= 100; i++) pts.push([i / 100 * GW, clampY(GH / 2 - db(fx(i, 100)) * (GH / 2) / 18)]);
    return `<line class="g0" x1="0" y1="${GH / 2}" x2="${GW}" y2="${GH / 2}"/><path class="ga" d="${gpath(pts)}L${GW},${GH}L0,${GH}Z"/><path class="gl" d="${gpath(pts)}"/>`;
  }
  if (type === "filter" || type === "eq") {
    const db = (f) => type === "filter" ? biquadDb(["lp", "hp", "bp"][+p.mode || 0], p.cutoff, p.res, 0, f)
      : biquadDb("lowshelf", 200, 0.707, p.low, f) + biquadDb("peak", p.freq, 0.9, p.mid, f) + biquadDb("highshelf", 5000, 0.707, p.high, f);
    const pts = []; for (let i = 0; i <= 80; i++) pts.push([i / 80 * GW, clampY(GH / 2 - db(fx(i, 80)) * (GH / 2) / 18)]);
    const mark = type === "filter" ? p.cutoff : p.freq, mx = Math.log(mark / 20) / Math.log(1000) * GW;
    return `<line class="g0" x1="0" y1="${GH / 2}" x2="${GW}" y2="${GH / 2}"/><line class="gm" x1="${mx}" y1="2" x2="${mx}" y2="${GH - 2}"/>`
      + `<path class="ga" d="${gpath(pts)}L${GW},${GH}L0,${GH}Z"/><path class="gl" d="${gpath(pts)}"/>`;
  }
  return "";
}
const fxMenu = (btn, slot) => menu(btn, [{ where: whereIn(slot, PFX.slots.length, "effect") }, ...FX_GROUPS.flatMap(([title, ts]) => [{ head: title },
  ...ts.filter((t) => PFX.types[t]).map((t) => ({ label: PFX.types[t].label, sub: FX_WHAT[t] || "", run: () => addFx(t, slot) }))])]);
function drawFx() {
  const row = $("#fxrow");
  row.innerHTML = ""; PFX.ctl = []; PFX.leds = {}; dropKnobs();
  const full = PFX.slots.length >= PFX.max;
  row.appendChild(h(`<div class="io" title="The model's sound, after auto level">in</div>`));
  PFX.slots.forEach((sl, i) => {
    const T = PFX.types[sl.type];
    if (!T) return;
    if (!full) row.appendChild(gapAdd(i, i === 0 ? "Add an effect before effect 1" : `Add an effect between effects ${i} and ${i + 1}`, fxMenu));
    const el = h(`<div class="fxs ${sl.on ? "" : "off"} ${FX_SPACE.has(sl.type) ? "space" : "tone"} ${FX_NARROW.has(sl.type) ? "narrow" : ""} ${FX_ZIG.has(sl.type) ? "zig" : ""}"><div class="fxh"><button class="pw ${sl.on ? "on" : ""}" title="On or bypassed"></button>
      <b class="fxn" title="Drag to move it in the chain">${esc(T.label)}</b><span class="fxi">${i + 1}</span><button class="spre" title="This effect's menu" aria-label="Effect menu" aria-haspopup="menu">▾</button></div>
      ${FX_GRAPH.has(sl.type) ? `<div class="fxg"><svg viewBox="0 0 ${GW} ${GH}" preserveAspectRatio="none" aria-hidden="true"></svg></div>` : ""}<div class="fxk"></div><div class="fxb"></div><button class="qhelp" type="button" aria-label="Help: ${esc(T.label)}" title="What ${esc(T.label)} does: how it works and its controls">?</button></div>`);
    el.querySelector(".pw").addEventListener("click", () => { fxSet(i, { on: !sl.on }); drawFx(); });
    const fm = el.querySelector(".spre");                         // the effect's menu: Remove for now, presets to come
    fm.addEventListener("click", () => menu(fm, [
      ...presetItems(T.presets, MY.fx[sl.type], (q) => { fxSet(i, { params: { ...PFX.slots[i].params, ...q.params } }); drawFx(); }, (q) => delMine("fx", sl.type, q)),
      ...mineItems("fx", sl.type, () => ({ ...PFX.slots[i].params }), () => undefined),
      { sep: true },
      { label: "Remove", danger: true, title: "Or select the card and press Delete", run: () => removeFx(sl, i) }], "cardmenu"));
    selectable(el, `fx${i}:${sl.type}`, () => removeFx(sl, i));     // by place: the slots are rebuilt from the engine's state
    el.querySelector(".qhelp").addEventListener("click", (e) => { e.stopPropagation(); fxHelp(sl.type, e.currentTarget); });
    sortable(el, el.querySelector(".fxn"), row, i, (from, to) => { moved(PFX.slots, from, to); PFX.touched = Date.now(); flushFx(); drawFx(); });
    const svg = el.querySelector(".fxg svg");
    const params = () => { const o = {}; for (const q of T.params) o[q.key] = +(PFX.slots[i].params[q.key] ?? q.default); return o; };
    const paintG = () => { if (svg) svg.innerHTML = fxGraph(sl.type, params()); };
    paintG(); if (svg) PFX.ctl.push({ paint: paintG });
    if (sl.type === "drive" || sl.type === "tape") {                // how hard the input drives the curve, in cells, as Output's meters
      const m = h(`<div class="fxmeter" title="How hard the input drives the curve: green bends it, amber flattens it, red holds it flat at the top (CLIP)"><span>drive</span><div class="cells">${"<i></i>".repeat(10)}</div><b>clip</b></div>`);
      el.querySelector(".fxk").before(m); PFX.leds[fxId(sl)] = { cells: [...m.querySelectorAll("i")], clip: m.querySelector("b"), hold: 0 };
    }
    if (sl.type === "pan") {                                        // where the sound sits now, L to R
      const m = h(`<div class="fxpan" title="Where the sound sits now, left to right, the auto-pan included"><span>L</span><div class="track"><i></i></div><span>R</span></div>`);
      el.querySelector(".fxk").before(m); PFX.leds[fxId(sl)] = { pan: m.querySelector(".track i") };
    }
    if (sl.type === "declick") {                                    // how many bursts it has redrawn
      const c = h(`<div class="fxcount" title="Bursts redrawn since this De-click was added"><span>fixed</span><b>0</b></div>`);
      el.querySelector(".fxk").before(c); PFX.leds[fxId(sl)] = { count: c.querySelector("b") };
    }
    const byKey = Object.fromEntries(T.params.map((q) => [q.key, q]));
    const get = (q) => +(PFX.slots[i].params[q.key] ?? q.default);
    const put = (q, v) => { PFX.slots[i].params[q.key] = v; PFX.touched = Date.now(); flushFx(); paintG(); };
    const fmt = (q) => (v) => q.key === "pos" && sl.type === "pan" ? (Math.abs(v) < 0.005 ? "C" : `${v < 0 ? "L" : "R"}${Math.round(Math.abs(v) * 100)}`) : (q.unit === "Hz" ? (v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v < 10 ? `${v.toFixed(2)}` : `${Math.round(v)}`) : q.unit === "ms" ? `${Math.round(v)} ms` : q.unit === "s" ? `${v.toFixed(1)} s` : q.unit === "dB" ? `${v > 0 ? "+" : ""}${v.toFixed(1)}` : q.unit === "×" ? `${v.toFixed(1)}×` : q.scale === "int" ? `${Math.round(v)}` : v.toFixed(2));
    const kbox = el.querySelector(".fxk");
    const rows = FX_ROWS[sl.type] || [T.params.filter((q) => !q.scale.startsWith("choice") && q.scale !== "toggle").map((q) => q.key)];
    for (const keys of rows) {
      const r = h(`<div class="fxkr"></div>`);
      for (const key of keys) {
        const q = byKey[key]; if (!q) continue;
        const tg = `pfx:${i}:${q.key}`;
        MODTGT[tg] = { label: `${T.label} ${q.label}`, min: q.min, max: q.max, log: q.scale === "log" };
        const k = knob({ label: q.label, min: q.min, max: q.max, log: q.scale === "log", int: q.scale === "int", step: q.scale === "int" ? 1 : q.scale === "log" ? undefined : (q.max - q.min) / 200, reset: q.default, get: () => get(q), put: (v) => put(q, v), fmt: fmt(q), mod: tg });
        r.appendChild(k); PFX.ctl.push(k);
      }
      kbox.appendChild(r);
    }
    const bot = el.querySelector(".fxb");
    for (const q of T.params) {                                     // menus as segmented buttons, switches as toggles
      if (q.scale === "toggle") { const t = toggle(q.label, () => get(q) >= 0.5, (v) => put(q, v ? 1 : 0)); bot.appendChild(t); PFX.ctl.push(t); continue; }
      if (!q.scale.startsWith("choice:")) continue;
      const opts = q.scale.slice(7).split(",");
      const old = !(q.key in PFX.slots[i].params);                // a slot saved before this param existed plays as it did then
      const seg = h(`<div class="fxseg" role="group" aria-label="${esc(q.label)}" title="${esc(q.label)}${old ? ": as saved before the colour modes (0.1.5); choose one to change it" : ""}"><span>${esc(q.label)}</span>${opts.map((o, j) => `<button type="button" data-j="${j}">${esc(o)}</button>`).join("")}</div>`);
      const paintSeg = () => seg.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(!(q.key in PFX.slots[i].params) ? false : +b.dataset.j === get(q))));
      seg.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { put(q, +b.dataset.j); paintSeg(); }));
      paintSeg(); PFX.ctl.push({ paint: paintSeg });
      bot.appendChild(seg);
    }
    if (!bot.children.length) bot.remove();
    row.appendChild(el);
  });
  const add = h(`<button class="fxadd" title="${full ? `The chain holds ${PFX.max} effects` : "Add an effect to the end of the chain"}" aria-haspopup="menu">+</button>`);
  add.disabled = full || !PFX.order.length;
  add.addEventListener("click", () => fxMenu(add, PFX.slots.length));
  if (!PFX.slots.length) row.appendChild(h(`<div class="empty">No effects: the model's sound goes straight on to Dynamics and the output</div>`));
  row.appendChild(add);
  row.appendChild(h(`<div class="io" title="On to Dynamics, Width, the low-pass, the limiter and the volume">out</div>`));
  drawEditor();                                                 // its links name the effects' knobs, which may have moved
}
// Drive's meter from the engine's readings: the peak drive into its curve, 0.25 to 8 on a log scale in
// ten cells; below 1 the curve is nearly straight, from 1 it bends, from 2 it flattens, above 4 it holds flat
// (CLIP, held for a second).
const DRIVE_CELLS = [0.25, 0.5, 0.75, 1, 1.4, 2, 2.8, 4, 5.6, 8];
function paintFxLeds(s) {
  const by = {};
  for (const r of (s.playing && s.stages && s.stages.fx) || []) if (r.id != null) by[r.id] = r;
  const now = Date.now();
  for (const [id, m] of Object.entries(PFX.leds || {})) {
    if (m.count) { if (by[id]?.fixed != null) m.count.textContent = by[id].fixed; continue; }
    if (m.pan) { m.pan.style.left = `${(((by[id]?.pos ?? 0) + 1) / 2 * 100).toFixed(1)}%`; continue; }
    const d = by[id]?.drive ?? 0;
    m.cells.forEach((c, j) => c.classList.toggle("on", d >= DRIVE_CELLS[j]));
    if (d > 4) m.hold = now + 1000;
    m.clip.classList.toggle("on", now < m.hold);
  }
}
function adoptFx(fx) {
  if (!fx || holding || Date.now() - PFX.touched < 800) return;
  const key = JSON.stringify(fx);
  if (key === PFX.key) return;
  const shape = JSON.stringify(fx.map((x) => [x.type, x.on]));
  const was = JSON.stringify(PFX.slots.map((x) => [x.type, x.on]));
  PFX.key = key; PFX.slots = JSON.parse(key);
  PFX.slots.forEach((x) => { if (typeof x.id === "number") PFX.uid = Math.max(PFX.uid, x.id); });
  PFX.sent = PFX.slots.map(fxId);
  if (shape !== was || !$("#fxrow").children.length) drawFx(); else PFX.ctl.forEach((c) => c.paint());
}

// the transport: the model in Synth, the tracks in Session
async function playModel() {
  if (!S.preset) return;
  $("#play").disabled = true;
  const body = { preset: S.preset.id, seed: Math.max(0, +$("#seed").value || 0) };
  if (S.engine === "mrt2") body.sampling = { guidance: +(S.sampling.guidance ?? 3), temperature: +(S.sampling.temperature ?? 1.3), top_k: +(S.sampling.top_k ?? 40) };
  else body.cfg = $("#cfg").checked ? (S.preset.cfg > 1 ? S.preset.cfg : 3) : 1;
  try { await api("/api/play", body); } catch (e) { fail(e); }
}
$("#play").addEventListener("click", () => (view !== "session" ? playModel() : deskCall("/api/desk/play")));
// Space plays or stops, as on a DAW's transport, except while typing in a field or a menu
document.addEventListener("keydown", (e) => {
  if (e.key !== " " || e.repeat || e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
  const t = e.target;
  if (t.closest?.("input, textarea, select, [contenteditable=''], [contenteditable='true'], .addmenu")) return;
  e.preventDefault();
  if (!$("#play").disabled) $("#play").click(); else if (!$("#stop").disabled) $("#stop").click();
});
$("#stop").addEventListener("click", async () => {
  if (view === "session") return deskCall("/api/desk/stop");
  try { const r = await api("/api/stop", {}); listSessions(); if (r.saved) info(`stopped; the session is saved as ${r.saved}`, "ok"); } catch (e) { fail(e); }
});
function transport() {
  const s = S.status;
  if (view !== "session") {
    $("#play").disabled = s.state !== "ready" || !!s.playing;
    $("#stop").disabled = !s.playing;
    $("#play").classList.toggle("on", !!s.playing);
  } else {
    const live = (D.state?.tracks || []).some((t) => t.live);
    $("#play").disabled = !(D.state?.tracks || []).some((t) => t.current !== null);
    $("#stop").disabled = !live && !D.state?.playing;
    $("#play").classList.toggle("on", live);
  }
  $("#snapbtn").disabled = !s.playing;
  $("#snapsave").disabled = !s.playing;
}

// saving sounds
async function saveLast() {
  try {
    const r = await api("/api/sounds/snapshot", { seconds: +$("#snaplen").value, name: "" });
    SND.list = r.sounds; SND.fresh = r.snapshot.snapshot; drawSounds();
    info(`saved “${r.snapshot.label}” (${r.snapshot.seconds} s)`, "ok");
  } catch (e) { fail(e); }
}
$("#snapsave").addEventListener("click", (e) => { if (e.target.tagName !== "SELECT") saveLast(); });
$("#snapbtn").addEventListener("click", saveLast);

// the scope: the last three seconds the model made; a click flips between the two channels' mix
// and left and right, one above the other
const sc = $("#scope"), scx = sc.getContext("2d");
let scopeLR = false, scopeLast = null;
try { scopeLR = localStorage.getItem("aa-scope-lr") === "1"; } catch (e) { /* private window */ }
function setScopeLR(on) {
  scopeLR = on;
  try { localStorage.setItem("aa-scope-lr", scopeLR ? "1" : "0"); } catch (e) { /* private window */ }
  if (typeof paintScopeMode === "function") paintScopeMode();
  if (scopeLast) drawScope(scopeLast.pairs, scopeLast.lr);
}
sc.addEventListener("click", () => { if (SPC.scope !== "spec") setScopeLR(!scopeLR); });
sc.title = "Click, or Mono and Stereo above: the two channels mixed, or left and right";
function trace(pairs, y0, hh, peak) {
  const n = pairs.length, w = sc.width;
  scx.save(); scx.translate(scopeOff, 0);
  const g = scx.createLinearGradient(0, y0, 0, y0 + hh);
  g.addColorStop(0, "#3fb6d9"); g.addColorStop(0.5, "#2a8fb3"); g.addColorStop(1, "#3fb6d9");
  scx.fillStyle = g;
  scx.beginPath();
  pairs.forEach(([, b], i) => scx.lineTo((i / (n - 1)) * w, y0 + hh / 2 - (b / peak) * hh * 0.45));
  for (let i = n - 1; i >= 0; i--) scx.lineTo((i / (n - 1)) * w, y0 + hh / 2 - (pairs[i][0] / peak) * hh * 0.45);
  scx.closePath(); scx.fill();
  scx.restore();
}
// cues for the eye: the older sound fades toward the left, and the right edge is marked as now
function scopeCues() {
  const w = sc.width, hh = sc.height;
  const f = scx.createLinearGradient(0, 0, w * 0.7, 0);
  f.addColorStop(0, "rgba(13,15,19,0.78)"); f.addColorStop(1, "rgba(13,15,19,0)");
  scx.fillStyle = f; scx.fillRect(0, 0, w * 0.7, hh);
  scx.fillStyle = "#f3a33b"; scx.fillRect(w - 4, 0, 4, hh);
  scx.font = "bold 17px ui-monospace, Menlo, monospace"; scx.textAlign = "right";
  scx.fillStyle = "rgba(13,15,19,.8)"; scx.fillRect(w - 62, hh - 28, 54, 22);
  scx.fillStyle = "#f3a33b"; scx.fillText("◀ now", w - 10, hh - 11); scx.textAlign = "left";
}
function drawScope(pairs, lr) {
  scopeLast = { pairs, lr };
  if (scopeLR && lr && lr.length === 2 && lr[0] && lr[0].length) {
    const w = sc.width, hh = sc.height, half = hh / 2;
    scx.clearRect(0, 0, w, hh);
    scx.strokeStyle = "#1c2028"; scx.lineWidth = 1;
    for (let k = 1; k < 6; k++) { const x = (w * k) / 6; scx.beginPath(); scx.moveTo(x, 0); scx.lineTo(x, hh); scx.stroke(); }
    scx.fillStyle = "#4a5160"; scx.font = "18px ui-monospace, Menlo, monospace";
    for (let k = 0; k < 6; k++) scx.fillText(`-${(3 - k * 0.5).toFixed(1)}s`, (w * k) / 6 + 6, 22);
    scx.strokeStyle = "#2a2f39";
    for (const y of [half / 2, half + half / 2]) { scx.beginPath(); scx.moveTo(0, y); scx.lineTo(w, y); scx.stroke(); }
    scx.strokeStyle = "#20242c"; scx.beginPath(); scx.moveTo(0, half); scx.lineTo(w, half); scx.stroke();
    const peak = Math.max(0.05, ...lr.flat().map(([a, b]) => Math.max(-a, b)));
    trace(lr[0], 0, half, peak); trace(lr[1], half, half, peak);
    scopeCues();
    scx.font = "bold 18px ui-monospace, Menlo, monospace";
    for (const [t, y] of [["L", half / 2], ["R", half + half / 2]]) { scx.fillStyle = "rgba(14,16,21,.55)"; scx.fillRect(4, y - 13, 22, 24); scx.fillStyle = "#5c6270"; scx.fillText(t, 9, y + 6); }
    return;
  }
  const w = sc.width, hh = sc.height;
  scx.clearRect(0, 0, w, hh);
  scx.strokeStyle = "#1c2028"; scx.lineWidth = 1;
  for (let k = 1; k < 6; k++) { const x = (w * k) / 6; scx.beginPath(); scx.moveTo(x, 0); scx.lineTo(x, hh); scx.stroke(); }
  scx.fillStyle = "#4a5160"; scx.font = "18px ui-monospace, Menlo, monospace";
  for (let k = 0; k < 6; k++) scx.fillText(`-${(3 - k * 0.5).toFixed(1)}s`, (w * k) / 6 + 6, 22);
  scx.strokeStyle = "#2a2f39"; scx.beginPath(); scx.moveTo(0, hh / 2); scx.lineTo(w, hh / 2); scx.stroke();
  if (!pairs || !pairs.length) return;
  trace(pairs, 0, hh, Math.max(0.05, ...pairs.map(([a, b]) => Math.max(-a, b))));
  scopeCues();
}
// Smooth scrolling: the engine sends the last three seconds at each reading (every 250 ms or so).
// The picture is drawn one reading behind and slides left between readings, so the sound moves
// steadily from right (now) to left (three seconds ago) instead of jumping a step at a time.
let scopeOff = 0, scopeAt = 0, scopeIv = 280;
function newScope(pairs, lr) {
  const t = performance.now();
  if (scopeAt) scopeIv = Math.min(600, Math.max(150, 0.8 * scopeIv + 0.2 * (t - scopeAt)));
  scopeAt = t; scopeLast = { pairs, lr };
}
requestAnimationFrame(function waveLoop() {                       // started on the next frame, once the page's state is all defined
  if (!document.hidden && view === "make" && SPC.scope === "wave" && scopeLast && S.playing) {
    const e = Math.min(performance.now() - scopeAt, scopeIv);
    scopeOff = ((scopeIv - e) / 3000) * sc.width;               // px still to travel before this reading reaches its place
    drawScope(scopeLast.pairs, scopeLast.lr);
  } else scopeOff = 0;
  requestAnimationFrame(waveLoop);
});

// Bend: the strips (VAU units inside the model)
const U = { units: [], chain: [], uid: 0, sentOn: null, noSnap: new Set() };   // chain: [{id, tk, layer, params, on, uid}]; noSnap: uids with snap-back off
// Modulation names a unit's knob by the strip's place among those playing (unit:<k>:<key>), as the
// engine counts them. When strips are switched off, removed or added, the routes follow their strips.
function followStrips() {
  const now = U.chain.filter((c) => c.on).map((c) => (c.uid ??= ++U.uid));
  const was = U.sentOn;
  U.sentOn = now;
  if (!was) return {};
  const ends = { anchor: now.filter((u) => !was.includes(u)), snap: was.filter((u) => !now.includes(u) && !U.noSnap.has(u)) };   // snap-back
  let moved = false;
  const routes = [];
  for (const r of MOD.spec.routes) {
    const m = /^unit:(\d+):(.+)$/.exec(r.tgt);
    if (!m) { routes.push(r); continue; }
    const k = now.indexOf(was[+m[1]]);
    if (k < 0) { moved = true; continue; }
    if (k !== +m[1]) moved = true;
    routes.push({ ...r, tgt: `unit:${k}:${m[2]}` });
  }
  if (moved) { MOD.spec.routes = routes; sendMod(); drawTiles(); }
  return ends;
}
async function loadUnits() {
  try {
    const j = await api("/api/units");
    U.units = (j.units || []).filter((u) => u.available);
    U.missing = (j.units || []).filter((u) => !u.available);
    const st = await api("/api/status");                 // the engine's chain, as it was before a reload
    U.chain = (st.chain || []).map((line) => {
      const [id, ...kv] = line.split(/\s+/), params = {};
      let layer = 6, tk = "layer";
      for (const x of kv) { const [k, v] = x.split("="); if (k === "layer") layer = +v; else if (k === "block") { tk = "block"; layer = v; } else params[k] = v; }
      const u = U.units.find((x) => x.id === id);
      if (u) for (const p of u.params) if (!(p.key in params)) params[p.key] = p.default ?? "";
      return { id, tk, layer, params, on: true };
    }).filter((c) => U.units.some((u) => u.id === c.id));
    U.sentOn = U.chain.filter((c) => c.on).map((c) => (c.uid ??= ++U.uid));
    SC.layers = (st.scopes || []).slice();
    drawStrips();
  } catch (e) { $("#stripdesk").innerHTML = `<div class="empty">${esc(e.message)}</div>`; }
}
const lineOf = (c) => [c.id, `${c.tk || "layer"}=${c.layer}`, ...Object.entries(c.params).map(([k, v]) => `${k}=${v}`)].join(" ");
// a strip's move re-attaches it (resetting its memory), so the chain goes at most every 150 ms
const sendChain = throttle(async () => {
  const ends = followStrips();
  try { const r = await api("/api/chain", { lines: U.chain.filter((c) => c.on).map(lineOf), anchor: ends.anchor || [], snap: ends.snap || [] }); showChainErrors(r.report || []); }
  catch (e) { info(e.message, "bad", 8000); }
}, 150);
// a strip the engine could not attach is marked on the desk with why; the others still play
function showChainErrors(report) {
  const on = U.chain.filter((c) => c.on);
  U.chain.forEach((c) => { c.error = null; });
  report.forEach((r, i) => { if (r.error && on[i]) on[i].error = r.error.replace(/^(Strip|Line) \d+: /, ""); });
  document.querySelectorAll("#stripdesk .strip").forEach((el, k) => {
    const c = U.chain[k], e = el.querySelector(".serr");
    el.classList.toggle("bad", !!(c && c.error));
    if (e) e.textContent = c && c.error ? c.error : "";
  });
  const bad = U.chain.filter((c) => c.error);
  if (bad.length) info(`${bad.length} unit${bad.length > 1 ? "s" : ""} not playing: ${bad[0].error}`, "bad", 6000);
}
function dropKnobs() { for (const k of MODKNOBS) if (!k.isConnected) MODKNOBS.delete(k); }    // after a panel is cleared
function drawStrips() {
  const desk = $("#stripdesk");
  desk.innerHTML = ""; dropKnobs();
  if (!U.chain.length && !SC.layers.length) desk.appendChild(h(`<div class="empty">${S.engine === "musicgen" ? "Units run on Magenta RT 2; MusicGen plays with Steer only" : "No VAU units"}</div>`));
  const unitMenu = (btn, slot) => menu(btn, [{ where: whereIn(slot, U.chain.length, "unit") },
    ...unitGroups().flatMap(([title, us]) => [{ head: title }, ...us.map((u) => ({ label: u.title, sub: u.what, run: () => addStrip(u, slot) }))])]);
  U.chain.forEach((c, k) => { if (U.units.length) desk.appendChild(gapAdd(k, k === 0 ? "Add a unit before unit 1" : `Add a unit between units ${k} and ${k + 1}`, unitMenu)); desk.appendChild(stripEl(c, k)); });
  if (U.units.length && SC.layers.length)                          // scopes sit after the units: a + after the last unit, before them
    desk.appendChild(gapAdd(U.chain.length, U.chain.length ? `Add a unit after unit ${U.chain.length}` : "Add the first unit", unitMenu));
  SC.layers.forEach((L, k) => desk.appendChild(scopeEl(L, k)));
  const add = h(`<button class="addend" title="Add a unit that acts on the model's stream at a layer (not an audio effect)" aria-haspopup="menu">+</button>`);
  add.addEventListener("click", () => menu(add, [
    ...unitGroups().flatMap(([title, us]) => [{ head: title }, ...us.map((u) => ({ label: u.title, sub: u.what, run: () => addStrip(u) }))]),
    { head: "Meters on the model's stream (they change nothing)" },
    { label: "Scope", sub: "draws the stream at a layer as it plays: its shape, its bands, and how it moves", run: addScope, disabled: SC.layers.length >= 4 },
    ...(U.missing?.length ? [{ head: "Not on this engine yet" }, ...U.missing.map((u) => ({ label: u.title, sub: `needs ${u.missing.join(", ")}`, disabled: true }))] : []),
  ]));
  if (!U.units.length) add.disabled = true;
  desk.appendChild(add);
}
// the units grouped by what they act on (units.space_of): along time, the stream's size, coordinate by
// coordinate, directions and bands, inside a block
const GROUP_ORDER = ["time", "size", "element", "subspace", "block", "other"];
function unitGroups() {
  const by = {};
  U.units.forEach((u) => { (by[u.group || "other"] = by[u.group || "other"] || { title: u.group_title || "VAU units", us: [] }).us.push(u); });
  return GROUP_ORDER.filter((g) => by[g]).map((g) => [by[g].title, by[g].us]);
}
function addStrip(u, slot = U.chain.length) {
  const params = {};
  for (const p of u.params) {
    let v = p.default ?? (p.kind === "direction" ? (S.dirs[0]?.name || "") : p.kind === "choice" ? p.choices[0] : "");
    if (p.kind === "direction" && !S.dirs.some((d) => d.name === v)) v = S.dirs[0]?.name || "";
    params[p.key] = v;
  }
  const tk = u.site === "output" ? "block" : "layer";
  const pd = u.place_default;                                   // measured: the nearest placement that holds up
  const layer = tk === "block" ? (pd === undefined || pd === -1 ? "all" : pd) : (pd === undefined ? 6 : pd);
  U.chain.splice(slot, 0, { id: u.id, tk, layer, params, on: true, fresh: true });
  drawStrips(); sendChain();
}
// Each strip's change to the stream (status.activity, by its line in the chain sent: the strips that are on,
// in order), as a dot that lights while it acts and a trace of the last ACT_N readings.
const ACT_N = 24;
const WIDE_UNITS = new Set(["neuron"]);                       // its cell needs the width
const actLevel = (v) => Math.max(0, Math.min(1, (Math.log10(Math.max(v, 1e-6)) + 3) / 3));    // 0.1% to 100%
function paintAct(c) {
  const a = c._act, hs = c._hist || [];
  if (!a) return;
  a.bars.forEach((b, j) => { const v = hs[hs.length - ACT_N + j]; b.style.height = v == null ? "0" : `${Math.max(v > 0 ? 1 : 0, actLevel(v) * 100)}%`; });
  const last = hs[hs.length - 1] || 0;
  if (!a.bars.length) {                                          // the header light: a slow green breath while it acts, quicker the more it changes
    const on = last > 1e-3, step = on ? Math.round(actLevel(last) * 3) : -1;
    a.dot.classList.toggle("live", on);
    if (step !== a.step) { a.step = step; a.dot.style.setProperty("--br", `${(3 - 0.66 * Math.max(0, step)).toFixed(2)}s`); }
    return;
  }
  a.dot.style.opacity = last > 1e-3 ? String(0.35 + 0.65 * actLevel(last)) : "0.12";
}
// Neuron's cell: the body swells with the value the unit leaves in the neuron it changes most (its value before,
// faint, inside), and pulses run down the axon at a rate set by how far that value is from its usual size.
function neuronAnim(c, box) {
  const el = box.appendChild(h(`<div class="sneu" title="The neuron this unit changes most, this frame: its value before (faint) and after (the body), and pulses at a rate set by how hard it is driven"><svg viewBox="0 0 120 46" aria-hidden="true">
    <g class="dend"><path d="M14 20 L4 10 M14 22 L2 22 M14 25 L5 36 M9 15 L6 7"/></g><path class="axon" d="M30 22 C 54 14, 78 30, 116 22"/>
    <circle class="soma" cx="20" cy="22" r="7"/><circle class="sin" cx="20" cy="22" r="3"/><g class="spk"></g></svg><small>·</small></div>`));
  const soma = el.querySelector(".soma"), sin = el.querySelector(".sin"), spk = el.querySelector(".spk"), axon = el.querySelector(".axon"), txt = el.querySelector("small");
  const len = axon.getTotalLength ? axon.getTotalLength() : 90;
  const N = { v: null, peak: 1e-6, lvl: 0, acc: 0, dots: [], t: performance.now() };
  c._neu = N;
  const frame = (now) => {
    if (!el.isConnected) return;
    const dt = Math.min(0.1, (now - N.t) / 1000); N.t = now;
    const v = N.v, live = !!v;
    if (live) { N.peak = Math.max(N.peak * 0.999, Math.abs(v[1]), Math.abs(v[2])); }
    const out = live ? Math.min(1, Math.abs(v[2]) / N.peak) : 0, inn = live ? Math.min(1, Math.abs(v[1]) / N.peak) : 0;
    N.lvl += (out - N.lvl) * Math.min(1, dt * 2.5);                  // about 0.4 s to follow, so it swells rather than flickers
    soma.setAttribute("r", (4 + 9 * N.lvl).toFixed(2)); soma.style.opacity = String(0.25 + 0.75 * N.lvl);
    sin.setAttribute("r", (1.5 + 9 * inn).toFixed(2));
    N.acc += dt * (live && !STILL ? 1 + 5 * N.lvl : 0);             // pulses a second: 1 to 6, none if the system asks for less motion
    while (N.acc >= 1) { N.acc -= 1; N.dots.push(0); }
    N.dots = N.dots.map((d) => d + dt / 0.9).filter((d) => d < 1);
    spk.innerHTML = N.dots.map((d) => { const p = axon.getPointAtLength ? axon.getPointAtLength(d * len) : { x: 30 + d * 86, y: 22 }; return `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="1.8" opacity="${(1 - d).toFixed(2)}"/>`; }).join("");
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
  N.show = (v) => { N.v = v; txt.textContent = v ? `n ${Math.round(v[0])}  ${numQ(v[1])} → ${numQ(v[2])}` : "·"; };
}
// Each unit's display, from status.probe (32 of the stream's values in and out at the strip, and |out| / |in|)
// and status.activity. A view is built once per strip and returns update(probe, activity); null while stopped.
const svgEl = (w, hgt, inner = "", fit = "none") => `<svg viewBox="0 0 ${w} ${hgt}" preserveAspectRatio="${fit}" aria-hidden="true">${inner}</svg>`;
const viewBox = (st, cls, tip, html) => st.appendChild(h(`<div class="uview ${cls}" title="${esc(tip)}">${html}</div>`));
const dbOf = (g) => 20 * Math.log10(Math.max(g, 1e-6));
function numberRows(st, tip, pick, n = 5) {                       // values in (grey) and out (amber), a row each
  const el = viewBox(st, "sq", tip, `${"<div><span>·</span><b>·</b></div>".repeat(n)}`);
  const rows = [...el.children].map((r) => [r.querySelector("span"), r.querySelector("b"), r]);
  let n0 = 0;
  return (p) => { if (!p || n0++ % 2) return; pick(p).forEach(([a, b], j) => { if (!rows[j]) return; rows[j][0].textContent = numQ(a); rows[j][1].textContent = numQ(b); rows[j][2].classList.toggle("hit", Math.abs(b) < Math.abs(a) - 1e-4); }); };
}
function grLeds(st, tip, bipolar) {                             // the stream's size after against before, in dB, on LEDs
  const n = bipolar ? 9 : 8;
  const el = viewBox(st, "sgr", tip, `<div class="leds">${"<i></i>".repeat(n)}</div><b>·</b>`);
  const cells = [...el.querySelectorAll("i")], txt = el.querySelector("b");
  let sm = 0;                                                     // held like a meter: quick to rise, about 0.6 s to fall
  return (p) => {
    const raw = p ? dbOf(p[2][0]) : 0;
    sm = Math.abs(raw) > Math.abs(sm) ? sm + (raw - sm) * 0.6 : sm + (raw - sm) * 0.2;
    const d = sm;
    txt.textContent = p ? `${d > 0 ? "+" : d < 0 ? "−" : ""}${Math.abs(d).toFixed(1)} dB` : "·";
    cells.forEach((c, j) => {
      if (!bipolar) { c.classList.toggle("on", p && -d >= 1.5 * (j + 1)); return; }
      const mid = 4, step = 1.5;
      c.classList.toggle("on", !!p && (j < mid ? d <= -step * (mid - j) : j > mid ? d >= step * (j - mid) : true));
    });
  };
}
const UNIT_VIEW = {
  erosion: (c, st) => numberRows(st, "Five of the stream's 1,024 values at this layer going in, and what the quantiser makes of them, this frame", (p) => p[0].slice(0, 5).map((a, j) => [a, p[1][j]])),
  limiter: (c, st) => numberRows(st, "The five largest of 32 of the stream's values going in, and what the clamp lets out; the rows it held back are marked", (p) => p[0].map((a, j) => [a, p[1][j]]).sort((x, y) => Math.abs(y[0]) - Math.abs(x[0])).slice(0, 5)),
  compressor: (c, st) => grLeds(st, "How much it shrinks the stream's size this frame: an LED for each 1.5 dB", false),
  multiband: (c, st) => grLeds(st, "How much it shrinks the stream's size this frame: an LED for each 1.5 dB", false),
  eq: (c, st) => grLeds(st, "The stream's size after the bands against before: cut to the left, boost to the right, 1.5 dB an LED", true),
  gate: (c, st) => {                                            // 32 values as pixels: lit by their size after, dark where zeroed
    const el = viewBox(st, "spx", "32 of the stream's values as pixels, lit by their size after the unit; the ones it set to zero are dark with an outline", `<div class="px">${"<i></i>".repeat(32)}</div><b>·</b>`);
    const px = [...el.querySelectorAll("i")], txt = el.querySelector("b");
    let n0 = 0;
    return (p) => {
      if (!p || n0++ % 2) return;
      const m = Math.max(1e-6, ...p[0].map(Math.abs));
      let kept = 0;
      px.forEach((x, j) => { const o = p[1][j], zero = Math.abs(o) < 1e-6 && Math.abs(p[0][j]) > 1e-6; kept += zero ? 0 : 1; x.classList.toggle("zero", zero); x.style.opacity = zero ? "" : String(0.15 + 0.85 * Math.min(1, Math.abs(o) / m)); });
      txt.textContent = `${kept} of 32 kept`;
    };
  },
  saturator: (c, st) => {                                       // the values out against in, the last ten frames, fading
    const el = viewBox(st, "sxy", "32 of the stream's values: out (up) against in (across), over the last ten frames; the bend is the squash", svgEl(100, 60, `<line class="ax" x1="50" y1="0" x2="50" y2="60"/><line class="ax" x1="0" y1="30" x2="100" y2="30"/><line class="id" x1="20" y1="60" x2="80" y2="0"/><g class="pts"></g>`));
    const g = el.querySelector(".pts"), hist = [];
    return (p) => {
      if (!p) return;
      hist.push(p); if (hist.length > 10) hist.shift();
      const m = Math.max(1e-6, ...hist.flatMap((q) => q[0].map(Math.abs)));
      g.innerHTML = hist.map((q, t) => q[0].map((a, j) => `<circle cx="${(50 + 50 * a / m).toFixed(1)}" cy="${(30 - 30 * q[1][j] / m).toFixed(1)}" r="1.4" opacity="${((t + 1) / hist.length).toFixed(2)}"/>`).join("")).join("");
    };
  },
  reverb: (c, st) => {                                          // one value over the last three seconds: in jagged, out smoothed
    const el = viewBox(st, "str", "One of the stream's values over the last few seconds: going in (grey) and after the average (amber)", svgEl(100, 40, `<polyline class="tin"/><polyline class="tout"/>`));
    const pin = el.querySelector(".tin"), pout = el.querySelector(".tout"), hin = [], hout = [];
    return (p) => {
      if (!p) return;
      hin.push(p[0][3]); hout.push(p[1][3]); if (hin.length > 48) { hin.shift(); hout.shift(); }
      const m = Math.max(1e-6, ...hin.map(Math.abs), ...hout.map(Math.abs));
      const pts = (a) => a.map((v, j) => `${(j * 100 / 47).toFixed(1)},${(20 - 18 * v / m).toFixed(1)}`).join(" ");
      pin.setAttribute("points", pts(hin)); pout.setAttribute("points", pts(hout));
    };
  },
  beatrepeat: (c, st) => {                                      // a lamp a reading: lit when it is playing a repeat
    const el = viewBox(st, "sstep", "Lights on each reading where the unit is repeating frames instead of passing the stream through", `<div class="leds">${"<i></i>".repeat(16)}</div>`);
    const cells = [...el.querySelectorAll("i")], hist = [];
    return (p, a) => { hist.push(a != null && a > 0.02); if (hist.length > 16) hist.shift(); cells.forEach((x, j) => x.classList.toggle("on", !!hist[hist.length - 16 + j])); };
  },
  delay: (c, st) => {                                           // the frames behind this one, the taps lit: Time back, and again for each pass of feedback
    const el = viewBox(st, "stape", "The last 32 frames, newest on the left: the frames it reads back (Time, then each pass of Feedback) light as it acts", `<div class="leds">${"<i></i>".repeat(32)}</div><b>·</b>`);
    const cells = [...el.querySelectorAll("i")], txt = el.querySelector("b");
    return (p, a) => {
      const t = Math.max(1, Math.round(+c.params.time || 1)), fb = +c.params.feedback || 0, lv = a == null ? 0 : actLevel(a);
      cells.forEach((x, j) => { const k = (j + 1) / t, tap = Number.isInteger(k); x.style.opacity = tap ? String(0.15 + 0.85 * lv * Math.pow(fb, k - 1)) : ""; x.classList.toggle("tap", tap); });
      txt.textContent = `${t} frames · ${t * 40} ms`;
    };
  },
  shifter: (c, st) => {                                         // the turn in the plane of the two directions; a point runs the arc while it acts
    const el = viewBox(st, "sdial", "The turn in the plane of the two directions: the arc is the angle, and a point runs along it while the unit acts", svgEl(60, 60, `<circle class="ring" cx="30" cy="30" r="22"/><line class="ax" x1="30" y1="30" x2="52" y2="30"/><path class="arc"/><circle class="run" r="2.4"/>`, "xMidYMid meet"));
    const arc = el.querySelector(".arc"), run = el.querySelector(".run");
    let ph = 0, t0 = performance.now(), lv = 0;
    const tick = (now) => {
      if (!el.isConnected) return;
      const dt = Math.min(0.1, (now - t0) / 1000); t0 = now;
      const ang = (+c.params.angle || 0) * Math.PI / 180, end = [30 + 22 * Math.cos(-ang), 30 + 22 * Math.sin(-ang)];
      arc.setAttribute("d", `M52 30 A22 22 0 ${Math.abs(ang) > Math.PI ? 1 : 0} ${ang > 0 ? 0 : 1} ${end[0].toFixed(1)} ${end[1].toFixed(1)}`);
      ph = STILL ? 1 : (ph + dt * (0.2 + 1.2 * lv)) % 1;
      const a = -ang * ph; run.setAttribute("cx", (30 + 22 * Math.cos(a)).toFixed(1)); run.setAttribute("cy", (30 + 22 * Math.sin(a)).toFixed(1)); run.style.opacity = String(0.2 + 0.8 * lv);
      requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
    return (p, a) => { lv = a == null ? 0 : actLevel(a); };
  },
};
const shortWord = (w) => (String(w).length > 7 ? String(w).slice(0, 6) + "." : String(w));   // a word on a knob, cut to fit its width
const numQ = (v) => (v < 0 ? "−" : "+") + Math.abs(v).toFixed(3);
// The Signal path panel, in Steer's old place: the stream through the model's twelve layers (Steer and each unit
// marked where it acts, lit by its reading), the codec, the model's sound and the effects, on to the Output panel
// beside it. Dots run along both paths while it plays. A stage scrolls to its panel when clicked.
const FLOW = { el: null, key: null };
function flowInit(N = 12, codec = "SpectroStream") {
  const host = $("#flow");
  if (!host) return;
  const dots = (path, n, dur, cls) => Array.from({ length: n }, (_, j) => `<circle class="${cls}" r="2"><animateMotion dur="${dur}s" repeatCount="indefinite" begin="${(-j * dur / n).toFixed(2)}s"><mpath href="#${path}"/></animateMotion></circle>`).join("");
  const X0 = 24, MW = 352;                                       // the model's box: twelve columns, a block each
  const cols = Array.from({ length: N + 1 }, (_, L) => `<line class="fcol" x1="${X0 + L * MW / N}" y1="24" x2="${X0 + L * MW / N}" y2="152"/>`).join("");
  FLOW.n = N; FLOW.codec = codec; FLOW.key = null;
  host.innerHTML = `<svg viewBox="0 0 400 292" aria-label="The signal path, live">
    <path id="fp1" class="fpath v" d="M2 88 H392 V180 H44 V208"/><path id="fp2" class="fpath a" d="M80 238 H396"/>
    <g class="fdots">${dots("fp1", 9, 6, "dv")}${dots("fp2", 7, 2.2, "da")}</g>
    <text class="fcap" id="fl-cap1" x="200" y="169">Model stream</text>
    <text class="fcap" id="fl-cap2" x="200" y="285">its sound</text>
    <g class="fnode fgo" data-go="#pn-engine"><title>From the Engine, beside this panel</title><path class="farr" d="M12 84 L18 88 L12 92Z"/></g><path class="farr" d="M40 196 L44 202 L48 196Z"/>
    <path class="farr" d="M96 234 L102 238 L96 242Z"/><path class="farr" d="M244 234 L250 238 L244 242Z"/>
    <g class="fnode" data-go="#pn-bend"><rect x="${X0}" y="24" width="${MW}" height="128" rx="5"/>${cols}<text class="fh" x="200" y="19">MODEL ${N} LAYERS - STEER${codec === "SpectroStream" ? " + MODEL UNITS" : ""}</text><rect id="fl-sweep" class="fsweep" x="${X0}" y="25" width="10" height="126"/><g id="fl-units"></g></g>
    <g class="fnode"><rect x="8" y="208" width="72" height="60" rx="5"/><text class="fh" x="44" y="232">CODEC</text><text class="fv" x="44" y="248">${codec}</text></g>
    <g class="fnode" data-go="#pn-engine"><rect x="108" y="208" width="128" height="60" rx="5"/><text class="fh" x="172" y="222">THE MODEL'S SOUND</text><polyline id="fl-wave" class="fwave"/></g>
    <g class="fnode" data-go="#pn-fx"><rect x="256" y="208" width="96" height="60" rx="5"/><text class="fh" x="304" y="222">EFFECTS</text><g id="fl-fx"></g><text class="fv" id="fl-fxn" x="304" y="256">·</text></g>
    <g class="fnode fgo" data-go="#pn-out"><title>On to the Output section, beside this panel</title><path class="farr" d="M390 234 L396 238 L390 242Z"/></g>
  </svg>`;
  host.querySelectorAll(".fnode[data-go]").forEach((g) => g.addEventListener("click", () => $(g.dataset.go)?.scrollIntoView({ behavior: "smooth", block: "center" })));
  FLOW.el = host;
  // a frame passing through the layers, slowed to 1.6 s a pass: each unit's dots flash as it passes, by the unit's
  // reading; Steer's diamond by the size of its doses
  const sweep = host.querySelector("#fl-sweep"), t0 = performance.now();
  const tick = (now) => {
    if (!host.isConnected) return;
    const live = FLOW.live && !STILL, x = X0 + ((now - t0) / 1600 % 1) * MW;
    sweep.style.opacity = live ? "1" : "0";
    sweep.setAttribute("x", (x - 5).toFixed(1));
    host.querySelectorAll(".fu").forEach((g) => {
      const lv = live ? +g.dataset.lv || 0 : 0;
      const pulse = live && g.classList.contains("hi") ? lv * (0.5 + 0.5 * Math.sin((now - t0) / 1400 * 2 * Math.PI)) : 0;   // the unit picked out pulses, by its reading
      g.querySelectorAll("circle").forEach((c) => { const f = Math.exp(-((((+c.getAttribute("cx")) - x) / 9) ** 2)) * lv; c.style.opacity = String(0.3 + 0.5 * lv + 0.2 * Math.max(f, pulse)); c.setAttribute("r", (3.2 + 2.6 * f + 1.6 * pulse).toFixed(2)); });
    });
    const cells = host.querySelectorAll("#fl-fx rect.on"), n = cells.length;   // the sound passing through the effects in turn
    cells.forEach((c, j) => { const w = 0.5 + 0.5 * Math.sin(2 * Math.PI * ((now - t0) / 1400 - j / Math.max(1, n))); c.style.opacity = String(live ? 0.7 + 0.3 * FLOW.lvl * w : 0.7); });
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}
// a lane's name, right-aligned at the lane's end on a dark backing drawn over the dots, so they cannot hide it
const laneTag = (x, y, t) => { const w = String(t).length * 4.6 + 6; return `<g class="flane"><rect x="${(x - w).toFixed(1)}" y="${(y - 5.5).toFixed(1)}" width="${w.toFixed(1)}" height="10" rx="2"/><text x="${(x - 3).toFixed(1)}" y="${(y + 2.5).toFixed(1)}">${esc(t)}</text></g>`; };
// A card under the pointer, or selected by a click on it, lights its lane in the Signal path and dims the others
function flowHi(k = FLOW.hover ?? FLOW.sel) {
  const g = FLOW.el?.querySelectorAll(".fu") || [];
  g.forEach((x) => { x.classList.toggle("hi", k != null && x.dataset.k === String(k)); x.classList.toggle("dim", k != null && x.dataset.k !== String(k)); });
  document.querySelectorAll(".strip[data-flow]").forEach((x) => x.classList.toggle("sel", FLOW.sel != null && x.dataset.flow === String(FLOW.sel)));
}
function flowLink(el, k) {
  el.dataset.flow = String(k);
  el.addEventListener("mouseenter", () => { FLOW.hover = k; flowHi(); });
  el.addEventListener("mouseleave", () => { FLOW.hover = null; flowHi(); });
  el.addEventListener("click", (e) => {
    if (e.target.closest("button, input, select, .knob, canvas, label, .sseg")) return;   // only a click on the card itself selects it
    FLOW.sel = FLOW.sel === k ? null : k; flowHi();
  });
}
function drawFlow(s) {
  const N = s.model?.n_layers || 12, codec = s.model?.codec || "SpectroStream";
  if (!FLOW.el || FLOW.n !== N || FLOW.codec !== codec) flowInit(N, codec);   // the engine's own shape
  const md = s.model || {}, fps = s.frame_ms ? Math.round(1000 / s.frame_ms) : 25;
  FLOW.el.querySelector("#fl-cap1").textContent = `Model stream: ${(md.d_model || 1024).toLocaleString("en-GB")} numbers/frame, ${fps} fps`;
  FLOW.el.querySelector("#fl-cap2").textContent = `its sound: ${Math.round((md.sample_rate || 48000) / 1000)} kHz ${(md.channels || 2) > 1 ? "stereo" : "mono"}`;
  const host = FLOW.el;
  if (!host || !s) return;
  const svg = host.querySelector("svg"), playing = !!s.playing;
  if (playing && !STILL) svg.unpauseAnimations(); else svg.pauseAnimations();
  host.classList.toggle("live", playing);
  // where Steer and the units act: a lane each, Steer's diamond at layer 6
  const W = 352 / N, xL = (L) => 24 + L * W, xB = (b) => 24 + (b + 0.5) * W, SL = s.model?.steer_layer ?? 6;
  const rep = (s.chain_report || []).filter((r) => !r.error), key = JSON.stringify(rep.map((r) => [r.title, r.at])) + s.steer_on + s.layers_on + SL + N;

  if (key !== FLOW.key) {
    FLOW.key = key;
    const n = rep.length + 1, lane = (k) => 28 + (k + 0.5) * 120 / n;   // Steer on the top lane, then a lane a unit
    host.querySelector("#fl-units").innerHTML = `<g class="fu fs ${s.steer_on === false ? "off" : ""}" data-k="steer"><title>Steer: layer ${SL}</title><circle cx="${xL(SL).toFixed(1)}" cy="${lane(0).toFixed(1)}" r="3.2"/>${laneTag(xL(N) - 2, lane(0), "Steer")}</g>`
      + rep.map((r, k) => `<g class="fu ${s.layers_on === false ? "off" : ""}" data-k="${k + 1}"><title>${esc(r.title)}: ${esc((r.at || []).join(", "))}</title>${(r.at || []).map((a) => { const b = /^block (\d+)$/.exec(String(a)); return `<circle cx="${(b ? xB(+b[1]) : xL(+a)).toFixed(1)}" cy="${lane(k + 1).toFixed(1)}" r="3.2"/>`; }).join("")}${laneTag(xL(N) - 2, lane(k + 1), r.short || r.title)}</g>`).join("");
  }
  flowHi();                                                       // the lanes were redrawn: keep the highlight
  const act = playing ? s.activity || {} : {};
  FLOW.live = playing;
  host.querySelectorAll(".fu:not(.fs)").forEach((g) => { const v = act[g.dataset.k]; g.dataset.lv = v ? actLevel(v).toFixed(3) : "0"; });
  const d = Object.values(s.doses || {}).reduce((m, v) => Math.max(m, Math.abs(+v || 0)), 0);
  FLOW.dose = s.steer_on === false ? 0 : Math.min(1, d / 0.15);
  const st = host.querySelector(".fu.fs");
  if (st) st.dataset.lv = FLOW.dose.toFixed(3);                  // Steer flashes and pulses by the size of its doses
  // the model's sound, its effects, the output section and OUT
  const sc = playing && s.scope ? s.scope : [];
  const pts = []; for (let j = 0; j < sc.length; j += 4) pts.push(`${(114 + j * 116 / Math.max(1, sc.length - 1)).toFixed(1)},${(244 - Math.max(-1, Math.min(1, sc[j][1] * 4)) * 14).toFixed(1)}`);
  host.querySelector("#fl-wave").setAttribute("points", pts.join(" ") || "114,244 230,244");
  FLOW.lvl = playing && s.level > 0 ? Math.max(0, Math.min(1, (20 * Math.log10(s.level) + 48) / 42)) : 0;
  const fx = (s.master && s.master.fx) || [], fxOn = !(s.master && s.master.fx_on === false), m = Math.max(1, fx.length);
  host.querySelector("#fl-fx").innerHTML = fx.map((x, j) => `<rect class="${x.on && fxOn ? "on" : ""}" x="${(264 + j * 80 / m).toFixed(1)}" y="232" width="${Math.max(3, 80 / m - 2).toFixed(1)}" height="9" rx="1"><title>${esc(x.type)}</title></rect>`).join("");
  host.querySelector("#fl-fxn").textContent = !fxOn ? "bypassed" : fx.length ? `${fx.filter((x) => x.on).length} of ${fx.length} on` : "none";
}
function paintActivity(s) {
  const act = (s.playing && s.layers_on !== false && s.activity) || null;
  U.chain.filter((c) => c.on).forEach((c, k) => {
    c._hist = (c._hist || []).concat([act ? act[k + 1] || 0 : null]).slice(-ACT_N);
    paintAct(c);
    const p = act && s.probe && s.probe[k + 1];
    if (c._neu) c._neu.show(p ? [p[0][0], p[0][1], p[1][1]] : null);
    if (c._view) c._view(p || null, act ? act[k + 1] || 0 : null);
  });
}
function stripEl(c, k) {
  const u = U.units.find((x) => x.id === c.id);
  const nonKnob = u.params.filter((p) => p.kind === "toggle" && !p.hidden).length, knobs = 1 + u.params.filter((p) => p.kind !== "toggle" && p.kind !== "text" && !p.hidden).length;
  const wide = nonKnob >= 3 || knobs > 6 || WIDE_UNITS.has(c.id);           // menus and switches in two columns, or room for a display
  const st = h(`<div class="strip ${c.on ? "" : "off"} ${wide ? "wide" : ""}"><div class="sh">
    <button class="led" aria-pressed="${c.on}" title="On or bypassed"></button><span class="st" title="${esc((u.short !== u.title ? `${u.title}. ` : "") + u.what + (u.note ? "\n" + u.note : ""))}">${esc(u.short || u.title)}</span></div></div>`);
  c.fresh = false;
  const uid = (c.uid ??= ++U.uid);
  {                                                            // the card's menu, by its name: presets, then the card's own switches
    const mb = h(`<button class="spre" title="Presets and settings for this card" aria-label="Card menu" aria-haspopup="menu">▾</button>`);
    const apply = (q) => { Object.assign(c.params, q.params); if (q.block !== undefined) c.layer = q.block; drawStrips(); sendChain(); };
    mb.addEventListener("click", () => menu(mb, [
      ...presetItems(u.presets, MY.units[c.id], apply, (q) => delMine("units", c.id, q)),
      ...mineItems("units", c.id, () => ({ ...c.params }), () => c.layer),
      { head: "Card" },
      { label: "Snap-back", check: !U.noSnap.has(uid), sub: "Return the model to its pre-unit state immediately when the unit is switched off",
        title: "On: switching this unit off or removing it returns the model to the state it had when the unit went on, at the next frame. Off: the unit's sound lingers for a while, since the model reads back about 20 s of its own frames.",
        run: () => { U.noSnap.has(uid) ? U.noSnap.delete(uid) : U.noSnap.add(uid); } },
      { sep: true },
      { label: "Remove", danger: true, title: "Or select the card and press Delete", run: () => removeUnit(c) },
    ], "cardmenu"));
    st.querySelector(".st").after(mb);
  }
  selectable(st, c, () => removeUnit(c));
  st.appendChild(h(`<div class="serr"></div>`));
  if (c.error) { st.classList.add("bad"); st.querySelector(".serr").textContent = c.error; }
  st.querySelector(".led").addEventListener("click", () => { c.on = !c.on; drawStrips(); sendChain(); });
  st.querySelector(".st").title += "\nDrag to move it along the row: units at the same layer act in this order.";
  sortable(st, st.querySelector(".st"), $("#stripdesk"), k, (from, to) => { moved(U.chain, from, to); drawStrips(); sendChain(); });
  const opts = c.tk === "block" ? ["all", ...Array.from({ length: 12 }, (_, j) => String(j))] : Array.from({ length: 12 }, (_, j) => String(j + 1));
  const places = u.places, measured = (o) => (c.tk === "block" ? (o === "all" ? -1 : +o) : +o);
  const swept = CAL.controls[`unit:${c.id}:at`]?.steps || [];
  const mark = (o) => (places && swept.includes(measured(o)) && !places.includes(measured(o)) ? " ⚠" : "");
  const at = () => Math.max(0, opts.indexOf(String(c.layer)));
  const place = knob({ label: c.tk === "block" ? "Block" : "Layer", min: 0, max: opts.length - 1, step: 1, int: true, reset: at(),
    get: at, put: (v) => { const o = opts[Math.round(v)]; if (o !== String(c.layer)) { c.layer = c.tk === "block" ? o : +o; sendChain(); } },
    fmt: (v) => opts[Math.round(v)] + mark(opts[Math.round(v)]),
    tip: `Where it acts: ${c.tk === "block" ? "a transformer block, or all twelve" : "between layers; at 6 it changes what layers 6 to 11 read"}. ⚠: the measures found the sound breaks with this unit here` });
  const box = wide ? st.appendChild(h(`<div class="sgrid"></div>`)) : st;
  const krow = h(`<div class="krow"></div>`);
  krow.appendChild(place);
  for (const p of u.params) {
    if (p.hidden) continue;                                       // cannot change the sound on this model (units.HOST_HIDDEN)
    const redraw = u.params.some((q) => q.when?.key === p.key);   // another setting's dimming depends on this one
    const set = (v) => { c.params[p.key] = v; sendChain(); if (redraw) drawStrips(); };
    const k0 = krow.children.length, b0 = box.children.length;
    const words = p.kind === "choice" ? p.choices.map(String) : p.kind === "direction" && S.dirs.length ? S.dirs.map((d) => d.name) : null;
    if (p.kind === "direction" && words && words.length >= 2 && words.length <= 4) {   // a few directions: one button each, the chosen one lit
      const g = h(`<div class="dseg" role="group" aria-label="${esc(p.label)}" title="${esc(p.tip || "")}"><span class="dsl">${esc(p.label)}</span><div class="dsb">${words.map((x) =>
        `<button type="button" data-v="${esc(x)}" title="${esc(x)}" aria-pressed="${x === String(c.params[p.key])}">${esc(x.charAt(0).toUpperCase() + x.slice(1))}</button>`).join("")}</div></div>`);
      g.querySelectorAll("button").forEach((btn) => btn.addEventListener("click", () => {
        if (btn.dataset.v === String(c.params[p.key])) return;
        g.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === btn)));
        set(btn.dataset.v);
      }));
      box.appendChild(g);
    } else if (words) {                                                  // a list of words: a stepped knob that reads the word
      const ch = words, at = () => Math.max(0, ch.indexOf(String(c.params[p.key])));
      krow.appendChild(knob({ label: p.label, min: 0, max: Math.max(1, ch.length - 1), step: 1, int: true, reset: Math.max(0, ch.indexOf(String(p.default ?? ch[0]))),
        get: at, put: (v) => { const x = ch[Math.min(ch.length - 1, Math.round(v))]; if (x !== String(c.params[p.key])) set(x); }, fmt: (v) => shortWord(ch[Math.min(ch.length - 1, Math.round(v))]), tip: p.tip || "" }));
    } else if (p.kind === "choice" || p.kind === "direction") {
      const ch = p.kind === "direction" ? S.dirs.map((d) => d.name) : p.choices.map(String);
      const s = h(`<label class="sl"><span>${esc(p.label)}</span><select class="sel" title="${esc(p.tip || "")}">${ch.map((x) => `<option ${x === String(c.params[p.key]) ? "selected" : ""}>${esc(x)}</option>`).join("")}</select></label>`);
      s.querySelector("select").addEventListener("change", (e) => set(e.target.value));
      box.appendChild(s);
    } else if (p.kind === "toggle") {
      const t = toggle(p.label, () => !!+c.params[p.key], (v) => set(v ? 1 : 0));
      t.title = p.tip || ""; box.appendChild(t);
    } else if (p.kind === "text" && c.id === "neuron" && p.key === "n" && S.status?.mlp_width && /^\s*\d+\s*$/.test(String(c.params[p.key]))) {
      const W = S.status.mlp_width;                               // one neuron: a knob over the MLP's hidden units; a list stays text
      krow.appendChild(knob({ label: p.label, min: 0, max: W - 1, step: 1, int: true, reset: +(p.default ?? 0), get: () => +c.params[p.key],
        put: (v) => { const n = String(Math.round(v)); if (n !== String(c.params[p.key])) set(n); }, fmt: (v) => String(Math.round(v)),
        tip: `${p.tip || ""}; one of ${W} hidden units` }));
    } else if (p.kind === "text") {
      const s = h(`<label class="sl"><span>${esc(p.label)}</span><input class="rawargs" value="${esc(c.params[p.key])}" title="${esc(p.tip || "")}"></label>`);
      s.querySelector("input").addEventListener("change", (e) => set(e.target.value));
      box.appendChild(s);
    } else {
      const k = U.chain.filter((x) => x.on).indexOf(c), tg = c.on && k >= 0 ? `unit:${k}:${p.key}` : null;
      if (tg) MODTGT[tg] = { label: `${u.short || u.title} ${p.label}`, min: +p.min, max: +p.max, log: false };
      krow.appendChild(knob({ label: p.label + (p.fragile ? " ⚠" : ""), min: +p.min, max: +p.max, step: p.step || (p.kind === "int" ? 1 : (p.max - p.min) / 200), int: p.kind === "int",
        reset: +(p.default ?? p.min), get: () => +c.params[p.key], put: set, curve: p.taper ? undefined : `unit:${c.id}:${p.key}`, taper: p.taper, mod: tg || undefined,
        fmt: (v) => (p.kind === "int" ? String(Math.round(v)) : Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2)),
        tip: (p.tip || "") + (p.fragile ? ". ⚠ Measured: away from this setting the sound breaks on at least one preset; listen in Tune" : "") }));
    }
    if (p.when && !p.when.in.includes(String(c.params[p.when.key]))) {   // dimmed: acts only for some values of another setting
      const el = krow.children.length > k0 ? krow.lastElementChild : box.children.length > b0 ? box.lastElementChild : null;
      if (el) { el.classList.add("unit-inert"); el.title = `${el.title ? el.title + ". " : ""}${p.when.why}`; }
    }
  }
  if (wide && krow.children.length > 6) { krow.classList.add("two"); krow.style.maxWidth = `${Math.ceil(krow.children.length / 2) * 46}px`; }
  if (wide && box.children.length % 2 && krow.children.length && krow.children.length <= 2) box.appendChild(krow);   // into the free cell
  else if (krow.children.length) st.appendChild(krow);
  c._act = c._view = null;
  if (c.on) flowLink(st, U.chain.filter((x) => x.on).indexOf(c) + 1);
  if (c.on && c.id === "neuron") neuronAnim(c, st);               // the unit's own live display
  else if (c.on && UNIT_VIEW[c.id]) c._view = UNIT_VIEW[c.id](c, st);
  else if (c.on) {                                              // no display of its own: a light in the header that breathes while it acts
    const led = h(`<i class="sled" title="Breathes while this unit changes the model's stream: quicker the more it changes it"></i>`);
    st.querySelector(".sh").appendChild(led);
    c._act = { dot: led, bars: [] };
    paintAct(c);
  }
  const q = h(`<button class="qhelp" type="button" aria-label="Help: ${esc(u.title)}" title="What ${esc(u.title)} does, with a figure and its parameters">?</button>`);
  q.addEventListener("click", (e) => { e.stopPropagation(); unitHelp(u, q); });
  st.appendChild(q);
  return st;
}
// A unit's help, as Abliterton Lens shows it: what it does, the animated figure VAU ships for it
// (vau.explain, served from /vau/explain/ and loaded the first time it is needed), its parameters
// with their tips, and VAU's own text; then where it sits here, and its sweep on the site.
const EXPLAIN = "/vau/explain/", SITE = "https://vector-lab-tools.github.io/abliterator/", VAUSITE = "https://vector-lab-tools.github.io/vector-abliteration-units/";
const loadingX = {};
function loadX(src, css) {
  if (!loadingX[src]) loadingX[src] = new Promise((ok, no) => {
    const el = css ? Object.assign(document.createElement("link"), { rel: "stylesheet", href: src }) : Object.assign(document.createElement("script"), { src });
    el.onload = ok; el.onerror = () => { delete loadingX[src]; no(new Error(src)); };
    document.head.appendChild(el);
  });
  return loadingX[src];
}
const numX = (v) => (typeof v === "number" ? String(+v.toFixed(4)).replace("-", "−") : esc(String(v)));
function paramRange(p) {
  if (p.kind === "choice") return (p.choices || []).map(esc).join(", ");
  if (p.kind === "direction") return "a direction";
  if (p.kind === "toggle") return "on or off";
  if (p.kind === "text") return p.hint ? esc(p.hint) : "text";
  return p.min !== undefined ? `${numX(+p.min)} to ${numX(+p.max)}` : "";
}
let helpFig = null;
async function unitHelp(u, opener) {
  const box = $("#unithelp"), body = box.querySelector(".uhbody");
  box.querySelector(".uht").innerHTML = `<b>${esc(u.title)}</b>${u.vau_title && u.vau_title !== u.title ? `<span>VAU: ${esc(u.vau_title)}</span>` : ""}`;
  const rows = (u.params || []).map((p) => `<tr><td><b>${esc(p.label)}</b> <code>${esc(p.key)}</code></td><td>${paramRange(p)}${p.default !== undefined && p.default !== null && p.default !== "" ? `<br><span class="uh-dim">default ${numX(typeof p.default === "number" ? p.default : p.default)}${p.neutral !== undefined ? `, neutral ${numX(p.neutral)}` : ""}</span>` : ""}</td><td>${p.tip ? `<code class="uh-tip">${esc(p.tip)}</code>` : ""}${p.fragile ? `<br><span class="uh-warn">⚠ measured: away from its default the sound breaks on at least one preset</span>` : ""}</td></tr>`).join("");
  body.innerHTML = `<p class="uh-what">${esc(u.what)}.</p>
    ${u.explain ? `<div class="uh-fig"><p class="uh-dim">Drawing the figure…</p></div>` : ""}
    ${u.group_title ? `<p class="uh-where"><b>${esc(u.group_title)}.</b> ${u.note ? esc(u.note) : ""}</p>` : ""}
    ${u.host_note ? `<h4>On this instrument</h4><p class="uh-help">${esc(u.host_note)}</p>` : ""}
    <h4>The ▾ menu by its name</h4>
    <p class="uh-help">The unit's presets, then your own under User Presets: type a name in Save preset… to keep the card's settings, and × deletes one. Then Snap-back, on or off, and Remove. A card clicked is selected, and Delete or Backspace removes it.</p>
    ${u.presets?.length ? `<h4>Presets</h4><ul class="uh-presets">${u.presets.map((q) => `<li><b>${esc(q.name)}</b>: ${esc(q.tip || "")}</li>`).join("")}</ul><p class="uh-dim">${u.id === "abliterate" ? "Each keeps the direction you chose." : "Each was kept only if it broke the sound on none of ten seeds the search did not use. Its tip names what moved on those seeds beyond chance."}</p>` : ""}
    <h4>Snap-back (on the ▾ menu)</h4>
    <p class="uh-help">On by default. The model writes each frame by reading back about 20 seconds of the frames it made, so a unit's sound would linger after it goes off. With snap-back, the model's state is kept when you switch the unit on. When you switch it off or remove it, the model returns to that state, crossfaded over one 40 ms frame, and goes on as if the unit had never played. With snap-back off, its sound fades over up to 20 seconds. A saved sound replays it as heard.</p>
    <h4>Order on the row</h4>
    <p class="uh-help">The order of the cards matters only among units on the same layer. Each unit acts at its own layer or block, wherever its card sits: Lookback at layer 6 and Tanh Squash at layer 10 do the same in either order. Units on the same layer run left to right, and at layer 6 after Steer. Where their edits do not commute, the order changes the sound: Tanh Squash then Norm Shrink is not Norm Shrink then Tanh Squash, and even two Principal Bands cards give a different take in each order, since the stream is rounded after each unit. Lookback and Running Average record the stream as it reaches them: a unit before them on that layer goes into what they repeat, and a unit after them acts on their output without being repeated. Neuron and Abliterate act inside the blocks, so their place among the stream units makes no difference. Drag a card by its name to move it. Measured on seven pairs over ten seeds, the two orders on a shared layer never gave the same audio, and three of the five pairs moved level, brightness or spectral shape the same way on most seeds. Tech specs has the figures.</p>
    <h4>Parameters</h4>
    ${rows ? `<table class="uh-params"><thead><tr><th>Parameter</th><th>Range</th><th>What it does</th></tr></thead><tbody>${rows}</tbody></table>` : `<p class="uh-dim">None.</p>`}
    ${u.example ? `<p class="uh-ex">Example line: <code>${esc(u.example)}</code></p>` : ""}
    ${u.inspired_by ? `<p class="uh-insp"><b>Inspired by</b> ${esc(u.inspired_by)}.</p>` : ""}
    ${u.help ? `<h4>More</h4><p class="uh-help">${u.help}</p>` : ""}
    <p class="uh-links"><a href="${SITE}units.html#u-${encodeURIComponent(u.id)}" target="_blank" rel="noopener">Its measured sweep, to listen to ↗</a> · <a href="${VAUSITE}units.html#${encodeURIComponent(u.id)}" target="_blank" rel="noopener">On the VAU site ↗</a></p>
    <p class="uh-src uh-dim">A VAU unit${u.file ? ` · ${esc(u.file)}` : ""}${u.version ? `, version ${esc(u.version)}` : ""}${u.author ? `, ${esc(u.author)}` : ""}</p>`;
  box.hidden = false; box.dataset.opener = "";
  box._opener = opener;
  box.querySelector(".uhclose").focus();
  if (helpFig) { try { helpFig.destroy(); } catch (e) { /* gone */ } helpFig = null; }
  const slot = body.querySelector(".uh-fig");
  if (!slot) return;
  try {
    await loadX(EXPLAIN + "explain.css", true);
    await loadX(EXPLAIN + "explain.js");
    await loadX(EXPLAIN + encodeURIComponent(u.id) + ".explain.js");
    if (!box.hidden && window.VAUExplain && VAUExplain.has(u.id)) helpFig = VAUExplain.mount(slot, u.id, { title: u.title, examples: false });
    else slot.remove();
  } catch (e) { slot.innerHTML = `<p class="uh-dim">The figure did not load (${esc(e.message)}).</p>`; }
}
// What each effect does, for its help box (?): how it works, each control, and where the Effects page says more
const FX_HELP = {
  phaser: {"at": "phaser", "how": "A chain of 4, 8 or 12 allpass filters, all at one frequency, which an LFO sweeps up to two octaves either side of Centre. Each filter shifts the phase and leaves the level alone. Added to the dry sound, the shifted copy cancels it where the chain's phase is an odd multiple of 180°: 2, 4 or 6 notches, which move with the sweep. Feedback sends the chain's output back to its input and raises peaks between the notches. The right side's LFO runs Spread degrees behind the left.", "p": {"rate": "how fast the sweep goes, 0.02 to 5 Hz", "depth": "how far the sweep goes: at 1, two octaves either side of Centre", "centre": "the middle of the sweep, 100 to 4000 Hz", "feedback": "how much of the chain's output goes back to its input, 0 to 0.9: higher gives sharper peaks", "stages": "4, 8 or 12 filters: 2, 4 or 6 notches", "spread": "how far the right side's sweep runs behind the left, 0 to 180°", "mix": "the swept copy against the dry sound; at 1 they are equal, for the deepest notches"}},
  flanger: {"at": "flanger", "how": "The sound delayed by 0.3 to 20 ms, the delay swept by an LFO, and added to the dry sound. The copy cancels the dry sound at every frequency where it arrives half a cycle late, so the notches sit at odd multiples of 1 / (2 × delay), evenly spaced, and move as the delay changes. Feedback sends the delayed sound back into the line and raises peaks between the notches; negative feedback puts the peaks where the notches were. The right side's LFO runs Spread degrees behind the left.", "p": {"rate": "how fast the sweep goes, 0.02 to 5 Hz", "depth": "how far above Delay the sweep goes, 0 to 10 ms", "delay": "the shortest delay, where the sweep starts, 0.3 to 10 ms", "feedback": "how much of the delayed sound goes back into the line, -0.95 to 0.95", "spread": "how far the right side's sweep runs behind the left, 0 to 180°", "mix": "the delayed copy against the dry sound; at 1 they are equal, for the deepest notches"}},
  tremolo: {"at": "tremolo", "how": "The level swung by an LFO. At its top the sound passes at unity; at its bottom it is turned down by Depth, to silence at 1. Square switches between the two with 5 ms edges, so it does not click. The right side runs Phase degrees behind the left; at 180° the sides take turns.", "p": {"rate": "how many times a second the level swings, 0.1 to 20 Hz", "depth": "how far the level falls: 0 none, 0.5 to half, 1 to silence", "shape": "Sine, Triangle or Square", "phase": "how far the right side runs behind the left, 0 to 180°"}},
  compressor: {"at": "compressor", "how": "A feed-forward compressor, the same gain on both sides. It reads the louder side's peak in dB. Above the Threshold the level rises by 1 dB for every Ratio dB in, and the Knee bends the curve over that many dB around the threshold. The gain falls over the Attack time and recovers over the Release time. A meter shows the most gain reduction since it last looked.", "p": {"thresh": "the level above which it acts, -50 to 0 dBFS", "ratio": "how many dB in for each dB out above the threshold; 20:1 acts as a limiter", "attack": "how fast the gain falls when the level rises", "release": "how fast the gain comes back when the level falls", "knee": "how many dB the curve takes to bend at the threshold; 0 is a sharp corner", "makeup": "gain after the compressor, up to 24 dB", "mix": "the compressed sound against the dry: below 1 it is parallel compression"}},
  resonator: {"at": "resonator", "how": "Up to four comb filters, delay lines fed back into themselves, each one period of a note long, so it rings at that note and its harmonics. The root sets the lowest note as a MIDI number (60 is middle C), and the Chord adds the notes above it. A low-pass in each loop makes the high harmonics die sooner, and an allpass sets the fraction of a sample that keeps each note in tune. The input is summed to mono; the notes are spread across the stereo field.", "p": {"root": "the lowest note, as a MIDI number from 24 (C1, 33 Hz) to 84 (C6, 1047 Hz)", "chord": "Uni (the root alone), 5th, Oct (octave), Maj (major), Min (minor) or Sus4, above the root", "decay": "how long each note takes to fall 60 dB, 0.1 to 10 s", "bright": "how long the high harmonics last: at 0 they die 100 times as fast as the note, at 1 as slowly", "mix": "the ringing against the dry sound"}},
  freqshift: {"at": "freqshift", "how": "Two chains of allpass filters make two copies of the sound 90 degrees apart. Multiplied by a sine and a cosine at the Shift frequency and combined, they give the sound moved up or down by that many hertz, with the mirror image cancelled. Every partial moves by the same amount, so harmonics stop lining up and the sound turns metallic; a shift of a few hertz mixed with the dry sound sweeps like a phaser.", "p": {"shift": "how many hertz every partial moves, -1000 to +1000", "spread": "moves the right side's shift the other way: at 1 the right goes down as far as the left goes up", "feedback": "sends the shifted sound back in through a 5 ms delay, so it is shifted again on each pass", "mix": "the shifted sound against the dry"}},
  pitch: {"at": "pitch", "how": "Pitch reads the sound back from a delay line through two heads. Each head sweeps its delay at the rate that gives the shift, then jumps back to the start of the window; the two heads are half a window apart and crossfade, so a head is silent when it jumps. Feedback sends the shifted sound back into the line, so each repeat is shifted again. The shifted sound is at least 5 ms late; the dry sound is not delayed.", "p": {"shift": "the shift in semitones, from two octaves down to two octaves up", "fine": "a further shift of up to 50 cents either way", "window": "how far each head sweeps, 20 to 200 ms: long windows suit low notes, short ones fast attacks", "feedback": "how much of the shifted sound goes back into the line, up to 0.9", "mix": "0 is the dry sound only, 1 the shifted sound only"}},
  shimmer: {"at": "shimmer", "how": "Shimmer is a reverb of eight delay lines, 44 to 100 ms long, mixed into each other on every pass. Two of the lines are read through a pitch shifter, so on each pass part of the tail moves by the interval and the tail climbs (or falls) as it decays. Every path round the loop has a gain below 1, so the tail always dies away.", "p": {"decay": "the time the tail takes to fall by 60 dB, 0.5 to 20 s", "shimmer": "how much of the two shifted lines is the shifted read: 0 is a plain reverb", "interval": "the shift on each pass, in semitones", "damp": "the high-frequency loss in each line: 0 keeps the top, 1 rolls off above 1.5 kHz", "mix": "0 is the dry sound only, 1 the reverb only"}},
  freeze: {"at": "freeze", "how": "Freeze measures the sound's spectrum every 21 ms over 85 ms windows. When you switch it on it keeps the average of the last few windows and plays that spectrum without end, with new random phases every 21 ms. It fades in and out over 100 ms. When it is off the sound passes unchanged. Its phases are drawn at random, so a held pure tone wanders in level by up to about 4 dB over a fifth of a second; noise-like sound holds within 0.5 dB.", "p": {"freeze": "on holds the sound; off lets it go", "blur": "how many windows, 21 ms apart, are averaged into the hold: 1 is a single 85 ms window, 32 covers 0.73 s", "drift": "slow random rises and falls of 24 frequency bands in the hold, up to about 10 dB, at the same total level", "mix": "the hold's share of the output while it is on"}},
  trim: { at: "trim", how: "No sound of its own: it sets the level and the stereo field before the next effect. In order: In picks Stereo, the left side (L) or the right side (R) on both sides, or swaps them (Swap); Ø L and Ø R flip each side's phase; Width scales the side against the mid (0% is mono, 100% as it came, 400% four times as wide), and Mono sets it to 0% to check how the sound holds up in mono; Bass mono takes the side out below Bass below, so the low end is the same on both sides, and Hear bass plays only what is below that frequency; DC takes out a constant offset; then Gain, Balance and Mute. Every switch moves over 10 ms, so none of them clicks.",
    p: { gain: "the level after everything else, −36 to +24 dB", balance: "turns the far side down; the near side stays at unity", width: "the side against the mid: 0% mono, 100% unchanged, up to 400%", chans: "Stereo; L or R on both sides; or Swap", phl: "flips the left side's phase", phr: "flips the right side's phase", mono: "width 0%: hear the sound in mono", bass: "takes the side out below Bass below, a 24 dB-an-octave high-pass on the side", bassf: "where Bass mono and Hear bass split the sound, 50 to 500 Hz", solo: "only the mid below Bass below, to hear what Bass mono keeps", dc: "a 5 Hz high-pass that takes out a constant offset", mute: "silence, faded over 10 ms" } },
  reverb: { at: "each", how: "A feedback delay network after Signalsmith's “Let's write a reverb”. The sound is pre-delayed, smeared by four allpass diffusers, then fed into eight delay lines, about 35 to 80 ms long, whose outputs are mixed by a Householder matrix and fed back, each through a low-pass. Each line falls 60 dB over the Decay time. Left and right are taken from alternate lines, so the tail is wide.",
    p: { size: "how long the tail takes to fall 60 dB", damp: "how dark the tail grows: the low-pass in each line, from 20 kHz at 0 to 1.5 kHz at 1", predelay: "a gap before the tail begins", mix: "the tail against the dry sound" } },
  plate: { at: "each", how: "Dattorro's 1997 plate tank: two halves, each a modulated allpass, a delay, damping and another allpass, each feeding the other, read out at several taps. Its allpasses are modulated and read with cubic interpolation, so the tail does not ring. Colour sets the era: 2020s is clean; 1980s is a darker tank with noisier modulation; 1970s runs the tank at half rate behind a 10 kHz band limit.",
    p: { colour: "2020s, 1980s or 1970s, as above", decay: "how long the tail lasts", size: "the tank's lengths and taps, scaled: a smaller or larger plate", damp: "how dark the tail grows", predelay: "a gap before the tail begins", mod: "how far the modulated allpasses wander", mix: "the tail against the dry sound" } },
  hall: { at: "each", how: "The reverb's network with longer lines, each line's delay wandering along its own smooth random path: a new target every so often, joined by a raised-cosine curve, drawn from the sample position, so it is the same at any block size. It has the plate's three colours.",
    p: { colour: "2020s, 1980s or 1970s: clean, darker with noisier movement, or at half rate behind a 10 kHz band limit", size: "how long the tail takes to fall 60 dB", damp: "how dark the tail grows", predelay: "a gap before the tail begins", mod: "how far each line's delay wanders", mix: "the tail against the dry sound" } },
  delay: { at: "each", how: "A delay line with a low-pass in its feedback loop, so each repeat is darker than the one before. With Ping-pong the input enters on the left and the repeats cross from side to side.",
    p: { time: "the time between repeats", feedback: "how much of each repeat comes round again", damp: "the low-pass in the loop, from 20 kHz at 0 to 1.5 kHz at 1", pingpong: "repeats thrown from side to side", mix: "the repeats against the dry sound; at full Mix the dry sound is at half" } },
  chorus: { at: "each", how: "Two copies of the sound delayed by about 12 ms, the delay swept by a sine, the right a quarter cycle behind the left, so the copies drift against the dry sound and widen it.",
    p: { rate: "how fast the delay swings", depth: "how far the delay swings", mix: "the copies against the dry sound" } },
  drive: { at: "each", how: "tanh saturation, gain-matched so that Drive changes the colour more than the level, then a one-pole low-pass. A small meter shows how hard the curve is driven, with a CLIP mark.",
    p: { amount: "how hard the sound is pushed into the curve", tone: "the low-pass after it", mix: "the driven sound against the dry" } },
  tape: { at: "tape", how: "A tape machine's colour. Saturation through a slightly lopsided tanh (the tilt adds even harmonics), gain-matched at −26 dBFS, about the model's usual level, so Drive changes the colour more than the level and louder passages are compressed, as on tape. Wow and flutter come from a delay line whose length wobbles. The tape speed sets a head bump in the lows and a roll-off in the highs; hiss is read from a fixed table by sample position. A meter shows how hard the tape is driven.",
    p: { drive: "how hard the tape is driven, 1 to 16 times into the curve", wow: "slow pitch wobble, two slow sines at unrelated rates (about ±2% at 1)", flutter: "faster wobble near 7 Hz (about ±1.3% at 1)", noise: "tape hiss, from −84 to −50 dBFS", mix: "the tape against the dry sound, delayed to match so they do not comb", speed: "3.75, 7.5, 15 or 30 inches a second: the slower, the larger and lower the head bump and the earlier the treble falls" } },
  eq: { at: "each", how: "Three RBJ biquads: a low shelf at 200 Hz, a bell at the frequency you set, and a high shelf at 5 kHz. It draws its response curve.",
    p: { low: "the shelf below 200 Hz, cut or boost", mid: "the bell, cut or boost", freq: "the bell's centre", high: "the shelf above 5 kHz, cut or boost" } },
  eq8: { at: "eq8", how: "A Butterworth low cut, six bells at fixed centres (60 Hz, 150 Hz, 400 Hz, 1 kHz, 2.5 kHz and 6 kHz, each about 1.2 octaves wide) and a Butterworth high cut. It draws its response curve.",
    p: { lowcut: "everything below this falls away", b60: "the bell at 60 Hz", b150: "the bell at 150 Hz", b400: "the bell at 400 Hz", b1k: "the bell at 1 kHz", b2k5: "the bell at 2.5 kHz", b6k: "the bell at 6 kHz", highcut: "everything above this falls away", slope: "how steep the cuts are, 12 or 24 dB an octave" } },
  filter: { at: "each", how: "One RBJ biquad, low-pass, high-pass or band-pass. It draws its response curve.",
    p: { mode: "low-pass, high-pass or band-pass", cutoff: "where it acts", res: "how sharp the peak at the cutoff is (Q)" } },
  crush: { at: "each", how: "A bit crusher. Each sample is held for a number of samples, counted from the start of the sound, which lowers the sample rate and folds high frequencies back down, then rounded to fewer bits.",
    p: { bits: "how many bits each sample keeps", hold: "how many samples each one is held for", mix: "the crushed sound against the dry" } },
  abliterated: { at: "abliterated", how: "Heavy distortion written for this instrument. Steer sets where it bites: a resonant boost of 12 dB before the clipping, after a low cut at 90 Hz that keeps the lows tight. Crunch drives two tanh stages, each tilted to one side and the second the other way, with a high-pass at 140 Hz between them. Edge blends in a sine wavefolder. Complicate adds an octave up, then a ring modulator. Tone is a two-pole low-pass standing in for a speaker cabinet. A make-up gain follows the input's power against the output's over 300 ms, so the controls change the character more than the level.",
    p: { crunch: "how hard the two stages are driven, from about 2× to 300× (+50 dB) in all", edge: "the wavefolder, folding the clipped wave back on itself up to seven times: it rings like metal", steer: "where the distortion bites: the boost before it", complicate: "over its first half an octave up (the wave rectified), over its second a ring modulator at 1.5 times Steer, whose sum and difference tones clang", tone: "the cabinet low-pass", level: "a trim after the make-up gain", mix: "the distortion against the dry sound: parallel distortion" } },
  pan: { at: "pan", how: "An equal-power balance: at the centre both sides pass at unity, at either end one side is silent and the other 3 dB up, so the sound keeps its loudness as it moves. Auto rate and depth swing it on a sine whose phase carries on from block to block. A mono sound passes as it is.",
    p: { pos: "where the sound sits, left (−1) to right (+1)", rate: "how fast the auto-pan swings", depth: "how far it swings" } },
  declick: { at: "each", how: "The Output section's De-click as an effect, so it can sit where you want it, before a reverb for example. It finds the few-sample bursts a unit pushed hard can make the model write and redraws each as a line between the good samples either side, 1.3 ms late. It counts the bursts it has redrawn.",
    p: { sens: "how readily it acts: at 0 a burst must stand 12 times above the local level, at 0.5 six times (as Output's), at 1 three times", longest: "the longest burst it redraws" } },
};
function fxRange(q) {
  if (q.scale && q.scale.startsWith("choice:")) return q.scale.slice(7).split(",").join(", ");
  if (q.scale === "toggle") return "off or on";
  return `${numX(q.min)} to ${numX(q.max)}${q.unit ? " " + esc(q.unit) : ""}`;
}
function fxHelp(type, opener) {
  const T = PFX.types[type], H = FX_HELP[type] || {}, box = $("#unithelp"), body = box.querySelector(".uhbody");
  if (!T) return;
  box.querySelector(".uht").innerHTML = `<b>${esc(T.label)}</b><span>an effect</span>`;
  const dflt = (q) => (q.scale && q.scale.startsWith("choice:") ? q.scale.slice(7).split(",")[Math.round(q.default)] : q.scale === "toggle" ? (q.default ? "on" : "off") : `${numX(q.default)}${q.unit ? " " + esc(q.unit) : ""}`);
  const rows = T.params.map((q) => `<tr><td><b>${esc(q.label)}</b> <code>${esc(q.key)}</code></td><td>${fxRange(q)}<br><span class="uh-dim">default ${dflt(q)}</span></td><td>${esc(H.p?.[q.key] || "")}</td></tr>`).join("");
  body.innerHTML = `<p class="uh-what">${esc((FX_WHAT[type] || "").replace(/^./, (c) => c.toUpperCase()))}.</p>
    ${H.how ? `<h4>How it works</h4><p class="uh-help">${esc(H.how)}</p>` : ""}
    <h4>Parameters</h4>
    <table class="uh-params"><thead><tr><th>Parameter</th><th>Range</th><th>What it does</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="uh-help">Every effect here is written for this instrument in numpy and scipy, and gives the same output whatever block size the sound card asks for, so a saved sound renders as it was heard. Switching it on or off crossfades it; the light on its card bypasses it, and its name drags it to another place. The ▾ menu by its name lists its presets and your own (Save preset… keeps the card's settings, × deletes one) and removes it; a card clicked is selected, and Delete or Backspace removes it too. Each knob takes modulation.</p>
    <p class="uh-links"><a href="${SITE}effects.html#${H.at || "each"}" target="_blank" rel="noopener">More on the Effects page ↗</a></p>`;
  box.hidden = false;
  box._opener = opener;
  box.querySelector(".uhclose").focus();
  if (helpFig) { try { helpFig.destroy(); } catch (e) { /* gone */ } helpFig = null; }
}
function closeHelp() {
  const box = $("#unithelp");
  if (box.hidden) return;
  box.hidden = true;
  if (helpFig) { try { helpFig.destroy(); } catch (e) { /* gone */ } helpFig = null; }
  if (box._opener && box._opener.isConnected) box._opener.focus();
}
// Spectrograms of what is heard (spectro.py): the engine's and the tracks' columns arrive with each
// status, a column every 20 ms, 150 log bands from 40 Hz to 20 kHz on the website's scale and in its
// colours. Each lane keeps the last 600 columns (12 s) in an offscreen canvas, and is drawn about
// 300 ms behind the newest column, moving smoothly between the polls.
const SPC = { scope: "wave", lanes: {} };
try { SPC.scope = localStorage.getItem("aa-scope") || "wave"; } catch (e) { /* private window */ }
const SPC_K = 600, SPC_LAG = 15;
const SPC_LUT = (() => {
  const stops = [[0, [7, 9, 12]], [0.3, [32, 30, 92]], [0.55, [123, 47, 190]], [0.78, [227, 164, 60]], [1, [255, 248, 228]]], lut = new Uint8ClampedArray(256 * 3);
  for (let v = 0; v < 256; v++) {
    const t = v / 255; let i = 1; while (i < stops.length - 1 && t > stops[i][0]) i++;
    const a = stops[i - 1], b = stops[i], k = Math.max(0, Math.min(1, (t - a[0]) / (b[0] - a[0])));
    for (let c = 0; c < 3; c++) lut[v * 3 + c] = a[1][c] + k * (b[1][c] - a[1][c]);
  }
  return lut;
})();
function specLane(kind) {
  if (!SPC.lanes[kind]) {
    const off = document.createElement("canvas"); off.width = SPC_K; off.height = 150;
    const g = off.getContext("2d"); g.fillStyle = "#07090c"; g.fillRect(0, 0, SPC_K, 150);
    SPC.lanes[kind] = { off, g, end: 0, t: 0, hop: 20, rows: 150, playing: false };
  }
  return SPC.lanes[kind];
}
function adoptSpec(kind, sp, playing) {
  const L = specLane(kind);
  L.playing = !!playing;
  if (!sp) return;
  if (sp.rows !== L.rows) { L.rows = sp.rows; L.off.height = sp.rows; }
  if (sp.end < L.end) { L.g.fillStyle = "#07090c"; L.g.fillRect(0, 0, SPC_K, L.rows); L.end = 0; }   // a new play
  const fresh = Math.min(sp.n, sp.end - L.end);
  if (fresh > 0) {
    const bin = atob(sp.data), rows = sp.rows, first = sp.n - fresh;
    L.g.drawImage(L.off, -fresh, 0);                                 // the older columns move left
    const img = L.g.createImageData(fresh, rows);
    for (let c = 0; c < fresh; c++) for (let r = 0; r < rows; r++) {
      const v = bin.charCodeAt((first + c) * rows + r), o = ((rows - 1 - r) * fresh + c) * 4;
      img.data[o] = SPC_LUT[v * 3]; img.data[o + 1] = SPC_LUT[v * 3 + 1]; img.data[o + 2] = SPC_LUT[v * 3 + 2]; img.data[o + 3] = 255;
    }
    L.g.putImageData(img, SPC_K - fresh, 0);
    L.end = sp.end; L.t = performance.now(); L.hop = sp.hop_ms || 20;
  }
}
function drawSpec(kind, cv, cols) {
  const L = SPC.lanes[kind], g = cv.getContext("2d"), W = cv.width, H = cv.height;
  g.fillStyle = "#07090c"; g.fillRect(0, 0, W, H);
  if (!L || !L.end) return false;
  const behind = L.playing ? Math.max(0, SPC_LAG - (performance.now() - L.t) / L.hop) : 0;   // how far the drawn edge trails the newest column
  const sx = SPC_K - behind - cols;
  g.imageSmoothingEnabled = true;
  g.drawImage(L.off, sx, 0, cols, L.rows, 0, 0, W, H);
  return true;
}
function paintScopeMode() {
  document.querySelectorAll(".scopemode button[data-sm]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.sm === SPC.scope));
  document.querySelectorAll(".chmode button").forEach((b) => b.setAttribute("aria-pressed", (b.dataset.ch === "stereo") === scopeLR));
  const ch = document.querySelector(".chmode"); if (ch) ch.hidden = SPC.scope === "spec";
}
document.querySelectorAll(".chmode button").forEach((b) => b.addEventListener("click", () => setScopeLR(b.dataset.ch === "stereo")));
document.querySelectorAll(".scopemode button[data-sm]").forEach((b) => b.addEventListener("click", () => {
  SPC.scope = b.dataset.sm; paintScopeMode();
  try { localStorage.setItem("aa-scope", SPC.scope); } catch (e) { /* private window */ }
  if (SPC.scope === "wave") drawScope(S.status?.scope, S.status?.scope_lr);
}));
paintScopeMode();
(function specLoop() {
  if (!document.hidden) {
    if (view === "make" && SPC.scope === "spec") drawSpec("model", $("#scope"), 150);
    if (view === "listen") {
      [["model", "#lane-model"], ["desk", "#lane-desk"]].forEach(([k, id]) => {
        const lane = $(id), cv = lane.querySelector("canvas"), L = SPC.lanes[k];
        const had = drawSpec(k, cv, 500);
        lane.classList.toggle("live", !!(L && L.playing));
        lane.querySelector(".lanestate").textContent = L && L.playing ? (k === "model" ? `playing ${S.status?.preset || ""} · seed ${S.status?.seed ?? ""}` : "playing") : had ? "stopped" : "not playing";
      });
    }
  }
  requestAnimationFrame(specLoop);
})();

// The scope: a meter on the model path (scope.py). It reads the stream at its layer, after the
// Steer and the units there, once a frame, and changes nothing. The engine keeps the last 5 s; each
// poll brings the newest frames' traces, played out here at one per 40 ms.
const SC = { layers: [], mode: {}, queue: {}, glow: {}, lastFrame: {}, meters: {}, next: 0 };
const SC_COL = { brightness: "#f3a33b", pulse: "#b28cff" };
const sendScopes = throttle(async () => {
  try { const r = await api("/api/scopes", { layers: SC.layers }); SC.layers = r.scopes; } catch (e) { fail(e); }
}, 150);
function addScope() {
  const used = new Set(SC.layers), L = [6, 9, 3, 12, 1, 2, 4, 5, 7, 8, 10, 11].find((x) => !used.has(x));
  SC.layers.push(L); SC.mode[L] = SC.mode[L] || "trace";
  drawStrips(); sendScopes();
}
function scopeEl(L, k) {
  const mode = SC.mode[L] || (SC.mode[L] = "trace");
  const st = h(`<div class="strip scopestrip"><div class="sh"><span class="st" title="Reads the model's stream at its layer, after the Steer and the units there, once a frame; changes nothing and is not saved with sounds">Scope<span class="meterbadge">meter</span></span><button class="spre" title="This scope's menu" aria-label="Scope menu" aria-haspopup="menu">▾</button></div>
    <div class="scopebar"><label class="sl"><span>Layer</span><select class="sel">${Array.from({ length: 12 }, (_, j) => `<option ${j + 1 === L ? "selected" : ""}>${j + 1}</option>`).join("")}</select></label>
    <div class="seg">${[["trace", "Trace", "the stream along its principal directions, the largest first; 0 is the ordinary stream, ±1 a typical swing"], ["bands", "Bands", "how far the stream swings in bands of its principal directions; the line is its ordinary level"], ["time", "Time", "the last 5 s: size against typical (white), the reading along each direction on the Steer's dose scale (colours), and how far the stream turns each frame (green)"]].map(([m, l, t]) => `<button data-m="${m}" aria-pressed="${m === mode}" title="${t}">${l}</button>`).join("")}</div></div>
    <canvas class="scopecv" width="464" height="256"></canvas><div class="scoperead"></div></div>`);
  const sm = st.querySelector(".spre");
  sm.addEventListener("click", () => menu(sm, [{ head: "Scope" }, { label: "Remove", danger: true, run: () => { SC.layers.splice(k, 1); drawStrips(); sendScopes(); } }], "cardmenu"));
  st.querySelector("select").addEventListener("change", (e) => {
    const v = +e.target.value;
    if (SC.layers.includes(v)) { info(`a scope already reads layer ${v}`, "bad", 3000); e.target.value = L; return; }
    SC.layers[k] = v; SC.mode[v] = mode; drawStrips(); sendScopes();
  });
  st.querySelectorAll(".seg button").forEach((b) => b.addEventListener("click", () => { SC.mode[L] = b.dataset.m; drawStrips(); }));
  st.dataset.layer = L;
  selectable(st, `scope:${L}`, () => { const j = SC.layers.indexOf(L); if (j >= 0) { SC.layers.splice(j, 1); drawStrips(); sendScopes(); } });
  return st;
}
function adoptMeters(meters) {
  SC.meters = meters || {};
  for (const [L, m] of Object.entries(SC.meters)) {
    const q = (SC.queue[L] = SC.queue[L] || []), last = SC.lastFrame[L] ?? -1;
    if (m.frame < last) SC.lastFrame[L] = -1;                 // a new play
    m.frames.forEach((f, i) => { if (f > (SC.lastFrame[L] ?? -1)) q.push(m.traces[i]); });
    SC.lastFrame[L] = m.frame;
    if (q.length > 12) q.splice(0, q.length - 12);            // behind: drop the oldest
  }
}
function drawScopes(now) {
  const els = document.querySelectorAll("#stripdesk .scopestrip");
  if (!els.length) return;
  const step = now >= SC.next;
  if (step) SC.next = now + 40;
  els.forEach((el) => {
    const L = el.dataset.layer, cv = el.querySelector("canvas"), g = cv.getContext("2d"), W = 232, H = 128, m = SC.meters[L];
    g.setTransform(cv.width / W, 0, 0, cv.height / H, 0, 0);
    const mode = SC.mode[L] || "trace", glow = (SC.glow[L] = SC.glow[L] || []);
    if (step && SC.queue[L]?.length) { glow.push(SC.queue[L].shift()); if (glow.length > 9) glow.shift(); }
    g.fillStyle = "#07090c"; g.fillRect(0, 0, W, H);
    g.strokeStyle = "rgba(63,182,217,.13)"; g.lineWidth = 1;
    for (let x = 0; x <= W; x += W / 8) { g.beginPath(); g.moveTo(Math.round(x) + 0.5, 0); g.lineTo(Math.round(x) + 0.5, H); g.stroke(); }
    for (let y = 0; y <= H; y += H / 4) { g.beginPath(); g.moveTo(0, Math.round(y) + 0.5); g.lineTo(W, Math.round(y) + 0.5); g.stroke(); }
    g.font = "9px ui-monospace, Menlo, monospace"; g.fillStyle = "#5d6472";
    const read = el.querySelector(".scoperead");
    if (!m) {
      g.strokeStyle = "rgba(63,182,217,.5)"; g.beginPath(); g.moveTo(0, H / 2); g.lineTo(W, H / 2); g.stroke();
      g.fillText(S.playing ? "waiting for the first frames" : "reads while the model plays", 6, 12);
      read.textContent = `layer ${L}`;
      return;
    }
    if (mode === "trace") {
      glow.forEach((tr, i) => {
        const a = (i + 1) / glow.length, newest = i === glow.length - 1;
        g.beginPath();
        tr.forEach((z, j) => { const x = (j / (tr.length - 1)) * W, y = H / 2 - (z / 6) * (H / 2 - 4); j ? g.lineTo(x, y) : g.moveTo(x, y); });
        g.strokeStyle = newest ? "#8fe6ff" : `rgba(63,182,217,${(a * 0.45).toFixed(2)})`;
        g.lineWidth = newest ? 1.4 : 1; g.shadowColor = "#3fb6d9"; g.shadowBlur = newest ? 8 : 0; g.stroke(); g.shadowBlur = 0;
      });
      g.fillText("largest directions →", 6, H - 5);
    } else if (mode === "bands") {
      const b = m.bands, n = b.length, bw = W / n, top = 3, yOf = (v) => H - 4 - Math.min(1, Math.log1p(v) / Math.log1p(top)) * (H - 14);
      SC.peak = SC.peak || {}; const pk = (SC.peak[L] = SC.peak[L] || []);
      b.forEach((v, i) => {
        pk[i] = Math.max(v, (pk[i] || 0) * 0.985);
        const y = yOf(v), grad = g.createLinearGradient(0, H, 0, y);
        grad.addColorStop(0, "#1f6f8a"); grad.addColorStop(1, v > 1.5 ? "#f3a33b" : "#3fb6d9");
        g.fillStyle = grad; g.fillRect(i * bw + 1, y, bw - 2, H - 4 - y);
        g.fillStyle = "#e9eaee"; g.fillRect(i * bw + 1, yOf(pk[i]) - 1, bw - 2, 1.5);
      });
      g.strokeStyle = "rgba(233,234,238,.55)"; g.setLineDash([3, 3]); g.beginPath(); g.moveTo(0, yOf(1)); g.lineTo(W, yOf(1)); g.stroke(); g.setLineDash([]);
      g.fillStyle = "#5d6472"; g.fillText("ordinary", W - 46, yOf(1) - 3);
    } else {
      const n = m.size.length, X = (i) => (i / Math.max(1, 124)) * W + (W - (n - 1) / 124 * W);
      const line = (arr, f, col, w = 1.2) => { g.beginPath(); arr.forEach((v, i) => { const y = f(v); i ? g.lineTo(X(i), y) : g.moveTo(X(i), y); }); g.strokeStyle = col; g.lineWidth = w; g.stroke(); };
      const mid = H * 0.42, span = H * 0.36;
      g.strokeStyle = "rgba(233,234,238,.25)"; g.beginPath(); g.moveTo(0, mid); g.lineTo(W, mid); g.stroke();
      line(m.size, (v) => mid - (v - 1) * span * 2, "rgba(233,234,238,.8)");
      Object.entries(m.along).forEach(([k, arr]) => line(arr, (v) => mid - (v / 0.8) * span, SC_COL[k] || "#9fb4d9", 1.5));
      line(m.motion, (v) => H - 3 - Math.min(1, v / 0.2) * (H * 0.18), "#4ccf6a", 1);
      g.fillStyle = "#5d6472"; g.fillText("5 s", 4, H - 5); g.fillText("now", W - 22, H - 5);
    }
    const al = Object.entries(m.along).map(([k, a]) => `<span style="color:${SC_COL[k] || "#9fb4d9"}">${esc(k)} ${a[a.length - 1] >= 0 ? "+" : "−"}${Math.abs(a[a.length - 1]).toFixed(2)}</span>`).join(" ");
    read.innerHTML = `layer ${L} · size ${m.size[m.size.length - 1].toFixed(2)}×<br>${al}`;
  });
}
(function scopeLoop(now) { drawScopes(now || 0); requestAnimationFrame(scopeLoop); })();

let unitsFor = null;
setInterval(() => { if (S.engine && S.engine !== unitsFor && S.dirs.length) { unitsFor = S.engine; loadUnits(); } }, 500);

// ------------------------------------------------------------- modulation: the tiles, the editor, the macros
function drawTiles() {
  const box = $("#modstrip");
  if (!box.children.length) {
    for (const id in SRC) {
      const s = SRC[id];
      const el = h(`<button class="mt" data-s="${id}" style="--c:${s.color}" title="${esc(s.label)}: drag onto a knob to link it, or click to assign it to knobs; double-click to edit it"><b>${esc(s.label)}</b><span></span><canvas width="240" height="84"></canvas></button>`);
      el.addEventListener("click", () => { if (el.dragged) { el.dragged = false; return; } if (MOD.on) assign(id); else info("Modulation is written for Magenta RT 2.", "bad"); });
      el.addEventListener("dblclick", () => { MOD.tab = tabOf(s.kind); drawEditor(); });
      dragSource(el, id); hintOn(el);
      box.appendChild(el);
    }
  }
  for (const el of box.children) {
    const id = el.dataset.s, n = MOD.spec.routes.filter((r) => r.src === id).length;
    el.classList.toggle("sel", MOD.assign === id); el.classList.toggle("used", n > 0); el.classList.toggle("off", !MOD.on);
    if (SRC[id].kind === "macro") el.hidden = !n && MOD.assign !== id;      // a macro's tile shows once it moves something; take one from the Macros tab
    if (id === "env2" || id === "env3") el.hidden = !n && MOD.assign !== id;   // one envelope shown; the other two from the Envelope tab, once they move something
    el.querySelector("span").textContent = n ? `→ ${n}` : "";
  }
}
// Drag and drop: a source's tile dragged over the page carries a small badge of its shape; dropped
// on a knob that modulation can move, it links the source to it (START_SEND, or the link's own) and
// selects the source, so dragging the knob then sets how far it moves it.
function glyph(id) {
  const k = SRC[id].kind, p = MOD.spec.sources[id] || {}, N = 32, W = 30, H = 14;
  let f;
  if (k === "lfo") f = (x) => lfoV({ ...p, phase: 0 }, x / (p.rate_hz || 1)) * 0.5 + 0.5;
  else if (k === "rand") f = (x) => randV({ ...p, smooth: Math.min(p.smooth ?? 0.5, 0.6) }, x * 4 / (p.rate_hz || 1)) * 0.5 + 0.5;
  else if (k === "env") { const L = (p.attack_s || 0) + (p.decay_s || 0) + 4 + (p.release_s || 0); f = (x) => envV(p, x * L, L); }
  else if (k === "steps") { const n = Math.round(p.length || 8); f = (x) => stepsV({ ...p, glide: 0, rate_hz: 1, amount: 1 }, Math.min(x, 0.999) * n) * 0.5 + 0.5; }   // one cycle
  else if (k === "drift") f = (x) => driftV(p, x * 4 / (p.rate_hz || 0.05)) * 0.5 + 0.5;
  else if (k === "chaos") f = (x) => chaosV(p, x * 4 / (p.rate_hz || 0.2)) * 0.5 + 0.5;
  else if (k === "follow") f = (x) => Math.max(0, Math.sin(x * 9) * Math.exp(-x * 1.5));
  else return `<svg viewBox="0 0 ${W} ${H}" aria-hidden="true"><circle cx="15" cy="7" r="5.5" fill="none" stroke="currentColor" stroke-width="1.6"/><line x1="15" y1="7" x2="15" y2="2.6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`;
  const pts = Array.from({ length: N + 1 }, (_, i) => `${((i / N) * W).toFixed(1)},${(H - 1 - f(i / N) * (H - 2)).toFixed(1)}`).join(" ");
  return `<svg viewBox="0 0 ${W} ${H}" aria-hidden="true"><polyline points="${pts}" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linejoin="round" stroke-linecap="round"/></svg>`;
}
function dragSource(el, id) {
  el.addEventListener("pointerdown", (e) => {
    if (e.button !== 0) return;
    const x0 = e.clientX, y0 = e.clientY;
    let ghost = null, over = null;
    const knobAt = (x, y) => document.elementFromPoint(x, y)?.closest(".knob[data-mod]") || null;
    const move = (ev) => {
      if (!ghost) {
        if (Math.hypot(ev.clientX - x0, ev.clientY - y0) < 6) return;
        if (!MOD.on) { info("Modulation is written for Magenta RT 2.", "bad"); end(); return; }
        ghost = h(`<div class="modghost" style="--c:${SRC[id].color}">${glyph(id)}<b>${esc(SRC[id].label)}</b></div>`);
        document.body.appendChild(ghost);
        document.body.classList.add("moddrag");
        document.body.style.setProperty("--dragc", SRC[id].color);
      }
      ghost.style.transform = `translate(${ev.clientX + 10}px, ${ev.clientY + 10}px)`;
      const k = knobAt(ev.clientX, ev.clientY);
      if (k !== over) { over?.classList.remove("drophover"); over = k; over?.classList.add("drophover"); }
    };
    const end = () => {
      el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); el.removeEventListener("pointercancel", end);
      ghost?.remove(); document.body.classList.remove("moddrag"); over?.classList.remove("drophover");
    };
    const up = (ev) => {
      const k = ghost ? knobAt(ev.clientX, ev.clientY) : null;
      if (ghost) { el.dragged = true; setTimeout(() => { el.dragged = false; }, 60); }     // the click that ends a drag is not a click
      end();
      if (!k) return;
      const tgt = k.dataset.mod, d = routeDepth(id, tgt) || START_SEND;
      setRoute(id, tgt, d);
      if (MOD.assign !== id) assign(id);
      MODKNOBS.forEach((x) => x.paintMod());
      info(`${SRC[id].label} → ${tgtOf(tgt)?.label || tgt}: depth ${d.toFixed(2)}. Drag the knob to change it; double-click it to remove the link.`, "ok", 6000);
    };
    el.setPointerCapture(e.pointerId);
    el.addEventListener("pointermove", move); el.addEventListener("pointerup", up); el.addEventListener("pointercancel", end);
  });
}
function traceTiles() {
  const t = modTime();
  for (const el of $("#modstrip").children) {
    const id = el.dataset.s, c = el.querySelector("canvas"), g = c.getContext("2d"), w = c.width, hh = c.height, col = CSSCOL[SRC[id].kind];
    g.clearRect(0, 0, w, hh);
    if (!MOD.spec.sources[id]) continue;
    const bip = SRC[id].bipolar, win = SRC[id].kind === "env" ? 30 : SRC[id].kind === "drift" ? 40 : 8, N = 120;
    const y = (v) => (bip ? hh / 2 - v * hh * 0.4 : hh * 0.9 - v * hh * 0.8);
    g.fillStyle = col + "22"; g.strokeStyle = col; g.lineWidth = 3;
    g.beginPath();
    for (let i = 0; i <= N; i++) {
      const tt = SRC[id].kind === "env" ? (MOD.playing ? t - win + (i / N) * win : (i / N) * win) : t - win + (i / N) * win;
      const v = srcV(id, tt);
      i ? g.lineTo((i / N) * w, y(v)) : g.moveTo(0, y(v));
    }
    g.stroke();
    g.lineTo(w, bip ? hh / 2 : hh * 0.9); g.lineTo(0, bip ? hh / 2 : hh * 0.9); g.closePath(); g.fill();
  }
}
const SHAPE_ICON = { sine: "M1 6 C4 -2 7 -2 10 6 S16 14 19 6", triangle: "M1 10 L6 2 L14 10 L19 2", square: "M1 10 V2 H10 V10 H19 V2", saw: "M1 10 L10 2 V10 L19 2", ramp: "M1 2 L10 10 V2 L19 10" };
const PARAM = {
  lfo: [["rate_hz", "Rate", 0.01, 4, { log: true, step: 0.01, fmt: (v) => (v < 0.1 ? v.toFixed(3) : v.toFixed(2)) + " Hz" }], ["amount", "Amount", 0, 1, { step: 0.01, fmt: (v) => `${Math.round(v * 100)}%`, tip: "the LFO's amplitude: scales every knob it moves" }], ["phase", "Phase", 0, 1, { step: 0.01 }], ["symmetry", "Symmetry", 0.05, 0.95, { step: 0.01 }], ["smooth", "Smooth", 0, 1, { step: 0.01 }]],
  env: [["attack_s", "Attack", 0, 60, { step: 0.1, fmt: (v) => `${v.toFixed(1)} s` }], ["decay_s", "Decay", 0, 60, { step: 0.1, fmt: (v) => `${v.toFixed(1)} s` }], ["sustain", "Sustain", 0, 1, { step: 0.01 }], ["hold_s", "Hold", 0, 60, { step: 0.1, fmt: (v) => `${v.toFixed(1)} s`, tip: "with Loop on: how long the envelope holds at Sustain before its release" }], ["release_s", "Release", 0, 60, { step: 0.1, fmt: (v) => `${v.toFixed(1)} s`, tip: "with Loop on: the fall from Sustain to zero before the envelope starts again" }]],
  rand: [["rate_hz", "Rate", 0.01, 4, { log: true, step: 0.01, fmt: (v) => (v < 0.1 ? v.toFixed(3) : v.toFixed(2)) + " Hz" }], ["amount", "Amount", 0, 1, { step: 0.01, fmt: (v) => `${Math.round(v * 100)}%`, tip: "the source's amplitude: scales every knob it moves" }], ["smooth", "Smooth", 0, 1, { step: 0.01 }], ["seed", "Seed", 0, 999, { int: true }]],
};
const HZ = (v) => (v < 0.1 ? v.toFixed(3) : v.toFixed(2)) + " Hz", PCT = (v) => `${Math.round(v * 100)}%`, SECS = (v) => (v < 1 ? `${Math.round(v * 1000)} ms` : `${v.toFixed(2)} s`);
Object.assign(PARAM, {
  steps: [["rate_hz", "Rate", 0.1, 16, { log: true, step: 0.01, fmt: (v) => `${v.toFixed(2)}/s`, tip: "steps a second" }], ["length", "Length", 2, 16, { int: true, tip: "how many steps before it repeats" }],
    ["glide", "Glide", 0, 1, { step: 0.01, tip: "the share of each step spent easing into the next" }], ["amount", "Amount", 0, 1, { step: 0.01, fmt: PCT, tip: "scales every knob it moves" }]],
  follow: [["attack_s", "Attack", 0.01, 2, { log: true, step: 0.001, fmt: SECS, tip: "how fast it rises with the sound" }], ["release_s", "Release", 0.04, 5, { log: true, step: 0.001, fmt: SECS, tip: "how fast it falls back" }],
    ["floor_db", "Floor", -70, -10, { step: 1, fmt: (v) => `${v.toFixed(0)} dB`, tip: "the level it reads as 0; 30 dB above it reads as 1" }], ["amount", "Amount", 0, 1, { step: 0.01, fmt: PCT, tip: "scales every knob it moves" }]],
  drift: [["rate_hz", "Rate", 0.005, 1, { log: true, step: 0.001, fmt: HZ, tip: "how fast it wanders" }], ["rough", "Roughness", 0, 1, { step: 0.01, tip: "how much faster detail rides on the wander" }],
    ["amount", "Amount", 0, 1, { step: 0.01, fmt: PCT, tip: "scales every knob it moves" }], ["seed", "Seed", 0, 999, { int: true, tip: "another path" }]],
  chaos: [["rate_hz", "Rate", 0.01, 4, { log: true, step: 0.01, fmt: HZ, tip: "how fast it runs along the attractor" }], ["amount", "Amount", 0, 1, { step: 0.01, fmt: PCT, tip: "scales every knob it moves" }],
    ["seed", "Seed", 0, 999, { int: true, tip: "another starting point: another path" }]],
});
const DEFK = { rand: { rate_hz: 2 }, steps: { rate_hz: 2, length: 8, glide: 0, amount: 1 }, follow: { attack_s: 0.05, release_s: 0.5, floor_db: -45, amount: 1 },
  drift: { rate_hz: 0.05, rough: 0.4, amount: 1, seed: 1 }, chaos: { rate_hz: 0.2, amount: 1, seed: 1 } };      // where a kind's default differs from DEF
const DEF = { rate_hz: 0.1, amount: 1, phase: 0, symmetry: 0.5, smooth: 0, attack_s: 4, decay_s: 4, sustain: 0.6, release_s: 4, hold_s: 4, loop: 0, seed: 1 };
function setSrc(id, k, v) { MOD.spec.sources[id][k] = v; sendMod(); }
let editorKnobs = [];
function drawEditor() {
  document.querySelectorAll("#modtabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.k === MOD.tab));
  const ed = $("#editor");
  ed.innerHTML = ""; editorKnobs = [];
  if (MOD.tab === "macro") { ed.appendChild(h(`<div class="macros-ed" id="macros"></div>`)); drawMacros(); return; }
  const ids = Object.keys(SRC).filter((id) => tabOf(SRC[id].kind) === MOD.tab);
  for (const id of ids) {
    const s = SRC[id], p = MOD.spec.sources[id];
    if (!p) continue;
    const col = h(`<div class="ecol" style="--c:${s.color}"><h4 class="egrab" title="Drag onto a knob to link ${esc(s.label)}, or click to assign it">${esc(s.label)}</h4></div>`);
    {                                                            // the name is a handle, so a source whose tile is hidden can still be linked
      const hd = col.querySelector("h4");
      hd.addEventListener("click", () => { if (hd.dragged) { hd.dragged = false; return; } if (MOD.on) assign(id); });
      dragSource(hd, id); hintOn(hd);
    }
    if (s.kind === "lfo") {
      const sh = h(`<div class="shapes"></div>`);
      for (const k in SHAPE_ICON) {
        const b = h(`<button title="${k}" aria-pressed="${p.shape === k}"><svg viewBox="0 0 20 12" fill="none" stroke="currentColor" stroke-width="1.6"><path d="${SHAPE_ICON[k]}"/></svg></button>`);
        b.addEventListener("click", () => { setSrc(id, "shape", k); drawEditor(); });
        sh.appendChild(b);
      }
      col.appendChild(sh);
    }
    if (s.kind === "env") {                                  // Loop: Hold and Release only act with it on
      const sh = h(`<div class="shapes"></div>`);
      const b = h(`<button class="loopbtn" aria-pressed="${!!p.loop}" title="Loop: hold at Sustain for Hold, fall over Release, and start again">Loop</button>`);
      b.addEventListener("click", () => { setSrc(id, "loop", p.loop ? 0 : 1); drawEditor(); });
      sh.appendChild(b); col.appendChild(sh);
    }
    if (s.kind === "follow") {                                // Invert: loud moves the knob down
      const sh = h(`<div class="shapes"></div>`);
      const b = h(`<button class="loopbtn" aria-pressed="${!!+p.invert}" title="Invert: quiet reads high and loud reads low, so the sound getting louder moves a knob down">Invert</button>`);
      b.addEventListener("click", () => { setSrc(id, "invert", +p.invert ? 0 : 1); drawEditor(); });
      sh.appendChild(b); col.appendChild(sh);
    }
    if (s.kind === "chaos") {                                 // which attractor
      const sh = h(`<div class="shapes"></div>`);
      for (const [k, name] of [["lorenz", "Lorenz"], ["rossler", "Rössler"]]) {
        const b = h(`<button class="loopbtn" aria-pressed="${p.system === k}" title="${name}'s attractor">${name}</button>`);
        b.addEventListener("click", () => { setSrc(id, "system", k); drawEditor(); });
        sh.appendChild(b);
      }
      col.appendChild(sh);
    }
    const g = h(`<canvas class="graph" width="600" height="92" data-g="${id}"></canvas>`);
    col.appendChild(g);
    if (s.kind === "steps") {                                 // draw the steps: press and drag across the bars
      g.classList.add("stepsed"); g.title = "Press and drag to draw the steps; up is positive";
      const at = (e) => { const r = g.getBoundingClientRect(), n = Math.round(p.length);
        const i = Math.max(0, Math.min(n - 1, Math.floor(((e.clientX - r.left) / r.width) * n))), v = Math.max(-1, Math.min(1, 1 - 2 * (e.clientY - r.top) / r.height));
        p[`s${i}`] = Math.round(v * 20) / 20; drawGraphs(); };
      g.addEventListener("pointerdown", (e) => { g.setPointerCapture(e.pointerId); at(e);
        const mv = (ev) => at(ev), up = () => { g.removeEventListener("pointermove", mv); g.removeEventListener("pointerup", up); MOD.touched = Date.now(); sendMod(); drawTiles(); };
        g.addEventListener("pointermove", mv); g.addEventListener("pointerup", up); });
    }
    const kr = h(`<div class="krow"></div>`);
    for (const [k, label, min, max, o] of PARAM[s.kind]) {
      const kn = knob({ label, min, max, reset: (id === "rand2" && k === "rate_hz" ? 0.25 : undefined) ?? DEFK[s.kind]?.[k] ?? DEF[k], get: () => +MOD.spec.sources[id][k], put: (v) => { setSrc(id, k, v); drawGraphs(); }, ...o });
      kr.appendChild(kn); editorKnobs.push(kn);
      if (s.kind === "env" && (k === "hold_s" || k === "release_s")) dimKnob(kn, !p.loop);
    }
    col.appendChild(kr);
    col.appendChild(routesEl(id));
    ed.appendChild(col);
  }
  if (MOD.tab === "env") ed.appendChild(h(`<div class="ecol"><p class="pnote">Each envelope runs from ▶ and holds at its sustain while it plays. ENV 2 and ENV 3 show above once they move something: drag one by its name here. With Loop on, it holds for Hold, falls to zero over Release, and starts again.</p></div>`));
  drawGraphs();
}
function routesEl(id) {
  const box = h(`<div class="routes"></div>`);
  const rs = MOD.spec.routes.filter((r) => r.src === id);
  if (!rs.length) box.appendChild(h(`<div class="noroutes">Not linked. Drag ${esc(SRC[id].label)} onto a knob.</div>`));
  for (const r of rs) {
    const row = h(`<div class="route"><span title="${esc(r.tgt)}">→ ${esc(tgtOf(r.tgt)?.label || r.tgt)}</span><input type="range" min="-1" max="1" step="0.01" value="${r.depth}" aria-label="depth"><i>${r.depth.toFixed(2)}</i><button title="Remove this route">✕</button></div>`);
    const inp = row.querySelector("input");
    inp.addEventListener("input", () => { r.depth = +inp.value; row.querySelector("i").textContent = r.depth.toFixed(2); MOD.touched = Date.now(); MODKNOBS.forEach((k) => k.paintMod()); });
    inp.addEventListener("change", () => setRoute(id, r.tgt, +inp.value));
    row.querySelector("button").addEventListener("click", () => { setRoute(id, r.tgt, 0); drawEditor(); MODKNOBS.forEach((k) => k.paintMod()); });
    box.appendChild(row);
  }
  return box;
}
function drawGraphs() {
  document.querySelectorAll("#editor canvas.graph").forEach((c) => {
    const dpr = window.devicePixelRatio || 1, cw = Math.round(c.clientWidth * dpr), ch = Math.round(c.clientHeight * dpr);
    if (cw > 0 && ch > 0 && (c.width !== cw || c.height !== ch)) { c.width = cw; c.height = ch; }   // drawn at the size shown, so text is not stretched
    const id = c.dataset.g, s = SRC[id], g = c.getContext("2d"), w = c.width, hh = c.height, col = CSSCOL[s.kind];
    const FONT = `${Math.round(10 * dpr)}px ui-monospace, Menlo, monospace`;
    g.lineJoin = "round";
    g.clearRect(0, 0, w, hh);
    g.strokeStyle = "#23272f"; g.lineWidth = 1; g.beginPath(); g.moveTo(0, hh / 2); g.lineTo(w, hh / 2); g.stroke();
    const p = MOD.spec.sources[id];
    const cyc = p.attack_s + p.decay_s + (p.hold_s ?? 4) + p.release_s;
    if (s.kind === "steps") {                                 // the steps as bars, the ones past Length dimmed
      const n = Math.round(p.length), bw = w / 16;
      for (let i = 0; i < 16; i++) { const v = +p[`s${i}`], y0 = hh / 2, y1 = hh / 2 - v * hh * 0.42;
        g.fillStyle = i < n ? col : "#2a2e37"; g.fillRect(i * bw + 3, Math.min(y0, y1), bw - 6, Math.max(2, Math.abs(y1 - y0))); }
      g.fillStyle = "#5a606d"; g.font = FONT; g.textAlign = "right"; g.fillText(`${n} steps · ${(n / p.rate_hz).toFixed(1)} s`, w - 6, hh - 4); g.textAlign = "left";
      return;
    }
    if (s.kind === "follow") {                                // what it has read in the last 20 s of play
      const H = MOD.folHist, t1 = H.length ? H[H.length - 1][0] : 0;
      g.strokeStyle = col; g.lineWidth = 2 * dpr; g.beginPath();
      H.filter((q) => q[0] >= t1 - 20).forEach((q, i) => { const x = w - ((t1 - q[0]) / 20) * w, y = hh * 0.92 - q[1] * hh * 0.84; i ? g.lineTo(x, y) : g.moveTo(x, y); });
      g.stroke(); g.fillStyle = "#5a606d"; g.font = FONT; g.fillText(H.length ? "the last 20 s" : "reads the sound while it plays", 6 * dpr, 13 * dpr);
      return;
    }
    const span = s.kind === "lfo" ? 1 / p.rate_hz : s.kind === "rand" ? 8 / p.rate_hz / 2 : s.kind === "drift" || s.kind === "chaos" ? Math.min(240, 4 / p.rate_hz)
      : p.loop ? Math.max(4, 2 * cyc) : Math.max(8, p.attack_s + p.decay_s + 4);
    const len = null, N = 400;
    const y = (v) => (s.bipolar ? hh / 2 - v * hh * 0.42 : hh * 0.92 - v * hh * 0.84);
    g.strokeStyle = col; g.lineWidth = 2 * dpr; g.beginPath();
    for (let i = 0; i <= N; i++) {
      const t = (i / N) * span, v = s.kind === "lfo" ? lfoV({ ...p, phase: p.phase }, t) : s.kind === "rand" ? randV(p, t) : s.kind === "drift" ? driftV(p, t)
        : s.kind === "chaos" ? chaosV(p, t) : envV(p, t, len);
      i ? g.lineTo((i / N) * w, y(v)) : g.moveTo(0, y(v));
    }
    g.stroke();
    g.fillStyle = "#5a606d"; g.font = FONT;
    g.fillText(s.kind === "lfo" ? "one cycle" : s.kind === "env" ? `${span.toFixed(0)} s` : `${span.toFixed(1)} s`, 6 * dpr, 13 * dpr);
  });
}
document.querySelectorAll("#modtabs button").forEach((b) => b.addEventListener("click", () => { MOD.tab = b.dataset.k; drawEditor(); }));
let macroEls = [];
function drawMacros() {
  const box = $("#macros");
  macroEls = [];
  if (!box) return;
  box.innerHTML = "";
  for (const id of ["m1", "m2", "m3", "m4"]) {
    const p = MOD.spec.sources[id];
    if (!p) continue;
    const el = h(`<div class="macro"></div>`);
    const grab = h(`<button class="mgrab" type="button" style="--c:${SRC[id].color}" title="Drag onto a knob to link ${SRC[id].label}, or click to assign it">${esc(SRC[id].label)}</button>`);
    grab.addEventListener("click", () => { if (grab.dragged) { grab.dragged = false; return; } if (MOD.on) assign(id); });
    dragSource(grab, id); hintOn(grab);
    el.appendChild(grab);
    const kn = knob({ label: "", min: 0, max: 1, step: 0.01, reset: 0, get: () => +MOD.spec.sources[id].value, put: (v) => setSrc(id, "value", v), fmt: (v) => v.toFixed(2), tip: `${SRC[id].label}: one knob moving every control routed from it` });
    el.appendChild(kn); macroEls.push(kn);
    const nm = h(`<input value="${esc(p.name || SRC[id].label)}" aria-label="${SRC[id].label} name" maxlength="24">`);
    nm.addEventListener("change", () => { setSrc(id, "name", nm.value); });
    el.appendChild(nm);
    const n = MOD.spec.routes.filter((r) => r.src === id).length;
    el.appendChild(h(`<div class="mcount">${n ? `moves ${n} control${n > 1 ? "s" : ""}` : "not routed"}</div>`));
    el.appendChild(routesEl(id));
    box.appendChild(el);
  }
}
let modKey = "";
function adoptMod(spec) {
  if (!spec || holding || Date.now() - MOD.touched < 800) return;
  const k = JSON.stringify(spec);
  if (k === modKey) return;
  const shape = JSON.stringify([Object.keys(spec.sources), spec.routes.map((r) => [r.src, r.tgt])]);
  const was = JSON.stringify([Object.keys(MOD.spec.sources), MOD.spec.routes.map((r) => [r.src, r.tgt])]);
  MOD.spec = JSON.parse(k); modKey = k;
  if (shape !== was || !$("#editor").children.length) { drawEditor(); drawMacros(); }
  else { editorKnobs.forEach((x) => x.paint()); macroEls.forEach((x) => x.paint()); drawGraphs(); }
  drawTiles(); MODKNOBS.forEach((x) => x.paint());
}
function animate() {
  if (view === "make" && !document.hidden) {
    traceTiles();
    MODKNOBS.forEach((k) => { if (k.isConnected) k.paintMod(); });
    if (MOD.spec.routes.some((r) => r.tgt.startsWith("dose:"))) drawXY();
  }
  setTimeout(() => requestAnimationFrame(animate), 40);
}

// the panels' switches: Steer and Model layers off bypass them, keeping their settings
const SW = { steer: true, layers: true, touched: 0 };
function paintSwitches() {
  for (const [id, on, pn] of [["#pw-steer", SW.steer, "#steercard"], ["#pw-layers", SW.layers, "#pn-bend"], ["#pw-fx", S.master?.fx_on !== false, "#pn-fx"]]) {
    $(id).classList.toggle("on", on); $(id).setAttribute("aria-pressed", on); $(pn).classList.toggle("bypassed", !on);
  }
  $("#steercard").title = SW.steer ? "" : "Steer is off: every dose plays at 0. Its light turns it back on, with the doses as you left them.";
}
// Steer's card folds to a narrow strip and back, on or off; the page remembers it for this viewer
function foldSteer(f) {
  $("#steercard").classList.toggle("folded", f);
  $("#steerfold").textContent = f ? "›" : "‹"; $("#steerfold").setAttribute("aria-expanded", String(!f));
  $("#steerfold").title = f ? "Unfold Steer" : "Fold Steer to a narrow strip; it keeps playing";
  try { localStorage.setItem("aa-steer-fold", f ? "1" : ""); } catch (e) { /* storage blocked: folding still works */ }
}
$("#steerfold").addEventListener("click", () => foldSteer(!$("#steercard").classList.contains("folded")));
flowLink($("#steercard"), "steer");
try { if (localStorage.getItem("aa-steer-fold")) foldSteer(true); } catch (e) { /* storage blocked */ }
$("#pw-steer").addEventListener("click", async () => {
  SW.steer = !SW.steer; SW.touched = Date.now(); paintSwitches();
  try { await api("/api/control", { steer_on: SW.steer }); info(SW.steer ? "Steer on" : "Steer off: every dose plays at 0", "", 2000); } catch (e) { fail(e); }
});
$("#pw-fx").addEventListener("click", () => {
  const on = S.master.fx_on === false;
  S.master.fx_on = on; paintSwitches(); sendMaster({ fx_on: on });
  info(on ? "Effects on" : "Effects off: the model's sound passes by the chain, which keeps its effects", "", 2000);
});
$("#pw-layers").addEventListener("click", async () => {
  SW.layers = !SW.layers; SW.touched = Date.now(); paintSwitches();
  try { const r = await api("/api/chain", { lines: U.chain.filter((c) => c.on).map(lineOf), on: SW.layers }); showChainErrors(r.report || []); info(SW.layers ? "Model layers on" : "Model layers off: the model plays without vector bending", "", 2000); } catch (e) { fail(e); }
});

// polling the engine
const ms = (v) => (v === null || v === undefined ? "–" : `${Math.round(v)} ms`);
let dirsKnown = 0;
async function poll() {
  let s;
  try { s = await api("/api/status"); }
  catch (e) {
    $("#modelchip").innerHTML = `<span class="busy">the server is not answering</span>`;
    setTimeout(poll, S.playing ? 120 : 250);
    return;
  }
  // each part on its own: a fault in one is reported (pageError) and the others still run, so one bad part
  // cannot leave the panels unpainted
  const part = (name, fn) => { try { fn(); } catch (e) { console.error(name, e); pageError(`${name}: ${e.message}`); } };
  part("engine", () => {
      S.status = s;
      $("#ver").textContent = s.version || "";
      if (s.engines) {
        drawEngine(s);
      }
      const sl = s.model?.steer_layer;
      if (sl != null && $("#steercard .sat").textContent !== `layer ${sl} · fixed`) {
        $("#steercard .sat").textContent = `layer ${sl} · fixed`;
        $("#steercard .st").title = `Steer adds a direction to the model's stream at layer ${sl}, scaled by the stream's size and the dose: h + |h| · dose · d. Its place is fixed: its directions and the doses' musical ranges were measured there.`;
      }
      if (s.engine !== S.engine) {
        S.engine = s.engine; dirsKnown = -1;
        $("#mg-only").hidden = s.engine !== "musicgen";
        engKnobs.forEach((k) => (k.hidden = s.engine !== "mrt2")); sampPair.hidden = s.engine !== "mrt2";
        speedPair.hidden = s.engine !== "mrt2";
      }
      MOD.on = s.engine === "mrt2" && s.state === "ready";
      S.playing = !!s.playing;
      MOD.playing = !!s.playing;
      if (s.playing) { MOD.t0 = s.seconds; MOD.tAt = performance.now(); }
      if (s.playing && s.mod_values && s.mod_values.follow != null) {      // the Follower is read from the engine, not computed here
        MOD.folHist.push([s.seconds, s.mod_values.follow]);
        while (MOD.folHist.length && MOD.folHist[0][0] < s.seconds - 60) MOD.folHist.shift();
      } else if (!s.playing && MOD.folHist.length) MOD.folHist = [];
      if (s.mod) adoptMod(s.mod);
      if (Date.now() - SW.touched > 800 && "steer_on" in s) { SW.steer = s.steer_on !== false; SW.layers = s.layers_on !== false; paintSwitches(); }
  });
  part("knobs", () => {
      if (s.doses && !dragging && !pendingDoses && !holding && Date.now() - dosesTouched > 400) {
        const changed = Object.keys(s.doses).some((k) => (S.doses[k] || 0) !== s.doses[k]);
        if (changed) { Object.assign(S.doses, s.doses); drawXY(); paintDoses(); }
      }
      if (s.sampling && Date.now() - samplingTouched > 400) { S.sampling = { ...s.sampling }; if (!holding) engKnobs.forEach((k) => k.paint?.()); }
      if (s.master && Date.now() - masterTouched > 400) { S.master = { ...s.master }; if (!holding) { for (const k of [...outKnobs, hdrVol, ...widenKnobs, ...outToggles, ...dynKnobs, ...dynToggles, speedKnob]) k.paint?.(); } }
  });
  part("effects", () => {
      if (s.master?.fx) { if (!Object.keys(PFX.types).length && !PFX.loading) { PFX.loading = true; loadFx(); } else adoptFx(s.master.fx); }
  });
  part("readout", () => {
      if (!holding) leadKnob.paint();
      {                                                            // the automatic buffer: the total in gold, the split on hover
        const b = S.status.lead_boost || 0, set = (S.status.lead || 2) * S.frameMs;
        leadKnob.classList.toggle("boosted", b > 0);
        leadKnob.querySelector(".kv").title = b ? `${set} ms set, +${b * S.frameMs} ms added by the engine after a step ran long` : "";
      }
      if (s.frame_ms) S.frameMs = s.frame_ms;
      const name = s.model?.model_id || (s.engines || []).find((e) => e.id === s.engine)?.label || "";
      $("#modelchip").innerHTML = `<b>${esc(name)}</b> · ` + (s.error ? `<span class="busy">${esc(s.error)}</span>` : s.freezing ? `<span class="busy">freezing ${s.freezing[0]} / ${s.freezing[1]} frames</span>` : s.playing ? `playing ${esc(s.preset)} · seed ${s.seed}` : `<span class="${s.state === "ready" ? "" : "busy"}">${esc(s.state)}</span>`);
      if (s.model) $("#modelinfo").textContent = `${s.model.dtype} · ${s.model.sample_rate / 1000} kHz ${s.model.channels === 2 ? "stereo" : "mono"}`;
      const L = s.latency || { measured_ms: [] };
      // the line under the waveform says one or two words; the figures are in its tip and in Settings
      const ro = $("#readout");
      if (s.playing) {
        const dropped = s.underruns > (S.lastUnderruns ?? s.underruns);
        S.lastUnderruns = s.underruns;
        if (dropped) S.dropAt = Date.now();
        const recent = Date.now() - (S.dropAt || 0) < 3000;
        ro.innerHTML = recent ? `<span class="bad">Dropouts</span>` : "Playing";
        ro.title = `A step takes ${s.step_ms_p50.toFixed(1)} ms (${s.step_ms_p95.toFixed(1)} at worst) of its ${S.frameMs}; ${s.underruns} dropouts in ${s.seconds.toFixed(0)} s; a move heard after ${L.measured_ms.length ? L.measured_ms.map(Math.round).join(", ") + " ms" : "–"}`;
        engMeter.set(s.level || 0, s.level_lr, s.peak_lr); hdrMeter.set(s.level || 0, s.level_lr, s.peak_lr);
      } else {
        S.lastUnderruns = undefined;
        const st = String(s.state || "");
        if (s.error) { ro.innerHTML = `<span class="bad">Error</span>`; ro.title = s.error; engMeter.set(0); hdrMeter.set(0); }
        else ro.textContent = st === "ready" ? "Ready" : /^loading/.test(st) ? "Loading" : /^measuring/.test(st) ? "Measuring" : /^(embedding|making)/.test(st) ? "Preparing" : st === "stopped" ? "Stopped" : st;
        if (!s.error) ro.title = st === "ready" ? `A move is heard about ${ms(L.declared_first_ms)} after it is made` : st;
        engMeter.set(0); hdrMeter.set(0);
      }
  });
  part("scope and meters", () => {
      if (SPC.scope !== "spec") { newScope(s.scope, s.scope_lr); if (!s.playing) drawScope(s.scope, s.scope_lr); }
      drawField(s.field);
      OM.set(s);
      drawDoseHist(s); ST.set(s); DYN.set(s); paintFxLeds(s); paintSpeedModes();
  });
    try { paintActivity(s); } catch (e) { console.error("unit displays", e); pageError(`unit displays: ${e.message}`); }   // a fault here leaves the rest of the panel updating
    try { drawFlow(s); } catch (e) { console.error("Signal path", e); pageError(`Signal path: ${e.message}`); }   // a fault here leaves the rest of the panel updating
  part("Listen and status line", () => {
      if (s.meters) adoptMeters(s.meters);
      adoptSpec("model", s.spec, s.playing);
      if (Date.now() > infoUntil) { $("#infotext").textContent = MOD.assign ? assignText() : ""; $("#infotext").className = ""; }
      $("#infostatus").textContent = s.playing ? `model: ${s.seconds.toFixed(0)} s, ${s.underruns} dropouts` : `model: ${s.state}`;
      transport();
  });
  try {
    if (s.state === "ready" && dirsKnown <= 0) { await loadPresets(); dirsKnown = S.dirs.length; listSessions(); }
  } catch (e) { pageError(`presets: ${e.message}`); }
  setTimeout(poll, S.playing ? 120 : 250);            // quicker while playing, so the waveform, meters and stereo field keep up
}

// ================================================================ THE BROWSER: saved sounds
const SND = { list: [], fresh: null, playing: null };
function drawSounds() {
  const q = $("#brsearch").value.trim().toLowerCase();
  const list = SND.list.filter((s) => !q || `${s.name} ${s.preset} ${s.strips.join(" ")}`.toLowerCase().includes(q));
  $("#nsounds").textContent = SND.list.length ? SND.list.length : "";
  const box = $("#sounds");
  box.innerHTML = list.length ? "" : `<div class="br-none">${SND.list.length ? "No sound matches." : "None yet. Play the model and save the last seconds (◉)."}</div>`;
  for (const s of list) {
    const strips = s.strips.length ? ` · ${s.strips.length} strip${s.strips.length > 1 ? "s" : ""}` : "";
    const el = h(`<div class="snd ${s.id === SND.fresh ? "fresh" : ""}" draggable="true" data-id="${esc(s.id)}" title="${esc(s.name)}\n${esc(s.preset || "")} · seed ${s.seed} · ${s.seconds} s${s.frozen ? " · frozen" : ""}${s.strips.length ? "\n" + esc(s.strips.join("\n")) : ""}\nDrag onto a track in the Session.">
      <button class="aud ${SND.playing === s.id ? "on" : ""}" title="Listen" aria-label="Listen">${SND.playing === s.id ? "■" : "▶"}</button>
      <span class="sl1"><span class="sname">${esc(s.name)}</span><span class="smeta">${s.seconds} s · ${s.frozen ? "❄" : "◉"} ${esc(s.made)}${strips}</span></span>
      <span class="sx"><button data-a="ren" title="Rename">✎</button><button data-a="del" title="Delete (moved to the trash folder)">✕</button></span></div>`);
    el.addEventListener("dragstart", (e) => { e.dataTransfer.setData("text/x-sound", s.id); e.dataTransfer.effectAllowed = "copy"; });
    el.querySelector(".aud").addEventListener("click", () => audition(s.id));
    el.querySelector('[data-a="ren"]').addEventListener("click", async () => {
      const n = prompt("Name", s.name); if (n === null) return;
      try { SND.list = (await api(`/api/sounds/${s.id}`, { name: n })).sounds; drawSounds(); refreshDesk(); } catch (e) { fail(e); }
    });
    el.querySelector('[data-a="del"]').addEventListener("click", async () => {
      if (!confirm(`Delete “${s.name}”? It is taken off every track and moved to data/snapshots/trash/.`)) return;
      try { const r = await api(`/api/sounds/${s.id}/delete`, {}); SND.list = r.sounds; D.state = r.desk; drawSounds(); showDesk(); } catch (e) { fail(e); }
    });
    box.appendChild(el);
  }
}
function audition(id) {
  const a = $("#audition");
  if (SND.playing === id) { a.pause(); SND.playing = null; drawSounds(); return; }
  a.src = `/api/sounds/${id}.wav`; a.play().catch(fail); SND.playing = id; drawSounds();
}
$("#audition").addEventListener("ended", () => { SND.playing = null; drawSounds(); });
$("#brsearch").addEventListener("input", drawSounds);
async function pollSounds() {
  try {
    const r = await api("/api/sounds");
    const key = JSON.stringify(r.sounds.map((s) => [s.id, s.name]));
    if (key !== SND.key) { SND.key = key; SND.list = r.sounds; drawSounds(); }
  } catch (e) { /* the status poll reports a dead server */ }
  setTimeout(pollSounds, 2000);
}
async function listSessions() {
  try {
    const j = await api("/api/sessions");
    $("#sessions").innerHTML = j.sessions.slice(0, 12).map((n) => `<div class="br-item"><a href="/api/sessions/${esc(n)}.wav" download title="The model's output">${esc(n)}</a><a class="minor" href="/api/sessions/${esc(n)}.post.wav" download title="After the master">post</a></div>`).join("") || `<div class="br-none">None yet. Stop saves the session.</div>`;
  } catch (e) { /* not yet */ }
}
async function listExports() {
  try {
    const j = await api("/api/exports");
    $("#exports").innerHTML = j.exports.slice(0, 16).map((n) => `<div class="br-item"><a href="/api/exports/${esc(n)}.wav" download>${esc(n)}</a><a class="minor" href="/api/exports/${esc(n)}.json" download>recipe</a></div>`).join("") || `<div class="br-none">None yet.</div>`;
  } catch (e) { /* not yet */ }
}

// ================================================================ SESSION
const FX = [["highpass_hz", "HP", (v) => `${Math.round(v)}`, 1], ["lowpass_hz", "LP", (v) => `${(v / 1000).toFixed(1)}k`, 100],
  ["delay_ms", "Delay", (v) => `${Math.round(v)}`, 1], ["delay_feedback", "Fdbk", (v) => v.toFixed(2), 0.01], ["delay_mix", "D mix", (v) => v.toFixed(2), 0.01],
  ["reverb_size", "Size", (v) => v.toFixed(2), 0.01], ["reverb_mix", "R mix", (v) => v.toFixed(2), 0.01], ["gain_db", "Gain", (v) => `${v.toFixed(1)}`, 0.5]];
const flushMix = throttle(() => api("/api/desk/mix", { mix_volume: D.state.mix_volume }).catch(fail), 40);
const D = { state: null, sessKey: null, mixBuilt: false, ctl: [], meters: [], touched: {} };
// a track fader's travel, as a console's: 0 dB about a quarter of the way down, more room around it
const LEVEL_TAPER = [[0, -60], [0.1, -50], [0.2, -40], [0.31, -30], [0.43, -20], [0.57, -10], [0.67, -5], [0.77, 0], [0.87, 5], [0.96, 10], [1, 12]];
const LEVEL_TICKS = [[10, "10"], [5, "5"], [0, "0"], [-5, "5"], [-10, "10"], [-20, "20"], [-30, "30"], [-40, "40"], [-50, "50"], [-60, "∞"]];
async function deskCall(path, body = {}) { try { D.state = await api(path, body); showDesk(); } catch (e) { fail(e); } }
// While a fader or knob on a track moves, its changes go at most one every 40 ms and the last
// always goes (a debounce sent nothing until the hand stopped). The answer is not adopted, since
// it may be older than the cap: the poll brings the desk's state once the track rests.
const trackPending = {}, trackFlush = {};
function sendTrack(i, body) {
  const t = D.state.tracks[i];
  if (body.fx) Object.assign(t.fx, body.fx); else Object.assign(t, body);
  D.touched[i] = Date.now();
  const pend = trackPending[i] || (trackPending[i] = {});
  if (body.fx) pend.fx = { ...(pend.fx || {}), ...body.fx };
  Object.assign(pend, { ...body, fx: pend.fx });
  if (!pend.fx) delete pend.fx;
  (trackFlush[i] || (trackFlush[i] = throttle(() => {
    const b = trackPending[i]; trackPending[i] = {};
    api(`/api/desk/track/${i}`, b).then(() => { D.touched[i] = Date.now(); }).catch(fail);
  }, 40)))();
}

function drawSession() {
  const st = D.state, grid = $("#sess");
  grid.innerHTML = "";
  for (const t of st.tracks) {
    const col = h(`<div class="scol" data-t="${t.i}"><div class="shd"><span title="${esc(t.name)}">${esc(t.name)}</span><i class="lp" title="${t.loop ? "Clips loop: click to play them once" : "Clips play once: click to loop them"}">${t.loop ? "↻ loop" : "→ once"}</i></div></div>`);
    col.querySelector(".lp").addEventListener("click", async () => { await api(`/api/desk/track/${t.i}`, { loop: !t.loop }).then((r) => { D.state = r; showDesk(); }).catch(fail); });
    t.clips.forEach((c, k) => {
      const on = t.live && t.current === k;
      const el = h(`<button class="clip ${t.current === k ? "cur" : ""} ${on ? "live" : ""}" title="${esc(c.label)}: ${on ? "click to stop" : "click to start"}"><b>${esc(c.label)}</b><span>${c.seconds ?? ""} s</span><span class="cx" title="Take off this track">✕</span><i class="pg"></i></button>`);
      el.addEventListener("click", (e) => {
        if (e.target.classList.contains("cx")) { deskCall(`/api/desk/track/${t.i}`, { remove_clip: k }); return; }
        deskCall(on ? `/api/desk/track/${t.i}/stop` : `/api/desk/track/${t.i}/trigger`, on ? {} : { clip: k });
      });
      col.appendChild(el);
    });
    const add = h(`<button class="slot" title="Put a saved sound on this track">+</button>`);
    add.addEventListener("click", () => menu(add, SND.list.length ? [{ head: `Onto ${t.name}` }, ...SND.list.slice(0, 40).map((s) => ({ label: s.name, sub: `${s.seconds} s · ${s.frozen ? "frozen" : "saved while playing"}`, run: () => putSound(t.i, s.id) }))]
      : [{ head: "No sounds yet" }, { label: "Make some first", sub: "In ☰ Synth, play the model and save the last seconds", disabled: true }]));
    col.appendChild(add);
    const stop = h(`<button class="slot stop" title="Stop this track">■</button>`);
    stop.disabled = !t.live;
    stop.addEventListener("click", () => deskCall(`/api/desk/track/${t.i}/stop`));
    col.appendChild(stop);
    col.addEventListener("dragover", (e) => { if (e.dataTransfer.types.includes("text/x-sound")) { e.preventDefault(); col.classList.add("over"); } });
    col.addEventListener("dragleave", () => col.classList.remove("over"));
    col.addEventListener("drop", (e) => { e.preventDefault(); col.classList.remove("over"); const id = e.dataTransfer.getData("text/x-sound"); if (id) putSound(t.i, id); });
    grid.appendChild(col);
  }
}
async function putSound(ti, sid) { deskCall(`/api/desk/track/${ti}/add`, { sound: sid }); }
function progress() {
  for (const t of D.state.tracks) {
    if (!t.live || t.current === null) continue;
    const el = document.querySelectorAll(`.scol[data-t="${t.i}"] .clip`)[t.current];
    if (el) el.style.setProperty("--p", `${(100 * t.pos_s) / Math.max(0.01, t.len_s)}%`);
  }
}

let mixMeter = null;
function buildMix() {
  const st = D.state, desk = $("#mixdesk");
  desk.innerHTML = ""; D.ctl = []; D.meters = [];
  for (const t of st.tracks) {
    const T = () => D.state.tracks[t.i];
    const el = h(`<div class="strip"><div class="sh"><span class="sn">${t.i + 1}</span><input class="tname" aria-label="track name"></div></div>`);
    const nm = el.querySelector(".tname");
    nm.value = t.name;
    nm.addEventListener("change", () => sendTrack(t.i, { name: nm.value }));
    D.ctl.push({ paint: () => { if (document.activeElement !== nm) nm.value = T().name; } });
    const pan = knob({ label: "Pan", min: -1, max: 1, step: 0.01, reset: 0, get: () => T().fx.pan, put: (v) => sendTrack(t.i, { fx: { pan: v } }), fmt: (v) => (Math.abs(v) < 0.005 ? "C" : `${v < 0 ? "L" : "R"}${Math.round(Math.abs(v) * 100)}`) });
    el.appendChild(pan); D.ctl.push(pan);
    const row = h(`<div class="tgrow"></div>`);
    for (const [k, lab, cls] of [["mute", "M", "m"], ["solo", "S", "s"]]) {
      const b = toggle(lab, () => T()[k], (v) => sendTrack(t.i, { [k]: v }), cls);
      b.title = k === "mute" ? "Mute" : "Solo"; row.appendChild(b); D.ctl.push(b);
    }
    const x = h(`<button class="tg" title="Export this track or its clip, before or after its effects">⤓</button>`);
    x.addEventListener("click", () => menu(x, [{ head: `Export ${T().name}` },
      ...[["clip_pre", "Clip, before effects", "the sound as the model made it"], ["clip_post", "Clip, after effects", `with a ${3} s tail`], ["track_pre", "Track, before effects", "looped to the mix length"], ["track_post", "Track, after effects", "looped, with level and pan"]]
        .map(([w, l, s]) => ({ label: l, sub: s, disabled: T().current === null, run: () => doExport({ what: w, track: t.i }) }))]));
    row.appendChild(x);
    el.appendChild(row);
    const fx = h(`<details class="hostfx"><summary>Effects <i></i></summary><div class="krow"></div></details>`);
    for (const [k, lab, fmt, step] of FX) {
      const [lo, hi] = st.fx_ranges[k];
      const kn = knob({ label: lab, min: lo, max: hi, step, reset: st.fx_defaults[k], get: () => T().fx[k], put: (v) => sendTrack(t.i, { fx: { [k]: v } }), fmt });
      fx.querySelector(".krow").appendChild(kn); D.ctl.push(kn);
    }
    D.ctl.push({ paint: () => { const f = T().fx, on = f.delay_mix > 0 || f.reverb_mix > 0 || f.highpass_hz > 20.5 || f.lowpass_hz < 19999 || f.gain_db !== 0; fx.querySelector("summary i").textContent = on ? "●" : ""; } });
    el.appendChild(fx);
    const lv = fader({ label: "Level", min: -60, max: 12, step: 0.5, reset: 0, taper: LEVEL_TAPER, ticks: LEVEL_TICKS, meter: () => T().peak,
      get: () => T().level_db, put: (v) => sendTrack(t.i, { level_db: v }), fmt: (v) => (v <= -60 ? "−∞ dB" : `${v.toFixed(1)} dB`) });
    el.appendChild(lv); D.ctl.push(lv); D.meters.push(lv);
    desk.appendChild(el);
  }
  const mx = h(`<div class="strip"><div class="sh"><span class="st">Mix</span></div><div class="badge">bus</div></div>`);
  const lim = toggle("Limiter", () => D.state.limit, (v) => { D.state.limit = v; api("/api/desk/mix", { limit: v }).catch(fail); });
  const vol = fader({ label: "Mix", min: 0, max: 1.5, step: 0.01, reset: 0.8, meter: () => D.state.peak,
    ticks: [[1.5, "1.5"], [1.25, ""], [1, "1"], [0.75, ""], [0.5, ".5"], [0.25, ""], [0, "0"]], get: () => D.state.mix_volume, put: (v) => { D.state.mix_volume = v; D.touched.mix = Date.now(); flushMix(); }, fmt: (v) => v.toFixed(2) });
  const xm = h(`<button class="tg" title="Export the mix: every track that would be heard, through the bus">⤓ Mix</button>`);
  xm.addEventListener("click", () => doExport({ what: "mix" }));
  mx.appendChild(lim); mx.appendChild(xm); mx.appendChild(vol);
  D.ctl.push(lim, vol); D.meters.push(vol);
  desk.appendChild(mx);
  mixMeter = meter("MIX");
  desk.appendChild(mixMeter);
  D.mixBuilt = true;
}
function showDesk() {
  const st = D.state;
  if (!st) return;
  if (!D.mixBuilt) buildMix();
  const key = JSON.stringify(st.tracks.map((t) => [t.name, t.clips, t.current, t.live, t.loop]));
  if (key !== D.sessKey) { D.sessKey = key; drawSession(); }
  progress();
  if (!holding) D.ctl.forEach((c) => c.paint());
  D.meters.forEach((m) => m.meter());
  mixMeter?.set(st.level || 0, st.level_lr);
  transport();
  const live = st.tracks.filter((t) => t.live).length;
  if (view === "session" || live) $("#infostatus").textContent = `tracks: ${live} playing` + (st.latency_ms ? ` · ${st.latency_ms} ms out` : "");
}
async function refreshDesk() { try { D.state = await api("/api/desk"); showDesk(); } catch (e) { /* reported by the status poll */ } }
async function pollDesk() {
  try {
    const st = await api("/api/desk");
    adoptSpec("desk", st.spec, (st.tracks || []).some((t) => t.live));
    const recent = Object.entries(D.touched).some(([, t]) => Date.now() - t < 400);
    if (!recent) D.state = st; else if (D.state) { D.state.level = st.level; D.state.level_lr = st.level_lr; D.state.peak = st.peak; D.state.tracks.forEach((t, i) => Object.assign(t, { live: st.tracks[i].live, peak: st.tracks[i].peak, pos_s: st.tracks[i].pos_s, len_s: st.tracks[i].len_s, clips: st.tracks[i].clips, current: st.tracks[i].current })); }
    showDesk();
  } catch (e) { /* reported by the status poll */ }
  setTimeout(pollDesk, D.state?.playing ? 200 : 700);
}
$("#dplay").addEventListener("click", () => deskCall("/api/desk/play"));
$("#dstop").addEventListener("click", () => deskCall("/api/desk/stop"));
async function doExport(body) {
  try {
    const r = await api("/api/desk/export", body);
    info(`exported ${r.export} (${r.seconds} s, peak ${r.peak.toFixed(2)})`, "ok");
    listExports();
  } catch (e) { fail(e); }
}

// ================================================================ TUNE: the listening grid
const T = { grid: null, ctl: null };
const unitTitle = (id) => U.units.find((u) => u.id === id)?.title || id;
const unitParam = (id, key) => (key === "at" ? "placement" : U.units.find((u) => u.id === id)?.params.find((q) => q.key === key)?.label || key);
const ctlLabel = (k) => k.startsWith("unit:") ? `${unitTitle(k.split(":")[1])} · ${unitParam(k.split(":")[1], k.split(":")[2])}` : (k.startsWith("dose:") ? `Steer: ${S.dirs.find((d) => d.name === k.slice(5))?.title || k.slice(5)}` : { guidance: "Strength", temperature: "Temperature", top_k: "Top-k" }[k] || k);
async function loadTune() {
  try { T.grid = await api("/api/calibration/grid"); } catch (e) { fail(e); return; }
  const ks = Object.keys(T.grid.controls);
  if (!ks.length) { $("#tunegrid").innerHTML = `<tr><td class="note">No measurements yet: run bench/sweep.py, then calibrate.</td></tr>`; return; }
  if (!T.ctl || !ks.includes(T.ctl)) T.ctl = ks[0];
  $("#tunemade").textContent = T.grid.made ? `measured ${T.grid.made}` : "";
  const mk = ks.filter((k) => !k.startsWith("unit:")), uk = ks.filter((k) => k.startsWith("unit:"));
  $("#tunetabs").innerHTML = mk.map((k) => `<button class="b small ${k === T.ctl ? "primary" : ""}" data-k="${esc(k)}">${esc(ctlLabel(k))}${T.grid.controls[k].ears ? " ✓" : ""}</button>`).join(" ")
    + (uk.length ? ` <select id="tuneunit" class="b small ${T.ctl.startsWith("unit:") ? "primary" : ""}"><option value="">Model layers…</option>${uk.map((k) => `<option value="${esc(k)}" ${k === T.ctl ? "selected" : ""}>${esc(ctlLabel(k))}${T.grid.controls[k].ears ? " ✓" : ""}</option>`).join("")}</select>` : "");
  $("#tuneunit")?.addEventListener("change", (e) => { if (e.target.value) { T.ctl = e.target.value; loadTune(); } });
  drawTune();
}
$("#tunetabs").addEventListener("click", (e) => { const k = e.target.closest("button")?.dataset.k; if (k) { T.ctl = k; loadTune(); } });
const fmtV = (v) => (Math.abs(v) < 1e-9 ? "0" : String(+v.toFixed(3)));
const showV = (k, v) => (k.endsWith(":at") && v === -1 ? "all" : fmtV(v));
function drawTune() {
  const c = T.grid.controls[T.ctl], vals = c.values, [lo, hi] = c.range;
  const presets = Object.keys(c.presets);
  const folder = T.ctl.replace(":", "-");
  let html = `<tr><th></th>${vals.map((v) => `<th class="${v === c.default ? "def" : ""}">${showV(T.ctl, v)}</th>`).join("")}</tr>`;
  for (const pid of presets) {
    const fl = T.grid.floor?.[pid];
    html += `<tr><th class="tpn" title="${fl ? `the music varies by itself: two seeds at the defaults differ by ${fl.timbre_db} dB of spectrum and ${fl.level_range_db} dB of level` : ""}">${esc(S.presets.find((x) => x.id === pid)?.name || pid)}</th>`;
    for (const m of c.presets[pid]) {
      const seed = (S.presets.find((x) => x.id === pid)?.seeds || [1])[0];
      const cls = m.broken.length ? "broken" : c.kind === "placement" || (m.value >= lo - 1e-9 && m.value <= hi + 1e-9) ? "in" : "out";
      const tip = `${ctlLabel(T.ctl)} ${fmtV(m.value)} on ${pid}\nlevel ${m.level_db >= 0 ? "+" : ""}${m.level_db} dB, spectrum moved ${m.timbre_db} dB${fl ? ` (seeds alone: ${fl.timbre_db})` : ""}\nbrightness ${Math.round(m.centroid_hz)} Hz, noisiness ${m.flatness}${m.broken.length ? "\n" + m.broken.join("; ") : ""}`;
      html += `<td><button class="tc ${cls} ${m.value === c.default ? "def" : ""}" data-src="/api/calibration/clip/${folder}/${pid}-s${seed}-${fmtV(m.value)}.flac" title="${esc(tip)}"><b>${m.level_db == null ? "–" : (m.level_db >= 0 ? "+" : "") + Math.round(m.level_db)}</b><i>${m.timbre_db == null ? "" : m.timbre_db.toFixed(0)}</i></button></td>`;
    }
    html += "</tr>";
  }
  $("#tunegrid").innerHTML = html;
  const fixed = c.kind === "placement" || !c.safe;
  ["#tunelo", "#tunehi", "#tunesave", "#tuneclear"].forEach((q) => ($(q).hidden = fixed));
  $("#tunenote").textContent = T.ctl.startsWith("unit:")
    ? `Each cell is three seconds of a preset with ${unitTitle(T.ctl.split(":")[1])} at that setting (its other settings at their defaults here). The figures are the level, and how far the spectrum moved with the level taken out, both against the same preset with no unit. Red: the measures found it broken.${c.kind === "placement" ? " The unit's card marks the placements that break with ⚠." : ""}${c.moved_default ? ` VAU's default (${fmtV(c.vau_default)}) breaks the sound here; the default is now ${fmtV(c.default)}.` : ""}`
    : "Each cell is three seconds of a preset with this control at that setting, and everything else at its default. Click to listen. Red: the measures found it broken. Dim: outside the knob's range now. The figures are the level, and how far the spectrum moved with the level taken out, both against the default.";
  if (fixed) { $("#tunefloor").textContent = ""; return; }
  const below = vals.filter((v) => v <= c.default && v >= c.safe[0]), above = vals.filter((v) => v >= c.default && v <= c.safe[1]);
  $("#tunelo").innerHTML = below.map((v) => `<option ${Math.abs(v - lo) < 1e-9 ? "selected" : ""}>${fmtV(v)}</option>`).join("");
  $("#tunehi").innerHTML = above.map((v) => `<option ${Math.abs(v - hi) < 1e-9 ? "selected" : ""}>${fmtV(v)}</option>`).join("");
  const fl = Object.values(T.grid.floor || {});
  $("#tunefloor").textContent = fl.length ? `Seeds alone move the spectrum about ${(fl.reduce((a, f) => a + f.timbre_db, 0) / fl.length).toFixed(1)} dB: a smaller figure is the music varying by itself.` : "";
}
$("#tunegrid").addEventListener("click", (e) => {
  const b = e.target.closest(".tc"); if (!b) return;
  const a = $("#audition");
  document.querySelectorAll(".tc.playing").forEach((x) => x.classList.remove("playing"));
  if (a.dataset.cur === b.dataset.src && !a.paused) { a.pause(); return; }
  a.src = b.dataset.src; a.dataset.cur = b.dataset.src; a.play().catch(fail); b.classList.add("playing");
});
$("#tunesave").addEventListener("click", async () => {
  try { const r = await api("/api/calibration/ears", { control: T.ctl, lo: +$("#tunelo").value, hi: +$("#tunehi").value }); Object.assign(CAL, r); info(`${ctlLabel(T.ctl)}: the knob now keeps to ${$("#tunelo").value} to ${$("#tunehi").value}`, "ok"); buildDoses(); engKnobs.forEach((k) => k.paint()); loadTune(); }
  catch (e) { fail(e); }
});
$("#tuneclear").addEventListener("click", async () => {
  try { const r = await api("/api/calibration/ears", { control: T.ctl, clear: true }); Object.assign(CAL, r); buildDoses(); engKnobs.forEach((k) => k.paint()); loadTune(); } catch (e) { fail(e); }
});

// ------------------------------------------------------------- start
setView(view);
drawTiles(); drawEditor();
loadPresets().catch(() => {});
loadMine();
poll(); pollDesk(); pollSounds(); listExports();
requestAnimationFrame(animate);


// ------------------------------------------------------------- settings: the model's data, and about
const SET = { tab: "model", data: null };
const kv = (rows) => `<dl>${rows.filter((r) => r[1] !== undefined && r[1] !== null && r[1] !== "").map(([k, v]) => `<dt>${esc(k)}</dt><dd>${v}</dd>`).join("")}</dl>`;
function sparkline(vals) {
  if (!vals || !vals.length) return "";
  const w = 600, hh = 46, mx = Math.max(...vals), mn = Math.min(...vals);
  const pts = vals.map((v, i) => `${(i / (vals.length - 1)) * w},${hh - 4 - ((v - mn) / (mx - mn || 1)) * (hh - 8)}`).join(" ");
  return `<svg class="spark" viewBox="0 0 ${w} ${hh}" preserveAspectRatio="none"><polyline points="${pts}" fill="none" stroke="#d9aa3f" stroke-width="2"/></svg>`;
}
function drawSettings() {
  document.querySelectorAll("#setpanel .settabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.t === SET.tab));
  const box = $("#setbody"), d = SET.data;
  if (!d) { box.innerHTML = `<p>Reading…</p>`; return; }
  const sw = d.software;
  if (SET.tab === "about") {
    const year = sw.year, harvard = `Berry, D. M. (${year}) ${sw.name}, version ${sw.version}. Software, ${sw.licence}. Available at: ${sw.repository}`;
    const bib = `@software{berry_abliterator_${year},\n  author  = {Berry, David M.},\n  title   = {${sw.name}},\n  version = {${sw.version}},\n  year    = {${year}},\n  license = {${sw.licence}},\n  url     = {${sw.repository}}\n}`;
    box.innerHTML = `<div class="who"><img src="mark.svg" alt=""><div><b>${esc(sw.name)}</b>Version ${esc(sw.version)} · alpha · ${esc(sw.licence)}</div></div>
      <p>Vector bending for a music model, played live: steer, squash and turn the vectors the model computes with, between its transformer layers, while it plays. A <a href="${sw.lab}" target="_blank" rel="noopener">Vector Lab</a> tool.</p>
      <h4>Author</h4>${kv([["Author", esc(sw.author)], ["Institution", esc(sw.institution)], ["ORCID", `<a href="https://orcid.org/${sw.orcid}" target="_blank" rel="noopener">${sw.orcid}</a>`], ["Licence", esc(sw.licence)], ["Repository", `<a href="${sw.repository}" target="_blank" rel="noopener">${esc(sw.repository.replace("https://", ""))}</a>`]])}
      <h4>How to cite it</h4><div class="cite" id="cite-h">${esc(harvard)}<button data-c="cite-h">Copy</button></div><div class="cite" id="cite-b">${esc(bib.replace(/\\n/g, "\n"))}<button data-c="cite-b">Copy</button></div>
      <h4>More on the site</h4><div class="links">
        <a href="${sw.site}" target="_blank" rel="noopener"><b>Abliterator</b><span>The instrument, its sounds and how it works</span></a>
        <a href="${sw.site}#chain" target="_blank" rel="noopener"><b>Signal chain</b><span>Engine, Model layers, Speed, Effects and Output, each with a page of its own</span></a>
        <a href="${sw.site}units.html" target="_blank" rel="noopener"><b>Model layers</b><span>Where the VAU units live and act, unit by unit</span></a>
        <a href="${sw.site}abliteration.html" target="_blank" rel="noopener"><b>What is Abliteration?</b><span>The word, and what it means for sound design</span></a>
        <a href="${sw.site}faq.html" target="_blank" rel="noopener"><b>FAQ</b><span>Questions people ask about it</span></a>
        <a href="${sw.repository}" target="_blank" rel="noopener"><b>GitHub</b><span>The code, the bench scripts and the tests</span></a>
        <a href="https://vector-lab-tools.github.io/vector-abliteration-units/" target="_blank" rel="noopener"><b>VAU</b><span>The unit standard this instrument hosts</span></a></div>
      <h4>Credits</h4><p>Magenta RealTime 2 by Google DeepMind's Magenta team (weights CC-BY 4.0, code Apache-2.0). MusicGen small by Meta (weights CC-BY-NC 4.0). Terence Broad's network bending and active divergence. Paul Nasca's Paulstretch, for Smear. Signalsmith Audio's “Let's write a reverb”.</p>`;
    box.querySelectorAll(".cite button").forEach((b) => b.addEventListener("click", async () => {
      const t = $("#" + b.dataset.c).firstChild.textContent;
      try { await navigator.clipboard.writeText(t); b.textContent = "Copied"; setTimeout(() => { b.textContent = "Copy"; }, 1200); } catch (e) { fail(e); }
    }));
    return;
  }
  const m = d.model || {}, mi = m.model || {}, t = m.timing || {}, v = m.versions || {}, st = m.stream_stats, cal = m.calibration, mc = m.machine || {};
  let h = `<h4>Model in use</h4><div class="engpick" id="engpick" role="group" aria-label="The model that plays"></div>
    <p class="setnote">Magenta RT 2 small is the instrument's model: Steer and the Model layers both act on it. MusicGen small plays with Steer and no model-layer units.</p>`;
  h += `<h4>Model</h4>` + kv([["Engine", esc(mi.model_id || d.engine)], ["Package", esc(mi.package)], ["Weights", esc(mi.dtype)], ["Device", esc(mi.device)],
    ["Licence", esc(mi.licence)], ["Sample rate", mi.sample_rate && `${mi.sample_rate} Hz, ${mi.channels} channels`], ["Frame", mi.frame_ms && `${mi.frame_ms} ms (${Math.round(mi.sample_rate * mi.frame_ms / 1000)} samples)`],
    ["Transformer layers", mi.n_layers], ["Steer layer", mi.steer_layer], ["Loaded in", m.load_s && `${m.load_s} s`], ["Presets", d.presets]]);
  h += `<h4>Timing</h4>` + kv([["Buffer", t.lead_frames && `${t.lead_frames} frames (${t.lead_frames * t.frame_ms} ms)`], ["Declared latency", t.latency && `${Math.round(t.latency.declared_first_ms)} ms from a move to the sound`],
    ["Measured latency", t.latency && t.latency.measured_median_ms != null ? `${Math.round(t.latency.measured_median_ms)} ms (median of the last ${t.latency.measured_ms.length})` : "play and move a knob to measure"],
    ["Step time", t.step_ms_p50 != null ? `${t.step_ms_p50} ms median, ${t.step_ms_p95} ms at the 95th percentile, budget ${t.frame_ms} ms` : "while playing"], ["Dropouts", t.underruns], ["Sound card block", t.block && `${t.block} samples`]]);
  if (m.directions && m.directions.length) h += `<h4>Directions</h4><table><tr><th>Name</th><th>Pairs</th><th>Separation at layer ${mi.steer_layer}</th><th>Best layer</th><th>Made</th></tr>${m.directions.map((x) => `<tr><td>${esc(x.name)}</td><td>${x.pairs}</td><td>${x.separation_at_steer ?? "–"}</td><td>${x.best_layer ?? "–"}</td><td>${esc(x.made || "")}</td></tr>`).join("")}</table>`;
  if (st) h += `<h4>Stream statistics</h4>` + kv([["Measured", esc(st.made)], ["Reference prompts", st.prompts], ["Samples", `${st.samples} frames (${st.frames} a prompt, the first ${st.skip} skipped)`], ["Stream", `${st.layers} layers × ${st.width} numbers`]]) + `<p style="margin:6px 0 0;color:#8f95a1">The stream's typical size, layer by layer:</p>` + sparkline(st.norm_by_layer);
  if (cal) h += `<h4>Calibration</h4>` + kv([["Measured", esc(cal.made)], ["Renders", cal.renders], ["Controls", cal.controls], ["Presets", esc((cal.presets || []).join(", "))]]);
  if (m.units) h += `<h4>VAU</h4>` + kv([["Units on this engine", m.units.available], ["Places it offers", esc((m.units.capabilities || []).join(", "))], ["On the Model layers row", m.chain && m.chain.length ? esc(m.chain.join("; ")) : "none"], ["Modulation links", m.routes]]);
  h += `<h4>Runtime</h4>` + kv([["Versions", esc(Object.entries(v).map(([k, x]) => `${k} ${x}`).join(" · "))], ["Python", esc(`engine ${m.python || "?"} · server ${d.server_python}`)], ["Machine", esc(`${mc.chip || ""} · ${mc.memory_gb ? mc.memory_gb + " GB" : ""} · ${mc.platform || ""}`)]]);
  if (d.model_error) h += `<p>${esc(d.model_error)}</p>`;
  box.innerHTML = h;
  if (S.status) drawEngine(S.status);
  $("#engpick").addEventListener("click", (e) => { const b = e.target.closest("button[data-e]"); if (b) switchEngine(b.dataset.e); });
}
async function openSettings(tab) {
  if (tab) SET.tab = tab;
  $("#setpanel").hidden = false; drawSettings();
  try { SET.data = await api("/api/about"); } catch (e) { SET.data = null; $("#setbody").innerHTML = `<p>${esc(e.message)}</p>`; return; }
  drawSettings();
}
$("#setbtn").addEventListener("click", () => openSettings());
$("#setclose").addEventListener("click", () => { $("#setpanel").hidden = true; });
$("#setpanel").addEventListener("click", (e) => { if (e.target.id === "setpanel") $("#setpanel").hidden = true; });
document.querySelectorAll("#setpanel .settabs button").forEach((b) => b.addEventListener("click", () => { SET.tab = b.dataset.t; drawSettings(); }));
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#setpanel").hidden) $("#setpanel").hidden = true; });
$("#brand").addEventListener("click", () => openSettings("about"));
$("#brand").addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openSettings("about"); } });
$("#unithelp .uhclose").addEventListener("click", closeHelp);
$("#unithelp").addEventListener("click", (e) => { if (e.target.id === "unithelp") closeHelp(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeHelp(); });

// The stereo field: each left-right sample pair as a dot, mid (L+R) up and side (L−R) across, so
// mono is a vertical line and out of phase a horizontal one; each reading fades the last, and the
// bar under it marks the correlation, −1 to +1.
const gf = $("#gonio"), gfx = gf.getContext("2d");
let gGain = 1;
function drawField(f) {
  const w = gf.width, c = w / 2, R0 = c - 3, mark = document.querySelector(".gcorr b");
  gfx.save(); gfx.beginPath(); gfx.arc(c, c, R0, 0, 2 * Math.PI); gfx.clip();
  gfx.fillStyle = "rgba(13,15,19,0.45)"; gfx.fillRect(0, 0, w, w);
  gfx.strokeStyle = "#1f232b"; gfx.lineWidth = 1.5;
  gfx.beginPath(); gfx.arc(c, c, R0 * 0.5, 0, 2 * Math.PI); gfx.moveTo(c, c - R0); gfx.lineTo(c, c + R0); gfx.moveTo(c - R0, c); gfx.lineTo(c + R0, c);
  const k = R0 * Math.SQRT1_2; gfx.moveTo(c - k, c - k); gfx.lineTo(c + k, c + k); gfx.moveTo(c + k, c - k); gfx.lineTo(c - k, c + k); gfx.stroke();
  gfx.fillStyle = "#4a5160"; gfx.font = "bold 18px ui-monospace, Menlo, monospace"; gfx.fillText("L", c - k + 2, c - k + 22); gfx.fillText("R", c + k - 14, c - k + 22);
  if (!f || !f.lr || !f.lr.length) { gfx.restore(); mark.style.display = "none"; return; }
  const pk = Math.max(1e-4, ...f.lr.map(([l, r]) => Math.max(Math.abs(l + r), Math.abs(l - r)) / Math.SQRT2));
  gGain = 0.8 * gGain + 0.2 * (0.85 / pk);                     // follows the level, so quiet music still fills it
  gfx.fillStyle = "rgba(63,182,217,0.55)";
  for (const [l, r] of f.lr) {
    const x = c + ((r - l) / Math.SQRT2) * gGain * c, y = c - ((l + r) / Math.SQRT2) * gGain * c;
    gfx.fillRect(x - 1.2, y - 1.2, 2.4, 2.4);
  }
  gfx.restore();
  if (f.corr === null || f.corr === undefined) { mark.style.display = "none"; return; }
  mark.style.display = "block"; mark.style.left = `${(f.corr + 1) * 50}%`;
  mark.parentElement.title = `Left-right correlation ${f.corr.toFixed(2)}: +1 mono, 0 wide, −1 out of phase`;
}
drawField(null);

// the output meters: left and right as bars (level) with a tick (peak), on a dB scale from −60
const OM = (() => {
  const el = $("#outm"), pos = (db) => clamp01((db + 60) / 60) * 100, dbOf = (v) => (v > 0 ? 20 * Math.log10(v) : -Infinity);
  el.innerHTML = `${["L", "R"].map((c) => `<div class="ob"><span>${c}</span><div class="ot"><i></i><b></b></div></div>`).join("")}
    <div class="os">${[-60, -40, -20, 0].map((d) => `<span style="left:${pos(d)}%">${d}</span>`).join("")}</div>
    <div class="ost"><div><b id="om-pk">–</b>Peak</div><div><b id="om-rms">–</b>RMS</div><div><b id="om-cor">–</b>Corr.</div></div>`;
  const bars = [...el.querySelectorAll(".ot")];
  const fmt = (db) => (isFinite(db) ? db.toFixed(1) : "–");
  return {
    set(s) {
      const on = !!s.playing, lr = (on && s.level_lr) || [0, 0], pk = (on && s.peak_lr) || [0, 0];
      bars.forEach((b, i) => {
        const d = dbOf(lr[i]), p = dbOf(pk[i]), fill = b.querySelector("i"), tick = b.querySelector("b");
        fill.style.width = `${pos(d)}%`; fill.className = d > -3 ? "r" : d > -12 ? "o" : "";
        tick.style.left = `${pos(p)}%`; tick.style.display = isFinite(p) ? "block" : "none"; tick.classList.toggle("r", p > -1); tick.classList.toggle("o", p > -6 && p <= -1);
      });
      const p = dbOf(Math.max(...pk)), r = dbOf(s.level || 0);
      $("#om-pk").textContent = on ? fmt(p) : "–"; $("#om-pk").className = on && p > -0.5 ? "bad" : "";
      $("#om-rms").textContent = on ? fmt(r) : "–";
      const c = on && s.field ? s.field.corr : null;
      $("#om-cor").textContent = c == null ? "–" : (c >= 0 ? "+" : "") + c.toFixed(2); $("#om-cor").className = c != null && c < 0 ? "bad" : "";
      // the larger of the 10-second 95th percentile and the slowest step of the last two seconds: a few long steps are what drop out
      const load = on && s.step_ms_p95 != null ? Math.round((Math.max(s.step_ms_p95, s.step_ms_peak || 0) / (S.frameMs || 40)) * 100) : null;
      const cpu = s.cpu_pct;
      $("#om-cpu").textContent = cpu == null ? "–" : `${Math.round(cpu)}%`; $("#om-cpu").className = cpu != null && cpu >= 80 ? "bad" : "";
      $("#om-load").textContent = load == null ? "–" : `${load}%`; $("#om-load").className = load != null && load >= 100 ? "bad" : "";
    },
  };
})();

// Steer's dose history: the effective doses (knob plus modulation) over the last ten seconds
const DH = { t: [], d: {} };
const DHCOL = ["#f3c969", "#4ac6e8", "#c38bff", "#7ddc84"];
function drawDoseHist(s) {
  const c = $("#dosehist"); if (!c) return;
  const g = c.getContext("2d"), w = c.width, hh = c.height, now = performance.now();
  const eff = s.playing && s.eff ? s.eff.doses : null;
  if (eff) { DH.t.push(now); for (const k in eff) (DH.d[k] = DH.d[k] || []).push(eff[k]); }
  while (DH.t.length && now - DH.t[0] > 10000) { DH.t.shift(); for (const k in DH.d) DH.d[k].shift(); }
  g.clearRect(0, 0, w, hh);
  const [lo, hi] = range(), y = (v) => hh / 2 - (v / Math.max(Math.abs(lo), Math.abs(hi), 1e-6)) * (hh / 2 - 6);
  g.strokeStyle = "#20242c"; g.lineWidth = 1; g.beginPath(); g.moveTo(0, hh / 2); g.lineTo(w, hh / 2); g.stroke();
  const keys = Object.keys(DH.d);
  keys.forEach((k, j) => {
    const v = DH.d[k]; if (!v.length) return;
    g.strokeStyle = DHCOL[j % DHCOL.length]; g.lineWidth = 2.5; g.beginPath();
    v.forEach((x, i) => { const px = w - ((now - DH.t[i]) / 10000) * w; i ? g.lineTo(px, y(x)) : g.moveTo(px, y(x)); });
    g.stroke();
  });
  const key = document.querySelector(".dosehist .dhk"), names = keys.map((k, j) => `<span><i style="background:${DHCOL[j % DHCOL.length]}"></i>${esc(S.dirs.find((d) => d.name === k)?.title || k)}</span>`).join("") + `<span style="margin-left:auto">10 s</span>`;
  if (key.innerHTML !== names) key.innerHTML = names;
}
// Output's automatic stages: auto level's gain (−12 to +18 dB about the middle) and the limiter's pull on the peak
const ST = (() => {
  const el = $("#stages");
  el.innerHTML = `<div class="sg"><span>Auto</span><div class="sb"><i></i></div><span class="sv">–</span></div><div class="sg"><span>Limit</span><div class="sb lim"><i></i></div><span class="sv">–</span></div>`;
  const [a, l] = el.querySelectorAll(".sg");
  return { set(s) {
    const st = s.playing ? s.stages : null, on = S.master || {};
    const ad = st && on.level ? st.auto_db : null, ld = st && on.limit ? st.limit_db : null;
    const ai = a.querySelector("i"), li = l.querySelector("i");
    if (ad == null) { ai.style.width = "0"; a.querySelector(".sv").textContent = on.level ? "–" : "off"; }
    else { const f = Math.min(50, (Math.abs(ad) / 18) * 50); ai.style.left = ad >= 0 ? "50%" : `${50 - f}%`; ai.style.width = `${f}%`; a.querySelector(".sv").textContent = `${ad >= 0 ? "+" : ""}${ad.toFixed(1)}`; }
    if (ld == null) { li.style.width = "0"; l.querySelector(".sv").textContent = on.limit ? "–" : "off"; }
    else { li.style.width = `${Math.min(100, (-ld / 6) * 100)}%`; l.querySelector(".sv").textContent = ld.toFixed(1); }
  } };
})();
