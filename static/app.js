"use strict";
// Spray Lab window. The server renders; this page holds the layer stack (bottom layer first, the order they are
// applied), edits it, and asks for previews. The list shows the top layer first, like Effect.app.

const $ = (id) => document.getElementById(id);
const el = (tag, cls, txt) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (txt != null) e.textContent = txt;
  return e;
};
const icon = (name) => `<svg><use href="#i-${name}"/></svg>`;
const clamp = (v, a, b) => Math.min(Math.max(v, a), b);
const deep = (o) => JSON.parse(JSON.stringify(o));
const rid = () => Math.random().toString(16).slice(2, 10);
const nice = (s) => s.charAt(0).toUpperCase() + s.slice(1).replace(/-/g, " ");
const SVGNS = "http://www.w3.org/2000/svg";

let ST = null; // static info from the server: presets, textures, blend modes, defaults
let media = null;
let params = { layers: [] };
let selId = null;
let selStop = null; // the selected gradient stop object
let frame = 0;
let zoom = "fit";
let pan = { x: 0, y: 0 };
let holdOrig = false;
let playing = false;
const play = { t0: 0, s0: 0 }; // at performance.now() == t0 the playhead was s0 seconds into the clip
let audioClock = false; // the clip's sound drove the playhead on the last tick
let lat = 0; // how long a preview frame takes to arrive while playing (ms, smoothed)
let muted = false;
let activePreset = null;

// --------------------------------------------------------------------------- helpers

async function api(path, body) {
  const opt = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const r = await fetch(path, opt);
  if (!r.ok) {
    let msg = r.statusText;
    try { msg = (await r.json()).detail || msg; } catch (e) { /* not json */ }
    throw new Error(msg);
  }
  return r.json();
}

let toastTimer = 0;
function toast(msg, err) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.toggle("err", !!err);
  t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), err ? 6000 : 3200);
}

function fmtTime(s) {
  const m = Math.floor(s / 60);
  return `${m}:${(s - m * 60).toFixed(2).padStart(5, "0")}`;
}

const hex2rgb = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const rgb2hex = (c) => "#" + c.map((v) => Math.round(clamp(v, 0, 255)).toString(16).padStart(2, "0")).join("");

// the same maths as effect.gradient_lut: `mid` is where two neighbouring colours are half-and-half
function sampleGradient(stops, x) {
  const s = [...stops].sort((a, b) => a.pos - b.pos);
  if (x <= s[0].pos) return hex2rgb(s[0].color);
  if (x >= s[s.length - 1].pos) return hex2rgb(s[s.length - 1].color);
  for (let i = 0; i < s.length - 1; i++) {
    const a = s[i], b = s[i + 1];
    if (x >= a.pos && x <= b.pos) {
      if (b.pos <= a.pos) return hex2rgb(b.color);
      let u = (x - a.pos) / (b.pos - a.pos);
      const mid = clamp(a.mid ?? 0.5, 0.05, 0.95);
      u = Math.pow(u, Math.log(0.5) / Math.log(mid));
      const ca = hex2rgb(a.color), cb = hex2rgb(b.color);
      return ca.map((v, k) => v + (cb[k] - v) * u);
    }
  }
  return hex2rgb(s[s.length - 1].color);
}

function cssGradient(stops) {
  const n = 48, parts = [];
  for (let i = 0; i <= n; i++) parts.push(`${rgb2hex(sampleGradient(stops, i / n))} ${(i / n * 100).toFixed(2)}%`);
  return `linear-gradient(90deg, ${parts.join(", ")})`;
}

const selLayer = () => params.layers.find((l) => l.id === selId) || null;
const anyBoil = () => params.layers.some((l) => l.type === "mix" && l.visible && l.boil > 0);

// --------------------------------------------------------------------------- history + persistence

const hist = { stack: [], i: -1, timer: 0 };

function save() {
  try {
    localStorage.setItem("spraylab.params", JSON.stringify(params));
    localStorage.setItem("spraylab.preset", activePreset || "");
  } catch (e) { /* storage off */ }
}

function restore() {
  try {
    const p = JSON.parse(localStorage.getItem("spraylab.params") || "null");
    if (p && Array.isArray(p.layers) && p.layers.length) {
      params = p;
      activePreset = localStorage.getItem("spraylab.preset") || null;
      return true;
    }
  } catch (e) { /* storage off */ }
  return false;
}

function commit(now) {
  clearTimeout(hist.timer);
  const go = () => {
    const s = JSON.stringify(params);
    if (hist.stack[hist.i] === s) return;
    hist.stack = hist.stack.slice(0, hist.i + 1);
    hist.stack.push(s);
    if (hist.stack.length > 120) hist.stack.shift();
    hist.i = hist.stack.length - 1;
    save();
  };
  if (now) go(); else hist.timer = setTimeout(go, 350);
}

function stepHistory(d) {
  commit(true);
  const i = hist.i + d;
  if (i < 0 || i >= hist.stack.length) return;
  hist.i = i;
  params = JSON.parse(hist.stack[i]);
  if (!selLayer()) selId = params.layers.length ? params.layers[params.layers.length - 1].id : null;
  renderLayers();
  buildSettings();
  requestPreview();
  save();
}

// every edit lands here
function changed(keepPreset) {
  if (!keepPreset && activePreset) {
    activePreset = null;
    markPresets();
  }
  updateTransport();
  requestPreview();
  commit(false);
}

// --------------------------------------------------------------------------- preview

let want = 0, inflight = false, lastUrl = null;

function requestPreview() {
  want++;
  pump();
}

async function pump() {
  if (inflight || !media) return;
  inflight = true;
  const ticket = want;
  const sent = performance.now();
  try {
    const r = await fetch("/api/preview", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame, side: previewSide(), params, original: holdOrig }),
    });
    if (r.ok) {
      const url = URL.createObjectURL(await r.blob());
      const v = $("view");
      await new Promise((res) => { v.onload = res; v.onerror = res; v.src = url; });
      if (playing) {
        const ms = performance.now() - sent;
        lat = lat ? lat * 0.7 + ms * 0.3 : ms;
      }
      if (lastUrl) URL.revokeObjectURL(lastUrl);
      lastUrl = url;
      $("renderMs").textContent = holdOrig ? "" : `${r.headers.get("X-Size")} · ${r.headers.get("X-Render-Ms")} ms`;
    } else {
      const j = await r.json().catch(() => ({}));
      toast(j.detail || "Preview failed", true);
      setPlaying(false);
    }
  } catch (e) {
    setPlaying(false);
  }
  inflight = false;
  if (playing) {
    advance();
    want++;
    pump();
  } else if (want !== ticket) {
    pump();
  }
}

function fitScale() {
  const r = $("viewport").getBoundingClientRect();
  return Math.min(r.width / media.width, r.height / media.height);
}

const curScale = () => (zoom === "fit" ? fitScale() : +zoom);

function previewSide() {
  const s = curScale() * (window.devicePixelRatio || 1);
  return Math.max(64, Math.round(Math.max(media.width, media.height) * Math.min(1, s)));
}

function layoutView() {
  if (!media) return;
  const vp = $("viewport").getBoundingClientRect();
  const s = curScale();
  const w = media.width * s, h = media.height * s;
  if (zoom === "fit") pan = { x: 0, y: 0 };
  const mx = Math.max(0, (w - vp.width) / 2), my = Math.max(0, (h - vp.height) / 2);
  pan.x = clamp(pan.x, -mx, mx);
  pan.y = clamp(pan.y, -my, my);
  const v = $("view");
  v.style.width = w + "px";
  v.style.height = h + "px";
  v.style.left = ((vp.width - w) / 2 + pan.x) + "px";
  v.style.top = ((vp.height - h) / 2 + pan.y) + "px";
  v.classList.toggle("pix", s * (window.devicePixelRatio || 1) > 1.5);
  $("viewport").classList.toggle("pan", w > vp.width + 1 || h > vp.height + 1);
  $("zoomSel").options[0].textContent = `Fit · ${Math.round(fitScale() * 100)}%`;
}

// --------------------------------------------------------------------------- playback

function updateTransport() {
  const show = media && (media.kind === "video" || anyBoil());
  $("transport").classList.toggle("hidden", !show);
  if (!show) {
    if (playing) setPlaying(false);
    return;
  }
  const vid = media.kind === "video";
  $("scrub").classList.toggle("hidden", !vid);
  if (vid) {
    $("scrub").max = media.frames - 1;
    $("scrub").value = frame;
    $("timeLabel").textContent = `${fmtTime(frame / media.fps)} / ${fmtTime(media.duration)}`;
  } else {
    $("timeLabel").textContent = playing ? `boil preview · frame ${frame}` : "Play to preview the boil";
  }
}

function setPlaying(on) {
  if (!media) on = false;
  if (on && media.kind !== "video" && !anyBoil()) on = false;
  playing = on;
  $("playBtn").innerHTML = icon(on ? "pause" : "play");
  if (on) {
    startClock(frame / (media.kind === "video" ? media.fps : 24));
    requestPreview();
  } else {
    $("audio").pause();
    if (media && media.kind !== "video") {
      frame = 0;
      requestPreview();
    }
  }
  updateTransport();
}

// While a clip with sound plays, the sound is the clock and the picture follows it. Without sound (or until it
// has loaded) the wall clock runs.
const hasSound = () => !!(media && media.kind === "video" && media.has_audio && $("audio").getAttribute("src"));

function startClock(s) {
  play.s0 = s;
  play.t0 = performance.now();
  audioClock = false;
  if (!hasSound()) return;
  const a = $("audio");
  try { a.currentTime = s; } catch (e) { /* not loaded yet */ }
  a.play().catch(() => { /* no sound: the wall clock runs */ });
}

// seconds into the clip, now
function playhead() {
  const a = $("audio"), now = performance.now();
  const wall = play.s0 + (now - play.t0) / 1000;
  if (!hasSound() || a.paused || a.ended || a.seeking || a.readyState < 3) {
    audioClock = false;
    return wall;
  }
  if (!audioClock && Math.abs(a.currentTime - wall) > 0.3) { // the sound arrived late: bring it to the picture
    a.currentTime = wall;
    return wall;
  }
  audioClock = true;
  play.s0 = a.currentTime; // the wall clock carries on from here if the sound stops (a clip whose sound ends early)
  play.t0 = now;
  return play.s0;
}

function advance() {
  let t = playhead();
  if (media.kind !== "video") {
    frame = Math.floor(t * 24);
  } else {
    if (t >= media.duration) { // the end: start again
      startClock(0);
      t = 0;
    }
    // ask for the frame that will be showing when the render arrives, so the picture lines up with the sound
    const ahead = Math.min(lat, 400) / 1000;
    frame = clamp(Math.floor((t + ahead) * media.fps + 1e-6), 0, media.frames - 1);
  }
  updateTransport();
}

function setMuted(on) {
  muted = on;
  $("audio").muted = on;
  $("muteBtn").innerHTML = icon(on ? "mute" : "sound");
  $("muteBtn").title = on ? "Sound off · press to hear the clip (M)" : "Sound on · press to mute (M)";
  try { localStorage.setItem("spraylab.muted", on ? "1" : ""); } catch (e) { /* storage off */ }
}

// --------------------------------------------------------------------------- layers list

function renderLayers() {
  const ul = $("layerList");
  ul.innerHTML = "";
  if (!params.layers.length) {
    ul.append(el("li", "emptyLayers", "No layers yet. Press + to add one."));
    return;
  }
  [...params.layers].reverse().forEach((lay) => {
    const li = el("li");
    li.dataset.id = lay.id;
    li.draggable = true;
    if (lay.id === selId) li.classList.add("sel");
    if (!lay.visible) li.classList.add("off");
    li.innerHTML = `<svg class="grip"><use href="#i-grip"/></svg><span class="name"></span><span class="act">
      <button class="ib" data-a="trash" title="Delete layer">${icon("trash")}</button>
      <button class="ib" data-a="eye" title="Hide / show">${icon(lay.visible ? "eye" : "eye-off")}</button></span>`;
    li.querySelector(".name").textContent = ST.layer_names[lay.type];
    li.onclick = (e) => {
      const a = e.target.closest("[data-a]");
      if (a) {
        e.stopPropagation();
        if (a.dataset.a === "trash") removeLayer(lay.id);
        else {
          lay.visible = !lay.visible;
          renderLayers();
          changed();
        }
        return;
      }
      selId = lay.id;
      renderLayers();
      buildSettings();
    };
    li.ondragstart = (e) => {
      e.dataTransfer.setData("text/x-layer", lay.id);
      e.dataTransfer.effectAllowed = "move";
    };
    const where = (e) => {
      const r = li.getBoundingClientRect();
      return e.clientY < r.top + r.height / 2;
    };
    li.ondragover = (e) => {
      if (!e.dataTransfer.types.includes("text/x-layer")) return;
      e.preventDefault();
      const above = where(e);
      li.classList.toggle("drop-above", above);
      li.classList.toggle("drop-below", !above);
    };
    li.ondragleave = () => li.classList.remove("drop-above", "drop-below");
    li.ondrop = (e) => {
      const id = e.dataTransfer.getData("text/x-layer");
      li.classList.remove("drop-above", "drop-below");
      if (!id) return;
      e.preventDefault();
      e.stopPropagation();
      if (id !== lay.id) moveLayer(id, lay.id, where(e));
    };
    ul.append(li);
  });
}

function moveLayer(id, targetId, above) {
  const L = params.layers;
  const [lay] = L.splice(L.findIndex((l) => l.id === id), 1);
  let to = L.findIndex((l) => l.id === targetId);
  if (above) to += 1; // higher in the list = later in the stack
  L.splice(to, 0, lay);
  renderLayers();
  changed();
}

async function addLayer(type) {
  $("addMenu").classList.add("hidden");
  try {
    const lay = await api("/api/layer/new", { type });
    const i = params.layers.findIndex((l) => l.id === selId);
    params.layers.splice(i >= 0 ? i + 1 : params.layers.length, 0, lay);
    selId = lay.id;
    renderLayers();
    buildSettings();
    changed();
  } catch (e) {
    toast(e.message, true);
  }
}

function removeLayer(id) {
  const i = params.layers.findIndex((l) => l.id === id);
  if (i < 0) return;
  params.layers.splice(i, 1);
  if (selId === id) {
    const n = params.layers[Math.min(i, params.layers.length - 1)];
    selId = n ? n.id : null;
  }
  renderLayers();
  buildSettings();
  changed();
}

// --------------------------------------------------------------------------- settings controls

function head(label) {
  const row = el("div", "row");
  row.append(el("label", null, label));
  return row;
}

function sliderCtl(lay, key, label, min, max, step, fmt) {
  fmt = fmt || ((v) => Number(v).toFixed(2));
  const d = el("div", "ctl");
  const row = head(label);
  const num = el("input", "num");
  num.value = fmt(lay[key]);
  num.title = "Type a value";
  row.append(num);
  const r = el("input", "rng");
  r.type = "range";
  r.min = min;
  r.max = max;
  r.step = step;
  r.value = lay[key];
  r.title = "Double-click to reset";
  const set = (v) => {
    v = clamp(Number(v), min, max);
    if (step >= 1) v = Math.round(v);
    lay[key] = v;
    r.value = v;
    num.value = fmt(v);
    changed();
  };
  r.oninput = () => set(r.value);
  r.ondblclick = () => set(ST.defaults[lay.type][key]);
  num.onfocus = () => num.select();
  num.onkeydown = (e) => { if (e.key === "Enter") num.blur(); };
  num.onchange = () => {
    const v = parseFloat(num.value);
    if (Number.isFinite(v)) set(v); else num.value = fmt(lay[key]);
  };
  d.append(row, r);
  return d;
}

function selectCtl(lay, key, label, options) {
  const d = el("div", "ctl");
  d.append(head(label));
  const w = el("label", "select");
  const s = el("select");
  options.forEach((o) => {
    const op = el("option", null, nice(o));
    op.value = o;
    s.append(op);
  });
  s.value = lay[key];
  s.onchange = () => { lay[key] = s.value; changed(); };
  w.append(s);
  w.insertAdjacentHTML("beforeend", icon("chev"));
  d.append(w);
  return d;
}

function segCtl(lay, key, label, options) {
  const d = el("div", "ctl");
  d.append(head(label));
  const seg = el("div", "seg");
  options.forEach((o) => {
    const b = el("button", lay[key] === o ? "on" : "", o.toUpperCase());
    b.onclick = () => {
      lay[key] = o;
      seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      changed();
    };
    seg.append(b);
  });
  d.append(seg);
  return d;
}

const texThumbUrl = (id, side) => `/api/texture-thumb?side=${side || 96}&id=${encodeURIComponent(id)}`;

function texLabel(id) {
  const t = ST.textures.find((x) => x.id === id);
  if (t) return t.label;
  return id.startsWith("file:") ? id.split(/[\\/]/).pop() : id;
}

function textureCtl(lay) {
  const d = el("div", "ctl");
  const b = el("button", "tex");
  b.title = "Choose a texture";
  b.innerHTML = `<span class="tl"><b>Texture</b><span></span></span><img alt="">`;
  b.querySelector(".tl span").textContent = texLabel(lay.texture);
  b.querySelector("img").src = texThumbUrl(lay.texture);
  b.onclick = () => openTexPicker(b, lay);
  d.append(b);
  return d;
}

// blend-if ramp: four corners slide sideways, the two end heights slide up and down
function rampCtl(lay, key, label) {
  const d = el("div", "ctl");
  d.append(head(label));
  const W = 272, H = 70, X0 = 16, X1 = 256, YB = 60, YT = 10;
  const X = (v) => X0 + v * (X1 - X0);
  const Y = (h) => YB - h * (YB - YT);
  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("class", "ramp");
  svg.style.height = "auto";
  const mk = (tag, attrs) => {
    const n = document.createElementNS(SVGNS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    svg.append(n);
    return n;
  };
  [0.25, 0.5, 0.75].forEach((g) => mk("line", { x1: X(g), x2: X(g), y1: YT, y2: YB, class: "grid" }));
  mk("line", { x1: X0, x2: X1, y1: YB, y2: YB, class: "grid" });
  const shape = mk("polygon", { class: "shape" });
  const edge = mk("polyline", { class: "edge" });
  const corners = [0, 1, 2, 3].map(() => mk("circle", { r: 5, class: "h" }));
  const ends = [4, 5].map((k) => mk("rect", { width: 10, height: 10, rx: 2, class: "hv", x: k === 4 ? 2 : W - 12 }));
  const draw = () => {
    const [a, b, c, dd, lo, hi] = lay[key];
    const pts = [[0, lo], [a, lo], [b, 1], [c, 1], [dd, hi], [1, hi]];
    const s = pts.map((p) => `${X(p[0])},${Y(p[1])}`);
    shape.setAttribute("points", [...s, `${X(1)},${Y(0)}`, `${X(0)},${Y(0)}`].join(" "));
    edge.setAttribute("points", s.join(" "));
    corners.forEach((h, i) => { h.setAttribute("cx", X(pts[i + 1][0])); h.setAttribute("cy", Y(pts[i + 1][1])); });
    ends[0].setAttribute("y", Y(lo) - 5);
    ends[1].setAttribute("y", Y(hi) - 5);
  };
  const drag = (node, i) => {
    node.onpointerdown = (e) => {
      e.preventDefault();
      node.setPointerCapture(e.pointerId);
      node.classList.add("drag");
      node.onpointermove = (ev) => {
        const r = svg.getBoundingClientRect();
        const px = (ev.clientX - r.left) * W / r.width, py = (ev.clientY - r.top) * H / r.height;
        const v = lay[key];
        if (i < 4) {
          const x = clamp((px - X0) / (X1 - X0), 0, 1);
          v[i] = +clamp(x, i > 0 ? v[i - 1] : 0, i < 3 ? v[i + 1] : 1).toFixed(3);
        } else {
          v[i] = +clamp((YB - py) / (YB - YT), 0, 1).toFixed(3);
        }
        draw();
        changed();
      };
      node.onpointerup = () => { node.onpointermove = null; node.classList.remove("drag"); };
    };
  };
  corners.forEach((h, i) => drag(h, i));
  ends.forEach((h, i) => drag(h, 4 + i));
  svg.ondblclick = () => { lay[key] = [0, 0, 1, 1, 0, 0]; draw(); changed(); };
  const t = svg.appendChild(document.createElementNS(SVGNS, "title"));
  t.textContent = "How much of the texture shows, by brightness (left dark, right light). Drag the dots; " +
    "the squares set the two ends. Double-click to reset.";
  draw();
  d.append(svg);
  return d;
}

function gradientCtl(lay) {
  const d = el("div", "ctl");
  const top = head("Gradient map");
  const rev = el("button", "mini", "Reverse");
  rev.title = "Flip the gradient";
  top.append(rev);
  d.append(top);
  const g = el("div", "grad");
  const bar = el("div", "bar");
  bar.title = "Click to add a colour stop";
  g.append(bar);
  const detail = el("div", "stopRow");
  d.append(g, detail);
  if (!lay.stops.includes(selStop)) selStop = lay.stops[0];

  const redraw = () => {
    bar.style.background = cssGradient(lay.stops);
    g.querySelectorAll(".stop, .mid").forEach((n) => n.remove());
    const sorted = [...lay.stops].sort((a, b) => a.pos - b.pos);
    sorted.forEach((s, i) => {
      const h = el("div", "stop" + (s === selStop ? " sel" : ""));
      h.style.left = s.pos * 100 + "%";
      const inner = el("i");
      inner.style.background = s.color;
      h.append(inner);
      h.title = "Drag to move, click to change the colour, drag down to remove";
      h.onpointerdown = (e) => stopDrag(e, s, h);
      g.append(h);
      if (i < sorted.length - 1) {
        const n = sorted[i + 1];
        const m = el("div", "mid");
        m.style.left = (s.pos + s.mid * (n.pos - s.pos)) * 100 + "%";
        m.title = "Midpoint: where the two colours are half-and-half";
        m.onpointerdown = (e) => midDrag(e, s, n, m);
        g.append(m);
      }
    });
    fillDetail();
  };

  const posFrom = (e) => {
    const r = g.getBoundingClientRect();
    return clamp((e.clientX - r.left) / r.width, 0, 1);
  };

  const stopDrag = (e, s, h) => {
    e.preventDefault();
    h.setPointerCapture(e.pointerId);
    selStop = s;
    g.querySelectorAll(".stop").forEach((n) => n.classList.toggle("sel", n === h));
    g.querySelectorAll(".mid").forEach((n) => n.classList.add("hidden"));
    const x0 = e.clientX, y0 = e.clientY;
    let moved = false, gone = false;
    h.onpointermove = (ev) => {
      if (!moved && Math.hypot(ev.clientX - x0, ev.clientY - y0) < 3) return;
      moved = true;
      gone = lay.stops.length > 2 && ev.clientY - y0 > 36;
      h.classList.toggle("gone", gone);
      s.pos = +posFrom(ev).toFixed(3);
      h.style.left = s.pos * 100 + "%";
      bar.style.background = cssGradient(gone ? lay.stops.filter((x) => x !== s) : lay.stops);
      fillDetail();
      changed();
    };
    h.onpointerup = () => {
      h.onpointermove = null;
      if (gone) {
        lay.stops.splice(lay.stops.indexOf(s), 1);
        selStop = lay.stops[0];
      }
      redraw();
      if (gone) changed();
      if (!moved) pickColor(h, s);
    };
  };

  const midDrag = (e, s, n, m) => {
    e.preventDefault();
    m.setPointerCapture(e.pointerId);
    m.classList.add("drag");
    m.onpointermove = (ev) => {
      const span = n.pos - s.pos;
      if (span <= 0) return;
      s.mid = +clamp((posFrom(ev) - s.pos) / span, 0.05, 0.95).toFixed(3);
      m.style.left = (s.pos + s.mid * span) * 100 + "%";
      bar.style.background = cssGradient(lay.stops);
      changed();
    };
    m.onpointerup = () => { m.onpointermove = null; m.classList.remove("drag"); };
  };

  bar.onclick = (e) => {
    if (lay.stops.length >= 16) return toast("16 colour stops at most", true);
    const pos = +posFrom(e).toFixed(3);
    const s = { pos, color: rgb2hex(sampleGradient(lay.stops, pos)), mid: 0.5 };
    lay.stops.push(s);
    selStop = s;
    redraw();
    changed();
  };

  rev.onclick = () => {
    const sorted = [...lay.stops].sort((a, b) => a.pos - b.pos);
    const mids = sorted.map((s) => s.mid);
    sorted.forEach((s, i) => {
      s.pos = +(1 - s.pos).toFixed(3);
      // the midpoint of the segment that ends up after this stop is the mirrored old one
      const j = sorted.length - 2 - i;
      s.mid = j >= 0 ? +(1 - mids[j]).toFixed(3) : 0.5;
    });
    redraw();
    changed();
  };

  function fillDetail() {
    detail.innerHTML = "";
    const s = selStop;
    if (!s) return;
    const sw = el("button", "swatch");
    sw.style.background = s.color;
    sw.title = "Pick a colour";
    sw.onclick = () => pickColor(sw, s);
    const hex = el("input", "hex");
    hex.value = s.color;
    hex.onkeydown = (e) => { if (e.key === "Enter") hex.blur(); };
    hex.onchange = () => {
      const v = hex.value.trim().replace(/^#?/, "#");
      if (/^#[0-9a-fA-F]{6}$/.test(v)) { s.color = v.toLowerCase(); redraw(); changed(); } else hex.value = s.color;
    };
    const loc = el("input", "num");
    loc.value = Math.round(s.pos * 100) + "%";
    loc.title = "Location";
    loc.onfocus = () => loc.select();
    loc.onkeydown = (e) => { if (e.key === "Enter") loc.blur(); };
    loc.onchange = () => {
      const v = parseFloat(loc.value);
      if (Number.isFinite(v)) { s.pos = clamp(v / 100, 0, 1); redraw(); changed(); } else loc.value = Math.round(s.pos * 100) + "%";
    };
    const del = el("button", "mini", "Delete");
    del.disabled = lay.stops.length <= 2;
    del.onclick = () => {
      if (lay.stops.length <= 2) return;
      lay.stops.splice(lay.stops.indexOf(s), 1);
      selStop = lay.stops[0];
      redraw();
      changed();
    };
    detail.append(sw, hex, loc, el("span", "spacer"), del);
  }

  function pickColor(anchor, s) {
    const c = $("colorInput");
    const r = anchor.getBoundingClientRect();
    c.style.left = r.left + "px";
    c.style.top = r.bottom + "px";
    c.value = s.color;
    c.oninput = () => {
      s.color = c.value;
      bar.style.background = cssGradient(lay.stops);
      g.querySelectorAll(".stop").forEach((n) => {
        if (n.classList.contains("sel")) n.querySelector("i").style.background = s.color;
      });
      fillDetail();
      changed();
    };
    c.onchange = () => redraw();
    c.click();
  }

  redraw();
  return d;
}

function buildSettings() {
  const box = $("settings");
  box.innerHTML = "";
  const lay = selLayer();
  $("rndBtn").disabled = $("resetBtn").disabled = !lay;
  if (!lay) {
    $("setTitle").textContent = "Settings";
    box.append(el("div", "note", "Select a layer, or add one with +."));
    return;
  }
  $("setTitle").textContent = ST.layer_names[lay.type];
  const add = (n) => box.append(n);
  if (lay.type === "blur") {
    add(sliderCtl(lay, "strength", "strength", 0, 100, 0.01));
    add(sliderCtl(lay, "aspect", "aspect", -1, 1, 0.01));
    add(sliderCtl(lay, "opacity", "opacity", 0, 1, 0.01));
    add(selectCtl(lay, "blend", "Blend mode", ST.blend_modes));
  } else if (lay.type === "mix") {
    add(textureCtl(lay));
    add(segCtl(lay, "fit", "Fit type", ST.fits));
    add(sliderCtl(lay, "scale", "texture scale", 0.25, 4, 0.01));
    add(selectCtl(lay, "channel", "Blend channel", ST.channels));
    add(rampCtl(lay, "ramp_top", "Current layer ramp"));
    add(rampCtl(lay, "ramp_under", "Underlying layer ramp"));
    add(sliderCtl(lay, "opacity", "opacity", 0, 1, 0.01));
    add(selectCtl(lay, "blend", "Blend mode", ST.blend_modes));
    add(sliderCtl(lay, "boil", "boil", 0, 12, 1, (v) => (v ? `${v} fr` : "off")));
    add(el("div", "note", "Boil moves the texture every few frames so videos (and photo clips) shimmer like " +
      "fresh spray. Off keeps it still."));
  } else {
    add(gradientCtl(lay));
    add(sliderCtl(lay, "shift", "Gradient shift", -1, 1, 0.01));
    add(sliderCtl(lay, "repeat", "Gradient repeat", 1, 8, 1, (v) => String(v)));
    add(sliderCtl(lay, "opacity", "opacity", 0, 1, 0.01));
    add(selectCtl(lay, "blend", "Blend mode", ST.blend_modes));
  }
}

async function randomizeSel() {
  const lay = selLayer();
  if (!lay) return;
  try {
    const r = await api("/api/layer/randomize", { layer: lay, seed: Math.floor(Math.random() * 1e9) });
    Object.assign(lay, r);
    selStop = null;
    buildSettings();
    changed();
  } catch (e) {
    toast(e.message, true);
  }
}

function resetSel() {
  const lay = selLayer();
  if (!lay) return;
  Object.assign(lay, deep(ST.defaults[lay.type]));
  selStop = null;
  buildSettings();
  changed();
}

// --------------------------------------------------------------------------- texture picker

let texLayer = null;

function openTexPicker(anchor, lay) {
  texLayer = lay;
  fillTexGrid();
  const p = $("texPicker");
  p.classList.remove("hidden");
  const r = anchor.getBoundingClientRect();
  p.style.left = Math.min(r.right + 16, innerWidth - p.offsetWidth - 12) + "px";
  p.style.top = clamp(r.top - 60, 12, innerHeight - p.offsetHeight - 12) + "px";
}

function fillTexGrid() {
  const g = $("texGrid");
  g.innerHTML = "";
  ST.textures.forEach((t) => {
    const b = el("button", texLayer && texLayer.texture === t.id ? "on" : "");
    const img = el("img");
    img.loading = "lazy";
    img.src = texThumbUrl(t.id, 200);
    b.append(img, el("span", null, t.label));
    b.title = t.builtin ? t.label : t.id.slice(5);
    b.onclick = () => setTexture(t.id);
    g.append(b);
  });
}

function setTexture(id) {
  if (!texLayer) return;
  texLayer.texture = id;
  fillTexGrid();
  buildSettings();
  changed();
}

async function loadOwnTexture() {
  try {
    const r = await api("/api/texture/browse", {});
    if (r.cancelled) return;
    ST.textures = r.textures;
    setTexture(r.id);
  } catch (e) {
    toast(e.message, true);
  }
}

async function uploadTexture(file) {
  try {
    const r = await fetch("/api/texture/upload?name=" + encodeURIComponent(file.name), { method: "POST", body: file });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "Could not use this texture");
    ST.textures = j.textures;
    setTexture(j.id);
  } catch (e) {
    toast(e.message, true);
  }
}

// --------------------------------------------------------------------------- presets

function withNewIds(p) {
  const q = deep(p);
  q.layers.forEach((l) => { l.id = rid(); });
  return q;
}

function applyPreset(p, key) {
  params = withNewIds(p.params);
  const top = params.layers[params.layers.length - 1];
  selId = top ? top.id : null;
  selStop = null;
  activePreset = key;
  renderLayers();
  buildSettings();
  markPresets();
  changed(true);
}

function presetCard(p, key, mine) {
  const b = el("button", "preset");
  b.dataset.key = key;
  const im = el("div", "pimg");
  b.append(im, el("span", null, p.name));
  b.title = p.note || p.name;
  b.onclick = () => applyPreset(p, key);
  if (mine) {
    const x = el("span", "del", "×");
    x.title = "Delete this preset";
    x.onclick = async (e) => {
      e.stopPropagation();
      const r = await api("/api/my-presets/delete", { name: p.name });
      ST.my_presets = r.my_presets;
      buildMyPresets();
    };
    b.append(x);
  }
  return b;
}

function markPresets() {
  document.querySelectorAll(".preset").forEach((b) => b.classList.toggle("on", b.dataset.key === activePreset));
}

function buildPresetGrid() {
  const g = $("presetGrid");
  g.innerHTML = "";
  ST.presets.forEach((p) => g.append(presetCard(p, "p:" + p.name, false)));
}

function buildMyPresets() {
  const g = $("myPresetGrid");
  g.innerHTML = "";
  ST.my_presets.forEach((p) => g.append(presetCard(p, "m:" + p.name, true)));
  $("myEmpty").classList.toggle("hidden", ST.my_presets.length > 0);
  markPresets();
  refreshThumbs();
}

let thumbGen = 0;

async function refreshThumbs() {
  const gen = ++thumbGen;
  if (!media) return;
  const all = [...ST.presets.map((p) => ["p:" + p.name, p]), ...ST.my_presets.map((p) => ["m:" + p.name, p])];
  for (const [key, p] of all) {
    if (gen !== thumbGen) return;
    const b = document.querySelector(`.preset[data-key="${CSS.escape(key)}"] .pimg`);
    if (!b) continue;
    try {
      const r = await fetch("/api/thumb", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ params: p.params, side: 360, frame: media.kind === "video" ? frame : 0 }),
      });
      if (!r.ok) continue;
      const url = URL.createObjectURL(await r.blob());
      if (b.dataset.url) URL.revokeObjectURL(b.dataset.url);
      b.dataset.url = url;
      b.style.backgroundImage = `url(${url})`;
    } catch (e) {
      return;
    }
  }
}

function openNamePrompt() {
  const p = $("namePrompt");
  const r = $("savePresetBtn").getBoundingClientRect();
  p.classList.remove("hidden");
  p.style.left = (r.right + 14) + "px";
  p.style.top = r.top + "px";
  $("presetName").value = "";
  $("presetName").focus();
}

async function savePreset() {
  const name = $("presetName").value.trim();
  if (!name) return $("presetName").focus();
  try {
    const r = await api("/api/my-presets", { name, params });
    ST.my_presets = r.my_presets;
    $("namePrompt").classList.add("hidden");
    activePreset = "m:" + name;
    buildMyPresets();
    toast(`Saved "${name}"`);
  } catch (e) {
    toast(e.message, true);
  }
}

// --------------------------------------------------------------------------- media

function buildFormatSelect() {
  const s = $("fmtSel");
  const prev = s.value;
  s.innerHTML = "";
  const opts = [];
  if (!media || media.kind === "image") {
    opts.push(["png", "PNG"], ["jpg", "JPG"], ["mp4:24", "MP4 · 24fps"], ["mp4:30", "MP4 · 30fps"]);
  } else {
    const src = Math.round(media.fps * 100) / 100;
    opts.push(["mp4:source", `MP4 · ${src}fps`]);
    if (Math.abs(src - 24) > 0.01) opts.push(["mp4:24", "MP4 · 24fps"]);
    if (Math.abs(src - 30) > 0.01) opts.push(["mp4:30", "MP4 · 30fps"]);
  }
  opts.forEach(([v, t]) => { const o = el("option", null, t); o.value = v; s.append(o); });
  if ([...s.options].some((o) => o.value === prev)) s.value = prev;
  onFormat();
}

function onFormat() {
  const photoClip = media && media.kind === "image" && $("fmtSel").value.startsWith("mp4");
  $("secWrap").classList.toggle("hidden", !photoClip);
  if (photoClip && !anyBoil()) toast("Tip: turn on Boil in Layer Mix so the photo clip moves");
}

async function onMedia(m) {
  setPlaying(false);
  media = m;
  frame = 0;
  lat = 0;
  pan = { x: 0, y: 0 };
  const a = $("audio"), sound = m.kind === "video" && m.has_audio;
  if (sound) a.src = "/api/audio?key=" + encodeURIComponent(m.key); // cut by the server while the clip opens
  else a.removeAttribute("src");
  a.load();
  $("muteBtn").classList.toggle("hidden", !sound);
  let info = `${m.name} · ${m.width}×${m.height}`;
  if (m.kind === "video") info += ` · ${Math.round(m.fps * 100) / 100} fps · ${fmtTime(m.duration)}${m.has_audio ? " · audio" : ""}`;
  $("fileName").textContent = info;
  $("fileName").title = m.path;
  $("empty").classList.add("hidden");
  $("view").style.display = "block";
  buildFormatSelect();
  updateTransport();
  layoutView();
  requestPreview();
  try {
    const r = await fetch("/api/preview", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frame: 0, side: 400, original: true }),
    });
    if (r.ok) $("srcThumb").src = URL.createObjectURL(await r.blob());
  } catch (e) { /* ignore */ }
  refreshThumbs();
}

async function openDialog() {
  try {
    const r = await api("/api/browse", {});
    if (!r.cancelled && r.media) await onMedia(r.media);
  } catch (e) {
    toast(e.message, true);
  }
}

async function uploadMedia(file) {
  toast(`Opening ${file.name}…`);
  try {
    const r = await fetch("/api/upload?name=" + encodeURIComponent(file.name), { method: "POST", body: file });
    const j = await r.json();
    if (!r.ok) throw new Error(j.detail || "Could not open this file");
    await onMedia(j.media);
    $("toast").classList.add("hidden");
  } catch (e) {
    toast(e.message, true);
  }
}

// --------------------------------------------------------------------------- export

let jobTimer = 0;

async function doExport() {
  if (!media) return toast("Open a photo or video first", true);
  const [format, fps] = $("fmtSel").value.split(":");
  commit(true);
  try {
    const j = await api("/api/export", {
      params, format, fps: fps || "source", max_side: +$("sizeSel").value, seconds: +$("secSel").value,
    });
    showJob(j);
    pollJob();
  } catch (e) {
    toast(e.message, true);
  }
}

function pollJob() {
  clearInterval(jobTimer);
  jobTimer = setInterval(async () => {
    try {
      const j = await api("/api/job");
      showJob(j);
      if (j.state !== "running") clearInterval(jobTimer);
    } catch (e) {
      clearInterval(jobTimer);
    }
  }, 300);
}

function showJob(j) {
  const p = $("jobPanel");
  if (!j || j.state === "idle") return p.classList.add("hidden");
  p.classList.remove("hidden");
  const run = j.state === "running";
  $("exportBtn").disabled = run;
  $("cancelBtn").classList.toggle("hidden", !run);
  $("showBtn").classList.toggle("hidden", j.state !== "done");
  $("closeJobBtn").classList.toggle("hidden", run);
  const pct = j.total ? (j.done / j.total) * 100 : 0;
  $("jobBar").style.width = (j.state === "done" ? 100 : pct) + "%";
  let t = "";
  if (run) {
    t = j.total > 1 ? `Exporting ${j.size} · frame ${j.done} / ${j.total}` : `Exporting ${j.size}…`;
    if (j.fps) t += ` · ${j.fps.toFixed(1)} fps · ${Math.ceil(j.eta)} s left`;
  } else if (j.state === "done") {
    t = "Saved " + (j.out || "").split(/[\\/]/).pop();
    if (j.report && j.report.problems && j.report.problems.length) t += " (check: " + j.report.problems.join("; ") + ")";
    $("showBtn").onclick = () => api("/api/reveal", { path: j.out });
  } else if (j.state === "cancelled") {
    t = "Export cancelled";
  } else if (j.state === "error") {
    t = "Export failed: " + j.error;
  }
  $("jobText").textContent = t;
  $("jobText").title = j.out || t;
}

// --------------------------------------------------------------------------- wiring

function wire() {
  $("openBtn").onclick = openDialog;
  $("openBtn2").onclick = openDialog;
  $("addLayerBtn").onclick = (e) => { e.stopPropagation(); $("addMenu").classList.toggle("hidden"); };
  $("addMenu").querySelectorAll("button").forEach((b) => { b.onclick = () => addLayer(b.dataset.type); });
  $("savePresetBtn").onclick = (e) => { e.stopPropagation(); openNamePrompt(); };
  $("nameOk").onclick = savePreset;
  $("nameCancel").onclick = () => $("namePrompt").classList.add("hidden");
  $("presetName").onkeydown = (e) => {
    if (e.key === "Enter") savePreset();
    if (e.key === "Escape") $("namePrompt").classList.add("hidden");
  };
  $("rndBtn").onclick = randomizeSel;
  $("resetBtn").onclick = resetSel;
  $("texLoadBtn").onclick = loadOwnTexture;
  $("exportBtn").onclick = doExport;
  $("cancelBtn").onclick = () => api("/api/cancel", {});
  $("closeJobBtn").onclick = () => $("jobPanel").classList.add("hidden");
  $("fmtSel").onchange = onFormat;
  $("zoomSel").onchange = () => {
    zoom = $("zoomSel").value;
    layoutView();
    requestPreview();
  };
  $("playBtn").onclick = () => setPlaying(!playing);
  $("muteBtn").onclick = () => setMuted(!muted);
  $("audio").onerror = () => {
    if (hasSound()) toast("Spray Lab couldn't load this clip's sound, so the preview plays silent.", true);
  };
  $("scrub").oninput = () => {
    if (playing) setPlaying(false);
    frame = +$("scrub").value;
    updateTransport();
    requestPreview();
  };
  $("scrub").onchange = () => refreshThumbs();
  $("srcThumbWrap").onpointerdown = () => showOriginal(true);
  $("srcThumbWrap").onpointerup = $("srcThumbWrap").onpointerleave = () => showOriginal(false);

  // pan when zoomed in
  const vp = $("viewport");
  vp.onpointerdown = (e) => {
    if (!vp.classList.contains("pan")) return;
    vp.setPointerCapture(e.pointerId);
    vp.classList.add("panning");
    const sx = e.clientX - pan.x, sy = e.clientY - pan.y;
    vp.onpointermove = (ev) => { pan = { x: ev.clientX - sx, y: ev.clientY - sy }; layoutView(); };
    vp.onpointerup = () => { vp.onpointermove = null; vp.classList.remove("panning"); };
  };

  let resizeT = 0;
  window.addEventListener("resize", () => {
    layoutView();
    clearTimeout(resizeT);
    resizeT = setTimeout(requestPreview, 150);
  });

  // close popovers on outside clicks
  document.addEventListener("pointerdown", (e) => {
    if (!e.target.closest("#texPicker") && !e.target.closest(".tex")) $("texPicker").classList.add("hidden");
    if (!e.target.closest("#addMenu") && !e.target.closest("#addLayerBtn")) $("addMenu").classList.add("hidden");
    if (!e.target.closest("#namePrompt") && !e.target.closest("#savePresetBtn")) $("namePrompt").classList.add("hidden");
  });

  // keys
  window.addEventListener("keydown", (e) => {
    const typing = (e.target.tagName === "INPUT" && !["range", "color"].includes(e.target.type)) || e.target.tagName === "SELECT";
    const k = e.key.toLowerCase();
    if (e.ctrlKey && k === "o") { e.preventDefault(); openDialog(); }
    else if (e.ctrlKey && k === "e") { e.preventDefault(); doExport(); }
    else if (typing) return;
    else if (e.ctrlKey && k === "z" && !e.shiftKey) { e.preventDefault(); stepHistory(-1); }
    else if (e.ctrlKey && (k === "y" || (k === "z" && e.shiftKey))) { e.preventDefault(); stepHistory(1); }
    else if (e.code === "Space") { e.preventDefault(); if (!e.repeat) showOriginal(true); }
    else if (!e.ctrlKey && !e.altKey && k === "r") randomizeSel();
    else if (!e.ctrlKey && !e.altKey && k === "p") setPlaying(!playing);
    else if (!e.ctrlKey && !e.altKey && k === "m" && hasSound()) setMuted(!muted);
    else if (media && media.kind === "video" && (k === "arrowleft" || k === "arrowright")) {
      e.preventDefault();
      if (playing) setPlaying(false);
      frame = clamp(frame + (k === "arrowright" ? 1 : -1) * (e.shiftKey ? 10 : 1), 0, media.frames - 1);
      updateTransport();
      requestPreview();
    } else if (k === "escape") {
      ["texPicker", "addMenu", "namePrompt"].forEach((id) => $(id).classList.add("hidden"));
    }
  });
  window.addEventListener("keyup", (e) => { if (e.code === "Space") showOriginal(false); });
  window.addEventListener("blur", () => showOriginal(false));

  // drag and drop files: onto the texture picker = a texture, anywhere else = open it
  let dragDepth = 0;
  const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
  window.addEventListener("dragenter", (e) => {
    if (!hasFiles(e)) return;
    dragDepth++;
    if ($("texPicker").classList.contains("hidden")) $("dropHint").classList.remove("hidden");
  });
  window.addEventListener("dragleave", (e) => {
    if (!hasFiles(e)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) $("dropHint").classList.add("hidden");
  });
  window.addEventListener("dragover", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    $("texPicker").classList.toggle("dropping", !!e.target.closest("#texPicker"));
  });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth = 0;
    $("dropHint").classList.add("hidden");
    $("texPicker").classList.remove("dropping");
    const f = e.dataTransfer.files[0];
    if (!f) return;
    if (e.target.closest("#texPicker")) uploadTexture(f);
    else uploadMedia(f);
  });
}

function showOriginal(on) {
  if (holdOrig === on) return;
  holdOrig = on;
  $("origBadge").classList.toggle("hidden", !on);
  requestPreview();
}

async function init() {
  wire();
  let wasMuted = false;
  try { wasMuted = localStorage.getItem("spraylab.muted") === "1"; } catch (e) { /* storage off */ }
  setMuted(wasMuted);
  try {
    ST = await api("/api/state");
  } catch (e) {
    toast("The Spray Lab server is not answering. Close this window and start it again.", true);
    return;
  }
  if (restore()) {
    try { params = await api("/api/normalize", { params }); } catch (e) { params = { layers: [] }; }
  }
  if (!params.layers.length) {
    params = withNewIds(ST.presets[0].params);
    activePreset = "p:" + ST.presets[0].name;
  }
  selId = params.layers.length ? params.layers[params.layers.length - 1].id : null;
  buildPresetGrid();
  buildMyPresets();
  renderLayers();
  buildSettings();
  buildFormatSelect();
  commit(true);
  if (ST.media) await onMedia(ST.media);
  if (ST.job && ST.job.state === "running") { showJob(ST.job); pollJob(); }
  const ping = () => fetch("/api/ping", { method: "POST" }).catch(() => {});
  ping();
  setInterval(ping, 4000);
}

init();
