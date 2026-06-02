// Replay + Correction — pinned, scroll-driven beat (sketch ④).
//
// Layer A (replay): the canonical DAgger takeover example builds bar-by-bar as
// you scroll — correct prefix (placement gradient) → wrong policy suffix (grey)
// → human correction (orange). Geometry from assets/correction/replay.json,
// coloured exactly like plot_dagger_individuals.py's dagger_failed_corrected.
//
// Smooth transition → Layer B (correction dataset): a grid of corrected demos
// that auto-scrolls horizontally on its own (independent rAF), feeding retraining.
//
//   p 0.00-0.18   prefix bars place (gradient)
//   p 0.18-0.34   wrong policy bars place (grey)
//   p 0.34-0.58   wrong fades; orange correction bars place
//   p 0.58-0.72   smooth transition (replay up/out, dataset up/in)
//   p 0.72-1.00   correction dataset on screen (auto-scrolling continues)

const BASE = "assets/correction";

async function loadJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} -> ${res.status}`);
  return res.json();
}

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
function smooth(v, a, b) {
  const t = clamp01((v - a) / (b - a));
  return t * t * (3 - 2 * t);
}
function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}
function hexToRgb(hex) {
  const h = hex.replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

export async function initCorrection({ full }) {
  const section = document.querySelector("#teaching-correction");
  if (!section) return;

  const stage = section.querySelector(".correction-stage");
  const replayLayer = section.querySelector("#correction-replay");
  const datasetLayer = section.querySelector("#correction-dataset");
  const canvas = section.querySelector("#correction-canvas");
  const rowsHost = section.querySelector("#correction-rows");
  const cap = section.querySelector(".correction-cap");
  if (!canvas || !rowsHost) return;
  const ctx = canvas.getContext("2d");

  let replay, manifest;
  try {
    [replay, manifest] = await Promise.all([
      loadJSON(`${BASE}/replay.json`),
      loadJSON(`${BASE}/manifest.json`),
    ]);
  } catch (err) {
    console.warn("correction: assets missing, leaving static.", err);
    return;
  }

  const COL = {
    barDark: cssVar("--bridge-bar-dark", "#0420bf"),
    barLight: cssVar("--bridge-bar-light", "#9beaf2"),
    wrong: cssVar("--bridge-bar-wrong", "#afb4b9"),
    correction: cssVar("--bridge-correction", "#ff7e43"),
    earth: cssVar("--bridge-earth", "#a3afbf40"),
    nodeStart: cssVar("--bridge-node-start", "#f26b83"),
    nodePrefix: cssVar("--bridge-node-free", "#d2f25e"),
    nodeCorrection: cssVar("--bridge-node-correction", "#ffb384"),
    nodeWrong: cssVar("--bridge-node-wrong", "#c3c8cd"),
    edge: cssVar("--bridge-node-edge", "#2c2c2c"),
    edgeWrong: cssVar("--bridge-node-wrong-edge", "#91969b"),
    edgeCorrection: cssVar("--bridge-correction", "#ff7e43"),
  };

  const nodes = replay.nodes;
  const vp = replay.viewport;
  const maxBars = replay.max_bars || 20;

  function placementColor(order) {
    const t = maxBars > 1 ? order / (maxBars - 1) : 0;
    const a = hexToRgb(COL.barDark), b = hexToRgb(COL.barLight);
    return `rgb(${Math.round(a[0] + (b[0] - a[0]) * t)},${Math.round(
      a[1] + (b[1] - a[1]) * t
    )},${Math.round(a[2] + (b[2] - a[2]) * t)})`;
  }

  // ---- canvas geometry (world → device px) ----
  let W = 0, H = 0, dpr = 1, lw = 4, rNode = 7;
  function sx(wx) {
    return ((wx - vp.x_min) / (vp.x_max - vp.x_min)) * W;
  }
  function sy(wy) {
    return H - ((wy - vp.y_min) / (vp.y_max - vp.y_min)) * H;
  }
  function resizeCanvas() {
    const r = canvas.getBoundingClientRect();
    W = r.width;
    H = r.height;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    lw = Math.max(3, W * 0.015);
    rNode = Math.max(6, W * 0.02); // slightly larger nodes
  }

  function nodeStyle(kind) {
    switch (kind) {
      case "support": return [COL.nodeStart, COL.edge, 1.35];
      case "prefix": return [COL.nodePrefix, COL.edge, 1.0];
      case "correction": return [COL.nodeCorrection, COL.edgeCorrection, 1.0];
      default: return [COL.nodeWrong, COL.edgeWrong, 1.0];
    }
  }

  // Draw a phase's bars progressively; returns map of nodeIdx -> alpha for those
  // whose incident bar is (at least partly) drawn.
  function drawPhase(list, frac, alpha, colorFn, visible) {
    const n = list.length;
    if (n === 0 || frac <= 0 || alpha <= 0) return;
    const drawn = frac * n;
    const full = Math.floor(drawn);
    const partial = drawn - full;
    ctx.lineCap = "round";
    ctx.lineWidth = lw;
    for (let i = 0; i < n; i++) {
      const f = i < full ? 1 : i === full ? partial : 0;
      if (f <= 0) break;
      const A = nodes[list[i].a], B = nodes[list[i].b];
      ctx.strokeStyle = colorFn(list[i], i);
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.moveTo(sx(A.x), sy(A.y));
      ctx.lineTo(sx(A.x) + (sx(B.x) - sx(A.x)) * f, sy(A.y) + (sy(B.y) - sy(A.y)) * f);
      ctx.stroke();
      visible.set(list[i].a, Math.max(visible.get(list[i].a) || 0, alpha));
      if (f >= 1) visible.set(list[i].b, Math.max(visible.get(list[i].b) || 0, alpha));
    }
    ctx.globalAlpha = 1;
  }

  function drawNodes(visible) {
    for (const [idx, a] of visible) {
      if (a <= 0) continue;
      const nd = nodes[idx];
      const [fill, edge, scale] = nodeStyle(nd.kind);
      ctx.globalAlpha = a;
      ctx.beginPath();
      ctx.arc(sx(nd.x), sy(nd.y), rNode * scale, 0, Math.PI * 2);
      ctx.fillStyle = fill;
      ctx.fill();
      ctx.lineWidth = Math.max(1.2, lw * 0.28);
      ctx.strokeStyle = edge;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  function drawCliffs() {
    const cw = replay.cliff.width, ch = replay.cliff.height;
    ctx.fillStyle = COL.earth;
    // left cliff: x in [-cw, 0]
    const lx = sx(-cw), lx2 = sx(0), top = sy(0), bot = sy(-ch);
    ctx.fillRect(Math.min(lx, lx2), Math.min(top, bot), Math.abs(lx2 - lx), Math.abs(bot - top));
    // right cliff: x in [chasm, chasm+cw]
    const rx = sx(replay.chasm_width), rx2 = sx(replay.chasm_width + cw);
    ctx.fillRect(Math.min(rx, rx2), Math.min(top, bot), Math.abs(rx2 - rx), Math.abs(bot - top));
  }

  function renderCanvas(p) {
    ctx.clearRect(0, 0, W, H);
    drawCliffs();

    const ph = replay.phases;
    const prefixFrac = smooth(p, 0.0, 0.18);
    const wrongFrac = smooth(p, 0.18, 0.34);
    const corrFrac = smooth(p, 0.34, 0.58);
    const wrongAlpha = 0.6 * (1 - smooth(p, 0.34, 0.5));

    const visible = new Map();
    // wrong first (lowest z), then prefix, then correction on top
    drawPhase(ph.wrong, wrongFrac, wrongAlpha, () => COL.wrong, visible);
    drawPhase(ph.prefix, prefixFrac, 1, (b) => placementColor(b.order), visible);
    drawPhase(ph.correction, corrFrac, 1, () => COL.correction, visible);

    // Wrong-only nodes fade out with their bars.
    for (const [idx] of visible) {
      if (nodes[idx].kind === "wrong") visible.set(idx, wrongAlpha / 0.6);
    }
    drawNodes(visible);

    if (cap) cap.style.opacity = String(smooth(p, 0.05, 0.18) * (1 - smooth(p, 0.5, 0.62)));
  }

  function renderLayers(p) {
    const out = smooth(p, 0.58, 0.72);
    replayLayer.style.opacity = String(1 - out);
    replayLayer.style.transform = `translateY(${-46 * out}px)`;
    const inn = smooth(p, 0.6, 0.74);
    datasetLayer.style.opacity = String(inn);
    datasetLayer.style.transform = `translateY(${46 * (1 - inn)}px)`;
  }

  // ---- dataset grid: inspection-style, round-organized rows ----
  const rowsData = manifest.rows || [];
  const widthBg = manifest.width_backgrounds || {};

  // Fade the width tint by the round alpha, composited over white so it reads
  // the same regardless of the orange section background (matches the plot).
  function compositeOverWhite(hex, alpha) {
    const [r, g, b] = hexToRgb(hex);
    const c = (v) => Math.round(255 * (1 - alpha) + v * alpha);
    return `rgb(${c(r)},${c(g)},${c(b)})`;
  }

  const rowTracks = []; // { track, setWidth, offset, speed }
  function buildRows() {
    rowsHost.innerHTML = "";
    rowTracks.length = 0;
    rowsData.forEach((row, i) => {
      const rowEl = document.createElement("div");
      rowEl.className = "dataset-row";
      const label = document.createElement("span");
      label.className = "dataset-row__label";
      label.textContent = row.round;
      const vp = document.createElement("div");
      vp.className = "dataset-row__viewport";
      const track = document.createElement("div");
      track.className = "dataset-row__track";
      const addCells = () => {
        row.cells.forEach((c) => {
          const cell = document.createElement("div");
          cell.className = "dataset-cell";
          const bg = widthBg[String(Math.round(c.width))];
          if (bg) cell.style.setProperty("--tint", compositeOverWhite(bg, row.alpha ?? 1));
          const img = document.createElement("img");
          img.src = `${BASE}/${c.file}`;
          img.alt = "";
          img.loading = "lazy";
          cell.appendChild(img);
          track.appendChild(cell);
        });
      };
      addCells();
      addCells(); // duplicate set for seamless wrap
      vp.appendChild(track);
      rowEl.appendChild(label);
      rowEl.appendChild(vp);
      rowsHost.appendChild(rowEl);
      rowTracks.push({ track, setWidth: 0, offset: 0, speed: 0.45 * (1 + 0.12 * i) });
    });
  }
  buildRows();

  function measureRows() {
    for (const rt of rowTracks) rt.setWidth = rt.track.scrollWidth / 2;
  }

  // ---- reduced-motion / small-screen fallback ----
  if (!full || window.innerWidth < 760) {
    resizeCanvas();
    renderCanvas(0.58); // final corrected state, before transition
    replayLayer.style.position = "relative";
    datasetLayer.style.position = "relative";
    replayLayer.style.opacity = "1";
    datasetLayer.style.opacity = "1";
    replayLayer.style.transform = "none";
    datasetLayer.style.transform = "none";
    if (cap) cap.style.opacity = "1";
    return;
  }

  resizeCanvas();
  measureRows();

  let scrollP = 0;
  let running = false;
  let rafId = 0;

  function frame() {
    for (const rt of rowTracks) {
      rt.offset -= rt.speed;
      if (rt.setWidth > 0 && -rt.offset >= rt.setWidth) rt.offset += rt.setWidth;
      rt.track.style.transform = `translateX(${rt.offset}px)`;
    }
    if (running) rafId = requestAnimationFrame(frame);
  }
  function start() {
    if (running) return;
    running = true;
    rafId = requestAnimationFrame(frame);
  }
  function stop() {
    running = false;
    cancelAnimationFrame(rafId);
  }

  function refresh() {
    resizeCanvas();
    measureRows();
    renderCanvas(scrollP);
    renderLayers(scrollP);
  }

  ScrollTrigger.create({
    trigger: section,
    start: "top top",
    end: "+=3200",
    pin: stage,
    scrub: true,
    onRefresh: refresh,
    onUpdate: (self) => {
      scrollP = self.progress;
      renderCanvas(scrollP);
      renderLayers(scrollP);
    },
  });

  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) (e.isIntersecting ? start() : stop());
    },
    { rootMargin: "20% 0px 20% 0px" }
  );
  io.observe(section);

  window.addEventListener("resize", refresh);
  renderCanvas(0);
  renderLayers(0);
}
