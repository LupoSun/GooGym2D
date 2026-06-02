// Human-AI Co-design — pinned, scroll-driven beat (sketch ⑤).
//
// The assistive co-design episode (same scripted sequence as
// assistive_codesign_episode.png) replays turn by turn. The card holding the
// canvas FLIES IN, then alternates LEFT (human turn) / RIGHT (AI turn) as the
// turn switches, then settles CENTER for the accepted co-designed bridge and
// FLIES OUT. Human bars use the placement gradient; AI suggestions are single
// magenta dashed ghosts (#e64980); committed AI/policy bars are solid magenta.
//
// Geometry from assets/codesign/codesign.json (tools/export_codesign_assets.py).

const BASE = "assets/codesign";

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
function lerp(a, b, t) {
  return a + (b - a) * t;
}
function cssVar(name, fallback) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}
function hexToRgb(hex) {
  const h = hex.replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

export async function initCodesign({ full }) {
  const section = document.querySelector("#teaming-codesign");
  if (!section) return;

  const stage = section.querySelector(".codesign-stage");
  const arena = section.querySelector("#codesign-arena");
  const card = section.querySelector("#codesign-card");
  const canvas = section.querySelector("#codesign-canvas");
  const turnEl = section.querySelector("#codesign-turn");
  const whoEl = section.querySelector("#codesign-who");
  const labelEl = section.querySelector("#codesign-label");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  let data;
  try {
    data = await loadJSON(`${BASE}/codesign.json`);
  } catch (err) {
    console.warn("codesign: assets missing, leaving static.", err);
    return;
  }

  const steps = data.steps || [];
  const vp = data.viewport;
  const maxBars = data.max_bars || 20;

  const COL = {
    barDark: cssVar("--bridge-bar-dark", "#0420bf"),
    barLight: cssVar("--bridge-bar-light", "#9beaf2"),
    nodeFree: cssVar("--bridge-node-free", "#d2f25e"),
    nodeStart: cssVar("--bridge-node-start", "#f26b83"),
    edge: cssVar("--bridge-node-edge", "#2c2c2c"),
    ghost: cssVar("--codesign-ghost", "#e64980"),
    ghostNode: cssVar("--codesign-ghost-node", "#f783ac"),
    earth: cssVar("--bridge-earth", "#a3afbf40"),
    humanBg: cssVar("--codesign-human-bg", "#e3fafc"),
    aiBg: cssVar("--codesign-ai-bg", "#f4fce3"),
    cardBg: cssVar("--surface-canvas-white-base", "#ffffff"),
  };

  function placementColor(order) {
    const t = maxBars > 1 ? order / (maxBars - 1) : 0;
    const a = hexToRgb(COL.barDark), b = hexToRgb(COL.barLight);
    return `rgb(${Math.round(lerp(a[0], b[0], t))},${Math.round(lerp(a[1], b[1], t))},${Math.round(
      lerp(a[2], b[2], t)
    )})`;
  }

  // ---- canvas geometry (world → device px) ----
  let W = 0, H = 0, dpr = 1, lw = 4, rNode = 6;
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
    lw = Math.max(3, W * 0.018);
    rNode = Math.max(4, W * 0.016);
  }

  function isSupport(x, y) {
    return Math.abs(y) < 0.06 && (Math.abs(x) < 0.06 || Math.abs(x - data.chasm_width) < 0.06);
  }

  function drawCliffs() {
    const cw = data.cliff.width, ch = data.cliff.height;
    ctx.fillStyle = COL.earth;
    const top = sy(0), bot = sy(-ch);
    const lx = sx(-cw), lx2 = sx(0);
    ctx.fillRect(Math.min(lx, lx2), Math.min(top, bot), Math.abs(lx2 - lx), Math.abs(bot - top));
    const rx = sx(data.chasm_width), rx2 = sx(data.chasm_width + cw);
    ctx.fillRect(Math.min(rx, rx2), Math.min(top, bot), Math.abs(rx2 - rx), Math.abs(bot - top));
  }

  function drawNode(x, y, kind) {
    const fill = kind === "support" ? COL.nodeStart : kind === "ghost" ? COL.ghostNode : COL.nodeFree;
    ctx.beginPath();
    ctx.arc(sx(x), sy(y), rNode, 0, Math.PI * 2);
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.lineWidth = Math.max(1.2, lw * 0.26);
    ctx.strokeStyle = kind === "ghost" ? COL.ghost : COL.edge;
    ctx.stroke();
  }

  // Draw a solid bar (human gradient or committed AI magenta), with optional
  // progressive fraction f (0..1).
  function drawBar(bar, f) {
    const [x1, y1] = bar.p1, [x2, y2] = bar.p2;
    ctx.lineCap = "round";
    ctx.lineWidth = lw;
    ctx.strokeStyle = bar.author === "policy" ? COL.ghost : placementColor(bar.order);
    ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.moveTo(sx(x1), sy(y1));
    ctx.lineTo(sx(x1) + (sx(x2) - sx(x1)) * f, sy(y1) + (sy(y2) - sy(y1)) * f);
    ctx.stroke();
  }

  function drawStep(step, drawT) {
    ctx.clearRect(0, 0, W, H);
    drawCliffs();

    const bars = step.bars || [];
    const newFrom = step.new_from ?? bars.length;
    const nodeSet = new Map(); // "x,y" -> kind

    const addNodes = (bar) => {
      for (const [x, y] of [bar.p1, bar.p2]) {
        const key = `${x.toFixed(3)},${y.toFixed(3)}`;
        nodeSet.set(key, { x, y, kind: isSupport(x, y) ? "support" : "free" });
      }
    };

    // Base (already-placed) bars at full; newly added bars draw progressively.
    bars.forEach((bar, i) => {
      let f = 1;
      if (i >= newFrom) {
        // Progressive draw across the newly-added bars.
        const drawn = drawT * (bars.length - newFrom);
        const k = i - newFrom;
        f = k < Math.floor(drawn) ? 1 : k === Math.floor(drawn) ? drawn - Math.floor(drawn) : 0;
      }
      if (f <= 0) return;
      drawBar(bar, f);
      if (f >= 1) addNodes(bar);
      else {
        // only the anchor node lands until the bar completes
        const key = `${bar.p1[0].toFixed(3)},${bar.p1[1].toFixed(3)}`;
        nodeSet.set(key, { x: bar.p1[0], y: bar.p1[1], kind: isSupport(bar.p1[0], bar.p1[1]) ? "support" : "free" });
      }
    });

    // Ghost suggestion (single, magenta dashed) — fades/draws in with drawT.
    if (step.ghost && step.ghost.length) {
      ctx.save();
      ctx.globalAlpha = clamp01(drawT) * 0.9;
      ctx.lineCap = "round";
      ctx.lineWidth = Math.max(2, lw * 0.7);
      ctx.strokeStyle = COL.ghost;
      ctx.setLineDash([Math.max(5, lw * 1.3), Math.max(4, lw)]);
      step.ghost.forEach((g) => {
        ctx.beginPath();
        ctx.moveTo(sx(g.p1[0]), sy(g.p1[1]));
        ctx.lineTo(sx(g.p2[0]), sy(g.p2[1]));
        ctx.stroke();
      });
      ctx.setLineDash([]);
      // ghost nodes
      const seen = new Set();
      step.ghost.forEach((g) => {
        for (const [x, y] of [g.p1, g.p2]) {
          const key = `${x.toFixed(3)},${y.toFixed(3)}`;
          if (seen.has(key)) continue;
          seen.add(key);
          drawNode(x, y, "ghost");
        }
      });
      ctx.restore();
    }

    // Solid nodes on top.
    ctx.globalAlpha = 1;
    for (const { x, y, kind } of nodeSet.values()) drawNode(x, y, kind);
  }

  // ---- layout slots (fly-in → L/R per turn → center → fly-out) ----
  let arenaW = 0;
  function slotX(side) {
    switch (side) {
      case "left": return -arenaW * 0.24;
      case "right": return arenaW * 0.24;
      case "flyIn": return -arenaW * 0.95;
      case "flyOut": return arenaW * 0.95;
      default: return 0; // center
    }
  }

  function render(p) {
    const n = steps.length;
    if (!n) return;
    const stepF = clamp01(p) * n;
    const i = Math.min(n - 1, Math.floor(stepF));
    const frac = stepF - i;
    const step = steps[i];

    const slide = smooth(frac, 0, 0.28);
    const prevSide = i > 0 ? steps[i - 1].side : "flyIn";
    let x = lerp(slotX(prevSide), slotX(step.side), slide);

    // Final step: after settling center, fly out to the right.
    if (i === n - 1) {
      const out = smooth(frac, 0.74, 1);
      x = lerp(x, slotX("flyOut"), out);
    }
    const scaleIn = lerp(0.85, 1, smooth(p, 0, 0.05));
    card.style.transform = `translate(calc(-50% + ${x}px), -50%) scale(${scaleIn})`;

    // Card tint by turn.
    card.style.backgroundColor =
      step.turn === "human" ? COL.humanBg : step.turn === "ai" ? COL.aiBg : COL.cardBg;

    // Turn caption on the opposite side.
    if (turnEl) {
      const onSide = step.side === "left" || step.side === "right";
      const capX = -x * 1.0;
      turnEl.style.transform = `translate(calc(-50% + ${capX}px), -50%)`;
      const vis = onSide ? smooth(frac, 0.12, 0.4) * (1 - smooth(frac, 0.82, 1)) : 0;
      turnEl.style.opacity = String(vis);
      if (whoEl) {
        whoEl.textContent = step.turn === "ai" ? "AI" : "Human";
        whoEl.style.color = step.turn === "ai" ? COL.ghost : COL.edge;
      }
      if (labelEl) labelEl.textContent = step.label || "";
    }

    const drawT = smooth(frac, 0.2, 0.64);
    drawStep(step, i === n - 1 ? Math.max(drawT, smooth(frac, 0.0, 0.5)) : drawT);
  }

  function refresh() {
    arenaW = arena.getBoundingClientRect().width;
    resizeCanvas();
    render(scrollP);
  }

  let scrollP = 0;

  // ---- reduced-motion / small-screen fallback: final co-designed bridge ----
  if (!full || window.innerWidth < 760) {
    arenaW = arena.getBoundingClientRect().width;
    resizeCanvas();
    card.style.transform = "translate(-50%, -50%)";
    if (window.innerWidth < 760) card.style.transform = "none";
    card.style.backgroundColor = COL.cardBg;
    drawStep(steps[steps.length - 1], 1);
    return;
  }

  arenaW = arena.getBoundingClientRect().width;
  resizeCanvas();

  ScrollTrigger.create({
    trigger: section,
    start: "top top",
    end: "+=3600",
    pin: stage,
    scrub: true,
    onRefresh: refresh,
    onUpdate: (self) => {
      scrollP = self.progress;
      render(scrollP);
    },
  });

  window.addEventListener("resize", refresh);
  render(0);
}
