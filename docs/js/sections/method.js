// Environment — copy/card reveal + three looping canvas panels illustrating the
// GooGym2D env mechanics (mirrors the paper figure):
//   a) Design action formulation  b) Hindsight nodal adjustment  c) FEM & constraints
// Pure client-side canvas (no assets). Each panel autoplays its own phased loop,
// gated by IntersectionObserver; reduced-motion draws a single static frame.

const PERIOD = 7; // seconds per loop

// Shared chasm geometry (fractions of the square): the two support endpoints of
// every structure sit exactly on these cliff inner corners.
const GX_L = -0.36, GX_R = 0.36, GY = 0.3;

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

const COL = {
  bar: "#3b5bdb",        // royal blue bars (panels a/b)
  barDark: "#1b3bbb",    // placement gradient: early bars
  barLight: "#99e9f2",   // placement gradient: late bars
  black: "#1a1a1a",      // FEM members (panel c)
  node: "#c0eb75",       // lime free nodes
  start: "#ff8787",      // coral support nodes
  edge: "#2c2c2c",
  ghost: "#cfd4da",
  ghostEdge: "#aeb4bc",
  earth: "rgba(163,175,191,0.32)",
  load: "#94d82d",       // distributed-load lime
  loadArrow: "#3b5bdb",  // nodal load arrows (blue)
  deflect: "#f4a6c0",    // dashed deflected shape (light pink)
  ink: cssVar("--ink", "#0a0d12"),
};

function hexToRgb(h) {
  h = h.replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}
function placement(order, max) {
  const t = max > 1 ? order / (max - 1) : 0;
  const a = hexToRgb(COL.barDark), b = hexToRgb(COL.barLight);
  return `rgb(${Math.round(lerp(a[0], b[0], t))},${Math.round(lerp(a[1], b[1], t))},${Math.round(lerp(a[2], b[2], t))})`;
}

// ---- small drawing helpers ----
function box(W, H) {
  const S = Math.min(W, H) * 0.84;
  return { cx: W / 2, cy: H / 2, S };
}
function pt(b, ox, oy) {
  return [b.cx + ox * b.S, b.cy + oy * b.S];
}
function node(ctx, p, r, fill, edge, alpha = 1) {
  if (alpha <= 0) return;
  ctx.globalAlpha = alpha;
  ctx.beginPath();
  ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = Math.max(1, r * 0.34);
  ctx.strokeStyle = edge;
  ctx.stroke();
  ctx.globalAlpha = 1;
}
function bar(ctx, a, b, color, lw, frac = 1, alpha = 1, dash = null) {
  if (alpha <= 0 || frac <= 0) return;
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = lw;
  ctx.lineCap = "round";
  if (dash) ctx.setLineDash(dash);
  ctx.beginPath();
  ctx.moveTo(a[0], a[1]);
  ctx.lineTo(a[0] + (b[0] - a[0]) * frac, a[1] + (b[1] - a[1]) * frac);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalAlpha = 1;
}
function arrow(ctx, x1, y1, x2, y2, color, lw, head, alpha = 1) {
  if (alpha <= 0) return;
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lw;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  const a = Math.atan2(y2 - y1, x2 - x1);
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - head * Math.cos(a - 0.4), y2 - head * Math.sin(a - 0.4));
  ctx.lineTo(x2 - head * Math.cos(a + 0.4), y2 - head * Math.sin(a + 0.4));
  ctx.closePath();
  ctx.fill();
  ctx.globalAlpha = 1;
}
function star8(ctx, p, len, color, alpha) {
  if (alpha <= 0) return;
  const head = len * 0.24;          // small arrowheads (was a fat wedge)
  const lw = Math.max(1, len * 0.045);
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2;
    const ox = Math.cos(a), oy = Math.sin(a);
    arrow(ctx, p[0] + ox * len * 0.5, p[1] + oy * len * 0.5,   // start clear of the node
      p[0] + ox * len, p[1] + oy * len, color, lw, head, alpha);
  }
}
function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
function badge(ctx, x, y, label, S, alpha = 1, opts = {}) {
  if (alpha <= 0) return;
  const fill = opts.fill || "#ffffff";
  const border = opts.border || "rgba(44,44,44,0.75)";
  const text = opts.text || COL.ink;
  const s = S * 0.08, rad = s * 0.28;
  ctx.globalAlpha = alpha;
  roundRect(ctx, x - s / 2, y - s / 2, s, s, rad);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = Math.max(1, S * 0.006);
  ctx.strokeStyle = border;
  ctx.stroke();
  ctx.fillStyle = text;
  ctx.font = `600 ${Math.round(s * 0.54)}px var(--font-aeonik, sans-serif)`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x, y + s * 0.04);
  ctx.globalAlpha = 1;
}
// Cliffs whose inner top corners sit exactly at (GX_L, GY) and (GX_R, GY).
function cliffs(ctx, b) {
  const top = b.cy + GY * b.S;
  const bottom = b.cy + 0.62 * b.S;
  const far = 0.62 * b.S;
  const lcx = b.cx + GX_L * b.S, rcx = b.cx + GX_R * b.S;
  ctx.fillStyle = COL.earth;
  ctx.fillRect(b.cx - far, top, lcx - (b.cx - far), bottom - top);
  ctx.fillRect(rcx, top, b.cx + far - rcx, bottom - top);
}

// ---- (a) Design action formulation ----
function drawAction(ctx, W, H, t) {
  const b = box(W, H);
  const S = b.S;
  const r = S * 0.026, lw = S * 0.016;
  const c = (t % PERIOD) / PERIOD;
  cliffs(ctx, b);

  // Triangle BL–BR–apex (left support on the left cliff); node 4 is placed
  // up-right of the apex at the same height (horizontal action), per the figure.
  const bl = pt(b, GX_L, GY);      // bottom-left support (coral), left cliff
  const br = pt(b, -0.04, GY);     // bottom-right node (over the gap)
  const apex = pt(b, -0.22, -0.06);// top node = action anchor
  const n4 = pt(b, 0.16, -0.06);   // node 4: right of the apex, same height
  const mid = (a, z) => [(a[0] + z[0]) / 2, (a[1] + z[1]) / 2];

  // base triangle (blue bars + numbered badges)
  bar(ctx, bl, apex, COL.bar, lw);  // bar 1
  bar(ctx, bl, br, COL.bar, lw);    // bar 2
  bar(ctx, apex, br, COL.bar, lw);  // bar 3

  // action phases
  const labelA = smooth(c, 0.06, 0.2) * (1 - smooth(c, 0.9, 1));
  const reach = smooth(c, 0.24, 0.56);
  const fade = 1 - smooth(c, 0.92, 1);

  // blue member growing horizontally from the anchor → node 4
  const tip = [lerp(apex[0], n4[0], reach), lerp(apex[1], n4[1], reach)];
  arrow(ctx, apex[0], apex[1], tip[0], tip[1], COL.bar, lw * 0.85, S * 0.055, 0.95 * reach * fade);
  node(ctx, n4, r, COL.node, COL.edge, smooth(c, 0.5, 0.66) * fade);

  // nodes: bl support (coral), br + apex lime
  node(ctx, bl, r, COL.start, COL.edge);
  node(ctx, br, r, COL.node, COL.edge);
  node(ctx, apex, r, COL.node, COL.edge);
  // pulsing coral ring on the anchor while the action forms
  if (labelA > 0) {
    ctx.globalAlpha = labelA;
    ctx.strokeStyle = COL.start;
    ctx.lineWidth = lw * 0.6;
    ctx.beginPath();
    ctx.arc(apex[0], apex[1], r * (1.8 + 0.28 * Math.sin(t * 4)), 0, Math.PI * 2);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // numbered bar badges (white rounded squares)
  badge(ctx, ...mid(bl, apex), "1", S);
  badge(ctx, ...mid(bl, br), "2", S);
  badge(ctx, ...mid(apex, br), "3", S);
  // "4" badge riding the new member
  const four = [lerp(apex[0], n4[0], 0.5), lerp(apex[1], n4[1], 0.5)];
  badge(ctx, four[0], four[1], "4", S, smooth(c, 0.34, 0.52) * fade,
    { fill: "#fff0f3", border: COL.start, text: COL.start });

  // action label pill: x4 y4 θ4 d4 (coral outline)
  if (labelA > 0) {
    const txt = "x₄  y₄  θ₄  d₄";
    ctx.font = `600 ${Math.round(S * 0.05)}px var(--font-aeonik, sans-serif)`;
    const tw = ctx.measureText(txt).width;
    const pw = tw + S * 0.07, ph = S * 0.1;
    const px = apex[0] - pw / 2;
    const py = apex[1] - S * 0.22;
    ctx.globalAlpha = labelA;
    roundRect(ctx, px, py - ph / 2, pw, ph, ph / 2);
    ctx.fillStyle = "#fff0f3";
    ctx.fill();
    ctx.lineWidth = Math.max(1, S * 0.006);
    ctx.strokeStyle = COL.start;
    ctx.stroke();
    ctx.fillStyle = COL.start;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(txt, px + pw / 2, py);
    ctx.globalAlpha = 1;
  }
}

// ---- (b) Hindsight nodal adjustment ----
function drawHindsight(ctx, W, H, t) {
  const b = box(W, H);
  const S = b.S;
  const r = S * 0.026, lw = S * 0.016;
  const c = (t % PERIOD) / PERIOD;
  cliffs(ctx, b);

  // fixed supports on the cliff corners
  const bl = pt(b, GX_L, GY);
  const br = pt(b, GX_R, GY);
  // movable interior nodes [TL, TR, BM]: original → adjusted
  const orig = [pt(b, -0.24, -0.04), pt(b, 0.06, 0.02), pt(b, 0.04, 0.24)];
  const adj = [pt(b, -0.30, -0.18), pt(b, 0.16, -0.12), pt(b, -0.02, 0.28)];

  // move amount: rise, hold, fall (clean loop)
  let m = smooth(c, 0.28, 0.6) - smooth(c, 0.85, 1.0);
  m = clamp01(m);
  const cur = orig.map((o, i) => [lerp(o[0], adj[i][0], m), lerp(o[1], adj[i][1], m)]);

  // nodes [BL, TL, TR, BM, BR] = indices 0..4
  const E = (n) => [bl, ...cur, br][n];
  // reference topology, ordered left→right so the placement gradient reads dark→cyan
  const bars = [[0, 1], [0, 3], [1, 3], [1, 2], [3, 2], [2, 4], [3, 4]];

  // ghost (original) faint while moving
  if (m > 0.02) {
    const go = [bl, ...orig, br];
    for (const [i, j] of bars) bar(ctx, go[i], go[j], COL.ghost, lw, 1, 0.45 * m);
    orig.forEach((p) => node(ctx, p, r * 0.85, COL.ghost, COL.ghostEdge, 0.45 * m));
  }

  // current truss — placement gradient (early bars dark blue → late bars cyan)
  bars.forEach(([i, j], k) => bar(ctx, E(i), E(j), placement(k, bars.length), lw));
  node(ctx, bl, r, COL.start, COL.edge);
  node(ctx, br, r, COL.start, COL.edge);
  cur.forEach((p) => node(ctx, p, r, COL.node, COL.edge));

  // 8-dir arrow stars on movable nodes (appear before/around the move)
  const starA = smooth(c, 0.12, 0.26) * (1 - smooth(c, 0.62, 0.78));
  if (starA > 0) cur.forEach((p) => star8(ctx, p, S * 0.1, COL.edge, starA * 0.7));
}

// ---- (c) FEM & constraints ----
function drawFem(ctx, W, H, t) {
  const b = box(W, H);
  const S = b.S;
  const r = S * 0.024, lw = S * 0.016;
  const c = (t % PERIOD) / PERIOD;
  cliffs(ctx, b);

  const bl = pt(b, GX_L, GY);
  const br = pt(b, GX_R, GY);
  const mb = pt(b, 0.0, GY);
  const tl = pt(b, -0.18, -0.06); // above the BL-BM midpoint
  const tr = pt(b, 0.18, -0.06);  // above the BM-BR midpoint
  const nodes = [bl, tl, tr, br, mb];
  const bars = [
    [0, 1], [1, 2], [2, 3], // left diag, top chord, right diag
    [0, 4], [4, 3],         // bottom chord
    [1, 4], [2, 4],         // inner diagonals
  ];

  // deflection: bottom-mid sags deepest; top chord drops + is pulled inward.
  const dA = clamp01(smooth(c, 0.4, 0.72) - smooth(c, 0.9, 1.0));
  const span = br[0] - bl[0];
  const disp = nodes.map((p) => {
    const u = (p[0] - bl[0]) / span; // 0..1 across span
    const w = Math.sin(Math.PI * clamp01(u)); // 0 at supports, 1 at mid
    const isTop = p[1] < bl[1] - 0.02 * S;
    const dy = w * S * (isTop ? 0.12 : 0.18) * dA;
    const dx = isTop ? (b.cx - p[0]) * 0.38 * dA : 0; // top nodes pulled toward center
    return [p[0] + dx, p[1] + dy];
  });

  // black FEM members
  for (const [i, j] of bars) bar(ctx, nodes[i], nodes[j], COL.black, lw * 1.05);

  // dashed deflected shape (light pink)
  if (dA > 0.02) {
    for (const [i, j] of bars)
      bar(ctx, disp[i], disp[j], COL.deflect, lw * 0.7, 1, 0.9, [S * 0.028, S * 0.028]);
  }

  // distributed load: white-filled lime rectangle (upper + lower chord) spanning
  // support-to-support, with down-arrows encased between the two chords.
  const loadA = smooth(c, 0.2, 0.4) * (1 - smooth(c, 0.92, 1));
  if (loadA > 0) {
    const x0 = b.cx + GX_L * S, x1 = b.cx + GX_R * S;
    const yTop = b.cy - 0.46 * S;
    const rectH = S * 0.13;
    const yBot = yTop + rectH;
    ctx.globalAlpha = loadA;
    ctx.fillStyle = "#ffffff";        // opaque white interior
    ctx.fillRect(x0, yTop, x1 - x0, rectH);
    ctx.strokeStyle = COL.load;       // lime upper + lower chords + ends
    ctx.lineWidth = Math.max(1.4, S * 0.009);
    ctx.lineJoin = "miter";
    ctx.strokeRect(x0, yTop, x1 - x0, rectH);
    ctx.globalAlpha = 1;
    for (let i = 0; i < 7; i++) {
      const x = lerp(x0, x1, (i + 0.5) / 7);
      arrow(ctx, x, yTop + S * 0.012, x, yBot - S * 0.006, COL.load, Math.max(1.2, S * 0.008), S * 0.03, loadA);
    }
  }

  // supports first, so the start/end nodes sit on top of the triangles
  drawPin(ctx, bl, S);
  drawRoller(ctx, br, S);

  // main nodes: white circles with black edge (above the support triangles)
  nodes.forEach((p) => node(ctx, p, r, "#ffffff", COL.black));

  // blue load dots + equal-length downward arrows at each member's midpoint
  const blueA = smooth(c, 0.26, 0.46) * (1 - smooth(c, 0.92, 1));
  if (blueA > 0) {
    const len = S * 0.13;
    for (const [i, j] of bars) {
      const a = nodes[i], z = nodes[j];
      const mx = (a[0] + z[0]) / 2, my = (a[1] + z[1]) / 2;
      node(ctx, [mx, my], r * 0.52, COL.loadArrow, COL.loadArrow, blueA);
      arrow(ctx, mx, my, mx, my + len, COL.loadArrow, Math.max(1.3, lw * 0.45), S * 0.04, blueA);
    }
  }
}
function drawPin(ctx, p, S) {
  const w = S * 0.07;
  ctx.strokeStyle = COL.edge;
  ctx.fillStyle = "#ffffff";
  ctx.lineWidth = Math.max(1, S * 0.006);
  ctx.beginPath();
  ctx.moveTo(p[0], p[1]);
  ctx.lineTo(p[0] - w, p[1] + w * 1.4);
  ctx.lineTo(p[0] + w, p[1] + w * 1.4);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  hatch(ctx, p[0] - w, p[1] + w * 1.4, p[0] + w, p[1] + w * 1.4, S);
}
function drawRoller(ctx, p, S) {
  const w = S * 0.07;
  const by = p[1] + w * 1.4;
  ctx.strokeStyle = COL.edge;
  ctx.fillStyle = "#ffffff";
  ctx.lineWidth = Math.max(1, S * 0.006);
  // triangle (like the pin)
  ctx.beginPath();
  ctx.moveTo(p[0], p[1]);
  ctx.lineTo(p[0] - w, by);
  ctx.lineTo(p[0] + w, by);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  // roller line just below the base, then hatched ground
  const ly = by + S * 0.022;
  ctx.beginPath();
  ctx.moveTo(p[0] - w, ly);
  ctx.lineTo(p[0] + w, ly);
  ctx.stroke();
  hatch(ctx, p[0] - w, ly, p[0] + w, ly, S);
}
function hatch(ctx, x1, y, x2, y2, S) {
  ctx.strokeStyle = "rgba(44,44,44,0.55)";
  ctx.lineWidth = Math.max(0.8, S * 0.004);
  ctx.beginPath();
  ctx.moveTo(x1, y);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  const n = 5, step = (x2 - x1) / n;
  for (let i = 0; i <= n; i++) {
    const x = x1 + i * step;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x - S * 0.03, y + S * 0.03);
    ctx.stroke();
  }
}

// ---- generic looping canvas ----
function makeLoop(canvas, drawFn, full) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let W = 0, H = 0, t = 0, raf = 0, running = false;

  function resize() {
    const r = canvas.getBoundingClientRect();
    W = r.width;
    H = r.height;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function paint(time) {
    ctx.clearRect(0, 0, W, H);
    drawFn(ctx, W, H, time);
  }
  function frame() {
    t += 0.016;
    paint(t);
    if (running) raf = requestAnimationFrame(frame);
  }
  resize();

  if (!full) {
    paint(0.82 * PERIOD); // representative static frame
    window.addEventListener("resize", () => { resize(); paint(0.82 * PERIOD); });
    return;
  }

  window.addEventListener("resize", () => { resize(); if (!running) paint(t); });
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) { if (!running) { running = true; raf = requestAnimationFrame(frame); } }
        else { running = false; cancelAnimationFrame(raf); }
      }
    },
    { rootMargin: "15% 0px 15% 0px" }
  );
  io.observe(canvas);
}

export function initMethod({ full }) {
  const section = document.querySelector("#environment");
  if (!section) return;

  const items = section.querySelectorAll(".js-anim");
  if (!full) {
    gsap.set(items, { opacity: 1, y: 0 });
  } else {
    gsap.set(items, { opacity: 0, y: 28 });
    gsap.to(items, {
      opacity: 1,
      y: 0,
      ease: "power2.out",
      duration: 0.7,
      stagger: 0.12,
      scrollTrigger: { trigger: section, start: "top 70%" },
    });
  }

  makeLoop(section.querySelector("#env-action"), drawAction, full);
  makeLoop(section.querySelector("#env-hindsight"), drawHindsight, full);
  makeLoop(section.querySelector("#env-fem"), drawFem, full);
}
