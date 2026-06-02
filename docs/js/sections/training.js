// Training — pinned, scroll-driven beat (flow matching).
//
// Story (sketch ③): random NOISE on the left is carried rightward through a
// learned VECTOR FIELD (continuously flowing dash-particles) and resolves into
// a small ACTION-CHUNK graph (stylized 3-node truss) on the right.
//
//   p 0.00-0.12  noise strip fizzes in
//   p 0.08-0.40  vector field fills with flowing particles
//   p 0.50-0.95  particles converge; action-chunk graph assembles (nodes + bars)
//
// The flow loops on its own rAF (so it keeps "flowing" without scrolling);
// scroll progress only gates reveal/assembly. Pure client-side, no assets.

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
  return [
    parseInt(h.slice(0, 2), 16),
    parseInt(h.slice(2, 4), 16),
    parseInt(h.slice(4, 6), 16),
  ];
}

// Rounded-rect path (avoids relying on ctx.roundRect for older engines).
function roundRectPath(c, x, y, w, h, r) {
  c.beginPath();
  c.moveTo(x + r, y);
  c.arcTo(x + w, y, x + w, y + h, r);
  c.arcTo(x + w, y + h, x, y + h, r);
  c.arcTo(x, y + h, x, y, r);
  c.arcTo(x, y, x + w, y, r);
  c.closePath();
}

// Formation-plot placement-order colour: linear dark→light over MAX_BARS steps,
// so a 3-bar chunk uses only the first three (darkest) colours — matching the
// paper figures.
function placementColor(idx, maxBars, dark, light) {
  const t = maxBars > 1 ? idx / (maxBars - 1) : 0;
  const a = hexToRgb(dark);
  const b = hexToRgb(light);
  const r = Math.round(a[0] + (b[0] - a[0]) * t);
  const g = Math.round(a[1] + (b[1] - a[1]) * t);
  const bl = Math.round(a[2] + (b[2] - a[2]) * t);
  return `rgb(${r},${g},${bl})`;
}

export function initTraining({ full }) {
  const section = document.querySelector("#teaching-training");
  if (!section) return;

  const stage = section.querySelector(".training-stage");
  const diagram = section.querySelector("#training-diagram");
  const canvas = section.querySelector("#training-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  const caps = {
    noise: section.querySelector(".training-cap--noise"),
    field: section.querySelector(".training-cap--field"),
    chunks: section.querySelector(".training-cap--chunks"),
  };

  const COL = {
    barDark: cssVar("--bridge-bar-dark", "#0420bf"),
    barLight: cssVar("--bridge-bar-light", "#9beaf2"),
    node: cssVar("--bridge-node-free", "#d2f25e"),
    nodeStart: cssVar("--bridge-node-start", "#f26b83"),
    nodeEdge: cssVar("--bridge-node-edge", "#2c2c2c"),
    flow: cssVar("--accent-teaching-training", "#4dabf7"),
    ink: cssVar("--ink", "#1b1f24"),
  };
  const MAX_BARS = 20; // canonical placement-order scale (matches formation plots)

  let W = 0, H = 0, dpr = 1;
  let particles = [];

  // Layout (CSS px), recomputed on resize.
  let geo = {};
  function layout() {
    // Composition lives in the right half (heading copy sits top-left and stays
    // legible via the head scrim). Caption x-anchors in styles.css mirror these.
    const cy = H * 0.5;
    const half = Math.min(H * 0.24, 180);
    geo = {
      cy,
      half,
      stripX: W * 0.42,
      stripW: Math.max(14, W * 0.04),
      flowStart: W * 0.48,
      targetX: W * 0.92,
      flowEnd: W * 0.88,
      s: Math.min(H * 0.13, 120), // truss scale
    };
  }

  function resize() {
    const r = diagram.getBoundingClientRect();
    W = r.width;
    H = r.height;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    layout();
    seedParticles();
  }

  function field(x, y, t) {
    const ny = (y - geo.cy) / geo.half;
    const nx = x / W;
    const base = Math.max(1.1, W * 0.0016);
    const vx = base * (0.85 + 0.35 * Math.cos(ny * 1.6 + t * 0.4));
    const vy =
      base * 0.7 * Math.sin(nx * 6.0 + ny * 1.4 + t * 0.8) - ny * base * 0.18;
    return [vx, vy];
  }

  function spawn(p) {
    p.x = geo.flowStart + Math.random() * (W * 0.02);
    p.y = geo.cy + (Math.random() * 2 - 1) * geo.half * 0.95;
    p.len = 6 + Math.random() * 10;
    p.a = 0.35 + Math.random() * 0.45;
  }

  function seedParticles() {
    const n = Math.round(clamp01(W / 1400) * 160) + 90;
    particles = [];
    for (let i = 0; i < n; i++) {
      const p = {};
      spawn(p);
      p.x = geo.flowStart + Math.random() * (geo.flowEnd - geo.flowStart);
      particles.push(p);
    }
  }

  // --- Action-chunk truss (stylized, in bridge colors) ---
  function nodes() {
    const { targetX, cy, s } = geo;
    return [
      { x: targetX + s * 0.15, y: cy - s * 0.95, n: 1 }, // top
      { x: targetX - s * 0.55, y: cy + s * 0.15, n: 2 }, // left
      { x: targetX + s * 0.75, y: cy + s * 0.85, n: 3 }, // bottom-right
    ];
  }

  function drawTruss(a) {
    if (a <= 0) return;
    const N = nodes();
    // [i, j, placementOrder, t0, t1] — bars coloured by placement order.
    const bars = [
      [0, 1, 0, 0.15, 0.42], // bar 1: 1-2
      [1, 2, 1, 0.45, 0.72], // bar 2: 2-3
      [0, 2, 2, 0.6, 0.86], // bar 3: 1-3
    ];
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = Math.max(3, geo.s * 0.11);
    for (const [i, j, order, t0, t1] of bars) {
      const f = clamp01((a - t0) / (t1 - t0));
      if (f <= 0) continue;
      const A = N[i], B = N[j];
      ctx.strokeStyle = placementColor(order, MAX_BARS, COL.barDark, COL.barLight);
      ctx.globalAlpha = 1;
      ctx.beginPath();
      ctx.moveTo(A.x, A.y);
      ctx.lineTo(A.x + (B.x - A.x) * f, A.y + (B.y - A.y) * f);
      ctx.stroke();
    }

    const nodeT = [0.05, 0.25, 0.5];
    const rNode = Math.max(5, geo.s * 0.14);
    N.forEach((nd, i) => {
      const f = smooth(a, nodeT[i], nodeT[i] + 0.12);
      if (f <= 0) return;
      ctx.globalAlpha = f;
      const r = rNode * (0.6 + 0.4 * f);
      ctx.beginPath();
      ctx.arc(nd.x, nd.y, r, 0, Math.PI * 2);
      // The first node placed is the start/support point -> pink.
      ctx.fillStyle = nd.n === 1 ? COL.nodeStart : COL.node;
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = COL.nodeEdge;
      ctx.stroke();
    });
    ctx.globalAlpha = 1;

    // Placement-order badges: white rounded squares centred on each bar's
    // midpoint, revealed as the bar finishes drawing.
    const boxSize = Math.max(8, geo.s * 0.2);
    const rad = boxSize * 0.28;
    ctx.lineWidth = Math.max(1, geo.s * 0.022);
    ctx.font = `600 ${Math.round(boxSize * 0.56)}px var(--font-aeonik, sans-serif)`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    for (const [i, j, order, , t1] of bars) {
      const boxA = clamp01((a - t1) / 0.1);
      if (boxA <= 0) continue;
      const A = N[i], B = N[j];
      const mx = (A.x + B.x) / 2;
      const my = (A.y + B.y) / 2;
      ctx.globalAlpha = boxA;
      roundRectPath(ctx, mx - boxSize / 2, my - boxSize / 2, boxSize, boxSize, rad);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.strokeStyle = "rgba(44,44,44,0.55)";
      ctx.stroke();
      ctx.fillStyle = COL.ink;
      ctx.fillText(String(order + 1), mx, my + boxSize * 0.04);
    }
    ctx.globalAlpha = 1;
  }

  function drawNoise(alpha, t) {
    if (alpha <= 0) return;
    const { stripX, stripW, cy, half } = geo;
    const cols = 4;
    const cell = stripW / cols;           // square cells
    const rows = Math.max(1, Math.round((half * 2) / cell));
    const top = cy - (rows * cell) / 2;   // centre the square grid in the band
    ctx.globalAlpha = alpha;
    // Cheap deterministic-ish flicker keyed to a slow time step.
    const seed = Math.floor(t * 6);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const k = Math.sin((r * 12.9898 + c * 78.233 + seed) * 1.0) * 43758.5;
        const g = k - Math.floor(k);
        const v = Math.round(40 + g * 200);
        ctx.fillStyle = `rgb(${v},${v},${v})`;
        ctx.fillRect(stripX + c * cell, top + r * cell, cell + 0.6, cell + 0.6);
      }
    }
    ctx.globalAlpha = 1;
  }

  let scrollP = 0;
  let t = 0;
  let running = false;
  let rafId = 0;

  function render() {
    t += 0.016;
    ctx.clearRect(0, 0, W, H);

    const p = scrollP;
    const noiseA = smooth(p, 0, 0.12);
    const flowA = smooth(p, 0.08, 0.4) * (1 - 0.55 * smooth(p, 0.72, 1));
    const assemble = smooth(p, 0.5, 0.95);

    drawNoise(noiseA, t);

    // Flowing particles (dash/comet segments along the velocity field).
    if (flowA > 0.01) {
      ctx.strokeStyle = COL.flow;
      ctx.lineCap = "round";
      ctx.lineWidth = Math.max(2, W * 0.0018);
      for (const pt of particles) {
        const [vx, vy] = field(pt.x, pt.y, t);
        pt.x += vx;
        pt.y += vy;
        if (
          pt.x > geo.flowEnd ||
          pt.y < geo.cy - geo.half * 1.15 ||
          pt.y > geo.cy + geo.half * 1.15
        ) {
          spawn(pt);
        }
        // Fade near both ends of the band.
        const edge =
          smooth(pt.x, geo.flowStart, geo.flowStart + W * 0.05) *
          (1 - smooth(pt.x, geo.flowEnd - W * 0.08, geo.flowEnd));
        const sp = Math.hypot(vx, vy);
        ctx.globalAlpha = flowA * pt.a * edge;
        ctx.beginPath();
        ctx.moveTo(pt.x - (vx / sp) * pt.len, pt.y - (vy / sp) * pt.len);
        ctx.lineTo(pt.x, pt.y);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    drawTruss(assemble);

    if (caps.noise) caps.noise.style.opacity = String(noiseA);
    if (caps.field) caps.field.style.opacity = String(smooth(p, 0.1, 0.35));
    if (caps.chunks) caps.chunks.style.opacity = String(smooth(p, 0.55, 0.82));

    if (running) rafId = requestAnimationFrame(render);
  }

  function start() {
    if (running) return;
    running = true;
    rafId = requestAnimationFrame(render);
  }
  function stop() {
    running = false;
    cancelAnimationFrame(rafId);
  }

  resize();

  // ---- Reduced-motion / small-screen fallback: static final frame ----
  if (!full || window.innerWidth < 760) {
    scrollP = 1;
    t = 2;
    render(); // single frame (running=false, so no loop)
    Object.values(caps).forEach((c) => c && (c.style.opacity = "1"));
    return;
  }

  ScrollTrigger.create({
    trigger: section,
    start: "top top",
    end: "+=2600",
    pin: stage,
    scrub: true,
    onRefresh: resize,
    onUpdate: (self) => {
      scrollP = self.progress;
    },
  });

  // Keep the flow animating only while the section is near the viewport.
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) start();
        else stop();
      }
    },
    { rootMargin: "20% 0px 20% 0px" }
  );
  io.observe(section);

  window.addEventListener("resize", resize);
}
