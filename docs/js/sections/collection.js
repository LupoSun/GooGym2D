// Collection — pinned, scroll-driven beat.
//
// Order of appearance (per scroll progress p):
//   A 0.00-0.12  GUI appears in the middle, scaling up to target size.
//   B 0.12-0.24  GUI moves to the right; explanatory text reveals on the left.
//   C 0.24-0.56  sequential placement: the demo builds bar by bar (GUI stays right).
//   D 0.56-0.68  GUI moves back to the middle; text fades.
//   E 0.68-0.80  mosaic thumbnails emerge from the middle; GUI fades out.
//   F 0.78-1.00  mosaic flies outward; stats reveal + count up in the middle.
//
// Assets come from project_website/assets/collection/ (see tools/export_collection_assets.py).

const BASE = "assets/collection";

async function loadJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} -> ${res.status}`);
  return res.json();
}

function rand(seed) {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
}

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function easeInOut(t) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

// Golden-angle scatter, pushed outward so the center stays clear for the stats.
function computeScatter(n, rect) {
  const maxRx = rect.width * 0.44;
  const maxRy = rect.height * 0.44;
  const golden = Math.PI * (3 - Math.sqrt(5));
  const out = [];
  for (let i = 0; i < n; i++) {
    const frac = 0.4 + 0.6 * Math.sqrt((i + 0.5) / n);
    const angle = i * golden;
    const jitter = (rand(i) - 0.5) * 0.16;
    out.push({
      tx: Math.cos(angle) * (frac + jitter) * maxRx,
      ty: Math.sin(angle) * (frac + jitter) * maxRy,
      rot: (rand(i + 7) - 0.5) * 24,
      sc: 0.7 + rand(i + 31) * 0.5,
    });
  }
  return out;
}

function buildCards(mosaicEl, mosaic) {
  mosaicEl.innerHTML = "";
  const cards = [];
  mosaic.forEach((entry) => {
    const card = document.createElement("div");
    card.className = "mosaic__card";
    const img = document.createElement("img");
    img.src = `${BASE}/${entry.file}`;
    img.alt = "";
    img.loading = "lazy";
    card.appendChild(img);
    mosaicEl.appendChild(card);
    cards.push(card);
  });
  return cards;
}

export async function initCollection({ full }) {
  const section = document.querySelector("#teaching-collection");
  if (!section) return;

  const stage = section.querySelector(".collection-stage");
  const guiWindow = section.querySelector("#collection-gui");
  const canvas = section.querySelector("#collection-gui-canvas");
  const mosaicEl = section.querySelector("#collection-mosaic");
  const head = section.querySelector(".collection-head");
  const statsPanel = section.querySelector("#collection-stats");
  const statNums = section.querySelectorAll("[data-count-to]");

  let manifest, states, constants;
  try {
    [manifest, states, constants] = await Promise.all([
      loadJSON(`${BASE}/manifest.json`),
      loadJSON(`${BASE}/gui/states.json`),
      loadJSON(`${BASE}/gui/constants.json`),
    ]);
  } catch (err) {
    console.warn("collection: assets missing, leaving static.", err);
    // Reveal the static copy so the section isn't blank (.collection-head is opacity:0 in CSS).
    if (head) head.style.opacity = "1";
    if (statsPanel) statsPanel.style.opacity = "1";
    statNums.forEach((el) => (el.textContent = el.dataset.countTo));
    return;
  }

  const hasRenderer = Boolean(window.BridgeRenderer);
  const nSteps = states.length;

  let drawnStep = -1;
  function drawStep(step) {
    if (!hasRenderer) return;
    const idx = Math.max(0, Math.min(step, nSteps - 1));
    if (idx === drawnStep) return;
    drawnStep = idx;
    window.BridgeRenderer.drawScene(canvas, states[idx], constants, { bars: states[idx].bars });
  }

  const cards = buildCards(mosaicEl, manifest.mosaic || []);
  const setNum = (el, v) => (el.textContent = String(Math.round(v)));

  // ---- Reduced-motion / small-screen fallback: static, everything visible ----
  const reduce = !full || window.innerWidth < 760;
  if (reduce) {
    drawStep(nSteps - 1);
    guiWindow.style.transform = "none";
    guiWindow.style.opacity = "1";
    if (head) head.style.opacity = "1";
    statsPanel.style.opacity = "1";
    mosaicEl.classList.add("mosaic--static");
    cards.forEach((c) => {
      c.style.opacity = "1";
      c.style.setProperty("--tx", "0px");
      c.style.setProperty("--ty", "0px");
      c.style.setProperty("--rot", "0deg");
      c.style.setProperty("--sc", "1");
      c.style.setProperty("--shadowA", "1");
    });
    statNums.forEach((el) => setNum(el, Number(el.dataset.countTo)));
    return;
  }

  // Radial scatter is sized from the full stage (its inner is height-collapsed).
  let scatter = computeScatter(cards.length, stage.getBoundingClientRect());
  let guiX = stage.getBoundingClientRect().width * 0.24;
  const refresh = () => {
    const rect = stage.getBoundingClientRect();
    scatter = computeScatter(cards.length, rect);
    guiX = rect.width * 0.24;
  };

  function render(p) {
    // --- GUI: entrance scale, move right, build, recenter, fade out ---
    const enter = easeInOut(clamp01(p / 0.12));

    let xf = 0;
    if (p < 0.24) xf = easeInOut(clamp01((p - 0.12) / 0.12)); // 0 -> 1 (move right)
    else if (p < 0.56) xf = 1;                                // hold right
    else xf = 1 - easeInOut(clamp01((p - 0.56) / 0.12));      // 1 -> 0 (recenter)

    const guiFade = clamp01((p - 0.68) / 0.1);
    const guiOpacity = enter * (1 - guiFade);
    const guiScale = enter * (1 - 0.08 * guiFade);
    guiWindow.style.opacity = String(guiOpacity);
    guiWindow.style.transform = `translateX(${guiX * xf}px) scale(${guiScale})`;

    // Sequential placement
    drawStep(Math.floor(clamp01((p - 0.24) / 0.32) * nSteps));

    // Explanatory copy on the left during the build
    if (head) {
      const headIn = clamp01((p - 0.12) / 0.1);
      const headOut = clamp01((p - 0.56) / 0.1);
      head.style.opacity = String(headIn * (1 - headOut));
    }

    // --- Mosaic: emerge from center, then fly outward ---
    const emerge = clamp01((p - 0.68) / 0.1);
    const fly = easeInOut(clamp01((p - 0.78) / 0.22));
    for (let i = 0; i < cards.length; i++) {
      const s = scatter[i];
      const card = cards[i];
      card.style.opacity = String(emerge);
      card.style.setProperty("--tx", `${s.tx * fly}px`);
      card.style.setProperty("--ty", `${s.ty * fly}px`);
      card.style.setProperty("--rot", `${s.rot * fly}deg`);
      card.style.setProperty("--sc", String(s.sc * emerge));
      // Shadows grow in as cards scatter, so the stacked-at-center start doesn't
      // pile overlapping shadows into a dark multiply blob.
      card.style.setProperty("--shadowA", String(fly));
    }

    // Centered stats reveal together with the fly-out
    statsPanel.style.opacity = String(clamp01((p - 0.8) / 0.12));
    const so = easeInOut(clamp01((p - 0.8) / 0.18));
    statNums.forEach((el) => setNum(el, Number(el.dataset.countTo) * so));
  }

  ScrollTrigger.create({
    trigger: section,
    start: "top top",
    end: `+=${nSteps * 80 + 3000}`,
    pin: section.querySelector(".collection-stage"),
    scrub: true,
    onRefresh: refresh,
    onUpdate: (self) => render(self.progress),
  });

  render(0);
}
