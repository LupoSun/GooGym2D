// Curation — pinned, scroll-driven beat.
//
// Gallery GUI lifecycle, then a horizontally-panning "formation with selection":
//   0.00-0.10  fly in (from left) + scale up to target
//   0.10-0.20  scale up (pulse)
//   0.20-0.30  scale down
//   0.30-0.40  fly out (to the right)
//   0.38-0.46  formation grid reveals
//   0.46-1.00  grid pans horizontally; selected demos lift + glow pink (staggered)
//
// Assets from project_website/assets/curation/ (see tools/export_curation_assets.py).

const BASE = "assets/curation";

async function loadJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} -> ${res.status}`);
  return res.json();
}

function rand(seed) {
  const x = Math.sin(seed * 91.7 + 47.3) * 43758.5453;
  return x - Math.floor(x);
}

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

function easeInOut(t) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

function buildCells(track, grid, rows) {
  track.innerHTML = "";
  track.style.setProperty("--rows", String(rows));
  const cells = [];
  grid.forEach((entry) => {
    const cell = document.createElement("div");
    cell.className = "formation__cell";
    const img = document.createElement("img");
    img.src = `${BASE}/${entry.file}`;
    img.alt = "";
    img.loading = "lazy";
    cell.appendChild(img);
    track.appendChild(cell);
    cells.push(cell);
  });
  return cells;
}

// Pick a spread of cells to lift, deterministically.
function pickSelected(nCells) {
  const nSel = Math.max(3, Math.round(nCells * 0.18));
  const order = Array.from({ length: nCells }, (_, i) => i).sort(
    (a, b) => rand(a + 1) - rand(b + 1)
  );
  return order.slice(0, nSel);
}

export async function initCuration({ full }) {
  const section = document.querySelector("#teaching-curation");
  if (!section) return;

  const stage = section.querySelector(".curation-stage");
  const gui = section.querySelector("#curation-gui");
  const formation = section.querySelector("#curation-formation");
  const track = section.querySelector("#curation-track");

  let manifest;
  try {
    manifest = await loadJSON(`${BASE}/manifest.json`);
  } catch (err) {
    console.warn("curation: assets missing, leaving static.", err);
    return;
  }

  const rows = manifest.rows || 4;
  const cells = buildCells(track, manifest.grid || [], rows);
  const selectedIdx = pickSelected(cells.length);

  // ---- Reduced-motion / small-screen fallback ----
  if (!full || window.innerWidth < 760) {
    gui.style.opacity = "0";
    formation.style.opacity = "1";
    track.style.transform = "none";
    selectedIdx.forEach((k) => cells[k].classList.add("is-lifted"));
    return;
  }

  let stageW = stage.getBoundingClientRect().width;
  let offIn = stageW * 0.65;
  let offOut = stageW * 0.75;
  let restX = 0;     // GUI rests right-of-center so the left copy doesn't cover it
  let panBase = 0;   // track x at pan progress 0
  let panRange = 0;  // total horizontal travel
  let viewW = stageW;
  let selected = []; // { el, threshold }

  function refresh() {
    stageW = stage.getBoundingClientRect().width;
    offIn = stageW * 0.65;
    offOut = stageW * 0.75;

    // Park the GUI right-of-center so the left copy doesn't cover it — but leave
    // room for the 1.12x magnify so the (scaled) window never crops at the edge.
    // Use offsetWidth (layout width) NOT getBoundingClientRect (which returns the
    // render()-applied scale, giving an unstable restX that drifts each refresh).
    const guiW = gui.offsetWidth || stageW * 0.72;
    const PEAK = 1.12;
    const MARGIN = 24;
    restX = Math.max(0, (stageW - guiW * PEAK) / 2 - MARGIN);

    // The track pans inside the .formation viewport.
    viewW = formation.getBoundingClientRect().width || stageW;
    const trackW = track.scrollWidth;
    const slack = Math.max(0, trackW - viewW);

    // Keep a buffer of content beyond BOTH edges at all times so the rail's
    // ends never enter view -> the pan never "feels like the end".
    const buf = viewW * 0.12;
    panRange = Math.max(0, slack - 2 * buf);
    panBase = -buf;

    // Lift the selected cells in a steady trickle across the whole pan, ordered
    // left-to-right by column. (Tying lift to a column "crossing the viewport
    // center" clusters at the start/end once the formation is full-bleed, since
    // most columns fall outside the narrowed center-crossing band.)
    const ordered = [...selectedIdx].sort(
      (a, b) => Math.floor(a / rows) - Math.floor(b / rows)
    );
    const n = ordered.length;
    selected = ordered.map((k, i) => ({
      el: cells[k],
      threshold: n > 1 ? (i + 0.5) / n : 0.5,
    }));
  }
  refresh();

  function render(p) {
    // --- Gallery GUI lifecycle (rests at restX, right-of-center) ---
    let x = restX, scale = 1, op = 1;
    if (p < 0.1) {
      const t = easeInOut(clamp01(p / 0.1));
      x = restX - offIn * (1 - t); // fly in from the left to the rest spot
      scale = 0.6 + 0.4 * t;
      op = t;
    } else if (p < 0.2) {
      const t = easeInOut(clamp01((p - 0.1) / 0.1));
      scale = 1 + 0.12 * t; // magnify
    } else if (p < 0.3) {
      const t = easeInOut(clamp01((p - 0.2) / 0.1));
      scale = 1.12 - 0.2 * t; // 1.12 -> 0.92
    } else if (p < 0.4) {
      const t = easeInOut(clamp01((p - 0.3) / 0.1));
      x = restX + offOut * t; // fly out to the right
      scale = 0.92;
      op = 1 - t;
    } else {
      x = restX + offOut;
      scale = 0.92;
      op = 0;
    }
    gui.style.transform = `translateX(${x}px) scale(${scale})`;
    gui.style.opacity = String(op);

    // --- Formation reveal + horizontal pan ---
    formation.style.opacity = String(clamp01((p - 0.38) / 0.08));
    const pan = clamp01((p - 0.46) / 0.54);
    track.style.transform = `translateX(${panBase - panRange * pan}px)`;

    // Staggered lift + glow as each selected column reaches center
    for (const s of selected) {
      s.el.classList.toggle("is-lifted", pan >= s.threshold);
    }
  }

  ScrollTrigger.create({
    trigger: section,
    start: "top top",
    end: `+=${(manifest.grid ? manifest.grid.length : 48) * 70 + 3200}`,
    pin: stage,
    scrub: true,
    onRefresh: refresh,
    onUpdate: (self) => render(self.progress),
  });

  render(0);
}
