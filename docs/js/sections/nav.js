// Nav scroll-spy — highlights the nav link for the chapter currently in view so
// the reader stays oriented. Sections are grouped under the four nav items:
//   Problem        → problem
//   Method         → method, collection, curation, training, correction, co-design
//   Results        → results
//   Papers & code  → project
// Works independently of GSAP/motion (plain scroll listener, rAF-throttled).

const GROUPS = [
  { sel: 'a[href="#problem"]', ids: ["problem"] },
  {
    sel: 'a[href="#environment"]',
    ids: [
      "environment",
      "framework",
      "teaching-collection",
      "teaching-curation",
      "teaching-training",
      "teaching-correction",
      "teaming-codesign",
    ],
  },
  { sel: 'a[href="#results"]', ids: ["results"] },
  { sel: 'a[href="#project"]', ids: ["project"] },
];

export function initNav() {
  const nav = document.querySelector(".site-nav");
  if (!nav) return;

  const groups = GROUPS.map((g) => ({ el: nav.querySelector(g.sel), ids: g.ids }));
  const idToLink = new Map();
  const sections = [];
  for (const g of groups) {
    for (const id of g.ids) {
      const sec = document.getElementById(id);
      if (sec && g.el) {
        idToLink.set(sec, g.el);
        sections.push(sec);
      }
    }
  }
  if (!sections.length) return;

  let ticking = false;
  function update() {
    ticking = false;
    const mid = window.innerHeight * 0.5;
    let active = null;
    for (const sec of sections) {
      const r = sec.getBoundingClientRect();
      if (r.top <= mid && r.bottom >= mid) {
        active = idToLink.get(sec);
        break;
      }
    }
    for (const g of groups) {
      if (!g.el) continue;
      const on = g.el === active;
      g.el.classList.toggle("is-active", on);
      if (on) g.el.setAttribute("aria-current", "true");
      else g.el.removeAttribute("aria-current");
    }
  }
  function onScroll() {
    if (!ticking) {
      ticking = true;
      requestAnimationFrame(update);
    }
  }

  window.addEventListener("scroll", onScroll, { passive: true });
  window.addEventListener("resize", onScroll);
  update();
}
