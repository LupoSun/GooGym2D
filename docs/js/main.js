// GooGym2D — GSAP entry point.
// Registers ScrollTrigger, gates motion on prefers-reduced-motion, inits sections.
// GSAP + ScrollTrigger are loaded as globals via CDN <script> tags in index.html.

import { initHero } from "./sections/hero.js";
import { initProblem } from "./sections/problem.js";
import { initMethod } from "./sections/method.js";
import { initFramework } from "./sections/framework.js";
import { initCollection } from "./sections/collection.js";
import { initCuration } from "./sections/curation.js";
import { initTraining } from "./sections/training.js";
import { initCorrection } from "./sections/correction.js";
import { initCodesign } from "./sections/codesign.js";
import { initResults } from "./sections/results.js";
import { initProject } from "./sections/project.js";
import { initNav } from "./sections/nav.js";

function boot() {
  initNav(); // nav scroll-spy; independent of GSAP/motion

  if (!window.gsap || !window.ScrollTrigger) {
    console.warn("project_website: GSAP not available; rendering static page.");
    document.querySelectorAll(".js-anim").forEach((el) => {
      el.style.opacity = "1";
    });
    return;
  }

  gsap.registerPlugin(ScrollTrigger);

  const mm = gsap.matchMedia();

  // Full scroll-driven motion.
  mm.add("(prefers-reduced-motion: no-preference)", () => {
    const ctx = { full: true };
    initHero(ctx);
    initProblem(ctx);
    initMethod(ctx);
    // Pinned sections must create their ScrollTriggers in DOM order so pin
    // spacing/start positions resolve correctly. Init sequentially, then refresh.
    (async () => {
      initFramework(ctx); // pinned — first in DOM order
      await initCollection(ctx);
      await initCuration(ctx);
      initTraining(ctx);
      await initCorrection(ctx);
      await initCodesign(ctx);
      await initResults(ctx);
      initProject(ctx);
      ScrollTrigger.refresh();
    })();
  });

  // Reduced motion: render everything visible, no scroll animation.
  mm.add("(prefers-reduced-motion: reduce)", () => {
    const ctx = { full: false };
    // If the user flips to reduced-motion mid-session, the no-preference branch's
    // pinned ScrollTriggers (created in the async IIFE) escape matchMedia's auto-
    // revert — kill any leftover triggers so stale pins don't conflict with the
    // static layout. (Residual rAF/observer/resize listeners aren't torn down; a
    // reload fully resets — acceptable for a static page. See WORKFLOW gotchas.)
    ScrollTrigger.getAll().forEach((t) => t.kill());
    initHero(ctx);
    initProblem(ctx);
    initMethod(ctx);
    initFramework(ctx);
    initCollection(ctx);
    initCuration(ctx);
    initTraining(ctx);
    initCorrection(ctx);
    initCodesign(ctx);
    initResults(ctx);
    initProject(ctx);
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
