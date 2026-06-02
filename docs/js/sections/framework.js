// Framework — pinned beat: copy on the left; the flowchart flies up from the
// bottom, holds in the middle, then flies out the top as you scroll.
//   p 0.00-0.33  fly in from below → centered
//   p 0.33-0.66  hold centered
//   p 0.66-1.00  fly out the top
// The stage clips (overflow hidden) so it enters/exits off the edges.

function clamp01(v) {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}
function easeInOut(t) {
  return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
}

export function initFramework({ full }) {
  const section = document.querySelector("#framework");
  if (!section) return;
  const stage = section.querySelector(".framework-stage");
  const copy = section.querySelectorAll(".framework-copy .js-anim");
  const flow = section.querySelector(".framework-flow");
  if (!stage || !flow) return;

  // ---- Reduced-motion / small screen: static, centered, visible ----
  if (!full || window.innerWidth < 760) {
    if (copy.length) gsap.set(copy, { opacity: 1, y: 0 });
    flow.style.transform = "none";
    flow.style.opacity = "1";
    return;
  }

  // Copy fades/slides in once and then stays put for the whole pin.
  gsap.set(copy, { opacity: 0, y: 28 });
  gsap.to(copy, {
    opacity: 1,
    y: 0,
    ease: "power2.out",
    duration: 0.7,
    stagger: 0.12,
    scrollTrigger: { trigger: section, start: "top 70%" },
  });

  let off = 0; // travel distance (px) to clear the stage top/bottom
  function refresh() {
    const h = stage.getBoundingClientRect().height || window.innerHeight;
    off = h * 0.62;
  }
  refresh();

  const PEAK = 1.2; // slight magnification while centered
  function render(p) {
    let y, s;
    if (p < 0.33) {
      const t = easeInOut(p / 0.33);
      y = off * (1 - t); //          +off → 0   (fly in)
      s = 1 + (PEAK - 1) * t; //     grow to peak as it centers
    } else if (p < 0.66) {
      y = 0; //                      hold centered
      s = PEAK; //                   magnified
    } else {
      const t = easeInOut((p - 0.66) / 0.34);
      y = -off * t; //               0 → -off   (fly out)
      s = PEAK - (PEAK - 1) * t; //  shrink back to original while leaving
    }
    const op = clamp01(p / 0.1) * (1 - clamp01((p - 0.9) / 0.1));
    flow.style.transform = `translateY(${y}px) scale(${s})`;
    flow.style.opacity = String(op);
  }

  ScrollTrigger.create({
    trigger: section,
    start: "top top",
    end: "+=2200",
    pin: stage,
    scrub: true,
    onRefresh: refresh,
    onUpdate: (self) => render(self.progress),
  });

  render(0);
}
