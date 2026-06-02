// Problem — scrub-in reveal on enter + stat count-up.

export function initProblem({ full }) {
  const section = document.querySelector("#problem");
  if (!section) return;

  const items = section.querySelectorAll(".js-anim");
  const counter = section.querySelector("[data-count-to]");

  if (!full) {
    gsap.set(items, { opacity: 1, y: 0 });
    if (counter) counter.textContent = formatCount(counter);
    return;
  }

  gsap.set(items, { opacity: 0, y: 32 });

  gsap.to(items, {
    opacity: 1,
    y: 0,
    ease: "power2.out",
    duration: 0.8,
    stagger: 0.15,
    scrollTrigger: {
      trigger: section,
      start: "top 70%",
    },
  });

  if (counter) {
    const target = parseFloat(counter.dataset.countTo) || 0;
    const suffix = counter.dataset.countSuffix || "";
    const obj = { val: 0 };
    gsap.to(obj, {
      val: target,
      ease: "power1.out",
      duration: 1.4,
      scrollTrigger: { trigger: counter, start: "top 80%" },
      onUpdate: () => {
        counter.textContent = formatValue(obj.val, target) + suffix;
      },
    });
  }
}

function formatValue(val, target) {
  return Number.isInteger(target) ? Math.round(val).toString() : val.toFixed(1);
}

function formatCount(el) {
  const target = parseFloat(el.dataset.countTo) || 0;
  const suffix = el.dataset.countSuffix || "";
  return formatValue(target, target) + suffix;
}
