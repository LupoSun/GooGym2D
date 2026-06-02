// Project information — closing beat: staggered fade-in + BibTeX copy buttons.

export function initProject({ full }) {
  const section = document.querySelector("#project");
  if (!section) return;

  const items = section.querySelectorAll(".js-anim");

  if (full) {
    gsap.set(items, { opacity: 0, y: 28 });
    gsap.to(items, {
      opacity: 1,
      y: 0,
      ease: "power2.out",
      duration: 0.7,
      stagger: 0.1,
      scrollTrigger: { trigger: section, start: "top 75%" },
    });
  } else {
    gsap.set(items, { opacity: 1, y: 0 });
  }

  // Cite toggle — smooth expand/collapse (grid-rows transition in CSS).
  section.querySelectorAll(".paper-cite__summary").forEach((btn) => {
    btn.addEventListener("click", () => {
      const cite = btn.closest(".paper-cite");
      const open = cite.classList.toggle("is-open");
      btn.setAttribute("aria-expanded", String(open));
    });
  });

  // BibTeX copy-to-clipboard.
  section.querySelectorAll(".paper-cite__copy").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const pre = section.querySelector(`#${btn.dataset.copyTarget}`);
      if (!pre) return;
      try {
        await navigator.clipboard.writeText(pre.textContent);
      } catch (err) {
        // Fallback: select the text so the user can copy manually.
        const range = document.createRange();
        range.selectNodeContents(pre);
        const sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
      }
      const original = btn.textContent;
      btn.textContent = "Copied";
      btn.classList.add("is-copied");
      setTimeout(() => {
        btn.textContent = original;
        btn.classList.remove("is-copied");
      }, 1600);
    });
  });
}
