// Hero — load-in stagger + an ambient Open Color flow-field background that
// reacts to the cursor (particles swirl/scatter around the pointer).
// `full` motion when prefers-reduced-motion: no-preference; reduced = static.

// Open Color accents (the flowchart family) — each particle picks one.
const PALETTE = ["#32b4dc", "#f06595", "#0ca678", "#fd7e14", "#845ef7", "#82c91e", "#4dabf7"];

function clamp(v, a, b) {
  return v < a ? a : v > b ? b : v;
}

export function initHero({ full }) {
  const hero = document.querySelector("#hero");
  if (!hero) return;

  // --- Text load-in stagger ---
  const items = hero.querySelectorAll(".js-anim");
  if (items.length) {
    if (!full) {
      gsap.set(items, { clearProps: "all", opacity: 1, y: 0 });
    } else {
      gsap.set(items, { opacity: 0, y: 24 });
      gsap
        .timeline({ defaults: { ease: "power3.out", duration: 0.8 }, delay: 0.1 })
        .to(items, { opacity: 1, y: 0, stagger: 0.12 });
    }
  }

  // --- Flow-field background ---
  const canvas = hero.querySelector("#hero-field");
  if (!canvas || !full) return; // reduced motion: leave the static gradient
  const ctx = canvas.getContext("2d");

  let W = 0, H = 0, dpr = 1, t = 0;
  let particles = [];
  let rafId = 0, running = false;
  const mouse = { x: 0, y: 0, active: false };

  function spawn(p, fresh) {
    p.x = Math.random() * W;
    p.y = Math.random() * H;
    p.life = 80 + Math.random() * 180;
    p.age = fresh ? Math.random() * p.life : 0;
    if (fresh) p.color = PALETTE[(Math.random() * PALETTE.length) | 0];
  }

  function seed() {
    const n = Math.round(clamp(W / 4.5, 200, 520));
    particles = [];
    for (let i = 0; i < n; i++) {
      const p = {};
      spawn(p, true);
      particles.push(p);
    }
  }

  function resize() {
    const r = hero.getBoundingClientRect();
    W = r.width;
    H = r.height;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    seed();
  }

  // Smooth, slowly-evolving direction field (cheap layered sines, curl-ish).
  function angle(x, y) {
    const nx = x / W, ny = y / H;
    return (
      (Math.sin(nx * 3.0 + t * 0.25) +
        Math.cos(ny * 3.0 - t * 0.2) +
        Math.cos(nx * 1.7 - t * 0.18) +
        Math.sin(ny * 2.3 + t * 0.22)) *
      0.8
    );
  }

  function frame() {
    t += 0.016;
    ctx.clearRect(0, 0, W, H);

    const speed = Math.max(0.7, W * 0.0013);
    const tail = Math.max(18, W * 0.022);
    const R = Math.min(220, Math.max(W, H) * 0.18); // cursor influence radius
    ctx.lineCap = "round";
    ctx.lineWidth = 1.7;

    for (const p of particles) {
      const a = angle(p.x, p.y);
      let vx = Math.cos(a) * speed;
      let vy = Math.sin(a) * speed;

      // Cursor reaction: swirl (tangential) + gentle push-out near the pointer.
      let boost = 0;
      if (mouse.active) {
        const dx = p.x - mouse.x, dy = p.y - mouse.y;
        const dist = Math.hypot(dx, dy) + 0.001;
        if (dist < R) {
          const f = 1 - dist / R;
          boost = f;
          const nx = dx / dist, ny = dy / dist;
          const swirl = speed * 4 * f;
          const push = speed * 2 * f;
          vx += -ny * swirl + nx * push;
          vy += nx * swirl + ny * push;
        }
      }

      const px = p.x, py = p.y;
      p.x += vx;
      p.y += vy;
      p.age++;
      if (p.x < -30 || p.x > W + 30 || p.y < -30 || p.y > H + 30 || p.age > p.life) {
        spawn(p, false);
        continue;
      }

      // Fade in at birth, out near death; brighten near the cursor.
      const lifeT = Math.min(p.age / 14, 1, (p.life - p.age) / 20);
      ctx.globalAlpha = clamp((0.34 + 0.4 * boost) * Math.max(0, lifeT), 0, 0.85);
      ctx.strokeStyle = p.color;
      const dirx = vx, diry = vy;
      const m = Math.hypot(dirx, diry) || 1;
      ctx.beginPath();
      ctx.moveTo(p.x - (dirx / m) * tail, p.y - (diry / m) * tail);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
    if (running) rafId = requestAnimationFrame(frame);
  }

  function start() {
    if (running) return;
    running = true;
    rafId = requestAnimationFrame(frame);
  }
  function stop() {
    running = false;
    cancelAnimationFrame(rafId);
  }

  function onMove(e) {
    const r = hero.getBoundingClientRect();
    mouse.x = e.clientX - r.left;
    mouse.y = e.clientY - r.top;
    mouse.active = true;
  }

  resize();
  start();
  window.addEventListener("resize", resize);
  hero.addEventListener("pointermove", onMove);
  hero.addEventListener("pointerleave", () => (mouse.active = false));

  // Only animate while the hero is on screen.
  const io = new IntersectionObserver(
    (entries) => {
      for (const e of entries) (e.isIntersecting ? start() : stop());
    },
    { rootMargin: "10% 0px 10% 0px" }
  );
  io.observe(hero);
}
