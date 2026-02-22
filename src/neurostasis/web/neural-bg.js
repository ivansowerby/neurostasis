(() => {
  const existing = document.getElementById("bg-net");
  const canvas = existing || document.createElement("canvas");
  canvas.id = "bg-net";
  if (!existing) {
    document.body.prepend(canvas);
  }

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  let graph = { nodes: [], edges: [] };
  let lastW = 0;
  let lastH = 0;
  let lastSpacing = 40;
  let mouseX = -1;
  let mouseY = -1;
  let mouseActive = false;
  let mouseMoveAt = 0;

  function makeRng(seed) {
    let s = seed >>> 0;
    return () => {
      s ^= s << 13;
      s ^= s >>> 17;
      s ^= s << 5;
      return (s >>> 0) / 4294967296;
    };
  }

  function palette() {
    if (document.body.classList.contains("phase-light")) {
      return { edge: "rgba(28, 38, 56, 0.10)", hub: "rgba(38, 52, 79, 0.36)", node: "rgba(38, 52, 79, 0.20)" };
    }
    if (document.body.classList.contains("phase-post")) {
      return { edge: "rgba(170, 188, 222, 0.09)", hub: "rgba(212, 226, 248, 0.30)", node: "rgba(198, 215, 245, 0.18)" };
    }
    return { edge: "rgba(148, 168, 207, 0.10)", hub: "rgba(188, 209, 247, 0.34)", node: "rgba(170, 194, 238, 0.20)" };
  }

  function interactionEnabled() {
    const inRun = document.body.classList.contains("run-active");
    const results = document.getElementById("results-panel");
    const inResults = results && !results.classList.contains("hidden");
    return !(inRun || inResults);
  }

  function buildGraph(w, h) {
    const seed = (w * 73856093) ^ (h * 19349663) ^ 0xa53f9b1d;
    const rng = makeRng(seed);
    const spacing = Math.max(30, Math.min(46, Math.round(w / 38)));
    lastSpacing = spacing;
    const dy = spacing * 0.866;
    const cols = Math.ceil(w / spacing) + 3;
    const rows = Math.ceil(h / dy) + 3;

    const nodes = [];
    const idx = Array.from({ length: rows }, () => Array(cols).fill(-1));
    for (let r = 0; r < rows; r += 1) {
      const rowOffset = (r % 2) * (spacing / 2);
      for (let c = 0; c < cols; c += 1) {
        const jx = (rng() - 0.5) * spacing * 0.18;
        const jy = (rng() - 0.5) * dy * 0.18;
        const x = c * spacing + rowOffset + jx - spacing;
        const y = r * dy + jy - dy;
        const i = nodes.length;
        nodes.push({ x, y, bx: x, by: y, vx: 0, vy: 0, deg: 0, hub: false });
        idx[r][c] = i;
      }
    }

    const edgeSet = new Set();
    const edges = [];
    const addEdge = (a, b) => {
      if (a < 0 || b < 0 || a === b) return;
      const k = a < b ? `${a}-${b}` : `${b}-${a}`;
      if (edgeSet.has(k)) return;
      edgeSet.add(k);
      edges.push([a, b]);
      nodes[a].deg += 1;
      nodes[b].deg += 1;
    };

    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const a = idx[r][c];
        addEdge(a, idx[r][c + 1] ?? -1);
        if (r + 1 < rows) {
          if (r % 2 === 0) {
            addEdge(a, idx[r + 1][c] ?? -1);
            addEdge(a, idx[r + 1][c - 1] ?? -1);
          } else {
            addEdge(a, idx[r + 1][c] ?? -1);
            addEdge(a, idx[r + 1][c + 1] ?? -1);
          }
        }
      }
    }

    const hubCount = Math.max(8, Math.round(nodes.length * 0.024));
    const chosen = new Set();
    for (let i = 0; i < hubCount; i += 1) {
      let pick = Math.floor(rng() * nodes.length);
      let guard = 0;
      while (chosen.has(pick) && guard < 60) {
        pick = Math.floor(rng() * nodes.length);
        guard += 1;
      }
      chosen.add(pick);
      nodes[pick].hub = true;
    }

    const maxRadius = spacing * 8.4;
    for (const hubIdx of chosen) {
      const hub = nodes[hubIdx];
      const candidates = [];
      for (let j = 0; j < nodes.length; j += 1) {
        if (j === hubIdx) continue;
        const n = nodes[j];
        const dx = n.x - hub.x;
        const dy2 = n.y - hub.y;
        const d = Math.hypot(dx, dy2);
        if (d < maxRadius) {
          candidates.push({ j, d });
        }
      }
      candidates.sort((a, b) => a.d - b.d);
      const links = 8 + Math.floor(rng() * 8);
      for (let k = 0; k < Math.min(links, candidates.length); k += 1) {
        addEdge(hubIdx, candidates[k].j);
      }

      // Add a few broader-range links per hub for global/local mixing.
      const farLinks = 2 + Math.floor(rng() * 2);
      for (let f = 0; f < farLinks; f += 1) {
        const idxPick = Math.floor(rng() * candidates.length);
        const target = candidates[candidates.length - 1 - idxPick];
        if (target) addEdge(hubIdx, target.j);
      }
    }
    return { nodes, edges };
  }

  function resizeMaybeRebuild(force = false) {
    const w = Math.max(1, window.innerWidth);
    const h = Math.max(1, window.innerHeight);
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(w * dpr);
    canvas.height = Math.floor(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    if (force || Math.abs(w - lastW) > 36 || Math.abs(h - lastH) > 36 || graph.nodes.length === 0) {
      graph = buildGraph(w, h);
      lastW = w;
      lastH = h;
    }
  }

  function draw() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const p = palette();
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = p.edge;
    ctx.lineWidth = 0.9;
    for (const [a, b] of graph.edges) {
      const na = graph.nodes[a];
      const nb = graph.nodes[b];
      ctx.beginPath();
      ctx.moveTo(na.x, na.y);
      ctx.lineTo(nb.x, nb.y);
      ctx.stroke();
    }

    for (const n of graph.nodes) {
      ctx.fillStyle = n.hub ? p.hub : p.node;
      const r = n.hub ? 2.3 : 1.35;
      ctx.beginPath();
      ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  function step(dt, nowMs) {
    const attractRadius = lastSpacing * 4.8;
    const attractRadiusSq = attractRadius * attractRadius;
    const springK = 8.8;
    const attractK = 2200;
    const hubBoost = 1.34;
    const damping = Math.exp(-6.4 * dt);
    const allowInteraction = interactionEnabled();
    const recentlyMoved = allowInteraction && mouseActive && nowMs - mouseMoveAt < 180;

    let energy = 0;
    for (const n of graph.nodes) {
      let ax = (n.bx - n.x) * springK;
      let ay = (n.by - n.y) * springK;

      if (recentlyMoved) {
        const dx = mouseX - n.x;
        const dy = mouseY - n.y;
        const d2 = dx * dx + dy * dy;
        if (d2 > 1e-6 && d2 < attractRadiusSq) {
          const d = Math.sqrt(d2);
          const t = 1 - d / attractRadius;
          const pull = attractK * t * t;
          const m = n.hub ? hubBoost : 1;
          ax += (dx / d) * pull * m;
          ay += (dy / d) * pull * m;
        }
      }

      n.vx = (n.vx + ax * dt) * damping;
      n.vy = (n.vy + ay * dt) * damping;
      n.x += n.vx * dt;
      n.y += n.vy * dt;
      energy += Math.abs(n.vx) + Math.abs(n.vy);
    }
    return energy / Math.max(1, graph.nodes.length);
  }

  let raf = 0;
  let running = false;
  let lastTs = 0;
  function frame(ts) {
    if (!running) return;
    const dt = Math.min(0.032, Math.max(0.008, (ts - lastTs) / 1000 || 0.016));
    lastTs = ts;
    resizeMaybeRebuild(false);
    const e = step(dt, ts);
    draw();
    const settling = e > 0.0013;
    const allowInteraction = interactionEnabled();
    const recentlyMoved = allowInteraction && mouseActive && ts - mouseMoveAt < 180;
    if (settling || recentlyMoved) {
      raf = requestAnimationFrame(frame);
    } else {
      running = false;
      raf = 0;
    }
  }

  function ensureRun(forceRebuild = false) {
    resizeMaybeRebuild(forceRebuild);
    if (!running) {
      running = true;
      lastTs = performance.now();
      raf = requestAnimationFrame(frame);
    }
  }

  const observer = new MutationObserver(() => ensureRun(false));
  observer.observe(document.body, { attributes: true, attributeFilter: ["class"] });
  window.addEventListener("resize", () => ensureRun(true));
  window.addEventListener("pointermove", (ev) => {
    if (!interactionEnabled()) return;
    mouseX = ev.clientX;
    mouseY = ev.clientY;
    mouseActive = true;
    mouseMoveAt = performance.now();
    ensureRun(false);
  });
  window.addEventListener("pointerleave", () => {
    mouseActive = false;
    mouseMoveAt = performance.now();
    ensureRun(false);
  });
  window.addEventListener(
    "touchmove",
    (ev) => {
      if (!interactionEnabled()) return;
      if (!ev.touches || ev.touches.length === 0) return;
      mouseX = ev.touches[0].clientX;
      mouseY = ev.touches[0].clientY;
      mouseActive = true;
      mouseMoveAt = performance.now();
      ensureRun(false);
    },
    { passive: true }
  );
  window.addEventListener("touchend", () => {
    mouseActive = false;
    mouseMoveAt = performance.now();
    ensureRun(false);
  });
  window.redrawNeuralBackground = () => ensureRun(false);
  ensureRun(true);
})();
