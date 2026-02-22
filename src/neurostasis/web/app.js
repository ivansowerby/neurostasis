const $ = (id) => document.getElementById(id);

let totalDuration = 55;

const views = {
  start: $("start-panel"),
  run: $("run-panel"),
  results: $("results-panel"),
};

const phaseLabel = $("phase-label");
const stimulus = $("stimulus");
const gazeCursor = $("gaze-cursor");
const preconnectVideo = $("preconnect-video");
const progressBar = $("progress-bar");
const logEl = $("log");
const graphCanvas = $("pupil-graph");
const metricReasons = $("metric-reasons");
const engagementReasons = $("engagement-reasons");
const gazeBounds = { minX: 0, maxX: 1280, minY: 0, maxY: 720 };
const graphPoints = [];
let lastGraphPlot = [];
let graphTooltip = null;
let preconnectStream = null;

function log(msg) {
  logEl.textContent = msg;
}

function fmt(value, decimals = 4) {
  return value == null ? "-" : Number(value).toFixed(decimals);
}

function fmtScore(value, decimals = 2) {
  return value == null ? "-" : `${Number(value).toFixed(decimals)}/100`;
}

function phaseLabelText(phase) {
  if (phase === "LIGHT_ON") return "Light On";
  if (phase === "POST_LIGHT") return "Post Light";
  if (phase === "BASELINE") return "Baseline";
  return phase;
}

function setPhase(phase) {
  phaseLabel.textContent = phaseLabelText(phase);
  document.body.classList.remove("phase-light", "phase-post");
  stimulus.classList.remove("active", "post");

  if (phase === "LIGHT_ON") {
    document.body.classList.add("phase-light");
    stimulus.classList.add("active");
  } else if (phase === "POST_LIGHT") {
    document.body.classList.add("phase-post");
    stimulus.classList.add("post");
  }
  if (typeof window.redrawNeuralBackground === "function") {
    window.redrawNeuralBackground();
  }
}

function hideGazeCursor() {
  gazeCursor.classList.add("hidden");
}

async function startPreconnectCamera() {
  if (!preconnectVideo) return;
  try {
    preconnectStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    preconnectVideo.srcObject = preconnectStream;
    preconnectVideo.classList.remove("hidden");
    log("Align your face in the center while connecting...");
  } catch {
    preconnectVideo.classList.add("hidden");
    log("Camera preview unavailable. Continuing connection.");
  }
}

function stopPreconnectCamera() {
  if (preconnectStream) {
    for (const track of preconnectStream.getTracks()) {
      track.stop();
    }
    preconnectStream = null;
  }
  if (preconnectVideo) {
    preconnectVideo.srcObject = null;
    preconnectVideo.classList.add("hidden");
  }
}

function getPhaseColor(phase) {
  if (phase === "LIGHT_ON") return "#ffbf2f";
  if (phase === "POST_LIGHT") return "#50e0b8";
  return "#8f9db8";
}

function drawGraph() {
  if (!graphCanvas || views.results.classList.contains("hidden")) return;
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = graphCanvas.clientWidth || 300;
  const cssHeight = graphCanvas.clientHeight || 220;
  graphCanvas.width = Math.floor(cssWidth * dpr);
  graphCanvas.height = Math.floor(cssHeight * dpr);
  const ctx = graphCanvas.getContext("2d");
  if (!ctx) return;
  ctx.scale(dpr, dpr);

  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.strokeStyle = "rgba(148, 160, 184, 0.35)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i += 1) {
    const y = (cssHeight / 4) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(cssWidth, y);
    ctx.stroke();
  }

  const valid = graphPoints.filter((p) => p.pupil != null);
  if (valid.length < 2) return;

  const maxT = Math.max(1, graphPoints[graphPoints.length - 1].t);
  const minP = Math.min(...valid.map((p) => p.pupil));
  const maxP = Math.max(...valid.map((p) => p.pupil));
  const spanP = Math.max(0.15, maxP - minP);
  const xPad = 8;
  const yPad = 8;
  const plotW = cssWidth - xPad * 2;
  const plotH = cssHeight - yPad * 2;

  const rendered = [];
  for (let i = 0; i < graphPoints.length; i += 1) {
    const p = graphPoints[i];
    if (p.pupil == null) continue;
    const x = xPad + (p.t / maxT) * plotW;
    const y = yPad + (1 - (p.pupil - minP) / spanP) * plotH;
    const phase = p.phase || "BASELINE";
    rendered.push({ x, y, t: p.t, pupil: p.pupil, phase });
  }

  lastGraphPlot = rendered;
  if (rendered.length < 2) return;

  ctx.lineWidth = 2;
  for (let i = 1; i < rendered.length; i += 1) {
    const a = rendered[i - 1];
    const b = rendered[i];
    ctx.strokeStyle = getPhaseColor(b.phase);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }
}

function ensureGraphTooltip() {
  if (graphTooltip) return graphTooltip;
  graphTooltip = document.createElement("div");
  graphTooltip.className = "canvas-tooltip hidden";
  document.body.appendChild(graphTooltip);
  return graphTooltip;
}

function showGraphTooltip(clientX, clientY, text) {
  const tip = ensureGraphTooltip();
  tip.textContent = text;
  tip.classList.remove("hidden");
  tip.style.left = `${clientX + 12}px`;
  tip.style.top = `${clientY + 12}px`;
}

function hideGraphTooltip() {
  if (!graphTooltip) return;
  graphTooltip.classList.add("hidden");
}

function onResultGraphMove(ev) {
  if (!graphCanvas || views.results.classList.contains("hidden") || lastGraphPlot.length === 0) {
    hideGraphTooltip();
    return;
  }
  const rect = graphCanvas.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const y = ev.clientY - rect.top;
  let best = null;
  let bestD = Infinity;
  for (const p of lastGraphPlot) {
    const d = Math.hypot(p.x - x, p.y - y);
    if (d < bestD) {
      bestD = d;
      best = p;
    }
  }
  if (!best || bestD > 28) {
    hideGraphTooltip();
    return;
  }
  const text = `${phaseLabelText(best.phase)} | ${Number(best.t).toFixed(1)}s | ${Number(best.pupil).toFixed(3)} mm`;
  showGraphTooltip(ev.clientX, ev.clientY, text);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalizeGaze(x, y) {
  if (x == null || y == null) {
    return null;
  }

  if (x >= 0 && x <= 1 && y >= 0 && y <= 1) {
    return { x, y };
  }

  const padX = 30;
  const padY = 20;
  gazeBounds.minX = Math.min(gazeBounds.minX, x - padX);
  gazeBounds.maxX = Math.max(gazeBounds.maxX, x + padX);
  gazeBounds.minY = Math.min(gazeBounds.minY, y - padY);
  gazeBounds.maxY = Math.max(gazeBounds.maxY, y + padY);

  const spanX = Math.max(1, gazeBounds.maxX - gazeBounds.minX);
  const spanY = Math.max(1, gazeBounds.maxY - gazeBounds.minY);
  return {
    x: clamp((x - gazeBounds.minX) / spanX, 0, 1),
    y: clamp((y - gazeBounds.minY) / spanY, 0, 1),
  };
}

function updateGazeCursor(gazeX, gazeY) {
  const norm = normalizeGaze(gazeX, gazeY);
  if (!norm) {
    hideGazeCursor();
    return;
  }

  const radius = stimulus.clientWidth / 2;
  const cursorRadius = gazeCursor.clientWidth / 2 || 12;
  let dx = (norm.x - 0.5) * 2;
  let dy = (norm.y - 0.5) * 2;
  const dist = Math.hypot(dx, dy);
  if (dist > 1) {
    dx /= dist;
    dy /= dist;
  }

  const maxOffset = radius - cursorRadius - 4;
  const px = dx * maxOffset;
  const py = dy * maxOffset;
  gazeCursor.style.transform = `translate(calc(-50% + ${px}px), calc(-50% + ${py}px))`;
  gazeCursor.classList.remove("hidden");
}

function toggleView(nextView) {
  for (const [key, node] of Object.entries(views)) {
    node.classList.toggle("hidden", key !== nextView);
  }
  document.body.classList.toggle("run-active", nextView === "run");
}

function updateReasons(result) {
  const reasons = [];
  if (result.reason_baseline) reasons.push(`Baseline: ${result.reason_baseline}`);
  if (result.reason_pipr6) reasons.push(`PIPR6: ${result.reason_pipr6}`);
  if (result.reason_pipr30) reasons.push(`PIPR30: ${result.reason_pipr30}`);

  if (reasons.length === 0) {
    metricReasons.classList.add("hidden");
    metricReasons.textContent = "";
    return;
  }

  metricReasons.textContent = reasons.join(" | ");
  metricReasons.classList.remove("hidden");
}

function updateEngagement(result) {
  const engagement = result.engagement || {};
  $("e-session").textContent = fmtScore(engagement.session_score, 2);
  $("e-ema").textContent = fmtScore(engagement.ema_score, 2);
  $("e-response").textContent = fmtScore(engagement.pupil_response_score, 2);
  $("e-recovery").textContent = fmtScore(engagement.recovery_score, 2);
  const stability = engagement.gaze_stability_score ?? engagement.attentiveness_score ?? engagement.concentration_score;
  $("e-gaze-stability").textContent = fmtScore(stability, 2);

  const reasons = [];
  if (engagement.reason_response) reasons.push(`Response: ${engagement.reason_response}`);
  if (engagement.reason_recovery) reasons.push(`Recovery: ${engagement.reason_recovery}`);
  const stabilityReason = engagement.reason_gaze_stability ?? engagement.reason_attentiveness ?? engagement.reason_concentration;
  if (stabilityReason) reasons.push(`Gaze stability proxy: ${stabilityReason}`);
  if (engagement.scientific_notes) reasons.push(engagement.scientific_notes);

  if (reasons.length === 0) {
    engagementReasons.classList.add("hidden");
    engagementReasons.textContent = "";
  } else {
    engagementReasons.textContent = reasons.join(" | ");
    engagementReasons.classList.remove("hidden");
  }
}

function showResults(result) {
  toggleView("results");
  document.body.classList.remove("phase-light", "phase-post", "run-active");
  if (typeof window.redrawNeuralBackground === "function") {
    window.redrawNeuralBackground();
  }

  $("r-baseline").textContent = fmt(result.baseline);

  const pipr6 = $("r-pipr6");
  pipr6.textContent = fmt(result.pipr_6);
  pipr6.className = result.pipr_6 == null ? "" : result.pipr_6 >= 0 ? "value-pos" : "value-neg";

  const pipr30 = $("r-pipr30");
  pipr30.textContent = fmt(result.pipr_30);
  pipr30.className = result.pipr_30 == null ? "" : result.pipr_30 >= 0 ? "value-pos" : "value-neg";

  $("r-n-base").textContent = result.n_base ?? "-";
  $("r-n-pipr6").textContent = result.n_pipr6 ?? "-";
  $("r-n-pipr30").textContent = result.n_pipr30 ?? "-";
  updateReasons(result);
  updateEngagement(result);
  drawGraph();
}

function updateTick(event) {
  $("st-elapsed").textContent = Number(event.elapsed).toFixed(2);
  $("st-pupil").textContent = event.pupil == null ? "-" : Number(event.pupil).toFixed(3);
  $("st-samples").textContent = event.samples;
  setPhase(event.phase);

  const pct = Math.min(100, (Number(event.elapsed) / totalDuration) * 100);
  progressBar.style.width = `${pct}%`;

  graphPoints.push({
    t: Number(event.elapsed),
    pupil: event.pupil == null ? null : Number(event.pupil),
    phase: event.phase,
  });
  if (graphPoints.length > 720) {
    graphPoints.splice(0, graphPoints.length - 720);
  }

  if (event.worn === false) {
    hideGazeCursor();
  } else {
    updateGazeCursor(event.gaze_x, event.gaze_y);
  }
}

async function waitForEvents() {
  let done = false;
  let connected = false;

  while (!done) {
    try {
      const resp = await fetch("/next-event");
      const event = await resp.json();

      if (event.type === "timeout") {
        continue;
      }

      if (event.type === "log") {
        log(event.msg);
      } else if (event.type === "phase") {
        if (!connected) {
          connected = true;
          stopPreconnectCamera();
        }
        setPhase(event.phase);
      } else if (event.type === "gaze") {
        if (!connected) {
          connected = true;
          stopPreconnectCamera();
        }
        if (event.worn === false) {
          hideGazeCursor();
        } else {
          updateGazeCursor(event.gaze_x, event.gaze_y);
        }
      } else if (event.type === "tick") {
        if (!connected) {
          connected = true;
          stopPreconnectCamera();
        }
        updateTick(event);
      } else if (event.type === "done") {
        if (!connected) {
          connected = true;
          stopPreconnectCamera();
        }
        progressBar.style.width = "100%";
        setTimeout(() => showResults(event.results), 350);
        done = true;
      }
    } catch (error) {
      log(`Event fetch error: ${error}. Retrying...`);
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
  }
}

async function startSession() {
  const config = {
    t_on: Number($("t_on").value),
    t_off: Number($("t_off").value),
    total_s: Number($("total_s").value),
    baseline_s: Number($("baseline_s").value),
    retries: Number($("retries").value),
    demo: $("demo-chk").checked,
  };

  totalDuration = config.total_s;
  toggleView("run");
  setPhase("BASELINE");
  progressBar.style.width = "0%";
  hideGazeCursor();
  graphPoints.length = 0;
  await startPreconnectCamera();

  try {
    const resp = await fetch("/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });

    if (!resp.ok) {
      const body = await resp.json();
      throw new Error(body.detail || "Failed to start session");
    }

    log("Connecting to tracker...");
    await waitForEvents();
  } catch (error) {
    stopPreconnectCamera();
    log(String(error));
    toggleView("start");
  }
}

$("start-btn").addEventListener("click", startSession);
$("restart-btn").addEventListener("click", () => window.location.reload());
window.addEventListener("resize", drawGraph);
graphCanvas?.addEventListener("mousemove", onResultGraphMove);
graphCanvas?.addEventListener("mouseleave", hideGraphTooltip);
