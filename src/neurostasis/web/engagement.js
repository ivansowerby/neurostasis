const $ = (id) => document.getElementById(id);
const graphCanvas = $("engagement-graph");
const historyList = $("engagement-list");
const logEl = $("log");
let trendTooltip = null;
let trendPlot = [];

function log(msg) {
  logEl.textContent = msg;
}

function fmt(value) {
  return value == null ? "-" : Number(value).toFixed(2);
}

function fmtScore(value) {
  return value == null ? "-" : `${Number(value).toFixed(2)}/100`;
}

function drawSmoothLine(ctx, points, color, width = 2) {
  if (!points || points.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(points[0][0], points[0][1]);

  for (let i = 1; i < points.length - 1; i += 1) {
    const xc = (points[i][0] + points[i + 1][0]) / 2;
    const yc = (points[i][1] + points[i + 1][1]) / 2;
    ctx.quadraticCurveTo(points[i][0], points[i][1], xc, yc);
  }

  const last = points[points.length - 1];
  ctx.lineTo(last[0], last[1]);
  ctx.stroke();
}

function drawEngagementGraph(records) {
  if (!graphCanvas) return;
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = graphCanvas.clientWidth || 320;
  const cssHeight = graphCanvas.clientHeight || 220;
  graphCanvas.width = Math.floor(cssWidth * dpr);
  graphCanvas.height = Math.floor(cssHeight * dpr);
  const ctx = graphCanvas.getContext("2d");
  if (!ctx) return;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const points = records.filter((r) => r.session_score != null && r.ema_score != null);
  if (points.length < 2) return;

  const xPad = 10;
  const yPad = 10;
  const plotW = cssWidth - 2 * xPad;
  const plotH = cssHeight - 2 * yPad;
  const maxIdx = points.length - 1;

  ctx.strokeStyle = "rgba(148, 160, 184, 0.35)";
  for (let i = 1; i < 4; i += 1) {
    const y = yPad + (plotH * i) / 4;
    ctx.beginPath();
    ctx.moveTo(xPad, y);
    ctx.lineTo(xPad + plotW, y);
    ctx.stroke();
  }

  const toXY = (i, score) => {
    const x = xPad + (i / maxIdx) * plotW;
    const y = yPad + (1 - score / 100) * plotH;
    return [x, y];
  };

  const sessionPts = [];
  const rendered = [];
  for (let i = 0; i < points.length; i += 1) {
    const xy = toXY(i, Number(points[i].session_score));
    sessionPts.push(xy);
    rendered.push({ x: xy[0], y: xy[1], series: "Session", value: Number(points[i].session_score), index: i + 1 });
  }
  drawSmoothLine(ctx, sessionPts, "#ffbf2f", 2);

  const emaPts = [];
  for (let i = 0; i < points.length; i += 1) {
    const xy = toXY(i, Number(points[i].ema_score));
    emaPts.push(xy);
    rendered.push({ x: xy[0], y: xy[1], series: "EMA", value: Number(points[i].ema_score), index: i + 1 });
  }
  drawSmoothLine(ctx, emaPts, "#50e0b8", 2);
  trendPlot = rendered;
}

function ensureTrendTooltip() {
  if (trendTooltip) return trendTooltip;
  trendTooltip = document.createElement("div");
  trendTooltip.className = "canvas-tooltip hidden";
  document.body.appendChild(trendTooltip);
  return trendTooltip;
}

function hideTrendTooltip() {
  if (!trendTooltip) return;
  trendTooltip.classList.add("hidden");
}

function onTrendMove(ev) {
  if (!graphCanvas || trendPlot.length === 0) {
    hideTrendTooltip();
    return;
  }
  const rect = graphCanvas.getBoundingClientRect();
  const x = ev.clientX - rect.left;
  const y = ev.clientY - rect.top;
  let best = null;
  let bestD = Infinity;
  for (const p of trendPlot) {
    const d = Math.hypot(p.x - x, p.y - y);
    if (d < bestD) {
      bestD = d;
      best = p;
    }
  }
  if (!best || bestD > 26) {
    hideTrendTooltip();
    return;
  }
  const tip = ensureTrendTooltip();
  tip.textContent = `${best.series} #${best.index}: ${fmtScore(best.value)}`;
  tip.classList.remove("hidden");
  tip.style.left = `${ev.clientX + 12}px`;
  tip.style.top = `${ev.clientY + 12}px`;
}

async function loadHistory() {
  try {
    const resp = await fetch("/api/engagement/history?limit=120");
    if (!resp.ok) {
      throw new Error("Failed to fetch engagement history");
    }
    const data = await resp.json();
    const records = data.records || [];
    const latest = data.latest || null;

    $("history-count").textContent = String(data.count ?? records.length);
    $("latest-session").textContent = fmtScore(latest?.session_score);
    $("latest-ema").textContent = fmtScore(latest?.ema_score);

    drawEngagementGraph(records);
    historyList.innerHTML = "";
    let rowOpacity = 100;
    for (const rec of records.slice(-12).reverse()) {
      const row = document.createElement("article");
      row.className = "metric metric-dim";
      row.style = `opacity: ${rowOpacity}%;`;
      row.innerHTML = `
        <span>${rec.timestamp_utc ?? "-"}</span>
        <span class="trend-row-scores">
          <span class="score-chip score-session">${rec.session_score.toFixed(2)}</span>
          <span class="score-chip score-ema">${rec.ema_score.toFixed(2)}</span>
        </span>
      `;
      historyList.appendChild(row);
      rowOpacity -= (100 / 12);
    }
    log("Engagement history loaded.");
  } catch (error) {
    log(String(error));
  }
}

window.addEventListener("resize", () => loadHistory());
graphCanvas?.addEventListener("mousemove", onTrendMove);
graphCanvas?.addEventListener("mouseleave", hideTrendTooltip);
loadHistory();
