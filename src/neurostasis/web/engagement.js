const $ = (id) => document.getElementById(id);
const graphCanvas = $("engagement-graph");
const historyList = $("engagement-list");
const logEl = $("log");

function log(msg) {
  logEl.textContent = msg;
}

function fmt(value) {
  return value == null ? "-" : Number(value).toFixed(2);
}

function fmtScore(value) {
  return value == null ? "-" : `${Number(value).toFixed(2)}/100`;
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

  ctx.lineWidth = 2;
  ctx.strokeStyle = "#ffbf2f";
  ctx.beginPath();
  for (let i = 0; i < points.length; i += 1) {
    const [x, y] = toXY(i, Number(points[i].session_score));
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.strokeStyle = "#50e0b8";
  ctx.beginPath();
  for (let i = 0; i < points.length; i += 1) {
    const [x, y] = toXY(i, Number(points[i].ema_score));
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
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
    for (const rec of records.slice(-12).reverse()) {
      const row = document.createElement("article");
      row.className = "metric metric-dim";
      row.innerHTML = `<span>${rec.timestamp_utc ?? "-"}</span><span>session ${fmtScore(rec.session_score)} | ema ${fmtScore(rec.ema_score)}</span>`;
      historyList.appendChild(row);
    }
    log("Engagement history loaded.");
  } catch (error) {
    log(String(error));
  }
}

window.addEventListener("resize", () => loadHistory());
loadHistory();
