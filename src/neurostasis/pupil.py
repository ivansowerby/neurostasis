"""
neurostasis.pupil  –  PIPR estimation + embedded web UI at http://localhost:8000

Architecture
============
- A ThreadingHTTPServer on :8000 serves the HTML/JS page and a long-poll API.
- When the browser clicks START, JS posts config to /start; Python launches
  the acquisition loop in a background thread.
- JS then enters a sequential event loop: it calls GET /next-event, which
  blocks server-side until the acquisition thread emits the next event, then
  returns immediately.  JS processes that event (updates the visual stimulus
  / stats), then calls /next-event again – and so on until type=="done".
- The timer and all phase logic live entirely in Python; the browser merely
  reacts to what Python tells it.

API
===
  GET  /                    → HTML UI
  POST /start               → { t_on, t_off, total_s, baseline_s, retries, demo }
  GET  /next-event          → next event JSON (blocks until available, ≤120 s)
  GET  /status              → latest tick snapshot (non-blocking)
  GET  /results             → final PIPR metrics (404 if not yet done)
"""
from __future__ import annotations

import json
import math
import queue
import statistics
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

# ---------------------------------------------------------------------------
# Device config
# ---------------------------------------------------------------------------
IP_ADDRESS = "172.20.10.3"
PORT       = "8080"

# ---------------------------------------------------------------------------
# Shared state (acquisition thread ↔ HTTP handlers)
# ---------------------------------------------------------------------------
_subscribers:      list[queue.Queue] = []
_subscribers_lock: threading.Lock   = threading.Lock()

_snapshot:      dict = {"phase": "idle", "elapsed": 0.0, "pupil": None, "samples": 0}
_snapshot_lock: threading.Lock = threading.Lock()

_results:             Optional[dict]            = None
_acquisition_thread:  Optional[threading.Thread] = None

# ---------------------------------------------------------------------------
# Event bus
# ---------------------------------------------------------------------------

def _broadcast(event: dict) -> None:
    with _subscribers_lock:
        for q in list(_subscribers):
            q.put(event)


def _subscribe() -> queue.Queue:
    q: queue.Queue = queue.Queue()
    with _subscribers_lock:
        _subscribers.append(q)
    return q


def _unsubscribe(q: queue.Queue) -> None:
    with _subscribers_lock:
        try:
            _subscribers.remove(q)
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# Pupil helpers
# ---------------------------------------------------------------------------

def _pick_pupil(pl: Optional[float], pr: Optional[float]) -> Optional[float]:
    if pl is None and pr is None:
        return None
    if pl is None:
        return pr
    if pr is None:
        return pl
    return (pl + pr) / 2.0


def _mean(xs: list) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


# ---------------------------------------------------------------------------
# Simulated pupil data (demo mode – no real device required)
# ---------------------------------------------------------------------------

def _simulated_pupil(elapsed: float, t_on: float, t_off: float) -> float:
    """Physiological caricature: ~5 mm baseline, constricts to ~3 mm during
    light, then recovers with a sustained PIPR offset."""
    base = 5.0 + 0.25 * math.sin(elapsed * 0.35)          # slow baseline drift
    if elapsed < t_on:
        return base
    if elapsed < t_off:
        frac = min(1.0, (elapsed - t_on) / 1.5)
        return base - frac * 2.0 + 0.12 * math.sin(elapsed * 3.1)
    # Post-light: exponential recovery with a sustained melanopsin residual
    tau = (elapsed - t_off) / 28.0
    residual = 1.1 * math.exp(-tau)
    return base - residual + 0.08 * math.sin(elapsed * 2.3)


# ---------------------------------------------------------------------------
# Acquisition / stimulus thread
# ---------------------------------------------------------------------------

def _run_acquisition(cfg: dict, demo: bool) -> None:
    global _results

    t_on       = float(cfg.get("t_on",       5.0))
    t_off      = float(cfg.get("t_off",      15.0))
    total_s    = float(cfg.get("total_s",    55.0))
    baseline_s = float(cfg.get("baseline_s",  2.0))
    retries    = int(cfg.get("retries",        5))

    # -- Connect (or fall back to demo) -------------------------------------
    device = None
    if not demo:
        for attempt in range(1, retries + 1):
            _broadcast({"type": "log",
                        "msg": f"Connection attempt {attempt}/{retries}…"})
            try:
                from pupil_labs.realtime_api.simple import Device  # type: ignore
                device = Device(IP_ADDRESS, PORT)
                _broadcast({"type": "log", "msg": "Connected to Pupil Labs device."})
                break
            except Exception as exc:
                _broadcast({"type": "log",
                            "msg": f"Attempt {attempt} failed: {exc}"})
                device = None
                time.sleep(0.3)
        else:
            _broadcast({"type": "log",
                        "msg": "Could not connect – switching to DEMO mode."})
            demo = True

    if demo:
        _broadcast({"type": "log",
                    "msg": "Running in DEMO MODE (simulated pupil data)."})

    # -- Stream -------------------------------------------------------------
    _broadcast({"type": "phase", "phase": "BASELINE", "elapsed": 0.0})

    t0            = time.time()
    samples:      list[tuple[float, Optional[float]]] = []
    last_tick     = -1.0
    current_phase = "BASELINE"

    try:
        while True:
            if demo:
                elapsed = time.time() - t0
                pupil: Optional[float] = _simulated_pupil(elapsed, t_on, t_off)
                t = t0 + elapsed
                time.sleep(0.033)   # ~30 Hz simulated
            else:
                gaze    = device.receive_gaze_datum()  # type: ignore[union-attr]
                t       = getattr(gaze, "timestamp_unix_seconds", None) or time.time()
                pl      = getattr(gaze, "pupil_diameter_left",  None)
                pr      = getattr(gaze, "pupil_diameter_right", None)
                pupil   = _pick_pupil(pl, pr)
                elapsed = time.time() - t0

            samples.append((t, pupil))

            # Phase transition detection
            if elapsed >= t_off:
                new_phase = "POST_LIGHT"
            elif elapsed >= t_on:
                new_phase = "LIGHT_ON"
            else:
                new_phase = "BASELINE"

            if new_phase != current_phase:
                current_phase = new_phase
                _broadcast({"type": "phase",
                            "phase": current_phase,
                            "elapsed": round(elapsed, 2)})

            # Per-second tick
            if elapsed - last_tick >= 1.0:
                tick = {
                    "type":    "tick",
                    "phase":   current_phase,
                    "elapsed": round(elapsed, 2),
                    "pupil":   round(pupil, 4) if pupil is not None else None,
                    "samples": len(samples),
                }
                with _snapshot_lock:
                    _snapshot.update(tick)
                _broadcast(tick)
                last_tick = elapsed

            if elapsed >= total_s:
                break

    finally:
        if device is not None:
            try:
                device.close()  # type: ignore[union-attr]
            except Exception:
                pass

    # -- Compute windowed metrics -------------------------------------------
    def in_window(ts: float, start: float, end: float) -> bool:
        return start <= ts <= end

    base_start   = t0 + (t_on - baseline_s)
    base_end     = t0 + t_on
    pipr6_start  = t0 + t_off +  5.0
    pipr6_end    = t0 + t_off +  7.0
    pipr30_start = t0 + t_off + 25.0
    pipr30_end   = t0 + t_off + 35.0

    base_vals   = [p for (ts, p) in samples if in_window(ts, base_start,   base_end)]
    pipr6_vals  = [p for (ts, p) in samples if in_window(ts, pipr6_start,  pipr6_end)]
    pipr30_vals = [p for (ts, p) in samples if in_window(ts, pipr30_start, pipr30_end)]

    baseline   = _mean(base_vals)
    pipr6_mean = _mean(pipr6_vals)
    pipr30_mean = _mean(pipr30_vals)

    pipr_6  = (baseline - pipr6_mean)  if (baseline is not None and pipr6_mean  is not None) else None
    pipr_30 = (baseline - pipr30_mean) if (baseline is not None and pipr30_mean is not None) else None

    _results = {
        "baseline":  baseline,
        "pipr_6":    pipr_6,
        "pipr_30":   pipr_30,
        "n_base":    len(base_vals),
        "n_pipr6":   len(pipr6_vals),
        "n_pipr30":  len(pipr30_vals),
    }
    _broadcast({"type": "done", "results": _results})


# ---------------------------------------------------------------------------
# HTML / JS (inline – no external files needed)
# ---------------------------------------------------------------------------
_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Neurostasis – PIPR</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #000;
    color: #d0d0d0;
    font-family: 'Courier New', monospace;
    height: 100vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    transition: background 0.35s ease, color 0.35s ease;
  }
  body.light-on { background: #f8f8f0; color: #111; }

  /* ── Start panel ── */
  #start-panel { display: flex; flex-direction: column; gap: 11px; width: min(400px, 90vw); }
  #start-panel h1 { font-size: 1.45rem; letter-spacing: .12em; margin-bottom: 6px; }

  .field { display: flex; justify-content: space-between; align-items: center; gap: 10px; }
  .field label { flex: 1; font-size: .82rem; opacity: .65; }
  .field input[type=number] {
    width: 82px; background: #111; border: 1px solid #3a3a3a; color: #d0d0d0;
    padding: 4px 8px; font-family: inherit; font-size: .9rem;
    text-align: right; border-radius: 3px;
  }
  #demo-row { display: flex; justify-content: flex-end; align-items: center;
              gap: 8px; font-size: .82rem; opacity: .75; }

  #start-btn {
    margin-top: 6px; padding: 10px 0;
    background: #1a5c35; color: #eee; border: none; border-radius: 4px;
    font-family: inherit; font-size: 1rem; letter-spacing: .08em; cursor: pointer;
  }
  #start-btn:hover { background: #217a45; }

  /* ── Run panel ── */
  #run-panel { display: none; text-align: center; }

  #stimulus {
    width: min(300px, 72vw); height: min(300px, 72vw);
    border-radius: 50%;
    background: #0d0d0d; border: 2px solid #1e1e1e;
    margin: 0 auto 26px;
    transition: background .25s ease, box-shadow .25s ease;
  }
  #stimulus.active {
    background: #ffffff;
    box-shadow: 0 0 90px 50px rgba(255,255,220,.55);
  }

  #phase-label {
    font-size: 1.35rem; letter-spacing: .22em; margin-bottom: 14px; min-height: 2rem;
    transition: color .3s ease;
  }
  #stats { font-size: .78rem; opacity: .55; line-height: 2.1; }

  /* ── Progress bar ── */
  #progress-wrap {
    width: min(300px, 72vw); height: 3px; background: #1a1a1a;
    border-radius: 2px; margin: 18px auto 0;
  }
  #progress-bar { height: 100%; width: 0%; background: #2a7a50; border-radius: 2px;
                  transition: width .9s linear; }

  /* ── Log strip ── */
  #log {
    position: fixed; bottom: 10px; left: 12px; right: 12px;
    font-size: .68rem; opacity: .38; white-space: nowrap;
    overflow: hidden; text-overflow: ellipsis; text-align: left;
  }

  /* ── Results panel ── */
  #results-panel { display: none; flex-direction: column; width: min(400px, 90vw); text-align: center; }
  #results-panel h2 { font-size: 1.25rem; letter-spacing: .14em; margin-bottom: 18px; }

  .metric {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 14px; border: 1px solid #222; border-radius: 4px; margin-bottom: 7px;
    font-size: .9rem;
  }
  .metric.dim { border-color: #111; font-size: .73rem; opacity: .45; }
  .val { font-weight: bold; letter-spacing: .04em; }
  .val.pos { color: #4cc46c; }
  .val.neg { color: #d95e5e; }

  #restart-btn {
    margin-top: 18px; padding: 8px 22px;
    background: #1e1e1e; color: #aaa; border: none; border-radius: 4px;
    font-family: inherit; cursor: pointer;
  }
  #restart-btn:hover { background: #2a2a2a; }
</style>
</head>
<body>

<!-- ── Start ── -->
<div id="start-panel">
  <h1>PIPR PROTOCOL</h1>
  <div class="field">
    <label>Light ON  (s from start)</label>
    <input id="t_on"       type="number" value="5"  min="1"  step="1">
  </div>
  <div class="field">
    <label>Light OFF (s from start)</label>
    <input id="t_off"      type="number" value="15" min="2"  step="1">
  </div>
  <div class="field">
    <label>Total duration (s)</label>
    <input id="total_s"    type="number" value="55" min="10" step="5">
  </div>
  <div class="field">
    <label>Baseline window (s)</label>
    <input id="baseline_s" type="number" value="2"  min="1"  step="1">
  </div>
  <div class="field">
    <label>Device retries</label>
    <input id="retries"    type="number" value="5"  min="1"  step="1">
  </div>
  <div id="demo-row">
    <label for="demo-chk">Demo mode (simulated pupil)</label>
    <input id="demo-chk" type="checkbox" checked>
  </div>
  <button id="start-btn" onclick="startSession()">&#9654;&#xFE0E;  START SESSION</button>
</div>

<!-- ── Running ── -->
<div id="run-panel">
  <div id="stimulus"></div>
  <div id="phase-label">–</div>
  <div id="stats">
    elapsed: <span id="st-elapsed">0.00</span> s &nbsp;|&nbsp;
    pupil: <span id="st-pupil">–</span> &nbsp;|&nbsp;
    samples: <span id="st-samples">0</span>
  </div>
  <div id="progress-wrap"><div id="progress-bar"></div></div>
</div>

<!-- ── Results ── -->
<div id="results-panel">
  <h2>PIPR RESULTS</h2>
  <div class="metric">
    <span>Baseline pupil (mean)</span>
    <span class="val" id="r-baseline">–</span>
  </div>
  <div class="metric">
    <span>PIPR&#x2086; &nbsp;(5–7 s post-light)</span>
    <span class="val" id="r-pipr6">–</span>
  </div>
  <div class="metric">
    <span>PIPR&#x2083;&#x2080; (25–35 s post-light)</span>
    <span class="val" id="r-pipr30">–</span>
  </div>
  <div class="metric dim">
    <span>Baseline samples</span><span id="r-n-base">–</span>
  </div>
  <div class="metric dim">
    <span>PIPR&#x2086; samples</span><span id="r-n-pipr6">–</span>
  </div>
  <div class="metric dim">
    <span>PIPR&#x2083;&#x2080; samples</span><span id="r-n-pipr30">–</span>
  </div>
  <button id="restart-btn" onclick="location.reload()">&#8635;  NEW SESSION</button>
</div>

<div id="log"></div>

<script>
const $ = id => document.getElementById(id);
let totalDuration = 55;

function log(msg) { $('log').textContent = msg; }

function setPhase(phase) {
  const label = $('phase-label');
  const stim  = $('stimulus');
  document.body.classList.remove('light-on');
  stim.classList.remove('active');
  switch (phase) {
    case 'BASELINE':
      label.textContent = 'BASELINE';
      label.style.color = '#777';
      break;
    case 'LIGHT_ON':
      label.textContent = 'LIGHT  ON';
      label.style.color = '#e8e8cc';
      document.body.classList.add('light-on');
      stim.classList.add('active');
      break;
    case 'POST_LIGHT':
      label.textContent = 'POST-LIGHT';
      label.style.color = '#4cc46c';
      break;
    default:
      label.textContent = phase;
  }
}

function fmt(v) {
  return (v == null) ? '–' : v.toFixed(4);
}

function showResults(res) {
  $('start-panel').style.display   = 'none';
  $('run-panel').style.display     = 'none';
  $('results-panel').style.display = 'flex';
  document.body.style.background   = '';
  document.body.classList.remove('light-on');

  $('r-baseline').textContent = fmt(res.baseline);
  $('r-baseline').className   = 'val';

  ['pipr6', 'pipr30'].forEach(key => {
    const v  = res['pipr_' + (key === 'pipr6' ? '6' : '30')];
    const el = $('r-' + key);
    el.textContent = fmt(v);
    el.className   = 'val' + (v == null ? '' : v >= 0 ? ' pos' : ' neg');
  });

  $('r-n-base').textContent  = res.n_base  ?? '–';
  $('r-n-pipr6').textContent = res.n_pipr6 ?? '–';
  $('r-n-pipr30').textContent = res.n_pipr30 ?? '–';
}

function handleEvent(event) {
  if (event.type === 'log') {
    log(event.msg);

  } else if (event.type === 'phase') {
    setPhase(event.phase);

  } else if (event.type === 'tick') {
    $('st-elapsed').textContent  = event.elapsed.toFixed(2);
    $('st-pupil').textContent    = event.pupil != null ? event.pupil.toFixed(3) : '–';
    $('st-samples').textContent  = event.samples;
    // keep phase label in sync with ticks too
    setPhase(event.phase);
    // update progress bar
    const pct = Math.min(100, (event.elapsed / totalDuration) * 100);
    $('progress-bar').style.width = pct + '%';

  } else if (event.type === 'done') {
    $('progress-bar').style.width = '100%';
    setTimeout(() => showResults(event.results), 600);
  }
}

async function startSession() {
  const cfg = {
    t_on:       parseFloat($('t_on').value),
    t_off:      parseFloat($('t_off').value),
    total_s:    parseFloat($('total_s').value),
    baseline_s: parseFloat($('baseline_s').value),
    retries:    parseInt($('retries').value),
    demo:       $('demo-chk').checked,
  };
  totalDuration = cfg.total_s;

  $('start-panel').style.display = 'none';
  $('run-panel').style.display   = 'block';
  setPhase('BASELINE');

  // Tell Python to begin acquisition
  await fetch('/start', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(cfg),
  });

  // Sequential event loop – each call blocks until Python emits the next event
  let done = false;
  while (!done) {
    let event;
    try {
      const resp = await fetch('/next-event');
      event = await resp.json();
    } catch (err) {
      log('fetch error: ' + err + ' – retrying…');
      await new Promise(r => setTimeout(r, 500));
      continue;
    }

    if (event.type === 'timeout') {
      // Server-side 120-s guard – just keep waiting
      continue;
    }

    handleEvent(event);

    if (event.type === 'done') done = true;
  }
}
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):   # silence request log
        pass

    # -- helpers -------------------------------------------------------------

    def _send(self, code: int, ctype: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: dict, code: int = 200) -> None:
        self._send(code, "application/json", json.dumps(obj).encode())

    # -- GET -----------------------------------------------------------------

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/":
            self._send(200, "text/html; charset=utf-8", _HTML.encode())

        elif path == "/next-event":
            # Block until the acquisition thread emits the next event (≤120 s).
            q = _subscribe()
            try:
                event = q.get(timeout=120)
                self._json(event)
            except queue.Empty:
                self._json({"type": "timeout"})
            finally:
                _unsubscribe(q)

        elif path == "/status":
            with _snapshot_lock:
                self._json(dict(_snapshot))

        elif path == "/results":
            if _results is not None:
                self._json(_results)
            else:
                self._json({"error": "not ready"}, 404)

        else:
            self._send(404, "text/plain", b"Not found")

    # -- POST ----------------------------------------------------------------

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path

        if path == "/start":
            length = int(self.headers.get("Content-Length", 0))
            body   = self.rfile.read(length)
            try:
                cfg = json.loads(body)
            except json.JSONDecodeError:
                self._json({"error": "bad json"}, 400)
                return

            demo = bool(cfg.pop("demo", False))

            global _acquisition_thread, _results
            if _acquisition_thread is None or not _acquisition_thread.is_alive():
                _results = None
                _acquisition_thread = threading.Thread(
                    target=_run_acquisition,
                    args=(cfg, demo),
                    daemon=True,
                )
                _acquisition_thread.start()
                self._json({"status": "started"})
            else:
                self._json({"status": "already running"})

        else:
            self._send(404, "text/plain", b"Not found")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host, port = "127.0.0.1", 8000
    server = ThreadingHTTPServer((host, port), _Handler)
    url = f"http://{host}:{port}/"
    print(f"PIPR server →  {url}")
    print("Press Ctrl-C to stop.\n")
    try:
        webbrowser.open(url)
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()
