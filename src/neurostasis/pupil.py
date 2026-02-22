"""FastAPI-backed PIPR acquisition server with static frontend assets."""
from __future__ import annotations

import math
import queue
import statistics
import threading
import time
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .eeg import EEGConfig, EEGRunner
from .eeg.attention import AttentionState, latest_attention_states
from .engagement import register_engagement_routes
from .engagement_store import append_engagement_record

IP_ADDRESS = "172.20.10.3"
PORT = "8080"
WEB_DIR = Path(__file__).resolve().parent / "web"


class StartRequest(BaseModel):
    t_on: float = Field(default=15.0, ge=0.0)
    t_off: float = Field(default=25.0, ge=0.0)
    total_s: float = Field(default=60.0, gt=0.0)
    baseline_s: float = Field(default=10.0, gt=0.0)
    retries: int = Field(default=5, ge=1)
    demo: bool = False


_subscribers: list[queue.Queue] = []
_subscribers_lock = threading.Lock()

_snapshot = {"phase": "idle", "elapsed": 0.0, "pupil": None, "samples": 0, "gaze_x": None, "gaze_y": None, "worn": None}
_snapshot_lock = threading.Lock()

_results: Optional[dict] = None
_acquisition_thread: Optional[threading.Thread] = None
_run_lock = threading.Lock()


def _broadcast(event: dict) -> None:
    with _subscribers_lock:
        for subscriber in list(_subscribers):
            subscriber.put(event)


def _subscribe() -> queue.Queue:
    subscriber: queue.Queue = queue.Queue()
    with _subscribers_lock:
        _subscribers.append(subscriber)
    return subscriber


def _unsubscribe(subscriber: queue.Queue) -> None:
    with _subscribers_lock:
        try:
            _subscribers.remove(subscriber)
        except ValueError:
            pass


def _pick_pupil(pl: Optional[float], pr: Optional[float]) -> Optional[float]:
    if pl is None and pr is None:
        return None
    if pl is None:
        return pr
    if pr is None:
        return pl
    return (pl + pr) / 2.0


def _mean(xs: list[Optional[float]]) -> Optional[float]:
    filtered = [x for x in xs if x is not None]
    return statistics.fmean(filtered) if filtered else None


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _normalize_sequence(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if all(0.0 <= x <= 1.0 for x in xs) and all(0.0 <= y <= 1.0 for y in ys):
        return points

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(1e-6, max_x - min_x)
    span_y = max(1e-6, max_y - min_y)
    return [((x - min_x) / span_x, (y - min_y) / span_y) for (x, y) in points]


def _simulated_pupil(elapsed: float, t_on: float, t_off: float) -> float:
    base = 5.0 + 0.25 * math.sin(elapsed * 0.35)
    if elapsed < t_on:
        return base
    if elapsed < t_off:
        frac = min(1.0, (elapsed - t_on) / 1.5)
        return base - frac * 2.0 + 0.12 * math.sin(elapsed * 3.1)
    tau = (elapsed - t_off) / 28.0
    residual = 1.1 * math.exp(-tau)
    return base - residual + 0.08 * math.sin(elapsed * 2.3)


def _simulated_gaze(elapsed: float) -> tuple[float, float]:
    x = 0.5 + 0.18 * math.sin(elapsed * 0.9) + 0.03 * math.sin(elapsed * 2.7)
    y = 0.5 + 0.12 * math.cos(elapsed * 0.7) + 0.02 * math.sin(elapsed * 2.2)
    return (max(0.0, min(1.0, x)), max(0.0, min(1.0, y)))


def _run_acquisition(cfg: StartRequest) -> None:
    global _results

    t_on = cfg.t_on
    t_off = cfg.t_off
    total_s = cfg.total_s
    baseline_s = cfg.baseline_s
    retries = cfg.retries
    demo = cfg.demo

    device = None
    if not demo:
        for attempt in range(1, retries + 1):
            _broadcast({"type": "log", "msg": f"Connection attempt {attempt}/{retries}..."})
            try:
                from pupil_labs.realtime_api.simple import Device  # type: ignore

                device = Device(IP_ADDRESS, PORT)
                _broadcast({"type": "log", "msg": "Connected to Pupil Labs device."})
                break
            except Exception as exc:
                _broadcast({"type": "log", "msg": f"Attempt {attempt} failed: {exc}"})
                device = None
                time.sleep(0.3)
        else:
            _broadcast({"type": "log", "msg": "Could not connect. Switching to DEMO mode."})
            demo = True

    if demo:
        _broadcast({"type": "log", "msg": "Running in DEMO mode (simulated pupil data)."})

    _broadcast({"type": "phase", "phase": "BASELINE", "elapsed": 0.0})
    t0 = time.time()
    samples: list[tuple[float, Optional[float]]] = []
    gaze_trace: list[tuple[float, float]] = []
    eeg_trace: list[AttentionState] = []
    last_tick = -1.0
    last_gaze_emit = -1.0
    last_eeg_poll = -1.0
    last_eeg_log = -1.0
    current_phase = "BASELINE"
    latest_eeg: Optional[AttentionState] = None
    eeg_runner: Optional[EEGRunner] = None
    eeg_thread: Optional[threading.Thread] = None
    eeg_error: Optional[str] = None
    eeg_status = "not_started"

    def _run_eeg() -> None:
        nonlocal eeg_error
        if eeg_runner is None:
            return
        try:
            eeg_runner.run()
        except Exception as exc:
            eeg_error = str(exc)
            _broadcast({"type": "log", "msg": f"EEG stopped: {eeg_error}"})

    try:
        eeg_runner = EEGRunner(EEGConfig(enable_ui=False))
        eeg_thread = threading.Thread(target=_run_eeg, daemon=True)
        eeg_thread.start()
        eeg_status = "running"
        _broadcast({"type": "log", "msg": "EEG attention stream started."})
    except Exception as exc:
        eeg_runner = None
        eeg_thread = None
        eeg_error = str(exc)
        eeg_status = "unavailable"
        _broadcast({"type": "log", "msg": f"EEG unavailable: {eeg_error}"})

    try:
        while True:
            if demo:
                elapsed = time.time() - t0
                pupil: Optional[float] = _simulated_pupil(elapsed, t_on, t_off)
                gaze_x, gaze_y = _simulated_gaze(elapsed)
                worn = True
                timestamp = t0 + elapsed
                time.sleep(0.033)
            else:
                gaze = device.receive_gaze_datum()  # type: ignore[union-attr]
                timestamp = getattr(gaze, "timestamp_unix_seconds", None) or time.time()
                pl = getattr(gaze, "pupil_diameter_left", None)
                pr = getattr(gaze, "pupil_diameter_right", None)
                pupil = _pick_pupil(pl, pr)
                gaze_x = getattr(gaze, "x", None)
                gaze_y = getattr(gaze, "y", None)
                worn = getattr(gaze, "worn", None)
                elapsed = time.time() - t0

            samples.append((timestamp, pupil))
            if worn is not False and gaze_x is not None and gaze_y is not None:
                gaze_trace.append((float(gaze_x), float(gaze_y)))

            if elapsed >= t_off:
                new_phase = "POST_LIGHT"
            elif elapsed >= t_on:
                new_phase = "LIGHT_ON"
            else:
                new_phase = "BASELINE"

            if new_phase != current_phase:
                current_phase = new_phase
                _broadcast({"type": "phase", "phase": current_phase, "elapsed": round(elapsed, 2)})

            if elapsed - last_gaze_emit >= 0.05:
                gaze_event = {
                    "type": "gaze",
                    "phase": current_phase,
                    "elapsed": round(elapsed, 3),
                    "gaze_x": gaze_x,
                    "gaze_y": gaze_y,
                    "worn": worn,
                }
                with _snapshot_lock:
                    _snapshot.update({"gaze_x": gaze_x, "gaze_y": gaze_y, "worn": worn})
                _broadcast(gaze_event)
                last_gaze_emit = elapsed

            if eeg_runner is not None and elapsed - last_eeg_poll >= 0.05:
                try:
                    eeg_states = latest_attention_states(eeg_runner, count=1)
                except Exception as exc:
                    eeg_states = []
                    if eeg_error is None:
                        eeg_error = str(exc)
                        _broadcast({"type": "log", "msg": f"EEG polling error: {eeg_error}"})
                if eeg_states:
                    latest_eeg = eeg_states[0]
                    eeg_trace.append(latest_eeg)
                    with _snapshot_lock:
                        _snapshot.update(
                            {
                                "eeg_concentration_score": round(latest_eeg.concentration_score, 3),
                                "eeg_alpha_theta_ratio": round(latest_eeg.alpha_theta_ratio, 6),
                                "eeg_status": eeg_status,
                            }
                        )
                    if elapsed - last_eeg_log >= 1.0:
                        _broadcast(
                            {
                                "type": "log",
                                "msg": (
                                    "EEG attention "
                                    f"{latest_eeg.concentration_score:.1f}/100 | "
                                    f"alpha/theta {latest_eeg.alpha_theta_ratio:.3f}"
                                ),
                            }
                        )
                        last_eeg_log = elapsed
                last_eeg_poll = elapsed

            if elapsed - last_tick >= 1.0:
                tick = {
                    "type": "tick",
                    "phase": current_phase,
                    "elapsed": round(elapsed, 2),
                    "pupil": round(pupil, 4) if pupil is not None else None,
                    "samples": len(samples),
                    "gaze_x": gaze_x,
                    "gaze_y": gaze_y,
                    "worn": worn,
                    "eeg_concentration_score": (
                        None if latest_eeg is None else round(latest_eeg.concentration_score, 3)
                    ),
                    "eeg_alpha_theta_ratio": (
                        None if latest_eeg is None else round(latest_eeg.alpha_theta_ratio, 6)
                    ),
                    "eeg_status": eeg_status,
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
        if eeg_runner is not None:
            try:
                eeg_runner.stop()
            except Exception:
                pass
        eeg_status = "stopped"
        with _snapshot_lock:
            _snapshot.update({"eeg_status": eeg_status})
        if eeg_thread is not None and eeg_thread.is_alive():
            eeg_thread.join(timeout=2.0)
        _broadcast({"type": "log", "msg": "EEG attention stream stopped."})

    def in_window(ts: float, start: float, end: float) -> bool:
        return start <= ts <= end

    base_start = t0 + (t_on - baseline_s)
    base_end = t0 + t_on
    pipr6_start = t0 + t_off + 5.0
    pipr6_end = t0 + t_off + 7.0
    pipr30_start = t0 + t_off + 25.0
    pipr30_end = t0 + t_off + 35.0

    base_vals = [p for (ts, p) in samples if in_window(ts, base_start, base_end)]
    pipr6_vals = [p for (ts, p) in samples if in_window(ts, pipr6_start, pipr6_end)]
    pipr30_vals = [p for (ts, p) in samples if in_window(ts, pipr30_start, pipr30_end)]

    baseline = _mean(base_vals)
    pipr6_mean = _mean(pipr6_vals)
    pipr30_mean = _mean(pipr30_vals)
    light_vals = [p for (ts, p) in samples if in_window(ts, t0 + t_on, t0 + t_off)]
    light_min = min((p for p in light_vals if p is not None), default=None)

    pipr_6 = (baseline - pipr6_mean) if (baseline is not None and pipr6_mean is not None) else None
    pipr_30 = (baseline - pipr30_mean) if (baseline is not None and pipr30_mean is not None) else None

    reason_baseline: Optional[str] = None
    reason_pipr6: Optional[str] = None
    reason_pipr30: Optional[str] = None
    if baseline is None:
        reason_baseline = "No valid pupil samples in baseline window."
    if pipr6_mean is None:
        reason_pipr6 = "No valid pupil samples in 5-7s post-light window."
    elif baseline is None:
        reason_pipr6 = "Baseline unavailable, so PIPR6 cannot be computed."
    if pipr30_mean is None:
        reason_pipr30 = "No valid pupil samples in 25-35s post-light window."
    elif baseline is None:
        reason_pipr30 = "Baseline unavailable, so PIPR30 cannot be computed."

    attentiveness_score: Optional[float] = None
    attentiveness_reason: Optional[str] = None
    gaze_jitter_rms: Optional[float] = None
    norm_trace = _normalize_sequence(gaze_trace)
    if len(norm_trace) < 6:
        attentiveness_reason = "Insufficient gaze samples to estimate jitter-based gaze stability."
    else:
        deltas = []
        for i in range(1, len(norm_trace)):
            dx = norm_trace[i][0] - norm_trace[i - 1][0]
            dy = norm_trace[i][1] - norm_trace[i - 1][1]
            deltas.append(math.sqrt(dx * dx + dy * dy))
        if deltas:
            gaze_jitter_rms = math.sqrt(statistics.fmean([d * d for d in deltas]))
            attentiveness_score = 100.0 * (1.0 - _clamp(gaze_jitter_rms / 0.08, 0.0, 1.0))
        else:
            attentiveness_reason = "Gaze jitter could not be computed."

    pupil_response_score: Optional[float] = None
    response_reason: Optional[str] = None
    if baseline is None or light_min is None:
        response_reason = "Missing baseline or light-phase pupil data for response scoring."
    else:
        constriction_amplitude = max(0.0, baseline - light_min)
        pupil_response_score = 100.0 * _clamp((constriction_amplitude - 0.2) / 2.0, 0.0, 1.0)
        if constriction_amplitude < 0.05:
            response_reason = (
                "Minimal constriction detected during LIGHT_ON. "
                "Stimulus intensity/timing may be insufficient or inverted."
            )

    recovery_score: Optional[float] = None
    recovery_reason: Optional[str] = None
    if pipr_30 is None:
        recovery_reason = "PIPR30 unavailable; sustained post-light recovery cannot be scored."
    else:
        recovery_score = 100.0 * _clamp(pipr_30 / 1.2, 0.0, 1.0)

    weighted_parts = []
    if pupil_response_score is not None:
        weighted_parts.append((0.65, pupil_response_score))
    if recovery_score is not None:
        weighted_parts.append((0.25, recovery_score))
    if attentiveness_score is not None:
        weighted_parts.append((0.10, attentiveness_score))

    if weighted_parts:
        weight_sum = sum(w for (w, _) in weighted_parts)
        session_score = sum(w * s for (w, s) in weighted_parts) / weight_sum
    else:
        session_score = 0.0

    eeg_concentration_score: Optional[float] = None
    eeg_alpha_theta_ratio: Optional[float] = None
    reason_eeg: Optional[str] = None
    retake_recommended = False
    if latest_eeg is not None:
        eeg_concentration_score = float(latest_eeg.concentration_score)
        eeg_alpha_theta_ratio = float(latest_eeg.alpha_theta_ratio)
        eeg_norm = _clamp(eeg_concentration_score / 100.0, 0.0, 1.0)
        gaze_norm = _clamp((attentiveness_score or 0.0) / 100.0, 0.0, 1.0)
        focus_match = eeg_norm * gaze_norm
        focus_mismatch = eeg_norm * (1.0 - gaze_norm)

        direction_components = []
        if pupil_response_score is not None:
            direction_components.append(_clamp((pupil_response_score - 50.0) / 50.0, -1.0, 1.0))
        if recovery_score is not None:
            direction_components.append(_clamp((recovery_score - 50.0) / 50.0, -1.0, 1.0))
        direction_signal = statistics.fmean(direction_components) if direction_components else 0.0

        boost = max(0.0, direction_signal) * (0.12 + 0.28 * focus_match)
        penalty = max(0.0, -direction_signal) * (0.12 + 0.28 * focus_match) + 0.36 * focus_mismatch
        adjustment_factor = _clamp(1.0 + boost - penalty, 0.6, 1.35)
        session_score = 100.0 * _clamp((session_score / 100.0) * adjustment_factor, 0.0, 1.0)

        if focus_mismatch > 0.45:
            retake_recommended = True
            reason_eeg = (
                "EEG attention was high while gaze was weakly aligned to the target, "
                "so the score was reduced and a lenient retake is recommended."
            )
        else:
            reason_eeg = "EEG and gaze were reasonably aligned; EEG provided a modest directional adjustment."
    elif eeg_error is not None:
        reason_eeg = f"EEG not used for this run: {eeg_error}"

    stored = append_engagement_record(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "session_score": round(session_score, 3),
            "pupil_response_score": None if pupil_response_score is None else round(pupil_response_score, 3),
            "recovery_score": None if recovery_score is None else round(recovery_score, 3),
            "attentiveness_score": None if attentiveness_score is None else round(attentiveness_score, 3),
            "eeg_concentration_score": None if eeg_concentration_score is None else round(eeg_concentration_score, 3),
            "eeg_alpha_theta_ratio": None if eeg_alpha_theta_ratio is None else round(eeg_alpha_theta_ratio, 6),
            "gaze_jitter_rms": None if gaze_jitter_rms is None else round(gaze_jitter_rms, 6),
            "weights": {"pupil_response": 0.65, "recovery": 0.25, "attentiveness": 0.10},
            "notes": (
                "Current score combines pupil dynamics, low-weight gaze stability proxy, and EEG attention direction."
            ),
        },
        alpha=0.3,
    )

    _results = {
        "baseline": baseline,
        "pipr_6": pipr_6,
        "pipr_30": pipr_30,
        "n_base": len(base_vals),
        "n_pipr6": len(pipr6_vals),
        "n_pipr30": len(pipr30_vals),
        "reason_baseline": reason_baseline,
        "reason_pipr6": reason_pipr6,
        "reason_pipr30": reason_pipr30,
        "engagement": {
            "session_score": round(session_score, 3),
            "ema_score": stored.get("ema_score"),
            "pupil_response_score": None if pupil_response_score is None else round(pupil_response_score, 3),
            "recovery_score": None if recovery_score is None else round(recovery_score, 3),
            "attentiveness_score": None if attentiveness_score is None else round(attentiveness_score, 3),
            "concentration_score": None if attentiveness_score is None else round(attentiveness_score, 3),
            "eeg_concentration_score": None if eeg_concentration_score is None else round(eeg_concentration_score, 3),
            "eeg_alpha_theta_ratio": None if eeg_alpha_theta_ratio is None else round(eeg_alpha_theta_ratio, 6),
            "gaze_jitter_rms": None if gaze_jitter_rms is None else round(gaze_jitter_rms, 6),
            "reason_response": response_reason,
            "reason_recovery": recovery_reason,
            "reason_attentiveness": attentiveness_reason,
            "reason_concentration": attentiveness_reason,
            "reason_eeg": reason_eeg,
            "retake_recommended": retake_recommended,
            "weights": {"pupil_response": 0.65, "recovery": 0.25, "attentiveness": 0.10},
            "scientific_notes": (
                "This is a proxy engagement metric. It is not a clinical diagnosis. "
                "Gaze stability can reflect relaxation as well as attentional state. "
                "EEG is used as a directional modulator with lenient penalties for mismatch."
            ),
        },
    }
    _broadcast({"type": "done", "results": _results})


app = FastAPI(title="Neurostasis PIPR")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
register_engagement_routes(app)


@app.get("/")
def root() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/next-event")
def next_event() -> dict:
    subscriber = _subscribe()
    try:
        first = subscriber.get(timeout=120)
        events = [first]
        for _ in range(63):
            try:
                events.append(subscriber.get_nowait())
            except queue.Empty:
                break
        if len(events) == 1:
            return first
        return {"type": "batch", "events": events}
    except queue.Empty:
        return {"type": "timeout"}
    finally:
        _unsubscribe(subscriber)


@app.get("/status")
def status() -> dict:
    with _snapshot_lock:
        return dict(_snapshot)


@app.get("/results")
def results() -> dict:
    if _results is None:
        raise HTTPException(status_code=404, detail="not ready")
    return _results


@app.post("/start")
def start(config: StartRequest) -> dict:
    global _acquisition_thread, _results

    if config.t_on >= config.t_off:
        raise HTTPException(status_code=400, detail="t_on must be less than t_off")
    if config.t_off >= config.total_s:
        raise HTTPException(status_code=400, detail="t_off must be less than total_s")
    if config.baseline_s > config.t_on:
        raise HTTPException(status_code=400, detail="baseline_s must be <= t_on")

    with _run_lock:
        if _acquisition_thread is not None and _acquisition_thread.is_alive():
            return {"status": "already running"}

        _results = None
        with _snapshot_lock:
            _snapshot.clear()
            _snapshot.update(
                {
                    "phase": "starting",
                    "elapsed": 0.0,
                    "pupil": None,
                    "samples": 0,
                    "gaze_x": None,
                    "gaze_y": None,
                    "worn": None,
                    "eeg_concentration_score": None,
                    "eeg_alpha_theta_ratio": None,
                    "eeg_status": "starting",
                }
            )
        _acquisition_thread = threading.Thread(target=_run_acquisition, args=(config,), daemon=True)
        _acquisition_thread.start()
    return {"status": "started"}


if __name__ == "__main__":
    import uvicorn

    host, port = "127.0.0.1", 8000
    url = f"http://{host}:{port}/"
    print(f"PIPR server -> {url}")
    print("Press Ctrl-C to stop.\n")
    webbrowser.open(url)
    uvicorn.run(app, host=host, port=port)
