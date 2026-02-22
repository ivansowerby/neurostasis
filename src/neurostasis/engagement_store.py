"""Persistence and EMA utilities for longitudinal engagement scoring."""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path.cwd() / "data"
STORE_PATH = DATA_DIR / "engagement_scores.json"
_store_lock = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp_score(value: float) -> float:
    return max(0.0, min(100.0, value))


def _read_store_unlocked() -> list[dict[str, Any]]:
    if not STORE_PATH.exists():
        return []
    try:
        raw = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(item)
    return out


def _write_store_unlocked(records: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STORE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
    tmp.replace(STORE_PATH)


def append_engagement_record(record: dict[str, Any], alpha: float = 0.3) -> dict[str, Any]:
    """Appends a session record and computes EMA based on prior stored value."""
    alpha = max(0.01, min(1.0, alpha))
    with _store_lock:
        records = _read_store_unlocked()
        last_ema = None
        if records:
            last_val = records[-1].get("ema_score")
            if isinstance(last_val, (int, float)):
                last_ema = _clamp_score(float(last_val))
        score = _clamp_score(float(record.get("session_score", 0.0)))
        ema = score if last_ema is None else _clamp_score(alpha * score + (1.0 - alpha) * last_ema)
        enriched = dict(record)
        enriched["session_score"] = round(score, 3)
        enriched["ema_score"] = round(ema, 3)
        enriched.setdefault("alpha", alpha)
        enriched.setdefault("timestamp_utc", _utc_now_iso())
        records.append(enriched)
        _write_store_unlocked(records)
        return enriched


def get_engagement_history(limit: int = 200) -> list[dict[str, Any]]:
    with _store_lock:
        records = _read_store_unlocked()
    limit = max(1, limit)
    return records[-limit:]
