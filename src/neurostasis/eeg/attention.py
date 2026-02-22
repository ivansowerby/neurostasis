from __future__ import annotations

from dataclasses import dataclass

from . import EEGRunner

@dataclass(frozen=True)
class AttentionState:
    timestamp_unix_seconds: float
    alpha_power: float
    theta_power: float
    alpha_theta_ratio: float
    concentration_score: float

def latest_attention_states(runner: EEGRunner, count: int = 1) -> list[AttentionState]:
    if count < 1:
        raise ValueError("count must be >= 1")

    metrics = runner.metric_history()
    if not metrics:
        return []

    selected = metrics[-count:]
    return [
        AttentionState(
            timestamp_unix_seconds=m.timestamp_unix_seconds,
            alpha_power=m.alpha_power,
            theta_power=m.theta_power,
            alpha_theta_ratio=m.alpha_theta_ratio,
            concentration_score=m.concentration_score,
        )
        for m in selected
    ]


__all__: list[str] = ["AttentionState", "latest_attention_states"]

if __name__ == '__main__':
    import asyncio
    import threading
    from . import EEGConfig, EEGRunner
    from .attention import latest_attention_states

    config = EEGConfig(enable_ui=False)
    runner = EEGRunner(config)

    # run() blocks, so start it in a background thread/process
    threading.Thread(target=runner.run, daemon=True).start()

    async def poll_loop():
        while True:
            states = latest_attention_states(runner, count=1)
            if states:
                state = states[0]
                print(state.concentration_score, state.alpha_theta_ratio)
            await asyncio.sleep(0.05)

    asyncio.run(poll_loop())
