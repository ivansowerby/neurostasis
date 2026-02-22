from __future__ import annotations

from neurostasis.eeg import ConcentrationMetric
from neurostasis.eeg.attention import AttentionState, latest_attention_states


class _FakeRunner:
    def __init__(self, metrics: list[ConcentrationMetric]) -> None:
        self._metrics = metrics

    def metric_history(self) -> list[ConcentrationMetric]:
        return list(self._metrics)


def _metric(ts: float, alpha: float, theta: float, ratio: float, score: float) -> ConcentrationMetric:
    return ConcentrationMetric(
        timestamp_unix_seconds=ts,
        alpha_power=alpha,
        theta_power=theta,
        alpha_theta_ratio=ratio,
        concentration_score=score,
    )


def test_latest_attention_states_returns_latest_n() -> None:
    runner = _FakeRunner(
        [
            _metric(1.0, 10.0, 5.0, 2.0, 40.0),
            _metric(2.0, 20.0, 8.0, 2.5, 60.0),
            _metric(3.0, 12.0, 6.0, 2.0, 50.0),
        ]
    )

    states = latest_attention_states(runner, count=2)

    assert len(states) == 2
    assert [s.timestamp_unix_seconds for s in states] == [2.0, 3.0]
    assert all(isinstance(s, AttentionState) for s in states)
    assert states[-1].concentration_score == 50.0


def test_latest_attention_states_handles_empty_and_bad_count() -> None:
    runner = _FakeRunner([])

    assert latest_attention_states(runner) == []

    try:
        latest_attention_states(runner, count=0)
        assert False, "Expected ValueError for count=0"
    except ValueError:
        pass
