from __future__ import annotations

import csv
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock
from typing import Any, Deque

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[3]
LOCAL_GPYPE_DIR = ROOT_DIR / "libs" / "gpype"
os.environ.setdefault("GPYPE_SETTINGS_DIR", str(ROOT_DIR / ".gpype"))
if LOCAL_GPYPE_DIR.exists() and str(LOCAL_GPYPE_DIR) not in sys.path:
    sys.path.insert(0, str(LOCAL_GPYPE_DIR))


@dataclass(frozen=True)
class ConcentrationMetric:
    timestamp_unix_seconds: float
    alpha_power: float
    theta_power: float
    alpha_theta_ratio: float
    concentration_score: float


@dataclass(frozen=True)
class EEGConfig:
    sampling_rate: int = 250
    include_accel: bool = True
    include_gyro: bool = True
    include_aux: bool = True
    bandpass_low_hz: float = 1.0
    bandpass_high_hz: float = 30.0
    notch50_low_hz: float = 48.0
    notch50_high_hz: float = 52.0
    notch60_low_hz: float = 58.0
    notch60_high_hz: float = 62.0
    fft_window_size: int = 250
    fft_overlap: float = 0.5
    alpha_low_hz: float = 8.0
    alpha_high_hz: float = 12.0
    theta_low_hz: float = 4.0
    theta_high_hz: float = 8.0
    ratio_low: float = 0.6
    ratio_high: float = 2.4
    enable_lsl: bool = False
    lsl_stream_name: str = "Neurostasis_HybridBlack_Filtered"
    lsl_stream_type: str = "EEG"
    enable_ui: bool = True
    enable_csv: bool = False
    csv_file: str = "eeg_filtered.csv"
    metrics_csv_file: str = "eeg_alpha_theta_metrics.csv"
    time_scope_limit_uv: float = 50.0
    time_scope_window_s: float = 10.0
    spectrum_limit: float = 20.0
    test_signal: bool = False
    duration_s: float = 0.0
    print_interval_s: float = 1.0
    history_size: int = 2048


def _require_gpype() -> Any:
    try:
        import gpype as gp

        return gp
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "gpype runtime dependency is missing. "
            "Install the g.tec runtime package that provides 'gtec_licensing', "
            "then run the EEG pipeline before polling attention state."
        ) from exc


class AlphaThetaMetricSink:
    def __init__(
        self,
        alpha_low_hz: float,
        alpha_high_hz: float,
        theta_low_hz: float,
        theta_high_hz: float,
        ratio_low: float,
        ratio_high: float,
        print_interval_s: float,
        history_size: int,
        metrics_csv_path: Path | None,
    ) -> None:
        self._alpha_low_hz: float = alpha_low_hz
        self._alpha_high_hz: float = alpha_high_hz
        self._theta_low_hz: float = theta_low_hz
        self._theta_high_hz: float = theta_high_hz
        self._ratio_low: float = ratio_low
        self._ratio_high: float = ratio_high
        self._print_interval_s: float = print_interval_s
        self._history: Deque[ConcentrationMetric] = deque(maxlen=max(8, history_size))
        self._lock: Lock = Lock()
        self._last_print_s: float = 0.0
        self._freqs_hz: np.ndarray = np.array([], dtype=np.float64)
        self._alpha_mask: np.ndarray = np.array([], dtype=bool)
        self._theta_mask: np.ndarray = np.array([], dtype=bool)
        self._metrics_csv_path: Path | None = metrics_csv_path
        self._metrics_file: csv.TextIOWrapper | None = None
        self._writer: csv.writer | None = None
        self.node: Any | None = None

    def build_node(self) -> Any:
        gp = _require_gpype()
        sink = self

        class _SinkNode(gp.INode):
            def setup(self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]) -> dict[str, dict]:
                context_out: dict[str, dict] = super().setup(data, port_context_in)
                first_context: dict = next(iter(port_context_in.values()))
                sampling_rate: float = float(first_context.get(gp.Constants.Keys.SAMPLING_RATE, 250.0))
                window_size: int = int(first_context.get(gp.Constants.Keys.FRAME_SIZE, 250))
                sink._freqs_hz = np.fft.rfftfreq(window_size, d=1.0 / sampling_rate)
                sink._alpha_mask = (sink._freqs_hz >= sink._alpha_low_hz) & (sink._freqs_hz < sink._alpha_high_hz)
                sink._theta_mask = (sink._freqs_hz >= sink._theta_low_hz) & (sink._freqs_hz < sink._theta_high_hz)
                if sink._metrics_csv_path is not None:
                    sink._metrics_csv_path.parent.mkdir(parents=True, exist_ok=True)
                    sink._metrics_file = sink._metrics_csv_path.open("w", newline="", encoding="utf-8")
                    sink._writer = csv.writer(sink._metrics_file)
                    sink._writer.writerow(
                        [
                            "timestamp_unix_seconds",
                            "alpha_power",
                            "theta_power",
                            "alpha_theta_ratio",
                            "concentration_score",
                        ]
                    )
                return context_out

            def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray] | None:
                block: np.ndarray = next(iter(data.values()))
                if block is None or block.size == 0:
                    return None
                amplitude: np.ndarray = np.asarray(block, dtype=np.float64)
                power: np.ndarray = np.square(amplitude, dtype=np.float64)
                alpha_power: float = float(np.mean(power[sink._alpha_mask, :])) if np.any(sink._alpha_mask) else 0.0
                theta_power: float = float(np.mean(power[sink._theta_mask, :])) if np.any(sink._theta_mask) else 0.0
                ratio: float = alpha_power / max(theta_power, 1e-12)
                scaled: float = (ratio - sink._ratio_low) / max(sink._ratio_high - sink._ratio_low, 1e-9)
                score: float = float(np.clip(100.0 * scaled, 0.0, 100.0))
                metric: ConcentrationMetric = ConcentrationMetric(
                    timestamp_unix_seconds=time.time(),
                    alpha_power=alpha_power,
                    theta_power=theta_power,
                    alpha_theta_ratio=ratio,
                    concentration_score=score,
                )
                with sink._lock:
                    sink._history.append(metric)
                if sink._writer is not None:
                    sink._writer.writerow(
                        [
                            f"{metric.timestamp_unix_seconds:.6f}",
                            f"{metric.alpha_power:.10f}",
                            f"{metric.theta_power:.10f}",
                            f"{metric.alpha_theta_ratio:.10f}",
                            f"{metric.concentration_score:.6f}",
                        ]
                    )
                    if sink._metrics_file is not None:
                        sink._metrics_file.flush()
                now_s: float = time.time()
                if now_s - sink._last_print_s >= sink._print_interval_s:
                    print(
                        " | ".join(
                            [
                                f"alpha={metric.alpha_power:.6f}",
                                f"theta={metric.theta_power:.6f}",
                                f"ratio={metric.alpha_theta_ratio:.4f}",
                                f"score={metric.concentration_score:.2f}/100",
                            ]
                        )
                    )
                    sink._last_print_s = now_s
                return None

            def stop(self) -> None:
                super().stop()
                if sink._metrics_file is not None:
                    sink._metrics_file.close()
                    sink._metrics_file = None
                    sink._writer = None

        self.node = _SinkNode()
        return self.node

    def latest(self) -> ConcentrationMetric | None:
        with self._lock:
            return self._history[-1] if self._history else None

    def history(self) -> list[ConcentrationMetric]:
        with self._lock:
            return list(self._history)


class EEGRunner:
    def __init__(self, config: EEGConfig) -> None:
        self.config: EEGConfig = config
        self.pipeline: Any | None = None
        self.app: Any | None = None
        self._stop_event: Event = Event()
        metrics_csv_path: Path | None = Path(config.metrics_csv_file) if config.enable_csv else None
        self.metric_sink: AlphaThetaMetricSink = AlphaThetaMetricSink(
            alpha_low_hz=config.alpha_low_hz,
            alpha_high_hz=config.alpha_high_hz,
            theta_low_hz=config.theta_low_hz,
            theta_high_hz=config.theta_high_hz,
            ratio_low=config.ratio_low,
            ratio_high=config.ratio_high,
            print_interval_s=config.print_interval_s,
            history_size=config.history_size,
            metrics_csv_path=metrics_csv_path,
        )

    def build(self) -> None:
        gp = _require_gpype()
        cfg: EEGConfig = self.config
        self.pipeline = gp.Pipeline()
        self.app = gp.MainApp() if cfg.enable_ui else None

        source: Any = gp.HybridBlack(
            include_accel=cfg.include_accel,
            include_gyro=cfg.include_gyro,
            include_aux=cfg.include_aux,
            test_signal=cfg.test_signal,
        )
        splitter: Any = gp.Router(
            input_channels=gp.Router.ALL,
            output_channels={
                "EEG": range(8),
                "ACC": [8, 9, 10],
                "GYRO": [11, 12, 13],
                "AUX": [14, 15, 16],
            },
        )
        bandpass: Any = gp.Bandpass(f_lo=cfg.bandpass_low_hz, f_hi=cfg.bandpass_high_hz)
        notch50: Any = gp.Bandstop(f_lo=cfg.notch50_low_hz, f_hi=cfg.notch50_high_hz)
        notch60: Any = gp.Bandstop(f_lo=cfg.notch60_low_hz, f_hi=cfg.notch60_high_hz)
        fft: Any = gp.FFT(
            window_size=cfg.fft_window_size,
            overlap=float(cfg.fft_overlap),
            window_function="hamming",
        )
        alpha_bp: Any = gp.Bandpass(f_lo=cfg.alpha_low_hz, f_hi=cfg.alpha_high_hz)
        theta_bp: Any = gp.Bandpass(f_lo=cfg.theta_low_hz, f_hi=cfg.theta_high_hz)
        alpha_pow: Any = gp.Equation("in**2")
        theta_pow: Any = gp.Equation("in**2")
        alpha_avg: Any = gp.MovingAverage(window_size=int(cfg.sampling_rate * 0.5))
        theta_avg: Any = gp.MovingAverage(window_size=int(cfg.sampling_rate * 0.5))
        sink_node: Any = self.metric_sink.build_node()

        self.pipeline.connect(source, splitter)
        self.pipeline.connect(splitter["EEG"], bandpass)
        self.pipeline.connect(bandpass, notch50)
        self.pipeline.connect(notch50, notch60)
        self.pipeline.connect(notch60, fft)
        self.pipeline.connect(fft, sink_node)
        self.pipeline.connect(notch60, alpha_bp)
        self.pipeline.connect(notch60, theta_bp)
        self.pipeline.connect(alpha_bp, alpha_pow)
        self.pipeline.connect(theta_bp, theta_pow)
        self.pipeline.connect(alpha_pow, alpha_avg)
        self.pipeline.connect(theta_pow, theta_avg)

        if cfg.enable_lsl:
            sender: Any = gp.LSLSender(stream_name=cfg.lsl_stream_name, stype=cfg.lsl_stream_type)
            self.pipeline.connect(notch60, sender)
        if cfg.enable_csv:
            writer: Any = gp.CsvWriter(file_name=cfg.csv_file)
            self.pipeline.connect(notch60, writer)
        if self.app is not None:
            time_scope: Any = gp.TimeSeriesScope(
                amplitude_limit=cfg.time_scope_limit_uv,
                time_window=cfg.time_scope_window_s,
            )
            spectrum_scope: Any = gp.SpectrumScope(amplitude_limit=cfg.spectrum_limit)
            alpha_scope: Any = gp.TimeSeriesScope(amplitude_limit=50, time_window=cfg.time_scope_window_s)
            theta_scope: Any = gp.TimeSeriesScope(amplitude_limit=50, time_window=cfg.time_scope_window_s)
            self.pipeline.connect(notch60, time_scope)
            self.pipeline.connect(fft, spectrum_scope)
            self.pipeline.connect(alpha_avg, alpha_scope)
            self.pipeline.connect(theta_avg, theta_scope)
            self.app.add_widget(time_scope)
            self.app.add_widget(spectrum_scope)
            self.app.add_widget(alpha_scope)
            self.app.add_widget(theta_scope)

    def run(self) -> ConcentrationMetric | None:
        self._stop_event.clear()
        self.build()
        assert self.pipeline is not None
        self.pipeline.start()
        try:
            if self.app is not None:
                self.app.run()
            else:
                start_s: float = time.time()
                while True:
                    if self._stop_event.is_set():
                        break
                    if self.config.duration_s > 0 and (time.time() - start_s) >= self.config.duration_s:
                        break
                    time.sleep(0.05)
        finally:
            self.pipeline.stop()
        return self.metric_sink.latest()

    def stop(self) -> None:
        self._stop_event.set()

    def latest_metric(self) -> ConcentrationMetric | None:
        return self.metric_sink.latest()

    def metric_history(self) -> list[ConcentrationMetric]:
        return self.metric_sink.history()


__all__: list[str] = [
    "ConcentrationMetric",
    "EEGConfig",
    "AlphaThetaMetricSink",
    "EEGRunner",
]
