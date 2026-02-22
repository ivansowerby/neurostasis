from __future__ import annotations

import argparse

from . import ConcentrationMetric, EEGConfig, EEGRunner

def parse_args(argv: list[str] | None = None) -> EEGConfig:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(prog="neurostasis.eeg")
    parser.add_argument("--enable-lsl", action="store_true")
    parser.add_argument("--enable-csv", action="store_true")
    parser.add_argument("--no-ui", action="store_true")
    parser.add_argument("--test-signal", action="store_true")
    parser.add_argument("--duration-s", type=float, default=0.0)
    parser.add_argument("--fft-window-size", type=int, default=250)
    parser.add_argument("--fft-overlap", type=float, default=0.5)
    parser.add_argument("--alpha-low-hz", type=float, default=8.0)
    parser.add_argument("--alpha-high-hz", type=float, default=12.0)
    parser.add_argument("--theta-low-hz", type=float, default=4.0)
    parser.add_argument("--theta-high-hz", type=float, default=8.0)
    parser.add_argument("--ratio-low", type=float, default=0.6)
    parser.add_argument("--ratio-high", type=float, default=2.4)
    parser.add_argument("--print-interval-s", type=float, default=1.0)
    parser.add_argument("--csv-file", type=str, default="eeg_filtered.csv")
    parser.add_argument("--metrics-csv-file", type=str, default="eeg_alpha_theta_metrics.csv")
    args: argparse.Namespace = parser.parse_args(argv)
    return EEGConfig(
        enable_lsl=bool(args.enable_lsl),
        enable_csv=bool(args.enable_csv),
        enable_ui=not bool(args.no_ui),
        test_signal=bool(args.test_signal),
        duration_s=float(args.duration_s),
        fft_window_size=int(args.fft_window_size),
        fft_overlap=float(args.fft_overlap),
        alpha_low_hz=float(args.alpha_low_hz),
        alpha_high_hz=float(args.alpha_high_hz),
        theta_low_hz=float(args.theta_low_hz),
        theta_high_hz=float(args.theta_high_hz),
        ratio_low=float(args.ratio_low),
        ratio_high=float(args.ratio_high),
        print_interval_s=float(args.print_interval_s),
        csv_file=str(args.csv_file),
        metrics_csv_file=str(args.metrics_csv_file),
    )


def main(argv: list[str] | None = None) -> int:
    config: EEGConfig = parse_args(argv)
    runner: EEGRunner = EEGRunner(config=config)
    latest: ConcentrationMetric | None = runner.run()
    if latest is not None:
        print(
            " | ".join(
                [
                    f"final_alpha={latest.alpha_power:.6f}",
                    f"final_theta={latest.theta_power:.6f}",
                    f"final_ratio={latest.alpha_theta_ratio:.4f}",
                    f"final_score={latest.concentration_score:.2f}/100",
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
