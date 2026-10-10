# Example: python experiments/tsr_prediction/analysis_line_sobolev.py --all_sky_metrics_path sobolev-results/tsr/0.25/all/metrics.json sobolev-results/tsr/0.5/all/metrics.json sobolev-results/tsr/1/all/metrics.json sobolev-results/tsr/2/all/metrics.json sobolev-results/tsr/4/all/metrics.json --clear_sky_metrics_path sobolev-results/tsr/0.25/clear/metrics.json sobolev-results/tsr/0.5/clear/metrics.json sobolev-results/tsr/1/clear/metrics.json sobolev-results/tsr/2/clear/metrics.json sobolev-results/tsr/4/clear/metrics.json
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tap import Tap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import parse_args_and_confirm

LAMBDA_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]


class SobolevLineArgs(Tap):
    all_sky_metrics_path: list[Path]
    clear_sky_metrics_path: list[Path]
    output_path: Path = Path("sobolev_tsr_performance.png")
    lambda_values: list[float] = LAMBDA_VALUES


def load_metrics(paths: list[Path]) -> dict[str, np.ndarray]:
    values = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return {
        key: np.array([value[key] for value in values])
        for key in ("mbe_mean", "mbe_std", "rmse_mean", "rmse_std")
    }


def plot(args: SobolevLineArgs) -> None:
    if len(args.lambda_values) != len(args.all_sky_metrics_path) or len(
        args.lambda_values
    ) != len(args.clear_sky_metrics_path):
        raise ValueError("Provide one all-sky and clear-sky metrics file per lambda.")
    all_sky = load_metrics(args.all_sky_metrics_path)
    clear_sky = load_metrics(args.clear_sky_metrics_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, metric, label in [
        (axes[0], "mbe", "MBE (W/m$^2$)"),
        (axes[1], "rmse", "RMSE (W/m$^2$)"),
    ]:
        ax.errorbar(
            args.lambda_values,
            all_sky[f"{metric}_mean"],
            yerr=all_sky[f"{metric}_std"],
            marker="o",
            label="all-sky",
        )
        ax.errorbar(
            args.lambda_values,
            clear_sky[f"{metric}_mean"],
            yerr=clear_sky[f"{metric}_std"],
            marker="o",
            label="clear-sky",
        )
        ax.set_xscale("log")
        ax.set_xlabel("Sobolev lambda")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    axes[0].legend()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args_and_confirm(SobolevLineArgs())
    plot(args)


if __name__ == "__main__":
    main()
