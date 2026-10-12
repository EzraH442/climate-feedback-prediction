# Example: python experiments/sensitivity_prediction/analysis_line_sobolev.py --all_sky_metrics_path sobolev-results/sensitivity/0.25/all/metrics.json sobolev-results/sensitivity/0.5/all/metrics.json sobolev-results/sensitivity/1/all/metrics.json sobolev-results/sensitivity/2/all/metrics.json sobolev-results/sensitivity/4/all/metrics.json --clear_sky_metrics_path sobolev-results/sensitivity/0.25/clear/metrics.json sobolev-results/sensitivity/0.5/clear/metrics.json sobolev-results/sensitivity/1/clear/metrics.json sobolev-results/sensitivity/2/clear/metrics.json sobolev-results/sensitivity/4/clear/metrics.json --baseline_metrics_path sobolev-results/sensitivity/baseline/all/metrics.json --baseline_clearsky_metrics_path sobolev-results/sensitivity/baseline_clearsky/clear/metrics.json
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator
from tap import Tap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import parse_args_and_confirm

LAMBDA_VALUES = [0.25, 0.5, 1.0, 2.0, 4.0]


class SobolevLineArgs(Tap):
    all_sky_metrics_path: list[Path]
    clear_sky_metrics_path: list[Path]
    baseline_metrics_path: Path | None = None
    baseline_clear_metrics_path: Path | None = None
    baseline_clearsky_all_metrics_path: Path | None = None
    baseline_clearsky_metrics_path: Path | None = None
    output_path: Path = Path("sobolev_sensitivity_performance.png")
    lambda_values: list[float] = LAMBDA_VALUES


def load_metric(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_metrics(paths: list[Path]) -> dict[str, np.ndarray]:
    values = [load_metric(path) for path in paths]
    return {
        key: np.array([value[key] for value in values])
        for key in ("mbe_mean", "mbe_std", "rmse_mean", "rmse_std")
    }


def plot_with_std(ax, x, mean, std, label, **plot_kwargs):
    (line,) = ax.plot(x, mean, label=label, **plot_kwargs)
    ax.fill_between(
        x,
        mean - std,
        mean + std,
        color=line.get_color(),
        alpha=0.15,
        linewidth=0,
    )


def plot(args: SobolevLineArgs) -> None:
    if len(args.lambda_values) != len(args.all_sky_metrics_path) or len(
        args.lambda_values
    ) != len(args.clear_sky_metrics_path):
        raise ValueError("Provide one all-sky and clear-sky metrics file per lambda.")
    lambda_values = np.array(args.lambda_values, dtype=float)
    all_sky = load_metrics(args.all_sky_metrics_path)
    clear_sky = load_metrics(args.clear_sky_metrics_path)
    baseline = (
        load_metric(args.baseline_metrics_path) if args.baseline_metrics_path else None
    )
    baseline_clear = (
        load_metric(args.baseline_clear_metrics_path)
        if args.baseline_clear_metrics_path
        else None
    )
    baseline_clearsky_all = (
        load_metric(args.baseline_clearsky_all_metrics_path)
        if args.baseline_clearsky_all_metrics_path
        else None
    )
    baseline_clearsky = (
        load_metric(args.baseline_clearsky_metrics_path)
        if args.baseline_clearsky_metrics_path
        else None
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for ax, sky_metrics, sky_baselines, metric, label, title in [
        (
            axes[0, 0],
            clear_sky,
            [(baseline_clear, "Baseline"), (baseline_clearsky, "Baseline clear-sky")],
            "mbe",
            "MBE (W/m$^2$)",
            "Clear-sky MBE",
        ),
        (
            axes[0, 1],
            clear_sky,
            [(baseline_clear, "Baseline"), (baseline_clearsky, "Baseline clear-sky")],
            "rmse",
            "RMSE (W/m$^2$)",
            "Clear-sky RMSE",
        ),
        (
            axes[1, 0],
            all_sky,
            [(baseline, "Baseline"), (baseline_clearsky_all, "Baseline clear-sky")],
            "mbe",
            "MBE (W/m$^2$)",
            "All-sky MBE",
        ),
        (
            axes[1, 1],
            all_sky,
            [(baseline, "Baseline"), (baseline_clearsky_all, "Baseline clear-sky")],
            "rmse",
            "RMSE (W/m$^2$)",
            "All-sky RMSE",
        ),
    ]:
        plot_with_std(
            ax,
            lambda_values,
            sky_metrics[f"{metric}_mean"],
            sky_metrics[f"{metric}_std"],
            marker="o",
            label="Sobolev",
        )
        for sky_baseline, baseline_label in sky_baselines:
            if not sky_baseline:
                continue
            plot_with_std(
                ax,
                lambda_values,
                np.full_like(lambda_values, sky_baseline[f"{metric}_mean"]),
                np.full_like(lambda_values, sky_baseline[f"{metric}_std"]),
                linestyle="--",
                label=baseline_label,
            )
        ax.set_xscale("log")
        ax.set_xticks(lambda_values)
        ax.set_xticklabels([f"{value:g}" for value in lambda_values])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Sobolev lambda")
        ax.set_ylabel(label)
        ax.set_title(title)
        ax.grid(alpha=0.3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(labels),
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args_and_confirm(SobolevLineArgs())
    plot(args)


if __name__ == "__main__":
    main()
