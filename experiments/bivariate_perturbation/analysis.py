import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from utils import figure_dir, kernel_label


def plot_contour_points(ax, result: xr.Dataset):
    ax.scatter(
        [float(result.attrs["base_var_a"])],
        [float(result.attrs["base_var_b"])],
        s=20,
        c="white",
        edgecolors="black",
    )


def plot_kernel_contour(result: xr.Dataset, figures_path: Path = Path(".")) -> None:
    var_a = result.attrs["var_a"]
    var_b = result.attrs["var_b"]
    date = result.attrs["date"]
    latitude = float(result.attrs["latitude"])
    longitude = float(result.attrs["longitude"])
    kernel = result["kernel"]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    max_abs = float(np.nanmax(np.abs(kernel)))
    levels = np.linspace(-max_abs, max_abs, 31) if max_abs > 0 else 31
    contour = ax.contourf(
        result.var_a,
        result.var_b,
        kernel,
        levels=levels,
        cmap="RdBu_r",
        extend="both",
    )
    plot_contour_points(ax, result)
    ax.set_xlabel(var_a)
    ax.set_ylabel(var_b)
    ax.set_title(
        f"NN surface albedo kernel over {var_a}/{var_b}; "
        f"lat={latitude:.2f}, lon={longitude:.2f}, date={date}"
    )
    cb = fig.colorbar(contour, ax=ax)
    cb.set_label(kernel_label(var_a))
    fig.tight_layout()
    output_dir = figure_dir(figures_path, "all", "kernel_contour_test", var_a, date)
    fig.savefig(output_dir / f"{var_a}_{var_b}.png")
    plt.close(fig)


def plot_tsr_contour(result: xr.Dataset, figures_path: Path = Path(".")) -> None:
    var_a = result.attrs["var_a"]
    var_b = result.attrs["var_b"]
    date = result.attrs["date"]
    latitude = float(result.attrs["latitude"])
    longitude = float(result.attrs["longitude"])
    target = result.attrs["target"]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    contour = ax.contourf(
        result.var_a,
        result.var_b,
        result["flux"],
        levels=31,
        cmap="Spectral",
    )
    plot_contour_points(ax, result)
    ax.set_xlabel(var_a)
    ax.set_ylabel(var_b)
    ax.set_title(
        f"NN {target.upper()} over {var_a}/{var_b};"
        f"lat={latitude:.2f}, lon={longitude:.2f}, date={date}"
    )
    cb = fig.colorbar(contour, ax=ax)
    cb.set_label("$W/m^2$")
    fig.tight_layout()
    output_dir = figure_dir(figures_path, "all", "tsr_contour_test", var_a, date)
    fig.savefig(output_dir / f"{var_a}_{var_b}.png")
    plt.close(fig)
