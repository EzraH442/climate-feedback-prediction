import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import xarray as xr

from experiments.bivariate_perturbation.analysis import (
    plot_kernel_contour,
    plot_tsr_contour,
)
from experiments.kernel_difference.analysis import (
    plot_hybrid_kernel_difference,
    plot_kernel_difference,
)
from utils import (
    figure_dir,
    kernel_label,
    kernel_title,
    plot_global_field,
    plot_north_pole_field,
)


def eval_kernel_vars(config):
    input_vars = list(config.dataset.input_vars)
    return [var for var in ("fal", "skt", "tcwv") if var in input_vars]


def plot_flux_summary(result: xr.Dataset, figures_path: Path) -> None:
    output_dir = figure_dir(figures_path, result.attrs["sky"], result.attrs["test_name"])
    target_label = result.attrs["target_label"]
    lon = result.longitude
    lat = result.latitude

    min_flux = min(float(result.truth_mean.min()), float(result.prediction_mean.min()))
    max_flux = max(float(result.truth_mean.max()), float(result.prediction_mean.max()))

    plot_global_field(
        result.truth_mean,
        lon,
        lat,
        f"{target_label} (ERA5)",
        output_dir / "era5.png",
        cmap="Spectral",
        vmin=min_flux,
        vmax=max_flux,
        label="$W/m^2$",
        annotation=f"{float(result.truth_mean.mean()):.2f}",
    )
    plot_global_field(
        result.prediction_mean,
        lon,
        lat,
        f"{target_label} (NN)",
        output_dir / "nn.png",
        cmap="Spectral",
        vmin=min_flux,
        vmax=max_flux,
        label="$W/m^2$",
        annotation=f"{float(result.prediction_mean.mean()):.2f}",
    )
    plot_global_field(
        result.mbe,
        lon,
        lat,
        "MBE",
        output_dir / "mbe.png",
        cmap="RdBu_r",
        vmin=-20,
        vmax=20,
        label="$W/m^2$",
        annotation=f"{float(result.mbe.mean()):.2f}",
    )
    plot_global_field(
        result.rmse,
        lon,
        lat,
        "RMSE",
        output_dir / "rmse.png",
        cmap="Blues",
        vmin=0,
        vmax=20,
        label="$W/m^2$",
        annotation=f"{float(np.sqrt((result.rmse**2).mean())):.2f}",
    )


def plot_kernel_date_result(result: xr.Dataset, figures_path: Path) -> None:
    kernel_name = result.attrs["kernel_name"]
    date = result.attrs["date"]
    title = kernel_title(kernel_name)
    label = kernel_label(kernel_name)
    lon = result.longitude
    lat = result.latitude
    north_mask = lat >= 60
    all_dir = figure_dir(figures_path, "all", "kernel_date_test", kernel_name, date)
    clr_dir = figure_dir(figures_path, "clr", "kernel_date_test", kernel_name, date)

    for sky, sky_dir, suffix in [("all", all_dir, "all"), ("clear", clr_dir, "clear")]:
        rrtm_name = f"rrtm_{sky}"
        fd_name = f"nn_fd_{sky}"
        grad_name = f"nn_grad_{sky}"
        diff_fd_name = f"diff_fd_{sky}"
        diff_grad_name = f"diff_grad_{sky}"

        plot_global_field(
            result[fd_name],
            lon,
            lat,
            f"NN {title} Kernel finite difference ({suffix})\n{date}",
            sky_dir / "nn.png",
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
            label=label,
            annotation=f"{float(result[fd_name].mean()):.2f}",
        )
        plot_global_field(
            result[grad_name],
            lon,
            lat,
            f"NN {title} Kernel via autograd ({suffix})\n{date}",
            sky_dir / "nn_grad.png",
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
            label=label,
            annotation=f"{float(result[grad_name].mean()):.2f}",
        )
        plot_north_pole_field(
            result[fd_name],
            lon,
            lat,
            f"NN {title} Kernel finite difference ({suffix})\n{date}",
            sky_dir / "nn_np.png",
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
            label=label,
            annotation=f"{float(result[fd_name].where(north_mask).mean()):.2f}",
        )
        plot_north_pole_field(
            result[grad_name],
            lon,
            lat,
            f"NN {title} Kernel via autograd ({suffix})\n{date}",
            sky_dir / "nn_grad_np.png",
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
            label=label,
            annotation=f"{float(result[grad_name].where(north_mask).mean()):.2f}",
        )
        plot_global_field(
            result[rrtm_name],
            lon,
            lat,
            f"RRTM {title} Kernel ({suffix})\n{date}",
            sky_dir / "rrtm.png",
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
            label=label,
            annotation=f"{float(result[rrtm_name].mean()):.2f}",
        )
        plot_north_pole_field(
            result[rrtm_name],
            lon,
            lat,
            f"RRTM {title} Kernel ({suffix})\n{date}",
            sky_dir / "rrtm_np.png",
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
            label=label,
            annotation=f"{float(result[rrtm_name].where(north_mask).mean()):.2f}",
        )
        plot_global_field(
            result[diff_fd_name],
            lon,
            lat,
            f"NN finite difference-RRTM {title} Kernel ({suffix})\n{date}",
            sky_dir / "nn-rrtm.png",
            cmap="RdBu_r",
            vmin=-0.6,
            vmax=0.6,
            label=label,
            annotation=f"{float(result[diff_fd_name].mean()):.2f}; {float(abs(result[diff_fd_name]).mean()):.2f}",
        )
        plot_global_field(
            result[diff_grad_name],
            lon,
            lat,
            f"NN autograd-RRTM {title} Kernel ({suffix})\n{date}",
            sky_dir / "nn_grad-rrtm.png",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            label=label,
            annotation=f"{float(result[diff_grad_name].mean()):.2f}; {float(abs(result[diff_grad_name]).mean()):.2f}",
        )
        plot_north_pole_field(
            result[diff_fd_name],
            lon,
            lat,
            f"NN finite difference-RRTM {title} Kernel ({suffix})\n{date}",
            sky_dir / "nn-rrtm_np.png",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            label=label,
            annotation=f"{float(result[diff_fd_name].where(north_mask).mean()):.2f}; {float(abs(result[diff_fd_name].where(north_mask)).mean()):.2f}",
        )
        plot_north_pole_field(
            result[diff_grad_name],
            lon,
            lat,
            f"NN autograd-RRTM {title} Kernel ({suffix})\n{date}",
            sky_dir / "nn_grad-rrtm_np.png",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            label=label,
            annotation=f"{float(result[diff_grad_name].where(north_mask).mean()):.2f}; {float(abs(result[diff_grad_name].where(north_mask)).mean()):.2f}",
        )


def kernel_difference_fields(result: xr.Dataset) -> dict:
    fields = {name: result[name] for name in result.data_vars}
    fields["base_date"] = result.attrs["base_date"]
    fields["perturbed_date"] = result.attrs["perturbed_date"]
    fields["lon"] = result.longitude
    fields["lat"] = result.latitude
    return fields


def plot_standard_eval_results(output_dir: Path) -> None:
    results_dir = output_dir / "results"
    plot_flux_summary(xr.open_dataset(results_dir / "tsr_test.nc"), output_dir)
    plot_flux_summary(xr.open_dataset(results_dir / "tsrc_test.nc"), output_dir)

    for path in sorted(results_dir.glob("kernel_date_*.nc")):
        plot_kernel_date_result(xr.open_dataset(path), output_dir)

    for path in sorted(results_dir.glob("second_order_*.nc")):
        fields = kernel_difference_fields(xr.open_dataset(path))
        second_order_dir = figure_dir(
            output_dir,
            "all",
            "second_order_test",
            path.stem.removeprefix("second_order_"),
            f"{fields['perturbed_date']}_minus_{fields['base_date']}",
        )
        plot_kernel_difference(fields, second_order_dir)

    for path in sorted(results_dir.glob("hybrid_second_order_*.nc")):
        fields = kernel_difference_fields(xr.open_dataset(path))
        second_order_dir = figure_dir(
            output_dir,
            "all",
            "second_order_test",
            path.stem.removeprefix("hybrid_second_order_"),
            f"{fields['perturbed_date']}_minus_{fields['base_date']}",
        )
        plot_hybrid_kernel_difference(fields, second_order_dir)

    plot_kernel_contour(xr.open_dataset(results_dir / "kernel_contour_fal_ecod.nc"), output_dir)
    plot_tsr_contour(xr.open_dataset(results_dir / "tsr_contour_fal_ecod.nc"), output_dir)
