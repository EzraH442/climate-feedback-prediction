import argparse
import sys
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config, variable_config_from_omegaconf
from constants import ALBEDO_KERNEL_PATH
from utils import (
    add_bottom_colorbar,
    annotate_response,
    compute_nn_kernel_autograd,
    generate_paths_yearly,
    load_model_and_preprocessor,
    make_era5_filename,
    panel_title,
    plot_shakirova_field,
)


def select_date_state(ds: xr.Dataset, date: str) -> xr.Dataset:
    selected = ds.sel(date=date)
    if "date" in selected.dims:
        selected = selected.squeeze("date", drop=True)
    return selected


def as_field(values, lon, lat):
    return xr.DataArray(
        values,
        coords={"latitude": lat, "longitude": lon},
        dims=("latitude", "longitude"),
    )


def load_two_states(config_path: Path, base_date: str, perturbed_date: str):
    config = load_config(config_path)
    years = sorted({int(base_date[:4]), int(perturbed_date[:4])})
    with xr.open_dataset(ALBEDO_KERNEL_PATH) as albedo_kernel:
        kernel_grid = {
            "latitude": albedo_kernel.latitude,
            "longitude": albedo_kernel.longitude,
        }
        raw_paths = generate_paths_yearly(
            config.dataset.era5.raw_path, years, make_era5_filename
        )
        with xr.open_mfdataset(
            raw_paths, combine="nested", concat_dim="date"
        ) as raw_for_dates:
            ds_for_dates = (
                raw_for_dates.sel(date=[base_date, perturbed_date])
                .interp(**kernel_grid)
                .load()
            )
    return config, select_date_state(ds_for_dates, base_date), select_date_state(
        ds_for_dates, perturbed_date
    )


def kernel_changes(config_path: Path, base, perturbed, base_date: str):
    config = load_config(config_path)
    vc = variable_config_from_omegaconf(config)
    model, preprocessor, _ = load_model_and_preprocessor(
        config, Path(config.train.checkpoint_dir) / "best_model.pt", downscaling=False
    )
    cloud_vars = [var for var in vc.clear_sky_zero_vars if var in base]

    def albedo_sensitivity(state):
        kernel, lon, lat = compute_nn_kernel_autograd(
            state.expand_dims(date=[base_date]),
            preprocessor,
            model,
            vc,
            var="fal",
        )
        return as_field(kernel, lon, lat)

    base_kernel = albedo_sensitivity(base)
    return {
        "a": albedo_sensitivity(base.assign(fal=perturbed.fal)) - base_kernel,
        "c": albedo_sensitivity(
            base.assign({var: perturbed[var] for var in cloud_vars})
        )
        - base_kernel,
        "preprocessor": preprocessor,
    }


def plot(args) -> None:
    if args.output_path is None:
        region = "global" if args.global_map else "arctic"
        args.output_path = Path(
            f"FiguresNeurIPS/neurips_albedo_cloud_kernel_change_"
            f"{args.perturbed_date}_minus_{args.base_date}_{region}.png"
        )
    baseline_config = Path(args.baseline_config)
    sobolev_config = Path(args.sobolev_config)
    _, base, perturbed = load_two_states(
        baseline_config, args.base_date, args.perturbed_date
    )
    anomaly = perturbed - base
    nn = kernel_changes(baseline_config, base, perturbed, args.base_date)
    sob = kernel_changes(sobolev_config, base, perturbed, args.base_date)
    delta_ecod = (
        nn["preprocessor"].transform(perturbed) - nn["preprocessor"].transform(base)
    ).ecod.compute()

    panels = [
        (0, 0, anomaly.fal, r"$\Delta a$", "a", -0.5, 0.5, 0.05, "", ""),
        (1, 0, delta_ecod, r"$\Delta c$", "b", -1, 1, 0.1, "", ""),
        (0, 1, nn["a"], r"$\Delta K_a^{NN}$", "c", -0.6, 0.6, 0.1, None, None),
        (1, 1, nn["c"], r"$\Delta K_c^{NN}$", "d", -0.6, 0.6, 0.1, None, None),
        (0, 2, sob["a"], r"$\Delta K_a^{Sob}$", "e", -0.6, 0.6, 0.1, None, None),
        (1, 2, sob["c"], r"$\Delta K_c^{Sob}$", "f", -0.6, 0.6, 0.1, None, None),
    ]

    fig, ax = plt.subplots(
        2,
        3,
        figsize=(11.69, 7.5),
        subplot_kw={
            "projection": (
                ccrs.NorthPolarStereo(central_longitude=0)
                if not args.global_map
                else ccrs.PlateCarree(central_longitude=180)
            )
        },
        layout="constrained",
    )
    fig.get_layout_engine().set(
        rect=[0.06, 0.12, 1, 0.94], h_pad=0.01, w_pad=0.01, hspace=0.01, wspace=0.01
    )
    kernel_im = None
    kernel_axes = []
    for row, col, field, title, letter, vmin, vmax, step, cbar_label, units in panels:
        im = plot_shakirova_field(
            ax[row][col],
            field,
            panel_title(letter, title),
            vmin=vmin,
            vmax=vmax,
            step=step,
            left_labels=(col == 0),
            bottom_labels=(row == 1),
            arctic=not args.global_map,
            extend=None if col == 0 else "both",
        )
        if cbar_label is None:
            kernel_im = im
            kernel_axes.append(ax[row][col])
            annotate_response(
                ax[row][col],
                field,
                residual=False,
                arctic=not args.global_map,
                units="W m$^{-2}$ %$^{-1}$",
            )
        else:
            annotate_response(ax[row][col], field, residual=False, units=units)
            fig.colorbar(im, ax=ax[row][col], orientation="horizontal", label=cbar_label)

    add_bottom_colorbar(fig, kernel_im, kernel_axes, "W m$^{-2}$ %$^{-1}$", height=0.02)
    fig.set_layout_engine("none")
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_date", default="2012-08")
    parser.add_argument("--perturbed_date", default="2015-08")
    parser.add_argument(
        "--baseline_config",
        default="configs/model/fal/2011-2014_3,6,9,12_baseline.yaml",
    )
    parser.add_argument(
        "--sobolev_config",
        default="configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky.yaml",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=None,
    )
    parser.add_argument("--global_map", action="store_true")
    plot(parser.parse_args())


if __name__ == "__main__":
    main()
