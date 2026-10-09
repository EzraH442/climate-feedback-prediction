import sys
from pathlib import Path

import pandas as pd
import xarray as xr
from tap import Tap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import analysis

from config_utils import load_config, variable_config_from_omegaconf
from constants import (
    ALBEDO_KERNEL_PATH,
    CLOSURE_NORTH_BOUNDARY,
    QT_PATH,
    WATER_VAPOR_KERNEL_PATH,
)
from experiments.common import (
    albedo_kernel_components,
    checkpoint_paths_for_args,
    model_label,
    parse_args_and_confirm,
    with_flux_targets,
)
from utils import (
    SECONDS_PER_DAY,
    generate_paths_yearly,
    integrate_over_pressure_levels,
    load_model_and_preprocessor,
    make_era5_filename,
    nn_pred,
    to_monthly,
)


class ClosureEvalArgs(Tap):
    config_file: str = "configs/model/fal/2011-2014_3,6,9,12_sob_fal.yaml"
    checkpoint_path: list[str] | None = None
    seeds: list[str] | None = None
    output_dir: str | None = None
    year: int = 2012
    month: int = 9
    start_date: str = "2007-01"
    end_date: str = "2016-12"
    data_path: str = "data/era5"
    north_boundary: float = CLOSURE_NORTH_BOUNDARY
    residual_samples: int = 2000
    skip_cross: bool = False
    skip_input_anomaly_plots: bool = False
    overwrite_responses: bool = False

    def process_args(self):
        if self.seeds and self.checkpoint_path:
            self.error("--seeds and --checkpoint_path are mutually exclusive")


def collect_closure_timeseries_input_data(
    data_path: Path, start_date: str, end_date: str
) -> xr.Dataset:
    years = range(pd.Timestamp(start_date).year, pd.Timestamp(end_date).year + 1)
    return xr.open_mfdataset(
        generate_paths_yearly(data_path, years, make_era5_filename),
        combine="nested",
        concat_dim="date",
    ).sel(date=slice(start_date, end_date))


def water_vapor_kernel_components(
    ds_monthly,
    ds_qt,
    water_vapor_kernel,
) -> tuple[xr.DataArray, xr.DataArray]:
    ds_qt_monthly = to_monthly(ds_qt.copy(deep=True))
    ds_qt_anomaly = ds_qt_monthly - ds_qt_monthly.mean("year")

    Rv = 461.52
    Lv = 2260000
    q = ds_qt_monthly.q
    T = ds_qt_monthly.t
    dq = ds_qt_anomaly.q
    sp = ds_monthly.sp
    dT = (dq / q) * (Rv / Lv) * (T**2)

    dR_q_k = integrate_over_pressure_levels(sp, water_vapor_kernel.TOA_all * dT)
    dR_q_k_clr = integrate_over_pressure_levels(sp, water_vapor_kernel.TOA_clr * dT)

    return dR_q_k.compute(), dR_q_k_clr.compute()


def load_qt(years, months, kernel_grid) -> xr.Dataset:
    ds_qt = xr.load_dataset(QT_PATH)
    ds_qt = ds_qt.sel(date=ds_qt.date.dt.year.isin(years))
    ds_qt = ds_qt.sel(date=ds_qt.date.dt.month.isin(months))
    return ds_qt.interp(**kernel_grid)


def generated_perturbed_dataset(mean_ds, full_ds, var_groups):
    perturbed = []
    for var_group in var_groups:
        perturbed.append(mean_ds.assign({var: full_ds[var] for var in var_group}))
    perturbed.append(full_ds)
    return xr.concat(perturbed, dim="perturbation")


def perturbation_responses(
    mean_ds,
    full_ds,
    model,
    preprocessor,
    variable_config,
    var_groups,
    clear=False,
):
    if clear:
        perturbed = []
        for var_group in var_groups:
            ds_p = mean_ds.assign({var: full_ds[var] for var in var_group})
            perturbed.append(variable_config.clear_sky_input(ds_p))
        perturbed.append(variable_config.clear_sky_input(full_ds))
        perturbed = xr.concat(perturbed, dim="perturbation")
        original = variable_config.clear_sky_input(mean_ds)
    else:
        perturbed = generated_perturbed_dataset(mean_ds, full_ds, var_groups)
        original = mean_ds

    pred_perturbed = nn_pred(
        preprocessor.transform(perturbed),
        model,
        preprocessor,
        variable_config,
        ["perturbation", "year", "month", "latitude", "longitude"],
        clear=clear,
    )
    pred_original = nn_pred(
        preprocessor.transform(original),
        model,
        preprocessor,
        variable_config,
        ["month", "latitude", "longitude"],
        clear=clear,
    )
    diff = (pred_perturbed - pred_original) / SECONDS_PER_DAY
    return [diff.sel(perturbation=i) for i in range(len(diff.perturbation))]


def compute_responses(
    ds_monthly,
    ds_monthly_means,
    model,
    preprocessor,
    variable_config,
    cloud_vars,
):
    print(ds_monthly)
    K_a = xr.open_dataset(ALBEDO_KERNEL_PATH)
    K_q = xr.open_dataset(WATER_VAPOR_KERNEL_PATH)

    kernel_grid = {"latitude": K_a.latitude, "longitude": K_a.longitude}

    dR_a_nn, dR_q_nn, dR_c_nn, dR_nn_all = perturbation_responses(
        ds_monthly_means,
        ds_monthly,
        model,
        preprocessor,
        variable_config,
        [["fal"], ["tcwv"], cloud_vars],
    )
    dR_a_nn_clr, dR_q_nn_clr, dR_nn_clr = perturbation_responses(
        ds_monthly_means,
        ds_monthly,
        model,
        preprocessor,
        variable_config,
        [["fal"], ["tcwv"]],
        clear=True,
    )

    anomaly = with_flux_targets(ds_monthly) - with_flux_targets(ds_monthly_means)
    dR_clr = anomaly.tsrc.compute()
    dR = anomaly.tsr.compute()

    ds_qt = load_qt(
        sorted(int(year) for year in ds_monthly.year.values),
        sorted(int(month) for month in ds_monthly.month.values),
        kernel_grid,
    )
    dR_a_k, dR_a_k_clr = albedo_kernel_components(anomaly.fal, K_a)
    dR_q_k, dR_q_k_clr = water_vapor_kernel_components(ds_monthly, ds_qt, K_q)
    dR_c_k = (dR - dR_clr) - (dR_a_k - dR_a_k_clr) - (dR_q_k - dR_q_k_clr)

    response_attrs = {
        "dR_era5_all": {"plot_label": "ERA5 all", "filename": "dR_era5_all.png"},
        "dR_era5_clr": {"plot_label": "ERA5 clear", "filename": "dR_era5_clr.png"},
        "dR_nn_all": {"plot_label": "NN all", "filename": "dR_nn_all.png"},
        "dR_nn_clr": {"plot_label": "NN clr", "filename": "dR_nn_clr.png"},
        "dR_a_nn_all": {"plot_label": "a", "filename": "dR_a.png", "vmax": 40},
        "dR_c_nn_all": {"plot_label": "c", "filename": "dR_c.png", "vmax": 60},
        "dR_q_nn_all": {"plot_label": "q", "filename": "dR_q.png", "vmax": 7},
        "dR_a_nn_clr": {"plot_label": "a,clr", "filename": "dR_a,clr.png", "vmax": 60},
        "dR_q_nn_clr": {"plot_label": "q,clr", "filename": "dR_q,clr.png", "vmax": 4},
        "dR_a_k_all": {"plot_label": "a", "filename": "k_dR_a.png", "vmax": 40},
        "dR_c_k_all": {"plot_label": "c", "filename": "k_dR_c.png", "vmax": 60},
        "dR_q_k_all": {"plot_label": "q", "filename": "k_dR_q.png", "vmax": 7},
        "dR_a_k_clr": {"plot_label": "a,clr", "filename": "k_dR_a,clr.png", "vmax": 40},
        "dR_q_k_clr": {"plot_label": "q,clr", "filename": "k_dR_q,clr.png", "vmax": 7},
    }

    return xr.Dataset(
        {
            name: values.assign_attrs(**response_attrs[name])
            for name, values in {
                "dR_era5_all": dR,
                "dR_era5_clr": dR_clr,
                "dR_nn_all": dR_nn_all,
                "dR_nn_clr": dR_nn_clr,
                "dR_a_nn_all": dR_a_nn,
                "dR_c_nn_all": dR_c_nn,
                "dR_q_nn_all": dR_q_nn,
                "dR_a_nn_clr": dR_a_nn_clr,
                "dR_q_nn_clr": dR_q_nn_clr,
                "dR_a_k_all": dR_a_k,
                "dR_c_k_all": dR_c_k,
                "dR_q_k_all": dR_q_k,
                "dR_a_k_clr": dR_a_k_clr,
                "dR_q_k_clr": dR_q_k_clr,
            }.items()
        }
    )


def compute_model_responses(
    checkpoint_paths,
    config,
    ds_monthly,
    ds_monthly_means,
    variable_config,
    cloud_vars,
):
    responses = []
    epochs = []
    labels = []
    for checkpoint_path in checkpoint_paths:
        model, preprocessor, epoch = load_model_and_preprocessor(
            config, checkpoint_path, downscaling=False
        )
        responses.append(
            compute_responses(
                ds_monthly,
                ds_monthly_means,
                model,
                preprocessor,
                variable_config,
                cloud_vars,
            )
        )
        epochs.append(epoch)
        labels.append(model_label(checkpoint_path))

    if len(responses) == 1:
        return responses[0], epochs[0]
    return (
        xr.concat(responses, dim=xr.IndexVariable("model", labels)),
        f"models_{len(responses)}",
    )


def responses_for_plots(responses: xr.Dataset) -> xr.Dataset:
    return responses.mean("model") if "model" in responses.dims else responses


def compute_closure_eval(args):
    K_a = xr.open_dataset(ALBEDO_KERNEL_PATH)
    kernel_grid = {"latitude": K_a.latitude, "longitude": K_a.longitude}

    north_boundary = float(args.north_boundary)
    north_mask = K_a.latitude.values > north_boundary

    config = load_config(args.config_file)
    variable_config = variable_config_from_omegaconf(config)
    checkpoint_paths = checkpoint_paths_for_args(args, config)

    year, month = args.year, args.month
    epoch = "cached"
    if len(checkpoint_paths) == 1:
        _, preprocessor, epoch = load_model_and_preprocessor(
            config, checkpoint_paths[0], downscaling=False
        )
    else:
        preprocessor = None
        epoch = f"models_{len(checkpoint_paths)}"
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config.train.checkpoint_dir) / "figures" / str(epoch) / "closure_test"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    response_save_path = output_root / "saved_responses_closure_test.nc"

    data_path = Path(args.data_path)
    ds = collect_closure_timeseries_input_data(
        data_path, args.start_date, args.end_date
    ).interp(**kernel_grid)

    ds_monthly = to_monthly(ds.copy(deep=True))
    ds_monthly_means = ds_monthly.mean("year")
    anomaly = with_flux_targets(ds_monthly) - with_flux_targets(ds_monthly_means)

    cloud_vars = variable_config.clear_sky_zero_vars
    use_cache = response_save_path.exists() and not args.overwrite_responses
    if use_cache:
        print("=" * 20 + " loaded cached responses " + "=" * 20)
        responses = xr.load_dataset(response_save_path)
        if len(checkpoint_paths) > 1 and "model" not in responses.dims:
            raise ValueError(
                f"{response_save_path} has no model dimension; rerun with --overwrite_responses."
            )
        responses.attrs["north_boundary"] = north_boundary
    else:
        responses, epoch = compute_model_responses(
            checkpoint_paths,
            config,
            ds_monthly,
            ds_monthly_means,
            variable_config,
            cloud_vars,
        )
        responses.attrs["north_boundary"] = north_boundary
        responses.to_netcdf(response_save_path)

    return {
        "anomaly": anomaly,
        "ds_monthly": ds_monthly,
        "ds_monthly_means": ds_monthly_means,
        "preprocessor": preprocessor,
        "responses": responses,
        "plot_responses": responses_for_plots(responses),
        "output_root": output_root,
        "north_mask": north_mask,
        "north_boundary": north_boundary,
        "year": year,
        "month": month,
        "residual_samples": args.residual_samples,
        "skip_input_anomaly_plots": args.skip_input_anomaly_plots,
    }


def main():
    args = parse_args_and_confirm(
        ClosureEvalArgs(description="Run radiative closure over a date range.")
    )
    eval_data = compute_closure_eval(args)
    analysis.timeseries_test(
        eval_data["plot_responses"],
        eval_data["output_root"],
        eval_data["residual_samples"],
        eval_data["north_mask"],
        eval_data["north_boundary"],
    )
    dt2m = eval_data["ds_monthly"].t2m - eval_data["ds_monthly"].t2m.mean("year")
    analysis.feedback_test(dt2m, eval_data["plot_responses"], eval_data["output_root"])


if __name__ == "__main__":
    main()
