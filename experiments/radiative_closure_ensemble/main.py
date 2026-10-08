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
    EnsembleModel,
    albedo_kernel_components,
    parse_seed_ranges,
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


class ClosureEnsembleEvalArgs(Tap):
    config_file: str = "configs/model/fal/2011-2014_3,6,9,12_sob_fal.yaml"
    checkpoint_path: list | None = None
    seeds: list | None = None
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
    config,
    var_groups,
    clear=False,
):
    vc = variable_config_from_omegaconf(config)
    if clear:
        perturbed = []
        for var_group in var_groups:
            ds_p = mean_ds.assign({var: full_ds[var] for var in var_group})
            perturbed.append(vc.clear_sky_input(ds_p))
        perturbed.append(vc.clear_sky_input(full_ds))
        perturbed = xr.concat(perturbed, dim="perturbation")
        original = vc.clear_sky_input(mean_ds)
    else:
        perturbed = generated_perturbed_dataset(mean_ds, full_ds, var_groups)
        original = mean_ds

    pred_perturbed = nn_pred(
        preprocessor.transform(perturbed),
        model,
        preprocessor,
        vc,
        ["perturbation", "year", "month", "latitude", "longitude"],
        clear=clear,
    )
    pred_original = nn_pred(
        preprocessor.transform(original),
        model,
        preprocessor,
        vc,
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
    config,
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
        config,
        [["fal"], ["tcwv"], cloud_vars],
    )
    dR_a_nn_clr, dR_q_nn_clr, dR_nn_clr = perturbation_responses(
        ds_monthly_means,
        ds_monthly,
        model,
        preprocessor,
        config,
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


def compute_closure_ensemble_eval(args):
    K_a = xr.open_dataset(ALBEDO_KERNEL_PATH)
    kernel_grid = {"latitude": K_a.latitude, "longitude": K_a.longitude}

    north_boundary = float(args.north_boundary)
    analysis.NORTH_MASK = K_a.latitude.values > north_boundary

    config = load_config(args.config_file)
    if args.seeds:
        checkpoint_paths = [
            Path(f"{config.train.checkpoint_dir}_seed_{seed}") / "best_model.pt"
            for seed in parse_seed_ranges(args.seeds)
        ]
    elif args.checkpoint_path:
        checkpoint_paths = [Path(path) for path in args.checkpoint_path]
    else:
        checkpoint_paths = [Path(config.train.checkpoint_dir) / "best_model.pt"]

    models = []
    epochs = []
    for checkpoint_path in checkpoint_paths:
        model, preprocessor, epoch = load_model_and_preprocessor(
            config, checkpoint_path, downscaling=False
        )
        models.append(model)
        epochs.append(epoch)
    model = EnsembleModel(models).eval()
    epoch = epochs[0] if len(epochs) == 1 else f"ensemble_{len(models)}"

    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config.train.checkpoint_dir) / "figures" / str(epoch) / "closure_test"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    response_save_path = output_root / "saved_responses_closure_test.nc"

    data_path = Path(args.data_path)
    data_years = range(
        pd.Timestamp(args.start_date).year,
        pd.Timestamp(args.end_date).year + 1,
    )
    ds = xr.open_mfdataset(
        generate_paths_yearly(data_path, data_years, make_era5_filename),
        combine="nested",
        concat_dim="date",
    )
    ds = ds.sel(date=slice(args.start_date, args.end_date)).interp(**kernel_grid)

    ds_monthly = to_monthly(ds.copy(deep=True))
    ds_monthly_means = ds_monthly.mean("year")

    cloud_vars = ["hcc", "mcc", "lcc", "tciw", "tclw"]
    if config.preprocess.ecod.enabled and "ecod" in ds_monthly:
        cloud_vars.append("ecod")

    if response_save_path.exists() and not args.overwrite_responses:
        print("=" * 20 + " loaded cached responses " + "=" * 20)
        responses = xr.load_dataset(response_save_path)
        responses.attrs["north_boundary"] = north_boundary
    else:
        responses = compute_responses(
            ds_monthly,
            ds_monthly_means,
            model,
            preprocessor,
            config,
            cloud_vars,
        )
        responses.attrs["north_boundary"] = north_boundary
        responses.to_netcdf(response_save_path)

    return ds_monthly, responses, output_root


def main():
    args = ClosureEnsembleEvalArgs(
        description="Run closure-test analysis."
    ).parse_args()
    ds_monthly, responses, output_root = compute_closure_ensemble_eval(args)
    analysis.timeseries_test(
        responses,
        output_root,
        args.residual_samples,
    )
    dt2m = ds_monthly.t2m - ds_monthly.t2m.mean("year")
    analysis.feedback_test(
        dt2m,
        responses,
        output_root,
    )


if __name__ == "__main__":
    main()
