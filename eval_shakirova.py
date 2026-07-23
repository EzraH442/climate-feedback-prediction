import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from config_utils import load_config, variable_config_from_omegaconf
from utils_cartopy import (
    SECONDS_PER_DAY,
    global_mean,
    integrate_over_pressure_levels,
    nn_pred,
    plot_global_field,
    plot_north_pole_field,
    load_model_and_preprocessor,
    generate_paths_yearly,
    make_era5_filename,
    make_qt_filename,
)

NORTH_BOUNDARY = 70
NORTH_MASK = None
ALBEDO_KERNEL_PATH = Path("data/ERA5_kernels/ERA5_kernel_alb_TOA.nc")
WATER_VAPOR_KERNEL_PATH = Path(
    "data/ERA5_kernels/layer_specified_ta_wv_kernel/ERA5_kernel_wv_sw_nodp_TOA.nc"
)
MONTH_NAMES = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def select_date_state(ds: xr.Dataset, date: str) -> xr.Dataset:
    selected = ds.sel(date=date)
    if "date" in selected.dims:
        selected = selected.squeeze("date", drop=True)
    return selected


def select_kernel_month(ds: xr.Dataset, date: str) -> xr.Dataset:
    if "month" not in ds.dims:
        return ds
    return ds.sel(month=pd.Timestamp(date).month, drop=True)


def albedo_kernel_components(da, albedo_kernel) -> tuple[xr.DataArray, xr.DataArray]:
    dR_a_k = albedo_kernel.TOA_all * da * 100
    dR_a_k_clr = albedo_kernel.TOA_clr * da * 100
    return dR_a_k, dR_a_k_clr


def load_qt_pair(
    path: Path,
    base_date: str,
    perturbed_date: str,
    kernel_grid,
) -> xr.Dataset:
    years = sorted({pd.Timestamp(date).year for date in (base_date, perturbed_date)})
    ds_qt = xr.open_mfdataset(
        generate_paths_yearly(path, years, make_qt_filename),
        combine="nested",
        concat_dim="date",
    )
    return ds_qt.sel(date=[base_date, perturbed_date]).interp(**kernel_grid)


def water_vapor_kernel_components_pair(
    base,
    ds_qt_pair,
    base_date: str,
    perturbed_date: str,
    water_vapor_kernel,
) -> tuple[xr.DataArray, xr.DataArray]:
    qt_base = select_date_state(ds_qt_pair, base_date)
    qt_perturbed = select_date_state(ds_qt_pair, perturbed_date)

    Rv = 461.52
    Lv = 2260000
    q = qt_base.q
    T = qt_base.t
    dq = qt_perturbed.q - qt_base.q
    dT = (dq / q) * (Rv / Lv) * (T**2)

    dR_q_k = integrate_over_pressure_levels(base.sp, water_vapor_kernel.TOA_all * dT)
    dR_q_k_clr = integrate_over_pressure_levels(base.sp, water_vapor_kernel.TOA_clr * dT)
    return dR_q_k.compute(), dR_q_k_clr.compute()


def generated_perturbed_dataset(base, perturbed, var_groups):
    states = []
    for var_group in var_groups:
        states.append(base.assign({var: perturbed[var] for var in var_group}))
    states.append(perturbed)
    return xr.concat(states, dim="perturbation")


def perturbation_responses(
    perturbed,
    base,
    model,
    preprocessor,
    variable_config,
    var_groups,
    clear=False,
):
    if clear:
        states = []
        for var_group in var_groups:
            ds_p = base.assign({var: perturbed[var] for var in var_group})
            states.append(variable_config.clear_sky_input(ds_p))
        states.append(variable_config.clear_sky_input(perturbed))
        perturbed = xr.concat(states, dim="perturbation")
        original = variable_config.clear_sky_input(base)
    else:
        perturbed = generated_perturbed_dataset(base, perturbed, var_groups)
        original = base

    perturbed_processed = preprocessor.transform(perturbed)
    original_processed = preprocessor.transform(original)
    pred_perturbed = nn_pred(
        perturbed_processed,
        model,
        preprocessor,
        variable_config,
        ["perturbation", "latitude", "longitude"],
        clear=clear,
    )
    pred_original = nn_pred(
        original_processed,
        model,
        preprocessor,
        variable_config,
        ["latitude", "longitude"],
        clear=clear,
    )
    diff = (pred_perturbed - pred_original) / SECONDS_PER_DAY
    return [diff.sel(perturbation=i) for i in range(len(diff.perturbation))]


def plot_field_pair(
    field,
    title: str,
    save_path: Path,
    np_save_path: Path,
    vmax: float,
    label: str,
    np_vmax: float | None = None,
    ann_rmse=False,
) -> None:
    if NORTH_MASK is None:
        raise RuntimeError("NORTH_MASK must be initialized before plotting.")
    lon, lat = field.longitude, field.latitude
    field_masked = field.isel(latitude=NORTH_MASK)

    print(global_mean(field).values)
    mean = float(global_mean(field).values)
    mean_np = float(global_mean(field_masked).values)
    ann = f"{mean:.2f}"
    ann_np = f"{mean_np:.2f}"

    if ann_rmse:
        rmse_val = np.sqrt(float(global_mean(field * field).values))
        rmse_val_np = np.sqrt(float(global_mean(field_masked * field_masked).values))

        ann += f"; {rmse_val:.2f}"
        ann_np += f"; {rmse_val_np:.2f}"

    plot_global_field(
        field,
        lon,
        lat,
        title=title,
        save_path=save_path,
        vmin=-vmax,
        vmax=vmax,
        annotation=ann,
        label=label,
    )
    plot_north_pole_field(
        field,
        lon,
        lat,
        title=title,
        save_path=np_save_path,
        vmin=-(np_vmax if np_vmax is not None else vmax),
        vmax=np_vmax if np_vmax is not None else vmax,
        annotation=ann_np,
        label=label,
        boundary=NORTH_BOUNDARY,
        contours=True,
    )


def plot_input_anomalies(
    anomaly,
    anomaly_ecod,
    dR_clr,
    output_dir: Path,
) -> None:
    specs = [
        ("fal", "", 0.5),
        ("hcc", "", None),
        ("mcc", "", None),
        ("lcc", "", None),
        ("tcwv", "kg/m^2", None),
        ("tco3", "kg/m^2", None),
        ("tsr", "W/m^2", 24),
        ("tciw", "W/m^2", None),
        ("tclw", "W/m^2", None),
    ]
    anomaly_slice = anomaly.compute()
    for var, label, fixed_vmax in specs:
        field = anomaly_slice[var]
        vmax = float(fixed_vmax if fixed_vmax is not None else np.max(np.abs(field)))
        print(var, np.max(np.abs(field)).values)
        plot_field_pair(
            field,
            title=rf"$\Delta${var}",
            save_path=output_dir / f"delta_{var}.png",
            np_save_path=output_dir / f"delta_{var}_np.png",
            vmax=vmax,
            label=label,
        )

    for var, field, label in [
        ("ecod", anomaly_ecod, "ecod"),
        ("tsrc", dR_clr, "tsrc"),
    ]:
        vmax = float(np.max(np.abs(field)))
        plot_field_pair(
            field,
            title=rf"$\Delta${var}",
            save_path=output_dir / f"delta_{var}.png",
            np_save_path=output_dir / f"delta_{var}_np.png",
            vmax=vmax,
            label=label,
        )


def plot_response_dataset(
    dataset: xr.Dataset,
    output_dir: Path,
    source: str,
    ann_rmse=False,
) -> None:
    for name, response in dataset.data_vars.items():
        label = response.attrs.get("plot_label", name)
        filename = response.attrs["filename"]
        vmax = response.attrs.get("vmax")
        field = response
        if vmax is None:
            vmax = float(np.max(np.abs(field)))
        plot_field_pair(
            field,
            title=rf"$\Delta R_{{{label}}}^{{{source}}}$",
            save_path=output_dir / filename,
            np_save_path=output_dir / "np" / filename,
            vmax=vmax,
            label="W/m^2",
            np_vmax=8,
            ann_rmse=ann_rmse,
        )


def date_closure_test(
    anomaly: xr.Dataset,
    perturbed: xr.Dataset,
    base: xr.Dataset,
    preprocessor,
    responses: xr.Dataset,
    output_root: Path,
    dR_co3_nn=None,
    dR_co3_nn_clr=None,
    skip_input_anomaly_plots=False,
) -> None:
    if NORTH_MASK is None:
        raise RuntimeError("NORTH_MASK must be initialized before date_closure_test.")
    date_clear = output_root / "clr"
    date_all = output_root / "all"
    date_clear.mkdir(exist_ok=True, parents=True)
    date_all.mkdir(exist_ok=True, parents=True)
    (date_clear / "np").mkdir(exist_ok=True, parents=True)
    (date_all / "np").mkdir(exist_ok=True, parents=True)
    dR = responses["dR_era5_all"]
    dR_clr = responses["dR_era5_clr"]
    pp = preprocessor.transform(perturbed)
    ppm = preprocessor.transform(base)
    anomaly_ecod = (pp - ppm).ecod.compute()

    if not skip_input_anomaly_plots:
        plot_input_anomalies(
            anomaly,
            anomaly_ecod,
            dR_clr,
            output_root,
        )
    nn_responses_to_print = [
        ("dR_nn", responses["dR_nn_all"]),
        ("dR_nn_clr", responses["dR_nn_clr"]),
        ("dR_a_nn", responses["dR_a_nn_all"]),
        ("dR_c_nn", responses["dR_c_nn_all"]),
        ("dR_q_nn", responses["dR_q_nn_all"]),
        ("dR_a_nn_clr", responses["dR_a_nn_clr"]),
        ("dR_q_nn_clr", responses["dR_q_nn_clr"]),
    ]
    for name, response in nn_responses_to_print:
        print(name, np.max(np.abs(response)).values)

    plot_response_dataset(
        responses[["dR_a_nn_all", "dR_c_nn_all", "dR_q_nn_all"]].rename(
            {
                "dR_a_nn_all": "a",
                "dR_c_nn_all": "c",
                "dR_q_nn_all": "q",
            }
        ),
        date_all,
        "NN",
    )
    plot_response_dataset(
        responses[["dR_a_nn_clr", "dR_q_nn_clr"]].rename(
            {"dR_a_nn_clr": "a", "dR_q_nn_clr": "q"}
        ),
        date_clear,
        "NN",
    )

    for name, response in [
        ("dR_a_k", responses["dR_a_k_all"]),
        ("dR_c_k", responses["dR_c_k_all"]),
        ("dR_q_k", responses["dR_q_k_all"]),
        ("dR_a_k_clr", responses["dR_a_k_clr"]),
        ("dR_q_k_clr", responses["dR_q_k_clr"]),
    ]:
        print(name, np.max(np.abs(response)).values)

    plot_response_dataset(
        responses[["dR_a_k_all", "dR_q_k_all", "dR_c_k_all"]].rename(
            {"dR_a_k_all": "a", "dR_q_k_all": "q", "dR_c_k_all": "c"}
        ),
        date_all,
        "K",
    )
    plot_response_dataset(
        responses[["dR_a_k_clr", "dR_q_k_clr"]].rename(
            {"dR_a_k_clr": "a", "dR_q_k_clr": "q"}
        ),
        date_clear,
        "K",
    )

    dR_sum_nn = (
        responses["dR_a_nn_all"] + responses["dR_c_nn_all"] + responses["dR_q_nn_all"]
    )
    dR_sum_nn_allcross = responses["dR_nn_all"]
    dR_sum_k = (
        responses["dR_a_k_all"] + responses["dR_c_k_all"] + responses["dR_q_k_all"]
    )
    dR_sum_nn_clr = responses["dR_a_nn_clr"] + responses["dR_q_nn_clr"]
    dR_sum_nn_allcross_clr = responses["dR_nn_clr"]
    dR_sum_k_clr = responses["dR_a_k_clr"] + responses["dR_q_k_clr"]

    dR_res_nn_allcross = dR - dR_sum_nn_allcross
    dR_res_nn = dR - dR_sum_nn
    dR_res_k = dR - dR_sum_k
    dR_res_nn_clr = dR_clr - dR_sum_nn_clr
    dR_res_nn_allcross_clr = dR_clr - dR_sum_nn_allcross_clr
    dR_res_k_clr = dR_clr - dR_sum_k_clr

    nn_all_closure = xr.Dataset(
        {
            "sum": dR_sum_nn.assign_attrs(
                plot_label="sum", filename="dR_sum.png", vmax=55
            ),
            "sum_allcross": dR_sum_nn_allcross.assign_attrs(
                plot_label="sum,allcross", filename="dR_sum_allcross.png", vmax=55
            ),
            "res": dR_res_nn.assign_attrs(
                plot_label="res", filename="dR_res.png", vmax=8
            ),
            "res_allcross": dR_res_nn_allcross.assign_attrs(
                plot_label="res,allcross", filename="dR_res_allcross.png", vmax=8
            ),
        }
    )

    kernel_all_closure = xr.Dataset(
        {
            "sum": dR_sum_k.assign_attrs(
                plot_label="sum", filename="k_dR_sum.png", vmax=55
            ),
            "res": dR_res_k.assign_attrs(
                plot_label="res", filename="k_dR_res.png", vmax=8
            ),
        }
    )

    nn_clr_closure = xr.Dataset(
        {
            "sum_clr": dR_sum_nn_clr.assign_attrs(
                plot_label="sum,clr", filename="dR_sum,clr.png", vmax=55
            ),
            "sum_allcross_clr": dR_sum_nn_allcross_clr.assign_attrs(
                plot_label="sum,allcross,clr",
                filename="dR_sum_allcross,clr.png",
                vmax=55,
            ),
            "res_clr": dR_res_nn_clr.assign_attrs(
                plot_label="res,clr", filename="dR_res,clr.png", vmax=8
            ),
            "res_allcross_clr": dR_res_nn_allcross_clr.assign_attrs(
                plot_label="res,allcross,clr",
                filename="dR_res_allcross,clr.png",
                vmax=8,
            ),
        }
    )

    kernel_clr_closure = xr.Dataset(
        {
            "sum_clr": dR_sum_k_clr.assign_attrs(
                plot_label="sum,clr", filename="k_dR_sum,clr.png", vmax=60
            ),
            "res_clr": dR_res_k_clr.assign_attrs(
                plot_label="res,clr", filename="k_dR_res,clr.png", vmax=8
            ),
        }
    )

    plot_response_dataset(nn_all_closure, date_all, "NN", ann_rmse=True)
    plot_response_dataset(kernel_all_closure, date_all, "K", ann_rmse=True)
    plot_response_dataset(nn_clr_closure, date_clear, "NN", ann_rmse=True)
    plot_response_dataset(
        kernel_clr_closure, date_clear, "K", ann_rmse=True
    )

    cross_vars = ["dR_aq_nn_all", "dR_ac_nn_all", "dR_qc_nn_all"]
    if not all(name in responses for name in cross_vars):
        return

    plot_response_dataset(
        responses[cross_vars].rename(
            {
                "dR_aq_nn_all": "a,q",
                "dR_ac_nn_all": "a,c",
                "dR_qc_nn_all": "q,c",
            }
        ),
        date_all,
        "NN",
    )

    dR_aq_nn = responses["dR_aq_nn_all"]
    dR_ac_nn = responses["dR_ac_nn_all"]
    dR_qc_nn = responses["dR_qc_nn_all"]
    dR_sum_nn_cross = dR_sum_nn + dR_aq_nn + dR_ac_nn + dR_qc_nn
    dR_res_nn_cross = dR - dR_sum_nn_cross

    cross_closure = xr.Dataset(
        {
            "sum": dR_sum_nn_cross.assign_attrs(
                plot_label="sum,cross", filename="cross_dR_sum.png", vmax=55
            ),
            "res": dR_res_nn_cross.assign_attrs(
                plot_label="res,cross", filename="cross_dR_res.png", vmax=55
            ),
        }
    )
    plot_response_dataset(
        cross_closure,
        date_all,
        "NN",
    )


def compute_responses(
    perturbed,
    base,
    model,
    preprocessor,
    variable_config,
    base_date: str,
    perturbed_date: str,
    data_path: Path,
):
    K_a = select_kernel_month(xr.open_dataset(ALBEDO_KERNEL_PATH), perturbed_date)
    K_q = select_kernel_month(xr.open_dataset(WATER_VAPOR_KERNEL_PATH), perturbed_date)

    kernel_grid = {"latitude": K_a.latitude, "longitude": K_a.longitude}

    dR_a_nn, dR_q_nn, dR_c_nn, dR_nn_all = perturbation_responses(
        perturbed,
        base,
        model,
        preprocessor,
        variable_config,
        [["fal"], ["tcwv"], variable_config.clear_sky_zero_vars],
    )
    dR_a_nn_clr, dR_q_nn_clr, dR_nn_clr = perturbation_responses(
        perturbed,
        base,
        model,
        preprocessor,
        variable_config,
        [["fal"], ["tcwv"]],
        clear=True,
    )

    anomaly = perturbed - base
    dR_clr = anomaly.tsrc.compute()
    dR = anomaly.tsr.compute()

    ds_qt = load_qt_pair(data_path, base_date, perturbed_date, kernel_grid)
    dR_a_k, dR_a_k_clr = albedo_kernel_components(anomaly.fal, K_a)
    dR_q_k, dR_q_k_clr = water_vapor_kernel_components_pair(
        base, ds_qt, base_date, perturbed_date, K_q
    )
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

    base_responses = xr.Dataset(
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
    return base_responses


def main():
    global NORTH_BOUNDARY, NORTH_MASK
    K_a = xr.open_dataset(ALBEDO_KERNEL_PATH)
    kernel_grid = {"latitude": K_a.latitude, "longitude": K_a.longitude}

    parser = argparse.ArgumentParser(description="Run closure-test analysis.")
    parser.add_argument(
        "--config_file",
        default="configs/model/fal/2011-2014_3,6,9,12_sob_fal_clear.yaml",
        help="Path to OmegaConf YAML config.",
    )
    parser.add_argument("--checkpoint_path", help="Path to model checkpoint.")
    parser.add_argument("--output_dir", help="Output directory.")
    parser.add_argument("--base_date", default="1992-09")
    parser.add_argument("--perturbed_date", default="2012-09")
    parser.add_argument("--data_path", default="data/era5")
    parser.add_argument("--north_boundary", type=float, default=NORTH_BOUNDARY)
    parser.add_argument("--skip_input_anomaly_plots", action="store_true")
    parser.add_argument("--overwrite_responses", action="store_true")
    args = parser.parse_args()

    # --- parse args ---
    d1, d2 = args.base_date, args.perturbed_date

    #  --- setup global vars ---
    NORTH_BOUNDARY = args.north_boundary
    NORTH_MASK = K_a.latitude.values > NORTH_BOUNDARY

    #  --- load model ---
    config = load_config(args.config_file)
    variable_config = variable_config_from_omegaconf(config)
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else Path(config.train.checkpoint_dir) / "best_model.pt"
    )
    model, preprocessor, epoch = load_model_and_preprocessor(
        config, checkpoint_path, downscaling=False
    )

    # --- setup ouputs dir ---
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config.train.checkpoint_dir)
        / "figures"
        / str(epoch)
        / "shakirova_test"
        / f"{d2} - {d1}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    response_save_path = output_root / "saved_responses.nc"

    # -- load necessary data
    data_path = Path(args.data_path)
    dates = [d1, d2]
    data_years = sorted({pd.Timestamp(d).year for d in dates})
    ds = xr.open_mfdataset(
        generate_paths_yearly(data_path, data_years, make_era5_filename),
        combine="nested",
        concat_dim="date",
    )
    ds["tsr"] = ds.tsr / SECONDS_PER_DAY
    ds["tsrc"] = ds.tsrc / SECONDS_PER_DAY

    base = select_date_state(ds, d1).interp(**kernel_grid)
    perturbed = select_date_state(ds, d2).interp(**kernel_grid)
    anomaly = perturbed - base

    if response_save_path.exists() and not args.overwrite_responses:
        print("=" * 20 + " loaded cached responses " + "=" * 20)
        responses = xr.load_dataset(response_save_path)

    else:
        responses = compute_responses(
            perturbed,
            base,
            model,
            preprocessor,
            variable_config,
            d1,
            d2,
            data_path,
        )
        responses.to_netcdf(response_save_path)
    date_closure_test(
        anomaly,
        perturbed,
        base,
        preprocessor,
        responses,
        output_root,
        skip_input_anomaly_plots=args.skip_input_anomaly_plots,
    )


if __name__ == "__main__":
    main()
