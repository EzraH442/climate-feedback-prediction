from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from config_utils import load_config, variable_config_from_omegaconf
from constants import (
    ABLATION_SPECS,
    ALBEDO_KERNEL_PATH,
    SHAKIROVA_NORTH_BOUNDARY,
    WATER_VAPOR_KERNEL_PATH,
)
from experiments.common import albedo_kernel_components, with_flux_targets
from utils import (
    SECONDS_PER_DAY,
    add_bottom_colorbar,
    add_right_colorbar,
    annotate_response,
    arctic_field,
    compute_nn_kernel,
    compute_nn_kernel_autograd,
    generate_paths_yearly,
    integrate_over_pressure_levels,
    kernel_delta,
    load_model_and_preprocessor,
    make_era5_filename,
    make_kernel_filename,
    make_qt_filename,
    nn_pred,
    panel_title,
    plot_input_anomalies,
    plot_response_dataset,
    plot_shakirova_field,
    plot_text_on_ax,
)

NORTH_BOUNDARY = SHAKIROVA_NORTH_BOUNDARY
NORTH_MASK = None


def select_date_state(ds: xr.Dataset, date: str) -> xr.Dataset:
    selected = ds.sel(date=date)
    if "date" in selected.dims:
        selected = selected.squeeze("date", drop=True)
    return selected


def select_kernel_month(ds: xr.Dataset, date: str) -> xr.Dataset:
    return ds.sel(month=pd.Timestamp(date).month - 1, drop=True)


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
    dR_q_k_clr = integrate_over_pressure_levels(
        base.sp, water_vapor_kernel.TOA_clr * dT
    )
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
            north_mask=NORTH_MASK,
            north_boundary=NORTH_BOUNDARY,
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
        source="NN",
        north_mask=NORTH_MASK,
        north_boundary=NORTH_BOUNDARY,
    )
    plot_response_dataset(
        responses[["dR_a_nn_clr", "dR_q_nn_clr"]].rename(
            {"dR_a_nn_clr": "a", "dR_q_nn_clr": "q"}
        ),
        date_clear,
        source="NN",
        north_mask=NORTH_MASK,
        north_boundary=NORTH_BOUNDARY,
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
        source="K",
        north_mask=NORTH_MASK,
        north_boundary=NORTH_BOUNDARY,
    )
    plot_response_dataset(
        responses[["dR_a_k_clr", "dR_q_k_clr"]].rename(
            {"dR_a_k_clr": "a", "dR_q_k_clr": "q"}
        ),
        date_clear,
        source="K",
        north_mask=NORTH_MASK,
        north_boundary=NORTH_BOUNDARY,
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

    plot_response_dataset(
        nn_all_closure,
        date_all,
        source="NN",
        north_mask=NORTH_MASK,
        north_boundary=NORTH_BOUNDARY,
        ann_rmse=True,
    )
    plot_response_dataset(
        kernel_all_closure,
        date_all,
        source="K",
        north_mask=NORTH_MASK,
        north_boundary=NORTH_BOUNDARY,
        ann_rmse=True,
    )
    plot_response_dataset(
        nn_clr_closure,
        date_clear,
        source="NN",
        north_mask=NORTH_MASK,
        north_boundary=NORTH_BOUNDARY,
        ann_rmse=True,
    )
    plot_response_dataset(
        kernel_clr_closure,
        date_clear,
        source="K",
        north_mask=NORTH_MASK,
        north_boundary=NORTH_BOUNDARY,
        ann_rmse=True,
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
        source="NN",
        north_mask=NORTH_MASK,
        north_boundary=NORTH_BOUNDARY,
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
        source="NN",
        north_mask=NORTH_MASK,
        north_boundary=NORTH_BOUNDARY,
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

    base_flux = with_flux_targets(base)
    perturbed_flux = with_flux_targets(perturbed)
    anomaly = perturbed_flux - base_flux
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


def compute_shakirova_eval(args):
    global NORTH_BOUNDARY, NORTH_MASK
    K_a = xr.open_dataset(ALBEDO_KERNEL_PATH)
    kernel_grid = {"latitude": K_a.latitude, "longitude": K_a.longitude}

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

    base = select_date_state(ds, d1).interp(**kernel_grid)
    perturbed = select_date_state(ds, d2).interp(**kernel_grid)
    anomaly = with_flux_targets(perturbed) - with_flux_targets(base)

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
    return {
        "anomaly": anomaly,
        "perturbed": perturbed,
        "base": base,
        "preprocessor": preprocessor,
        "responses": responses,
        "output_root": output_root,
        "skip_input_anomaly_plots": args.skip_input_anomaly_plots,
    }


def plot_shakirova_eval(eval_data):
    date_closure_test(
        eval_data["anomaly"],
        eval_data["perturbed"],
        eval_data["base"],
        eval_data["preprocessor"],
        eval_data["responses"],
        eval_data["output_root"],
        skip_input_anomaly_plots=eval_data["skip_input_anomaly_plots"],
    )


def run_shakirova_eval(args):
    print("=" * 20 + " computation " + "=" * 20)
    eval_data = compute_shakirova_eval(args)
    print("=" * 20 + " plotting " + "=" * 20)
    plot_shakirova_eval(eval_data)


# --------------------------------------------------------------------------- #
# NeurIPS/Shakirova figure helpers
# --------------------------------------------------------------------------- #


# %%
def zero_like_response(field):
    return xr.zeros_like(field)


def add_fields(fields):
    total = fields[0]
    for field in fields[1:]:
        total = total + field
    return total


def shakirova_terms(responses, sky):
    suffix = "all" if sky == "all" else "clr"
    target = responses[f"dR_era5_{suffix}"]
    if sky == "all":
        kernel_parts = [
            responses["dR_a_k_all"],
            responses["dR_q_k_all"],
            responses["dR_c_k_all"],
        ]
        nn_parts = [
            responses["dR_a_nn_all"],
            responses["dR_q_nn_all"],
            responses["dR_c_nn_all"],
        ]
        nn_allcross = responses["dR_nn_all"]
    else:
        kernel_parts = [responses["dR_a_k_clr"], responses["dR_q_k_clr"]]
        nn_parts = [responses["dR_a_nn_clr"], responses["dR_q_nn_clr"]]
        nn_allcross = responses["dR_nn_clr"]

    kernel_sum = add_fields(kernel_parts)
    nn_sum = add_fields(nn_parts)
    return {
        "target": target,
        "kernel": [*kernel_parts, kernel_sum, target - kernel_sum],
        "nn": [*nn_parts, nn_sum, target - nn_sum],
        "allcross": [nn_allcross, target - nn_allcross],
    }


def sum_response_expr(source, sky):
    suffix = "K" if source == "kernel" else "NN"
    subscript = "a+c+q" if sky == "all" else "a+q"
    return rf"\Delta R_{{{subscript}}}^{{{suffix}}}"


def sum_response_title(source, sky):
    return "$" + sum_response_expr(source, sky) + "$"


def plot_feedback_total_response(sky, arctic=False):
    terms = shakirova_terms(shakirova_responses, sky)
    rows = [
        (terms["target"], None),
        (terms["kernel"][-2], terms["kernel"][-1]),
        (terms["nn"][-2], terms["nn"][-1]),
        (terms["allcross"][0], terms["allcross"][1]),
    ]

    figsize = (11.69, 7) if arctic else (11.69, 3.75)
    fig, ax = plt.subplots(
        2,
        4,
        figsize=figsize,
        subplot_kw={
            "projection": (
                ccrs.NorthPolarStereo(central_longitude=0)
                if arctic
                else ccrs.PlateCarree(central_longitude=180)
            )
        },
        layout="constrained",
    )
    fig.get_layout_engine().set(
        rect=[0.06, 0.08, 1, 0.94], h_pad=0.01, w_pad=0.01, hspace=0.01, wspace=0.01
    )

    response_im = residual_im = None
    response_titles = [
        r"$\Delta R$",
        sum_response_title("kernel", sky),
        sum_response_title("nn", sky),
        r"$\Delta R_{allcross}^{NN}$",
    ]
    residual_titles = [
        "",
        r"(a) - (b)",
        r"(a) - (c)",
        r"(a) - (d)",
    ]
    response_labels = list("abcd")
    residual_labels = ["", "e", "f", "g"]
    for col, (response, residual) in enumerate(rows):
        response_im = plot_shakirova_field(
            ax[0][col],
            response,
            panel_title(response_labels[col], response_titles[col]),
            vmin=-60,
            vmax=60,
            step=5,
            left_labels=(col == 0),
            bottom_labels=(col == 0),
            arctic=arctic,
            # titlepad=24,
        )
        annotate_response(ax[0][col], response, arctic=arctic)
        if residual is None:
            ax[1][col].set_visible(False)
            continue
        residual_im = plot_shakirova_field(
            ax[1][col],
            residual,
            panel_title(residual_labels[col], residual_titles[col]),
            vmin=-8,
            vmax=8,
            step=1,
            left_labels=(col == 1),
            bottom_labels=True,
            arctic=arctic,
            # titlepad=24,
        )
        annotate_response(ax[1][col], residual, arctic=arctic)

    fig.canvas.draw()
    # def add_right_colorbar(fig, im, axes_list, label, gap=0.01, width=0.035):
    add_right_colorbar(
        fig,
        response_im,
        [ax[0][col] for col in range(4)],
        "W m$^{-2}$",
    )
    add_right_colorbar(
        fig, residual_im, [ax[1][col] for col in range(1, 4)], "W m$^{-2}$"
    )
    fig.set_layout_engine("none")
    region = "_arctic" if arctic else ""
    fig.savefig(
        shakirova_output_dir / f"shakirova_{sky}_total{region}.png", bbox_inches="tight"
    )
    return fig


def plot_feedback_component_responses(sky, arctic=False):
    terms = shakirova_terms(shakirova_responses, sky)
    if sky == "all":
        inputs = [
            (shakirova_anomaly.fal, r"$\Delta a$", "a", -0.5, 0.5, 0.05, ""),
            (shakirova_anomaly.tcwv, r"$\Delta q$", "q", -20, 20, 2, "kg m$^{-2}$"),
            (shakirova_anomaly_ecod, r"$\Delta c$", "c", -1, 1, 0.1, ""),
            (terms["target"], r"$\Delta R$", "sum", -60, 60, 5, "W m$^{-2}$"),
        ]
    else:
        inputs = [
            (shakirova_anomaly.fal, r"$\Delta a$", "a", -0.5, 0.5, 0.05, ""),
            (shakirova_anomaly.tcwv, r"$\Delta q$", "q", -20, 20, 2, "kg m$^{-2}$"),
            (terms["target"], r"$\Delta R$", "sum", -60, 60, 5, "W m$^{-2}$"),
        ]

    if arctic:
        figsize = (11.69, 10.75) if sky == "all" else (8.27, 9.5)
    else:
        figsize = (11.69, 6) if sky == "all" else (8.27, 6)

    fig, ax = plt.subplots(
        3,
        len(inputs),
        figsize=figsize,
        subplot_kw={
            "projection": (
                ccrs.NorthPolarStereo(central_longitude=0)
                if arctic
                else ccrs.PlateCarree(central_longitude=180)
            )
        },
        layout="constrained",
    )
    fig.get_layout_engine().set(
        rect=[0.06, 0.09, 1, 0.94], h_pad=0.01, w_pad=0.01, hspace=0.01, wspace=0.01
    )

    response_im = None
    panel_labels = [
        list("abcdefghijkl"[row * len(inputs) : (row + 1) * len(inputs)])
        for row in range(3)
    ]
    for col, (
        input_field,
        input_title,
        response_part,
        vmin,
        vmax,
        step,
        cbar_label,
    ) in enumerate(inputs):
        im = plot_shakirova_field(
            ax[0][col],
            input_field,
            panel_title(panel_labels[0][col], input_title),
            vmin=vmin,
            vmax=vmax,
            step=step,
            left_labels=(col == 0),
            bottom_labels=False,
            arctic=arctic,
        )
        plot_text_on_ax(
            ax[0][col],
            f"{response_mean(input_field, arctic=arctic):.2f}"
            + (f" {cbar_label}" if cbar_label else ""),
        )
        fig.colorbar(im, ax=ax[0][col], orientation="horizontal", label=cbar_label)

        for row, source in enumerate(["kernel", "nn"], start=1):
            field = terms[source][col]
            title = (
                sum_response_title(source, sky)
                if response_part == "sum"
                else rf"$\Delta R_{{{response_part}}}^{{{'K' if source == 'kernel' else 'NN'}}}$"
            )
            response_im = plot_shakirova_field(
                ax[row][col],
                field,
                panel_title(panel_labels[row][col], title),
                vmin=-60,
                vmax=60,
                step=5,
                left_labels=(col == 0),
                bottom_labels=(row == 2),
                arctic=arctic,
            )
            annotate_response(ax[row][col], field, arctic=arctic)

    fig.canvas.draw()
    add_bottom_colorbar(
        fig,
        response_im,
        [ax[row][col] for row in (1, 2) for col in range(len(inputs))],
        "W m$^{-2}$",
    )
    fig.set_layout_engine("none")
    region = "_arctic" if arctic else ""
    fig.savefig(
        shakirova_output_dir / f"shakirova_{sky}_components{region}.png",
        bbox_inches="tight",
    )
    return fig


def setup_neurips_feedback_context(
    config_path: Path = Path(
        "configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky.yaml"
    ),
    checkpoint_path: Path = Path(
        "checkpoints/fal/2011-2014_3,6,9,12_sob_fal_clearsky/best_model.pt"
    ),
    output_dir: Path = Path("FiguresNeurIPS"),
    base_date: str = "1992-09",
    perturbed_date: str = "2020-09",
    response_cache_name: str = "saved_responses_no_downscale.nc",
):
    global config, vc, model, preprocessor
    global shakirova_output_dir, shakirova_base_date, shakirova_perturbed_date
    global shakirova_grid, shakirova_ds, shakirova_base, shakirova_perturbed
    global shakirova_anomaly, shakirova_anomaly_ecod, shakirova_responses

    config = load_config(config_path)
    vc = variable_config_from_omegaconf(config)
    model, preprocessor, _ = load_model_and_preprocessor(
        config, checkpoint_path, downscaling=False
    )

    years = sorted({pd.Timestamp(base_date).year, pd.Timestamp(perturbed_date).year})
    raw_paths = generate_paths_yearly(
        config.dataset.era5.raw_path, years, make_era5_filename
    )
    raw = xr.open_mfdataset(raw_paths, combine="nested", concat_dim="date")

    shakirova_output_dir = output_dir
    shakirova_output_dir.mkdir(exist_ok=True, parents=True)
    shakirova_base_date = base_date
    shakirova_perturbed_date = perturbed_date
    response_save_path = shakirova_output_dir / response_cache_name

    K_a = xr.open_dataset(ALBEDO_KERNEL_PATH).isel(
        month=pd.Timestamp(base_date).month - 1
    )
    shakirova_grid = {"latitude": K_a.latitude, "longitude": K_a.longitude}
    shakirova_ds = raw.interp(**shakirova_grid)
    shakirova_base = select_date_state(shakirova_ds, shakirova_base_date)
    shakirova_perturbed = select_date_state(shakirova_ds, shakirova_perturbed_date)
    shakirova_anomaly = shakirova_perturbed - shakirova_base
    shakirova_anomaly_ecod = (
        preprocessor.transform(shakirova_perturbed)
        - preprocessor.transform(shakirova_base)
    ).ecod.compute()

    if response_save_path.exists():
        shakirova_responses = xr.load_dataset(response_save_path)
    else:
        shakirova_responses = compute_responses(
            shakirova_perturbed,
            shakirova_base,
            model,
            preprocessor,
            vc,
            shakirova_base_date,
            shakirova_perturbed_date,
            Path(config.dataset.era5.raw_path),
        )
        shakirova_responses.to_netcdf(response_save_path)
    return shakirova_responses


def load_feedback_ablation_responses(label, config_path, checkpoint_path, source):
    if source == "kernel":
        return shakirova_responses

    ablation_config = load_config(config_path)
    ablation_vc = variable_config_from_omegaconf(ablation_config)
    checkpoint_path = (
        checkpoint_path or Path(ablation_config.train.checkpoint_dir) / "best_model.pt"
    )
    cache_path = (
        shakirova_output_dir / f"ablation_responses_{checkpoint_path.parent.name}.nc"
    )
    if cache_path.exists():
        return xr.load_dataset(cache_path)

    ablation_model, ablation_preprocessor, _ = load_model_and_preprocessor(
        ablation_config, checkpoint_path, downscaling=False
    )
    responses = compute_responses(
        shakirova_perturbed,
        shakirova_base,
        ablation_model,
        ablation_preprocessor,
        ablation_vc,
        shakirova_base_date,
        shakirova_perturbed_date,
        Path(ablation_config.dataset.era5.raw_path),
    )
    responses.to_netcdf(cache_path)
    return responses


def feedback_ablation_residual_fields(responses, source):
    all_terms = shakirova_terms(responses, "all")
    clr_terms = shakirova_terms(responses, "clr")
    if source == "kernel":
        return [all_terms["kernel"][-1], clr_terms["kernel"][-1], None, None]
    return [
        all_terms["nn"][-1],
        clr_terms["nn"][-1],
        all_terms["allcross"][-1],
        clr_terms["allcross"][-1],
    ]


def plot_feedback_ablation_residuals(arctic=True):
    fig, ax = plt.subplots(
        len(ABLATION_SPECS),
        4,
        figsize=(11.69, 13),
        subplot_kw={
            "projection": (
                ccrs.NorthPolarStereo(central_longitude=0)
                if arctic
                else ccrs.PlateCarree(central_longitude=180)
            )
        },
        layout="constrained",
    )
    fig.get_layout_engine().set(
        rect=[0.08, 0.08, 1, 0.94], h_pad=0.01, w_pad=0.01, hspace=0.01, wspace=0.01
    )
    column_titles = [
        "All-sky sum",
        "Clear-sky sum",
        "All-sky allcross",
        "Clear-sky allcross",
    ]
    labels = iter("abcdefghijklmn")
    residual_im = None
    for row, (label, config_path, checkpoint_path, source) in enumerate(ABLATION_SPECS):
        responses = load_feedback_ablation_responses(
            label, config_path, checkpoint_path, source
        )
        for col, field in enumerate(
            feedback_ablation_residual_fields(responses, source)
        ):
            if field is None:
                ax[row][col].set_visible(False)
                continue
            title = (
                column_titles[col]
                if not any(ax[prev][col].get_visible() for prev in range(row))
                else ""
            )
            residual_im = plot_shakirova_field(
                ax[row][col],
                field,
                panel_title(next(labels), title),
                vmin=-8,
                vmax=8,
                step=1,
                left_labels=(col == 0),
                bottom_labels=(row == len(ABLATION_SPECS) - 1),
                arctic=arctic,
            )
            annotate_response(ax[row][col], field, residual=False, arctic=arctic)
        ax[row][0].text(
            -0.08,
            0.5,
            label,
            transform=ax[row][0].transAxes,
            ha="right",
            va="center",
            rotation=90,
        )

    add_bottom_colorbar(
        fig,
        residual_im,
        [
            ax[row][col]
            for row in range(len(ABLATION_SPECS))
            for col in range(4)
            if ax[row][col].get_visible()
        ],
        "W m$^{-2}$",
    )
    fig.set_layout_engine("none")
    region = "_arctic" if arctic else ""
    fig.savefig(
        shakirova_output_dir
        / f"shakirova_feedback_ablation_residual_fields{region}.png",
        bbox_inches="tight",
    )
    return fig


def coupling_slug(label):
    return "".join(ch if ch.isalnum() else "_" for ch in str(label).lower()).strip("_")


def load_albedo_cloud_model(config_path, checkpoint_path):
    coupling_config = load_config(config_path)
    coupling_vc = variable_config_from_omegaconf(coupling_config)
    checkpoint_path = (
        checkpoint_path or Path(coupling_config.train.checkpoint_dir) / "best_model.pt"
    )
    coupling_model, coupling_preprocessor, _ = load_model_and_preprocessor(
        coupling_config, checkpoint_path, downscaling=False
    )
    return coupling_model, coupling_preprocessor, coupling_vc


def load_sensitivity_model(config_path, checkpoint_path=None):
    sensitivity_config = load_config(config_path)
    sensitivity_vc = variable_config_from_omegaconf(sensitivity_config)
    checkpoint_path = (
        checkpoint_path
        if checkpoint_path is not None
        else Path(sensitivity_config.train.checkpoint_dir) / "best_model.pt"
    )
    sensitivity_model, sensitivity_preprocessor, _ = load_model_and_preprocessor(
        sensitivity_config, checkpoint_path, downscaling=False
    )
    return (
        sensitivity_config,
        sensitivity_vc,
        sensitivity_model,
        sensitivity_preprocessor,
    )


def compute_albedo_cloud_coupling(label, config_path, checkpoint_path):
    cache_path = (
        shakirova_output_dir
        / f"{coupling_slug(label)}_compute_albedo_cloud_coupling.nc"
    )
    if cache_path.exists():
        return xr.load_dataarray(cache_path)

    coupling_model, coupling_preprocessor, coupling_vc = load_albedo_cloud_model(
        config_path, checkpoint_path
    )
    cloud_vars = [
        var for var in coupling_vc.clear_sky_zero_vars if var in shakirova_base
    ]
    states = xr.concat(
        [
            shakirova_base,
            shakirova_base.assign(fal=shakirova_perturbed.fal),
            shakirova_base.assign(
                {var: shakirova_perturbed[var] for var in cloud_vars}
            ),
            shakirova_base.assign(
                {
                    "fal": shakirova_perturbed.fal,
                    **{var: shakirova_perturbed[var] for var in cloud_vars},
                }
            ),
        ],
        dim="perturbation",
    )
    response = predict_flux_for_states(
        states,
        ["perturbation", "latitude", "longitude"],
        coupling_model,
        coupling_preprocessor,
        coupling_vc,
    )
    coupling = (
        response.sel(perturbation=3)
        - response.sel(perturbation=1)
        - response.sel(perturbation=2)
        + response.sel(perturbation=0)
    )
    coupling.to_netcdf(cache_path)
    return coupling


def predict_flux_for_states(
    states, dim_names, coupling_model, coupling_preprocessor, coupling_vc
):
    processed = coupling_preprocessor.transform(states)
    return (
        nn_pred(
            processed, coupling_model, coupling_preprocessor, coupling_vc, dim_names
        )
        / SECONDS_PER_DAY
    )


def albedo_cloud_diagnostic_fields(label, config_path, checkpoint_path):
    responses = load_feedback_ablation_responses(
        label, config_path, checkpoint_path, "nn"
    )
    terms = shakirova_terms(responses, "all")
    allcross_residual = terms["allcross"][-1]
    sum_residual = terms["nn"][-1]
    residual_diff = sum_residual - allcross_residual
    coupling = compute_albedo_cloud_coupling(label, config_path, checkpoint_path)
    return [sum_residual, allcross_residual, residual_diff, coupling]


def spatial_field_correlation(a, b, arctic=True):
    if arctic:
        a, b = arctic_field(a), arctic_field(b)
    a_values, b_values = xr.align(a, b, join="inner")
    a_values = a_values.to_numpy().ravel()
    b_values = b_values.to_numpy().ravel()
    valid = np.isfinite(a_values) & np.isfinite(b_values)
    return float(np.corrcoef(a_values[valid], b_values[valid])[0, 1])


def plot_albedo_cloud_coupling_residuals(arctic=True):
    nn_specs = [
        (label, config_path, checkpoint_path)
        for label, config_path, checkpoint_path, source in ABLATION_SPECS
        if source == "nn"
    ]
    fig, ax = plt.subplots(
        len(nn_specs),
        4,
        figsize=(11.69, 11),
        subplot_kw={
            "projection": (
                ccrs.NorthPolarStereo(central_longitude=0)
                if arctic
                else ccrs.PlateCarree(central_longitude=180)
            )
        },
        layout="constrained",
    )
    fig.get_layout_engine().set(
        rect=[0.08, 0.10, 1, 0.94], h_pad=0.01, w_pad=0.01, hspace=0.01, wspace=0.01
    )
    column_titles = [
        r"NN residual sum",
        r"NN residual allcross",
        r"(a) - (b)",
        r"$\Delta R_{ac}^{NN}$",
    ]
    letters = iter("abcdefghijkl")
    im = None
    for row, (label, config_path, checkpoint_path) in enumerate(nn_specs):
        fields = albedo_cloud_diagnostic_fields(label, config_path, checkpoint_path)
        print(
            f"{label}: corr((a)-(b), Delta R_ac) = {spatial_field_correlation(fields[2], fields[3], arctic=arctic):.3f}"
        )
        for col, field in enumerate(fields):
            im = plot_shakirova_field(
                ax[row][col],
                field,
                panel_title(next(letters), column_titles[col] if row == 0 else ""),
                vmin=-8,
                vmax=8,
                step=1,
                left_labels=(col == 0),
                bottom_labels=(row == len(nn_specs) - 1),
                arctic=arctic,
            )
            annotate_response(ax[row][col], field, residual=False, arctic=arctic)
        ax[row][0].text(
            -0.08,
            0.5,
            label,
            transform=ax[row][0].transAxes,
            ha="right",
            va="center",
            rotation=90,
        )

    add_bottom_colorbar(fig, im, ax.ravel().tolist(), "W m$^{-2}$")
    fig.set_layout_engine("none")
    region = "_arctic" if arctic else ""
    fig.savefig(
        shakirova_output_dir / f"shakirova_ablation_albedo_cloud_residuals{region}.png",
        bbox_inches="tight",
    )
    return fig


def plot_albedo_cloud_coupling_diagnostics(arctic=True):
    label, config_path, checkpoint_path, _ = ABLATION_SPECS[-1]
    responses = load_feedback_ablation_responses(
        label, config_path, checkpoint_path, "nn"
    )
    terms = shakirova_terms(responses, "all")
    target = terms["target"]
    a_field = target - terms["allcross"][0]
    b_field = target - terms["nn"][-2]
    c_field = terms["kernel"][-1]
    d_field = b_field - a_field
    e_field = compute_albedo_cloud_coupling(label, config_path, checkpoint_path)
    f_field = e_field - d_field

    panels = [
        (
            0,
            0,
            a_field,
            r"$\Delta R - \Delta R_{\text{allcross}}^{NN}$",
            "a",
            -8,
            8,
            1,
            None,
            None,
        ),
        (
            1,
            0,
            b_field,
            r"$\Delta R - \Delta R_{a+c+q}^{NN}$",
            "b",
            -8,
            8,
            1,
            None,
            None,
        ),
        (
            2,
            0,
            c_field,
            r"$\Delta R - \Delta R_{a+c+q}^{K}$",
            "c",
            -8,
            8,
            1,
            None,
            None,
        ),
        (0, 1, d_field, r"(b) - (a)", "d", -8, 8, 1, None, None),
        (1, 1, e_field, r"$\Delta R_{ac}^{NN}$", "e", -8, 8, 1, None, None),
        (2, 1, f_field, r"(e) - (d)", "f", -8, 8, 1, None, None),
        (0, 2, shakirova_anomaly.fal, r"$\Delta a$", "g", -0.5, 0.5, 0.05, "", ""),
        (1, 2, shakirova_anomaly_ecod, r"$\Delta c$", "h", -1, 1, 0.1, "", ""),
        (
            2,
            2,
            shakirova_anomaly.tcwv,
            r"$\Delta q$",
            "i",
            -20,
            20,
            2,
            "kg m$^{-2}$",
            "kg m$^{-2}$",
        ),
    ]

    fig, ax = plt.subplots(
        3,
        3,
        figsize=(9, 9.5),
        subplot_kw={
            "projection": (
                ccrs.NorthPolarStereo(central_longitude=0)
                if arctic
                else ccrs.PlateCarree(central_longitude=180)
            )
        },
        layout="constrained",
    )
    fig.get_layout_engine().set(
        rect=[0.06, 0.10, 1, 0.94], h_pad=0.01, w_pad=0.01, hspace=0.01, wspace=0.01
    )
    response_im = None
    response_axes = []
    for row, col, field, title, letter, vmin, vmax, step, cbar_label, units in panels:
        im = plot_shakirova_field(
            ax[row][col],
            field,
            panel_title(letter, title),
            vmin=vmin,
            vmax=vmax,
            step=step,
            left_labels=(col == 0),
            bottom_labels=(row == 2),
            arctic=arctic,
        )
        if cbar_label is None:
            response_im = im
            response_axes.append(ax[row][col])
            annotate_response(ax[row][col], field, residual=False, arctic=arctic)
        else:
            annotate_response(ax[row][col], field, residual=False, units=units)
            fig.colorbar(im, ax=ax[row][col], orientation="vertical", label=cbar_label)

    add_bottom_colorbar(fig, response_im, response_axes, "W m$^{-2}$", height=0.015)
    fig.set_layout_engine("none")
    region = "_arctic" if arctic else ""
    fig.savefig(
        shakirova_output_dir
        / f"shakirova_ablation_albedo_cloud_residuals_bottom_row{region}.png",
        bbox_inches="tight",
    )
    return fig


# %%
# plot_feedback_ablation_residuals()

# %%
# fig_shakirova_ablation_albedo_cloud_residuals_arctic = plot_albedo_cloud_coupling_diagnostics(arctic=True)

# %%
# fig_shakirova_ablation_albedo_cloud_residuals_arctic = plot_albedo_cloud_coupling_residuals(arctic=True)


# %%
def plot_albedo_response_stack(arctic=True):
    label, config_path, checkpoint_path, _ = ABLATION_SPECS[-1]
    responses = load_feedback_ablation_responses(
        label, config_path, checkpoint_path, "nn"
    )
    terms = shakirova_terms(responses, "all")
    fields = [
        (terms["target"], r"$\Delta R$"),
        (terms["nn"][0], r"$\Delta R_a^{NN}$"),
        (terms["kernel"][0], r"$\Delta R_a^K$"),
    ]
    fig, ax = plt.subplots(
        3,
        1,
        figsize=(3.8, 8.4),
        subplot_kw={
            "projection": (
                ccrs.NorthPolarStereo(central_longitude=0)
                if arctic
                else ccrs.PlateCarree(central_longitude=180)
            )
        },
        layout="constrained",
    )
    fig.get_layout_engine().set(
        rect=[0.10, 0.10, 0.94, 0.94], h_pad=0.01, w_pad=0.01, hspace=0.01, wspace=0.01
    )
    im = None
    for row, (field, title) in enumerate(fields):
        im = plot_shakirova_field(
            ax[row],
            field,
            panel_title("abc"[row], title),
            vmin=-30,
            vmax=30,
            step=5,
            left_labels=True,
            bottom_labels=(row == len(fields) - 1),
            arctic=arctic,
        )
        annotate_response(ax[row], field, residual=False, arctic=arctic)
    add_bottom_colorbar(fig, im, list(ax), "W m$^{-2}$", height=0.015)
    fig.set_layout_engine("none")
    region = "_arctic" if arctic else ""
    fig.savefig(
        shakirova_output_dir / f"shakirova_ablation_albedo_stack{region}.png",
        bbox_inches="tight",
    )
    return fig


# %%
# plot_albedo_response_stack()


# %%
def plot_albedo_sensitivity_change(base_date=None, perturbed_date=None, arctic=True):
    base_date = base_date or shakirova_base_date
    perturbed_date = perturbed_date or shakirova_perturbed_date
    label, config_path, checkpoint_path, _ = ABLATION_SPECS[1]
    cache_path = (
        shakirova_output_dir
        / f"{coupling_slug(label)}_albedo_sensitivity_differences_{perturbed_date}_minus_{base_date}.nc"
    )

    coupling_config = load_config(config_path)
    years = sorted({pd.Timestamp(base_date).year, pd.Timestamp(perturbed_date).year})
    raw_paths = generate_paths_yearly(
        coupling_config.dataset.era5.raw_path, years, make_era5_filename
    )
    with xr.open_mfdataset(
        raw_paths, combine="nested", concat_dim="date"
    ) as raw_for_dates:
        ds_for_dates = (
            raw_for_dates.sel(date=[base_date, perturbed_date])
            .interp(**shakirova_grid)
            .load()
        )
    base = select_date_state(ds_for_dates, base_date)
    perturbed = select_date_state(ds_for_dates, perturbed_date)
    anomaly = perturbed - base

    coupling_model, coupling_preprocessor, coupling_vc = load_albedo_cloud_model(
        config_path, checkpoint_path
    )
    cloud_vars = [var for var in coupling_vc.clear_sky_zero_vars if var in base]
    delta_ecod = (
        coupling_preprocessor.transform(perturbed)
        - coupling_preprocessor.transform(base)
    ).ecod.compute()

    if cache_path.exists():
        ds = xr.load_dataset(cache_path)
    else:

        def albedo_sensitivity(state):
            kernel, lon, lat = compute_nn_kernel_autograd(
                state.expand_dims(date=[base_date]),
                coupling_preprocessor,
                coupling_model,
                coupling_vc,
                var="fal",
            )
            return xr.DataArray(
                kernel,
                coords={"latitude": lat, "longitude": lon},
                dims=("latitude", "longitude"),
            )

        base_kernel = albedo_sensitivity(base)
        states = {
            "a": base.assign(fal=perturbed.fal),
            "c": base.assign({var: perturbed[var] for var in cloud_vars}),
            "q": base.assign(tcwv=perturbed.tcwv),
            "all": perturbed,
        }
        ds = xr.Dataset(
            {
                name: albedo_sensitivity(state) - base_kernel
                for name, state in states.items()
            }
        )
        ds.to_netcdf(cache_path)

    def true_albedo_kernel(date):
        kernels = xr.open_dataset(
            Path("data/fal/kernels")
            / make_kernel_filename(pd.Timestamp(date).year, "fal")
        )
        try:
            return (
                kernels.sel(date=date).fal.squeeze("date", drop=True).sel(all_clr="all")
                * kernel_delta("fal")
            ).load()
        finally:
            kernels.close()

    true_delta_k = true_albedo_kernel(perturbed_date) - true_albedo_kernel(base_date)
    panels = [
        (0, 0, anomaly.fal, r"$\Delta a$", "a", -0.5, 0.5, 0.05, ""),
        (1, 0, delta_ecod, r"$\Delta c$", "b", -1, 1, 0.1, ""),
        (2, 0, anomaly.tcwv, r"$\Delta q$", "c", -8, 8, 2, "kg m$^{-2}$"),
        (0, 1, ds["a"], r"$\Delta K_a^{NN}$", "d", -0.6, 0.6, 0.1, None),
        (1, 1, ds["c"], r"$\Delta K_c^{NN}$", "e", -0.6, 0.6, 0.1, None),
        (2, 1, ds["q"], r"$\Delta K_q^{NN}$", "f", -0.6, 0.6, 0.1, None),
        (0, 2, true_delta_k, r"$\Delta K^{RRTM}$", "h", -0.6, 0.6, 0.1, None),
        (1, 2, ds["all"], r"$\Delta K^{NN}$", "g", -0.6, 0.6, 0.1, None),
        (
            2,
            2,
            true_delta_k - ds["all"],
            r"$\Delta K^{RRTM} - \Delta K^{NN}$",
            "i",
            -0.6,
            0.6,
            0.1,
            None,
        ),
    ]

    fig, ax = plt.subplots(
        3,
        3,
        figsize=(11.69, 15),
        subplot_kw={
            "projection": (
                ccrs.NorthPolarStereo(central_longitude=0)
                if arctic
                else ccrs.PlateCarree(central_longitude=180)
            )
        },
        layout="constrained",
    )
    fig.get_layout_engine().set(
        rect=[0.06, 0.08, 1, 0.94], h_pad=0.01, w_pad=0.01, hspace=0.01, wspace=0.01
    )

    kernel_im = None
    kernel_axes = []
    for row, col, field, title, letter, vmin, vmax, step, cbar_label in panels:
        im = plot_shakirova_field(
            ax[row][col],
            field,
            panel_title(letter, title),
            vmin=vmin,
            vmax=vmax,
            step=step,
            left_labels=(col == 0),
            bottom_labels=(row == 2),
            arctic=arctic,
            extend=None if col == 0 else "both",
        )
        if cbar_label is None:
            kernel_im = im
            kernel_axes.append(ax[row][col])
            annotate_response(
                ax[row][col],
                field,
                residual=False,
                arctic=arctic,
                units="W m$^{-2}$ %$^{-1}$",
            )
        else:
            fig.colorbar(
                im, ax=ax[row][col], orientation="horizontal", label=cbar_label
            )

    add_bottom_colorbar(fig, kernel_im, kernel_axes, "W m$^{-2}$ %$^{-1}$", height=0.02)
    fig.set_layout_engine("none")
    region = "_arctic" if arctic else ""
    fig.savefig(
        shakirova_output_dir
        / f"shakirova_albedo_sensitivity_differences_{perturbed_date}_minus_{base_date}{region}.png",
        bbox_inches="tight",
    )
    return fig


def plot_albedo_kernel_comparison(base_date=None, perturbed_date=None, arctic=True):
    base_date = base_date or shakirova_base_date
    perturbed_date = perturbed_date or shakirova_perturbed_date
    label, config_path, checkpoint_path, _ = ABLATION_SPECS[1]
    cache_path = (
        shakirova_output_dir
        / f"{coupling_slug(label)}_albedo_kernels_{base_date}_{perturbed_date}.nc"
    )

    def true_albedo_kernel(date):
        kernels = xr.open_dataset(
            Path("data/fal/kernels")
            / make_kernel_filename(pd.Timestamp(date).year, "fal")
        )
        try:
            return (
                kernels.sel(date=date).fal.squeeze("date", drop=True).sel(all_clr="all")
                * kernel_delta("fal")
            ).load()
        finally:
            kernels.close()

    if cache_path.exists():
        ds = xr.load_dataset(cache_path)
    else:
        coupling_config = load_config(config_path)
        years = sorted(
            {pd.Timestamp(base_date).year, pd.Timestamp(perturbed_date).year}
        )
        raw_paths = generate_paths_yearly(
            coupling_config.dataset.era5.raw_path, years, make_era5_filename
        )
        with xr.open_mfdataset(
            raw_paths, combine="nested", concat_dim="date"
        ) as raw_for_dates:
            ds_for_dates = (
                raw_for_dates.sel(date=[base_date, perturbed_date])
                .interp(**shakirova_grid)
                .load()
            )
        coupling_model, coupling_preprocessor, coupling_vc = load_albedo_cloud_model(
            config_path, checkpoint_path
        )

        def nn_albedo_kernel(date):
            state = select_date_state(ds_for_dates, date)
            kernel, lon, lat = compute_nn_kernel_autograd(
                state.expand_dims(date=[date]),
                coupling_preprocessor,
                coupling_model,
                coupling_vc,
                var="fal",
            )
            return xr.DataArray(
                kernel,
                coords={"latitude": lat, "longitude": lon},
                dims=("latitude", "longitude"),
            )

        ds = xr.Dataset(
            {
                "rrtm_base": true_albedo_kernel(base_date),
                "rrtm_perturbed": true_albedo_kernel(perturbed_date),
                "nn_base": nn_albedo_kernel(base_date),
                "nn_perturbed": nn_albedo_kernel(perturbed_date),
            }
        )
        ds.to_netcdf(cache_path)

    kernel_panels = [
        (0, 0, ds["rrtm_base"], r"$K^{RRTM}$", "a"),
        (0, 1, ds["nn_base"], r"$K^{NN}$", "b"),
        (1, 0, ds["rrtm_perturbed"], r"$K'^{RRTM}$", "d"),
        (1, 1, ds["nn_perturbed"], r"$K'^{NN}$", "e"),
    ]
    residual_panels = [
        (0, 2, ds["rrtm_base"] - ds["nn_base"], r"$K^{RRTM} - K^{NN}$", "c"),
        (
            1,
            2,
            ds["rrtm_perturbed"] - ds["nn_perturbed"],
            r"$K'^{RRTM} - K'^{NN}$",
            "f",
        ),
    ]
    fig, ax = plt.subplots(
        2,
        3,
        figsize=(12, 9),
        subplot_kw={
            "projection": (
                ccrs.NorthPolarStereo(central_longitude=0)
                if arctic
                else ccrs.PlateCarree(central_longitude=180)
            )
        },
        layout="constrained",
    )
    fig.get_layout_engine().set(
        rect=[0.06, 0.14, 1, 0.94], h_pad=0.01, w_pad=0.01, hspace=0.01, wspace=0.01
    )
    kernel_im = residual_im = None
    for row, col, field, title, letter in kernel_panels:
        kernel_im = plot_shakirova_field(
            ax[row][col],
            field,
            panel_title(letter, title),
            vmin=-2.6,
            vmax=0,
            step=0.2,
            left_labels=(col == 0),
            bottom_labels=(row == 1),
            arctic=arctic,
            cmap="Blues_r",
            extend="both",
        )
        annotate_response(
            ax[row][col],
            field,
            residual=False,
            arctic=arctic,
            units="W m$^{-2}$ %$^{-1}$",
        )
    for row, col, field, title, letter in residual_panels:
        residual_im = plot_shakirova_field(
            ax[row][col],
            field,
            panel_title(letter, title),
            vmin=-0.4,
            vmax=0.4,
            step=0.1,
            left_labels=False,
            bottom_labels=(row == 1),
            arctic=arctic,
            extend="both",
        )
        annotate_response(
            ax[row][col],
            field,
            residual=False,
            arctic=arctic,
            units="W m$^{-2}$ %$^{-1}$",
        )

    add_bottom_colorbar(
        fig,
        kernel_im,
        [ax[0][0], ax[0][1], ax[1][0], ax[1][1]],
        "W m$^{-2}$ %$^{-1}$",
        gap=0.01,
    )
    add_bottom_colorbar(
        fig, residual_im, [ax[0][2], ax[1][2]], "W m$^{-2}$ %$^{-1}$", gap=0.01
    )
    fig.set_layout_engine("none")
    region = "_arctic" if arctic else ""
    fig.savefig(
        shakirova_output_dir
        / f"shakirova_albedo_kernels_{perturbed_date}_and_{base_date}{region}.png",
        bbox_inches="tight",
    )
    return fig


def plot_neurips_albedo_sensitivity(date="2015-07", arctic=False):
    output_path = (
        shakirova_output_dir / f"neurips_albedo_radiative_sensitivity_{date}.png"
    )
    raw_ds = xr.open_dataset(
        Path("data/era5") / make_era5_filename(pd.Timestamp(date).year)
    )
    kernels = xr.open_dataset(
        Path("data/fal/kernels") / make_kernel_filename(pd.Timestamp(date).year, "fal")
    )
    kernel_grid = {"latitude": kernels.latitude, "longitude": kernels.longitude}
    raw_fd = raw_ds.sel(date=[date]).interp(**kernel_grid)
    raw_autograd = raw_ds.sel(date=[date]).interp(**kernel_grid)
    true_kernel = kernels.sel(date=date).fal.squeeze("date", drop=True) * kernel_delta(
        "fal"
    )

    def as_field(values, lon, lat):
        return xr.DataArray(
            values,
            coords={"longitude": lon, "latitude": lat},
            dims=("latitude", "longitude"),
        )

    def get_prediction(config, method="fd", clear=False):
        _, vc, m, pp = load_sensitivity_model(config)
        if method == "fd":
            pred, lon, lat = compute_nn_kernel(
                raw_fd, pp, m, vc, clear=clear, perturbation_var="fal"
            )
        else:
            pred, lon, lat = compute_nn_kernel_autograd(
                raw_autograd, pp, m, vc, var="fal", clear=clear
            )
        return as_field(pred, lon, lat)

    rows = [
        ("RRTM", true_kernel.sel(all_clr="all"), true_kernel.sel(all_clr="clr")),
        (
            "NN",
            get_prediction(
                "configs/model/fal/2011-2014_3,6,9,12_baseline.yaml",
                method="fd",
                clear=False,
            ),
            get_prediction(
                "configs/model/fal/2011-2014_3,6,9,12_baseline.yaml",
                method="fd",
                clear=True,
            ),
        ),
        (
            "NN+clear-sky",
            get_prediction(
                "configs/model/fal/2011-2014_3,6,9,12_baseline_clearsky.yaml",
                method="fd",
                clear=False,
            ),
            get_prediction(
                "configs/model/fal/2011-2014_3,6,9,12_baseline_clearsky.yaml",
                method="fd",
                clear=True,
            ),
        ),
        (
            "NN+clear-sky+Sob.",
            get_prediction(
                "configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky.yaml",
                method="autograd",
                clear=False,
            ),
            get_prediction(
                "configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky.yaml",
                method="autograd",
                clear=True,
            ),
        ),
    ]
    fig, ax = plt.subplots(
        4,
        2,
        figsize=(11.69, 13.5),
        subplot_kw={
            "projection": (
                ccrs.NorthPolarStereo(central_longitude=0)
                if arctic
                else ccrs.PlateCarree(central_longitude=180)
            )
        },
        layout="constrained",
    )
    fig.get_layout_engine().set(
        rect=[0.06, 0.10, 0.96, 0.94], h_pad=0.01, w_pad=0.01, hspace=0.01, wspace=0.01
    )
    im = None
    letters = iter("aebfcgdh")
    for row, (label, all_field, clear_field) in enumerate(rows):
        for col, (field, sky) in enumerate(
            [(all_field, "all-sky"), (clear_field, "clear-sky")]
        ):
            im = plot_shakirova_field(
                ax[row][col],
                field,
                panel_title(
                    next(letters), f"{label}" + ((" " + sky) if row == 0 else "")
                ),
                vmin=-3,
                vmax=3,
                step=0.2,
                left_labels=(col == 0),
                bottom_labels=(row == len(rows) - 1),
                arctic=arctic,
            )
            annotate_response(
                ax[row][col], field, arctic=arctic, units="W m$^{-2}$ %$^{-1}$"
            )

    add_right_colorbar(fig, im, ax.ravel().tolist(), "W m$^{-2}$ %$^{-1}$", width=0.03)
    fig.set_layout_engine("none")
    fig.savefig(output_path, bbox_inches="tight")
    raw_ds.close()
    kernels.close()
    return fig


def plot_tsi_albedo_sensitivity(date="2015-07", arctic=False):
    Path("FiguresTSI").mkdir(exist_ok=True)
    output_path = Path("FiguresTSI") / f"albedo_radiative_sensitivity_{date}.png"
    plt.style.use("dark_background")
    raw_ds = xr.open_dataset(
        Path("data/era5") / make_era5_filename(pd.Timestamp(date).year)
    )
    kernels = xr.open_dataset(
        Path("data/fal/kernels") / make_kernel_filename(pd.Timestamp(date).year, "fal")
    )
    kernel_grid = {"latitude": kernels.latitude, "longitude": kernels.longitude}
    raw_fd = raw_ds.sel(date=[date]).interp(**kernel_grid)
    raw_autograd = raw_ds.sel(date=[date]).interp(**kernel_grid)
    true_kernel = kernels.sel(date=date).fal.squeeze("date", drop=True) * kernel_delta(
        "fal"
    )

    def as_field(values, lon, lat):
        return xr.DataArray(
            values,
            coords={"longitude": lon, "latitude": lat},
            dims=("latitude", "longitude"),
        )

    def get_prediction(config, method="fd", clear=False):
        _, vc, m, pp = load_sensitivity_model(config)
        if method == "fd":
            pred, lon, lat = compute_nn_kernel(
                raw_fd, pp, m, vc, clear=clear, perturbation_var="fal"
            )
        else:
            pred, lon, lat = compute_nn_kernel_autograd(
                raw_autograd, pp, m, vc, var="fal", clear=clear
            )
        return as_field(pred, lon, lat)

    rows = [
        ("RRTM", true_kernel.sel(all_clr="all"), true_kernel.sel(all_clr="clr")),
        (
            "NN",
            get_prediction(
                "configs/model/fal/2011-2014_3,6,9,12_baseline.yaml",
                method="fd",
                clear=False,
            ),
            get_prediction(
                "configs/model/fal/2011-2014_3,6,9,12_baseline.yaml",
                method="fd",
                clear=True,
            ),
        ),
        (
            "NN+clear-sky+Sob.",
            get_prediction(
                "configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky.yaml",
                method="autograd",
                clear=False,
            ),
            get_prediction(
                "configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky.yaml",
                method="autograd",
                clear=True,
            ),
        ),
    ]
    fig, ax = plt.subplots(
        1,
        3,
        figsize=(11.69, 4),
        subplot_kw={
            "projection": (
                ccrs.NorthPolarStereo(central_longitude=0)
                if arctic
                else ccrs.PlateCarree(central_longitude=180)
            )
        },
        layout="constrained",
        dpi=600,
    )
    fig.get_layout_engine().set(
        rect=[0.06, 0.10, 0.96, 0.94], h_pad=0.01, w_pad=0.01, hspace=0.01, wspace=0.01
    )
    im = None
    letters = iter("abc")
    for col, (label, all_field, clear_field) in enumerate(rows):
        field = all_field.where(all_field <= 0, other=0)
        im = plot_shakirova_field(
            ax[col],
            field,
            panel_title(next(letters), f"{label}"),
            vmin=-3.6,
            vmax=0,
            step=0.2,
            left_labels=(col == 0),
            bottom_labels=True,
            arctic=arctic,
            cmap="Blues_r",
            extend=None,
        )
        annotate_response(ax[col], field, arctic=arctic, units="W m$^{-2}$ %$^{-1}$")

    add_bottom_colorbar(
        fig, im, ax.ravel().tolist(), "W m$^{-2}$ %$^{-1}$", height=0.05
    )
    fig.set_layout_engine("none")
    fig.savefig(output_path, bbox_inches="tight")
    raw_ds.close()
    kernels.close()
    return fig


# %%


def run_neurips_feedback_figures():
    setup_neurips_feedback_context()
    plot_feedback_ablation_residuals(arctic=True)
    plot_albedo_cloud_coupling_diagnostics(arctic=True)
    plot_albedo_cloud_coupling_residuals(arctic=True)
    plot_albedo_response_stack(arctic=True)
    plot_albedo_kernel_comparison(
        base_date="2012-08", perturbed_date="2015-08", arctic=True
    )
    plot_albedo_sensitivity_change(
        base_date="2012-08", perturbed_date="2015-08", arctic=True
    )
