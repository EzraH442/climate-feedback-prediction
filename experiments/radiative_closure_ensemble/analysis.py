from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from constants import (
    CLOSURE_NORTH_BOUNDARY,
    MONTH_NAMES,
)
from utils import (
    global_date_series,
    plot_input_anomalies,
    plot_response_dataset,
    setup_timeseries_plot,
    weighted_residuals_by_month,
    weighted_residuals_by_year_month,
)

NORTH_MASK = None


def closure_north_boundary(responses: xr.Dataset) -> float:
    return float(responses.attrs.get("north_boundary", CLOSURE_NORTH_BOUNDARY))


def plot_component_timeseries(series, output_dir: Path, clear_sky: bool) -> None:
    variables = ["a", "q"] if clear_sky else ["a", "q", "c"]
    suffix = "_clr" if clear_sky else ""
    for var in variables:
        fig, ax = setup_timeseries_plot()
        ax.axhline(0, alpha=0.1)
        ylabel = (
            rf"$\Delta R_{{{var},clr}}$ ($W /m^2$)"
            if clear_sky
            else rf"$\Delta R_{{{var}}}$ ($W m^2$)"
        )
        for source in ["nn", "k"]:
            name = f"dR_{var}_{source}{suffix}"
            ax.scatter(x=series[name]["date"], y=series[name], s=1, label=source)
        ax.set_yticks(np.arange(-2, 2.5, 0.5))
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"timeseries_{var}.png", dpi=200)
        plt.close(fig)


def _residual_key(source: str, clear_sky: bool) -> str:
    return f"dR_res_{source}" + ("_clr" if clear_sky else "")


def build_residual_series(residual_fields: dict, north_pole: bool = False) -> dict:
    """
    build the {dR_res_<source>[_clr]: timeseries} dict
    residual_fields keys are expected to look like "nn", "k", "nn_allcross",
    "nn_clr", "k_clr", "nn_allcross_clr".
    """
    series = {}
    for name, field in residual_fields.items():
        if north_pole:
            field = field.isel(latitude=NORTH_MASK)
        series[f"dR_res_{name}"] = global_date_series(field)
    return series


def build_rmse_residual_series(residual_fields: dict, north_pole: bool = False) -> dict:
    series = {}
    for name, field in residual_fields.items():
        if north_pole:
            field = field.isel(latitude=NORTH_MASK)
        series[f"dR_res_{name}"] = np.sqrt(global_date_series(field * field))
    return series


def plot_residual_rmse_timeseries(series, output_dir: Path, clear_sky: bool) -> None:
    fig, ax = setup_timeseries_plot()
    ylabel = (
        rf"RMSE $\Delta R_{{net,clr}}$ ($W m^{{-2}}$)"
        if clear_sky
        else rf"RMSE $\Delta R_{{net}}$ ($W m^{{-2}}$)"
    )
    for source in ["nn", "k", "nn_allcross"]:
        name = f"dR_res_{source}" + ("_clr" if clear_sky else "")
        ax.scatter(x=series[name]["date"], y=series[name], s=1, label=source)
    ax.set_yticks(np.arange(0, 7, 2))
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "timeseries_rmse.png", dpi=200)
    plt.close(fig)


def plot_residual_mbe_timeseries(series, output_dir: Path, clear_sky: bool) -> None:
    fig, ax = setup_timeseries_plot()
    ylabel = (
        rf"MBE $\Delta R_{{net,clr}}$ ($W m^{{-2}}$)"
        if clear_sky
        else rf"MBE $\Delta R_{{net}}$ ($W m^{{-2}}$)"
    )
    for source in ["nn", "k", "nn_allcross"]:
        name = f"dR_res_{source}" + ("_clr" if clear_sky else "")
        ax.scatter(x=series[name]["date"], y=series[name], s=1, label=source)
    ax.set_yticks(np.arange(-2, 3, 1))
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_dir / "timeseries_mbe.png", dpi=200)
    plt.close(fig)


def plot_net_timeseries(net_series, save_path: Path) -> None:
    fig, ax = setup_timeseries_plot()
    ax.axhline(0, alpha=0.1)
    for name in ["nn", "kernel", "era5"]:
        ax.scatter(x=net_series[name]["date"], y=net_series[name], s=1, label=name)
    ax.set_yticks(np.arange(-2, 2.5, 0.5))
    ax.set_ylabel("$\Delta R$ (W/m$^2$)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def setup_box_plot(figsize=(14, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color="black", alpha=0.2, linewidth=0.8)
    ax.set_ylim(-20, 20)
    ax.set_yticks(np.arange(-20, 21, 5))
    ax.set_ylabel(r"$\Delta R_{\rm res}$ (W m$^{-2}$)")
    ax.grid(axis="y", alpha=0.3)
    return fig, ax


def plot_boxes_on_ax(ax, data, positions, width, color, label) -> None:
    box = ax.boxplot(
        data,
        positions=positions,
        widths=width,
        patch_artist=True,
        showfliers=False,
    )
    for patch in box["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor(color)
    for key in ["whiskers", "caps", "medians"]:
        for artist in box[key]:
            artist.set_color(color)
    ax.scatter([], [], color=color, label=label, s=20)


def plot_residual_boxplots(residuals, residuals_np, output_dir: Path) -> None:
    colors = {"nn": "tab:blue", "nn_cross": "purple", "kernel": "tab:orange"}
    keys = [key for key in ["nn", "nn_cross", "kernel"] if key in residuals]
    width = 0.7 / len(keys)
    offsets = dict(
        zip(keys, np.linspace(-0.35 + width / 2, 0.35 - width / 2, len(keys)))
    )
    months = np.arange(1, 13)

    for title, data_dict, save_name in [
        ("Global", residuals, "boxplot_residuals_by_month_global.png"),
        ("North Pole (>75 deg N)", residuals_np, "boxplot_residuals_by_month_np.png"),
    ]:
        fig, ax = setup_box_plot()
        for key in keys:
            plot_boxes_on_ax(
                ax,
                data_dict[key],
                months + offsets[key],
                width * 0.9,
                colors[key],
                key,
            )

        ax.set_xticks(months)
        ax.set_xticklabels(MONTH_NAMES)
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / save_name, dpi=200)
        plt.close(fig)


def plot_residual_year_month_boxplots(
    nn_residual: xr.DataArray,
    k_residual: xr.DataArray,
    title: str,
    save_path: Path,
    residual_samples: int,
) -> None:
    labels, nn_residuals = weighted_residuals_by_year_month(
        nn_residual, residual_samples, np.random.default_rng(0)
    )
    k_labels, k_residuals = weighted_residuals_by_year_month(
        k_residual, residual_samples, np.random.default_rng(1)
    )
    if labels != k_labels:
        raise ValueError("NN and K residuals do not share year-month coordinates.")

    positions = np.arange(len(labels))
    colors = {"NN": "tab:blue", "K": "tab:orange"}
    width = 0.35
    fig_width = max(14, 0.18 * len(labels))
    fig, ax = setup_box_plot(figsize=(fig_width, 4))
    for name, data, offset in [
        ("NN", nn_residuals, -width / 2),
        ("K", k_residuals, width / 2),
    ]:
        plot_boxes_on_ax(
            ax,
            data,
            positions + offset,
            width,
            colors[name],
            name,
        )

    tick_step = 6
    tick_positions = positions[::tick_step]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([labels[i] for i in tick_positions], rotation=45, ha="right")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def timeseries_test(
    responses: xr.Dataset,
    output_root: Path,
    residual_samples: int,
) -> None:
    if NORTH_MASK is None:
        raise RuntimeError("NORTH_MASK must be initialized before timeseries_test.")
    output_root.mkdir(exist_ok=True, parents=True)
    timeseries_clear = output_root / "clr"
    timeseries_all = output_root / "all"
    timeseries_clear.mkdir(exist_ok=True, parents=True)
    timeseries_all.mkdir(exist_ok=True, parents=True)
    (timeseries_clear / "np").mkdir(exist_ok=True, parents=True)
    (timeseries_all / "np").mkdir(exist_ok=True, parents=True)
    dR = responses["dR_era5_all"]
    dR_clr = responses["dR_era5_clr"]
    dR_a_nn = responses["dR_a_nn_all"]
    dR_c_nn = responses["dR_c_nn_all"]
    dR_q_nn = responses["dR_q_nn_all"]
    dR_a_nn_clr = responses["dR_a_nn_clr"]
    dR_q_nn_clr = responses["dR_q_nn_clr"]
    dR_a_k = responses["dR_a_k_all"]
    dR_c_k = responses["dR_c_k_all"]
    dR_q_k = responses["dR_q_k_all"]
    dR_a_k_clr = responses["dR_a_k_clr"]
    dR_q_k_clr = responses["dR_q_k_clr"]

    dR_sum_nn = dR_a_nn + dR_c_nn + dR_q_nn
    dR_sum_nn_allcross = responses["dR_nn_all"]
    dR_sum_k = dR_a_k + dR_c_k + dR_q_k
    dR_sum_nn_clr = dR_a_nn_clr + dR_q_nn_clr
    dR_sum_nn_allcross_clr = responses["dR_nn_clr"]
    dR_sum_k_clr = dR_a_k_clr + dR_q_k_clr

    dR_res_nn = dR - dR_sum_nn
    dR_res_nn_allcross = dR - dR_sum_nn_allcross
    dR_res_k = dR - dR_sum_k
    dR_res_nn_clr = dR_clr - dR_sum_nn_clr
    dR_res_nn_allcross_clr = dR_clr - dR_sum_nn_allcross_clr
    dR_res_k_clr = dR_clr - dR_sum_k_clr
    fields = {
        "dR": dR,
        "dR_clr": dR_clr,
        "dR_sum_nn": dR_sum_nn,
        "dR_sum_nn_allcross": dR_sum_nn_allcross,
        "dR_sum_k": dR_sum_k,
        "dR_sum_nn_clr": dR_sum_nn_clr,
        "dR_sum_nn_allcross_clr": dR_sum_nn_allcross_clr,
        "dR_sum_k_clr": dR_sum_k_clr,
        "dR_res_nn": dR_res_nn,
        "dR_res_nn_allcross": dR_res_nn_allcross,
        "dR_res_k": dR_res_k,
        "dR_res_nn_clr": dR_res_nn_clr,
        "dR_res_nn_allcross_clr": dR_res_nn_allcross_clr,
        "dR_res_k_clr": dR_res_k_clr,
    }
    field_series = {key: global_date_series(value) for key, value in fields.items()}
    field_series_np = {
        key: global_date_series(value.isel(latitude=NORTH_MASK))
        for key, value in fields.items()
    }

    component_inputs = {
        "dR_a_nn": dR_a_nn,
        "dR_a_k": dR_a_k,
        "dR_q_nn": dR_q_nn,
        "dR_q_k": dR_q_k,
        "dR_c_nn": dR_c_nn,
        "dR_c_k": dR_c_k,
        "dR_a_nn_clr": dR_a_nn_clr,
        "dR_a_k_clr": dR_a_k_clr,
        "dR_q_nn_clr": dR_q_nn_clr,
        "dR_q_k_clr": dR_q_k_clr,
    }
    component_series = {
        key: global_date_series(value) for key, value in component_inputs.items()
    }
    component_series_np = {
        key: global_date_series(value.isel(latitude=NORTH_MASK))
        for key, value in component_inputs.items()
    }
    plot_component_timeseries(component_series, timeseries_all, clear_sky=False)
    plot_component_timeseries(component_series, timeseries_clear, clear_sky=True)
    plot_component_timeseries(
        component_series_np, timeseries_all / "np", clear_sky=False
    )
    plot_component_timeseries(
        component_series_np, timeseries_clear / "np", clear_sky=True
    )

    plot_net_timeseries(
        {
            "nn": field_series["dR_sum_nn"],
            "kernel": field_series["dR_sum_k"],
            "era5": field_series["dR"],
        },
        timeseries_all / "timeseries_net.png",
    )
    plot_net_timeseries(
        {
            "nn": field_series["dR_sum_nn_clr"],
            "kernel": field_series["dR_sum_k_clr"],
            "era5": field_series["dR_clr"],
        },
        timeseries_clear / "timeseries_net.png",
    )
    plot_net_timeseries(
        {
            "nn": field_series_np["dR_sum_nn"],
            "kernel": field_series_np["dR_sum_k"],
            "era5": field_series_np["dR"],
        },
        timeseries_all / "np" / "timeseries_net.png",
    )
    plot_net_timeseries(
        {
            "nn": field_series_np["dR_sum_nn_clr"],
            "kernel": field_series_np["dR_sum_k_clr"],
            "era5": field_series_np["dR_clr"],
        },
        timeseries_clear / "np" / "timeseries_net.png",
    )

    cross_vars = ["dR_aq_nn_all", "dR_ac_nn_all", "dR_qc_nn_all"]
    has_cross = all(name in responses for name in cross_vars)
    if has_cross:
        dR_aq_nn = responses["dR_aq_nn_all"]
        dR_ac_nn = responses["dR_ac_nn_all"]
        dR_qc_nn = responses["dR_qc_nn_all"]
        dR_sum_nn_cross = dR_sum_nn + dR_aq_nn + dR_ac_nn + dR_qc_nn

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axhline(0, alpha=0.1)
        colors = ["tab:green", "tab:blue", "tab:orange", "purple", "tab:red"]
        net_series = {
            "era5": field_series["dR"],
            "nn": field_series["dR_sum_nn"],
            "kernel": field_series["dR_sum_k"],
            "nn_cross": global_date_series(dR_sum_nn_cross),
            "cross": global_date_series(dR_aq_nn + dR_ac_nn + dR_qc_nn),
        }
        for i, (name, values) in enumerate(net_series.items()):
            ax.scatter(
                x=values["date"],
                y=values,
                s=3 if name == "era5" else 1,
                label=name,
                color=colors[i],
            )
        ax.set_yticks([-2, 0, 2])
        ax.set_xticks(
            pd.date_range(start="2007", end="2017", freq="YS", inclusive="both"),
            np.arange(2007, 2018),
        )
        ax.set_ylabel("W/m^2")
        ax.legend()
        fig.tight_layout()
        fig.savefig(timeseries_all / "timeseries_sum_cross.png", dpi=200)
        plt.close(fig)

    mbe_fields_all = {
        "nn": dR_res_nn,
        "k": dR_res_k,
        "nn_allcross": dR_res_nn_allcross,
    }
    mbe_fields_clr = {
        "nn_clr": dR_res_nn_clr,
        "k_clr": dR_res_k_clr,
        "nn_allcross_clr": dR_res_nn_allcross_clr,
    }
    mbe_series_all = build_residual_series(mbe_fields_all, north_pole=False)
    mbe_series_clr = build_residual_series(mbe_fields_clr, north_pole=False)
    mbe_series_all_np = build_residual_series(mbe_fields_all, north_pole=True)
    mbe_series_clr_np = build_residual_series(mbe_fields_clr, north_pole=True)

    plot_residual_mbe_timeseries(
        {**mbe_series_all, **mbe_series_clr}, timeseries_all, clear_sky=False
    )
    plot_residual_mbe_timeseries(
        {**mbe_series_all, **mbe_series_clr}, timeseries_clear, clear_sky=True
    )
    plot_residual_mbe_timeseries(
        {**mbe_series_all_np, **mbe_series_clr_np},
        timeseries_all / "np",
        clear_sky=False,
    )
    plot_residual_mbe_timeseries(
        {**mbe_series_all_np, **mbe_series_clr_np},
        timeseries_clear / "np",
        clear_sky=True,
    )

    rmse_series_all = build_rmse_residual_series(mbe_fields_all, north_pole=False)
    rmse_series_clr = build_rmse_residual_series(mbe_fields_clr, north_pole=False)
    rmse_series_all_np = build_rmse_residual_series(mbe_fields_all, north_pole=True)
    rmse_series_clr_np = build_rmse_residual_series(mbe_fields_clr, north_pole=True)

    plot_residual_rmse_timeseries(
        {**rmse_series_all, **rmse_series_clr}, timeseries_all, clear_sky=False
    )
    plot_residual_rmse_timeseries(
        {**rmse_series_all, **rmse_series_clr}, timeseries_clear, clear_sky=True
    )
    plot_residual_rmse_timeseries(
        {**rmse_series_all_np, **rmse_series_clr_np},
        timeseries_all / "np",
        clear_sky=False,
    )
    plot_residual_rmse_timeseries(
        {**rmse_series_all_np, **rmse_series_clr_np},
        timeseries_clear / "np",
        clear_sky=True,
    )

    plot_residual_year_month_boxplots(
        dR_res_nn,
        dR_res_k,
        "Global",
        timeseries_all / "boxplot_residuals_by_year_month_global.png",
        residual_samples,
    )
    plot_residual_year_month_boxplots(
        dR_res_nn.isel(latitude=NORTH_MASK),
        dR_res_k.isel(latitude=NORTH_MASK),
        "North Pole (>75 deg N)",
        timeseries_all / "np" / "boxplot_residuals_by_year_month_np.png",
        residual_samples,
    )
    plot_residual_year_month_boxplots(
        dR_res_nn_clr,
        dR_res_k_clr,
        "Global clear-sky",
        timeseries_clear / "boxplot_residuals_by_year_month_global.png",
        residual_samples,
    )
    plot_residual_year_month_boxplots(
        dR_res_nn_clr.isel(latitude=NORTH_MASK),
        dR_res_k_clr.isel(latitude=NORTH_MASK),
        "North Pole (>75 deg N) clear-sky",
        timeseries_clear / "np" / "boxplot_residuals_by_year_month_np.png",
        residual_samples,
    )
    rng = np.random.default_rng(0)
    residuals_clear = {
        "nn": weighted_residuals_by_month(dR_res_nn_clr, residual_samples, rng),
        "kernel": weighted_residuals_by_month(dR_res_k_clr, residual_samples, rng),
    }
    residuals_clear_np = {
        "nn": weighted_residuals_by_month(
            dR_res_nn_clr.isel(latitude=NORTH_MASK), residual_samples, rng
        ),
        "kernel": weighted_residuals_by_month(
            dR_res_k_clr.isel(latitude=NORTH_MASK), residual_samples, rng
        ),
    }
    plot_residual_boxplots(residuals_clear, residuals_clear_np, timeseries_clear)

    # if not has_cross:
    #    return

    # rng = np.random.default_rng(0)
    # residuals = {
    #    "nn": weighted_residuals_by_month(dR_res_nn, residual_samples, rng),
    #    "nn_cross": weighted_residuals_by_month(dR_res_nn_cross, residual_samples, rng),
    #    "kernel": weighted_residuals_by_month(dR_res_k, residual_samples, rng),
    # }
    # residuals_np = {
    #    "nn": weighted_residuals_by_month(
    #        dR_res_nn.isel(latitude=NORTH_MASK), residual_samples, rng
    #    ),
    #    "nn_cross": weighted_residuals_by_month(
    #        dR_res_nn_cross.isel(latitude=NORTH_MASK), residual_samples, rng
    #    ),
    #    "kernel": weighted_residuals_by_month(
    #        dR_res_k.isel(latitude=NORTH_MASK), residual_samples, rng
    #    ),
    # }
    # plot_residual_boxplots(residuals, residuals_np, timeseries_all)


def date_closure_test(
    anomaly: xr.Dataset,
    ds_monthly: xr.Dataset,
    ds_monthly_means: xr.Dataset,
    preprocessor,
    responses: xr.Dataset,
    output_root: Path,
    year: int,
    month: int,
    dR_co3_nn=None,
    dR_co3_nn_clr=None,
    skip_input_anomaly_plots=False,
) -> None:
    if NORTH_MASK is None:
        raise RuntimeError("NORTH_MASK must be initialized before date_closure_test.")
    north_boundary = closure_north_boundary(responses)
    date_root = output_root / f"{year}_{month:02d}"
    date_root.mkdir(exist_ok=True, parents=True)
    date_clear = date_root / "clr"
    date_all = date_root / "all"
    date_clear.mkdir(exist_ok=True, parents=True)
    date_all.mkdir(exist_ok=True, parents=True)
    (date_clear / "np").mkdir(exist_ok=True, parents=True)
    (date_all / "np").mkdir(exist_ok=True, parents=True)
    dR = responses["dR_era5_all"]
    dR_clr = responses["dR_era5_clr"]
    pp = preprocessor.transform(ds_monthly.sel(month=month, year=year))
    ppm = preprocessor.transform(ds_monthly_means.sel(month=month))
    anomaly_ecod = (pp - ppm).ecod.compute()

    if not skip_input_anomaly_plots:
        plot_input_anomalies(
            anomaly,
            anomaly_ecod,
            dR_clr,
            date_root,
            north_mask=NORTH_MASK,
            north_boundary=north_boundary,
            year=year,
            month=month,
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
        print(name, np.max(np.abs(response.sel(month=month, year=year))).values)

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
        north_boundary=north_boundary,
        year=year,
        month=month,
        np_vmax=24,
    )
    plot_response_dataset(
        responses[["dR_a_nn_clr", "dR_q_nn_clr"]].rename(
            {"dR_a_nn_clr": "a", "dR_q_nn_clr": "q"}
        ),
        date_clear,
        source="NN",
        north_mask=NORTH_MASK,
        north_boundary=north_boundary,
        year=year,
        month=month,
        np_vmax=24,
    )

    for name, response in [
        ("dR_a_k", responses["dR_a_k_all"]),
        ("dR_c_k", responses["dR_c_k_all"]),
        ("dR_q_k", responses["dR_q_k_all"]),
        ("dR_a_k_clr", responses["dR_a_k_clr"]),
        ("dR_q_k_clr", responses["dR_q_k_clr"]),
    ]:
        print(name, np.max(np.abs(response.sel(month=month, year=year))).values)

    plot_response_dataset(
        responses[["dR_a_k_all", "dR_q_k_all", "dR_c_k_all"]].rename(
            {"dR_a_k_all": "a", "dR_q_k_all": "q", "dR_c_k_all": "c"}
        ),
        date_all,
        source="K",
        north_mask=NORTH_MASK,
        north_boundary=north_boundary,
        year=year,
        month=month,
        np_vmax=24,
    )
    plot_response_dataset(
        responses[["dR_a_k_clr", "dR_q_k_clr"]].rename(
            {"dR_a_k_clr": "a", "dR_q_k_clr": "q"}
        ),
        date_clear,
        source="K",
        north_mask=NORTH_MASK,
        north_boundary=north_boundary,
        year=year,
        month=month,
        np_vmax=24,
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
                plot_label="res", filename="dR_res.png", vmax=24
            ),
            "res_allcross": dR_res_nn_allcross.assign_attrs(
                plot_label="res,allcross", filename="dR_res_allcross.png", vmax=24
            ),
        }
    )

    kernel_all_closure = xr.Dataset(
        {
            "sum": dR_sum_k.assign_attrs(
                plot_label="sum", filename="k_dR_sum.png", vmax=55
            ),
            "res": dR_res_k.assign_attrs(
                plot_label="res", filename="k_dR_res.png", vmax=24
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
                plot_label="res,clr", filename="dR_res,clr.png", vmax=24
            ),
            "res_allcross_clr": dR_res_nn_allcross_clr.assign_attrs(
                plot_label="res,allcross,clr",
                filename="dR_res_allcross,clr.png",
                vmax=24,
            ),
        }
    )

    kernel_clr_closure = xr.Dataset(
        {
            "sum_clr": dR_sum_k_clr.assign_attrs(
                plot_label="sum,clr", filename="k_dR_sum,clr.png", vmax=60
            ),
            "res_clr": dR_res_k_clr.assign_attrs(
                plot_label="res,clr", filename="k_dR_res,clr.png", vmax=24
            ),
        }
    )

    plot_response_dataset(
        nn_all_closure,
        date_all,
        source="NN",
        north_mask=NORTH_MASK,
        north_boundary=north_boundary,
        year=year,
        month=month,
        np_vmax=24,
        ann_rmse=True,
    )
    plot_response_dataset(
        kernel_all_closure,
        date_all,
        source="K",
        north_mask=NORTH_MASK,
        north_boundary=north_boundary,
        year=year,
        month=month,
        np_vmax=24,
        ann_rmse=True,
    )
    plot_response_dataset(
        nn_clr_closure,
        date_clear,
        source="NN",
        north_mask=NORTH_MASK,
        north_boundary=north_boundary,
        year=year,
        month=month,
        np_vmax=24,
        ann_rmse=True,
    )
    plot_response_dataset(
        kernel_clr_closure,
        date_clear,
        source="K",
        north_mask=NORTH_MASK,
        north_boundary=north_boundary,
        year=year,
        month=month,
        np_vmax=24,
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
        north_boundary=north_boundary,
        year=year,
        month=month,
        np_vmax=24,
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
        north_boundary=north_boundary,
        year=year,
        month=month,
        np_vmax=24,
    )


def feedback_test(
    temperature_anomaly: xr.Dataset,
    responses: xr.Dataset,
    output_root: Path,
) -> None:
    output_root.mkdir(exist_ok=True, parents=True)
    output_clr = output_root / "clr"
    output_all = output_root / "all"
    output_clr.mkdir(exist_ok=True, parents=True)
    output_all.mkdir(exist_ok=True, parents=True)

    responses["dt2m"] = temperature_anomaly  # (lat, lon, year, month) -> (year, month)
    responses_global_mean = global_date_series(responses)
    responses_global_mean = responses_global_mean.set_coords("dt2m")
    print(responses_global_mean)

    respones_regression_results = responses_global_mean.polyfit("dt2m", deg=1, cov=True)
    print(respones_regression_results)
    # respones_regression_results.to_netcdf(output_root / 'feedbacks.nc')

    xmin, xmax = (
        responses_global_mean.dt2m.min().values,
        responses_global_mean.dt2m.max().values,
    )
    for var in responses_global_mean.data_vars:
        responses_global_mean[var].plot.scatter(x="dt2m")

        m = respones_regression_results[f"{var}_polyfit_coefficients"].sel(degree=1)
        b = respones_regression_results[f"{var}_polyfit_coefficients"].sel(degree=0)
        plt.plot([xmin, xmax], [xmin * m + b, xmax * m + b], linestyle="--", alpha=0.3)
        plt.savefig(output_root / f"reg_{var}.png")
        plt.close()
