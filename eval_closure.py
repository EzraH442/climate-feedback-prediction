import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

from config_utils import load_config
from model import SimpleModel
from preprocess import load_ecod
from preprocessing import create_2024_preprocessor_no_downscaling
from utils import (
    SECONDS_PER_DAY,
    global_date_series,
    global_mean,
    integrate_over_pressure_levels,
    nn_radiative_response,
    nn_radiative_response_cross,
    plot_global_field,
    plot_north_pole_field,
    setup_timeseries_plot,
    to_monthly,
    weighted_residuals_by_month,
    weighted_residuals_by_year_month,
)


NORTH_BOUNDARY = 75
NORTH_MASK = None
ALBEDO_KERNEL_PATH = Path("data/ERA5_kernels/ERA5_kernel_fal_TOA.nc")
WATER_VAPOR_KERNEL_PATH = Path(
    "data/ERA5_kernels/layer_specified_ta_wv_kernel/ERA5_kernel_wv_sw_nodp_TOA.nc"
)
QT_PATH = Path("data/era5/era5_plev_qt_monthly_downscaled.nc")
MONTH_NAMES = [
    "Jan", "Feb", "Mar",
    "Apr", "May", "Jun",
    "Jul", "Aug", "Sep",
    "Oct", "Nov", "Dec",
]


def load_model_and_preprocessor(config, checkpoint_path: Path):
    checkpoint_data = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model_config = checkpoint_data.get("config", config)
    model = SimpleModel(model_config)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    preprocessor = create_2024_preprocessor_no_downscaling(
        input_vars=list(config.dataset.input_vars),
        target_var=config.dataset.target_var,
        ecod=config.preprocess.ecod.enabled,
    )
    preprocessor.load(model.config.preprocess.params_dir)

    return model, preprocessor


def albedo_kernel_components(
    anomaly, albedo_kernel
) -> tuple[xr.DataArray, xr.DataArray]:
    da = anomaly.fal.compute()
    dR_a_k = albedo_kernel.TOA_all * da * 100
    dR_a_k_clr = albedo_kernel.TOA_clr * da * 100
    return dR_a_k, dR_a_k_clr


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

    dR_q_k = dR_q_k.compute()
    dR_q_k_clr = dR_q_k_clr.compute()

    return dR_q_k, dR_q_k_clr


def plot_field_pair(
    field,
    title: str,
    save_path: Path,
    np_save_path: Path,
    vmax: float,
    label: str,
    np_vmax: float | None = None,
    ann_rmse = False,
) -> None:
    if NORTH_MASK is None:
        raise RuntimeError("NORTH_MASK must be initialized before plotting.")
    lon, lat = field.longitude, field.latitude
    field_masked = field.isel(latitude=NORTH_MASK)
    
    mean    = global_mean(field).values
    mean_np = global_mean(field_masked).values
    ann     = f"{mean:.2f}"
    ann_np  = f"{mean_np:.2f}"
    
    if ann_rmse:
        rmse_val    = np.sqrt(global_mean(field * field).values)
        rmse_val_np = np.sqrt(global_mean(field_masked * field_masked).values)
        
        ann    += f"; {rmse_val:.2f}"
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
        contours=True
    )


def plot_input_anomalies(
    anomaly,
    anomaly_ecod,
    dR_clr,
    output_dir: Path,
    year: int,
    month: int,
) -> None:
    specs = [
        ("fal", "", 0.5),
        ("hcc", "", None),
        ("mcc", "", None),
        ("lcc", "", None),
        ("tcwv", "kg/m^2", None),
        ("tco3", "kg/m^2", None),
        ("tsr", "W/m^2", 24),
    ]
    for var, label, fixed_vmax in specs:
        field = anomaly.sel(month=month, year=year)[var].compute()
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
        ("tsrc", dR_clr.sel(month=month, year=year), "tsrc"),
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
    year: int,
    month: int,
    source: str,
    ann_rmse = False,
) -> None:
    for name, response in dataset.data_vars.items():
        label = response.attrs.get("plot_label", name)
        filename = response.attrs["filename"]
        vmax = response.attrs.get("vmax")
        field = response.sel(month=month, year=year)
        if vmax is None:
            vmax = float(np.max(np.abs(field)))
        plot_field_pair(
            field,
            title=rf"$\Delta R_{{{label}}}^{{{source}}}$",
            save_path=output_dir / filename,
            np_save_path=output_dir / "np" / filename,
            vmax=vmax,
            label="W/m^2",
            np_vmax=24,
            ann_rmse=ann_rmse,
        )


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


def plot_residual_rmse_timeseries(series, output_dir: Path, clear_sky: bool) -> None:
    fig, ax = setup_timeseries_plot()
    ylabel = (
        rf"RMSE $\Delta R_{{net,clr}}$ ($W m^{{-2}}$)"
        if clear_sky
        else rf"RMSE $\Delta R_{{net}}$ ($W m^{{-2}}$)"
    )
    for source in ["nn", "k"]:
        name = f"dR_res_{source}" + ("_clr" if clear_sky else "")
        ax.scatter(x=series[name]["date"], y=series[name], s=1, label=source)
    ax.set_yticks(np.arange(0, 11, 2))
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
    for source in ["nn", "k"]:
        name = f"dR_res_{source}" + ("_clr" if clear_sky else "")
        ax.scatter(x=series[name]["date"], y=series[name], s=1, label=source)
    ax.set_yticks(np.arange(0, 11, 2))
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
    dR_sum_k = dR_a_k + dR_c_k + dR_q_k
    dR_sum_nn_clr = dR_a_nn_clr + dR_q_nn_clr
    dR_sum_k_clr = dR_a_k_clr + dR_q_k_clr

    dR_res_nn = dR - dR_sum_nn
    dR_res_k = dR - dR_sum_k
    dR_res_nn_clr = dR_clr - dR_sum_nn_clr
    dR_res_k_clr = dR_clr - dR_sum_k_clr

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
    plot_component_timeseries(
        component_series, timeseries_all, clear_sky=False
    )
    plot_component_timeseries(
        component_series, timeseries_clear, clear_sky=True
    )
    plot_component_timeseries(
        component_series_np, timeseries_all / "np", clear_sky=False
    )
    plot_component_timeseries(
        component_series_np, timeseries_clear / "np", clear_sky=True
    )

    plot_net_timeseries(
        {
            "nn": global_date_series(dR_sum_nn),
            "kernel": global_date_series(dR_sum_k),
            "era5": global_date_series(dR),
        },
        timeseries_all / "timeseries_net.png",
    )
    plot_net_timeseries(
        {
            "nn": global_date_series(dR_sum_nn_clr),
            "kernel": global_date_series(dR_sum_k_clr),
            "era5": global_date_series(dR_clr),
        },
        timeseries_clear / "timeseries_net.png",
    )
    plot_net_timeseries(
        {
            "nn": global_date_series(dR_sum_nn.isel(latitude=NORTH_MASK)),
            "kernel": global_date_series(dR_sum_k.isel(latitude=NORTH_MASK)),
            "era5": global_date_series(dR.isel(latitude=NORTH_MASK)),
        },
        timeseries_all / "np" / "timeseries_net.png",
    )
    plot_net_timeseries(
        {
            "nn": global_date_series(dR_sum_nn_clr.isel(latitude=NORTH_MASK)),
            "kernel": global_date_series(dR_sum_k_clr.isel(latitude=NORTH_MASK)),
            "era5": global_date_series(dR_clr.isel(latitude=NORTH_MASK)),
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
        dR_res_nn_cross = dR - dR_sum_nn_cross

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axhline(0, alpha=0.1)
        colors = ["tab:green", "tab:blue", "tab:orange", "purple", "tab:red"]
        net_series = {
            "era5": global_date_series(dR),
            "nn": global_date_series(dR_sum_nn),
            "kernel": global_date_series(dR_sum_k),
            "nn_cross": global_date_series(dR_sum_nn_cross),
            "cross": global_date_series(dR_aq_nn + dR_ac_nn + dR_qc_nn),
        }
        for i, (name, series) in enumerate(net_series.items()):
            ax.scatter(
                x=series["date"],
                y=series,
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

    residual_series = {
        "dR_res_nn": global_date_series(dR_res_nn),
        "dR_res_k": global_date_series(dR_res_k),
        "dR_res_nn_clr": global_date_series(dR_res_nn_clr),
        "dR_res_k_clr": global_date_series(dR_res_k_clr),
    }
    residual_series_np = {
        "dR_res_nn": global_date_series(dR_res_nn.isel(latitude=NORTH_MASK)),
        "dR_res_k": global_date_series(dR_res_k.isel(latitude=NORTH_MASK)),
        "dR_res_nn_clr": global_date_series(dR_res_nn_clr.isel(latitude=NORTH_MASK)),
        "dR_res_k_clr": global_date_series(dR_res_k_clr.isel(latitude=NORTH_MASK)),
    }
    plot_residual_mbe_timeseries(
        residual_series, timeseries_all, clear_sky=False
    )
    plot_residual_mbe_timeseries(
        residual_series, timeseries_clear, clear_sky=True
    )
    plot_residual_mbe_timeseries(
        residual_series_np, timeseries_all / "np", clear_sky=False
    )
    plot_residual_mbe_timeseries(
        residual_series_np, timeseries_clear / "np", clear_sky=True
    )

    residual_series = {
        "dR_res_nn": np.sqrt(global_date_series(np.power(dR_res_nn, 2))),
        "dR_res_k": np.sqrt(global_date_series(np.power(dR_res_k, 2))),
        "dR_res_nn_clr": np.sqrt(global_date_series(np.power(dR_res_nn_clr, 2))),
        "dR_res_k_clr": np.sqrt(global_date_series(np.power(dR_res_k_clr, 2))),
    }
    residual_series_np = {
        "dR_res_nn": np.sqrt(
            global_date_series(np.power(dR_res_nn.isel(latitude=NORTH_MASK), 2))
        ),
        "dR_res_k": np.sqrt(
            global_date_series(np.power(dR_res_k.isel(latitude=NORTH_MASK), 2))
        ),
        "dR_res_nn_clr": np.sqrt(
            global_date_series(np.power(dR_res_nn_clr.isel(latitude=NORTH_MASK), 2))
        ),
        "dR_res_k_clr": np.sqrt(
            global_date_series(np.power(dR_res_k_clr.isel(latitude=NORTH_MASK), 2))
        ),
    }
    plot_residual_rmse_timeseries(
        residual_series, timeseries_all, clear_sky=False
    )
    plot_residual_rmse_timeseries(
        residual_series, timeseries_clear, clear_sky=True
    )
    plot_residual_rmse_timeseries(
        residual_series_np, timeseries_all / "np", clear_sky=False
    )
    plot_residual_rmse_timeseries(
        residual_series_np, timeseries_clear / "np", clear_sky=True
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

    if not has_cross:
        return

    rng = np.random.default_rng(0)
    residuals = {
        "nn": weighted_residuals_by_month(dR_res_nn, residual_samples, rng),
        "nn_cross": weighted_residuals_by_month(dR_res_nn_cross, residual_samples, rng),
        "kernel": weighted_residuals_by_month(dR_res_k, residual_samples, rng),
    }
    residuals_np = {
        "nn": weighted_residuals_by_month(
            dR_res_nn.isel(latitude=NORTH_MASK), residual_samples, rng
        ),
        "nn_cross": weighted_residuals_by_month(
            dR_res_nn_cross.isel(latitude=NORTH_MASK), residual_samples, rng
        ),
        "kernel": weighted_residuals_by_month(
            dR_res_k.isel(latitude=NORTH_MASK), residual_samples, rng
        ),
    }
    plot_residual_boxplots(residuals, residuals_np, timeseries_all)


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
) -> None:
    if NORTH_MASK is None:
        raise RuntimeError("NORTH_MASK must be initialized before date_closure_test.")
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

    plot_input_anomalies(
        anomaly,
        anomaly_ecod,
        dR_clr,
        date_root,
        year,
        month,
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
        responses[["dR_a_nn_all", "dR_c_nn_all", "dR_q_nn_all"]].rename({
            "dR_a_nn_all": "a",
            "dR_c_nn_all": "c",
            "dR_q_nn_all": "q",
        }),
        date_all,
        year,
        month,
        "NN",
    )
    plot_response_dataset(
        responses[["dR_a_nn_clr", "dR_q_nn_clr"]].rename(
            {"dR_a_nn_clr": "a", "dR_q_nn_clr": "q"}
        ),
        date_clear,
        year,
        month,
        "NN",
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
        year,
        month,
        "K",
    )
    plot_response_dataset(
        responses[["dR_a_k_clr", "dR_q_k_clr"]].rename(
            {"dR_a_k_clr": "a", "dR_q_k_clr": "q"}
        ),
        date_clear,
        year,
        month,
        "K",
    )

    dR_sum_nn              = responses["dR_a_nn_all"] + responses["dR_c_nn_all"] + responses["dR_q_nn_all"]
    dR_sum_nn_allcross     = responses["dR_nn_all"]
    dR_sum_k               = responses["dR_a_k_all"]  + responses["dR_c_k_all"]  + responses["dR_q_k_all"]
    dR_sum_nn_clr          = responses["dR_a_nn_clr"] + responses["dR_q_nn_clr"]
    dR_sum_nn_allcross_clr = responses["dR_nn_clr"]
    dR_sum_k_clr           = responses["dR_a_k_clr"]  + responses["dR_q_k_clr"]

    dR_res_nn_allcross     = dR     - responses['dR_nn_all']
    dR_res_nn              = dR     - dR_sum_nn
    dR_res_k               = dR     - dR_sum_k
    dR_res_nn_clr          = dR_clr - dR_sum_nn_clr
    dR_res_nn_allcross_clr = dR     - responses['dR_nn_clr']
    dR_res_k_clr           = dR_clr - dR_sum_k_clr

    nn_all_closure = xr.Dataset({
        "sum": dR_sum_nn.assign_attrs(plot_label="sum", filename="dR_sum.png", vmax=55),
        "sum_allcross": dR_sum_nn_allcross.assign_attrs(plot_label="sum,allcross", filename="dR_sum_allcross.png", vmax=55),
        "res": dR_res_nn.assign_attrs(plot_label="res", filename="dR_res.png", vmax=24),
        "res_allcross": dR_res_nn_allcross.assign_attrs(plot_label="res,allcross", filename="dR_res_allcross.png", vmax=24),
    })

    kernel_all_closure = xr.Dataset({
        "sum": dR_sum_k.assign_attrs(plot_label="sum", filename="k_dR_sum.png", vmax=55),
        "res": dR_res_k.assign_attrs(plot_label="res", filename="k_dR_res.png", vmax=24),
    })

    nn_clr_closure = xr.Dataset({
        "sum_clr": dR_sum_nn_clr.assign_attrs(plot_label="sum,clr", filename="dR_sum,clr.png", vmax=55),
        "sum_allcross_clr": dR_sum_nn_allcross_clr.assign_attrs(plot_label="sum,allcross,clr", filename="dR_sum_allcross,clr.png", vmax=55),
        "res_clr": dR_res_nn_clr.assign_attrs(plot_label="res,clr", filename="dR_res,clr.png", vmax=24),
        "res_allcross_clr": dR_res_nn_allcross_clr.assign_attrs(plot_label="res,allcross,clr", filename="dR_res_allcross,clr.png", vmax=24),
    })

    kernel_clr_closure = xr.Dataset({
        "sum_clr": dR_sum_k_clr.assign_attrs(plot_label="sum,clr", filename="k_dR_sum,clr.png", vmax=60),
        "res_clr": dR_res_k_clr.assign_attrs(plot_label="res,clr", filename="k_dR_res,clr.png", vmax=24),
    })

    plot_response_dataset(nn_all_closure,      date_all,   year, month, "NN", ann_rmse=True)
    plot_response_dataset(kernel_all_closure,  date_all,   year, month, "K",  ann_rmse=True)
    plot_response_dataset(nn_clr_closure,      date_clear, year, month, "NN", ann_rmse=True)
    plot_response_dataset(kernel_clr_closure,  date_clear, year, month, "K",  ann_rmse=True)

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
        year,
        month,
        "NN",
    )

    dR_aq_nn = responses["dR_aq_nn_all"]
    dR_ac_nn = responses["dR_ac_nn_all"]
    dR_qc_nn = responses["dR_qc_nn_all"]
    dR_sum_nn_cross = dR_sum_nn + dR_aq_nn + dR_ac_nn + dR_qc_nn
    dR_res_nn_cross = dR - dR_sum_nn_cross

    cross_closure = xr.Dataset({
        "sum": dR_sum_nn_cross.assign_attrs(plot_label="sum,cross", filename="cross_dR_sum.png", vmax=55),
        "res": dR_res_nn_cross.assign_attrs(plot_label="res,cross", filename="cross_dR_res.png", vmax=55),
    })
    plot_response_dataset(
        cross_closure,
        date_all,
        year,
        month,
        "NN",
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
    
    dR = responses["dR_era5_all"]
    dR_clr = responses["dR_era5_clr"]

    responses['dt2m'] = temperature_anomaly # (lat, lon, year, month) -> (year, month)
    responses_global_mean = global_date_series(responses)
    responses_global_mean = responses_global_mean.set_coords('dt2m')
    print(responses_global_mean)

    respones_regression_results = responses_global_mean.polyfit('dt2m', deg=1, cov=True)
    print(respones_regression_results)
    #respones_regression_results.to_netcdf(output_root / 'feedbacks.nc')  

    xmin, xmax = responses_global_mean.dt2m.min().values, responses_global_mean.dt2m.max().values
    for var in responses_global_mean.data_vars:
        responses_global_mean[var].plot.scatter(x='dt2m')

        m = respones_regression_results[f"{var}_polyfit_coefficients"].sel(degree=1)
        b = respones_regression_results[f"{var}_polyfit_coefficients"].sel(degree=0)
        plt.plot([xmin, xmax], [xmin * m + b, xmax * m + b], linestyle='--', alpha=0.3)
        plt.savefig(output_root / f'reg_{var}.png')
        plt.close()

def main():
    global NORTH_BOUNDARY, NORTH_MASK

    parser = argparse.ArgumentParser(description="Run closure-test analysis.")
    parser.add_argument(
        "--config_file",
        default="configs/model/fal/2011-2014_3,6,9,12_sob_fal.yaml",
        help="Path to OmegaConf YAML config.",
    )
    parser.add_argument("--checkpoint_path", help="Path to model checkpoint.")
    parser.add_argument("--output_dir", help="Output directory.")
    parser.add_argument("--year", type=int, default=2012)
    parser.add_argument("--month", type=int, default=9)
    parser.add_argument("--start_date", default="2007-01")
    parser.add_argument("--end_date", default="2016-12")
    parser.add_argument("--data_path", default="data/era5")
    parser.add_argument("--north_boundary", type=float, default=NORTH_BOUNDARY)
    parser.add_argument("--residual_samples", type=int, default=2000)
    parser.add_argument("--skip_cross", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config_file)
    year, month = args.year, args.month
    output_root = (
        Path(args.output_dir)
        if args.output_dir
        else Path(config.train.checkpoint_dir) / "figures" / "closure_test"
    )
    response_save_path = output_root / "saved_responses_closure_test.nc"
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else Path(config.train.checkpoint_dir) / "best_model.pt"
    )
    model, preprocessor = load_model_and_preprocessor(config, checkpoint_path)

    K_a = xr.open_dataset(ALBEDO_KERNEL_PATH)
    K_q = xr.open_dataset(WATER_VAPOR_KERNEL_PATH)
    kernel_grid = {"latitude": K_a.latitude, "longitude": K_a.longitude}
    NORTH_BOUNDARY = args.north_boundary
    NORTH_MASK = K_a.latitude.values > NORTH_BOUNDARY

    data_path = Path(args.data_path)
    ds = xr.open_mfdataset(list(data_path.glob("era5_single_levels_monthly_*.nc")))
    ds = ds.sel(date=slice(args.start_date, args.end_date)).interp(**kernel_grid)
    if config.preprocess.ecod.enabled:
        ds = ds.assign(
            ecod=load_ecod(
                args.data_path,
                range(
                    pd.Timestamp(args.start_date).year,
                    pd.Timestamp(args.end_date).year + 1,
                ),
                range(1, 13),
                config.preprocess.ecod.method == "fast",
            ).interp(**kernel_grid)
        )
    ds["tsr"] = ds.tsr / SECONDS_PER_DAY
    ds["tsrc"] = ds.tsrc / SECONDS_PER_DAY

    ds_monthly = to_monthly(ds.copy(deep=True))
    ds_monthly_means = ds_monthly.mean("year")
    anomaly = ds_monthly - ds_monthly_means
    # print(ds_monthly_means)

    cloud_vars = ["tcc", "hcc", "mcc", "lcc", "tciw", "tclw"]
    if config.preprocess.ecod.enabled:
        cloud_vars.append("ecod")
    cloud_vars = [v for v in cloud_vars if v in ds_monthly]
    ds_monthly_clr = ds_monthly.assign(
        {v: xr.zeros_like(ds_monthly[v]) for v in cloud_vars}
    )
    ds_monthly_means_clr = ds_monthly_clr.mean('year')
    anomaly_clr = anomaly.assign({v: xr.zeros_like(anomaly[v]) for v in cloud_vars})

    dR_clr = ds_monthly.tsrc - ds_monthly.tsrc.mean("year")
    dR = anomaly.tsr.interp(**kernel_grid).compute()

    if response_save_path.exists():
        responses = xr.load_dataset(response_save_path)
    else:
        dR_a_nn, dR_c_nn, dR_q_nn, dR_nn_all = nn_radiative_response(
            ds_monthly_means,
            anomaly,
            model,
            preprocessor,
            config,
            ["fal", cloud_vars, "tcwv"],
        )
        dR_a_nn_clr, dR_q_nn_clr, dR_nn_clr = nn_radiative_response(
            ds_monthly_means_clr,
            anomaly_clr,
            model,
            preprocessor,
            config,
            ["fal", "tcwv"],
        )
    
        ds_qt = xr.load_dataset(QT_PATH)
        dR_a_k, dR_a_k_clr = albedo_kernel_components(anomaly, K_a)
        dR_q_k, dR_q_k_clr = water_vapor_kernel_components(ds_monthly, ds_qt, K_q)
        dR_c_k = (dR - dR_clr) - (dR_a_k - dR_a_k_clr) - (dR_q_k - dR_q_k_clr)
    
        cross_data_vars = xr.Dataset()
        if not args.skip_cross:
            variable_pairs = [
                ("fal", "tcwv"),
                ("fal", ["hcc", "mcc", "lcc", "tcc", "tciw", "tclw"]),
                ("tcwv", ["hcc", "mcc", "lcc", "tcc", "tciw", "tclw"]),
            ]
            dR_aq_nn, dR_ac_nn, dR_qc_nn = nn_radiative_response_cross(
                ds_monthly_means,
                anomaly,
                model,
                preprocessor,
                config,
                variable_pairs,
            )
            cross_data_vars = xr.Dataset(
                {
                    "dR_aq_nn_all": dR_aq_nn.assign_attrs(
                        plot_label="a,q", filename="cross_dR_a,q.png"
                    ),
                    "dR_ac_nn_all": dR_ac_nn.assign_attrs(
                        plot_label="a,c", filename="cross_dR_a,c.png"
                    ),
                    "dR_qc_nn_all": dR_qc_nn.assign_attrs(
                        plot_label="q,c", filename="cross_dR_q,c.png"
                    ),
                }
            )
    
        response_attrs = {
            "dR_era5_all": {"plot_label": "ERA5 all",   "filename": "dR_era5_all.png"},
            "dR_era5_clr": {"plot_label": "ERA5 clear", "filename": "dR_era5_clr.png"},
            "dR_nn_all":   {"plot_label": "NN all",     "filename": "dR_nn_all.png"},
            "dR_nn_clr":   {"plot_label": "NN clr",     "filename": "dR_nn_clr.png"},
            "dR_a_nn_all": {"plot_label": "a",          "filename": "dR_a.png",       "vmax": 40},
            "dR_c_nn_all": {"plot_label": "c",          "filename": "dR_c.png",       "vmax": 60},
            "dR_q_nn_all": {"plot_label": "q",          "filename": "dR_q.png",       "vmax": 7},
            "dR_a_nn_clr": {"plot_label": "a,clr",      "filename": "dR_a,clr.png",   "vmax": 60},
            "dR_q_nn_clr": {"plot_label": "q,clr",      "filename": "dR_q,clr.png",   "vmax": 4},
            "dR_a_k_all":  {"plot_label": "a",          "filename": "k_dR_a.png",     "vmax": 40},
            "dR_c_k_all":  {"plot_label": "c",          "filename": "k_dR_c.png",     "vmax": 60},
            "dR_q_k_all":  {"plot_label": "q",          "filename": "k_dR_q.png",     "vmax": 7},
            "dR_a_k_clr":  {"plot_label": "a,clr",      "filename": "k_dR_a,clr.png", "vmax": 40},
            "dR_q_k_clr":  {"plot_label": "q,clr",      "filename": "k_dR_q,clr.png", "vmax": 7},
        }
        bad_attr_names = [
            key
            for attrs in response_attrs.values()
            for key in attrs
            if key.strip() != key
        ]
        if bad_attr_names:
            raise ValueError(f"Illegal response attribute names: {bad_attr_names}")
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
        responses = xr.merge([base_responses, cross_data_vars])
        # print(responses)
        # print(list(responses.data_vars))
        # print(responses.attrs)
        output_root.mkdir(exist_ok=True, parents=True)
        responses.to_netcdf(response_save_path)
    
    
    date_closure_test(
        anomaly,
        ds_monthly,
        ds_monthly_means,
        preprocessor,
        responses,
        output_root,
        year,
        month,
    )
    timeseries_test(
        responses,
        output_root,
        args.residual_samples,
    )

    """
    dt2m = ds_monthly.t2m - ds_monthly.t2m.mean("year")
    feedback_test(
        dt2m,
        responses,
        output_root,
    )
"""

if __name__ == "__main__":
    main()
