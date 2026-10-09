from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import xarray as xr

from config_utils import VariableConfig, load_config, variable_config_from_omegaconf
from experiments.kernel_difference.analysis import (
    plot_hybrid_kernel_difference,
    plot_kernel_difference,
)
from experiments.kernel_difference.main import (
    compute_hybrid_kernel_difference,
    compute_kernel_difference,
)
from experiments.radiative_closure_single_date.analysis import (
    arctic_field,
    load_feedback_ablation_responses,
    make_neurips_feedback_context,
    shakirova_terms,
)
from model import SimpleModel
from preprocessing import Preprocessor
from utils import (
    SECONDS_PER_DAY,
    compute_nn_kernel,
    compute_nn_kernel_autograd,
    figure_dir,
    generate_paths_yearly,
    interpolate_spatial_field,
    kernel_delta,
    kernel_label,
    kernel_title,
    load_model_and_preprocessor,
    make_era5_filename,
    make_kernel_filename,
    nn_pred,
    plot_global_field,
    plot_north_pole_field,
    scale_minmax_value,
    target_flux,
)

TABLE_MODEL_SPECS = [
    ("NN", Path("configs/model/fal/2011-2014_3,6,9,12_baseline.yaml")),
    (
        "NN + clear-sky",
        Path("configs/model/fal/2011-2014_3,6,9,12_baseline_clearsky.yaml"),
    ),
    (
        "NN + clear-sky + Sob.",
        Path("configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky.yaml"),
    ),
]
SEED_AGGREGATED_MODELS = ["NN", "NN + clear-sky", "NN + clear-sky + Sob."]
SOBOLEV_LAMBDA_CONFIG = Path("configs/experiments/fal/sobolev_lambda.yaml")
SOBOLEV_ABLATION_CHECKPOINT_ROOT = Path("sobolev-ablation/checkpoints")
SOBOLEV_LAMBDA_SPECS = [
    (
        "lambda=0.25",
        SOBOLEV_LAMBDA_CONFIG,
        SOBOLEV_ABLATION_CHECKPOINT_ROOT
        / "2011-2014_3,6,9,12_sob_fal_0.25_clearsky",
    ),
    (
        "lambda=0.5",
        SOBOLEV_LAMBDA_CONFIG,
        SOBOLEV_ABLATION_CHECKPOINT_ROOT
        / "2011-2014_3,6,9,12_sob_fal_0.5_clearsky",
    ),
    (
        "lambda=1",
        SOBOLEV_LAMBDA_CONFIG,
        SOBOLEV_ABLATION_CHECKPOINT_ROOT
        / "2011-2014_3,6,9,12_sob_fal_1_clearsky",
    ),
    (
        "lambda=2",
        SOBOLEV_LAMBDA_CONFIG,
        SOBOLEV_ABLATION_CHECKPOINT_ROOT
        / "2011-2014_3,6,9,12_sob_fal_2_clearsky",
    ),
    (
        "lambda=4",
        SOBOLEV_LAMBDA_CONFIG,
        SOBOLEV_ABLATION_CHECKPOINT_ROOT
        / "2011-2014_3,6,9,12_sob_fal_4_clearsky",
    ),
]


def global_tsr_test(
    ds: xr.Dataset,
    preprocessor: Preprocessor,
    model: SimpleModel,
    vc: VariableConfig,
    figures_path: Path = Path("."),
):
    lat = ds.latitude.values
    lon = ds.longitude.values

    target = vc.target_var
    target_label = "TSR" if target == "pal" else target.upper()
    output_dir = figure_dir(figures_path, "all", "tsr_test")

    pred = nn_pred(ds, model, preprocessor, vc, ["date", "latitude", "longitude"])
    true = target_flux(ds, preprocessor, target)

    tsr_true = true.to_numpy() / SECONDS_PER_DAY
    tsr_pred = pred.to_numpy() / SECONDS_PER_DAY
    diff_full = tsr_pred - tsr_true

    tsr_mean = np.mean(tsr_true, axis=0)
    tsr_pred_mean = np.mean(tsr_pred, axis=0)
    mbe_map = np.mean(diff_full, axis=0)
    rmse_map = np.sqrt(np.mean(diff_full**2, axis=0))

    min_tsr = min(np.min(tsr_mean), np.min(tsr_pred_mean))
    max_tsr = max(np.max(tsr_mean), np.max(tsr_pred_mean))
    max_abs_mbe = np.max(np.abs(mbe_map))
    max_rmse = np.max(rmse_map)

    global_tsr_mean = np.mean(tsr_mean)
    global_tsr_pred_mean = np.mean(tsr_pred_mean)
    global_mbe = np.mean(mbe_map)
    global_rmse = np.sqrt(np.mean(rmse_map**2))

    print(f"Global MBE :  {global_mbe:.4f} W/m²")
    print(f"Max MBE    :  {max_abs_mbe:.4f} W/m²")
    print(f"Global RMSE:  {global_rmse:.4f} W/m²")
    print(f"Max RMSE   :  {max_rmse:.4f} W/m²")

    plot_global_field(
        tsr_mean,
        lon,
        lat,
        f"{target_label} (ERA5)",
        output_dir / "era5.png",
        cmap="Spectral",
        vmin=min_tsr,
        vmax=max_tsr,
        label="$W/m^2$",
        annotation=f"{global_tsr_mean:.2f}",
    )
    plot_global_field(
        tsr_pred_mean,
        lon,
        lat,
        f"{target_label} (NN)",
        output_dir / "nn.png",
        cmap="Spectral",
        vmin=min_tsr,
        vmax=max_tsr,
        label="$W/m^2$",
        annotation=f"{global_tsr_pred_mean:.2f}",
    )
    plot_global_field(
        mbe_map,
        lon,
        lat,
        "MBE",
        output_dir / "mbe.png",
        cmap="RdBu_r",
        vmin=-20,
        vmax=20,
        label="$W/m^2$",
        annotation=f"{global_mbe:.2f}",
    )
    plot_global_field(
        rmse_map,
        lon,
        lat,
        "RMSE",
        output_dir / "rmse.png",
        cmap="Blues",
        vmin=0,
        vmax=20,
        label="$W/m^2$",
        annotation=f"{global_rmse:.2f}",
    )

    # slope, intercept, r_value, p_value, std_err = linregress(
    #     predictions_inversed, dataset["tsr"].values
    # )
    # r_squared = r_value**2
    # mse = np.mean((dataset["tsr"].values - predictions_inversed) ** 2)
    # rmse = np.sqrt(mse)

    # val_str = f"$R^2$ = {r_squared:.3f} \nRMSE = {rmse:.2f} $W/m^2$"

    # fig, ax = setup_correlation_plot()
    # plot_correlation(
    #     ax, predictions_inversed.flatten(), dataset["tsr"].values.flatten()
    # )

    # ax.set_ylabel("TSR from ERA5 $[W/m^2$]")
    # ax.set_xlabel("TSR predicted by the NN [$W/m^2$]")
    # ax.set_title("Validation of the climatological TSR")

    # ax.text(0, 370, val_str, fontsize=12)
    # fig.savefig("correlation.png")


def global_tsrc_test(
    raw_ds: xr.Dataset,
    preprocessor: Preprocessor,
    model: SimpleModel,
    vc: VariableConfig,
    figures_path: Path = Path("."),
):
    clear_processed = preprocessor.transform(vc.clear_sky_input(raw_ds))
    lat = clear_processed.latitude.values
    lon = clear_processed.longitude.values

    target = vc.clear_sky_target
    target_label = "TSRC" if target == "palc" else target.upper()
    output_dir = figure_dir(figures_path, "clr", "tsrc_test")

    pred = nn_pred(
        clear_processed,
        model,
        preprocessor,
        vc,
        ["date", "latitude", "longitude"],
        clear=True,
    )
    true = target_flux(clear_processed, preprocessor, target)

    tsrc_true = true.to_numpy() / SECONDS_PER_DAY
    tsrc_pred = pred.to_numpy() / SECONDS_PER_DAY
    diff_full = tsrc_pred - tsrc_true

    tsrc_mean = np.mean(tsrc_true, axis=0)
    tsrc_pred_mean = np.mean(tsrc_pred, axis=0)
    mbe_map = np.mean(diff_full, axis=0)
    rmse_map = np.sqrt(np.mean(diff_full**2, axis=0))

    min_tsrc = min(np.min(tsrc_mean), np.min(tsrc_pred_mean))
    max_tsrc = max(np.max(tsrc_mean), np.max(tsrc_pred_mean))
    max_abs_mbe = np.max(np.abs(mbe_map))
    max_rmse = np.max(rmse_map)

    global_tsrc_mean = np.mean(tsrc_mean)
    global_tsrc_pred_mean = np.mean(tsrc_pred_mean)
    global_mbe = np.mean(mbe_map)
    global_rmse = np.sqrt(np.mean(rmse_map**2))

    print(f"Global MBE :  {global_mbe:.4f} W/m²")
    print(f"Max MBE    :  {max_abs_mbe:.4f} W/m²")
    print(f"Global RMSE:  {global_rmse:.4f} W/m²")
    print(f"Max RMSE   :  {max_rmse:.4f} W/m²")

    plot_global_field(
        tsrc_mean,
        lon,
        lat,
        f"{target_label} (ERA5)",
        output_dir / "era5.png",
        cmap="Spectral",
        vmin=min_tsrc,
        vmax=max_tsrc,
        label="$W/m^2$",
        annotation=f"{global_tsrc_mean:.2f}",
    )
    plot_global_field(
        tsrc_pred_mean,
        lon,
        lat,
        f"{target_label} (NN)",
        output_dir / "nn.png",
        cmap="Spectral",
        vmin=min_tsrc,
        vmax=max_tsrc,
        label="$W/m^2$",
        annotation=f"{global_tsrc_pred_mean:.2f}",
    )
    plot_global_field(
        mbe_map,
        lon,
        lat,
        "MBE",
        output_dir / "mbe.png",
        cmap="RdBu_r",
        vmin=-20,
        vmax=20,
        label="$W/m^2$",
        annotation=f"{global_mbe:.2f}",
    )
    plot_global_field(
        rmse_map,
        lon,
        lat,
        "RMSE",
        output_dir / "rmse.png",
        cmap="Blues",
        vmin=0,
        vmax=20,
        label="$W/m^2$",
        annotation=f"{global_rmse:.2f}",
    )


def kernel_ecod_fal_contour_test(
    processed_ds: xr.Dataset,
    preprocessor: Preprocessor,
    model: SimpleModel,
    config: VariableConfig,
    figures_path: Path = Path("."),
    date="2015-09",
    latitude=83.625,
    longitude=17.375,
    scatter=True,
    extrapolate=False,
    n=100,
):
    return kernel_contour_test(
        "fal",
        "ecod",
        processed_ds,
        preprocessor,
        model,
        config,
        figures_path,
        date,
        latitude,
        longitude,
        scatter,
        extrapolate,
        n,
    )


def kernel_contour_test(
    v1,
    v2,
    processed_ds: xr.Dataset,
    preprocessor: Preprocessor,
    model: SimpleModel,
    config: VariableConfig,
    figures_path: Path = Path("."),
    date="2015-09",
    latitude=83.625,
    longitude=17.375,
    scatter=True,
    extrapolate=False,
    n=100,
):
    scaler = preprocessor.scalar
    feature_names = config.input_order()

    ordered = config.inputs(processed_ds)

    base_point = ordered.sel(date=date, latitude=latitude, longitude=longitude)
    observed_base_point = scaler.inverse_transform(base_point)

    base_features = np.array(
        [float(base_point[name]) for name in feature_names],
        dtype=np.float32,
    )

    vmin, vmax = scaler.get_data_min(), scaler.get_data_max()

    v1_values = np.linspace(float(vmin[v1]), float(vmax[v1]), n)
    v2_values = np.linspace(float(vmin[v2]), float(vmax[v2]), n)
    if extrapolate:
        v1_values = np.linspace(0, float(vmax[v1]), n)
        v2_values = np.linspace(0, float(vmax[v2]), n)

    v1_grid, v2_grid = np.meshgrid(v1_values, v2_values)

    inputs_np = np.broadcast_to(base_features, (n * n, len(feature_names))).copy()
    inputs_np[:, feature_names.index(v1)] = scale_minmax_value(
        scaler, v1, v1_grid.ravel()
    )
    inputs_np[:, feature_names.index(v2)] = scale_minmax_value(
        scaler, v2, v2_grid.ravel()
    )
    if "ecod_fal" in feature_names:
        inputs_np[:, feature_names.index("ecod_fal")] = scale_minmax_value(
            scaler, "ecod_fal", (v2_grid * v1_grid).ravel()
        )

    inputs = torch.from_numpy(inputs_np).float().requires_grad_(True)
    outputs = model(inputs)
    kernel_name = v1
    grads = torch.autograd.grad(outputs=outputs.sum(), inputs=inputs)[0][
        :, feature_names.index(kernel_name)
    ]

    target = config.target_var
    target_range = float(vmax[target] - vmin[target])
    input_range = float(vmax[kernel_name] - vmin[kernel_name])
    kernel = (
        grads.detach().cpu().numpy() * (target_range / input_range) / SECONDS_PER_DAY
    ).reshape(n, n) * kernel_delta(kernel_name)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    max_abs = float(np.nanmax(np.abs(kernel)))
    levels = np.linspace(-max_abs, max_abs, 31) if max_abs > 0 else 31
    contour = ax.contourf(
        v1_grid,
        v2_grid,
        kernel,
        levels=levels,
        cmap="RdBu_r",
        extend="both",
    )
    observed = scaler.inverse_transform(ordered[[v1, v2]])
    observed_v1 = observed[v1].to_numpy().ravel()
    observed_v2 = observed[v2].to_numpy().ravel()
    # step = max(1, observed_fal.size // 50000)
    if scatter:
        ax.scatter(
            observed_v1,  # [::step],
            observed_v2,  # [::step],
            s=1,
            c="black",
            alpha=0.05,
            linewidths=0,
            label="dataset",
        )
        ax.legend(loc="upper right", markerscale=4)
    ax.scatter(
        [float(observed_base_point[v1])],
        [float(observed_base_point[v2])],
        s=20,
        c="white",
        edgecolors="black",
    )
    ax.set_xlabel(v1)
    ax.set_ylabel(v2)
    ax.set_title(
        f"NN surface albedo kernel over {v1}/{v2}; lat={latitude:.2f}, lon={longitude:.2f}, date={date}"
    )
    cb = fig.colorbar(contour, ax=ax)
    cb.set_label(kernel_label(kernel_name))
    fig.tight_layout()
    output_dir = figure_dir(figures_path, "all", "kernel_contour_test", v1, date)
    fig.savefig(output_dir / f"{v1}_{v2}.png")
    plt.close(fig)


def tsr_ecod_fal_contour_test(
    processed_ds: xr.Dataset,
    preprocessor: Preprocessor,
    model: SimpleModel,
    config: VariableConfig,
    figures_path: Path = Path("."),
    date="2015-09",
    latitude=83.625,
    longitude=17.375,
    scatter=True,
    extrapolate=False,
    n_fal: int = 100,
    n_ecod: int = 100,
):
    return tsr_contour_test(
        "fal",
        "ecod",
        processed_ds,
        preprocessor,
        model,
        config,
        figures_path,
        date,
        latitude,
        longitude,
        scatter,
        extrapolate,
        n_fal,
    )


def tsr_contour_test(
    v1,
    v2,
    processed_ds: xr.Dataset,
    preprocessor: Preprocessor,
    model: SimpleModel,
    config: VariableConfig,
    figures_path: Path = Path("."),
    date="2015-09",
    latitude=83.625,
    longitude=17.375,
    scatter=True,
    extrapolate=False,
    n=100,
):
    scaler = preprocessor.scalar
    target_var = config.target_var
    feature_names = config.input_order()

    ordered = config.inputs(processed_ds)
    observed = scaler.inverse_transform(ordered)
    base_point = ordered.sel(date=date, latitude=latitude, longitude=longitude)
    observed_base_point = scaler.inverse_transform(base_point)

    base_features = np.array(
        [float(base_point[name]) for name in feature_names],
        dtype=np.float32,
    )

    vmin, vmax = scaler.get_data_min(), scaler.get_data_max()
    v1_values = np.linspace(float(vmin[v1]), float(vmax[v1]), n)
    v2_values = np.linspace(float(vmin[v2]), float(vmax[v2]), n)
    if extrapolate:
        v1_values = np.linspace(0, float(vmax[v1]), n)
        v2_values = np.linspace(0, float(vmax[v2]), n)

    v1_grid, v2_grid = np.meshgrid(v1_values, v2_values)

    inputs_np = np.broadcast_to(base_features, (n * n, len(feature_names))).copy()
    inputs_np[:, feature_names.index(v1)] = scale_minmax_value(
        scaler, v1, v1_grid.ravel()
    )
    inputs_np[:, feature_names.index(v2)] = scale_minmax_value(
        scaler, v2, v2_grid.ravel()
    )
    if "ecod_fal" in feature_names:
        inputs_np[:, feature_names.index("ecod_fal")] = scale_minmax_value(
            scaler, "ecod_fal", (v2_grid * v1_grid).ravel()
        )

    with torch.no_grad():
        outputs = model(torch.from_numpy(inputs_np).float()).numpy().reshape(n, n)

    target_min = float(vmin[target_var])
    target_range = float(vmax[target_var] - vmin[target_var])
    target_scaled = (outputs - scaler.min_val) / (scaler.max_val - scaler.min_val)
    target_values = (target_scaled * target_range + target_min) / SECONDS_PER_DAY

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    contour = ax.contourf(
        v1_grid,
        v2_grid,
        target_values,
        levels=31,
        cmap="Spectral",
    )
    if scatter:
        ax.scatter(
            observed[v1].to_numpy().ravel(),
            observed[v2].to_numpy().ravel(),
            s=1,
            c="black",
            alpha=0.05,
            linewidths=0,
            label="dataset",
        )
        ax.legend(loc="upper right", markerscale=4)
    ax.scatter(
        [float(observed_base_point[v1])],
        [float(observed_base_point[v2])],
        s=20,
        c="white",
        edgecolors="black",
    )
    ax.set_xlabel(v1)
    ax.set_ylabel(v2)
    ax.set_title(
        f"NN {target_var.upper()} over {v1}/{v2};"
        f"lat={latitude:.2f}, lon={longitude:.2f}, date={date}"
    )
    cb = fig.colorbar(contour, ax=ax)
    cb.set_label("$W/m^2$")
    fig.tight_layout()
    output_dir = figure_dir(figures_path, "all", "tsr_contour_test", v1, date)
    fig.savefig(output_dir / f"{v1}_{v2}.png")
    plt.close(fig)


def eval_kernel_vars(config):
    input_vars = list(config.dataset.input_vars)
    return [var for var in ("fal", "skt", "tcwv") if var in input_vars]


def kernel_date_test(
    ds: xr.Dataset,
    preprocessor: Preprocessor,
    model: SimpleModel,
    config: VariableConfig,
    date,
    true_kernel: xr.Dataset,
    figures_path: Path = Path("."),
    kernel_name=None,
):
    kernel_name = kernel_name or eval_kernel_vars(config)[0]
    title = kernel_title(kernel_name)
    label = kernel_label(kernel_name)
    all_dir = figure_dir(figures_path, "all", "kernel_date_test", kernel_name, date)
    clr_dir = figure_dir(figures_path, "clr", "kernel_date_test", kernel_name, date)

    nn_kern_cld, lon, lat = compute_nn_kernel(
        ds, preprocessor, model, config, perturbation_var=kernel_name
    )
    nn_kern_clr, _, _ = compute_nn_kernel(
        ds, preprocessor, model, config, clear=True, perturbation_var=kernel_name
    )
    nn_grad_cld, grad_lon, grad_lat = compute_nn_kernel_autograd(
        ds, preprocessor, model, config, var=kernel_name
    )
    nn_grad_clr, _, _ = compute_nn_kernel_autograd(
        ds, preprocessor, model, config, var=kernel_name, clear=True
    )

    rrtm_kern_cld = true_kernel[kernel_name].sel(all_clr="all").to_numpy()[
        0
    ] * kernel_delta(kernel_name)
    rrtm_kern_clr = true_kernel[kernel_name].sel(all_clr="clr").to_numpy()[
        0
    ] * kernel_delta(kernel_name)
    kern_lon, kern_lat = true_kernel.longitude, true_kernel.latitude
    # --- clear and all sky plots of nn kernel, global and north pole
    plot_global_field(
        nn_kern_cld,
        lon,
        lat,
        f"NN {title} Kernel finite difference (all)\n{date}",
        all_dir / "nn.png",
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
        label=label,
        annotation=f"{np.mean(nn_kern_cld):.2f}",
    )
    plot_global_field(
        nn_kern_clr,
        lon,
        lat,
        f"NN {title} Kernel finite difference (clear)\n{date}",
        clr_dir / "nn.png",
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
        label=label,
        annotation=f"{np.mean(nn_kern_clr):.2f}",
    )
    plot_global_field(
        nn_grad_cld,
        grad_lon,
        grad_lat,
        f"NN {title} Kernel via autograd (all)\n{date}",
        all_dir / "nn_grad.png",
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
        label=label,
        annotation=f"{np.mean(nn_grad_cld):.2f}",
    )
    plot_global_field(
        nn_grad_clr,
        grad_lon,
        grad_lat,
        f"NN {title} Kernel via autograd (clear)\n{date}",
        clr_dir / "nn_grad.png",
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
        label=label,
        annotation=f"{np.mean(nn_grad_clr):.2f}",
    )
    north_mask = lat >= 60
    plot_north_pole_field(
        nn_kern_cld,
        lon,
        lat,
        f"NN {title} Kernel finite difference (all)\n{date}",
        all_dir / "nn_np.png",
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
        label=label,
        annotation=f"{np.mean(nn_kern_cld[north_mask]):.2f}",
    )
    plot_north_pole_field(
        nn_kern_clr,
        lon,
        lat,
        f"NN {title} Kernel finite difference (clear)\n{date}",
        clr_dir / "nn_np.png",
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
        label=label,
        annotation=f"{np.mean(nn_kern_clr[north_mask]):.2f}",
    )
    grad_north_mask = grad_lat >= 60
    plot_north_pole_field(
        nn_grad_cld,
        grad_lon,
        grad_lat,
        f"NN {title} Kernel via autograd (all)\n{date}",
        all_dir / "nn_grad_np.png",
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
        label=label,
        annotation=f"{np.mean(nn_grad_cld[grad_north_mask]):.2f}",
    )
    plot_north_pole_field(
        nn_grad_clr,
        grad_lon,
        grad_lat,
        f"NN {title} Kernel via autograd (clear)\n{date}",
        clr_dir / "nn_grad_np.png",
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
        label=label,
        annotation=f"{np.mean(nn_grad_clr[grad_north_mask]):.2f}",
    )
    # --- clear and all sky plots of rrtm kernel, global and north pole
    plot_global_field(
        rrtm_kern_cld,
        kern_lon,
        kern_lat,
        f"RRTM {title} Kernel (all)\n{date}",
        all_dir / "rrtm.png",
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
        label=label,
        annotation=f"{np.mean(rrtm_kern_cld):.2f}",
    )
    plot_global_field(
        rrtm_kern_clr,
        kern_lon,
        kern_lat,
        f"RRTM {title} Kernel (clear)\n{date}",
        clr_dir / "rrtm.png",
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
        label=label,
        annotation=f"{np.mean(rrtm_kern_clr):.2f}",
    )
    north_mask = kern_lat >= 60
    plot_north_pole_field(
        rrtm_kern_cld,
        kern_lon,
        kern_lat,
        f"RRTM {title} Kernel (all)\n{date}",
        all_dir / "rrtm_np.png",
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
        label=label,
        annotation=f"{np.mean(rrtm_kern_clr[north_mask]):.2f}",
    )
    plot_north_pole_field(
        rrtm_kern_clr,
        kern_lon,
        kern_lat,
        f"RRTM {title} Kernel (clear)\n{date}",
        clr_dir / "rrtm_np.png",
        cmap="RdBu_r",
        vmin=-3,
        vmax=3,
        label=label,
        annotation=f"{np.mean(rrtm_kern_cld[north_mask]):.2f}",
    )
    # --- clear and all sky plots of nn-rrtm kernel difference, global and north pole

    plot_lon, plot_lat = lon, lat
    plot_grad_lon, plot_grad_lat = grad_lon, grad_lat
    if nn_kern_clr.shape != true_kernel[kernel_name].sel(all_clr="clr").shape:
        nn_kern_cld = interpolate_spatial_field(
            nn_kern_cld,
            lon,
            lat,
            kern_lon,
            kern_lat,
        )
        nn_kern_clr = interpolate_spatial_field(
            nn_kern_clr,
            lon,
            lat,
            kern_lon,
            kern_lat,
        )
        plot_lon = kern_lon
        plot_lat = kern_lat
    if nn_grad_clr.shape != true_kernel[kernel_name].sel(all_clr="clr").shape:
        nn_grad_cld = interpolate_spatial_field(
            nn_grad_cld, grad_lon, grad_lat, kern_lon, kern_lat
        )
        nn_grad_clr = interpolate_spatial_field(
            nn_grad_clr, grad_lon, grad_lat, kern_lon, kern_lat
        )
        plot_grad_lon = kern_lon
        plot_grad_lat = kern_lat
    grad_north_mask = plot_grad_lat >= 60

    diff_cld = nn_kern_cld - rrtm_kern_cld
    diff_clr = nn_kern_clr - rrtm_kern_clr
    diff_grad_cld = nn_grad_cld - rrtm_kern_cld
    diff_grad_clr = nn_grad_clr - rrtm_kern_clr

    plot_global_field(
        diff_cld,
        plot_lon,
        plot_lat,
        f"NN finite difference-RRTM {title} Kernel (all)\n{date}",
        all_dir / "nn-rrtm.png",
        cmap="RdBu_r",
        vmin=-0.6,
        vmax=0.6,
        label=label,
        annotation=f"{np.mean(diff_cld):.2f}; {np.mean(np.abs(diff_cld)):.2f}",
    )
    plot_global_field(
        diff_clr,
        plot_lon,
        plot_lat,
        f"NN finite difference-RRTM {title} Kernel (clear)\n{date}",
        clr_dir / "nn-rrtm.png",
        cmap="RdBu_r",
        vmin=-0.6,
        vmax=0.6,
        label=label,
        annotation=f"{np.mean(diff_clr):.2f}; {np.mean(np.abs(diff_clr)):.2f}",
    )
    plot_global_field(
        diff_grad_cld,
        plot_grad_lon,
        plot_grad_lat,
        f"NN autograd-RRTM {title} Kernel (all)\n{date}",
        all_dir / "nn_grad-rrtm.png",
        cmap="RdBu_r",
        vmin=-0.6,
        vmax=0.6,
        label=label,
        annotation=f"{np.mean(diff_grad_cld):.2f}; {np.mean(np.abs(diff_grad_cld)):.2f}",
    )
    plot_global_field(
        diff_grad_clr,
        plot_grad_lon,
        plot_grad_lat,
        f"NN autograd-RRTM {title} Kernel (clear)\n{date}",
        clr_dir / "nn_grad-rrtm.png",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        label=label,
        annotation=f"{np.mean(diff_grad_clr):.2f}; {np.mean(np.abs(diff_grad_clr)):.2f}",
    )
    north_mask = plot_lat >= 60
    plot_north_pole_field(
        diff_cld,
        plot_lon,
        plot_lat,
        f"NN finite difference-RRTM {title} Kernel (all)\n{date}",
        all_dir / "nn-rrtm_np.png",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        label=label,
        annotation=f"{np.mean(diff_cld[north_mask]):.2f}; {np.mean(np.abs(diff_cld[north_mask])):.2f}",
    )
    plot_north_pole_field(
        diff_clr,
        plot_lon,
        plot_lat,
        f"NN finite difference-RRTM {title} Kernel (clear)\n{date}",
        clr_dir / "nn-rrtm_np.png",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        label=label,
        annotation=f"{np.mean(diff_clr[north_mask]):.2f}; {np.mean(np.abs(diff_clr[north_mask])):.2f}",
    )
    plot_north_pole_field(
        diff_grad_cld,
        plot_grad_lon,
        plot_grad_lat,
        f"NN autograd-RRTM {title} Kernel (all)\n{date}",
        all_dir / "nn_grad-rrtm_np.png",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        label=label,
        annotation=f"{np.mean(diff_grad_cld[grad_north_mask]):.2f}; {np.mean(np.abs(diff_grad_cld[grad_north_mask])):.2f}",
    )
    plot_north_pole_field(
        diff_grad_clr,
        plot_grad_lon,
        plot_grad_lat,
        f"NN autograd-RRTM {title} Kernel (clear)\n{date}",
        clr_dir / "nn_grad-rrtm_np.png",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        label=label,
        annotation=f"{np.mean(diff_grad_clr[grad_north_mask]):.2f}; {np.mean(np.abs(diff_grad_clr[grad_north_mask])):.2f}",
    )


def select_date_state(ds: xr.Dataset, date: str) -> xr.Dataset:
    selected = ds.sel(date=date)
    if "date" in selected.dims:
        selected = selected.squeeze("date", drop=True)
    return selected


def second_order_test(
    ds: xr.Dataset,
    preprocessor: Preprocessor,
    model: SimpleModel,
    config,
    true_kernel: xr.Dataset,
    dates=["2012-09", "2013-09"],
    figures_path: Path = Path("."),
    kernel_name=None,
):
    kernel_name = kernel_name or eval_kernel_vars(config)[0]
    base_date, perturbed_date = dates
    output_dir = figure_dir(
        figures_path,
        "all",
        "second_order_test",
        kernel_name,
        f"{perturbed_date}_minus_{base_date}",
    )
    fields = compute_kernel_difference(
        ds,
        true_kernel,
        preprocessor,
        model,
        config,
        base_date,
        perturbed_date,
        kernel_name,
    )
    plot_kernel_difference(fields, output_dir)
    plot_hybrid_kernel_difference(
        compute_hybrid_kernel_difference(fields, preprocessor, model, config, kernel_name),
        output_dir,
    )


def compute_standard_eval(args):
    config = load_config(args.config_file)
    variable_config = variable_config_from_omegaconf(config)
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else Path(config.train.checkpoint_dir) / "best_model.pt"
    )
    model, preprocessor, epoch = load_model_and_preprocessor(config, checkpoint_path)

    # --- setup output directory ---
    output_dir = Path(config.train.checkpoint_dir) / "figures" / str(epoch)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- load test data ---
    eval_years = [2012, 2013, 2015]
    raw_era5_paths = generate_paths_yearly(
        config.dataset.era5.raw_path, eval_years, make_era5_filename
    )
    test_era5_paths = generate_paths_yearly(
        config.dataset.era5.path, config.dataset.test_years, make_era5_filename
    )

    raw_dataset = xr.open_mfdataset(raw_era5_paths, combine="nested", concat_dim="date")
    raw_test_dataset = raw_dataset.sel(
        date=raw_dataset.date.dt.year.isin(config.dataset.test_years)
    )
    processed_dataset = xr.open_mfdataset(
        test_era5_paths, combine="nested", concat_dim="date"
    )

    # --- load preprocessor ---
    preprocessor = Preprocessor(config)
    preprocessor.load(config.preprocess.params_dir)

    return {
        "config": config,
        "variable_config": variable_config,
        "model": model,
        "preprocessor": preprocessor,
        "output_dir": output_dir,
        "eval_years": eval_years,
        "raw_dataset": raw_dataset,
        "raw_test_dataset": raw_test_dataset,
        "processed_dataset": processed_dataset,
    }


def plot_standard_eval(eval_data):
    config = eval_data["config"]
    variable_config = eval_data["variable_config"]
    model = eval_data["model"]
    preprocessor = eval_data["preprocessor"]
    output_dir = eval_data["output_dir"]
    eval_years = eval_data["eval_years"]
    raw_dataset = eval_data["raw_dataset"]
    raw_test_dataset = eval_data["raw_test_dataset"]
    processed_dataset = eval_data["processed_dataset"]

    global_tsr_test(
        ds=processed_dataset,
        preprocessor=preprocessor,
        model=model,
        vc=variable_config,
        figures_path=output_dir,
    )
    global_tsrc_test(
        raw_ds=raw_test_dataset,
        preprocessor=preprocessor,
        model=model,
        vc=variable_config,
        figures_path=output_dir,
    )

    for kernel_name in eval_kernel_vars(config):
        kernel_paths = generate_paths_yearly(
            config.dataset.kernels.raw_path,
            eval_years,
            lambda year, var=kernel_name: make_kernel_filename(year, var),
        )
        kernels_dataset = xr.open_mfdataset(
            kernel_paths, combine="nested", concat_dim="date"
        )
        # kernel_date_test(
        #    ds=raw_dataset.sel(date="2013-09"),
        #    preprocessor=preprocessor,
        #    model=model,
        #    config=variable_config,
        #    figures_path=output_dir,
        #    true_kernel=kernels_dataset.sel(date="2013-09"),
        #    date="2013-09",
        #    kernel_name=kernel_name,
        # )
        kernel_date_test(
            ds=raw_dataset.sel(date="2015-09"),
            preprocessor=preprocessor,
            model=model,
            config=variable_config,
            figures_path=output_dir,
            true_kernel=kernels_dataset.sel(date="2015-09"),
            date="2015-09",
            kernel_name=kernel_name,
        )
        second_order_test(
            ds=raw_dataset,
            preprocessor=preprocessor,
            model=model,
            config=variable_config,
            true_kernel=kernels_dataset,
            # dates=["2012-09", "2013-09"],
            figures_path=output_dir,
            kernel_name=kernel_name,
        )
        # kernel_date_test(
        #    ds=raw_dataset.sel(date="2015-12"),
        #    preprocessor=preprocessor,
        #    model=model,
        #    config=config,
        #    figures_path=output_dir,
        #    true_kernel=kernels_dataset.sel(date="2015-12"),
        #    date="2015-12",
        #    kernel_name=kernel_name,
        # )
        kernels_dataset.close()

    kernel_ecod_fal_contour_test(
        processed_ds=processed_dataset,
        preprocessor=preprocessor,
        model=model,
        config=variable_config,
        figures_path=output_dir,
        scatter=False,
        extrapolate=False,
        date="2015-09",
    )
    tsr_ecod_fal_contour_test(
        processed_ds=processed_dataset,
        preprocessor=preprocessor,
        model=model,
        config=variable_config,
        figures_path=output_dir,
        scatter=False,
        extrapolate=False,
        date="2015-09",
    )


def run_standard_eval(args):
    print("=" * 20 + " computation " + "=" * 20)
    eval_data = compute_standard_eval(args)
    print("=" * 20 + " plotting " + "=" * 20)
    plot_standard_eval(eval_data)


# --------------------------------------------------------------------------- #
# NeurIPS summary table helpers
# --------------------------------------------------------------------------- #

table_output_dir = Path("FiguresNeurIPS") / "ablation_tables"

validation_output_dir = table_output_dir / "validation_points"

tsr_table_path = table_output_dir / "tsr_test_1990_2020.csv"
kernel_table_path = table_output_dir / "kernel_date_test_2015_heldout_months.csv"
kernel_fd_table_path = (
    table_output_dir / "kernel_finite_difference_date_test_2015_heldout_months.csv"
)
closure_table_path = table_output_dir / "closure_1990_2020.csv"
feedback_table_path = table_output_dir / "feedback_quantification_two_year.csv"


def seed_checkpoint_paths(config_path, checkpoint_dir=None):
    config_for_path = load_config(config_path)
    base_dir = Path(checkpoint_dir or config_for_path.train.checkpoint_dir)
    return sorted(base_dir.parent.glob(f"{base_dir.name}_seed_*/best_model.pt"))


def expanded_table_model_specs():
    for label, config_path in TABLE_MODEL_SPECS:
        seeds = (
            seed_checkpoint_paths(config_path)
            if label in SEED_AGGREGATED_MODELS
            else []
        )
        if len(seeds) > 1:
            for i, seed_path in enumerate(seeds, start=1):
                yield f"{label} [seed {i}]", config_path, seed_path
        else:
            yield label, config_path, None


def seed_rows_for(table, label):
    """Boolean mask selecting rows of `table` belonging to seed variants of `label`."""
    index = (
        table.index.get_level_values(0)
        if isinstance(table.index, pd.MultiIndex)
        else table.index
    )
    return index.astype(str).str.startswith(f"{label} [seed ")


def format_mean_std(mean, std):
    return mean.combine(std, lambda m, s: f"{m:.3f} \u00b1 {s:.3f}")


def aggregate_seed_rows(table):
    print(table)
    """Collapse per-seed rows into a single mean +/- std row per seed-aggregated model."""
    multi = isinstance(table.index, pd.MultiIndex)
    rows, seen_mask = [], pd.Series(False, index=table.index)

    for label in SEED_AGGREGATED_MODELS:
        mask = seed_rows_for(table, label)
        seen_mask |= mask
        if not mask.any():
            continue
        seed_table = table.loc[mask].astype(float)

        if not multi:
            rows.append(
                pd.DataFrame(
                    [format_mean_std(seed_table.mean(), seed_table.std(ddof=1))],
                    index=[label],
                )
            )
            continue

        for residual in seed_table.index.get_level_values(1).unique():
            residual_table = seed_table.xs(residual, level=1)
            row = format_mean_std(residual_table.mean(), residual_table.std(ddof=1))
            row.name = (label, residual)
            rows.append(row)

    if not rows:
        return table

    keep = table.loc[~seen_mask.to_numpy()]
    if not multi:
        return pd.concat([*rows, keep])

    aggregated = pd.DataFrame(rows)
    aggregated.index = pd.MultiIndex.from_tuples(
        aggregated.index, names=table.index.names
    )
    return pd.concat([aggregated, keep])


def seed_aggregation_available():
    """Whether any seed-aggregated model currently has >1 discovered seed checkpoint."""
    return any(
        len(seed_checkpoint_paths(config_path)) > 1
        for label, config_path in TABLE_MODEL_SPECS
        if label in SEED_AGGREGATED_MODELS
    )


def table_has_seed_aggregation(table):
    """Whether a cached table already contains mean +/- std formatted seed rows."""
    values = table.astype(str).to_numpy().ravel()
    return any("\u00b1" in value for value in values)


def safe_name(value):
    """Filesystem-safe version of a label, for use in output filenames."""
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


def load_evaluation_model(config_path, checkpoint_path):
    table_config = load_config(config_path)
    table_vc = variable_config_from_omegaconf(table_config)
    checkpoint_path = (
        checkpoint_path or Path(table_config.train.checkpoint_dir) / "best_model.pt"
    )
    table_model, table_preprocessor, _ = load_model_and_preprocessor(
        table_config, checkpoint_path, downscaling=False
    )
    return table_config, table_vc, table_model, table_preprocessor


def collect_model_metric_rows(row_fn, *args, **kwargs):
    """Load each model spec and collect the rows `row_fn` produces for it."""
    rows = []
    for label, config_path, checkpoint_path in expanded_table_model_specs():
        table_config, table_vc, table_model, table_preprocessor = load_evaluation_model(
            config_path, checkpoint_path
        )
        rows.extend(
            row_fn(
                label,
                table_config,
                table_vc,
                table_model,
                table_preprocessor,
                *args,
                **kwargs,
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Validation-point snapshots (for spot-checking predictions vs truth)
# --------------------------------------------------------------------------- #


def validation_point_dataset(test, model, date, sky, prediction, truth):
    prediction = (
        prediction.reset_coords(drop=True)
        .transpose("latitude", "longitude")
        .astype("float32")
    )
    truth = (
        truth.reset_coords(drop=True)
        .transpose("latitude", "longitude")
        .astype("float32")
    )
    return xr.Dataset(
        {
            "truth": truth.expand_dims(record=1),
            "prediction": prediction.expand_dims(record=1),
            "residual": (prediction - truth).astype("float32").expand_dims(record=1),
        },
        coords={
            "test": ("record", [test]),
            "model": ("record", [model]),
            "date": ("record", [pd.Timestamp(date)]),
            "sky": ("record", [sky]),
        },
    )


def write_validation_points(path, datasets):
    if not datasets:
        return
    path.parent.mkdir(exist_ok=True, parents=True)
    ds = xr.concat(datasets, dim="record").assign_coords(
        record=np.arange(len(datasets))
    )
    ds.to_netcdf(path)


# --------------------------------------------------------------------------- #
# Metric bookkeeping (mean bias, RMSE, across-model summary tables)
# --------------------------------------------------------------------------- #


def global_bias_and_mean_square(diff, arctic=False):
    print(diff)

    field = arctic_field(diff) if arctic else diff
    mean = float(global_mean(field).compute())
    mean_sq = float(global_mean(field**2).compute())
    return mean, mean_sq


def append_prediction_metric(rows, model, sky, diff, arctic=False):
    mean, mean_sq = global_bias_and_mean_square(diff, arctic=arctic)
    rows.append({"Model": model, "sky": sky, "mean": mean, "mean_sq": mean_sq})


def summarize_prediction_metrics(rows, include_std=True):
    rows_df = pd.DataFrame(rows)
    table_rows = []
    for model, model_df in rows_df.groupby("Model", sort=False):
        row = {"Model": model}
        for sky in ["all", "clear"]:
            sky_df = model_df[model_df.sky == sky]
            row[f"{sky} MBE"] = float(sky_df["mean"].mean())
            row[f"{sky} RMSE"] = float(np.sqrt(sky_df["mean_sq"].mean()))
            if include_std:
                row[f"{sky} STD"] = float(sky_df["mean"].std(ddof=0))
        table_rows.append(row)
    return pd.DataFrame(table_rows).set_index("Model")


def append_feedback_metric(rows, model, residual, sky, region, diff):
    mean, mean_sq = global_bias_and_mean_square(diff, arctic=(region == "arctic"))
    rows.append(
        {
            "Model": model,
            "Residual": residual,
            "sky": sky,
            "region": region,
            "mean": mean,
            "mean_sq": mean_sq,
        }
    )


def summarize_feedback_metrics(rows):
    rows_df = pd.DataFrame(rows)
    table_rows = []
    for (model, residual), group_df in rows_df.groupby(
        ["Model", "Residual"], sort=False
    ):
        row = {"Model": model, "Residual": residual}
        for region in ["global", "arctic"]:
            for sky in ["all", "clear"]:
                subset = group_df[(group_df.region == region) & (group_df.sky == sky)]
                row[f"{region} {sky} MBE"] = float(subset["mean"].mean())
                row[f"{region} {sky} RMSE"] = float(np.sqrt(subset["mean_sq"].mean()))
        table_rows.append(row)
    return pd.DataFrame(table_rows).set_index(["Model", "Residual"])


# --------------------------------------------------------------------------- #
# TSR test: NN TSR predictions vs ERA5 TSR, 1990-2020, all-sky & clear-sky
# --------------------------------------------------------------------------- #


def collect_tsr_prediction_metrics(
    label, table_config, table_vc, table_model, table_preprocessor
):
    rows, validation_points = [], []
    for year in range(1990, 2021):
        raw_ds = xr.open_dataset(
            Path(table_config.dataset.era5.raw_path) / make_era5_filename(year)
        )
        for date in raw_ds.date.values:
            raw_date = raw_ds.sel(date=date)

            all_pred = (
                nn_pred(
                    table_preprocessor.transform(raw_date),
                    table_model,
                    table_preprocessor,
                    table_vc,
                    ["latitude", "longitude"],
                )
                / SECONDS_PER_DAY
            )
            clear_pred = (
                nn_pred(
                    table_preprocessor.transform(table_vc.clear_sky_input(raw_date)),
                    table_model,
                    table_preprocessor,
                    table_vc,
                    ["latitude", "longitude"],
                    clear=True,
                )
                / SECONDS_PER_DAY
            )

            all_truth = raw_date.tsr / SECONDS_PER_DAY
            clear_truth = raw_date.tsrc / SECONDS_PER_DAY

            append_prediction_metric(rows, label, "all", all_pred - all_truth)
            append_prediction_metric(rows, label, "clear", clear_pred - clear_truth)

            if len(validation_points) < MAX_VALIDATION_POINTS:
                validation_points.extend(
                    [
                        validation_point_dataset(
                            "tsr", label, date, "all", all_pred, all_truth
                        ),
                        validation_point_dataset(
                            "tsr", label, date, "clear", clear_pred, clear_truth
                        ),
                    ]
                )
        raw_ds.close()

    write_validation_points(
        validation_output_dir / f"tsr_{safe_name(label)}.nc", validation_points
    )
    return rows


def save_table(table, path):
    path.parent.mkdir(exist_ok=True, parents=True)
    table.to_csv(path)
    return table


def write_tsr_prediction_table():
    table = aggregate_seed_rows(
        summarize_prediction_metrics(
            collect_model_metric_rows(collect_tsr_prediction_metrics), include_std=True
        )
    )
    return save_table(table, tsr_table_path)


# --------------------------------------------------------------------------- #
# Kernel date test: 2015 held-out months, NN albedo kernel vs FAL kernel
# (autograd-derived and, optionally, finite-difference-derived)
# --------------------------------------------------------------------------- #


def collect_albedo_kernel_metrics(
    label,
    table_config,
    table_vc,
    table_model,
    table_preprocessor,
    finite_difference=False,
):
    rows, validation_points = [], []
    held_out_months = [month for month in range(1, 13) if month not in (3, 6, 9, 12)]
    dates = [f"2015-{month:02d}" for month in held_out_months]

    raw_ds = xr.open_dataset(
        Path(table_config.dataset.era5.raw_path) / make_era5_filename(2015)
    )
    kernels = xr.open_dataset(
        Path(table_config.dataset.kernels.raw_path) / make_kernel_filename(2015, "fal")
    )

    for date in dates:
        raw_date = raw_ds.sel(date=[date] if finite_difference else date).interp(
            latitude=kernels.latitude, longitude=kernels.longitude
        )
        true_kernel = kernels.sel(date=date).fal.squeeze(
            "date", drop=True
        ) * kernel_delta("fal")

        if finite_difference:
            pred_all, lon, lat = compute_nn_kernel(
                raw_date,
                table_preprocessor,
                table_model,
                table_vc,
                perturbation_var="fal",
            )
            pred_clear, _, _ = compute_nn_kernel(
                raw_date,
                table_preprocessor,
                table_model,
                table_vc,
                clear=True,
                perturbation_var="fal",
            )
        else:
            pred_all, lon, lat = compute_nn_kernel_autograd(
                raw_date,
                table_preprocessor,
                table_model,
                table_vc,
                var="fal",
                clear=False,
            )
            pred_clear, _, _ = compute_nn_kernel_autograd(
                raw_date,
                table_preprocessor,
                table_model,
                table_vc,
                var="fal",
                clear=True,
            )

        all_pred = xr.DataArray(
            pred_all,
            coords={"latitude": lat, "longitude": lon},
            dims=("latitude", "longitude"),
        )
        clear_pred = xr.DataArray(
            pred_clear,
            coords={"latitude": lat, "longitude": lon},
            dims=("latitude", "longitude"),
        )
        all_truth = true_kernel.sel(all_clr="all")
        clear_truth = true_kernel.sel(all_clr="clr")

        append_prediction_metric(rows, label, "all", all_pred - all_truth)
        append_prediction_metric(rows, label, "clear", clear_pred - clear_truth)

        if len(validation_points) < MAX_VALIDATION_POINTS:
            test_name = "kernel_finite_difference" if finite_difference else "kernel"
            validation_points.extend(
                [
                    validation_point_dataset(
                        test_name, label, date, "all", all_pred, all_truth
                    ),
                    validation_point_dataset(
                        test_name, label, date, "clear", clear_pred, clear_truth
                    ),
                ]
            )

    raw_ds.close()
    kernels.close()

    suffix = "kernel_finite_difference" if finite_difference else "kernel"
    write_validation_points(
        validation_output_dir / f"{suffix}_{safe_name(label)}.nc", validation_points
    )
    return rows


def write_albedo_kernel_table():
    table = aggregate_seed_rows(
        summarize_prediction_metrics(
            collect_model_metric_rows(collect_albedo_kernel_metrics), include_std=True
        )
    )
    return save_table(table, kernel_table_path)


def write_albedo_kernel_finite_difference_table():
    rows = collect_model_metric_rows(
        collect_albedo_kernel_metrics, finite_difference=True
    )
    table = aggregate_seed_rows(summarize_prediction_metrics(rows, include_std=True))
    return save_table(table, kernel_fd_table_path)


# --------------------------------------------------------------------------- #
# Feedback quantification test: context base_date -> perturbed_date response residuals
# --------------------------------------------------------------------------- #


def collect_feedback_response_metrics(ctx, label, config_path, checkpoint_path):
    rows = []
    responses = load_feedback_ablation_responses(
        ctx, label, config_path, "nn", checkpoint_path
    )
    for sky, response_sky in [("all", "all"), ("clear", "clr")]:
        terms = shakirova_terms(responses, response_sky)
        for residual, diff in [
            ("allcross", terms["allcross"][-1]),
            ("nn sum", terms["nn"][-1]),
        ]:
            for region in ["global", "arctic"]:
                append_feedback_metric(rows, label, residual, sky, region, diff)
    return rows


def collect_feedback_kernel_metrics(ctx):
    rows = []
    responses = load_feedback_ablation_responses(ctx, None, None, "kernel")
    for sky, response_sky in [("all", "all"), ("clear", "clr")]:
        terms = shakirova_terms(responses, response_sky)
        diff = terms["kernel"][-1]
        for region in ["global", "arctic"]:
            append_feedback_metric(rows, "Kernel", "kernel sum", sky, region, diff)
    return rows


def write_feedback_response_table(ctx):
    rows = []
    for label, config_path, checkpoint_path in expanded_table_model_specs():
        rows.extend(
            collect_feedback_response_metrics(ctx, label, config_path, checkpoint_path)
        )
    rows.extend(collect_feedback_kernel_metrics(ctx))

    table = aggregate_seed_rows(summarize_feedback_metrics(rows))
    return save_table(table, feedback_table_path)


def ensure_feedback_kernel_method(table, ctx):
    """Backfill the kernel-method row into a cached feedback table that predates it."""
    if (
        isinstance(table.index, pd.MultiIndex)
        and ("Kernel", "kernel sum") in table.index
    ):
        return table
    print("Feedback quantification test: recomputing for method-level mean/std rows")
    return write_feedback_response_table(ctx)


# --------------------------------------------------------------------------- #
# Cache loading and top-level table computation
# --------------------------------------------------------------------------- #


def load_or_compute_table(path, description, compute_table, index_col=0):
    """Load a cached CSV table if present and up to date, else compute and cache it."""
    if path.exists():
        table = pd.read_csv(path, index_col=index_col)
        if seed_aggregation_available() and not table_has_seed_aggregation(table):
            print(f"{description}: recomputing for seed mean/std")
            return compute_table()
        print(f"{description}: loaded {path}")
        return table
    print(f"{description}: computing")
    return compute_table()


def write_neurips_summary_tables(ctx=None):
    ctx = ctx or make_neurips_feedback_context()
    tsr_table = load_or_compute_table(
        tsr_table_path,
        "TSR test: monthly 1990-2020 NN TSR predictions vs ERA5 TSR, all-sky and clear-sky",
        write_tsr_prediction_table,
    )
    kernel_table = load_or_compute_table(
        kernel_table_path,
        "Kernel date test: 2015 held-out months except March, June, September, December; NN autograd albedo kernel vs FAL kernel",
        write_albedo_kernel_table,
    )
    kernel_fd_table = load_or_compute_table(
        kernel_fd_table_path,
        "Kernel finite-difference date test: same 2015 held-out months; NN finite-difference albedo kernel vs FAL kernel",
        write_albedo_kernel_finite_difference_table,
    )
    feedback_table = ensure_feedback_kernel_method(
        load_or_compute_table(
            feedback_table_path,
            "Feedback quantification test: context base_date to perturbed_date response residuals, global and Arctic",
            lambda: write_feedback_response_table(ctx),
            index_col=[0, 1],
        ),
        ctx,
    )
    return tsr_table, kernel_table, kernel_fd_table, feedback_table


# %%
import re

sobolev_lambda_tsr_raw_path = (
    table_output_dir / "sobolev_lambda_seed_metrics_tsr_test_1990_2020.csv"
)
sobolev_lambda_kernel_raw_path = (
    table_output_dir
    / "sobolev_lambda_seed_metrics_kernel_date_test_2015_heldout_months.csv"
)
sobolev_lambda_feedback_raw_path = (
    table_output_dir
    / "sobolev_lambda_seed_metrics_feedback_quantification_two_year.csv"
)


def sobolev_lambda_seed_paths(label, config_path, checkpoint_dir):
    seeds = seed_checkpoint_paths(config_path, checkpoint_dir)
    if len(seeds) != 5:
        raise FileNotFoundError(
            f"{label}: expected 5 seeded checkpoints, found {len(seeds)}"
        )
    return seeds


def sobolev_seed_labels(label):
    return [f"{label} [seed {i}]" for i in range(1, 6)]


def strip_sobolev_seed_label(value):
    return re.sub(r" \[seed \d+\](?=$| )", "", str(value))


def sobolev_seed_mean_std_table(table):
    table = table.astype(float).copy()
    if isinstance(table.index, pd.MultiIndex):
        tuples = [
            (strip_sobolev_seed_label(parts[0]), *parts[1:]) for parts in table.index
        ]
        table.index = pd.MultiIndex.from_tuples(tuples, names=table.index.names)
        grouped = table.groupby(level=list(range(table.index.nlevels)), sort=False)
    else:
        table.index = table.index.map(strip_sobolev_seed_label)
        grouped = table.groupby(level=0, sort=False)

    mean = grouped.mean()
    std = grouped.std(ddof=1)
    return mean.map(lambda x: f"{x:.3f}") + " ± " + std.map(lambda x: f"{x:.3f}")


def missing_sobolev_specs(table):
    existing = set(
        (
            table.index.get_level_values(0)
            if isinstance(table.index, pd.MultiIndex)
            else table.index
        ).astype(str)
    )
    return [
        (label, config_path, checkpoint_dir)
        for label, config_path, checkpoint_dir in SOBOLEV_LAMBDA_SPECS
        if not set(sobolev_seed_labels(label)).issubset(existing)
    ]


def sobolev_lambda_collect_model_metric_rows(row_fn, specs=None, *args, **kwargs):
    rows = []
    for label, config_path, checkpoint_dir in specs or SOBOLEV_LAMBDA_SPECS:
        for seed_label, checkpoint_path in zip(
            sobolev_seed_labels(label),
            sobolev_lambda_seed_paths(label, config_path, checkpoint_dir),
        ):
            table_config, table_vc, table_model, table_preprocessor = (
                load_evaluation_model(config_path, checkpoint_path)
            )
            rows.extend(
                row_fn(
                    seed_label,
                    table_config,
                    table_vc,
                    table_model,
                    table_preprocessor,
                    *args,
                    **kwargs,
                )
            )
    return rows


def load_or_update_raw_sobolev_table(path, description, compute_table, index_col=0):
    if not path.exists():
        print(f"{description}: computing")
        table = compute_table(SOBOLEV_LAMBDA_SPECS)
        table.to_csv(path)
        return table

    table = pd.read_csv(path, index_col=index_col)
    missing = missing_sobolev_specs(table)
    if not missing:
        print(f"{description}: loaded {path}")
        return table

    print(f"{description}: computing missing configs {[label for label, *_ in missing]}")
    table = pd.concat([table, compute_table(missing)])
    table = table[~table.index.duplicated(keep="last")]
    table.to_csv(path)
    return table


def compute_sobolev_lambda_tsr_raw_table(specs):
    return summarize_prediction_metrics(
        sobolev_lambda_collect_model_metric_rows(collect_tsr_prediction_metrics, specs),
        include_std=False,
    )


def compute_sobolev_lambda_kernel_raw_table(specs):
    return summarize_prediction_metrics(
        sobolev_lambda_collect_model_metric_rows(collect_albedo_kernel_metrics, specs),
        include_std=False,
    )


def compute_sobolev_lambda_feedback_raw_table(specs, ctx):
    rows = []
    for label, config_path, checkpoint_dir in specs:
        for seed_label, checkpoint_path in zip(
            sobolev_seed_labels(label),
            sobolev_lambda_seed_paths(label, config_path, checkpoint_dir),
        ):
            rows.extend(
                collect_feedback_response_metrics(
                    ctx, seed_label, config_path, checkpoint_path
                )
            )
    return summarize_feedback_metrics(rows)


def write_sobolev_lambda_summary_tables(ctx=None):
    ctx = ctx or make_neurips_feedback_context()
    sobolev_lambda_tsr_raw_table = load_or_update_raw_sobolev_table(
        sobolev_lambda_tsr_raw_path,
        "Sobolev lambda ablation TSR seed metrics",
        compute_sobolev_lambda_tsr_raw_table,
    )
    sobolev_lambda_kernel_raw_table = load_or_update_raw_sobolev_table(
        sobolev_lambda_kernel_raw_path,
        "Sobolev lambda ablation kernel sensitivity seed metrics",
        compute_sobolev_lambda_kernel_raw_table,
    )
    sobolev_lambda_feedback_raw_table = load_or_update_raw_sobolev_table(
        sobolev_lambda_feedback_raw_path,
        "Sobolev lambda ablation two-year feedback seed metrics",
        lambda specs: compute_sobolev_lambda_feedback_raw_table(specs, ctx),
        index_col=[0, 1],
    )

    return (
        sobolev_lambda_tsr_raw_table,
        sobolev_lambda_kernel_raw_table,
        sobolev_lambda_feedback_raw_table,
        sobolev_seed_mean_std_table(sobolev_lambda_tsr_raw_table),
        sobolev_seed_mean_std_table(sobolev_lambda_kernel_raw_table),
        sobolev_seed_mean_std_table(sobolev_lambda_feedback_raw_table),
    )
