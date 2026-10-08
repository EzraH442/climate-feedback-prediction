from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from config_utils import load_config, variable_config_from_omegaconf
from experiments.common import EnsembleModel, parse_seed_ranges
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
)


def global_tsr_test(
    ds: xr.Dataset,
    preprocessor: Preprocessor,
    model: SimpleModel,
    config,
    figures_path: Path = Path("."),
):
    lat = ds.latitude.values
    lon = ds.longitude.values

    target = config.dataset.target_var
    target_label = target.upper()
    output_dir = figure_dir(figures_path, "all", "tsr_test")

    pred = nn_pred(ds, model, preprocessor, config, ["date", "latitude", "longitude"])
    true = preprocessor.preprocessors[-1].inverse_transform(ds)[target]

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


def kernel_ecod_fal_contour_test(
    processed_ds: xr.Dataset,
    preprocessor: Preprocessor,
    model: SimpleModel,
    config,
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
    config,
    figures_path: Path = Path("."),
    date="2015-09",
    latitude=83.625,
    longitude=17.375,
    scatter=True,
    extrapolate=False,
    n=100,
):
    scaler = preprocessor.scalar
    vconf = variable_config_from_omegaconf(config)
    feature_names = vconf.input_order()

    ordered = vconf.inputs(processed_ds)

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

    target = config.dataset.target_var
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
    config,
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
    config,
    figures_path: Path = Path("."),
    date="2015-09",
    latitude=83.625,
    longitude=17.375,
    scatter=True,
    extrapolate=False,
    n=100,
):
    scaler = preprocessor.scalar
    vconf = variable_config_from_omegaconf(config)
    target_var = vconf.target_var
    feature_names = vconf.input_order()

    ordered = vconf.inputs(processed_ds)
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
    config,
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
    north_mask = plot_lat >= 60
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
        vmin=-1,
        vmax=1,
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
        vmin=-1,
        vmax=1,
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
        vmin=-1,
        vmax=1,
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


def second_order_test(
    ds: xr.Dataset,
    preprocessor: Preprocessor,
    model: SimpleModel,
    config,
    true_kernel: xr.Dataset,
    dates=["2013-09", "2012-09"],
    figures_path: Path = Path("."),
    kernel_name=None,
):
    kernel_name = kernel_name or eval_kernel_vars(config)[0]
    output_dir = figure_dir(
        figures_path,
        "clr",
        "second_order_test",
        kernel_name,
        f"{dates[1]}_minus_{dates[0]}",
    )
    nn_kern_cld_1, lon, lat = compute_nn_kernel(
        ds.sel(date=dates[1]), preprocessor, model, config, perturbation_var=kernel_name
    )
    nn_kern_cld_0, _, _ = compute_nn_kernel(
        ds.sel(date=dates[0]), preprocessor, model, config, perturbation_var=kernel_name
    )
    delta_kern_nn = nn_kern_cld_1 - nn_kern_cld_0
    nn_grad_cld_1, grad_lon, grad_lat = compute_nn_kernel_autograd(
        ds.sel(date=dates[1]), preprocessor, model, config, var=kernel_name
    )
    nn_grad_cld_0, _, _ = compute_nn_kernel_autograd(
        ds.sel(date=dates[0]), preprocessor, model, config, var=kernel_name
    )
    delta_kern_nn_grad = nn_grad_cld_1 - nn_grad_cld_0

    rrtm_lat, rrtm_lon = true_kernel.latitude, true_kernel.longitude
    rrtm_kern_cld_1 = true_kernel[kernel_name].sel(
        all_clr="clr", date=dates[1]
    ).as_numpy()[0] * kernel_delta(kernel_name)
    rrtm_kern_cld_0 = true_kernel[kernel_name].sel(
        all_clr="clr", date=dates[0]
    ).as_numpy()[0] * kernel_delta(kernel_name)
    delta_kern_rrtm = rrtm_kern_cld_1 - rrtm_kern_cld_0

    if delta_kern_nn.shape != delta_kern_rrtm.shape:
        delta_kern_nn = interpolate_spatial_field(
            delta_kern_nn,
            lon,
            lat,
            rrtm_lon,
            rrtm_lat,
        )
        plot_lon = rrtm_lon
        plot_lat = rrtm_lat
    else:
        plot_lon = rrtm_lon
        plot_lat = rrtm_lat
    if delta_kern_nn_grad.shape != delta_kern_rrtm.shape:
        delta_kern_nn_grad = interpolate_spatial_field(
            delta_kern_nn_grad,
            grad_lon,
            grad_lat,
            rrtm_lon,
            rrtm_lat,
        )
        plot_grad_lon = rrtm_lon
        plot_grad_lat = rrtm_lat
    else:
        plot_grad_lon = rrtm_lon
        plot_grad_lat = rrtm_lat
    grad_north_mask = plot_grad_lat >= 60

    delta_k_diff_grad = delta_kern_nn_grad - delta_kern_rrtm
    delta_k_nn_grad = delta_kern_nn_grad
    delta_k_rrtm = delta_kern_rrtm.to_numpy()

    # plot_north_pole_field(
    #    delta_k_nn, plot_lon, plot_lat,
    #    "NN surface albedo kernel difference\n"+f"({dates[1]} minus f{dates[0]})",
    #    figures_path / "delta_k_nn_np.png",
    #    vmin=-1, vmax=1,
    # )
    plot_north_pole_field(
        delta_k_nn_grad,
        plot_grad_lon,
        plot_grad_lat,
        "NN autograd kernel difference\n" + f"({dates[1]} minus {dates[0]})",
        output_dir / "nn_grad_np.png",
        vmin=-1,
        vmax=1,
    )
    plot_north_pole_field(
        delta_k_rrtm,
        plot_lon,
        plot_lat,
        "ERA5 surface albedo kernel difference\n" + f"({dates[1]} minus {dates[0]})",
        output_dir / "rrtm_np.png",
        vmin=-1,
        vmax=1,
    )
    # plot_north_pole_field(
    #    delta_k_diff, plot_lon, plot_lat,
    #    r"$K_{NN} - K_{ERA5}$" + "\n"+f"({dates[1]} minus f{dates[0]})",
    #    figures_path / "delta_k_nn-rrtm_north_pole_2013-09_minus_2012-09.png",
    #    vmin=-1, vmax=1,
    # )
    plot_north_pole_field(
        delta_k_diff_grad,
        plot_grad_lon,
        plot_grad_lat,
        r"$K_{NN,\mathrm{grad}} - K_{ERA5}$" + "\n" + f"({dates[1]} minus {dates[0]})",
        output_dir / "nn_grad-rrtm_np.png",
        vmin=-1,
        vmax=1,
        annotation=f"{np.mean(delta_k_diff_grad[grad_north_mask]):.2f}; {np.mean(np.abs(delta_k_diff_grad[grad_north_mask])):.2f}",
    )


def run_ensemble_eval(args):
    # --- load model ---
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
            config, checkpoint_path
        )
        models.append(model)
        epochs.append(epoch)
    model = EnsembleModel(models).eval()
    epoch = epochs[0] if len(epochs) == 1 else f"ensemble_{len(models)}"

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

    raw_dataset = xr.open_mfdataset(raw_era5_paths, combine="nested", concat_dim="date")

    # --- load preprocessor ---
    preprocessor = Preprocessor(config)
    preprocessor.load(config.preprocess.params_dir)

    # --- run tests ---
    # global_tsr_test(
    #    ds=processed_dataset,
    #    preprocessor=preprocessor,
    #    model=model,
    #    config=config,
    #    figures_path=output_dir,
    # )#

    for kernel_name in ["tcwv"]:  ##eval_kernel_vars(config):
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
        #    config=config,
        #    figures_path=output_dir,
        #    true_kernel=kernels_dataset.sel(date="2013-09"),
        #    date="2013-09",
        #    kernel_name=kernel_name,
        # )
        kernel_date_test(
            ds=raw_dataset.sel(date="2015-09"),
            preprocessor=preprocessor,
            model=model,
            config=config,
            figures_path=output_dir,
            true_kernel=kernels_dataset.sel(date="2015-09"),
            date="2015-09",
            kernel_name=kernel_name,
        )
        # second_order_test(
        #    ds=raw_dataset,
        #    preprocessor=preprocessor,
        #    model=model,
        #    config=config,
        #    true_kernel=kernels_dataset,
        #    dates=["2012-09", "2013-09"],
        #    figures_path=output_dir,
        #    kernel_name=kernel_name,
        # )
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

    # kernel_ecod_fal_contour_test(
    #    processed_ds=processed_dataset,
    #    preprocessor=preprocessor,
    #    model=model,
    #    config=config,
    #    figures_path=output_dir,
    #    scatter=False,
    #    extrapolate=False,
    #    date="2015-09",
    # )
    # tsr_ecod_fal_contour_test(
    #    processed_ds=processed_dataset,
    #    preprocessor=preprocessor,
    #    model=model,
    #    config=config,
    #    figures_path=output_dir,
    #    scatter=False,
    #    extrapolate=False,
    #    date="2015-09",
    # )
