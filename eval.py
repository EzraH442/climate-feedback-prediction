import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from scipy.stats import linregress

from config_utils import load_config
from preprocess import load_cloud_profiles
from preprocessing import (
    Preprocessor,
    SequentialPreprocessor,
    XarrayMinMaxScaler,
    create_2024_preprocessor,
)

from utils import (
    plot_global_field, plot_north_pole_field,
    interpolate_spatial_field,
    dataset_from_array, ordered_dataset,
    preprocessed_feature_target_arrays, empirical_copula_values,
    unpreprocess_feature_values, inverse_transform_without_upsampling, make_kernel_filename,
    make_era5_filename,
    SECONDS_PER_DAY,
    ordered_dataset_for_config, kernel_delta,
    kernel_title, kernel_label, clear_sky_raw, scale_minmax_value,
    compute_nn_kernel, compute_nn_kernel_autograd,
)

from model import SimpleModel


def setup_correlation_plot():
    fig, ax = plt.subplots(figsize=(6, 6), dpi=300)
    return fig, ax


def plot_correlation(ax, predictions, actuals):
    ax.scatter(predictions, actuals, alpha=0.5)
    slope, intercept, r_value, p_value, std_err = linregress(predictions, actuals)
    line_x = np.array([predictions.min(), predictions.max()])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, color="red", label=f"R²={r_value**2:.2f}")
    ax.legend()

def global_tsr_test(
    ds: xr.Dataset,
    preprocessor: Preprocessor,
    model: torch.nn.Module,
    config,
    figures_path: Path = Path("."),
):
    assert isinstance(preprocessor, SequentialPreprocessor)
    # preprocessors=[
    #    ECOD_Calculator(),
    #    Downscaler(factor=[("latitude", 4), ("longitude", 4)]),
    #    XarrayMinMaxScaler(dim=("date", "latitude", "longitude")),
    # ]
    
    date = ds.date.values
    lat = ds.latitude.values
    lon = ds.longitude.values

    target = config.dataset.target_var
    target_label = target.upper()

    ds_ordered_np = (
        ordered_dataset_for_config(ds, config)
        .to_dataarray()
        .transpose("date", "latitude", "longitude", "variable")
        .to_numpy()
    )

    data_torch = torch.from_numpy(ds_ordered_np).float()
    model_outputs = model(data_torch[:, :, :, : model.input_dim]).detach()

    pred = (
        preprocessor.preprocessors[-1]
        .inverse_transform(
            dataset_from_array(
                arr=model_outputs.numpy(), date=date, lon=lon, lat=lat, target_var=target
            ),
        )
        .to_dataarray()
        .squeeze(dim="variable", drop=True)
    )
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
        tsr_mean, lon, lat, f"{target_label} (ERA5)", figures_path / f"global_{target}_era5.png",
        cmap="Spectral", vmin=min_tsr, vmax=max_tsr, label="$W/m^2$",
        annotation=f"{global_tsr_mean:.2f}",
    )
    plot_global_field(
        tsr_pred_mean, lon, lat, f"{target_label} (NN)", figures_path / f"global_{target}_nn.png",
        cmap="Spectral", vmin=min_tsr, vmax=max_tsr, label="$W/m^2$",
        annotation=f"{global_tsr_pred_mean:.2f}",
    )
    plot_global_field(
        mbe_map, lon, lat, "MBE", figures_path / f"global_{target}_mbe.png",
        cmap="RdBu_r", vmin=-20, vmax=20, label="$W/m^2$",
        annotation=f"{global_mbe:.2f}",
    )
    plot_global_field(
        rmse_map, lon, lat, "RMSE", figures_path / f"global_{target}_rmse.png",
        cmap="Blues", vmin=0, vmax=20, label="$W/m^2$",
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
    date='2015-09',
    latitude=83.625,
    longitude=17.375,
    scatter=True,
    extrapolate=False,
    n_fal: int = 100,
    n_ecod: int = 100,
):
    assert isinstance(preprocessor, SequentialPreprocessor)
    scaler = preprocessor.preprocessors[-1]
    assert isinstance(scaler, XarrayMinMaxScaler)
    if 'fal' not in config.dataset.input_vars:
        print("Skipping fal/ecod kernel contour; model input variable is not fal.")
        return

    ordered = ordered_dataset_for_config(processed_ds, config)
    feature_names = [v for v in ordered.data_vars if v != config.dataset.target_var]
    required = {"fal", "ecod"}
    missing = required.difference(feature_names)
    if missing:
        print(f"Skipping fal/ecod kernel contour; missing features: {sorted(missing)}")
        return

    base_point = ordered.sel(
        date=date,
        latitude=latitude,
        longitude=longitude,
    )
    observed_base_point = scaler.inverse_transform(ordered[["fal", "ecod"]]).sel(
        date=date,
        latitude=latitude,
        longitude=longitude,
    )
    base_features = np.array(
        [float(base_point[name]) for name in feature_names],
        dtype=np.float32,
    )

    fal_values = np.linspace(
        float(scaler.data_min_["fal"]),
        float(scaler.data_max_["fal"]),
        n_fal,
    )
    ecod_values = np.linspace(
        float(scaler.data_min_["ecod"]),
        float(scaler.data_max_["ecod"]),
        n_ecod,
    )
    if extrapolate:
        fal_values = np.linspace(
            0,
            float(scaler.data_max_["fal"])*2,
            n_fal,
        )
        ecod_values = np.linspace(
            0,
            float(scaler.data_max_["ecod"])*2,
            n_ecod,
        )
    fal_grid, ecod_grid = np.meshgrid(fal_values, ecod_values)

    inputs_np = np.broadcast_to(
        base_features, (n_ecod * n_fal, len(feature_names))
    ).copy()
    inputs_np[:, feature_names.index("fal")] = scale_minmax_value(
        scaler, "fal", fal_grid.ravel()
    )
    inputs_np[:, feature_names.index("ecod")] = scale_minmax_value(
        scaler, "ecod", ecod_grid.ravel()
    )
    if "ecod_fal" in feature_names:
        inputs_np[:, feature_names.index("ecod_fal")] = scale_minmax_value(
            scaler, "ecod_fal", (ecod_grid * fal_grid).ravel()
        )

    inputs = torch.from_numpy(inputs_np).float().requires_grad_(True)
    outputs = model(inputs)
    grads = torch.autograd.grad(outputs=outputs.sum(), inputs=inputs)[0][
        :, feature_names.index("fal")
    ]

    target = config.dataset.target_var
    target_range = float(scaler.data_max_[target] - scaler.data_min_[target])
    fal_range = float(scaler.data_max_["fal"] - scaler.data_min_["fal"])
    kernel = (
        grads.detach().cpu().numpy() * (target_range / fal_range) / SECONDS_PER_DAY
    ).reshape(n_ecod, n_fal) * kernel_delta(config)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    max_abs = float(np.nanmax(np.abs(kernel)))
    levels = np.linspace(-max_abs, max_abs, 31) if max_abs > 0 else 31
    contour = ax.contourf(
        fal_grid,
        ecod_grid,
        kernel,
        levels=levels,
        cmap="RdBu_r",
        extend="both",
    )
    observed = scaler.inverse_transform(ordered[["fal", "ecod"]])
    observed_fal = observed["fal"].to_numpy().ravel()
    observed_ecod = observed["ecod"].to_numpy().ravel()
    #step = max(1, observed_fal.size // 50000)
    if scatter:
        ax.scatter(
            observed_fal,#[::step],
            observed_ecod,#[::step],
            s=1,
            c="black",
            alpha=0.05,
            linewidths=0,
            label="dataset",
        )
        ax.legend(loc="upper right", markerscale=4)
    ax.scatter(
        [float(observed_base_point["fal"])],
        [float(observed_base_point["ecod"])],
        s=20,
        c="white",
        edgecolors="black",
    )
    ax.set_xlabel("fal")
    ax.set_ylabel("ecod")
    ax.set_title(f"NN surface albedo kernel over fal/ecod; lat={latitude:.2f}, lon={longitude:.2f}, date={date}")
    cb = fig.colorbar(contour, ax=ax)
    cb.set_label(kernel_label(config))
    fig.tight_layout()
    fig.savefig(figures_path / "kernel_contour_fal_ecod.png")
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
    assert isinstance(preprocessor, SequentialPreprocessor)
    scaler = preprocessor.preprocessors[-1]
    assert isinstance(scaler, XarrayMinMaxScaler)
    if 'fal' not in config.dataset.input_vars:
        print("Skipping fal/ecod TSR contour; model input variable is not fal.")
        return

    ordered = ordered_dataset_for_config(processed_ds, config)
    target = config.dataset.target_var
    feature_names = [v for v in ordered.data_vars if v != target]
    required = {"fal", "ecod"}
    missing = required.difference(feature_names)
    if missing:
        print(f"Skipping fal/ecod TSR contour; missing features: {sorted(missing)}")
        return

    observed = scaler.inverse_transform(ordered[["fal", "ecod"]])
    base_point = ordered.sel(
        date=date,
        latitude=latitude,
        longitude=longitude,
    )
    observed_base_point = observed.sel(
        date=date,
        latitude=latitude,
        longitude=longitude,
    )
    base_features = np.array(
        [float(base_point[name]) for name in feature_names],
        dtype=np.float32,
    )

    fal_values = np.linspace(
        float(scaler.data_min_["fal"]),
        float(scaler.data_max_["fal"]),
        n_fal,
    )
    ecod_values = np.linspace(
        float(scaler.data_min_["ecod"]),
        float(scaler.data_max_["ecod"]),
        n_ecod,
    )
    if extrapolate:
        fal_values = np.linspace(0, float(scaler.data_max_["fal"]) * 2, n_fal)
        ecod_values = np.linspace(0, float(scaler.data_max_["ecod"]) * 2, n_ecod)
    fal_grid, ecod_grid = np.meshgrid(fal_values, ecod_values)

    inputs_np = np.broadcast_to(
        base_features, (n_ecod * n_fal, len(feature_names))
    ).copy()
    inputs_np[:, feature_names.index("fal")] = scale_minmax_value(
        scaler, "fal", fal_grid.ravel()
    )
    inputs_np[:, feature_names.index("ecod")] = scale_minmax_value(
        scaler, "ecod", ecod_grid.ravel()
    )
    if "ecod_fal" in feature_names:
        inputs_np[:, feature_names.index("ecod_fal")] = scale_minmax_value(
            scaler, "ecod_fal", (ecod_grid * fal_grid).ravel()
        )

    with torch.no_grad():
        outputs = (
            model(torch.from_numpy(inputs_np).float())
            .cpu()
            .numpy()
            .reshape(n_ecod, n_fal)
        )

    target_min = float(scaler.data_min_[target])
    target_range = float(scaler.data_max_[target] - scaler.data_min_[target])
    target_scaled = (outputs - scaler.min_val) / (scaler.max_val - scaler.min_val)
    target_values = (target_scaled * target_range + target_min) / SECONDS_PER_DAY

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    contour = ax.contourf(
        fal_grid,
        ecod_grid,
        target_values,
        levels=31,
        cmap="Spectral",
    )
    if scatter:
        ax.scatter(
            observed["fal"].to_numpy().ravel(),
            observed["ecod"].to_numpy().ravel(),
            s=1,
            c="black",
            alpha=0.05,
            linewidths=0,
            label="dataset",
        )
        ax.legend(loc="upper right", markerscale=4)
    ax.scatter(
        [float(observed_base_point["fal"])],
        [float(observed_base_point["ecod"])],
        s=20,
        c="white",
        edgecolors="black",
    )
    ax.set_xlabel("fal")
    ax.set_ylabel("ecod")
    ax.set_title(
        f"NN {target.upper()} over fal/ecod; "
        f"lat={latitude:.2f}, lon={longitude:.2f}, date={date}"
    )
    cb = fig.colorbar(contour, ax=ax)
    cb.set_label("$W/m^2$")
    fig.tight_layout()
    fig.savefig(figures_path / f"{target}_contour_fal_ecod.png")
    plt.close(fig)


def kernel_date_test(
    ds: xr.Dataset,
    preprocessor: Preprocessor,
    model: SimpleModel,
    config,
    date,
    true_kernel: xr.Dataset,
    figures_path: Path = Path(".")
):
    ds_clr = clear_sky_raw(ds, config)
    title = kernel_title(config)
    label = kernel_label(config)

    nn_kern_cld, lon, lat = compute_nn_kernel(ds, preprocessor, model, config)
    nn_kern_clr, _, _ = compute_nn_kernel(ds_clr, preprocessor, model, config)
    nn_grad_cld, grad_lon, grad_lat = compute_nn_kernel_autograd(ds, preprocessor, model, config)
    nn_grad_clr, _, _ = compute_nn_kernel_autograd(ds_clr, preprocessor, model, config)
    
    rrtm_kern_cld = true_kernel["TOA_cld"].as_numpy()[0] * kernel_delta(config)
    rrtm_kern_clr = true_kernel["TOA_clr"].as_numpy()[0] * kernel_delta(config)
    kern_lon, kern_lat = true_kernel.longitude, true_kernel.latitude
    # --- clear and all sky plots of nn kernel, global and north pole
    #plot_global_field(
    #    nn_kern_cld, lon, lat,
    #    f"NN Surface Albedo Kernel (all)\n{date}",
    #    figures_path / f"kern_all_nn_{date}.png",
    #    cmap="RdBu_r", vmin=-4, vmax=4, label=r"$W/m^2 1\%$",
    #    annotation=f"{np.mean(nn_kern_cld):.2f}",
    #)
    #plot_global_field(
    #    nn_kern_clr, lon, lat,
    #    f"NN Surface Albedo Kernel (clear)\n{date}",
    #    figures_path / f"kern_clr_nn_{date}.png",
    #    cmap="RdBu_r", vmin=-4, vmax=4, label=r"$W/m^2 1\%$",
    #    annotation=f"{np.mean(nn_kern_clr):.2f}",
    #)
    plot_global_field(
        nn_grad_cld, grad_lon, grad_lat,
        f"NN {title} Kernel via autograd (all)\n{date}",
        figures_path / f"kern_all_nn_grad_{date}.png",
        cmap="RdBu_r", vmin=-3, vmax=3, label=label,
        annotation=f"{np.mean(nn_grad_cld):.2f}",
    )
    plot_global_field(
        nn_grad_clr, grad_lon, grad_lat,
        f"NN {title} Kernel via autograd (clear)\n{date}",
        figures_path / f"kern_clr_nn_grad_{date}.png",
        cmap="RdBu_r", vmin=-3, vmax=3, label=label,
        annotation=f"{np.mean(nn_grad_clr):.2f}",
    )
    north_mask = lat >= 60
    #plot_north_pole_field(
    #    nn_kern_cld[north_mask], lon, lat[north_mask],
    #    f"NN Surface Albedo Kernel (all)\n{date}",
    #    figures_path / f"kern_all_nn_np_{date}.png",
    #    cmap="RdBu_r", vmin=-4, vmax=4, label=r"$W/m^2 1\%$",
    #    annotation=f"{np.mean(nn_kern_cld[north_mask]):.2f}",
    #)
    #plot_north_pole_field(
    #    nn_kern_clr[north_mask], lon, lat[north_mask],
    #    f"NN Surface Albedo Kernel (clear)\n{date}",
    #    figures_path / f"kern_clr_nn_np_{date}.png",
    #    cmap="RdBu_r", vmin=-4, vmax=4, label=r"$W/m^2 1\%$",
    #    annotation=f"{np.mean(nn_kern_clr[north_mask]):.2f}",
    #)
    grad_north_mask = grad_lat >= 60
    plot_north_pole_field(
        nn_grad_cld, grad_lon, grad_lat,
        f"NN {title} Kernel via autograd (all)\n{date}",
        figures_path / f"kern_all_nn_grad_np_{date}.png",
        cmap="RdBu_r", vmin=-3, vmax=3, label=label,
        annotation=f"{np.mean(nn_grad_cld[grad_north_mask]):.2f}",
    )
    plot_north_pole_field(
        nn_grad_clr, grad_lon, grad_lat,
        f"NN {title} Kernel via autograd (clear)\n{date}",
        figures_path / f"kern_clr_nn_grad_np_{date}.png",
        cmap="RdBu_r", vmin=-3, vmax=3, label=label,
        annotation=f"{np.mean(nn_grad_clr[grad_north_mask]):.2f}",
    )
    # --- clear and all sky plots of rrtm kernel, global and north pole
    plot_global_field(
        rrtm_kern_cld, kern_lon, kern_lat,
        f"RRTM {title} Kernel (all)\n{date}",
        figures_path / f"kern_all_rrtm_{date}.png",
        cmap="RdBu_r", vmin=-3, vmax=3, label=label,
        annotation=f"{np.mean(rrtm_kern_cld):.2f}",
    )
    plot_global_field(
        rrtm_kern_clr, kern_lon, kern_lat,
        f"RRTM {title} Kernel (clear)\n{date}",
        figures_path / f"kern_clr_rrtm_{date}.png",
        cmap="RdBu_r", vmin=-3, vmax=3, label=label,
        annotation=f"{np.mean(rrtm_kern_clr):.2f}",
    )    
    north_mask = kern_lat >= 60
    plot_north_pole_field(
        rrtm_kern_cld, kern_lon, kern_lat,
        f"RRTM {title} Kernel (all)\n{date}",
        figures_path / f"kern_all_rrtm_np_{date}.png",
        cmap="RdBu_r", vmin=-3, vmax=3, label=label,
        annotation=f"{np.mean(rrtm_kern_clr[north_mask]):.2f}",
    )
    plot_north_pole_field(
        rrtm_kern_clr, kern_lon, kern_lat,
        f"RRTM {title} Kernel (clear)\n{date}",
        figures_path / f"kern_clr_rrtm_np_{date}.png",
        cmap="RdBu_r", vmin=-3, vmax=3, label=label,
        annotation=f"{np.mean(rrtm_kern_cld[north_mask]):.2f}",
    )
    # --- clear and all sky plots of nn-rrtm kernel difference, global and north pole

    plot_lon, plot_lat = lon, lat
    plot_grad_lon, plot_grad_lat = grad_lon, grad_lat
    if nn_kern_clr.shape != true_kernel["TOA_clr"].shape:
        nn_kern_cld = interpolate_spatial_field(
            nn_kern_cld, lon, lat, kern_lon, kern_lat
        )
        nn_kern_clr = interpolate_spatial_field(
            nn_kern_clr, lon, lat, kern_lon, kern_lat
        )
        plot_lon = kern_lon
        plot_lat = kern_lat
    if nn_grad_clr.shape != true_kernel["TOA_clr"].shape:
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
    
    diff_cld      = nn_kern_cld - rrtm_kern_cld
    diff_clr      = nn_kern_clr - rrtm_kern_clr
    diff_grad_cld = nn_grad_cld - rrtm_kern_cld
    diff_grad_clr = nn_grad_clr - rrtm_kern_clr
    
    #plot_global_field(
    #    diff_cld, plot_lon, plot_lat,
    #    f"NN-RRTM Surface Albedo Kernel (all)\n{date}",
    #    figures_path / f"kern_all_nn-rrtm_{date}.png",
    #    cmap="RdBu_r", vmin=-2, vmax=2, label=r"$W/m^2 1\%$",
    #    annotation=f"{np.mean(diff_cld):.2f}",
    #)
    #plot_global_field(
    #    diff_clr, plot_lon, plot_lat,
    #    f"NN-RRTM Surface Albedo Kernel (clear)\n{date}",
    #    figures_path / f"kern_clr_nn-rrtm_{date}.png",
    #    cmap="RdBu_r", vmin=-2, vmax=2, label=r"$W/m^2 1\%$",
    #    annotation=f"{np.mean(diff_clr):.2f}",
    #)    
    plot_global_field(
        diff_grad_cld, plot_grad_lon, plot_grad_lat,
        f"NN autograd-RRTM {title} Kernel (all)\n{date}",
        figures_path / f"kern_all_nn_grad-rrtm_{date}.png",
        cmap="RdBu_r", vmin=-0.3, vmax=0.3, label=label,
        annotation=f"{np.mean(diff_grad_cld):.2f}; {np.mean(np.abs(diff_grad_cld)):.2f}",
    )
    plot_global_field(
        diff_grad_clr, plot_grad_lon, plot_grad_lat,
        f"NN autograd-RRTM {title} Kernel (clear)\n{date}",
        figures_path / f"kern_clr_nn_grad-rrtm_{date}.png",
        cmap="RdBu_r", vmin=-0.3, vmax=0.3, label=label,
        annotation=f"{np.mean(diff_grad_clr):.2f}; {np.mean(np.abs(diff_grad_clr)):.2f}",
    )
    north_mask = plot_lat >= 60
    #plot_north_pole_field(
    #    diff_cld[north_mask], plot_lon, plot_lat[north_mask],
    #    f"NN-RRTM Surface Albedo Kernel (all)\n{date}",
    #    figures_path / f"kern_all_nn-rrtm_np_{date}.png",
    #    cmap="RdBu_r", vmin=-1, vmax=1, label=r"$W/m^2 1\%$",
    #    annotation=f"{np.mean(diff_cld[north_mask]):.2f}",
    #)
    #plot_north_pole_field(
    #    diff_clr[north_mask], plot_lon, plot_lat[north_mask],
    #    f"NN-RRTM Surface Albedo Kernel (clear)\n{date}",
    #    figures_path / f"kern_clr_nn-rrtm_np_{date}.png",
    #    cmap="RdBu_r", vmin=-1, vmax=1, label=r"$W/m^2 1\%$",
    #    annotation=f"{np.mean(diff_clr[north_mask]):.2f}",
    #)
    plot_north_pole_field(
        diff_grad_cld, plot_grad_lon, plot_grad_lat,
        f"NN autograd-RRTM {title} Kernel (all)\n{date}",
        figures_path / f"kern_all_nn_grad-rrtm_np_{date}.png",
        cmap="RdBu_r", vmin=-1, vmax=1, label=label,
        annotation=f"{np.mean(diff_grad_cld[grad_north_mask]):.2f}; {np.mean(np.abs(diff_grad_cld[grad_north_mask])):.2f}",
    )
    plot_north_pole_field(
        diff_grad_clr, plot_grad_lon, plot_grad_lat,
        f"NN autograd-RRTM {title} Kernel (clear)\n{date}",
        figures_path / f"kern_clr_nn_grad-rrtm_np_{date}.png",
        cmap="RdBu_r", vmin=-1, vmax=1, label=label,
        annotation=f"{np.mean(diff_grad_clr[grad_north_mask]):.2f}; {np.mean(np.abs(diff_grad_clr[grad_north_mask])):.2f}",
    )

def second_order_test(
    ds: xr.Dataset,
    preprocessor: Preprocessor,
    model: SimpleModel,
    config,
    true_kernel: xr.Dataset,
    dates = ["2013-09", "2012-09"],
    figures_path: Path = Path(".")
):
    nn_kern_cld_1, lon, lat = compute_nn_kernel(ds.sel(date=dates[1]), preprocessor, model, config)
    nn_kern_cld_0, _, _ = compute_nn_kernel(ds.sel(date=dates[0]), preprocessor, model, config)
    delta_kern_nn = nn_kern_cld_1 - nn_kern_cld_0
    nn_grad_cld_1, grad_lon, grad_lat = compute_nn_kernel_autograd(
        ds.sel(date=dates[1]), preprocessor, model, config
    )
    nn_grad_cld_0, _, _ = compute_nn_kernel_autograd(
        ds.sel(date=dates[0]), preprocessor, model, config
    )
    delta_kern_nn_grad = nn_grad_cld_1 - nn_grad_cld_0

    rrtm_lat, rrtm_lon = true_kernel.latitude, true_kernel.longitude
    rrtm_kern_cld_1 = true_kernel["TOA_cld"].sel(date=dates[1]).as_numpy()[0] * kernel_delta(config)
    rrtm_kern_cld_0 = true_kernel["TOA_cld"].sel(date=dates[0]).as_numpy()[0] * kernel_delta(config)
    delta_kern_rrtm = rrtm_kern_cld_1 - rrtm_kern_cld_0


    if delta_kern_nn.shape != delta_kern_rrtm.shape:
        delta_kern_nn = interpolate_spatial_field(
            delta_kern_nn, lon, lat, rrtm_lon, rrtm_lat,
        )
        plot_lon = rrtm_lon
        plot_lat = rrtm_lat
    else:
        plot_lon = rrtm_lon
        plot_lat = rrtm_lat
    if delta_kern_nn_grad.shape != delta_kern_rrtm.shape:
        delta_kern_nn_grad = interpolate_spatial_field(
            delta_kern_nn_grad, grad_lon, grad_lat, rrtm_lon, rrtm_lat,
        )
        plot_grad_lon = rrtm_lon
        plot_grad_lat = rrtm_lat
    else:
        plot_grad_lon = rrtm_lon
        plot_grad_lat = rrtm_lat
    north_mask = plot_lat >= 60
    grad_north_mask = plot_grad_lat >= 60
    
    delta_k_diff = (delta_kern_nn - delta_kern_rrtm)
    delta_k_diff_grad = (delta_kern_nn_grad - delta_kern_rrtm)
    delta_k_nn = delta_kern_nn
    delta_k_nn_grad = delta_kern_nn_grad
    delta_k_rrtm = delta_kern_rrtm

    max_abs_diff = np.max(np.abs(delta_k_diff))

    #plot_north_pole_field(
    #    delta_k_nn, plot_lon, plot_lat,
    #    "NN surface albedo kernel difference\n"+f"({dates[1]} minus f{dates[0]})",
    #    figures_path / "delta_k_nn_np.png",
    #    vmin=-1, vmax=1,
    #)
    plot_north_pole_field(
        delta_k_nn_grad, plot_grad_lon, plot_grad_lat,
        "NN autograd kernel difference\n"+f"({dates[1]} minus {dates[0]})",
        figures_path / "delta_k_nn_grad_np.png",
        vmin=-1, vmax=1,
    )
    plot_north_pole_field(
        delta_k_rrtm, plot_lon, plot_lat,
        "ERA5 surface albedo kernel difference\n"+f"({dates[1]} minus {dates[0]})",
        figures_path / "delta_k_rrtm_np.png",
        vmin=-1, vmax=1,
    )
    #plot_north_pole_field(
    #    delta_k_diff, plot_lon, plot_lat,
    #    r"$K_{NN} - K_{ERA5}$" + "\n"+f"({dates[1]} minus f{dates[0]})",
    #    figures_path / "delta_k_nn-rrtm_north_pole_2013-09_minus_2012-09.png",
    #    vmin=-1, vmax=1,
    #)
    plot_north_pole_field(
        delta_k_diff_grad, plot_grad_lon, plot_grad_lat,
        r"$K_{NN,\mathrm{grad}} - K_{ERA5}$" + "\n" + f"({dates[1]} minus {dates[0]})",
        figures_path / "delta_k_nn_grad-rrtm_north_pole.png",
        vmin=-1, vmax=1, annotation=f"{np.mean(delta_k_diff_grad[grad_north_mask]):.2f}; {np.mean(np.abs(delta_k_diff_grad[grad_north_mask])):.2f}"
    )



def test_4(
    raw_dataset: xr.Dataset,
    preprocessor: Preprocessor,
    model: SimpleModel,
    figures_path: Path = Path("."),
    batch_size: int = 8192,
):
    try:
        from captum.attr import IntegratedGradients
    except ImportError as exc:
        raise ImportError(
            "Captum is required for test_4. Install the `captum` package first."
        ) from exc

    preprocessed_dataset = preprocessor.transform(raw_dataset)
    ordered_preprocessed = ordered_dataset(preprocessed_dataset)
    feature_names = [v for v in ordered_preprocessed.data_vars if v != "tsr"]

    data = (
        ordered_preprocessed.to_dataarray()
        .transpose("date", "latitude", "longitude", "variable")
        .to_numpy()
    )
    inputs = torch.from_numpy(data[..., : model.input_dim]).float()
    baselines_inputs = inputs.mean(dim=0).expand(inputs.shape)

    flattened_inputs = inputs.reshape(-1, model.input_dim)
    baselines = baselines_inputs.reshape(-1, model.input_dim)
    # baselines = torch.zeros_like(flattened_inputs) gives different results

    print("Running integrated gradients...")
    ig = IntegratedGradients(model)
    feature_sums = torch.zeros(model.input_dim, dtype=torch.float32)
    total_samples = flattened_inputs.shape[0]

    for start in range(0, total_samples, batch_size):
        print(f"{start}/{total_samples}")
        end = min(start + batch_size, total_samples)
        batch_inputs = flattened_inputs[start:end]
        batch_baselines = baselines[start:end]
        attributions = ig.attribute(batch_inputs, baselines=batch_baselines)
        feature_sums += attributions.abs().sum(dim=0).cpu()
        if start > 1000:
            break

    mean_feature_importance = (feature_sums / total_samples).numpy()

    print("Average Captum feature importance over validation data:")
    for name, value in zip(feature_names, mean_feature_importance):
        print(f"  {name}: {value:.6e}")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.bar(feature_names, mean_feature_importance)
    ax.set_ylabel("Mean absolute attribution")
    ax.set_title("Captum Feature Importance on Validation Data")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(figures_path / "captum_feature_importance_validation.png")
    plt.close(fig)

    output_csv = figures_path / "captum_feature_importance_validation.csv"
    with output_csv.open("w", encoding="ascii") as f:
        f.write("feature,mean_absolute_attribution\n")
        for name, value in zip(feature_names, mean_feature_importance):
            f.write(f"{name},{value:.10e}\n")


def test_5(
    raw_dataset: xr.Dataset,
    preprocessor: Preprocessor,
    model: SimpleModel,
    figures_path: Path = Path("."),
    num_bins: int = 100,
    max_scatter_points: int = 50000,
):
    inputs, targets, feature_names, latitudes, ordered_preprocessed = (
        preprocessed_feature_target_arrays(
        raw_dataset=raw_dataset,
        preprocessor=preprocessor,
        model=model,
        )
    )

    with torch.no_grad():
        predictions = model(torch.from_numpy(inputs).float()).cpu().numpy()
    losses = (predictions - targets) ** 2
    squared_errors = losses

    target_unprocessed = inverse_transform_without_upsampling(
        dataset_from_array(
            arr=targets.reshape(
                len(ordered_preprocessed["date"]),
                len(ordered_preprocessed["latitude"]),
                len(ordered_preprocessed["longitude"]),
            ),
            date=ordered_preprocessed["date"].values,
            lon=ordered_preprocessed["longitude"].values,
            lat=ordered_preprocessed["latitude"].values,
        ),
        preprocessor,
    )["tsr"].to_numpy().reshape(-1)

    rng = np.random.default_rng(0)
    scatter_indices = np.arange(len(losses))
    if len(scatter_indices) > max_scatter_points:
        scatter_indices = rng.choice(
            scatter_indices, size=max_scatter_points, replace=False
        )

    style = {
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.8,
    }

    latitude_bands = [
        ("0-30", (np.abs(latitudes) >= 0) & (np.abs(latitudes) < 30)),
        ("30-60", (np.abs(latitudes) >= 30) & (np.abs(latitudes) < 60)),
        ("60-90", (np.abs(latitudes) >= 60) & (np.abs(latitudes) <= 90)),
    ]
    band_colors = {
        "0-30": "#4FC3F7",
        "30-60": "#A6E22E",
        "60-90": "#FFB86C",
    }

    summary_csv = figures_path / "loss_landscape_summary.csv"
    with summary_csv.open("w", encoding="ascii") as f:
        f.write("feature,latitude_band,x_mean,y_mean,y_std,sample_count\n")

        for feature_index, feature_name in enumerate(feature_names):
            feature_values = inputs[:, feature_index]
            feature_values_unprocessed = unpreprocess_feature_values(
                feature_values=feature_values,
                feature_name=feature_name,
                preprocessor=preprocessor,
            )

            bin_edges = np.quantile(feature_values, np.linspace(0.0, 1.0, num_bins + 1))
            if np.unique(bin_edges).size < 2:
                bin_edges = np.linspace(
                    feature_values.min(), feature_values.max(), num_bins + 1
                )
            bin_edges = np.unique(bin_edges)

            if bin_edges.size < 2:
                print(
                    f"Skipping loss-landscape plots for {feature_name}; feature is constant."
                )
                continue

            # ── loss landscape plot ───────────────────────────────────────────
            with plt.rc_context(style):
                fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

                for band_name, band_mask in latitude_bands:
                    band_feature_values = feature_values[band_mask]
                    if band_feature_values.size == 0:
                        continue

                    band_bin_edges = np.quantile(
                        band_feature_values,
                        np.linspace(0.0, 1.0, num_bins + 1),
                    )
                    if np.unique(band_bin_edges).size < 2:
                        band_bin_edges = np.linspace(
                            band_feature_values.min(),
                            band_feature_values.max(),
                            num_bins + 1,
                        )
                    band_bin_edges = np.unique(band_bin_edges)
                    if band_bin_edges.size < 2:
                        continue

                    band_bin_indices = np.digitize(
                        feature_values[band_mask],
                        band_bin_edges[1:-1],
                        right=False,
                    )
                    band_x = []
                    band_y = []
                    band_y_std = []
                    band_counts = []

                    for bin_index in range(band_bin_edges.size - 1):
                        mask = band_bin_indices == bin_index
                        if not np.any(mask):
                            continue

                        x_values = feature_values_unprocessed[band_mask][mask]
                        if feature_name == "tsr":
                            x_values = target_unprocessed[band_mask][mask]
                            rmse = np.sqrt(np.mean(squared_errors[band_mask][mask]))
                            normalization = np.mean(np.abs(x_values))
                            y_mean = (
                                rmse / normalization if normalization > 0 else np.nan
                            )
                            y_std = 0.0
                        else:
                            y_values = losses[band_mask][mask]
                            y_mean = y_values.mean()
                            y_std = y_values.std()

                        band_x.append(x_values.mean())
                        band_y.append(y_mean)
                        band_y_std.append(y_std)
                        band_counts.append(mask.sum())
                        f.write(
                            f"{feature_name},{band_name},{band_x[-1]:.10e},{band_y[-1]:.10e},{band_y_std[-1]:.10e},{band_counts[-1]}\n"
                        )

                    if not band_x:
                        continue

                    band_x = np.asarray(band_x)
                    band_y = np.asarray(band_y)
                    band_y_std = np.asarray(band_y_std)
                    lower_band = np.clip(band_y - band_y_std, a_min=0.0, a_max=None)
                    upper_band = band_y + band_y_std
                    color = band_colors[band_name]

                    ax.fill_between(
                        band_x,
                        lower_band,
                        upper_band,
                        color=color,
                        alpha=0.12,
                        linewidth=0,
                    )
                    ax.plot(
                        band_x,
                        band_y,
                        color=color,
                        linewidth=1.2,
                        marker="o",
                        markersize=2.5,
                        markerfacecolor=color,
                        markeredgewidth=0,
                        label=band_name,
                    )

                ax.set_xlabel(
                    "tsr target value (unpreprocessed)"
                    if feature_name == "tsr"
                    else f"{feature_name}  (unpreprocessed)"
                )
                ax.set_ylabel(
                    "Normalized RMSE" if feature_name == "tsr" else "Mean squared error"
                )
                ax.set_title(
                    f"Loss landscape — {feature_name}"
                    if feature_name != "tsr"
                    else "Normalized RMSE vs tsr"
                )
                ax.legend(frameon=False, fontsize=8)
                fig.tight_layout()
                fig.savefig(figures_path / f"loss_vs_{feature_name}.png")
                plt.close(fig)

        if "tsr" not in feature_names:
            with plt.rc_context(style):
                fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
                for band_name, band_mask in latitude_bands:
                    band_target_values = target_unprocessed[band_mask]
                    if band_target_values.size == 0:
                        continue
                    band_bin_edges = np.quantile(
                        band_target_values, np.linspace(0.0, 1.0, num_bins + 1)
                    )
                    if np.unique(band_bin_edges).size < 2:
                        band_bin_edges = np.linspace(
                            band_target_values.min(),
                            band_target_values.max(),
                            num_bins + 1,
                        )
                    band_bin_edges = np.unique(band_bin_edges)
                    if band_bin_edges.size < 2:
                        continue
                    band_bin_indices = np.digitize(
                        band_target_values, band_bin_edges[1:-1], right=False
                    )

                    band_x = []
                    band_y = []
                    for bin_index in range(band_bin_edges.size - 1):
                        mask = band_bin_indices == bin_index
                        if not np.any(mask):
                            continue
                        x_values = band_target_values[mask]
                        rmse = np.sqrt(np.mean(squared_errors[band_mask][mask]))
                        normalization = np.mean(np.abs(x_values))
                        band_x.append(x_values.mean())
                        band_y.append(rmse / normalization if normalization > 0 else np.nan)
                        f.write(
                            f"tsr,{band_name},{band_x[-1]:.10e},{band_y[-1]:.10e},{0.0:.10e},{int(mask.sum())}\n"
                        )

                    if not band_x:
                        continue
                    color = band_colors[band_name]
                    ax.plot(
                        band_x,
                        band_y,
                        color=color,
                        linewidth=1.2,
                        marker="o",
                        markersize=2.5,
                        markerfacecolor=color,
                        markeredgewidth=0,
                        label=band_name,
                    )

                ax.set_xlabel("tsr target value (unpreprocessed)")
                ax.set_ylabel("Normalized RMSE")
                ax.set_title("Normalized RMSE vs tsr")
                ax.legend(frameon=False, fontsize=8)
                fig.tight_layout()
                fig.savefig(figures_path / "loss_vs_tsr.png")
                plt.close(fig)

            # ── copula plot ───────────────────────────────────────────────────
            feature_copula = empirical_copula_values(feature_values[scatter_indices])
            loss_copula = empirical_copula_values(losses[scatter_indices])

            with plt.rc_context(style):
                fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=300)

                hb = ax.hexbin(
                    feature_copula,
                    loss_copula,
                    gridsize=60,
                    cmap="inferno",
                    bins="log",
                    mincnt=1,
                    linewidths=0.2,
                )
                # diagonal = independence reference
                ax.plot(
                    [0, 1],
                    [0, 1],
                    color="white",
                    linewidth=0.7,
                    linestyle="--",
                    alpha=0.4,
                    label="independence",
                )

                ax.set_xlabel(f"Copula rank: {feature_name}")
                ax.set_ylabel("Copula rank: loss")
                ax.set_title(f"Loss-feature copula: {feature_name}")
                ax.set_aspect("equal")
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

                cb = fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.02)
                cb.set_label("log10(count)", fontsize=8)

                fig.tight_layout()
                fig.savefig(figures_path / f"empirical_copula_loss_vs_{feature_name}.png")
                plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Test the trained model on validation data."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to OmegaConf YAML config",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to a training checkpoint (.pt) or imported PyTorch weights (.pth)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to output directory (optional)",
    )
    args = parser.parse_args()

    # --- load config ---
    config = load_config(args.config_file)

    # --- load model checkpoint ---
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else Path(config.train.checkpoint_dir) / "best_model.pt"
    )
    checkpoint_data = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )

    checkpoint_epoch_label = checkpoint_path.stem
    if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
        model_weights = checkpoint_data["model_state_dict"]
        model_config = checkpoint_data.get("config", config)
        checkpoint_epoch_label = str(
            checkpoint_data.get(
                "best_epoch", checkpoint_data.get("epoch", checkpoint_path.stem)
            )
        )
    else:
        model_weights = checkpoint_data
        model_config = config

    model = SimpleModel(model_config)
    model.load_state_dict(model_weights)
    model.eval()

    # --- setup output directory ---
    output_dir = Path(config.train.checkpoint_dir) / "figures" / checkpoint_epoch_label
    if args.output_dir:
        output_dir = Path(args.output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- load test data ---
    raw_era5_paths = [
        f"{config.dataset.era5.raw_path}/{make_era5_filename(year)}"
        for year in range(1990, 2021)
    ]
    test_era5_paths = [
        f"{config.dataset.era5.path}/{make_era5_filename(year)}"
        for year in config.dataset.test_years
    ]
    kernel_paths = [
        Path(config.dataset.kernels.raw_path) / make_kernel_filename(year)
        for year in range(2011, 2016)
    ]
    
    raw_dataset = xr.open_mfdataset(raw_era5_paths, combine="nested", concat_dim="date")
    if config.preprocess.ecod:
        cloud_profiles = load_cloud_profiles(
            config.dataset.era5.raw_path,
            range(1990, 2021),
            range(1, 13),
            raw_dataset,
        )
        raw_dataset = raw_dataset.assign(
            ciwc=cloud_profiles["ciwc"],
            clwc=cloud_profiles["clwc"],
        )
    processed_dataset = xr.open_mfdataset(
        test_era5_paths, combine="nested", concat_dim="date"
    )
    kernels_dataset = xr.open_mfdataset(kernel_paths, combine="nested", concat_dim="date")

    # --- load preprocessor ---
    preprocessor = create_2024_preprocessor(
        input_vars=list(config.dataset.input_vars),
        target_var=config.dataset.target_var,
        ecod=config.preprocess.ecod,
    )
    preprocessor.load(config.preprocess.params_dir)

    # --- run tests ---
    global_tsr_test(
        ds=processed_dataset,
        preprocessor=preprocessor,
        model=model,
        config=config,
        figures_path=output_dir,
    )
    kernel_date_test(
        ds=raw_dataset.sel(date="2013-09"),
        preprocessor=preprocessor,
        model=model,
        config=config,
        figures_path=output_dir,
        true_kernel=kernels_dataset.sel(date="2013-09"),
        date="2013-09"
    )    
    kernel_date_test(
        ds=raw_dataset.sel(date="2015-09"),
        preprocessor=preprocessor,
        model=model,
        config=config,
        figures_path=output_dir,
        true_kernel=kernels_dataset.sel(date="2015-09"),
        date="2015-09"
    )
    kernel_ecod_fal_contour_test(
        processed_ds=processed_dataset,
        preprocessor=preprocessor,
        model=model,
        config=config,
        figures_path=output_dir,
        scatter=False,
        extrapolate=False,
        date="2015-09"
    )
    tsr_ecod_fal_contour_test(
        processed_ds=processed_dataset,
        preprocessor=preprocessor,
        model=model,
        config=config,
        figures_path=output_dir,
        scatter=False,
        extrapolate=False,
        date="2015-09"
    )
    second_order_test(
        ds=raw_dataset,
        preprocessor=preprocessor,
        model=model,
        config=config,
        true_kernel=kernels_dataset,
        dates=["2012-09", "2013-09"],
        figures_path=output_dir,
    )
    kernel_date_test(
        ds=raw_dataset.sel(date="2015-12"),
        preprocessor=preprocessor,
        model=model,
        config=config,
        figures_path=output_dir,
        true_kernel=kernels_dataset.sel(date="2015-12"),
        date="2015-12"
    )
    #test_4(
    #    raw_dataset=filter_by_years(raw_dataset, [2015]),
    #    preprocessor=preprocessor,
    #    model=model,
    #    figures_path=output_dir,
    #    batch_size=256,
    #)
    #test_5(
    #    raw_dataset=filter_by_years(raw_dataset, [2015]),
    #    preprocessor=preprocessor,
    #    model=model,
    #    figures_path=output_dir,
    #)


if __name__ == "__main__":
    main()
