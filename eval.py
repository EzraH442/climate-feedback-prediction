import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from omegaconf import OmegaConf
from scipy.stats import linregress, rankdata

from dataloader import make_era5_filename
from preprocessing import (
    Preprocessor,
    SequentialPreprocessor,
    XarrayMinMaxScaler,
    XarrayStandardScaler,
    create_2024_preprocessor,
)

from model import SimpleModel

from mpl_toolkits.basemap import Basemap


def setup_global_map():
    m = Basemap(
        projection="cyl",
        resolution="l",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=0,
        urcrnrlon=360,
    )
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary()
    # Draw parallels (latitude lines) and meridians (longitude lines) with labels

    parallels = np.arange(-90.0, 91.0, 30.0)
    meridians = np.arange(-180.0, 181.0, 60.0)

    # Draw latitude lines with labels on left and right
    m.drawparallels(parallels, labels=[True, False, False, True])
    # Draw longitude lines with labels on top and bottom
    m.drawmeridians(meridians, labels=[True, False, False, True])
    return m


def setup_north_pole_map():
    m = Basemap(
        projection="npstere",
        boundinglat=60,
        lon_0=0,
        # round=True,
        resolution="l",
    )
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color="white")
    m.drawparallels(np.arange(60.0, 91.0, 30.0))
    m.drawmeridians(np.arange(0.0, 360.0, 60.0))
    return m


def plot_colormesh_on_map(m, lon, lat, data, cmap, vmin, vmax):
    lon_grid, lat_grid = np.meshgrid(np.asarray(lon), np.asarray(lat))
    x, y = m(lon_grid, lat_grid)
    m.pcolormesh(
        x,
        y,
        np.asarray(data),
        shading="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )


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


# fal        (date, latitude, longitude) float64 100MB 0.7555 0.7555 ... 0.85
# hcc        (date, latitude, longitude) float64 100MB 0.3654 0.3654 ... 0.085
# mcc        (date, latitude, longitude) float64 100MB 0.474 0.474 ... 0.1886
# lcc        (date, latitude, longitude) float64 100MB 0.8883 ... 0.1287
# sp         (date, latitude, longitude) float64 100MB 1.011e+05 ... 7.03e+04
# tciw       (date, latitude, longitude) float64 100MB 0.02062 ... 0.003697
# tclw       (date, latitude, longitude) float64 100MB 0.005785 ... 2.902e-05
# tco3       (date, latitude, longitude) float64 100MB 0.006788 ... 0.005639
# tcwv       (date, latitude, longitude) float64 100MB 2.86 2.86 ... 1.031
# totalx     (date, latitude, longitude) float64 100MB 33.78 33.78 ... 24.45
# tsr        (date, latitude, longitude) float64 100MB 0.0 0.0 ... 1.344e+07
def dataset_from_array(arr, date, lon, lat):
    """
    arr must have shape (date, lat, lon)
    """
    array = xr.Dataset(
        data_vars={"tsr": (("date", "latitude", "longitude"), arr)},
        coords={"date": date, "longitude": lon, "latitude": lat},
    )
    return array


def ordered_dataset(dataset: xr.Dataset, target_var: str = "tsr") -> xr.Dataset:
    all_vars = [v for v in dataset.data_vars if v != target_var] + [target_var]
    return dataset[all_vars]


def load_rrtm_kernel(reference_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = xr.open_dataset(reference_path)
    data_array = dataset["TOA"].sel(up_down_net=3, band=1)
    data_array = data_array.transpose("latitude", "longitude")
    return (
        data_array.to_numpy(),
        dataset["longitude"].values,
        dataset["latitude"].values,
    )


def filter_by_years(ds, years, time_coord="date"):
    return ds.sel({time_coord: ds[time_coord].dt.year.isin(years)})


def interpolate_spatial_field(
    data: np.ndarray,
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    dst_lon: np.ndarray,
    dst_lat: np.ndarray,
) -> np.ndarray:
    dataset = xr.Dataset(
        data_vars={"field": (("latitude", "longitude"), data)},
        coords={"longitude": src_lon, "latitude": src_lat},
    )
    return dataset.interp(
        longitude=dst_lon,
        latitude=dst_lat,
        method="linear",
    )["field"].to_numpy()


def load_raw_date_dataset(raw_root: str, date: str) -> xr.Dataset:
    year = int(date.split("-")[0])
    raw_path = Path(raw_root) / make_era5_filename(year)
    raw_dataset = xr.open_dataset(raw_path)
    return raw_dataset.sel(date=[date])


def preprocessed_feature_target_arrays(
    raw_dataset: xr.Dataset,
    preprocessor: Preprocessor,
    model: SimpleModel,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    preprocessed_dataset = preprocessor.transform(raw_dataset)
    ordered_preprocessed = ordered_dataset(preprocessed_dataset)
    feature_names = [v for v in ordered_preprocessed.data_vars if v != "tsr"]
    data = (
        ordered_preprocessed.to_dataarray()
        .transpose("date", "latitude", "longitude", "variable")
        .to_numpy()
    )

    inputs = data[..., : model.input_dim].reshape(-1, model.input_dim)
    targets = data[..., model.input_dim].reshape(-1)
    return inputs, targets, feature_names


def empirical_copula_values(values: np.ndarray) -> np.ndarray:
    ranks = rankdata(values, method="average")
    return (ranks - 0.5) / len(values)


def unpreprocess_feature_values(
    feature_values: np.ndarray,
    feature_name: str,
    preprocessor: Preprocessor,
) -> np.ndarray:
    if isinstance(preprocessor, SequentialPreprocessor):
        values = feature_values.copy()
        for child in reversed(preprocessor.preprocessors):
            if isinstance(child, XarrayMinMaxScaler):
                data_min = child.data_min_[feature_name].to_numpy().item()
                data_max = child.data_max_[feature_name].to_numpy().item()
                denom = data_max - data_min
                if denom == 0:
                    return np.full_like(values, fill_value=data_min, dtype=np.float64)
                values = (values - child.min_val) / (
                    child.max_val - child.min_val
                ) * denom + data_min
            elif isinstance(child, XarrayStandardScaler):
                mean = child.mean_[feature_name].to_numpy().item()
                std = child.std_[feature_name].to_numpy().item()
                values = values * std + mean
        return values

    return feature_values


def inverse_transform_without_upsampling(
    dataset: xr.Dataset,
    preprocessor: Preprocessor,
) -> xr.Dataset:
    if isinstance(preprocessor, SequentialPreprocessor):
        transformed = dataset
        for child in reversed(preprocessor.preprocessors):
            if child.__class__.__name__ == "Downscaler":
                continue
            transformed = child.inverse_transform(transformed)
        return transformed

    return preprocessor.inverse_transform(dataset)


def test_1(
    ds: xr.Dataset,
    preprocessor: Preprocessor,
    model: torch.nn.Module,
    figures_path: Path = Path("."),
):
    # preprocessors=[
    #    ECOD_Calculator(),
    #    Downscaler(factor=[("latitude", 4), ("longitude", 4)]),
    #    XarrayMinMaxScaler(dim=("date", "latitude", "longitude")),
    # ]
    assert isinstance(preprocessor, SequentialPreprocessor)

    date = ds["date"].values
    lat = ds["latitude"].values
    lon = ds["longitude"].values

    ds_ordered_np = (
        ordered_dataset(ds)
        .to_dataarray()
        .transpose("date", "latitude", "longitude", "variable")
        .to_numpy()
    )

    data_torch = torch.from_numpy(ds_ordered_np).float()

    # (date, lat, lon)
    model_outputs = model(data_torch[:, :, :, : model.input_dim]).detach()

    pred = (
        preprocessor.preprocessors[-1]
        .inverse_transform(
            dataset_from_array(arr=model_outputs.numpy(), date=date, lon=lon, lat=lat),
        )
        .to_dataarray()
        .squeeze(dim="variable", drop=True)
    )
    true = preprocessor.preprocessors[-1].inverse_transform(ds)["tsr"]

    # (date, lat, lon)
    tsr_true = true.to_numpy() / (3600 * 24)
    tsr_pred = pred.to_numpy() / (3600 * 24)

    diff_full = tsr_true - tsr_pred  # (date, lat, lon)

    tsr_mean = np.mean(tsr_true, axis=0)  # (lat, lon)
    tsr_pred_mean = np.mean(tsr_pred, axis=0)  # (lat, lon)
    mbe_map = np.mean(diff_full, axis=0)  # (lat, lon)
    rmse_map = np.sqrt(np.mean(diff_full**2, axis=0))  # (lat, lon)

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

    fig = plt.figure(figsize=(8, 6), dpi=300)
    m = setup_global_map()
    plot_colormesh_on_map(m, lon, lat, tsr_mean, cmap="Spectral", vmin=0, vmax=max_tsr)
    plt.text(x=300, y=np.max(lat) + 5, s=f"{global_tsr_mean:.2f}", fontsize=20)
    plt.colorbar(orientation="horizontal", fraction=0.075, label="$W/m^2$")
    plt.title("TSR (ERA5)")
    plt.savefig(figures_path / "tsr_era5.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6), dpi=300)
    m = setup_global_map()
    plot_colormesh_on_map(
        m, lon, lat, tsr_pred_mean, cmap="Spectral", vmin=0, vmax=max_tsr
    )
    plt.text(x=300, y=np.max(lat) + 5, s=f"{global_tsr_pred_mean:.2f}", fontsize=20)
    plt.colorbar(orientation="horizontal", fraction=0.075, label="$W/m^2$")
    plt.title("TSR (NN)")
    plt.savefig(figures_path / "tsr_nn.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6), dpi=300)
    m = setup_global_map()
    plot_colormesh_on_map(
        m, lon, lat, mbe_map, cmap="RdBu_r", vmin=-max_abs_mbe, vmax=max_abs_mbe
    )
    plt.text(x=300, y=np.max(lat) + 5, s=f"{global_mbe:.2f}", fontsize=20)
    plt.colorbar(orientation="horizontal", fraction=0.075, label="$W/m^2$")
    plt.title("MBE")
    plt.savefig(figures_path / "mbe.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6), dpi=300)
    m = setup_global_map()
    plot_colormesh_on_map(m, lon, lat, rmse_map, cmap="Blues", vmin=0, vmax=max_rmse)

    plt.text(x=300, y=np.max(lat) + 5, s=f"{global_rmse:.2f}", fontsize=20)
    plt.colorbar(orientation="horizontal", fraction=0.075, label="$W/m^2$")
    plt.title("RMSE")
    plt.savefig(figures_path / "rmse.png")
    plt.close(fig)

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


def test_2(
    raw_dataset: xr.Dataset,
    preprocessor: Preprocessor,
    date,
    model: SimpleModel,
    figures_path: Path = Path("."),
    max_true_kernel=0,
    true_kernel: np.ndarray | None = None,
    true_lon: np.ndarray | None = None,
    true_lat: np.ndarray | None = None,
):
    raw_date_specific_data = raw_dataset.sel(date=[date])
    raw_date_specific_data_perturbed = raw_date_specific_data.copy(deep=True)
    raw_date_specific_data_perturbed["fal"] = raw_date_specific_data["fal"] + 0.01
    # raw_date_specific_data_perturbed["fal"] = xr.where(
    #     raw_date_specific_data["fal"] + 0.01 > 1.0,
    #     1.0,
    #     raw_date_specific_data["fal"] + 0.01,
    # )

    processed_date_specific_data = preprocessor.transform(raw_date_specific_data)
    processed_date_specific_data_perturbed = preprocessor.transform(
        raw_date_specific_data_perturbed
    )

    lon = processed_date_specific_data["longitude"].values
    lat = processed_date_specific_data["latitude"].values
    raw_lon = raw_dataset["longitude"].values
    raw_lat = raw_dataset["latitude"].values

    ordered_base = ordered_dataset(processed_date_specific_data)
    ordered_perturbed = ordered_dataset(processed_date_specific_data_perturbed)
    data_torch_base = torch.from_numpy(
        ordered_base.to_dataarray()
        .transpose("date", "latitude", "longitude", "variable")
        .to_numpy()
    ).float()
    data_torch_perturbed = torch.from_numpy(
        ordered_perturbed.to_dataarray()
        .transpose("date", "latitude", "longitude", "variable")
        .to_numpy()
    ).float()

    model_output_base = model(data_torch_base[... ,: model.input_dim]).detach()
    model_output_perturbed = model(data_torch_perturbed[... ,: model.input_dim]).detach()

    predictions_base = preprocessor.inverse_transform(
        dataset_from_array(
            arr=model_output_base.numpy(),
            date=[date],
            lon=lon,
            lat=lat,
        )
    ).squeeze(dim='date').to_dataarray().to_numpy()[0] / (3600 * 24)
    predictions_perturbed = preprocessor.inverse_transform(
        dataset_from_array(
            arr=model_output_perturbed.numpy(),
            date=[date],
            lon=lon,
            lat=lat,
        )
    ).squeeze(dim='date').to_dataarray().to_numpy()[0] / (3600 * 24)

    diff = predictions_perturbed - predictions_base
    max_diff = max(max_true_kernel, np.max(np.abs(diff)))

    fig = plt.figure(figsize=(8, 6), dpi=300)
    m = setup_global_map()
    plot_colormesh_on_map(
        m, raw_lon, raw_lat, diff, cmap="RdBu_r", vmin=-max_diff, vmax=max_diff
    )

    plt.text(x=300, y=np.max(lat) + 5, s=f"{np.mean(diff):.2f}", fontsize=20)
    plt.colorbar(orientation="horizontal", fraction=0.075, label=r"$W/m^2 1\%$")
    plt.title(f"NN Surface Albedo Kernel\n{date}")
    plt.savefig(figures_path / f"nn_kernel_{date}.png")

    plt.close(fig)

    if true_kernel is not None and true_lon is not None and true_lat is not None:
        model_kernel_for_diff = diff
        plot_lon = raw_lon
        plot_lat = raw_lat

        if model_kernel_for_diff.shape != true_kernel.shape:
            model_kernel_for_diff = interpolate_spatial_field(
                model_kernel_for_diff, raw_lon, raw_lat, true_lon, true_lat
            )
            plot_lon = true_lon
            plot_lat = true_lat

        north_mask = plot_lat >= 60
        kernel_diff_north_pole = (true_kernel - model_kernel_for_diff)[north_mask, :]
        plot_lat_north = plot_lat[north_mask]
        max_abs_diff = np.max(np.abs(kernel_diff_north_pole))

        fig = plt.figure(figsize=(8, 8), dpi=300)
        m = setup_north_pole_map()
        plot_colormesh_on_map(
            m,
            plot_lon,
            plot_lat_north,
            kernel_diff_north_pole,
            cmap="RdBu_r",
            vmin=-max_abs_diff,
            vmax=max_abs_diff,
        )
        plt.title(rf"$K_{{ERA5}}({date}) - K_{{NN}}({date})$")
        plt.colorbar(
            orientation="horizontal", fraction=0.05, pad=0.07, label=r"$W/m^2 1\%$"
        )
        plt.savefig(figures_path / f"k_era5_minus_k_nn_north_pole_{date}.png")
        plt.close(fig)

    return diff, raw_lon, raw_lat


def test_2013_09_against_rrtm(
    raw_root: str,
    preprocessor: Preprocessor,
    model: SimpleModel,
    figures_path: Path = Path("."),
    reference_path: Path = Path("old_data/RRTM_2013_cld_alb_TOA_SFC_09.nc"),
):
    if not reference_path.exists():
        print(f"Skipping 2013-09 RRTM comparison; missing file: {reference_path}")
        return

    rrtm_kernel, rrtm_lon, rrtm_lat = load_rrtm_kernel(reference_path)
    rrtm_kernel = rrtm_kernel * 0.01

    model_kernel, raw_lon, raw_lat = test_2(
        raw_dataset=load_raw_date_dataset(raw_root, "2013-09"),
        preprocessor=preprocessor,
        date="2013-09",
        model=model,
        figures_path=figures_path,
        max_true_kernel=np.max(np.abs(rrtm_kernel)),  # for same colorbar
        true_kernel=rrtm_kernel,
        true_lon=rrtm_lon,
        true_lat=rrtm_lat,
    )

    if model_kernel.shape != rrtm_kernel.shape:
        model_kernel = interpolate_spatial_field(
            model_kernel,
            raw_lon,
            raw_lat,
            rrtm_lon,
            rrtm_lat,
        )
        raw_lon = rrtm_lon
        raw_lat = rrtm_lat

    comparison_diff = model_kernel - rrtm_kernel
    max_abs_kernel = max(np.max(np.abs(model_kernel)), np.max(np.abs(rrtm_kernel)))
    max_abs_comparison = np.max(np.abs(comparison_diff))
    max_rrtm_lat = np.max(rrtm_lat)

    print(model_kernel)
    print(rrtm_kernel)
    print(np.max(np.abs(model_kernel)), np.max(np.abs(rrtm_kernel)))
    print(max_abs_kernel)
    print(max_abs_comparison)

    fig = plt.figure(figsize=(8, 6), dpi=300)
    m = setup_global_map()
    plot_colormesh_on_map(
        m,
        rrtm_lon,
        rrtm_lat,
        rrtm_kernel,
        cmap="RdBu_r",
        vmin=-max_abs_kernel,
        vmax=max_abs_kernel,
    )
    plt.text(x=300, y=max_rrtm_lat + 5, s=f"{np.mean(rrtm_kernel):.2f}", fontsize=20)
    plt.colorbar(orientation="horizontal", fraction=0.075, label=r"$W/m^2 1\%$")
    plt.title("RRTM Surface Albedo Kernel\n2013-09")
    plt.savefig(figures_path / "rrtm_kernel_2013-09.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6), dpi=300)
    m = setup_global_map()
    plot_colormesh_on_map(
        m,
        raw_lon,
        raw_lat,
        comparison_diff,
        cmap="RdBu_r",
        vmin=-max_abs_comparison,
        vmax=max_abs_comparison,
    )
    plt.text(
        x=300, y=max_rrtm_lat + 5, s=f"{np.mean(comparison_diff):.2f}", fontsize=20
    )
    plt.colorbar(orientation="horizontal", fraction=0.075, label=r"$W/m^2 1\%$")
    plt.title("NN - RRTM Surface Albedo Kernel\n2013-09")
    plt.savefig(figures_path / "nn_vs_rrtm_kernel_2013-09.png")
    plt.close(fig)


def test_3(
    raw_root: str,
    kernels_root: Path,
    preprocessor: Preprocessor,
    model: SimpleModel,
    figures_path: Path = Path("."),
):
    nn_kernel_2013, nn_lon_2013, nn_lat_2013 = test_2(
        raw_dataset=load_raw_date_dataset(raw_root, "2013-09"),
        preprocessor=preprocessor,
        date="2013-09",
        model=model,
        figures_path=figures_path,
    )
    nn_kernel_2012, _, _ = test_2(
        raw_dataset=load_raw_date_dataset(raw_root, "2012-09"),
        preprocessor=preprocessor,
        date="2012-09",
        model=model,
        figures_path=figures_path,
    )

    delta_k_nn = nn_kernel_2013 - nn_kernel_2012
    kernel_2013_diff = None

    kernels_root_path = Path(kernels_root)
    era5_kernel_2013_path = kernels_root_path / "RRTM_kernel_2013_cld_alb_TOA_SFC_09.nc"
    era5_kernel_2012_path = kernels_root_path / "RRTM_kernel_2012_cld_alb_TOA_SFC_09.nc"
    if not era5_kernel_2013_path.exists() or not era5_kernel_2012_path.exists():
        missing = []
        if not era5_kernel_2013_path.exists():
            missing.append(str(era5_kernel_2013_path))
        if not era5_kernel_2012_path.exists():
            missing.append(str(era5_kernel_2012_path))
        raise FileNotFoundError("Missing ERA5 kernel file(s): " + ", ".join(missing))

    era5_kernel_2013, era5_lon_2013, era5_lat_2013 = load_rrtm_kernel(
        era5_kernel_2013_path
    )
    era5_kernel_2012, _, _ = load_rrtm_kernel(era5_kernel_2012_path)
    era5_kernel_2013 = era5_kernel_2013 * 0.01
    era5_kernel_2012 = era5_kernel_2012 * 0.01

    delta_k_era5 = era5_kernel_2013 - era5_kernel_2012

    if delta_k_nn.shape != delta_k_era5.shape:
        nn_kernel_2013 = interpolate_spatial_field(
            nn_kernel_2013,
            nn_lon_2013,
            nn_lat_2013,
            era5_lon_2013,
            era5_lat_2013,
        )
        delta_k_nn = interpolate_spatial_field(
            delta_k_nn,
            nn_lon_2013,
            nn_lat_2013,
            era5_lon_2013,
            era5_lat_2013,
        )
        plot_lon = era5_lon_2013
        plot_lat = era5_lat_2013
    else:
        plot_lon = nn_lon_2013
        plot_lat = nn_lat_2013

    delta_k_diff = delta_k_nn - delta_k_era5
    kernel_2013_diff = era5_kernel_2013 - nn_kernel_2013

    north_mask = plot_lat >= 60
    kernel_2013_diff = kernel_2013_diff[north_mask, :]
    delta_k_nn = delta_k_nn[north_mask, :]
    delta_k_era5 = delta_k_era5[north_mask, :]
    delta_k_diff = delta_k_diff[north_mask, :]

    max_abs_kernel_2013_diff = np.max(np.abs(kernel_2013_diff))
    max_kernel = max(np.max(np.abs(delta_k_nn)), np.max(np.abs(delta_k_era5)))
    max_abs_diff = np.max(np.abs(delta_k_diff))
    plot_lat = plot_lat[north_mask]

    fig = plt.figure(figsize=(8, 8), dpi=300)
    m = setup_north_pole_map()
    plot_colormesh_on_map(
        m,
        plot_lon,
        plot_lat,
        kernel_2013_diff,
        cmap="RdBu_r",
        vmin=-max_abs_kernel_2013_diff,
        vmax=max_abs_kernel_2013_diff,
    )
    plt.title(r"$K_{ERA5}(2013\mathrm{-}09) - K_{NN}(2013\mathrm{-}09)$")
    plt.colorbar(
        orientation="horizontal", fraction=0.05, pad=0.07, label=r"$W/m^2 1\%$"
    )
    plt.savefig(figures_path / "k_era5_minus_k_nn_north_pole_2013-09.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 8), dpi=300)
    m = setup_north_pole_map()
    plot_colormesh_on_map(
        m,
        plot_lon,
        plot_lat,
        delta_k_nn,
        cmap="RdBu_r",
        vmin=-max_kernel,
        vmax=max_kernel,
    )
    plt.title("NN surface albedo kernel difference" + "\n(2013-09 minus 2012-09)")
    plt.colorbar(
        orientation="horizontal", fraction=0.05, pad=0.07, label=r"$W/m^2 1\%$"
    )
    plt.savefig(figures_path / "delta_k_nn_np.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 8), dpi=300)
    m = setup_north_pole_map()
    plot_colormesh_on_map(
        m,
        plot_lon,
        plot_lat,
        delta_k_era5,
        cmap="RdBu_r",
        vmin=-max_kernel,
        vmax=max_kernel,
    )
    plt.title(r"ERA5 surface albedo kernel difference" + "\n(2013-09 minus 2012-09)")
    plt.colorbar(
        orientation="horizontal", fraction=0.05, pad=0.07, label=r"$W/m^2 1\%$"
    )
    plt.savefig(figures_path / "delta_k_era5_np.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 8), dpi=300)
    m = setup_north_pole_map()
    plot_colormesh_on_map(
        m,
        plot_lon,
        plot_lat,
        -delta_k_diff,
        cmap="RdBu_r",
        vmin=-max_abs_diff,
        vmax=max_abs_diff,
    )
    plt.title(r"$K_{ERA5} - K_{NN}$" + "\n2013-09 minus 2012-09")
    plt.colorbar(
        orientation="horizontal", fraction=0.05, pad=0.07, label=r"$W/m^2 1\%$"
    )
    plt.savefig(
        figures_path / "delta_k_nn_minus_era5_north_pole_2013-09_minus_2012-09.png"
    )
    plt.close(fig)


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
    inputs, targets, feature_names = preprocessed_feature_target_arrays(
        raw_dataset=raw_dataset,
        preprocessor=preprocessor,
        model=model,
    )

    with torch.no_grad():
        predictions = model(torch.from_numpy(inputs).float()).cpu().numpy()
    losses = (predictions - targets) ** 2

    rng = np.random.default_rng(0)
    scatter_indices = np.arange(len(losses))
    if len(scatter_indices) > max_scatter_points:
        scatter_indices = rng.choice(
            scatter_indices, size=max_scatter_points, replace=False
        )

    # ── shared style ──────────────────────────────────────────────────────────
    ACCENT = "#4FC3F7"  # sky blue line
    BAND = "#4FC3F7"
    BG = "#0F1117"
    GRID = "#1E2130"
    TEXT = "#CDD6F4"
    style = {
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "axes.titlepad": 10,
        "font.family": "monospace",
        "text.color": TEXT,
    }
    # ─────────────────────────────────────────────────────────────────────────

    summary_csv = figures_path / "loss_landscape_summary.csv"
    with summary_csv.open("w", encoding="ascii") as f:
        f.write("feature,feature_mean,mean_loss,loss_std,sample_count\n")

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

            bin_indices = np.digitize(feature_values, bin_edges[1:-1], right=False)
            bin_centers = []
            bin_losses = []
            bin_loss_stds = []
            bin_counts = []

            for bin_index in range(bin_edges.size - 1):
                mask = bin_indices == bin_index
                if not np.any(mask):
                    continue
                bin_centers.append(feature_values_unprocessed[mask].mean())
                bin_losses.append(losses[mask].mean())
                bin_loss_stds.append(losses[mask].std())
                bin_counts.append(mask.sum())
                f.write(
                    f"{feature_name},{bin_centers[-1]:.10e},{bin_losses[-1]:.10e},{bin_loss_stds[-1]:.10e},{bin_counts[-1]}\n"
                )

            # ── loss landscape plot ───────────────────────────────────────────
            with plt.rc_context(style):
                fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

                bin_centers = np.asarray(bin_centers)
                bin_losses = np.asarray(bin_losses)
                bin_loss_stds = np.asarray(bin_loss_stds)
                lower_band = np.clip(bin_losses - bin_loss_stds, a_min=0.0, a_max=None)
                upper_band = bin_losses + bin_loss_stds

                ax.fill_between(
                    bin_centers,
                    lower_band,
                    upper_band,
                    color=BAND,
                    alpha=0.12,
                    linewidth=0,
                )
                ax.plot(
                    bin_centers,
                    bin_losses,
                    color=ACCENT,
                    linewidth=1.2,
                    marker="o",
                    markersize=2.5,
                    markerfacecolor=ACCENT,
                    markeredgewidth=0,
                )
                ax.set_xlabel(f"{feature_name}  (unpreprocessed)")
                ax.set_ylabel("Mean squared error")
                ax.set_title(f"Loss landscape — {feature_name}")
                ax.spines[["top", "right", "left"]].set_visible(False)
                ax.tick_params(length=0)
                fig.tight_layout()
                fig.savefig(
                    figures_path / f"loss_vs_{feature_name}.png",
                    facecolor=BG,
                )
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
                ax.spines[["top", "right"]].set_visible(False)
                ax.tick_params(length=0)

                cb = fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.02)
                cb.set_label("log₁₀(count)", fontsize=8)
                cb.ax.yaxis.set_tick_params(color=TEXT, labelsize=7)

                fig.tight_layout()
                fig.savefig(
                    figures_path / f"empirical_copula_loss_vs_{feature_name}.png",
                    facecolor=BG,
                )
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
    config = OmegaConf.load(args.config_file)

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
    raw_dataset = xr.open_mfdataset(raw_era5_paths, combine="nested", concat_dim="date")
    processed_dataset = xr.open_mfdataset(
        test_era5_paths, combine="nested", concat_dim="date"
    )

    # --- load preprocessor ---
    preprocessor = create_2024_preprocessor()
    preprocessor.load(config.preprocess.params_dir)

    # --- run tests ---
    test_1(
        ds=processed_dataset,
        preprocessor=preprocessor,
        model=model,
        figures_path=output_dir,
    )
    test_2(
        raw_dataset=load_raw_date_dataset(config.dataset.era5.raw_path, "2005-09"),
        date="2005-09",
        preprocessor=preprocessor,
        model=model,
        figures_path=output_dir,
    )
    test_2013_09_against_rrtm(
        raw_root=config.dataset.era5.raw_path,
        preprocessor=preprocessor,
        model=model,
        figures_path=output_dir,
        reference_path=Path("data/other/RRTM_kernel_2013_cld_alb_TOA_SFC_09.nc"),
    )
    test_3(
        raw_root=config.dataset.era5.raw_path,
        kernels_root=Path("data/other"),
        preprocessor=preprocessor,
        model=model,
        figures_path=output_dir,
    )
    test_4(
        raw_dataset=filter_by_years(raw_dataset, [2015]),
        preprocessor=preprocessor,
        model=model,
        figures_path=output_dir,
        batch_size=256,
    )
    test_5(
        raw_dataset=filter_by_years(raw_dataset, [2015]),
        preprocessor=preprocessor,
        model=model,
        figures_path=output_dir,
    )


if __name__ == "__main__":
    main()
