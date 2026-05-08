import argparse
from omegaconf import OmegaConf
import torch
import numpy as np
import xarray as xr
from scipy.stats import linregress
from dataloader import make_era5_filename
from preprocessing import Preprocessor, create_2024_preprocessor
from pathlib import Path

from model import SimpleModel
import matplotlib.pyplot as plt

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


def test_1(
    raw_dataset: xr.Dataset,
    preprocessor: Preprocessor,
    model: torch.nn.Module,
    figures_path: Path = Path("."),
):
    date = raw_dataset["date"].values
    raw_lat = raw_dataset["latitude"].values
    raw_lon = raw_dataset["longitude"].values

    preprocessed_dataset = preprocessor.transform(raw_dataset)
    processed_lat = preprocessed_dataset["latitude"].values
    processed_lon = preprocessed_dataset["longitude"].values

    ordered_preprocessed = ordered_dataset(preprocessed_dataset)
    data_torch = torch.from_numpy(
        ordered_preprocessed.to_dataarray()
        .transpose("date", "latitude", "longitude", "variable")
        .to_numpy()
    ).float()
    print(data_torch.shape)

    model_outputs = model(data_torch[:, :, :, : model.input_dim]).detach()
    # (date, lat, lon)

    print(model_outputs.shape)
    predictions = preprocessor.inverse_transform(
        dataset_from_array(
            arr=model_outputs.numpy(), date=date, lon=processed_lon, lat=processed_lat
        )
    )

    tsr_raw = raw_dataset["tsr"].to_numpy() / (3600 * 24)  # (date, lat, lon)
    tsr_pred = predictions.to_dataarray().to_numpy()[0] / (
        3600 * 24
    )  # (date, lat, lon)

    diff_full = tsr_raw - tsr_pred  # (date, lat, lon)

    tsr_mean = np.mean(tsr_raw, axis=0)  # (lat, lon)
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

    print(f"Global MBE:  {global_mbe:.4f} W/m²")
    print(f"Global RMSE: {global_rmse:.4f} W/m²")

    fig = plt.figure(figsize=(8, 6), dpi=300)
    m = setup_global_map()
    plot_colormesh_on_map(
        m, raw_lon, raw_lat, tsr_mean, cmap="Spectral", vmin=0, vmax=max_tsr
    )
    plt.text(x=300, y=np.max(raw_lat) + 5, s=f"{global_tsr_mean:.2f}", fontsize=20)
    plt.colorbar(orientation="horizontal", fraction=0.075, label="$W/m^2$")
    plt.title("TSR (ERA5)")
    plt.savefig(figures_path / "tsr_era5.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6), dpi=300)
    m = setup_global_map()
    plot_colormesh_on_map(
        m, raw_lon, raw_lat, tsr_pred_mean, cmap="Spectral", vmin=0, vmax=max_tsr
    )
    plt.text(x=300, y=np.max(raw_lat) + 5, s=f"{global_tsr_pred_mean:.2f}", fontsize=20)
    plt.colorbar(orientation="horizontal", fraction=0.075, label="$W/m^2$")
    plt.title("TSR (NN)")
    plt.savefig(figures_path / "tsr_nn.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6), dpi=300)
    m = setup_global_map()
    plot_colormesh_on_map(
        m, raw_lon, raw_lat, mbe_map, cmap="RdBu_r", vmin=-max_abs_mbe, vmax=max_abs_mbe
    )
    plt.text(x=300, y=np.max(raw_lat) + 5, s=f"{global_mbe:.2f}", fontsize=20)
    plt.colorbar(orientation="horizontal", fraction=0.075, label="$W/m^2$")
    plt.title("MBE")
    plt.savefig(figures_path / "mbe.png")
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6), dpi=300)
    m = setup_global_map()
    plot_colormesh_on_map(
        m, raw_lon, raw_lat, rmse_map, cmap="Blues", vmin=0, vmax=max_rmse
    )

    plt.text(x=300, y=np.max(raw_lat) + 5, s=f"{global_rmse:.2f}", fontsize=20)
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

    model_output_base = model(data_torch_base[:, :, :, : model.input_dim]).detach()
    model_output_perturbed = model(
        data_torch_perturbed[:, :, :, : model.input_dim]
    ).detach()

    predictions_base = preprocessor.inverse_transform(
        dataset_from_array(
            arr=model_output_base.numpy(),
            date=[date],
            lon=lon,
            lat=lat,
        )
    ).mean(dim="date").to_dataarray().to_numpy()[0] / (3600 * 24)
    predictions_perturbed = preprocessor.inverse_transform(
        dataset_from_array(
            arr=model_output_perturbed.numpy(),
            date=[date],
            lon=lon,
            lat=lat,
        )
    ).mean(dim="date").to_dataarray().to_numpy()[0] / (3600 * 24)

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

    north_mask = plot_lat >= 60
    delta_k_nn = delta_k_nn[north_mask, :]
    delta_k_era5 = delta_k_era5[north_mask, :]
    delta_k_diff = delta_k_diff[north_mask, :]

    max_kernel = max(np.max(np.abs(delta_k_nn)), np.max(np.abs(delta_k_era5)))
    max_abs_diff = np.max(np.abs(delta_k_diff))
    plot_lat = plot_lat[north_mask]

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
    plt.title(r"NN surface albedo kernel difference" + "(2013-09 minus 2012-09)")
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
    plt.title(r"ERA5 surface albedo kernel difference" + "(2013-09 minus 2012-09)")
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
        delta_k_diff,
        cmap="RdBu_r",
        vmin=-max_abs_diff,
        vmax=max_abs_diff,
    )
    plt.title(r"$\Delta K_{NN} - \Delta K_{ERA5}$" + "\n2013-09 minus 2012-09")
    plt.colorbar(
        orientation="horizontal", fraction=0.05, pad=0.07, label=r"$W/m^2 1\%$"
    )
    plt.savefig(
        figures_path / "delta_k_nn_minus_era5_north_pole_2013-09_minus_2012-09.png"
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
        "--output_dir",
        type=str,
        help="Path to output directory (optional)",
    )
    args = parser.parse_args()

    # --- load config ---
    config = OmegaConf.load(args.config_file)

    # --- setup output directory ---
    output_dir = Path(config.train.checkpoint_dir) / "figures"
    if args.output_dir:
        output_dir = args.output_dir

    output_path = Path(output_dir)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # --- load model checkpoint ---
    checkpoint_path = Path(config.train.checkpoint_dir) / "best_model.pt"
    checkpoint_data = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    # checkpoint_data = {
    #     "epoch": epoch,
    #     "best_epoch": self.best_epoch,
    #     "best_val_loss": self.best_val_loss,
    #     "total_training_time": self.total_training_time,
    #     "model_state_dict": self.model.state_dict(),
    #     "optimizer_state_dict": self.optimizer.state_dict(),
    #     "config": self.config,
    # }
    model_weights = checkpoint_data["model_state_dict"]
    model = SimpleModel(checkpoint_data["config"])
    model.load_state_dict(model_weights)
    model.eval()

    # --- load test data ---
    test_years = getattr(config.dataset, "test_years", config.dataset.val_years)
    raw_era5_paths = [
        f"{config.dataset.era5.raw_path}/{make_era5_filename(year)}"
        for year in test_years
    ]
    processed_era5_paths = [
        f"{config.dataset.era5.path}/{make_era5_filename(year)}" for year in test_years
    ]
    print(f"Loading ERA5 data from: {raw_era5_paths} and {processed_era5_paths}")
    raw_dataset = xr.open_mfdataset(raw_era5_paths, combine="nested", concat_dim="date")
    preprocessed_dataset = xr.open_mfdataset(
        processed_era5_paths, combine="nested", concat_dim="date"
    )

    # --- load preprocessor ---
    preprocessor = create_2024_preprocessor()
    preprocessor.load(config.preprocess.params_dir)

    # --- run tests ---
    test_1(
        raw_dataset=filter_by_years(raw_dataset, list(range(1991, 2021, 2))),
        preprocessor=preprocessor,
        model=model,
        figures_path=output_path,
    )
    test_2(
        raw_dataset=load_raw_date_dataset(config.dataset.era5.raw_path, "2005-09"),
        date="2005-09",
        preprocessor=preprocessor,
        model=model,
        figures_path=output_path,
    )
    test_2013_09_against_rrtm(
        raw_root=config.dataset.era5.raw_path,
        preprocessor=preprocessor,
        model=model,
        figures_path=output_path,
        reference_path=Path("data/other/RRTM_kernel_2013_cld_alb_TOA_SFC_09.nc"),
    )
    test_3(
        raw_root=config.dataset.era5.raw_path,
        kernels_root=Path("data/other"),
        preprocessor=preprocessor,
        model=model,
        figures_path=output_path,
    )


if __name__ == "__main__":
    main()
