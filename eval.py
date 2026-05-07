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


def plot_colormesh_on_map(m, lon, lat, data, cmap, vmin, vmax):
    m.pcolormesh(
        np.asarray(lon),
        np.asarray(lat),
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


def test_1(
    preprocessed_dataset: xr.Dataset,
    raw_dataset: xr.Dataset,
    preprocessor: Preprocessor,
    model: torch.nn.Module,
    figures_path: Path = Path("."),
):
    date = raw_dataset["date"].values
    raw_lat = raw_dataset["latitude"].values
    raw_lon = raw_dataset["longitude"].values
    processed_lat = preprocessed_dataset["latitude"].values
    processed_lon = preprocessed_dataset["longitude"].values

    data_torch = torch.from_numpy(
        preprocessed_dataset.to_dataarray()
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
    preprocessed_dataset: xr.Dataset,
    preprocessor: Preprocessor,
    date,
    model: SimpleModel,
    figures_path: Path = Path("."),
):
    date_specific_data = preprocessed_dataset.sel(date=date)
    date_specific_data_perturbed = date_specific_data.copy(deep=True)
    date_specific_data_perturbed["fal"] = date_specific_data["fal"] + 0.01
    # todo clamping

    lon = date_specific_data["longitude"].values
    lat = date_specific_data["latitude"].values
    raw_lon = raw_dataset['longitude'].values
    raw_lat = raw_dataset['latitude'].values

    data_torch_base = torch.from_numpy(
        date_specific_data.to_dataarray()
        .transpose("date", "latitude", "longitude", "variable")
        .to_numpy()
    ).float()
    data_torch_perturbed = torch.from_numpy(
        date_specific_data_perturbed.to_dataarray()
        .transpose("date", "latitude", "longitude", "variable")
        .to_numpy()
    ).float()

    model_output_base = model(data_torch_base[:, :, :, : model.input_dim]).detach()
    model_output_perturbed = model(
        data_torch_perturbed[:, :, :, : model.input_dim]
    ).detach()
    print(model_output_base)

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
    print(predictions_base)

    diff = predictions_perturbed - predictions_base
    max_diff = np.max(np.abs(diff))
    print(max_diff)

    fig = plt.figure(figsize=(8, 6), dpi=300)
    m = setup_global_map()
    plot_colormesh_on_map(
        m, raw_lon, raw_lat, diff, cmap="RdBu_r", vmin=-max_diff, vmax=max_diff
    )

    plt.text(x=300, y=np.max(lat) + 5, s=f"{np.mean(diff):.2f}", fontsize=20)
    plt.colorbar(orientation="horizontal", fraction=0.075, label=r"$W/m^2 1\%$")
    plt.title(f"NN Surface Albedo Kernel\n{date}")
    plt.savefig(figures_path / f"nn_kernel_{date}.png")

    pass


def test_3(
    dataset: xr.Dataset,
    date,
    preprocessor: Preprocessor,
    model: SimpleModel,
    figures_path: Path = Path("."),
):
    # Second-Order Test: Radiative Sensitivity Difference
    # \Delta K^* =
    pass


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

    # --- load val data ---
    raw_era5_paths = [
        f"{config.dataset.era5.raw_path}/{make_era5_filename(year)}"
        for year in config.dataset.val_years
    ]
    processed_era5_paths = [
        f"{config.dataset.era5.path}/{make_era5_filename(year)}"
        for year in config.dataset.val_years
    ]
    print(f"Loading ERA5 data from: {raw_era5_paths} and {processed_era5_paths}")
    raw_dataset = xr.open_mfdataset(raw_era5_paths, combine="nested", concat_dim="date")
    preprocessed_dataset = xr.open_mfdataset(
        processed_era5_paths, combine="nested", concat_dim="date"
    )

    # --- load val preprocessor ---
    preprocessor = create_2024_preprocessor()
    preprocessor.load(config.preprocess.params_dir)

    # --- run tests ---
    test_1(
        preprocessed_dataset=preprocessed_dataset,
        raw_dataset=raw_dataset,
        preprocessor=preprocessor,
        model=model,
        figures_path=output_path,
    )
    test_2(
        raw_dataset=raw_dataset,
        preprocessed_dataset=preprocessed_dataset,
        date="2005-09",
        preprocessor=preprocessor,
        model=model,
        figures_path=output_path,
    )


if __name__ == "__main__":
    main()
