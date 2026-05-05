import argparse
from omegaconf import OmegaConf
import torch
import numpy as np
import xarray as xr
from scipy.stats import linregress
from dataloader import make_era5_filename
from preprocessing import XarrayMinMaxScaler
from pathlib import Path
import copy

from model import SimpleModel
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap


def setup_map():
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
    array = xr.Dataset(
        data_vars={
            "tsr": (("date", "latitude", "longitude"), arr[:,:,:,0]),
        },
        coords={
            "date": date,
            "longitude": lon,
            "latitude": lat,
        },
    )
    return array


def test_1(
    dataset: xr.Dataset,
    scaler: XarrayMinMaxScaler,
    model: torch.nn.Module,
):
    data_torch = torch.from_numpy(
        scaler.transform(dataset.transpose("date", "latitude", "longitude")).to_array().to_numpy()
    ).permute(3,2,1,0).float()
    # print(data_torch.shape)

    model_outputs=model(data_torch[:, :, :, :10]).permute(2,1,0,3).detach().numpy()
    # print(model_outputs.shape)
    predictions = scaler.inverse_transform(
        dataset_from_array(
            arr=model_outputs,
            date=dataset["date"].values,
            lon=dataset["longitude"].values,
            lat=dataset["latitude"].values,
        )
    )

    tsr_mean = dataset["tsr"].mean(dim="date").to_numpy() / (3600 * 24)
    tsr_pred_mean = predictions.mean(dim="date").to_array().to_numpy()[0] / (3600 * 24)
    max_tsr = max(np.max(tsr_mean), np.max(tsr_pred_mean))
    print('tsr_mean', tsr_mean)
    print('tsr_pred_mean', tsr_mean)
    print('max_tsr', max_tsr)
    
    
    fig = plt.figure(figsize=(8, 6), dpi=300)
    m = setup_map()

    lon, lat = dataset["longitude"].values, dataset["latitude"].values
    plot_colormesh_on_map(m, lon, lat, tsr_mean, cmap="Spectral", vmin=0, vmax=max_tsr)

    plt.text(
        x=300,
        y=np.max(dataset["latitude"].values) + 5,
        s=f"{np.mean(tsr_mean):.2f}",
        fontsize=20,
    )
    plt.colorbar(orientation="horizontal", fraction=0.075, label="$W/m^2$")
    plt.title("TSR (ERA5)")

    plt.savefig("tsr_era5.png")

    fig = plt.figure(figsize=(8, 6), dpi=300)
    m = setup_map()
    plot_colormesh_on_map(
        m, lon, lat, tsr_pred_mean, cmap="Spectral", vmin=0, vmax=max_tsr
    )

    plt.text(
        x=300,
        y=np.max(dataset["latitude"].values) + 5,
        s=f"{np.mean(tsr_pred_mean):.2f}",
        fontsize=20,
    )
    plt.colorbar(orientation="horizontal", fraction=0.075, label="$W/m^2$")
    plt.title("TSR (NN)")

    plt.savefig("tsr_nn.png")

    diff = tsr_mean - tsr_pred_mean
    max_abs_diff = np.max(np.abs(diff))

    fig = plt.figure(figsize=(8, 6), dpi=300)
    m = setup_map()
    plot_colormesh_on_map(
        m, lon, lat, diff, cmap="RdBu_r", vmin=-max_abs_diff, vmax=max_abs_diff
    )

    plt.text(
        x=300,
        y=np.max(dataset["latitude"].values) + 5,
        s=f"{np.mean(diff):.2f}",
        fontsize=20,
    )
    plt.colorbar(orientation="horizontal", fraction=0.075, label="$W/m^2$")
    plt.title("MBE")
    plt.savefig("mbe.png")

    mse = np.mean(diff**2)
    rmse = np.sqrt(mse)
    max_rmse = np.max(rmse)

    fig = plt.figure(figsize=(8, 6), dpi=300)
    m = setup_map()
    plot_colormesh_on_map(m, lon, lat, np.abs(diff), cmap="Blues", vmin=0, vmax=max_rmse)

    plt.text(
        x=300,
        y=np.max(dataset["latitude"].values) + 5,
        s=f"{np.mean(rmse):.2f}",
        fontsize=20,
    )
    plt.colorbar(orientation="horizontal", fraction=0.075, label="$W/m^2$")
    plt.title("RMSE")
    plt.savefig("rmse.png")

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


def main():
    parser = argparse.ArgumentParser(
        description="Test the trained model on validation data."
    )
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to OmegaConf YAML config",
    )

    # --- load config ---
    config_path = parser.parse_args().config_path
    config = OmegaConf.load(config_path)

    # --- load model checkpoint ---
    checkpoint_path = Path(config.train.checkpoint_dir) / "best_model.pt"
    checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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
    model = SimpleModel(config)
    model.load_state_dict(model_weights)
    model.eval()

    # --- load val data ---
    era5_paths = [
        f"{config.dataset.era5.raw_path}/{make_era5_filename(year)}"
        for year in config.dataset.val_years
    ]
    print(f"Loading ERA5 data from: {era5_paths}")
    dataset = xr.open_mfdataset(era5_paths, combine="nested", concat_dim="date")

    # --- load val preprocessor ---
    path_prefix = "scaler_small_val"
    scaler = XarrayMinMaxScaler(dim=("date", "latitude", "longitude"))
    scaler.load(path_prefix)

    # --- run tests ---
    test_1(dataset, scaler, model)


if __name__ == "__main__":
    main()
