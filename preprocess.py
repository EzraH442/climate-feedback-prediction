import os
from utils import compute_cloud_optical_depth
import torch
from omegaconf import OmegaConf
from preprocessing import XarrayMinMaxScaler
import xarray as xr
import argparse


def make_era5_filename(year):
    return f"era5_single_levels_monthly_{year}.nc"


def transform_dataset(ds: xr.Dataset):
    new_ds = ds.copy()

    new_ds["ecod"] = compute_cloud_optical_depth(
        tclw=ds["tclw"], tciw=ds["tciw"], tcc=ds["tcc"]
    )
    new_ds = new_ds.drop_vars(["tcc"])
    new_ds["ecod_fal"] = new_ds["ecod"] * ds["fal"]

    # 4x downscaling in both lat and lon
    ds_downscaled = new_ds.coarsen(latitude=4, longitude=4, boundary="trim").mean()

    return ds_downscaled


def preprocess(config_path):
    conf = OmegaConf.load(config_path)
    torch.manual_seed(conf.seed)

    train_paths = [
        f"{conf.dataset.era5.raw_path}/{make_era5_filename(year)}"
        for year in conf.dataset.train_years
    ]

    val_paths = [
        f"{conf.dataset.era5.raw_path}/{make_era5_filename(year)}"
        for year in conf.dataset.val_years
    ]

    print(f"Loading ERA5 train data from: {train_paths}")
    train_data = xr.open_mfdataset(train_paths, combine="nested", concat_dim="date")

    print(f"Loading ERA5 val data from: {val_paths}")
    val_data = xr.open_mfdataset(val_paths, combine="nested", concat_dim="date")
    """
       tcc        (date, latitude, longitude) float64 100MB 
       fal        (date, latitude, longitude) float64 100MB 0.7555 0.7555 ... 0.85
       hcc        (date, latitude, longitude) float64 100MB 0.3654 0.3654 ... 0.085
       mcc        (date, latitude, longitude) float64 100MB 0.474 0.474 ... 0.1886
       lcc        (date, latitude, longitude) float64 100MB 0.8883 ... 0.1287
       sp         (date, latitude, longitude) float64 100MB 1.011e+05 ... 7.03e+04
       tciw       (date, latitude, longitude) float64 100MB 0.02062 ... 0.003697
       tclw       (date, latitude, longitude) float64 100MB 0.005785 ... 2.902e-05
       tco3       (date, latitude, longitude) float64 100MB 0.006788 ... 0.005639
       tcwv       (date, latitude, longitude) float64 100MB 2.86 2.86 ... 1.031
       totalx     (date, latitude, longitude) float64 100MB 33.78 33.78 ... 24.45
       tsr        (date, latitude, longitude) float64 100MB 0.0 0.0 ... 1.344e+07
    """

    train_dataset = transform_dataset(train_data)
    val_dataset = transform_dataset(val_data)

    scaler_train = XarrayMinMaxScaler(
        dim=("date", "latitude", "longitude"), min=-1, max=1
    )
    scaler_train.fit(train_dataset)
    train_scaled = scaler_train.transform(train_dataset)
    scaler_train.save(conf.preprocess.params_dir)

    val_scaled = scaler_train.transform(val_dataset)

    if not os.path.exists(conf.dataset.era5.path):
        os.makedirs(conf.dataset.era5.path)

    for year in conf.dataset.train_years:
        path = f"{conf.dataset.era5.path}/era5_single_levels_monthly_{year}.nc"
        print(f"Saving scaled data for year {year} to {path}...")
        train_scaled.sel(date=str(year)).to_netcdf(path)

    for year in conf.dataset.val_years:
        path = f"{conf.dataset.era5.path}/era5_single_levels_monthly_{year}.nc"
        print(f"Saving scaled data for year {year} to {path}...")
        val_scaled.sel(date=str(year)).to_netcdf(path)


def main():
    parser = argparse.ArgumentParser(description="Preprocess ERA5 data")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to config file",
        default="config_preprocess.yaml",
    )
    args = parser.parse_args()

    preprocess(args.config_file)


if __name__ == "__main__":
    main()
