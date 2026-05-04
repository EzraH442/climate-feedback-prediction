import torch
from omegaconf import OmegaConf
from preprocessing import XarrayMinMaxScaler
import xarray as xr


def make_era5_filename(year):
    return f"era5_single_levels_monthly_{year}.nc"


config_path = "config_preprocess.yaml"
conf = OmegaConf.load(config_path)
torch.manual_seed(conf.seed)

era5_paths = [
    f"{conf.dataset.era5.raw_path}/{make_era5_filename(year)}"
    for year in conf.dataset.years
]
print(f"Loading ERA5 data from: {era5_paths}")

train_dataset = xr.open_mfdataset(era5_paths, combine="nested", concat_dim="date")
"""
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

scaler = XarrayMinMaxScaler(dim=("date", "latitude", "longitude"), min=-1, max=1)
scaler.fit(train_dataset)
train_scaled = scaler.transform(train_dataset)
scaler.save("scaler_small")

for year in conf.dataset.years:
    path = f"{conf.dataset.era5.path}/era5_single_levels_monthly_{year}.nc"
    print(f"Saving scaled data for year {year} to {path}...")
    train_scaled.sel(date=str(year)).to_netcdf(path)
