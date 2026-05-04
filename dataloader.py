import os
import numpy as np
import torch
import xarray as xr
from omegaconf import OmegaConf


def make_era5_filename(year):
    return f"era5_single_levels_monthly_{year}.nc"


def make_kernel_filename(year):
    pass


class ClimateTorchDataset(torch.utils.data.Dataset):
    def __init__(self, config_path="config.yaml", data_type="train", device=None):
        """
        Args:
            config_path (str): Path to OmegaConf YAML config
            data_type (str): Type of data to load ("train" or "val")
            device (optional): Device for tensors
        """
        conf = OmegaConf.load(config_path)

        if data_type == "train":
            era5_paths = [
                f"{conf.dataset.era5.path}/{make_era5_filename(year)}"
                for year in conf.dataset.train_years
            ]
        else:
            era5_paths = [
                f"{conf.dataset.era5.path}/{make_era5_filename(year)}"
                for year in conf.dataset.val_years
            ]
        print(f"Loading ERA5 data from: {era5_paths}")

        self.dataset_era5 = xr.open_mfdataset(
            era5_paths, combine="nested", concat_dim="date"
        )
        data_array = self.dataset_era5.to_dataarray(dim="variable").transpose(
            "date", "latitude", "longitude", "variable"
        )
        # Write to a memmap once
        mmap_path = data_type + "_era5_cache.npy"
        if not os.path.exists(mmap_path):
            np.save(mmap_path, data_array.values)

        self.n_dates = len(self.dataset_era5.date)
        self.n_lat = len(self.dataset_era5.latitude)
        self.n_lon = len(self.dataset_era5.longitude)
        self.n_vars = len(self.dataset_era5.data_vars)

        self.shape = (self.n_dates, self.n_lat, self.n_lon, self.n_vars)
        self.data = np.load(mmap_path, mmap_mode="r")  # OS handles paging

        # dataset_kernels = xr.open(mfdataset(...))
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

        self.device = device

    def __len__(self):
        return self.n_dates * self.n_lat * self.n_lon

    def __getitem__(self, idx):
        # Convert flat index to 4D indices
        date_idx = idx // (self.n_lat * self.n_lon)
        rem = idx % (self.n_lat * self.n_lon)
        lat_idx = rem // self.n_lon
        lon_idx = rem % self.n_lon

        # Extract the data for this index
        data_point = self.data[date_idx, lat_idx, lon_idx, :]

        # Convert to torch tensor
        tensor_data = torch.from_numpy(data_point).float()

        if self.device:
            tensor_data = tensor_data.to(self.device)

        X = tensor_data[:-1]  # All but last variable as input
        y = tensor_data[-1]  # Last variable as target
        return X, y
