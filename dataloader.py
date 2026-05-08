import os
from pathlib import Path
import numpy as np
import torch
import xarray as xr
from omegaconf import OmegaConf


def make_era5_filename(year):
    return f"era5_single_levels_monthly_{year}.nc"


def make_kernel_filename(year):
    pass


class ClimateTorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_path="configs/model/small.yaml",
        data_type="train",
        target_var="tsr",
    ):
        """
        Args:
            config_path (str): Path to OmegaConf YAML config
            data_type (str): Type of data to load ("train" or "val")
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
        all_vars = [v for v in self.dataset_era5.data_vars if v != target_var] + [
            target_var
        ]
        self.n_dates = len(self.dataset_era5.date)
        self.n_lat = len(self.dataset_era5.latitude)
        self.n_lon = len(self.dataset_era5.longitude)
        self.n_vars = len(all_vars)

        expected_shape = (self.n_dates, self.n_lat, self.n_lon, self.n_vars)
        slurm_tmpdir = os.getenv("SLURM_TMPDIR")
        if not slurm_tmpdir:
            raise EnvironmentError("SLURM_TMPDIR environment variable is not set.")

        mmap_path = Path(slurm_tmpdir) / f"{data_type}_data_mmap.npy"
        if not mmap_path.exists():
            mmap = np.lib.format.open_memmap(
                mmap_path, mode="w+", dtype=np.float32, shape=expected_shape
            )
            for t in range(self.n_dates):
                data_slice = (
                    self.dataset_era5[all_vars]
                    .isel(date=t)
                    .to_dataarray()
                    .transpose("latitude", "longitude", "variable")
                    .values
                )
                mmap[t] = data_slice
                if t % 12 == 0:
                    print(
                        f"Processed {t}/{self.n_dates} dates into memory-mapped file."
                    )
            mmap.flush()  # Ensure data is written to disk
            del mmap  # Close the memmap
            print(f"Data successfully written to memory-mapped file: {mmap_path}")
        else:
            existing = np.load(mmap_path, mmap_mode="r")
            if existing.shape != expected_shape:
                raise ValueError(
                    f"Existing memory-mapped file shape {existing.shape} does not match expected shape {expected_shape}."
                )

        self.dataset_era5.close()
        self.data = np.load(mmap_path, mmap_mode="r")  # OS handles paging
        self.shape = (self.n_dates, self.n_lat, self.n_lon, self.n_vars)

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
           tisr       (date, latitude, longitude) float64 100MB 33.78 33.78 ... 24.45
           tsr        (date, latitude, longitude) float64 100MB 0.0 0.0 ... 1.344e+07
        """

    def __len__(self):
        return self.n_dates * self.n_lat * self.n_lon

    def __getitem__(self, idx):
        # Convert flat index to 4D indices
        date_idx = idx // (self.n_lat * self.n_lon)
        rem = idx % (self.n_lat * self.n_lon)
        lat_idx = rem // self.n_lon
        lon_idx = rem % self.n_lon

        # Extract the data for this index
        data_point = self.data[date_idx, lat_idx, lon_idx, :].copy()

        # Convert to torch tensor
        tensor_data = torch.from_numpy(data_point)
        X = tensor_data[:-1]  # All but last variable as input
        y = tensor_data[-1]  # Last variable as target
        return X, y
