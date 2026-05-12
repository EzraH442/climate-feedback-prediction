import os
from pathlib import Path
import numpy as np
import torch
import xarray as xr

from config_utils import load_config


def make_era5_filename(year):
    return f"era5_single_levels_monthly_{year}.nc"


def make_kernel_path(year):
    return f"RRTM_kernel_monthly_{year}_cld_alb_TOA_SFC.nc"


def make_mmap_stem(model_name: str, data_type: str, suffix: str) -> str:
    return f"{model_name}_{data_type}_{suffix}.npy"


def ordered_vars(dataset: xr.Dataset, target_var: str) -> list[str]:
    vars_without_target = [v for v in dataset.data_vars if v != target_var]
    if "fal" not in vars_without_target:
        raise ValueError("'fal' was not found in dataset variables")
    return ["fal", *[v for v in vars_without_target if v != "fal"], target_var]


class ClimateTorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_path="configs/model/baseline.yaml",
        data_type="train",
        target_var="tsr",
    ):
        """
        Args:
            config_path (str): Path to OmegaConf YAML config
            data_type (str): Type of data to load ("train" or "val")
        """
        conf = load_config(config_path)
        model_name = conf.train.name
        years = (
            conf.dataset.train_years if data_type == "train" else conf.dataset.val_years
        )

        era5_paths = [
            Path(conf.dataset.era5.path) / make_era5_filename(year) for year in years
        ]

        print(f"Loading ERA5 data from: {era5_paths}")

        self.dataset_era5 = xr.open_mfdataset(
            era5_paths, combine="nested", concat_dim="date"
        )
        all_vars = ordered_vars(self.dataset_era5, target_var)
        self.n_dates = len(self.dataset_era5.date)
        self.n_lat = len(self.dataset_era5.latitude)
        self.n_lon = len(self.dataset_era5.longitude)
        self.n_vars = len(all_vars)

        expected_shape = (self.n_dates, self.n_lat, self.n_lon, self.n_vars)
        slurm_tmpdir = os.getenv("SLURM_TMPDIR")
        if not slurm_tmpdir:
            raise EnvironmentError("SLURM_TMPDIR environment variable is not set.")

        mmap_path = Path(slurm_tmpdir) / make_mmap_stem(
            model_name, data_type, "data_mmap"
        )
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


class KernelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_path="configs/model/baseline.yaml",
        data_type="train",
        target_var="tsr",
    ):
        """
        Args:
            config_path (str): Path to OmegaConf YAML config
            data_type (str): Type of data to load ("train" or "val")
        """
        conf = load_config(config_path)
        model_name = conf.train.name
        years = (
            conf.dataset.train_years if data_type == "train" else conf.dataset.val_years
        )

        # --- load era5 data ---
        era5_paths = [
            Path(conf.dataset.era5.path) / make_era5_filename(year) for year in years
        ]
        print(f"Loading ERA5 data from: {era5_paths}")
        dataset_era5 = xr.open_mfdataset(
            era5_paths, combine="nested", concat_dim="date"
        )

        # --- load kernel data ---
        kernel_paths = [
            Path(conf.dataset.kernels.path) / make_kernel_path(year) for year in years
        ]
        print(f"Loading kernel data from: {kernel_paths}")
        dataset_kernels = xr.open_mfdataset(
            kernel_paths, combine="nested", concat_dim="date"
        )

        # --- dimension setup ---
        all_vars = ordered_vars(dataset_era5, target_var)

        self.n_dates = len(dataset_era5.date)
        self.n_lat = len(dataset_era5.latitude)
        self.n_lon = len(dataset_era5.longitude)
        self.n_vars = len(all_vars)

        expected_shape_era5 = (self.n_dates, self.n_lat, self.n_lon, self.n_vars)
        expected_shape_kern = (self.n_dates, self.n_lat, self.n_lon, 1)

        # --- save data to slurm tempdir ---
        slurm_tmpdir = os.getenv("SLURM_TMPDIR")
        if not slurm_tmpdir:
            raise EnvironmentError("SLURM_TMPDIR environment variable is not set.")

        mmap_path_era5 = Path(slurm_tmpdir) / make_mmap_stem(
            model_name, data_type, "mmap_era5"
        )
        mmap_path_kern = Path(slurm_tmpdir) / make_mmap_stem(
            model_name, data_type, "mmap_kern"
        )

        if not mmap_path_era5.exists():
            mmap = np.lib.format.open_memmap(
                mmap_path_era5, mode="w+", dtype=np.float32, shape=expected_shape_era5
            )
            for t in range(self.n_dates):
                data_slice = (
                    dataset_era5[all_vars]
                    .isel(date=t)
                    .to_dataarray()
                    .transpose("latitude", "longitude", "variable")
                    .values
                )
                mmap[t] = data_slice
                if t % 12 == 0:
                    print(
                        f"Processed {t}/{self.n_dates} dates into ERA5 memory-mapped file."
                    )
            mmap.flush()  # Ensure data is written to disk
            del mmap  # Close the memmap
            print(f"Data successfully written to memory-mapped file: {mmap_path_era5}")
        else:
            existing = np.load(mmap_path_era5, mmap_mode="r")
            if existing.shape != expected_shape_era5:
                raise ValueError(
                    f"Existing memory-mapped file shape {existing.shape} does not match expected shape {expected_shape_era5}."
                )

        if not mmap_path_kern.exists():
            mmap = np.lib.format.open_memmap(
                mmap_path_kern, mode="w+", dtype=np.float32, shape=expected_shape_kern
            )
            for t in range(self.n_dates):
                data_slice = (
                    dataset_kernels.isel(date=t)
                    .to_dataarray()
                    .transpose("latitude", "longitude", "variable")
                    .values
                )
                mmap[t] = data_slice
                if t % 12 == 0:
                    print(
                        f"Processed {t}/{self.n_dates} dates into kernel memory-mapped file."
                    )
            mmap.flush()  # Ensure data is written to disk
            del mmap  # Close the memmap
            print(f"Data successfully written to memory-mapped file: {mmap_path_kern}")
        else:
            existing = np.load(mmap_path_kern, mmap_mode="r")
            if existing.shape != expected_shape_kern:
                raise ValueError(
                    f"Existing memory-mapped file shape {existing.shape} does not match expected shape {expected_shape_kern}."
                )

        dataset_era5.close()
        dataset_kernels.close()
        self.data_era5 = np.load(mmap_path_era5, mmap_mode="r")  # OS handles paging
        self.data_kernels = np.load(mmap_path_kern, mmap_mode="r")  # OS handles paging

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
        era5_data_point = self.data_era5[date_idx, lat_idx, lon_idx, :].copy()
        kern_data_point = self.data_kernels[date_idx, lat_idx, lon_idx, :].copy()

        # Convert to torch tensor
        era5_tensor_data = torch.from_numpy(era5_data_point)
        kern_tensor_data = torch.from_numpy(kern_data_point)
        X = era5_tensor_data[:-1]  # All but last variable as input
        y = era5_tensor_data[-1]  # Last variable as target

        return X, y, kern_tensor_data
