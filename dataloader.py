import os
from pathlib import Path
import numpy as np
import torch
import xarray as xr

from config_utils import load_config
from utils import (
    filter_by_months,
    make_kernel_filename,
    make_era5_filename,
    ordered_vars,
)


def make_mmap_stem(model_name: str, data_type: str, suffix: str) -> str:
    return f"{model_name}_{data_type}_{suffix}.npy"


def area_weights_from_latitudes(
    latitudes_deg: np.ndarray,
    n_dates: int,
    n_lon: int,
) -> torch.Tensor:
    lat_weights = np.cos(np.deg2rad(latitudes_deg)).astype(np.float32)
    lat_weights = np.clip(lat_weights, 1e-6, None)
    sample_weights = np.broadcast_to(
        lat_weights[None, :, None],
        (n_dates, len(latitudes_deg), n_lon),
    ).reshape(-1)
    return torch.as_tensor(sample_weights, dtype=torch.double)


class ClimateTorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config_path="configs/model/baseline.yaml",
        data_type="train",
        target_var=None,
    ):
        """
        Args:
            config_path (str): Path to OmegaConf YAML config
            data_type (str): Type of data to load ("train" or "val")
        """
        conf = load_config(config_path)
        model_name = conf.train.name

        ### input and ouputput setup
        target_var = target_var or conf.dataset.target_var
        input_var = conf.dataset.input_var
        input_vars = conf.dataset.input_vars

        ### years and months to load
        years = (
            conf.dataset.train_years if data_type == "train" else conf.dataset.val_years
        )
        months = conf.dataset.months
        self.sobolev = bool(conf.train.sobolev)

        ### optional clear sky training setup
        clear_sky = conf.dataset.clear_sky_training
        self.clear_sky_enabled = bool(clear_sky.enabled)
        self.clear_sky_target_var = clear_sky.output_var
        zero_vars = conf.dataset.clear_sky_zero_vars
        self.clear_sky_zero_indices = [input_vars.index(var) for var in zero_vars]

        era5_paths = [
            Path(conf.dataset.era5.path) / make_era5_filename(year) for year in years
        ]

        print(f"Loading ERA5 data from: {era5_paths}")

        self.dataset_era5 = xr.open_mfdataset(
            era5_paths, combine="nested", concat_dim="date"
        )
        self.dataset_era5 = filter_by_months(self.dataset_era5, months)

        if self.sobolev:
            kernel_paths = [
                Path(conf.dataset.kernels.path) / make_kernel_filename(year)
                for year in years
            ]
            kernel_vars = list(conf.dataset.kernels.vars)
            print(f"Loading kernel data from: {kernel_paths}")
            dataset_kernels = xr.open_mfdataset(
                kernel_paths, combine="nested", concat_dim="date"
            )
            dataset_kernels = filter_by_months(dataset_kernels, months)
            missing_kernel_vars = [v for v in kernel_vars if v not in dataset_kernels]
            if missing_kernel_vars:
                raise ValueError(f"Kernel dataset missing variables: {missing_kernel_vars}")
            self.kernel_clear_sky_idx = (
                kernel_vars.index("TOA_clr") if self.clear_sky_enabled else None
            )
            if len(conf.train.sobolev_vars) == 1:
                self.kernel_output_indices = [0]
            else:
                self.kernel_output_indices = list(range(len(kernel_vars)))
        order_dataset = self.dataset_era5
        if self.clear_sky_target_var in order_dataset:
            order_dataset = order_dataset.drop_vars(self.clear_sky_target_var)
        all_vars = ordered_vars(order_dataset, target_var, ecod=conf.preprocess.ecod, input_var=input_var, input_vars=input_vars)
        mmap_vars = [*all_vars]
        if self.clear_sky_enabled:
            if self.clear_sky_target_var not in self.dataset_era5:
                raise ValueError(
                    f"Clear-sky training enabled, but {self.clear_sky_target_var!r} is missing from preprocessed data."
                )
            mmap_vars.append(self.clear_sky_target_var)
        self.n_dates = len(self.dataset_era5.date)
        self.n_lat = len(self.dataset_era5.latitude)
        self.n_lon = len(self.dataset_era5.longitude)
        self.n_vars = len(all_vars)
        self.base_len = self.n_dates * self.n_lat * self.n_lon
        self.sample_weights = area_weights_from_latitudes(
            self.dataset_era5.latitude.to_numpy(),
            n_dates=self.n_dates,
            n_lon=self.n_lon,
        )
        if self.clear_sky_enabled:
            self.sample_weights = torch.cat([self.sample_weights, self.sample_weights])

        expected_shape = (self.n_dates, self.n_lat, self.n_lon, len(mmap_vars))
        slurm_tmpdir = os.getenv("SLURM_TMPDIR")
        if not slurm_tmpdir:
            raise EnvironmentError("SLURM_TMPDIR environment variable is not set.")

        mmap_path = Path(slurm_tmpdir) / make_mmap_stem(
            model_name, data_type, "data_mmap"
        )
        mmap_path_kern = Path(slurm_tmpdir) / make_mmap_stem(
            model_name, data_type, "mmap_kern"
        )
        if not mmap_path.exists():
            mmap = np.lib.format.open_memmap(
                mmap_path, mode="w+", dtype=np.float32, shape=expected_shape
            )
            for t in range(self.n_dates):
                data_slice = (
                    self.dataset_era5[mmap_vars]
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

        if self.sobolev:
            expected_shape_kern = (
                self.n_dates,
                self.n_lat,
                self.n_lon,
                len(kernel_vars),
            )
            if not mmap_path_kern.exists():
                mmap = np.lib.format.open_memmap(
                    mmap_path_kern,
                    mode="w+",
                    dtype=np.float32,
                    shape=expected_shape_kern,
                )
                for t in range(self.n_dates):
                    data_slice = (
                        dataset_kernels.isel(date=t)
                        [kernel_vars]
                        .to_dataarray()
                        .transpose("latitude", "longitude", "variable")
                        .values
                    )
                    mmap[t] = data_slice
                    if t % 12 == 0:
                        print(
                            f"Processed {t}/{self.n_dates} dates into kernel memory-mapped file."
                        )
                mmap.flush()
                del mmap
                print(f"Data successfully written to memory-mapped file: {mmap_path_kern}")
            else:
                existing = np.load(mmap_path_kern, mmap_mode="r")
                if existing.shape != expected_shape_kern:
                    raise ValueError(
                        f"Existing memory-mapped file shape {existing.shape} does not match expected shape {expected_shape_kern}."
                    )
            dataset_kernels.close()
            self.data_kernels = np.load(mmap_path_kern, mmap_mode="r")

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
        return self.base_len * (2 if self.clear_sky_enabled else 1)

    def __getitem__(self, idx):
        clear_sky_sample = self.clear_sky_enabled and idx >= self.base_len
        if clear_sky_sample:
            idx -= self.base_len

        # Convert flat index to 4D indices
        date_idx = idx // (self.n_lat * self.n_lon)
        rem = idx % (self.n_lat * self.n_lon)
        lat_idx = rem // self.n_lon
        lon_idx = rem % self.n_lon

        # Extract the data for this index
        data_point = self.data[date_idx, lat_idx, lon_idx, :].copy()
        if self.sobolev:
            kern_data_point = self.data_kernels[date_idx, lat_idx, lon_idx, :].copy()
        if clear_sky_sample:
            data_point[: self.n_vars][self.clear_sky_zero_indices] = 0
            data_point[self.n_vars - 1] = data_point[self.n_vars]
            if self.sobolev:
                kern_data_point[0] = kern_data_point[self.kernel_clear_sky_idx]
        data_point = data_point[: self.n_vars]

        # Convert to torch tensor
        tensor_data = torch.from_numpy(data_point)
        X = tensor_data[:-1]  # All but last variable as input
        y = tensor_data[-1]  # Last variable as target
        if self.sobolev:
            kern_data_point = kern_data_point[self.kernel_output_indices]
            return X, y, torch.from_numpy(kern_data_point)
        return X, y
