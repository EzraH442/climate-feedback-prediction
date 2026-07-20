import os
from pathlib import Path

import numpy as np
import torch

from config_utils import load_config, variable_config_from_omegaconf
from preprocessing import DianaPreprocessor
from utils import (
    make_combined_kernel_filename,
    make_era5_filename,
    load_yearly_and_filter_by_months,
)


def make_mmap_stem(model_name: str, data_type: str, suffix: str) -> str:
    return f"{model_name}_{data_type}_{suffix}.npy"


def load_or_create_mmap(path: Path, shape: tuple[int, ...], build_array):
    if path.exists():
        mmap = np.load(path, mmap_mode="r")
        if mmap.shape == shape and mmap.dtype == np.float32:
            return mmap
        path.unlink()

    values = build_array().astype(np.float32, copy=False)
    if values.shape != shape:
        raise ValueError(f"{path.name} shape {values.shape} does not match {shape}.")
    mmap = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    mmap[:] = values
    mmap.flush()
    del mmap
    return np.load(path, mmap_mode="r")


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
        config_path="configs/model/fal/1990-2020_1-12_baseline.yaml",
        data_type="train",
    ):
        """
        Args:
            config_path (str): Path to OmegaConf YAML config
            data_type (str): Type of data to load ("train" or "val")
        """
        conf = load_config(config_path)
        self.variable_config = variable_config_from_omegaconf(conf)
        model_name = conf.train.name

        ### input setup

        ### years and months to load
        years = (
            conf.dataset.train_years if data_type == "train" else conf.dataset.val_years
        )
        months = conf.dataset.months
        self.sobolev = bool(conf.train.sobolev)

        ### sky setup
        self.sky = conf.dataset.sky

        self.clear_sky_target_var = self.variable_config.clear_sky_target
        self.dataset_era5 = load_yearly_and_filter_by_months(
            path=conf.dataset.era5.path,
            years=years,
            months=months,
            filename_fn=make_era5_filename,
        )

        if self.sobolev:
            self.dataset_kernels = load_yearly_and_filter_by_months(
                path=conf.dataset.kernels.path,
                years=years,
                months=months,
                filename_fn=make_combined_kernel_filename,
            )

        input_order = self.variable_config.input_order()
        kernel_order = self.variable_config.kern_input_order()

        self.n_dates = len(self.dataset_era5.date)
        self.n_lat = len(self.dataset_era5.latitude)
        self.n_lon = len(self.dataset_era5.longitude)
        self.base_len = self.n_dates * self.n_lat * self.n_lon
        self.sample_weights = area_weights_from_latitudes(
            self.dataset_era5.latitude.to_numpy(),
            n_dates=self.n_dates,
            n_lon=self.n_lon,
        )
        self.sample_weights = torch.cat([self.sample_weights] * len(self.sky))

        slurm_tmpdir = os.getenv("SLURM_TMPDIR")
        if not slurm_tmpdir:
            raise EnvironmentError("SLURM_TMPDIR environment variable is not set.")
        mmap_dir = Path(slurm_tmpdir)
        mmap_dir.mkdir(parents=True, exist_ok=True)

        grid_shape = (self.n_dates, self.n_lat, self.n_lon)
        dim_order = ["date", "latitude", "longitude", "variable"]

        if "all" in self.sky:
            self.x = load_or_create_mmap(
                mmap_dir / make_mmap_stem(model_name, data_type, "x"),
                (*grid_shape, len(input_order)),
                lambda: self.variable_config.inputs_np(self.dataset_era5, dim_order),
            )
            self.y = load_or_create_mmap(
                mmap_dir / make_mmap_stem(model_name, data_type, "y"),
                (*grid_shape,),
                lambda: self.variable_config.outputs_np(self.dataset_era5),
            )
        if "clear" in self.sky:

            def build_x_clear():
                preprocessor = DianaPreprocessor(conf)
                preprocessor.load(conf.preprocess.params_dir)
                raw = load_yearly_and_filter_by_months(
                    path=conf.dataset.era5.raw_path,
                    years=years,
                    months=months,
                    filename_fn=make_era5_filename,
                )
                clear = self.variable_config.clear_sky_input(raw)
                processed = preprocessor.transform(clear)
                values = self.variable_config.inputs_np(processed, dim_order)
                raw.close()
                processed.close()
                return values

            self.x_clear = load_or_create_mmap(
                mmap_dir / make_mmap_stem(model_name, data_type, "x_clear_raw_zero"),
                (*grid_shape, len(input_order)),
                build_x_clear,
            )
            self.y_clear = load_or_create_mmap(
                mmap_dir / make_mmap_stem(model_name, data_type, "y_clear"),
                (*grid_shape,),
                lambda: self.variable_config.outputs_np(self.dataset_era5, clear=True),
            )
        if self.sobolev:
            self.k = load_or_create_mmap(
                mmap_dir / make_mmap_stem(model_name, data_type, "k"),
                (*grid_shape, len(kernel_order)),
                lambda: self.variable_config.kern_inputs_np(
                    self.dataset_kernels, dim_order
                ),
            )
            if "sky" in self.sky:
                self.k_clear = load_or_create_mmap(
                    mmap_dir / make_mmap_stem(model_name, data_type, "k_clear"),
                    (*grid_shape, len(kernel_order)),
                    lambda: self.variable_config.kern_inputs_np(
                        self.dataset_kernels, dim_order, clear=True
                    ),
                )

        self.dataset_era5.close()
        if self.sobolev:
            self.dataset_kernels.close()

    def __len__(self):
        return self.base_len * len(self.sky)

    def _get_all(self, idx):
        date_idx = idx // (self.n_lat * self.n_lon)
        rem = idx % (self.n_lat * self.n_lon)
        lat_idx = rem // self.n_lon
        lon_idx = rem % self.n_lon

        x = self.x[date_idx, lat_idx, lon_idx]
        y = self.y[date_idx, lat_idx, lon_idx]
        if not self.sobolev:
            return x, y, np.array(0)
        k = self.k[date_idx, lat_idx, lon_idx]

        return x, y, k

    def _get_clear(self, idx):
        date_idx = idx // (self.n_lat * self.n_lon)
        rem = idx % (self.n_lat * self.n_lon)
        lat_idx = rem // self.n_lon
        lon_idx = rem % self.n_lon

        x = self.x_clear[date_idx, lat_idx, lon_idx]
        y = self.y_clear[date_idx, lat_idx, lon_idx]
        if not self.sobolev:
            return x, y, np.array(0)
        k = self.k_clear[date_idx, lat_idx, lon_idx]

        return x, y, k

    def __getitem__(self, idx):
        if len(self.sky) == 2:
            clear_sky_sample = idx >= self.base_len
            if clear_sky_sample:
                idx -= self.base_len
                x, y, k = self._get_clear(idx)
            else:
                x, y, k = self._get_all(idx)
            pass
        elif self.sky[0] == "all":
            x, y, k = self._get_all(idx)
        elif self.sky[0] == "clear":
            x, y, k = self._get_clear(idx)
        else:
            raise NotImplementedError('path not implemented')

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float(),
            torch.from_numpy(k).float(),
        )
