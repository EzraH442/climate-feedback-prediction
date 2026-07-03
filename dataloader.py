import numpy as np
import torch

from config_utils import load_config, variable_config_from_omegaconf
from utils import (
    make_combined_kernel_filename,
    make_era5_filename,
    load_yearly_and_filter_by_months,
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
        config_path="configs/model/fal/1990-2020_1-12_baseline.yaml",
        data_type="train",
        target_var=None,
    ):
        """
        Args:
            config_path (str): Path to OmegaConf YAML config
            data_type (str): Type of data to load ("train" or "val")
        """
        conf = load_config(config_path)
        self.variable_config = variable_config_from_omegaconf(conf)
        model_name = conf.train.name

        ### input and output setup
        target_var = target_var or conf.dataset.target_var
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

        all_vars = self.variable_config.all_vars()

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

        # select points
        data_point = self.dataset_era5.isel(
            date=date_idx, latitude=lat_idx, longitude=lon_idx
        )
        kern_data_point = self.dataset_kernels.isel(
            date=date_idx, latitude=lat_idx, longitude=lon_idx
        )

        # order points
        x = self.variable_config.inputs_np(data_point, clear=clear_sky_sample)
        y = self.variable_config.outputs_np(data_point, clear=clear_sky_sample)
        dy_dx = self.variable_config.kern_inputs_np(
            kern_data_point, clear=clear_sky_sample
        )

        if self.sobolev:
            return torch.tensor(x), torch.tensor(y), torch.tensor(dy_dx)

        else:
            return torch.tensor(x), torch.tensor(y), None
