import os
import pickle
from pathlib import Path

import numpy as np
import xarray as xr

from config_utils import variable_config_from_omegaconf
from ecod_calculation import cod_from_water_path


class Preprocessor:
    def __init__(self):
        pass

    def fit(self, ds: xr.Dataset):
        pass

    def transform(self, ds: xr.Dataset):
        return ds

    def inverse_transform(self, ds: xr.Dataset):
        return ds

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessor state not found at: {path}")


class ECOD_Calculator(Preprocessor):
    def __init__(self, log=False):
        super().__init__()
        self.log = log

    def transform(self, ds):
        ds["ecod"] = ds.tcc * cod_from_water_path(ds.tclw * 1000, ds.tciw * 1000)
        if self.log:
            ds = ds.assign(ecod=np.log1p(ds["ecod"]))
        ds["ecod_fal"] = ds["ecod"] * ds["fal"]
        return ds

    def inverse_transform(self, ds):
        if self.log:
            ds = ds.assign(ecod=(np.exp(ds["ecod"]) - 1))

        """ECOD is derived, so we can't reverse it. Just return the dataset."""
        return ds


class PlanetaryAlbedoCalculator(Preprocessor):
    def transform(self, ds):
        if "tsr" in ds and "tisr" in ds:
            ds["pal"] = ds["tsr"] / ds["tisr"]
        if "tsrc" in ds and "tisr" in ds:
            ds["palc"] = ds["tsrc"] / ds["tisr"]
        return ds


class VariableSelector(Preprocessor):
    def __init__(self, vars: list[str]):
        super().__init__()
        self.vars = vars

    def transform(self, ds):
        missing = [var for var in self.vars if var not in ds.data_vars]
        if missing:
            raise ValueError(f"Dataset missing variables for model input: {missing}")
        return ds[self.vars]


class Downscaler(Preprocessor):
    def __init__(self, factor: list[tuple[str, int]] | None = None):
        super().__init__()
        self.factor = factor or [("latitude", 4), ("longitude", 4)]
        self.original_dims_sizes = {}

    def transform(self, ds: xr.Dataset):
        """Downscale data by averaging over 4x4 blocks in lat/lon."""
        print("Downscaling dataset")
        coarsen_kwargs = {}
        for dim, factor in self.factor:
            assert dim in ds.coords, f"Dimension '{dim}' not found in dataset."

            self.original_dims_sizes[dim] = ds[dim].values
            coarsen_kwargs[dim] = factor

        ds_downscaled = ds.coarsen(**coarsen_kwargs, boundary="trim").mean()
        return ds_downscaled

    def inverse_transform(self, ds):
        interp_kwargs = {}
        for coord, vals in self.original_dims_sizes.items():
            interp_kwargs[coord] = vals

        ds_upscaled = ds.interp(
            **interp_kwargs, method="nearest", kwargs={"fill_value": "extrapolate"}
        )
        return ds_upscaled

    def save(self, path):
        super().save(path)
        data = {
            "original_dims_sizes": self.original_dims_sizes,
            "factor": self.factor,
        }
        with open(Path(path) / "downscaler_dims.pkl", "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        super().load(path)
        with open(Path(path) / "downscaler_dims.pkl", "rb") as f:
            data = pickle.load(f)
        self.original_dims_sizes = data["original_dims_sizes"]
        self.factor = data["factor"]


class Identity(Preprocessor):
    def __init__(self):
        super().__init__()

    def transform(self, ds: xr.Dataset):
        return ds

    def inverse_transform(self, ds):
        return ds

    def save(self, path):
        pass

    def load(self, path):
        pass


class XarrayMinMaxScaler(Preprocessor):
    def __init__(self, dim, min_val=-1, max_val=1):
        self.dim = dim
        self.min_val = min_val
        self.max_val = max_val
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, ds):
        """Find min and max across the specified dimension for all variables."""
        print(f"Calculating global min/max across {self.dim}...")
        # Only the final min/max values are computed and pulled into memory
        self.data_min_ = ds.min(dim=self.dim).compute()
        self.data_max_ = ds.max(dim=self.dim).compute()

    def get_data_min(self):
        if self.data_min_ is None:
            raise RuntimeError("Scaler must be fitted before accessing data_min_.")
        return self.data_min_

    def get_data_max(self):
        if self.data_max_ is None:
            raise RuntimeError("Scaler must be fitted before accessing data_max_.")
        return self.data_max_

    def update_shared_range(self, var1, var2):
        print(f"Linking {var1} and {var2}")

        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("Scaler must be fitted before updating shared range.")

        new_min = min(self.data_min_[var1].values, self.data_min_[var2].values)
        new_max = max(self.data_max_[var1].values, self.data_max_[var2].values)

        self.data_min_[var1] = new_min
        self.data_max_[var1] = new_max
        self.data_min_[var2] = new_min
        self.data_max_[var2] = new_max

    def transform(self, ds):
        """Scale data to the [min_val, max_val] range."""
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("Scaler must be fitted before transforming.")

        # Avoid division by zero
        denom = self.data_max_ - self.data_min_
        denom = denom.where(denom != 0, 1.0)

        ds_std = (ds - self.data_min_) / denom

        scaled = ds_std * (self.max_val - self.min_val) + self.min_val
        self._warn_if_out_of_range(scaled)
        return scaled

    def _warn_if_out_of_range(self, ds):
        data_min = float(ds.to_array().min(skipna=True).compute())
        data_max = float(ds.to_array().max(skipna=True).compute())
        if data_min >= self.min_val and data_max <= self.max_val:
            return

        message = (
            f"Preprocessor transformed data outside [{self.min_val}, {self.max_val}]: "
            f"min={data_min:.6g}, max={data_max:.6g}"
        )
        print(f"WARNING: {message}")

    def inverse_transform(self, ds):
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("Scaler must be fitted before transforming.")

        denom = self.data_max_ - self.data_min_
        denom = denom.where(denom != 0, 1.0)
        ds_std = (ds - self.min_val) / (self.max_val - self.min_val)
        return ds_std * denom + self.data_min_

    def save(self, path):
        super().save(path)
        self.get_data_min().to_netcdf(Path(path) / "min.nc")
        self.get_data_max().to_netcdf(Path(path) / "max.nc")

    def load(self, path):
        super().load(path)
        self.data_min_ = xr.open_dataset(Path(path) / "min.nc")
        self.data_max_ = xr.open_dataset(Path(path) / "max.nc")


class XarrayStandardScaler(Preprocessor):
    def __init__(self, dim="date"):
        self.dim = dim
        self.mean_ = None
        self.std_ = None

    def get_mean(self):
        if self.mean_ is None:
            raise RuntimeError("Scaler must be fitted before accessing mean_.")
        return self.mean_

    def get_std(self):
        if self.std_ is None:
            raise RuntimeError("Scaler must be fitted before accessing std_.")
        return self.std_

    def fit(self, ds):
        print(f"Fitting scaler across dimension: {self.dim}...")
        self.mean_ = ds.mean(dim=self.dim).compute()
        self.std_ = ds.std(dim=self.dim).compute()
        # Prevent division by zero if std is 0
        self.std_ = self.std_.where(self.std_ != 0, 1.0)

    def transform(self, ds):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fitted before transforming data.")
        return (ds - self.mean_) / self.std_

    def inverse_transform(self, ds):
        return (ds * self.get_std()) + self.get_mean()

    def save(self, path):
        super().save(path)
        self.get_mean().to_netcdf(Path(path) / "mean.nc")
        self.get_std().to_netcdf(Path(path) / "std.nc")

    def load(self, path):
        super().load(path)
        self.mean_ = xr.open_dataset(Path(path) / "mean.nc")
        self.std_ = xr.open_dataset(Path(path) / "std.nc")


class SequentialPreprocessor(Preprocessor):
    def __init__(self, preprocessors: list[Preprocessor]):
        self.preprocessors = preprocessors

    def fit(self, ds):
        for preprocessor in self.preprocessors:
            preprocessor.fit(ds)
            ds = preprocessor.transform(ds)

    def transform(self, ds):
        for preprocessor in self.preprocessors:
            ds = preprocessor.transform(ds)
        return ds

    def inverse_transform(self, ds):
        for preprocessor in reversed(self.preprocessors):
            ds = preprocessor.inverse_transform(ds)
        return ds

    def save(self, path):
        super().save(path)
        for i, preprocessor in enumerate(self.preprocessors):
            preprocessor.save(Path(path) / f"preprocessor_{i}")

    def load(self, path):
        super().load(path)
        for i, preprocessor in enumerate(self.preprocessors):
            preprocessor.load(Path(path) / f"preprocessor_{i}")


class Preprocessor(SequentialPreprocessor):
    def __init__(self, config, downscaling=True):
        self.config = config
        self.variable_config = variable_config_from_omegaconf(config)
        vars = self.variable_config.all_vars()

        self.scalar = XarrayMinMaxScaler(dim=("date", "latitude", "longitude"))
        preprocessors: list[Preprocessor] = [
            PlanetaryAlbedoCalculator(),
            (
                ECOD_Calculator(log=config.preprocess.ecod.log)
                if config.preprocess.ecod.enabled
                else Identity()
            ),
            VariableSelector(vars),
            (
                Downscaler(factor=[("latitude", 4), ("longitude", 4)])
                if downscaling
                else Identity()
            ),
            self.scalar,
        ]

        super().__init__(preprocessors=preprocessors)

    def fit(self, ds):
        super().fit(ds)

        if ("all" in self.config.dataset.sky) and ("clear" in self.config.dataset.sky):
            self.scalar.update_shared_range(
                var1=self.variable_config.target_var,
                var2=self.variable_config.clear_sky_target,
            )


def fast_compute_cloud_optical_depth(
    tclw,
    tciw,
    tcc,
    re_liquid=10e-6,
    re_ice=30e-6,
    rho_water=1000.0,
    rho_ice=917.0,
):
    tau_l = 1.5 * tclw / (rho_water * re_liquid)
    tau_i = 1.5 * tciw / (rho_ice * re_ice)
    return (tau_l + tau_i) * tcc
