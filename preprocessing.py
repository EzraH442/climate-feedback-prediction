import os
import xarray as xr
from pathlib import Path


class XarrayMinMaxScaler:
    def __init__(self, dim, min=-1, max=1):
        self.dim = dim
        self.min_val = min
        self.max_val = max
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, ds):
        """Find min and max across the specified dimension for all variables."""
        print(f"Calculating global min/max across {self.dim}...")
        # Only the final min/max values are computed and pulled into memory
        self.data_min_ = ds.min(dim=self.dim).compute()
        self.data_max_ = ds.max(dim=self.dim).compute()
        return self

    def transform(self, ds):
        """Scale data to the [min_val, max_val] range."""
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("Scaler must be fitted before transforming.")

        # Standard 0-1 scaling first
        # We add a tiny epsilon or check to avoid division by zero
        denom = self.data_max_ - self.data_min_
        denom = denom.where(denom != 0, 1.0)

        ds_std = (ds - self.data_min_) / denom

        # Scale to the custom range (e.g., -1 to 1)
        return ds_std * (self.max_val - self.min_val) + self.min_val

    def inverse_transform(self, ds_scaled):
        """Convert scaled data back to original physical values."""
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("Scaler must be fitted before transforming.")

        denom = self.data_max_ - self.data_min_
        ds_std = (ds_scaled - self.min_val) / (self.max_val - self.min_val)
        return ds_std * denom + self.data_min_

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        self.data_min_.to_netcdf(Path(path) / "min.nc")
        self.data_max_.to_netcdf(Path(path) / "max.nc")

    def load(self, path):
        self.data_min_ = xr.open_dataset(Path(path) / "min.nc")
        self.data_max_ = xr.open_dataset(Path(path) / "max.nc")


class XarrayStandardScaler:
    def __init__(self, dim="date"):
        self.dim = dim
        self.mean_ = None
        self.std_ = None

    def fit(self, ds):
        print(f"Fitting scaler across dimension: {self.dim}...")
        self.mean_ = ds.mean(dim=self.dim).compute()
        self.std_ = ds.std(dim=self.dim).compute()
        # Prevent division by zero if std is 0
        self.std_ = self.std_.where(self.std_ != 0, 1.0)
        return self

    def transform(self, ds):
        """Apply the saved mean and std to new data."""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fitted before transforming data.")
        return (ds - self.mean_) / self.std_

    def inverse_transform(self, ds_norm):
        """Convert normalized data back to original scale."""
        return (ds_norm * self.std_) + self.mean_

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        """Save stats to NetCDF for production use."""
        self.mean_.to_netcdf(Path(path) / "mean.nc")
        self.std_.to_netcdf(Path(path) / "std.nc")

    def load(self, path):
        """Load stats from NetCDF."""
        self.mean_ = xr.open_dataset(Path(path) / "mean.nc")
        self.std_ = xr.open_dataset(Path(path) / "std.nc")
