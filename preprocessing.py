import os
import xarray as xr
from pathlib import Path
import pickle

def compute_cloud_optical_depth(
    tclw,  # total column liquid water
    tciw,  # total column ice water
    tcc,  # total cloud cover (0–1)
    re_liquid=10e-6,  # liquid effective radius (m)
    re_ice=30e-6,  # ice effective radius (m)
    rho_water=1000.0,  # water density (kg/m^3)
    rho_ice=917.0,  # ice density (kg/m^3)
):

    tau_l = 1.5 * tclw / (rho_water * re_liquid)
    tau_i = 1.5 * tciw / (rho_ice * re_ice)

    tau_total = tau_l + tau_i
    tau_effective = tau_total * tcc

    return tau_effective

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
        pass

    def load(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Preprocessor state not found at: {path}")
        pass


class ECOD_Calculator(Preprocessor):
    def __init__(self):
        super().__init__()

    def transform(self, ds):
        """Calculate ECOD from ERA5 variables."""
        print("Calculating ECOD...")
        ecod_ds = compute_cloud_optical_depth(
            tclw=ds["tclw"], tciw=ds["tciw"], tcc=ds["tcc"]
        )
        ds["ecod"] = ecod_ds
        ds["ecod_fal"] = ds["ecod"] * ds["fal"]
        ds = ds.drop_vars("tcc")
        return ds

    def inverse_transform(self, ds):
        """ECOD is derived, so we can't reverse it. Just return the dataset."""
        return ds


class VariableSelector(Preprocessor):
    def __init__(self, input_vars, target_var):
        super().__init__()
        self.vars = list(dict.fromkeys([*input_vars, target_var]))

    def transform(self, ds):
        missing = [var for var in self.vars if var not in ds.data_vars]
        if missing:
            raise ValueError(f"Dataset missing variables for model input: {missing}")
        return ds[self.vars]


class Downscaler(Preprocessor):
    def __init__(
        self, factor: list[tuple[str, int]] = [("latitude", 4), ("longitude", 4)]
    ):
        super().__init__()
        self.factor = factor
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

        ds_upscaled = ds.interp(**interp_kwargs, method="nearest", kwargs={"fill_value": "extrapolate"})
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

    def transform(self, ds):
        """Scale data to the [min_val, max_val] range."""
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("Scaler must be fitted before transforming.")

        # Avoid division by zero
        denom = self.data_max_ - self.data_min_
        denom = denom.where(denom != 0, 1.0)

        ds_std = (ds - self.data_min_) / denom

        return ds_std * (self.max_val - self.min_val) + self.min_val

    def inverse_transform(self, ds):
        if self.data_min_ is None or self.data_max_ is None:
            raise RuntimeError("Scaler must be fitted before transforming.")

        denom = self.data_max_ - self.data_min_
        denom = denom.where(denom != 0, 1.0)
        ds_std = (ds - self.min_val) / (self.max_val - self.min_val)
        return ds_std * denom + self.data_min_

    def save(self, path):
        super().save(path)
        self.data_min_.to_netcdf(Path(path) / "min.nc")
        self.data_max_.to_netcdf(Path(path) / "max.nc")

    def load(self, path):
        super().load(path)
        self.data_min_ = xr.open_dataset(Path(path) / "min.nc")
        self.data_max_ = xr.open_dataset(Path(path) / "max.nc")


class XarrayStandardScaler(Preprocessor):
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

    def transform(self, ds):
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler must be fitted before transforming data.")
        return (ds - self.mean_) / self.std_

    def inverse_transform(self, ds):
        return (ds * self.std_) + self.mean_

    def save(self, path):
        super().save(path)
        self.mean_.to_netcdf(Path(path) / "mean.nc")
        self.std_.to_netcdf(Path(path) / "std.nc")

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


def create_2024_preprocessor(input_vars=None, target_var="tsr", ecod=True):
    preprocessors = []
    if ecod:
        preprocessors.append(ECOD_Calculator())
    if input_vars is not None:
        preprocessors.append(VariableSelector(input_vars, target_var))
    preprocessors.extend(
        [
            Downscaler(factor=[("latitude", 4), ("longitude", 4)]),
            XarrayMinMaxScaler(dim=("date", "latitude", "longitude")),
        ]
    )
    return SequentialPreprocessor(
        preprocessors=preprocessors
    )

def create_2024_preprocessor_no_downscaling():
    return SequentialPreprocessor(
        preprocessors=[
            ECOD_Calculator(),
            Identity(),
            XarrayMinMaxScaler(dim=("date", "latitude", "longitude")),
        ]
    )
