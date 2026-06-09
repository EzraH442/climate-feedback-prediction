from pathlib import Path

import numpy as np
import xarray as xr
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

from preprocessing import (
    Preprocessor,
    SequentialPreprocessor,
    XarrayMinMaxScaler,
    XarrayStandardScaler,
)
from model import SimpleModel



def make_era5_filename(year):
    return f"era5_single_levels_monthly_{year}.nc"


def make_kernel_filename(year):
    return f"RRTM_kernel_monthly_{year}_alb_TOA_SFC.nc"


# --- map setup functions ---
def setup_global_map() -> Basemap:
    m = Basemap(
        projection="cyl",
        resolution="l",
        llcrnrlat=-90,
        urcrnrlat=90,
        llcrnrlon=0,
        urcrnrlon=360,
    )
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary()
    m.drawparallels(np.arange(-90.0, 91.0, 30.0), labels=[True, False, False, True])
    m.drawmeridians(np.arange(-180.0, 181.0, 60.0), labels=[True, False, False, True])
    return m


def setup_north_pole_map() -> Basemap:
    m = Basemap(
        projection="npstere",
        boundinglat=60,
        lon_0=0,
        resolution="l",
    )
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color="white")
    m.drawparallels(np.arange(60.0, 91.0, 30.0))
    m.drawmeridians(np.arange(0.0, 360.0, 60.0))
    return m


def plot_colormesh_on_map(m, lon, lat, data, cmap, vmin, vmax) -> None:
    #lon_grid, lat_grid = np.meshgrid(np.asarray(lon), np.asarray(lat))
    #m.pcolormesh(lon_grid, lat_grid, np.asarray(data), latlon=True, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

    lon_arr = np.asarray(lon)
    lat_arr = np.asarray(lat)
    data_arr = np.asarray(data)

    # Pre-slice to map bounds — avoids Basemap producing masked/overflow
    # coordinates for out-of-bounds points, which pcolormesh rejects
    lat_mask = (lat_arr >= m.latmin) & (lat_arr <= m.latmax)
    lat_idx = np.where(lat_mask)[0]
    # Add one row of padding on each side so boundary cell quads are complete
    i_min = max(lat_idx.min() - 1, 0)
    i_max = min(lat_idx.max() + 1, len(lat_arr) - 1)

    lat_sub = lat_arr[i_min:i_max + 1]
    data_sub = data_arr[i_min:i_max + 1, :]

    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_sub)
    x, y = m(lon_grid, lat_grid)
    m.pcolormesh(x, y, data_sub, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

def plot_global_field(
    field: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    title: str,
    save_path: Path,
    cmap: str = "RdBu_r",
    vmin: float = -5,
    vmax: float = 5,
    label: str = r"$W/m^2 1\%$",
    annotation: str | None = None,
) -> None:
    fig = plt.figure(figsize=(8, 6), dpi=300)
    m = setup_global_map()
    plot_colormesh_on_map(m, lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax)
    if annotation is not None:
        plt.text(x=300, y=np.max(lat) + 5, s=annotation, fontsize=20)
    plt.colorbar(orientation="horizontal", fraction=0.075, label=label)
    plt.title(title)
    plt.savefig(save_path)
    plt.close(fig)

def plot_north_pole_field(
    field: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    title: str,
    save_path: Path,
    cmap: str = "RdBu_r",
    vmin: float = -5,
    vmax: float = 5,
    label: str = r"$W/m^2 1\%$",
    annotation: str | None = None,
) -> None:
    fig = plt.figure(figsize=(8, 8), dpi=300)
    m = setup_north_pole_map()
    plot_colormesh_on_map(m, lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax)
    if annotation is not None:
        plt.text(x=300, y=np.max(lat) + 5, s=annotation, fontsize=20)
    plt.title(title)
    plt.colorbar(orientation="horizontal", fraction=0.05, pad=0.07, label=label)
    plt.savefig(save_path)
    plt.close(fig)


# --- rrtm kernel loading ---

def load_rrtm_kernel(year, path: Path = Path('data/kernels')) -> xr.Dataset:
    ds = xr.open_dataset(path / make_kernel_filename(year))
    return ds

def interpolate_spatial_field(
    data: np.ndarray,
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    dst_lon: np.ndarray,
    dst_lat: np.ndarray,
) -> np.ndarray:
    dataset = xr.Dataset(
        data_vars={"field": (("latitude", "longitude"), data)},
        coords={"longitude": src_lon, "latitude": src_lat},
    )
    return dataset.interp(longitude=dst_lon, latitude=dst_lat, method="linear")["field"].to_numpy()


# ── Dataset construction & ordering ──────────────────────────────────────────

def dataset_from_array(arr, date, lon, lat) -> xr.Dataset:
    return xr.Dataset(
        data_vars={"tsr": (("date", "latitude", "longitude"), arr)},
        coords={"date": date, "longitude": lon, "latitude": lat},
    )


def ordered_vars(dataset: xr.Dataset, target_var: str, ecod: bool = True) -> list[str]:
    clear_sky_vars = ["hcc", "mcc", "lcc", "tciw", "tclw"]
    if ecod:
        clear_sky_vars += ["ecod", "ecod_fal"]
    clear_sky_vars_set = set(clear_sky_vars)
    other_vars = [
        v for v in dataset.data_vars
        if v != target_var and v != "fal" and v not in clear_sky_vars_set
    ]
    result = ["fal"] + clear_sky_vars + other_vars + [target_var]
    assert set(result) == set(dataset.data_vars), (
        f"ordered_vars is missing or adding vars: "
        f"{set(result).symmetric_difference(set(dataset.data_vars))}"
    )
    return result


def ordered_dataset(ds: xr.Dataset, target_var: str = "tsr", ecod: bool = True) -> xr.Dataset:
    return ds[ordered_vars(ds, target_var, ecod)]


def filter_by_years(ds: xr.Dataset, years, time_coord: str = "date") -> xr.Dataset:
    return ds.sel({time_coord: ds[time_coord].dt.year.isin(years)})


def filter_by_months(
    ds: xr.Dataset,
    months: list[int] | tuple[int, ...] | None,
    time_coord: str = "date",
) -> xr.Dataset:
    if months is None:
        return ds
    if len(months) == 0:
        raise ValueError("months must be non-empty when provided.")
    return ds.sel({time_coord: ds[time_coord].dt.month.isin(months)})



# ── Feature/target extraction & preprocessing helpers ────────────────────────

def preprocessed_feature_target_arrays(
    raw_dataset: xr.Dataset,
    preprocessor: Preprocessor,
    model: SimpleModel,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray, xr.Dataset]:
    preprocessed_dataset = preprocessor.transform(raw_dataset)
    ordered_preprocessed = ordered_dataset(preprocessed_dataset, "tsr")
    feature_names = [v for v in ordered_preprocessed.data_vars if v != "tsr"]
    data = (
        ordered_preprocessed.to_dataarray()
        .transpose("date", "latitude", "longitude", "variable")
        .to_numpy()
    )
    inputs = data[..., : model.input_dim].reshape(-1, model.input_dim)
    targets = data[..., model.input_dim].reshape(-1)
    latitudes = np.broadcast_to(
        ordered_preprocessed["latitude"].to_numpy()[None, :, None],
        data.shape[:3],
    ).reshape(-1)
    return inputs, targets, feature_names, latitudes, ordered_preprocessed


def empirical_copula_values(values: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    ranks = rankdata(values, method="average")
    return (ranks - 0.5) / len(values)


def unpreprocess_feature_values(
    feature_values: np.ndarray,
    feature_name: str,
    preprocessor: Preprocessor,
) -> np.ndarray:
    if isinstance(preprocessor, SequentialPreprocessor):
        values = feature_values.copy()
        for child in reversed(preprocessor.preprocessors):
            if isinstance(child, XarrayMinMaxScaler):
                data_min = child.data_min_[feature_name].to_numpy().item()
                data_max = child.data_max_[feature_name].to_numpy().item()
                denom = data_max - data_min
                if denom == 0:
                    return np.full_like(values, fill_value=data_min, dtype=np.float64)
                values = (values - child.min_val) / (child.max_val - child.min_val) * denom + data_min
            elif isinstance(child, XarrayStandardScaler):
                mean = child.mean_[feature_name].to_numpy().item()
                std = child.std_[feature_name].to_numpy().item()
                values = values * std + mean
        return values
    return feature_values


def inverse_transform_without_upsampling(
    dataset: xr.Dataset,
    preprocessor: Preprocessor,
) -> xr.Dataset:
    if isinstance(preprocessor, SequentialPreprocessor):
        transformed = dataset
        for child in reversed(preprocessor.preprocessors):
            if child.__class__.__name__ == "Downscaler":
                continue
            transformed = child.inverse_transform(transformed)
        return transformed
    return preprocessor.inverse_transform(dataset)
