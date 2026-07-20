from pathlib import Path
from config_utils import variable_config_from_omegaconf, VariableConfig

import numpy as np
import pandas as pd
import xarray as xr
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

from preprocessing import XarrayMinMaxScaler, DianaPreprocessor
from model import SimpleModel
import glob

import torch

SECONDS_PER_DAY = 3600 * 24


def make_era5_filename(year):
    return f"era5_single_levels_monthly_{year}.nc"


def make_kernel_filename(year, var):
    file_var = "ts" if var == "skt" else var
    assert file_var in ["fal", "tcwv", "ts"]
    return f"RRTM_kernel_monthly_{year}_{file_var}_TOA_SFC.nc"


def make_combined_kernel_filename(year):
    return f"RRTM_kernel_monthly_{year}_TOA_SFC.nc"


def make_cloud_profile_filename(year):
    return f"era5_plev_ciwc_clwc_monthly_{year}.nc"


def make_qt_filename(year):
    return f"era5_plev_qt_monthly_{year}.nc"


def generate_paths_yearly(base, years, filename_fn):
    return [Path(base) / filename_fn(year) for year in years]


def load_yearly_and_filter_by_months(path: str, years, months, filename_fn):
    paths = generate_paths_yearly(path, years, filename_fn)
    ds = xr.open_mfdataset(paths, combine="nested", concat_dim="date")
    return filter_by_months(ds, months)


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


def setup_north_pole_map(boundary: float = 60) -> Basemap:
    m = Basemap(
        projection="npstere",
        boundinglat=boundary,
        lon_0=0,
        resolution="l",
    )
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color="white")
    m.drawparallels(np.arange(boundary, 91.0, 30.0))
    m.drawmeridians(np.arange(0.0, 360.0, 60.0))
    return m


def setup_south_pole_map(boundary: float = -60) -> Basemap:
    m = Basemap(
        projection="spstere",
        boundinglat=boundary,
        lon_0=0,
        resolution="l",
    )
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary(fill_color="white")
    m.drawparallels(np.arange(-90.0, boundary + 1, 30.0))
    m.drawmeridians(np.arange(0.0, 360.0, 60.0))
    return m


def plot_colormesh_on_map(m, lon, lat, data, cmap, vmin, vmax) -> None:
    lon_arr = np.asarray(lon)
    lat_arr = np.asarray(lat)
    data_arr = np.asarray(data)

    lat_mask = (lat_arr >= m.latmin) & (lat_arr <= m.latmax)
    lat_idx = np.where(lat_mask)[0]
    # Add one row of padding on each side so boundary cell quads are complete
    i_min = max(lat_idx.min() - 1, 0)
    i_max = min(lat_idx.max() + 1, len(lat_arr) - 1)

    lat_sub = lat_arr[i_min : i_max + 1]
    data_sub = data_arr[i_min : i_max + 1, :]

    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_sub)
    x, y = m(lon_grid, lat_grid)
    return m.pcolormesh(
        x, y, data_sub, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax
    )


def plot_contours_on_map(m, lon, lat, data, cmap, vmin, vmax) -> None:
    # lon_grid, lat_grid = np.meshgrid(np.asarray(lon), np.asarray(lat))
    # m.pcolormesh(lon_grid, lat_grid, np.asarray(data), latlon=True, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)

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

    lat_sub = lat_arr[i_min : i_max + 1]
    data_sub = data_arr[i_min : i_max + 1, :]

    lon_grid, lat_grid = np.meshgrid(lon_arr, lat_sub)
    x, y = m(lon_grid, lat_grid)

    data_range = vmax - vmin
    step = data_range // 12
    if step == 0:
        step = data_range / 12
    true_min = np.min(data).values
    true_max = np.max(data).values
    # print(true_min, true_max)
    levels_up = np.arange(0, true_max - 1 + step, step)
    levels_down = -np.arange(step, -true_min - 1 + step, step)
    levels = list(reversed(list(levels_down))) + list(levels_up)
    # print(levels)

    cs_halo = m.contour(x, y, data_sub, colors="white", linewidths=3.0, levels=levels)
    contours = m.contour(
        x, y, data_sub, colors="#333333", linewidths=1.2, levels=levels
    )
    plt.clabel(contours, inline=True, fmt="%.1f", fontsize=10)
    return contours


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
    contours=False,
) -> None:
    fig = plt.figure(figsize=(8, 6), dpi=300)
    m = setup_global_map()
    artist = plot_colormesh_on_map(m, lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax)
    if contours:
        plot_contours_on_map(m, lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax)
    if annotation is not None:
        plt.text(x=300, y=np.max(lat) + 5, s=annotation, fontsize=20)
    # fig.colorbar(artist, m, orientation="horizontal", fraction=0.075, label=label)
    plt.colorbar(artist, orientation="horizontal", fraction=0.075, label=label)
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
    boundary: float = 60,
    contours=False,
) -> None:
    fig = plt.figure(figsize=(8, 8), dpi=300)
    m = setup_north_pole_map(boundary=boundary)
    artist = plot_colormesh_on_map(m, lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax)
    if contours:
        plot_contours_on_map(m, lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax)
    if annotation is not None:
        plt.text(x=300, y=np.max(lat) + 5, s=annotation, fontsize=20)
    plt.title(title)
    plt.colorbar(artist, orientation="horizontal", fraction=0.05, pad=0.07, label=label)
    plt.savefig(save_path)
    plt.close(fig)


def plot_south_pole_field(
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
    boundary: float = -60,
    contours=False,
) -> None:
    fig = plt.figure(figsize=(8, 8), dpi=300)
    m = setup_south_pole_map(boundary=boundary)
    artist = plot_colormesh_on_map(m, lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax)
    if contours:
        plot_contours_on_map(m, lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax)
    if annotation is not None:
        plt.text(x=300, y=np.max(lat) + 5, s=annotation, fontsize=20)
    plt.title(title)
    plt.colorbar(artist, orientation="horizontal", fraction=0.05, pad=0.07, label=label)
    plt.savefig(save_path)
    plt.close(fig)


def setup_timeseries_plot():
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_xticks(
        pd.date_range(start="2007", end="2017", freq="YS", inclusive="both"),
        np.arange(2007, 2018),
    )
    ax.set_xlabel("date")
    ax.grid(alpha=0.3)
    return fig, ax


# --- rrtm kernel loading ---


def load_rrtm_kernel(year, path: Path = Path("data/kernels")) -> xr.Dataset:
    ds = xr.open_dataset(path / make_kernel_filename(year, "fal"))
    return ds


def load_rrtm_kernel_instant(
    year,
    month,
    day,
    path: Path = Path(
        "/lustre09/project/6003571/hanhuang/kernel_data_latest_from_scratch/spectral_kernel_era5_2015_multiple_profile/instant_TOA_SFC_broadband/sw"
    ),
) -> xr.Dataset:
    dir_no_perturb = path / "no_perturbation"
    dir_perturb = path / "alb"

    path_no_perturb = dir_no_perturb / f"TOA_SFC_all_clr_{month:02d}.nc"
    path_perturb = dir_perturb / f"TOA_SFC_all_clr_{month:02d}.nc"

    d1 = xr.open_dataset(path_no_perturb)
    d2 = xr.open_dataset(path_perturb)

    tstart = 1 + 8 * (day - 1)
    tend = 1 + 8 * (day)

    diff = (
        (d2 - d1)
        .sel(up_down_net=3, drop=True)
        .sel(time=range(tstart, tend))
        .mean(dim="time")
    )
    diff["TOA_cld"] = diff.TOA
    diff = diff.expand_dims(axis=0, dim="date") * 100
    return diff


def load_era5_profile_instant(year, month, day):
    raw_era5_paths = glob.glob(f"data/era5_1hr_point/{year}/{month:02d}/{day:02d}/*.nc")
    raw_dataset = xr.open_mfdataset(raw_era5_paths, combine="nested")
    ds = raw_dataset.rename({"valid_time": "date"})
    ds["tisr"] = ds.tisr * 24
    ds["tsr"] = ds.tsr * 24
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
    return dataset.interp(
        longitude=dst_lon,
        latitude=dst_lat,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )["field"].to_numpy()


# ── Dataset construction & ordering ──────────────────────────────────────────


def kernel_delta(var: str) -> float:
    return 0.01 if var == "fal" else 1.0


def kernel_title(var: str) -> str:
    return {
        "fal": "Surface Albedo",
        "tcwv": "Water Vapor",
        "skt": "Surface Temperature",
        "ts": "Surface Temperature",
        "ecod": "Effective Cloud Optical Depth",
    }[var]


def kernel_label(var: str) -> str:
    return {
        "fal": r"$W/m^2 1\%$",
        "tcwv": r"$W/m^2 kg^{-1} m^2$",
        "skt": r"$W/m^2 K^{-1}$",
        "ts": r"$W/m^2 K^{-1}$",
        "ecod": r"$W/m^2$",
    }[var]


def scale_minmax_value(scaler: XarrayMinMaxScaler, name: str, values):
    denom = scaler.get_data_max()[name] - scaler.get_data_min()[name]
    denom = denom.where(denom != 0, 1.0)
    return ((values - float(scaler.get_data_min()[name])) / float(denom)) * (
        scaler.max_val - scaler.min_val
    ) + scaler.min_val


def load_model_and_preprocessor(
    config, checkpoint_path: Path, downscaling=True, override=None
):
    checkpoint_data = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    epoch = checkpoint_data["epoch"]
    model_config = checkpoint_data.get("config", config)
    model = SimpleModel(model_config)
    model.load_state_dict(checkpoint_data["model_state_dict"])
    model.eval()

    preprocessor = DianaPreprocessor(config, downscaling=downscaling)
    preprocessor.load(model.config.preprocess.params_dir)

    return model, preprocessor, epoch


def nn_pred(
    ds: xr.Dataset,
    model: SimpleModel,
    preprocessor: DianaPreprocessor,
    variable_config: VariableConfig,
    dim_names=["month", "latitude", "longitude"],
) -> xr.DataArray:
    inputs_np = variable_config.inputs_np(ds, dim_names + ["variable"])
    data_torch = torch.from_numpy(inputs_np).float()
    model_outputs = model(data_torch).detach()

    pred = (
        preprocessor.scalar.inverse_transform(
            xr.Dataset(
                data_vars={
                    variable_config.target_var: (dim_names, model_outputs.numpy())
                },
                coords=ds.coords,
            )
        )
        .to_dataarray()
        .squeeze(dim="variable", drop=True)
    )
    return pred


def compute_nn_kernel(
    ds: xr.Dataset,
    preprocessor: DianaPreprocessor,
    model: SimpleModel,
    config,
    clear=False,
    dim_order=["date", "latitude", "longitude"],
    perturbation_var="fal",
):
    vconf = variable_config_from_omegaconf(config)
    ds_base = vconf.clear_sky_input(ds) if clear else ds
    ds_p = ds_base.assign(
        {perturbation_var: ds_base[perturbation_var] + kernel_delta(perturbation_var)}
    )

    pred = nn_pred(
        preprocessor.transform(ds_base), model, preprocessor, config, dim_order
    )
    pred_perturbed = nn_pred(
        preprocessor.transform(ds_p), model, preprocessor, config, dim_order
    )

    kernel = (pred_perturbed - pred) / SECONDS_PER_DAY
    return kernel.to_numpy().squeeze(axis=0), kernel.longitude, kernel.latitude


def compute_nn_kernel_autograd(
    ds: xr.Dataset,
    preprocessor: DianaPreprocessor,
    model: SimpleModel,
    variable_config,
    var="fal",
    clear=False,
    dim_order=["date", "latitude", "longitude", "variable"],
):
    ds_base = variable_config.clear_sky_input(ds) if clear else ds
    processed_ds = preprocessor.transform(ds_base)
    lon = processed_ds.longitude
    lat = processed_ds.latitude

    target = variable_config.target_var
    input_var = var
    feature_names = variable_config.input_order()
    input_idx = feature_names.index(input_var)

    inputs_np = variable_config.inputs_np(processed_ds, dim_order)
    inputs = torch.from_numpy(inputs_np).float().requires_grad_(True)

    outputs = model(inputs)
    grads = torch.autograd.grad(
        outputs=outputs.sum(), inputs=inputs, create_graph=False
    )[0][..., input_idx]

    scaler = preprocessor.scalar
    vmin, vmax = scaler.get_data_min(), scaler.get_data_max()
    target_range = float(vmax[target] - vmin[target])
    input_range = float(vmax[input_var] - vmin[input_var])
    # print(f"{target.upper()} range", target_range)
    # print(f"{input_var} range", input_range)
    grad_physical_per_unit = (
        grads.detach().cpu().numpy().squeeze(axis=0)
        * (target_range / input_range)
        / SECONDS_PER_DAY
    )
    return grad_physical_per_unit * kernel_delta(input_var), lon, lat


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


def to_monthly(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    year = ds.date.dt.year
    month = ds.date.dt.month
    ds = ds.assign_coords(year=("date", year.data), month=("date", month.data))
    return ds.set_index(date=("year", "month")).unstack("date")


def to_dates(ds: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    ds_stacked = ds.stack(date=("year", "month"))
    datetime_index = pd.to_datetime(
        {
            "year": ds_stacked.year.values,
            "month": ds_stacked.month.values,
            "day": 1,
        }
    )
    return (
        ds_stacked.drop_vars(["date", "year", "month"])
        .assign_coords(date=datetime_index)
        .sortby("date")
    )


def global_mean(da: xr.DataArray) -> xr.DataArray:
    weights = np.cos(np.deg2rad(da.latitude))
    weights.name = "weights"
    return da.weighted(weights).mean(dim=["latitude", "longitude"])


def global_date_series(da: xr.DataArray) -> xr.DataArray:
    return global_mean(to_dates(da.copy(deep=True))).compute()


def integrate_over_pressure_levels(
    sp: xr.DataArray,
    da: xr.DataArray,
) -> xr.DataArray:
    p = da.level * 100.0
    return da.where(p <= sp, 0).sum(dim="level")


def weighted_residuals_by_month(
    da: xr.DataArray,
    n_samples: int = 400,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    rng = rng or np.random.default_rng()
    da = da.transpose("year", "month", "latitude", "longitude").compute()

    lat_vals = da.latitude.values
    lat_weights = np.clip(np.cos(np.deg2rad(lat_vals)), 0, None)
    lat_p = lat_weights / lat_weights.sum()

    n_year = da.sizes["year"]
    n_lat = da.sizes["latitude"]
    n_lon = da.sizes["longitude"]

    result = []
    for month in range(1, 13):
        field = da.sel(month=month).values
        year_idx = rng.integers(0, n_year, size=n_samples)
        lat_idx = rng.choice(n_lat, size=n_samples, p=lat_p)
        lon_idx = rng.integers(0, n_lon, size=n_samples)
        result.append(field[year_idx, lat_idx, lon_idx])
    return result


def weighted_residuals_by_year_month(
    da: xr.DataArray,
    n_samples: int = 400,
    rng: np.random.Generator | None = None,
) -> tuple[list[str], list[np.ndarray]]:
    rng = rng or np.random.default_rng()
    da = da.transpose("year", "month", "latitude", "longitude").compute()

    lat_vals = da.latitude.values
    lat_weights = np.clip(np.cos(np.deg2rad(lat_vals)), 0, None)
    lat_p = lat_weights / lat_weights.sum()

    n_lat = da.sizes["latitude"]
    n_lon = da.sizes["longitude"]

    labels = []
    result = []
    for year in da.year.values:
        for month in da.month.values:
            field = da.sel(year=year, month=month).values
            lat_idx = rng.choice(n_lat, size=n_samples, p=lat_p)
            lon_idx = rng.integers(0, n_lon, size=n_samples)
            labels.append(f"{int(year)}-{int(month):02d}")
            result.append(field[lat_idx, lon_idx])
    return labels, result
