from pathlib import Path
from config_utils import variable_config_from_omegaconf

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
    assert var in ["fal", "tcwv", "ts"]
    return f"RRTM_kernel_monthly_{year}_{var}_TOA_SFC.nc"


def make_combined_kernel_filename(year):
    return f"RRTM_kernel_monthly_{year}_TOA_SFC.nc"


def make_cloud_profile_filename(year):
    return f"era5_plev_ciwc_clwc_monthly_{year}.nc"


def make_qt_filename(year):
    return f"era5_plev_qt_monthly_{year}.nc"


def make_ecod_filename(year, fast_ecod=False):
    prefix = "era5_fast_ecod" if fast_ecod else "era5_ecod"
    return f"{prefix}_monthly_{year}.nc"


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
    print(true_min, true_max)
    levels_up = np.arange(0, true_max - 1 + step, step)
    levels_down = -np.arange(step, -true_min - 1 + step, step)
    levels = list(reversed(list(levels_down))) + list(levels_up)
    print(levels)

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
    return dataset.interp(longitude=dst_lon, latitude=dst_lat, method="linear", kwargs={"fill_value": "extrapolate"})[
        "field"
    ].to_numpy()


# ── Dataset construction & ordering ──────────────────────────────────────────


def kernel_delta(var: str) -> float:
    return 0.01 if var == "fal" else 1.0


def kernel_title(var: str) -> str:
    return "Surface Albedo" if var == "fal" else "Surface Temperature"


def kernel_label(var: str) -> str:
    return r"$W/m^2 1\%$" if var == "fal" else r"$W/m^2 K^{-1}$"


def scale_minmax_value(scaler: XarrayMinMaxScaler, name: str, values):
    denom = scaler.get_data_max()[name] - scaler.get_data_min()[name]
    denom = denom.where(denom != 0, 1.0)
    return ((values - float(scaler.get_data_min()[name])) / float(denom)) * (
        scaler.max_val - scaler.min_val
    ) + scaler.min_val


def load_model_and_preprocessor(config, checkpoint_path: Path, downscaling=True, override=None):
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
    config,
    dim_names=["month", "latitude", "longitude"],
    clear=False,
) -> xr.DataArray:
    target_var = config.dataset.target_var
    variable_config = variable_config_from_omegaconf(config)

    inputs_np = variable_config.inputs_np(ds, dim_names + ["variable"], clear)
    data_torch = torch.from_numpy(inputs_np).float()
    model_outputs = model(data_torch).detach()

    pred = (
        preprocessor.scalar.inverse_transform(
            xr.Dataset(
                data_vars={target_var: (dim_names, model_outputs.numpy())},
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
    ds_p = ds.assign(
        {perturbation_var: ds[perturbation_var] + kernel_delta(perturbation_var)}
    )

    pred = nn_pred(preprocessor.transform(ds), model, preprocessor, config, dim_order, clear)
    pred_perturbed = nn_pred(preprocessor.transform(ds_p), model, preprocessor, config, dim_order, clear)

    kernel = ((pred_perturbed - pred) / SECONDS_PER_DAY)
    return kernel.to_numpy().squeeze(axis=0), kernel.longitude, kernel.latitude


def compute_nn_kernel_autograd(
    ds: xr.Dataset,
    preprocessor: DianaPreprocessor,
    model: SimpleModel,
    config,
    var="fal",
    clear=False,
    dim_order=["date", "latitude", "longitude", "variable"],
):
    processed_ds = preprocessor.transform(ds)
    lon = processed_ds.longitude
    lat = processed_ds.latitude

    vconf = variable_config_from_omegaconf(config)
    target = vconf.target_var
    input_var = var
    feature_names = vconf.input_order()
    input_idx = feature_names.index(input_var)

    inputs_np = vconf.inputs_np(processed_ds, dim_order, clear)
    inputs = torch.from_numpy(inputs_np).float().requires_grad_(True)

    outputs = model(inputs)
    grads = torch.autograd.grad(
        outputs=outputs.sum(), inputs=inputs, create_graph=False
    )[0][..., input_idx]

    scaler = preprocessor.scalar
    vmin, vmax = scaler.get_data_min(), scaler.get_data_max()
    target_range = float(vmax[target] - vmin[target])
    input_range = float(vmax[input_var] - vmin[input_var])
    print(f"{target.upper()} range", target_range)
    print(f"{input_var} range", input_range)
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


def nn_radiative_response(
    ds: xr.Dataset,
    anomaly: xr.Dataset,
    model: SimpleModel,
    preprocessor: DianaPreprocessor,
    config,
    variables,
    clear=False
) -> list[xr.DataArray]:
    datasets_to_test = []
    for var_list in variables:
        if not isinstance(var_list, list):
            var_list = [var_list]

        modified = ds.assign({v: ds[v] + anomaly[v] for v in var_list})
        datasets_to_test.append(modified)
    datasets_to_test.append(ds + anomaly)

    ds_original = ds
    ds_stacked = xr.concat(datasets_to_test, dim="run")

    dim_names_original = ["month", "latitude", "longitude"]
    dim_names = ["run", "year", "month", "latitude", "longitude"]

    pred_original = nn_pred(
        preprocessor.transform(ds_original), model, preprocessor, config, dim_names=dim_names_original, clear=clear
    )
    pred_perturbed = nn_pred(
        preprocessor.transform(ds_stacked), model, preprocessor, config, dim_names=dim_names, clear=clear
    )

    results = (pred_perturbed - pred_original) / SECONDS_PER_DAY
    return [results.isel(run=i) for i in range(0, len(variables) + 1)]


def nn_radiative_response_cross(
    ds: xr.Dataset,
    anomaly: xr.Dataset,
    model: SimpleModel,
    preprocessor: DianaPreprocessor,
    config,
    variable_pairs,
) -> list[xr.DataArray]:
    all_var_lists = set()
    for vi, vj in variable_pairs:
        if not isinstance(vi, list):
            vi = [vi]
        if not isinstance(vj, list):
            vj = [vj]
        all_var_lists.add(tuple(vi))
        all_var_lists.add(tuple(vj))
        all_var_lists.add(tuple(vi + vj))

    ds_original = preprocessor.transform(ds)

    dim_names_original = ["month", "latitude", "longitude"]
    dim_names = ["year", "month", "latitude", "longitude"]

    pred_original = nn_pred(
        ds_original, model, preprocessor, config, dim_names=dim_names_original
    )

    preds = {}
    for var_list in all_var_lists:
        ds_perturbed = preprocessor.transform(
            ds.assign({v: ds[v] + anomaly[v] for v in var_list})
        )
        preds[var_list] = nn_pred(
            ds_perturbed, model, preprocessor, config, dim_names=dim_names
        )

    results = []
    for vi, vj in variable_pairs:
        if not isinstance(vi, list):
            vi = [vi]
        if not isinstance(vj, list):
            vj = [vj]
        ki, kj, kij = tuple(vi), tuple(vj), tuple(vi + vj)
        results.append(
            (preds[kij] - preds[ki] - preds[kj] + pred_original) / SECONDS_PER_DAY
        )

    return results


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
    n_samples: int = 2000,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    rng = rng or np.random.default_rng()
    result = []
    for month in range(1, 13):
        field = (
            da.sel(month=month)
            .transpose("year", "latitude", "longitude")
            .compute()
            .values
        )
        n_year, n_lat, n_lon = field.shape

        lat_vals = da.latitude.values
        weights = np.clip(np.cos(np.deg2rad(lat_vals)), 0, None)
        weights_3d = weights[None, :, None] * np.ones((n_year, n_lat, n_lon))

        flat_res = field.ravel()
        p = weights_3d.ravel().copy()
        p = p / p.sum()

        indices = rng.choice(len(flat_res), size=n_samples, p=p)
        result.append(flat_res[indices])
    return result


def weighted_residuals_by_year_month(
    da: xr.DataArray,
    n_samples: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[list[str], list[np.ndarray]]:
    rng = rng or np.random.default_rng()
    da = da.transpose("year", "month", "latitude", "longitude")
    lat_vals = da.latitude.values
    # print(lat_vals)
    weights = np.clip(np.cos(np.deg2rad(lat_vals)), 0, None)
    labels = []
    result = []

    for year in da.year.values:
        for month in da.month.values:
            field = da.sel(year=year, month=month).compute().values
            n_lat, n_lon = field.shape
            weights_2d = weights[:, None] * np.ones((n_lat, n_lon))
            # print(weights_2d)

            flat_res = field.ravel()
            p = weights_2d.ravel().copy()
            p = p / p.sum()
            # print(p)
            indices = rng.choice(len(flat_res), size=n_samples, p=p)
            labels.append(f"{int(year)}-{int(month):02d}")
            result.append(flat_res[indices])

    return labels, result


def empirical_copula_values(values: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    ranks = rankdata(values, method="average")
    return (ranks - 0.5) / len(values)
