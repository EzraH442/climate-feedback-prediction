from pathlib import Path

import numpy as np
import pandas as pd
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
import glob

import torch


SECONDS_PER_DAY = 3600 * 24


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
    boundary: float = 60,
) -> None:
    fig = plt.figure(figsize=(8, 8), dpi=300)
    m = setup_north_pole_map(boundary=boundary)
    plot_colormesh_on_map(m, lon, lat, field, cmap=cmap, vmin=vmin, vmax=vmax)
    if annotation is not None:
        plt.text(x=300, y=np.max(lat) + 5, s=annotation, fontsize=20)
    plt.title(title)
    plt.colorbar(orientation="horizontal", fraction=0.05, pad=0.07, label=label)
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

def load_rrtm_kernel(year, path: Path = Path('data/kernels')) -> xr.Dataset:
    ds = xr.open_dataset(path / make_kernel_filename(year))
    return ds

def load_rrtm_kernel_instant(year, month, day, path: Path = Path('/lustre09/project/6003571/hanhuang/kernel_data_latest_from_scratch/spectral_kernel_era5_2015_multiple_profile/instant_TOA_SFC_broadband/sw')) -> xr.Dataset:
    dir_no_perturb = path / "no_perturbation"
    dir_perturb = path / "alb"
    
    path_no_perturb = dir_no_perturb / f"TOA_SFC_all_clr_{month:02d}.nc"
    path_perturb = dir_perturb / f"TOA_SFC_all_clr_{month:02d}.nc"
    
    d1 = xr.open_dataset(path_no_perturb)
    d2 = xr.open_dataset(path_perturb)

    tstart = 1 + 8 * (day - 1)
    tend   = 1 + 8 * (day)
    
    diff = (d2 - d1).sel(up_down_net=3, drop=True).sel(time=range(tstart, tend)).mean(dim='time')
    diff['TOA_cld'] = diff.TOA
    diff = diff.expand_dims(axis=0, dim="date") * 100
    return diff

def load_era5_profile_instant(year, month, day):
    raw_era5_paths = glob.glob(f"data/era5_1hr_point/{year}/{month:02d}/{day:02d}/*.nc")
    raw_dataset = xr.open_mfdataset(raw_era5_paths, combine="nested")
    ds = raw_dataset.rename({'valid_time': 'date'})
    ds['tisr'] = ds.tisr * 24
    ds['tsr'] = ds.tsr * 24
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

def target_var_from_config(config) -> str:
    return getattr(config.dataset, "target_var", None) or "tsr"


def input_var_from_config(config) -> str:
    return getattr(config.dataset, "input_var", None) or "fal"


def input_vars_from_config(config) -> list[str]:
    return list(getattr(config.dataset, "input_vars", []) or [])


def dataset_from_array(arr, date, lon, lat, target_var: str = "tsr") -> xr.Dataset:
    return xr.Dataset(
        data_vars={target_var: (("date", "latitude", "longitude"), arr)},
        coords={"date": date, "longitude": lon, "latitude": lat},
    )


def ordered_vars(
    dataset: xr.Dataset,
    target_var: str,
    ecod: bool = True,
    input_var: str = "fal",
    input_vars: list[str] | None = None,
) -> list[str]:
    if input_vars:
        result = list(dict.fromkeys([*input_vars, target_var]))
        assert set(result) == set(dataset.data_vars), (
            f"ordered_vars is missing or adding vars: "
            f"{set(result).symmetric_difference(set(dataset.data_vars))}"
        )
        return result

    clear_sky_vars = ["hcc", "mcc", "lcc", "tciw", "tclw"]
    if ecod:
        clear_sky_vars += ["ecod", "ecod_fal"]
    clear_sky_vars_set = set(clear_sky_vars)
    other_vars = [
        v for v in dataset.data_vars
        if v != target_var and v != input_var and v not in clear_sky_vars_set
    ]
    result = [input_var] + clear_sky_vars + other_vars + [target_var]
    assert set(result) == set(dataset.data_vars), (
        f"ordered_vars is missing or adding vars: "
        f"{set(result).symmetric_difference(set(dataset.data_vars))}"
    )
    return result


def ordered_dataset(
    ds: xr.Dataset,
    target_var: str = "tsr",
    ecod: bool = True,
    input_var: str = "fal",
    input_vars: list[str] | None = None,
) -> xr.Dataset:
    return ds[ordered_vars(ds, target_var, ecod, input_var, input_vars)]


def ordered_dataset_for_config(ds: xr.Dataset, config) -> xr.Dataset:
    return ordered_dataset(
        ds,
        target_var=target_var_from_config(config),
        ecod=getattr(config.preprocess, "ecod", True),
        input_var=input_var_from_config(config),
        input_vars=input_vars_from_config(config),
    )


def kernel_delta(config) -> float:
    return 0.01 if input_var_from_config(config) == "fal" else 1.0


def kernel_title(config) -> str:
    return (
        "Surface Albedo"
        if input_var_from_config(config) == "fal"
        else "Surface Temperature"
    )


def kernel_label(config) -> str:
    return (
        r"$W/m^2 1\%$"
        if input_var_from_config(config) == "fal"
        else r"$W/m^2 K^{-1}$"
    )


def clear_sky_raw(ds: xr.Dataset, config) -> xr.Dataset:
    ds_clr = ds.copy(deep=True)
    zero_vars = set(getattr(config.dataset, "clear_sky_zero_vars", []) or [])
    zero_vars.update(["hcc", "mcc", "lcc", "tcc", "tciw", "tclw"])
    for name in zero_vars.intersection(ds_clr.data_vars):
        ds_clr[name][:] = 0
    return ds_clr


def scale_minmax_value(scaler: XarrayMinMaxScaler, name: str, values):
    denom = scaler.data_max_[name] - scaler.data_min_[name]
    denom = denom.where(denom != 0, 1.0)
    return (
        (values - float(scaler.data_min_[name])) / float(denom)
    ) * (scaler.max_val - scaler.min_val) + scaler.min_val


def compute_nn_kernel(
    ds: xr.Dataset,
    preprocessor: Preprocessor,
    model: SimpleModel,
    config,
):
    input_var = input_var_from_config(config)
    target = target_var_from_config(config)
    ds_perturbed = ds.copy(deep=True)
    ds_perturbed[input_var] = ds[input_var] + kernel_delta(config)

    processed_ds = preprocessor.transform(ds)
    processed_ds_perturbed = preprocessor.transform(ds_perturbed)
    date = processed_ds.date
    lon = processed_ds.longitude
    lat = processed_ds.latitude

    def _run_model(dataset):
        data = torch.from_numpy(
            ordered_dataset_for_config(dataset, config)
            .to_dataarray()
            .transpose("date", "latitude", "longitude", "variable")
            .to_numpy()
        ).float()
        return model(data[..., : model.input_dim]).detach()

    def _invert(output):
        return (
            preprocessor.preprocessors[-1]
            .inverse_transform(
                dataset_from_array(
                    arr=output.numpy(),
                    date=date,
                    lon=lon,
                    lat=lat,
                    target_var=target,
                )
            )
            .to_dataarray()
            .squeeze(dim=["variable", "date"])
            .to_numpy()
            / SECONDS_PER_DAY
        )

    return (
        _invert(_run_model(processed_ds_perturbed)) - _invert(_run_model(processed_ds)),
        lon,
        lat,
    )


def compute_nn_kernel_autograd(
    ds: xr.Dataset,
    preprocessor: Preprocessor,
    model: SimpleModel,
    config,
):
    assert isinstance(preprocessor, SequentialPreprocessor)
    scaler = preprocessor.preprocessors[-1]
    assert isinstance(scaler, XarrayMinMaxScaler)

    processed_ds = preprocessor.transform(ds)
    lon = processed_ds.longitude
    lat = processed_ds.latitude

    target = target_var_from_config(config)
    input_var = input_var_from_config(config)

    ordered = ordered_dataset_for_config(processed_ds, config)
    feature_names = [v for v in ordered.data_vars if v != target]
    input_idx = feature_names.index(input_var)

    data = torch.from_numpy(
        ordered.to_dataarray()
        .transpose("date", "latitude", "longitude", "variable")
        .to_numpy()
    ).float()
    inputs = data[..., : model.input_dim].requires_grad_(True)

    outputs = model(inputs)
    grads = torch.autograd.grad(
        outputs=outputs.sum(),
        inputs=inputs,
        create_graph=False,
    )[0][..., input_idx]

    target_range = float(scaler.data_max_[target] - scaler.data_min_[target])
    input_range = float(scaler.data_max_[input_var] - scaler.data_min_[input_var])
    print(f"{target.upper()} range", target_range)
    print(f"{input_var} range", input_range)
    grad_physical_per_unit = grads.detach().cpu().numpy().squeeze(axis=0) * (
        target_range / input_range
    ) / SECONDS_PER_DAY
    return grad_physical_per_unit * kernel_delta(config), lon, lat


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


def nn_pred(
    ds: xr.Dataset,
    model: SimpleModel,
    preprocessor: Preprocessor,
    config,
) -> xr.DataArray:
    year, month = ds.year.values, ds.month.values
    lon, lat = ds.longitude.values, ds.latitude.values
    target_var = target_var_from_config(config)

    ds_ordered_np = (
        ordered_dataset_for_config(ds, config)
        .to_dataarray()
        .transpose("year", "month", "latitude", "longitude", "variable")
        .to_numpy()
    )

    data_torch = torch.from_numpy(ds_ordered_np).float()
    model_outputs = model(data_torch[..., : model.input_dim]).detach()

    pred = (
        preprocessor.preprocessors[-1]
        .inverse_transform(
            xr.Dataset(
                data_vars={
                    target_var: (
                        ("year", "month", "latitude", "longitude"),
                        model_outputs.numpy(),
                    )
                },
                coords={
                    "year": year,
                    "month": month,
                    "longitude": lon,
                    "latitude": lat,
                },
            )
        )
        .to_dataarray()
        .squeeze(dim="variable", drop=True)
    )
    return pred


def nn_radiative_response(
    ds: xr.Dataset,
    anomaly: xr.Dataset,
    model: SimpleModel,
    preprocessor: Preprocessor,
    config,
    variables,
) -> list[xr.DataArray]:
    ds_copies = []
    for var_list in variables:
        if not isinstance(var_list, list):
            var_list = [var_list]
        ds_copies.append(ds.assign({v: ds[v] + anomaly[v] for v in var_list}))

    ds_original = preprocessor.transform(ds)
    ds_perturbed = [preprocessor.transform(d) for d in ds_copies]

    pred_original = nn_pred(ds_original, model, preprocessor, config)
    return [
        (nn_pred(ds_p, model, preprocessor, config) - pred_original)
        / SECONDS_PER_DAY
        for ds_p in ds_perturbed
    ]


def nn_radiative_response_cross(
    ds: xr.Dataset,
    anomaly: xr.Dataset,
    model: SimpleModel,
    preprocessor: Preprocessor,
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
    pred_original = nn_pred(ds_original, model, preprocessor, config)

    preds = {}
    for var_list in all_var_lists:
        ds_perturbed = preprocessor.transform(
            ds.assign({v: ds[v] + anomaly[v] for v in var_list})
        )
        preds[var_list] = nn_pred(ds_perturbed, model, preprocessor, config)

    results = []
    for vi, vj in variable_pairs:
        if not isinstance(vi, list):
            vi = [vi]
        if not isinstance(vj, list):
            vj = [vj]
        ki, kj, kij = tuple(vi), tuple(vj), tuple(vi + vj)
        results.append((preds[kij] - preds[ki] - preds[kj] + pred_original) / SECONDS_PER_DAY)

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
        valid = np.isfinite(flat_res)
        p[~valid] = 0.0
        p = p / p.sum()

        indices = rng.choice(len(flat_res), size=n_samples, p=p)
        result.append(flat_res[indices])
    return result



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
