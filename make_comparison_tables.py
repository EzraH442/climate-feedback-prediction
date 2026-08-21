import argparse
import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from config_utils import load_config, variable_config_from_omegaconf
from eval import compute_nn_kernel_autograd, nn_pred
from preprocess import load_ecod
from utils import (
    SECONDS_PER_DAY,
    generate_paths_yearly,
    global_mean,
    kernel_delta,
    load_model_and_preprocessor,
    make_era5_filename,
    make_kernel_filename,
    plot_colormesh_on_map,
    setup_global_map,
)


NORTH_BOUNDARY = 75

MODELS = {
    "sob_fal": "configs/model/fal/2011-2014_3,6,9,12_sob_fal.yaml",
    "sob_fal_noozone": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_noozone.yaml",
    "sob_fal_clear": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky.yaml",
    "sob_fal_clear_noozone": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky_noozone.yaml",
    "sob_fal_fast_ecod": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_fast-ecod.yaml",
    "sob_fal_fast_ecod_noozone": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_fast-ecod_noozone.yaml",
    "sob_fal_clear_fast_ecod": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky_fast-ecod.yaml",
    "sob_fal_tcwv": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_tcwv.yaml",
    "sob_fal_tcwv_noozone": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_tcwv_noozone.yaml",
    "sob_fal_tcwv_clear_noozone": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_tcwv_clearsky_noozone.yaml",
}

COMPARISONS = {
    "noozone": [
        ("sob_fal -> sob_fal_noozone", "sob_fal", "sob_fal_noozone"),
        ("sob_fal_clear -> sob_fal_clear_noozone", "sob_fal_clear", "sob_fal_clear_noozone"),
        ("sob_fal_fast_ecod -> sob_fal_fast_ecod_noozone", "sob_fal_fast_ecod", "sob_fal_fast_ecod_noozone"),
    ],
    "fast_vs_slow_ecod": [
        ("sob_fal -> sob_fal_fast_ecod", "sob_fal", "sob_fal_fast_ecod"),
        ("sob_fal_clear -> sob_fal_clear_fast_ecod", "sob_fal_clear", "sob_fal_clear_fast_ecod"),
        ("sob_fal_noozone -> sob_fal_fast_ecod_noozone", "sob_fal_noozone", "sob_fal_fast_ecod_noozone"),
    ],
    "clearsky_vs_nonclearsky": [
        ("sob_fal -> sob_fal_clear", "sob_fal", "sob_fal_clear"),
        ("sob_fal_fast_ecod -> sob_fal_clear_fast_ecod", "sob_fal_fast_ecod", "sob_fal_clear_fast_ecod"),
        ("sob_fal_noozone -> sob_fal_clear_noozone", "sob_fal_noozone", "sob_fal_clear_noozone"),
    ],
    "tcwv": [
        ("sob_fal -> sob_fal_tcwv", "sob_fal", "sob_fal_tcwv"),
        ("sob_fal_noozone -> sob_fal_tcwv_noozone", "sob_fal_noozone", "sob_fal_tcwv_noozone"),
        ("sob_fal_clear_noozone -> sob_fal_tcwv_clear_noozone", "sob_fal_clear_noozone", "sob_fal_tcwv_clear_noozone"),
    ],
}

ECOD_VALIDATION_MODELS = {
    "baseline": "configs/model/fal/2011-2014_3,6,9,12_baseline.yaml",
    "baseline_true_tcc": "configs/model/fal/2011-2014_3,6,9,12_baseline_true-tcc-ecod.yaml",
    "sob_fal": "configs/model/fal/2011-2014_3,6,9,12_sob_fal.yaml",
    "sob_fal_fast_ecod": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_fast-ecod.yaml",
    "sob_fal_true_tcc": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_true-tcc-ecod.yaml",
    "sob_fal_clear": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky.yaml",
    "sob_fal_clear_fast_ecod": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky_fast-ecod.yaml",
    "sob_fal_clear_true_tcc": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky_true-tcc-ecod.yaml",
    "sob_fal_tcwv": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_tcwv.yaml",
    "sob_fal_tcwv_true_tcc": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_tcwv_true-tcc-ecod.yaml",
    "sob_fal_noozone": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_noozone.yaml",
    "sob_fal_noozone_fast_ecod": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_fast-ecod_noozone.yaml",
    "sob_fal_noozone_true_tcc": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_noozone_true-tcc-ecod.yaml",
}

ECOD_VALIDATION_COMPARISONS = {
    "true_tcc_vs_regular_ecod": [
        ("baseline -> baseline_true_tcc", "baseline", "baseline_true_tcc"),
        ("sob_fal -> sob_fal_true_tcc", "sob_fal", "sob_fal_true_tcc"),
        ("sob_fal_clear -> sob_fal_clear_true_tcc", "sob_fal_clear", "sob_fal_clear_true_tcc"),
        ("sob_fal_tcwv -> sob_fal_tcwv_true_tcc", "sob_fal_tcwv", "sob_fal_tcwv_true_tcc"),
        ("sob_fal_noozone -> sob_fal_noozone_true_tcc", "sob_fal_noozone", "sob_fal_noozone_true_tcc"),
    ],
    "true_tcc_vs_fast_ecod": [
        ("sob_fal_fast_ecod -> sob_fal_true_tcc", "sob_fal_fast_ecod", "sob_fal_true_tcc"),
        ("sob_fal_clear_fast_ecod -> sob_fal_clear_true_tcc", "sob_fal_clear_fast_ecod", "sob_fal_clear_true_tcc"),
        ("sob_fal_noozone_fast_ecod -> sob_fal_noozone_true_tcc", "sob_fal_noozone_fast_ecod", "sob_fal_noozone_true_tcc"),
    ],
}

OZONE_VALIDATION_MODELS = {
    "sob_fal": "configs/model/fal/2011-2014_3,6,9,12_sob_fal.yaml",
    "sob_fal_noozone": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_noozone.yaml",
    "sob_fal_clear": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky.yaml",
    "sob_fal_clear_noozone": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky_noozone.yaml",
    "sob_fal_tcwv": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_tcwv.yaml",
    "sob_fal_tcwv_noozone": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_tcwv_noozone.yaml",
    "sob_fal_tcwv_clear_noozone": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_tcwv_clearsky_noozone.yaml",
    "sob_fal_true_tcc": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_true-tcc-ecod.yaml",
    "sob_fal_noozone_true_tcc": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_noozone_true-tcc-ecod.yaml",
    "sob_fal_clear_true_tcc": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky_true-tcc-ecod.yaml",
    "sob_fal_clear_noozone_true_tcc": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky_noozone_true-tcc-ecod.yaml",
}

OZONE_VALIDATION_COMPARISONS = {
    "ozone_vs_noozone": [
        ("sob_fal -> sob_fal_noozone", "sob_fal", "sob_fal_noozone"),
        ("sob_fal_clear -> sob_fal_clear_noozone", "sob_fal_clear", "sob_fal_clear_noozone"),
        ("sob_fal_tcwv -> sob_fal_tcwv_noozone", "sob_fal_tcwv", "sob_fal_tcwv_noozone"),
        ("sob_fal_true_tcc -> sob_fal_noozone_true_tcc", "sob_fal_true_tcc", "sob_fal_noozone_true_tcc"),
        ("sob_fal_clear_true_tcc -> sob_fal_clear_noozone_true_tcc", "sob_fal_clear_true_tcc", "sob_fal_clear_noozone_true_tcc"),
    ],
    "tcwv_noozone": [
        ("sob_fal_clear_noozone -> sob_fal_tcwv_clear_noozone", "sob_fal_clear_noozone", "sob_fal_tcwv_clear_noozone"),
    ],
}

SETUPS = {
    "default": (MODELS, COMPARISONS, Path("analysis_tables")),
    "ecod_validation": (
        ECOD_VALIDATION_MODELS,
        ECOD_VALIDATION_COMPARISONS,
        Path("ecod_validation"),
    ),
    "ozone_validation": (
        OZONE_VALIDATION_MODELS,
        OZONE_VALIDATION_COMPARISONS,
        Path("ozone_validation"),
    ),
}


def metric(diff):
    mbe = float(np.mean(global_mean(diff)))
    results = {
        "mbe": mbe, 
        "abs_mbe": abs(mbe), 
        "rmse": float(np.sqrt(np.mean(global_mean(diff**2))))
    }
    return results


def correlation(a, b):
    a_values, b_values = xr.align(a, b, join="inner")
    a_flat = np.asarray(a_values).ravel()
    b_flat = np.asarray(b_values).ravel()
    mask = np.isfinite(a_flat) & np.isfinite(b_flat)
    if mask.sum() < 2:
        return {"corr": float("nan"), "n": int(mask.sum())}
    return {
        "corr": float(np.corrcoef(a_flat[mask], b_flat[mask])[0, 1]),
        "n": int(mask.sum()),
    }


def task_slug(name):
    return name.replace("/", "_").replace(" ", "_").replace(",", "").lower()


def write_tables(cache, output_dir, comparisons=COMPARISONS):
    output_dir.mkdir(parents=True, exist_ok=True)
    for group, group_comparisons in comparisons.items():
        task_names = sorted({
            task
            for model in cache.values()
            for task, values in model.items()
            if isinstance(values, dict) and {"mbe", "abs_mbe", "rmse"} <= set(values)
        })
        for task in task_names:
            path = output_dir / f"{group}__{task_slug(task)}.csv"
            with path.open("w", newline="", encoding="ascii") as f:
                writer = csv.writer(f)
                writer.writerow(["comparison", "MBE change", "|MBE| change", "RMSE change"])
                for label, src, dst in group_comparisons:
                    old = cache.get(src, {}).get(task, {})
                    new = cache.get(dst, {}).get(task, {})
                    row = [label]
                    for key in ("mbe", "abs_mbe", "rmse"):
                        row.append(new[key] - old[key] if key in new and key in old else "")
                    writer.writerow(row)


def write_tsr_field_correlations(cache, output_dir):
    path = output_dir / "tsr_field_correlations.csv"
    with path.open("w", newline="", encoding="ascii") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model",
            "field",
            "tsr_residual_field_corr",
            "tsr_residual_field_corr_n",
            "tsr_field_corr",
            "tsr_field_corr_n",
            "tsrc_residual_field_corr",
            "tsrc_residual_field_corr_n",
            "tsrc_field_corr",
            "tsrc_field_corr_n",
        ])
        for model_name in sorted(cache):
            values = cache[model_name].get("tsr", {})
            for field in ("tco3", "tcwv"):
                if (
                    f"tsr_residual_{field}_corr" not in values
                    and f"tsr_{field}_corr" not in values
                    and f"tsrc_residual_{field}_corr" not in values
                    and f"tsrc_{field}_corr" not in values
                ):
                    continue
                writer.writerow([
                    model_name,
                    field,
                    values.get(f"tsr_residual_{field}_corr", ""),
                    values.get(f"tsr_residual_{field}_corr_n", ""),
                    values.get(f"tsr_{field}_corr", ""),
                    values.get(f"tsr_{field}_corr_n", ""),
                    values.get(f"tsrc_residual_{field}_corr", ""),
                    values.get(f"tsrc_residual_{field}_corr_n", ""),
                    values.get(f"tsrc_{field}_corr", ""),
                    values.get(f"tsrc_{field}_corr_n", ""),
                ])


def write_tsr_field_scatter_plots(output_dir, manifest_path, max_points=200000):
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    plot_dir = output_dir / "field_scatter_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    field_labels = {"tco3": "ozone", "tcwv": "tcwv"}
    y_labels = {
        "tsr_true": "tsr",
        "tsr_diff": "tsr_residual",
        "tsrc_true": "tsrc",
        "tsrc_diff": "tsrc_residual",
    }
    for model_name, path in manifest.items():
        if not Path(path).exists():
            continue
        ds = xr.load_dataset(path)
        try:
            for field in ("tco3", "tcwv"):
                field_var = f"{field}_field"
                if field_var not in ds:
                    continue
                for y_name in ("tsr_true", "tsr_diff", "tsrc_true", "tsrc_diff"):
                    if y_name not in ds:
                        continue
                    field_label = field_labels[field]
                    y_label = y_labels[y_name]
                    plot_scatter(
                        ds[field_var],
                        ds[y_name],
                        plot_dir / f"{model_name}__{field_label}_vs_{y_label}.png",
                        f"{model_name}: {field_label} vs {y_label}",
                        field_label,
                        y_label,
                        max_points,
                    )
                if field == "tco3":
                    date_dim = next(
                        (dim for dim in ds[field_var].dims if dim.endswith("date")),
                        None,
                    )
                    if date_dim is None:
                        continue
                    delta_ozone = ds[field_var] - ds[field_var].mean(date_dim)
                    for y_name in ("tsr_diff", "tsrc_diff"):
                        if y_name not in ds:
                            continue
                        y_label = y_labels[y_name]
                        plot_scatter(
                            delta_ozone,
                            ds[y_name],
                            plot_dir / f"{model_name}__delta_ozone_vs_{y_label}.png",
                            f"{model_name}: delta ozone vs {y_label}",
                            "delta ozone",
                            y_label,
                            max_points,
                        )
        finally:
            ds.close()


def plot_scatter(x, y, save_path, title, x_label, y_label, max_points):
    x_values, y_values = xr.align(x, y, join="inner")
    x_flat = np.asarray(x_values).ravel()
    y_flat = np.asarray(y_values).ravel()
    mask = np.isfinite(x_flat) & np.isfinite(y_flat)
    x_flat = x_flat[mask]
    y_flat = y_flat[mask]
    if x_flat.size > max_points:
        idx = np.linspace(0, x_flat.size - 1, max_points, dtype=int)
        x_flat = x_flat[idx]
        y_flat = y_flat[idx]
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    ax.scatter(x_flat, y_flat, s=1, alpha=0.15, linewidths=0)
    if x_flat.size >= 2 and np.ptp(x_flat) > 0 and np.ptp(y_flat) > 0:
        slope, intercept = np.polyfit(x_flat, y_flat, 1)
        x_line = np.array([float(np.min(x_flat)), float(np.max(x_flat))])
        ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1)
        r = float(np.corrcoef(x_flat, y_flat)[0, 1])
        ax.text(
            0.05,
            0.95,
            f"y = {slope:.3g}x {intercept:+.3g}\nR = {r:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def metric_tasks(cache):
    return sorted({
        task
        for model in cache.values()
        for task, values in model.items()
        if isinstance(values, dict) and {"mbe", "abs_mbe", "rmse"} <= set(values)
    })


def var_name(name):
    return name.replace("/", "__").replace(" ", "_").replace(",", "").replace("-", "_")


def cached_residual_name(task):
    if task == "tsr":
        return "tsr_diff"
    suffix = "diff" if task.startswith("kernel/") else "residual"
    return f"{var_name(task)}__{suffix}"


def store_prediction(predictions, name, data):
    da = data if isinstance(data, xr.DataArray) else xr.DataArray(data)
    da = da.astype("float64")
    da = da.reset_coords(drop=True)
    predictions[name] = da.rename({dim: f"{name}__{dim}" for dim in da.dims})


def spatial_field(da):
    lat_dim = next((dim for dim in da.dims if dim.endswith("latitude")), None)
    lon_dim = next((dim for dim in da.dims if dim.endswith("longitude")), None)
    if lat_dim is None or lon_dim is None:
        return None
    reduce_dims = [dim for dim in da.dims if dim not in (lat_dim, lon_dim)]
    if reduce_dims:
        da = da.mean(reduce_dims)
    da = da.rename({
        dim: name
        for dim, name in ((lat_dim, "latitude"), (lon_dim, "longitude"))
        if dim != name
    })
    return da, da.longitude.values, da.latitude.values


def normalize_spatial_dims(da):
    lat_dim = next(dim for dim in da.dims if dim.endswith("latitude"))
    lon_dim = next(dim for dim in da.dims if dim.endswith("longitude"))
    return da.rename({
        dim: name
        for dim, name in ((lat_dim, "latitude"), (lon_dim, "longitude"))
        if dim != name
    })


def north_pole_cache(cache, manifest_path, boundary):
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    out = {}
    for model_name, path in manifest.items():
        if not Path(path).exists():
            continue
        ds = xr.load_dataset(path)
        try:
            model_metrics = {}
            for task in cache.get(model_name, {}):
                var = cached_residual_name(task)
                if var not in ds:
                    continue
                da = normalize_spatial_dims(ds[var]).sel(latitude=slice(boundary, None))
                model_metrics[task] = metric(da)
            out[model_name] = model_metrics
        finally:
            ds.close()
    return out


def write_residual_plots(cache, output_dir, manifest_path, comparisons=COMPARISONS):
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text(encoding="ascii"))
    datasets = {
        name: xr.load_dataset(path)
        for name, path in manifest.items()
        if Path(path).exists()
    }
    plot_dir = output_dir / "residual_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    try:
        for group, group_comparisons in comparisons.items():
            for task in metric_tasks(cache):
                var = cached_residual_name(task)
                rows = []
                for label, src, dst in group_comparisons:
                    if src not in datasets or dst not in datasets:
                        continue
                    if var not in datasets[src] or var not in datasets[dst]:
                        continue
                    old = spatial_field(datasets[src][var])
                    new = spatial_field(datasets[dst][var])
                    if old is None or new is None:
                        continue
                    old_da, lon, lat = old
                    new_da, _, _ = new
                    rows.append((label, lon, lat, old_da, new_da, new_da - old_da))
                if rows:
                    plot_residual_grid(rows, plot_dir / f"{group}__{task_slug(task)}.png", f"{group}: {task}")
    finally:
        for ds in datasets.values():
            ds.close()


def plot_residual_grid(rows, save_path, title):
    values = [np.asarray(field) for row in rows for field in row[3:]]
    vmax = max(float(np.max(np.abs(value))) for value in values)
    vmax = vmax if vmax > 0 else 1.0
    fig = plt.figure(figsize=(12, 3.4 * len(rows)), dpi=200)
    last_artist = None
    for row_idx, (label, lon, lat, old, new, change) in enumerate(rows):
        for col_idx, (col_title, field) in enumerate(
            [("before", old), ("after", new), ("change", change)]
        ):
            ax = fig.add_subplot(len(rows), 3, row_idx * 3 + col_idx + 1)
            plt.sca(ax)
            m = setup_global_map()
            last_artist = plot_colormesh_on_map(
                m, lon, lat, field, cmap="RdBu_r", vmin=-vmax, vmax=vmax
            )
            stats = metric(field)
            ax.text(
                0.02,
                0.04,
                f"MBE {stats['mbe']:.2f}\nRMSE {stats['rmse']:.2f}",
                transform=ax.transAxes,
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
            )
            ax.set_title(f"{label}\n{col_title}", fontsize=9)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    cbar_ax = fig.add_axes([0.2, 0.025, 0.6, 0.02])
    fig.colorbar(last_artist, cax=cbar_ax, orientation="horizontal", label="residual")
    fig.savefig(save_path)
    plt.close(fig)


def load_processed_dataset(config):
    paths = generate_paths_yearly(config.dataset.era5.path, config.dataset.test_years, make_era5_filename)
    return xr.open_mfdataset(paths, combine="nested", concat_dim="date")


def load_raw_test_dataset(config):
    paths = generate_paths_yearly(config.dataset.era5.raw_path, config.dataset.test_years, make_era5_filename)
    ds = xr.open_mfdataset(paths, combine="nested", concat_dim="date")
    if config.preprocess.ecod.enabled:
        ds = ds.assign(
            ecod=load_ecod(
                config.dataset.era5.raw_path,
                config.dataset.test_years,
                range(1, 13),
                config.preprocess.ecod.method,
            )
        )
    return ds


def raw_field_on_grid(raw, field, target):
    da = raw[field].sel(date=target.date)
    return da.interp(latitude=target.latitude, longitude=target.longitude)


def tsr_metric(config, model, preprocessor, predictions):
    ds = load_processed_dataset(config)
    raw = None
    try:
        pred = nn_pred(ds, model, preprocessor, config, ["date", "latitude", "longitude"])
        true = preprocessor.preprocessors[-1].inverse_transform(ds)[config.dataset.target_var]
        pred = pred / SECONDS_PER_DAY
        true = true / SECONDS_PER_DAY
        diff = pred - true
        store_prediction(predictions, "tsr_pred", pred)
        store_prediction(predictions, "tsr_true", true)
        store_prediction(predictions, "tsr_diff", diff)
        results = metric(diff)
        physical_ds = preprocessor.preprocessors[-1].inverse_transform(ds)
        clear_true = None
        clear_diff = None
        clear_var = config.dataset.clear_sky.var
        if clear_var in physical_ds:
            clear_true = physical_ds[clear_var] / SECONDS_PER_DAY
            if config.dataset.clear_sky.enabled:
                raw = load_raw_test_dataset(config)
                vconf = variable_config_from_omegaconf(config)
                clear_processed = preprocessor.transform(vconf.clear_sky_input(raw))
                clear_pred = nn_pred(
                    clear_processed,
                    model,
                    preprocessor,
                    config,
                    ["date", "latitude", "longitude"],
                ) / SECONDS_PER_DAY
                clear_diff = clear_pred - clear_true
                store_prediction(predictions, "tsrc_pred", clear_pred)
                store_prediction(predictions, "tsrc_true", clear_true)
                store_prediction(predictions, "tsrc_diff", clear_diff)
        for field in ("tco3", "tcwv"):
            if field in physical_ds:
                field_da = physical_ds[field]
            else:
                if raw is None:
                    raw = load_raw_test_dataset(config)
                if field not in raw:
                    continue
                field_da = raw_field_on_grid(raw, field, diff)
            store_prediction(predictions, f"{field}_field", field_da)
            corr = correlation(field_da, diff)
            results[f"tsr_residual_{field}_corr"] = corr["corr"]
            results[f"tsr_residual_{field}_corr_n"] = corr["n"]
            corr = correlation(field_da, true)
            results[f"tsr_{field}_corr"] = corr["corr"]
            results[f"tsr_{field}_corr_n"] = corr["n"]
            if clear_true is not None:
                corr = correlation(field_da, clear_true)
                results[f"tsrc_{field}_corr"] = corr["corr"]
                results[f"tsrc_{field}_corr_n"] = corr["n"]
            if clear_diff is not None:
                corr = correlation(field_da, clear_diff)
                results[f"tsrc_residual_{field}_corr"] = corr["corr"]
                results[f"tsrc_residual_{field}_corr_n"] = corr["n"]
        return results
    finally:
        if raw is not None:
            raw.close()
        ds.close()


def kernel_metrics(config, model, preprocessor, years, dates, predictions):
    out = {}
    raw_paths = generate_paths_yearly(config.dataset.era5.raw_path, years, make_era5_filename)
    raw = xr.open_mfdataset(raw_paths, combine="nested", concat_dim="date")
    try:
        raw = raw.assign(
            ecod=load_ecod(
                config.dataset.era5.raw_path,
                years,
                range(1, 13),
                config.preprocess.ecod.method,
            )
        )
        available_dates = {str(value)[:7] for value in raw.date.values}
        for var in [v for v in ("fal", "tcwv") if v in config.dataset.input_vars]:
            paths = generate_paths_yearly(
                config.dataset.kernels.raw_path,
                years,
                lambda year, name=var: make_kernel_filename(year, name),
            )
            kernels = xr.open_mfdataset(paths, combine="nested", concat_dim="date")
            try:
                for date in dates:
                    if date not in available_dates:
                        continue
                    true = kernels.sel(date=date)[var]
                    for sky, clear in (("all", False), ("clear", True)):
                        pred, lon, lat = compute_nn_kernel_autograd(
                            raw.sel(date=date), preprocessor, model, config, var=var, clear=clear
                        )
                        truth_da = (
                            true.sel(all_clr="clr" if clear else "all").squeeze(drop=True)
                            * kernel_delta(var)
                        )
                        pred_da = xr.DataArray(
                            pred,
                            coords={"latitude": lat, "longitude": lon},
                            dims=("latitude", "longitude"),
                        ).interp(latitude=truth_da.latitude, longitude=truth_da.longitude)
                        task = f"kernel/{sky}/{var}/{date}"
                        diff = pred_da - truth_da
                        stem = var_name(task)
                        store_prediction(predictions, f"{stem}__pred", pred_da)
                        store_prediction(predictions, f"{stem}__true", truth_da)
                        store_prediction(predictions, f"{stem}__diff", diff)
                        out[task] = metric(diff)
            finally:
                kernels.close()
    finally:
        raw.close()
    return out


def closure_metrics(config, predictions):
    root = Path(config.train.checkpoint_dir)
    paths = sorted(root.glob("figures/*/closure_test/saved_responses_closure_test.nc"))
    if not paths:
        return {}
    ds = xr.load_dataset(paths[-1])
    dR = ds["dR_era5_all"]
    dR_clr = ds["dR_era5_clr"]
    all_sum = ds["dR_a_nn_all"] + ds["dR_c_nn_all"] + ds["dR_q_nn_all"]
    clear_sum = ds["dR_a_nn_clr"] + ds["dR_q_nn_clr"]
    cross = {"dR_aq_nn_all", "dR_ac_nn_all", "dR_qc_nn_all"}
    all_cross_sum = all_sum
    if cross <= set(ds.data_vars):
        all_cross_sum = all_cross_sum + ds["dR_aq_nn_all"] + ds["dR_ac_nn_all"] + ds["dR_qc_nn_all"]
    residuals = {
        "closure/all/cross": dR - all_cross_sum,
        "closure/all/allcross": dR - ds["dR_nn_all"],
        "closure/clear/allcross": dR_clr - ds["dR_nn_clr"],
        "closure/clear/cross": dR_clr - clear_sum,
    }
    for task, residual in residuals.items():
        store_prediction(predictions, f"{var_name(task)}__residual", residual)
    out = {task: metric(residual) for task, residual in residuals.items()}
    return out


def compute_cache(output_dir, years, dates, models=MODELS):
    cache = {}
    manifest = {}
    prediction_dir = output_dir / "prediction_cache"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for name, config_path in models.items():
        config = load_config(config_path)
        predictions = {}
        tasks = closure_metrics(config, predictions)
        try:
            model, pp, _ = load_model_and_preprocessor(config, Path(config.train.checkpoint_dir) / "best_model.pt")
        except FileNotFoundError:
            if predictions:
                path = prediction_dir / f"{name}.nc"
                xr.Dataset(predictions).to_netcdf(path)
                manifest[name] = str(path)
            cache[name] = tasks
            continue
        tasks["tsr"] = tsr_metric(config, model, pp, predictions)
        tasks.update(kernel_metrics(config, model, pp, years, dates, predictions))
        if predictions:
            path = prediction_dir / f"{name}.nc"
            xr.Dataset(predictions).to_netcdf(path)
            manifest[name] = str(path)
        cache[name] = tasks
    path = output_dir / "comparison_metric_cache.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="ascii")
    (output_dir / "prediction_cache_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="ascii",
    )
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--setup", choices=SETUPS, default="default")
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--north-boundary", type=float, default=NORTH_BOUNDARY)
    parser.add_argument("--years", type=int, nargs="+", default=[2015])
    parser.add_argument("--dates", nargs="+", default=["2015-09"])
    args = parser.parse_args()

    models, comparisons, default_output_dir = SETUPS[args.setup]
    output_dir = args.output_dir or default_output_dir

    if args.cache:
        cache = json.loads(args.cache.read_text(encoding="ascii"))
    else:
        cache = compute_cache(output_dir, args.years, args.dates, models)
    write_tables(cache, output_dir, comparisons)
    north_cache = north_pole_cache(
        cache,
        output_dir / "prediction_cache_manifest.json",
        args.north_boundary,
    )
    if north_cache:
        write_tables(north_cache, output_dir / "north_pole", comparisons)
    write_residual_plots(cache, output_dir, output_dir / "prediction_cache_manifest.json", comparisons)
    write_tsr_field_correlations(cache, output_dir)
    write_tsr_field_scatter_plots(output_dir, output_dir / "prediction_cache_manifest.json")
    print(f"wrote tables and cache under {output_dir}")


if __name__ == "__main__":
    main()
