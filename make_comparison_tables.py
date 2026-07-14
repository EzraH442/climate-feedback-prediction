import argparse
import csv
import json
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from config_utils import load_config
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
    "sob_fal_fast_ecod_noozone": "configs/model/fal/2011-2014_3,6,9,12_sob_fal_fast-ecod_noozone_legacy.yaml",
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


def metric(diff):
    mbe = float(np.mean(global_mean(diff)))
    return {"mbe": mbe, "abs_mbe": abs(mbe), "rmse": float(np.sqrt(np.mean(global_mean(diff**2))))}


def task_slug(name):
    return name.replace("/", "_").replace(" ", "_").replace(",", "").lower()


def write_tables(cache, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for group, comparisons in COMPARISONS.items():
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
                for label, src, dst in comparisons:
                    old = cache.get(src, {}).get(task, {})
                    new = cache.get(dst, {}).get(task, {})
                    row = [label]
                    for key in ("mbe", "abs_mbe", "rmse"):
                        row.append(new[key] - old[key] if key in new and key in old else "")
                    writer.writerow(row)


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


def write_residual_plots(cache, output_dir, manifest_path):
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
        for group, comparisons in COMPARISONS.items():
            for task in metric_tasks(cache):
                var = cached_residual_name(task)
                rows = []
                for label, src, dst in comparisons:
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


def tsr_metric(config, model, preprocessor, predictions):
    ds = load_processed_dataset(config)
    try:
        pred = nn_pred(ds, model, preprocessor, config, ["date", "latitude", "longitude"])
        true = preprocessor.preprocessors[-1].inverse_transform(ds)[config.dataset.target_var]
        pred = pred / SECONDS_PER_DAY
        true = true / SECONDS_PER_DAY
        diff = pred - true
        store_prediction(predictions, "tsr_pred", pred)
        store_prediction(predictions, "tsr_true", true)
        store_prediction(predictions, "tsr_diff", diff)
        return metric(diff)
    finally:
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
                config.preprocess.ecod.method == "fast",
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


def compute_cache(output_dir, years, dates):
    cache = {}
    manifest = {}
    prediction_dir = output_dir / "prediction_cache"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    for name, config_path in MODELS.items():
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
    parser.add_argument("--output-dir", type=Path, default=Path("analysis_tables"))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--north-boundary", type=float, default=NORTH_BOUNDARY)
    parser.add_argument("--years", type=int, nargs="+", default=[2015])
    parser.add_argument("--dates", nargs="+", default=["2015-09"])
    args = parser.parse_args()

    if args.cache:
        cache = json.loads(args.cache.read_text(encoding="ascii"))
    else:
        cache = compute_cache(args.output_dir, args.years, args.dates)
    write_tables(cache, args.output_dir)
    north_cache = north_pole_cache(
        cache,
        args.output_dir / "prediction_cache_manifest.json",
        args.north_boundary,
    )
    if north_cache:
        write_tables(north_cache, args.output_dir / "north_pole")
    write_residual_plots(cache, args.output_dir, args.output_dir / "prediction_cache_manifest.json")
    print(f"wrote tables and cache under {args.output_dir}")


if __name__ == "__main__":
    main()

