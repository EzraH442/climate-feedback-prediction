import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xarray as xr

from config_utils import load_config, variable_config_from_omegaconf
from experiments.bivariate_perturbation.analysis import (
    plot_kernel_contour,
    plot_tsr_contour,
)
from experiments.kernel_difference.analysis import (
    plot_hybrid_kernel_difference,
    plot_kernel_difference,
)
from experiments.radiative_closure_single_date.analysis import (
    arctic_field,
    load_feedback_ablation_responses,
    make_neurips_feedback_context,
    shakirova_terms,
)
from experiments.sensitivity_prediction.main import predict_sensitivity
from experiments.tsr_prediction.main import predict_tsr
from utils import (
    SECONDS_PER_DAY,
    figure_dir,
    kernel_delta,
    kernel_label,
    kernel_title,
    load_model_and_preprocessor,
    make_era5_filename,
    make_kernel_filename,
    plot_global_field,
    plot_north_pole_field,
)

TABLE_MODEL_SPECS = [
    ("NN", Path("configs/model/fal/2011-2014_3,6,9,12_baseline.yaml")),
    (
        "NN + clear-sky",
        Path("configs/model/fal/2011-2014_3,6,9,12_baseline_clearsky.yaml"),
    ),
    (
        "NN + clear-sky + Sob.",
        Path("configs/model/fal/2011-2014_3,6,9,12_sob_fal_clearsky.yaml"),
    ),
]
SEED_AGGREGATED_MODELS = ["NN", "NN + clear-sky", "NN + clear-sky + Sob."]
SOBOLEV_LAMBDA_CONFIG = Path("configs/experiments/fal/sobolev_lambda.yaml")
SOBOLEV_ABLATION_CHECKPOINT_ROOT = Path("sobolev-ablation/checkpoints")
SOBOLEV_LAMBDA_SPECS = [
    (
        "lambda=0.25",
        SOBOLEV_LAMBDA_CONFIG,
        SOBOLEV_ABLATION_CHECKPOINT_ROOT
        / "2011-2014_3,6,9,12_sob_fal_0.25_clearsky",
    ),
    (
        "lambda=0.5",
        SOBOLEV_LAMBDA_CONFIG,
        SOBOLEV_ABLATION_CHECKPOINT_ROOT
        / "2011-2014_3,6,9,12_sob_fal_0.5_clearsky",
    ),
    (
        "lambda=1",
        SOBOLEV_LAMBDA_CONFIG,
        SOBOLEV_ABLATION_CHECKPOINT_ROOT
        / "2011-2014_3,6,9,12_sob_fal_1_clearsky",
    ),
    (
        "lambda=2",
        SOBOLEV_LAMBDA_CONFIG,
        SOBOLEV_ABLATION_CHECKPOINT_ROOT
        / "2011-2014_3,6,9,12_sob_fal_2_clearsky",
    ),
    (
        "lambda=4",
        SOBOLEV_LAMBDA_CONFIG,
        SOBOLEV_ABLATION_CHECKPOINT_ROOT
        / "2011-2014_3,6,9,12_sob_fal_4_clearsky",
    ),
]


def eval_kernel_vars(config):
    input_vars = list(config.dataset.input_vars)
    return [var for var in ("fal", "skt", "tcwv") if var in input_vars]


def plot_flux_summary(result: xr.Dataset, figures_path: Path) -> None:
    output_dir = figure_dir(figures_path, result.attrs["sky"], result.attrs["test_name"])
    target_label = result.attrs["target_label"]
    lon = result.longitude
    lat = result.latitude

    min_flux = min(float(result.truth_mean.min()), float(result.prediction_mean.min()))
    max_flux = max(float(result.truth_mean.max()), float(result.prediction_mean.max()))

    plot_global_field(
        result.truth_mean,
        lon,
        lat,
        f"{target_label} (ERA5)",
        output_dir / "era5.png",
        cmap="Spectral",
        vmin=min_flux,
        vmax=max_flux,
        label="$W/m^2$",
        annotation=f"{float(result.truth_mean.mean()):.2f}",
    )
    plot_global_field(
        result.prediction_mean,
        lon,
        lat,
        f"{target_label} (NN)",
        output_dir / "nn.png",
        cmap="Spectral",
        vmin=min_flux,
        vmax=max_flux,
        label="$W/m^2$",
        annotation=f"{float(result.prediction_mean.mean()):.2f}",
    )
    plot_global_field(
        result.mbe,
        lon,
        lat,
        "MBE",
        output_dir / "mbe.png",
        cmap="RdBu_r",
        vmin=-20,
        vmax=20,
        label="$W/m^2$",
        annotation=f"{float(result.mbe.mean()):.2f}",
    )
    plot_global_field(
        result.rmse,
        lon,
        lat,
        "RMSE",
        output_dir / "rmse.png",
        cmap="Blues",
        vmin=0,
        vmax=20,
        label="$W/m^2$",
        annotation=f"{float(np.sqrt((result.rmse**2).mean())):.2f}",
    )


def plot_kernel_date_result(result: xr.Dataset, figures_path: Path) -> None:
    kernel_name = result.attrs["kernel_name"]
    date = result.attrs["date"]
    title = kernel_title(kernel_name)
    label = kernel_label(kernel_name)
    lon = result.longitude
    lat = result.latitude
    north_mask = lat >= 60
    all_dir = figure_dir(figures_path, "all", "kernel_date_test", kernel_name, date)
    clr_dir = figure_dir(figures_path, "clr", "kernel_date_test", kernel_name, date)

    for sky, sky_dir, suffix in [("all", all_dir, "all"), ("clear", clr_dir, "clear")]:
        rrtm_name = f"rrtm_{sky}"
        fd_name = f"nn_fd_{sky}"
        grad_name = f"nn_grad_{sky}"
        diff_fd_name = f"diff_fd_{sky}"
        diff_grad_name = f"diff_grad_{sky}"

        plot_global_field(
            result[fd_name],
            lon,
            lat,
            f"NN {title} Kernel finite difference ({suffix})\n{date}",
            sky_dir / "nn.png",
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
            label=label,
            annotation=f"{float(result[fd_name].mean()):.2f}",
        )
        plot_global_field(
            result[grad_name],
            lon,
            lat,
            f"NN {title} Kernel via autograd ({suffix})\n{date}",
            sky_dir / "nn_grad.png",
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
            label=label,
            annotation=f"{float(result[grad_name].mean()):.2f}",
        )
        plot_north_pole_field(
            result[fd_name],
            lon,
            lat,
            f"NN {title} Kernel finite difference ({suffix})\n{date}",
            sky_dir / "nn_np.png",
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
            label=label,
            annotation=f"{float(result[fd_name].where(north_mask).mean()):.2f}",
        )
        plot_north_pole_field(
            result[grad_name],
            lon,
            lat,
            f"NN {title} Kernel via autograd ({suffix})\n{date}",
            sky_dir / "nn_grad_np.png",
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
            label=label,
            annotation=f"{float(result[grad_name].where(north_mask).mean()):.2f}",
        )
        plot_global_field(
            result[rrtm_name],
            lon,
            lat,
            f"RRTM {title} Kernel ({suffix})\n{date}",
            sky_dir / "rrtm.png",
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
            label=label,
            annotation=f"{float(result[rrtm_name].mean()):.2f}",
        )
        plot_north_pole_field(
            result[rrtm_name],
            lon,
            lat,
            f"RRTM {title} Kernel ({suffix})\n{date}",
            sky_dir / "rrtm_np.png",
            cmap="RdBu_r",
            vmin=-3,
            vmax=3,
            label=label,
            annotation=f"{float(result[rrtm_name].where(north_mask).mean()):.2f}",
        )
        plot_global_field(
            result[diff_fd_name],
            lon,
            lat,
            f"NN finite difference-RRTM {title} Kernel ({suffix})\n{date}",
            sky_dir / "nn-rrtm.png",
            cmap="RdBu_r",
            vmin=-0.6,
            vmax=0.6,
            label=label,
            annotation=f"{float(result[diff_fd_name].mean()):.2f}; {float(abs(result[diff_fd_name]).mean()):.2f}",
        )
        plot_global_field(
            result[diff_grad_name],
            lon,
            lat,
            f"NN autograd-RRTM {title} Kernel ({suffix})\n{date}",
            sky_dir / "nn_grad-rrtm.png",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            label=label,
            annotation=f"{float(result[diff_grad_name].mean()):.2f}; {float(abs(result[diff_grad_name]).mean()):.2f}",
        )
        plot_north_pole_field(
            result[diff_fd_name],
            lon,
            lat,
            f"NN finite difference-RRTM {title} Kernel ({suffix})\n{date}",
            sky_dir / "nn-rrtm_np.png",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            label=label,
            annotation=f"{float(result[diff_fd_name].where(north_mask).mean()):.2f}; {float(abs(result[diff_fd_name].where(north_mask)).mean()):.2f}",
        )
        plot_north_pole_field(
            result[diff_grad_name],
            lon,
            lat,
            f"NN autograd-RRTM {title} Kernel ({suffix})\n{date}",
            sky_dir / "nn_grad-rrtm_np.png",
            cmap="RdBu_r",
            vmin=-1,
            vmax=1,
            label=label,
            annotation=f"{float(result[diff_grad_name].where(north_mask).mean()):.2f}; {float(abs(result[diff_grad_name].where(north_mask)).mean()):.2f}",
        )


def kernel_difference_fields(result: xr.Dataset) -> dict:
    fields = {name: result[name] for name in result.data_vars}
    fields["base_date"] = result.attrs["base_date"]
    fields["perturbed_date"] = result.attrs["perturbed_date"]
    fields["lon"] = result.longitude
    fields["lat"] = result.latitude
    return fields


def plot_standard_eval_results(output_dir: Path) -> None:
    results_dir = output_dir / "results"
    plot_flux_summary(xr.open_dataset(results_dir / "tsr_test.nc"), output_dir)
    plot_flux_summary(xr.open_dataset(results_dir / "tsrc_test.nc"), output_dir)

    for path in sorted(results_dir.glob("kernel_date_*.nc")):
        plot_kernel_date_result(xr.open_dataset(path), output_dir)

    for path in sorted(results_dir.glob("second_order_*.nc")):
        fields = kernel_difference_fields(xr.open_dataset(path))
        second_order_dir = figure_dir(
            output_dir,
            "all",
            "second_order_test",
            path.stem.removeprefix("second_order_"),
            f"{fields['perturbed_date']}_minus_{fields['base_date']}",
        )
        plot_kernel_difference(fields, second_order_dir)

    for path in sorted(results_dir.glob("hybrid_second_order_*.nc")):
        fields = kernel_difference_fields(xr.open_dataset(path))
        second_order_dir = figure_dir(
            output_dir,
            "all",
            "second_order_test",
            path.stem.removeprefix("hybrid_second_order_"),
            f"{fields['perturbed_date']}_minus_{fields['base_date']}",
        )
        plot_hybrid_kernel_difference(fields, second_order_dir)

    plot_kernel_contour(xr.open_dataset(results_dir / "kernel_contour_fal_ecod.nc"), output_dir)
    plot_tsr_contour(xr.open_dataset(results_dir / "tsr_contour_fal_ecod.nc"), output_dir)


# --------------------------------------------------------------------------- #
# NeurIPS summary table helpers
# --------------------------------------------------------------------------- #

table_output_dir = Path("FiguresNeurIPS") / "ablation_tables"

validation_output_dir = table_output_dir / "validation_points"

tsr_table_path = table_output_dir / "tsr_test_1990_2020.csv"
kernel_table_path = table_output_dir / "kernel_date_test_2015_heldout_months.csv"
kernel_fd_table_path = (
    table_output_dir / "kernel_finite_difference_date_test_2015_heldout_months.csv"
)
closure_table_path = table_output_dir / "closure_1990_2020.csv"
feedback_table_path = table_output_dir / "feedback_quantification_two_year.csv"


def seed_checkpoint_paths(config_path, checkpoint_dir=None):
    config_for_path = load_config(config_path)
    base_dir = Path(checkpoint_dir or config_for_path.train.checkpoint_dir)
    return sorted(base_dir.parent.glob(f"{base_dir.name}_seed_*/best_model.pt"))


def expanded_table_model_specs():
    for label, config_path in TABLE_MODEL_SPECS:
        seeds = (
            seed_checkpoint_paths(config_path)
            if label in SEED_AGGREGATED_MODELS
            else []
        )
        if len(seeds) > 1:
            for i, seed_path in enumerate(seeds, start=1):
                yield f"{label} [seed {i}]", config_path, seed_path
        else:
            yield label, config_path, None


def seed_rows_for(table, label):
    """Boolean mask selecting rows of `table` belonging to seed variants of `label`."""
    index = (
        table.index.get_level_values(0)
        if isinstance(table.index, pd.MultiIndex)
        else table.index
    )
    return index.astype(str).str.startswith(f"{label} [seed ")


def format_mean_std(mean, std):
    return mean.combine(std, lambda m, s: f"{m:.3f} \u00b1 {s:.3f}")


def aggregate_seed_rows(table):
    print(table)
    """Collapse per-seed rows into a single mean +/- std row per seed-aggregated model."""
    multi = isinstance(table.index, pd.MultiIndex)
    rows, seen_mask = [], pd.Series(False, index=table.index)

    for label in SEED_AGGREGATED_MODELS:
        mask = seed_rows_for(table, label)
        seen_mask |= mask
        if not mask.any():
            continue
        seed_table = table.loc[mask].astype(float)

        if not multi:
            rows.append(
                pd.DataFrame(
                    [format_mean_std(seed_table.mean(), seed_table.std(ddof=1))],
                    index=[label],
                )
            )
            continue

        for residual in seed_table.index.get_level_values(1).unique():
            residual_table = seed_table.xs(residual, level=1)
            row = format_mean_std(residual_table.mean(), residual_table.std(ddof=1))
            row.name = (label, residual)
            rows.append(row)

    if not rows:
        return table

    keep = table.loc[~seen_mask.to_numpy()]
    if not multi:
        return pd.concat([*rows, keep])

    aggregated = pd.DataFrame(rows)
    aggregated.index = pd.MultiIndex.from_tuples(
        aggregated.index, names=table.index.names
    )
    return pd.concat([aggregated, keep])


def seed_aggregation_available():
    """Whether any seed-aggregated model currently has >1 discovered seed checkpoint."""
    return any(
        len(seed_checkpoint_paths(config_path)) > 1
        for label, config_path in TABLE_MODEL_SPECS
        if label in SEED_AGGREGATED_MODELS
    )


def table_has_seed_aggregation(table):
    """Whether a cached table already contains mean +/- std formatted seed rows."""
    values = table.astype(str).to_numpy().ravel()
    return any("\u00b1" in value for value in values)


def safe_name(value):
    """Filesystem-safe version of a label, for use in output filenames."""
    return "".join(ch if ch.isalnum() else "_" for ch in str(value)).strip("_")


def load_evaluation_model(config_path, checkpoint_path):
    table_config = load_config(config_path)
    table_vc = variable_config_from_omegaconf(table_config)
    checkpoint_path = (
        checkpoint_path or Path(table_config.train.checkpoint_dir) / "best_model.pt"
    )
    table_model, table_preprocessor, _ = load_model_and_preprocessor(
        table_config, checkpoint_path, downscaling=False
    )
    return table_config, table_vc, table_model, table_preprocessor


def collect_model_metric_rows(row_fn, *args, **kwargs):
    """Load each model spec and collect the rows `row_fn` produces for it."""
    rows = []
    for label, config_path, checkpoint_path in expanded_table_model_specs():
        table_config, table_vc, table_model, table_preprocessor = load_evaluation_model(
            config_path, checkpoint_path
        )
        rows.extend(
            row_fn(
                label,
                table_config,
                table_vc,
                table_model,
                table_preprocessor,
                *args,
                **kwargs,
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Validation-point snapshots (for spot-checking predictions vs truth)
# --------------------------------------------------------------------------- #


def validation_point_dataset(test, model, date, sky, prediction, truth):
    prediction = (
        prediction.reset_coords(drop=True)
        .transpose("latitude", "longitude")
        .astype("float32")
    )
    truth = (
        truth.reset_coords(drop=True)
        .transpose("latitude", "longitude")
        .astype("float32")
    )
    return xr.Dataset(
        {
            "truth": truth.expand_dims(record=1),
            "prediction": prediction.expand_dims(record=1),
            "residual": (prediction - truth).astype("float32").expand_dims(record=1),
        },
        coords={
            "test": ("record", [test]),
            "model": ("record", [model]),
            "date": ("record", [pd.Timestamp(date)]),
            "sky": ("record", [sky]),
        },
    )


def write_validation_points(path, datasets):
    if not datasets:
        return
    path.parent.mkdir(exist_ok=True, parents=True)
    ds = xr.concat(datasets, dim="record").assign_coords(
        record=np.arange(len(datasets))
    )
    ds.to_netcdf(path)


# --------------------------------------------------------------------------- #
# Metric bookkeeping (mean bias, RMSE, across-model summary tables)
# --------------------------------------------------------------------------- #


def global_bias_and_mean_square(diff, arctic=False):
    print(diff)

    field = arctic_field(diff) if arctic else diff
    mean = float(global_mean(field).compute())
    mean_sq = float(global_mean(field**2).compute())
    return mean, mean_sq


def append_prediction_metric(rows, model, sky, diff, arctic=False):
    mean, mean_sq = global_bias_and_mean_square(diff, arctic=arctic)
    rows.append({"Model": model, "sky": sky, "mean": mean, "mean_sq": mean_sq})


def summarize_prediction_metrics(rows, include_std=True):
    rows_df = pd.DataFrame(rows)
    table_rows = []
    for model, model_df in rows_df.groupby("Model", sort=False):
        row = {"Model": model}
        for sky in ["all", "clear"]:
            sky_df = model_df[model_df.sky == sky]
            row[f"{sky} MBE"] = float(sky_df["mean"].mean())
            row[f"{sky} RMSE"] = float(np.sqrt(sky_df["mean_sq"].mean()))
            if include_std:
                row[f"{sky} STD"] = float(sky_df["mean"].std(ddof=0))
        table_rows.append(row)
    return pd.DataFrame(table_rows).set_index("Model")


def append_feedback_metric(rows, model, residual, sky, region, diff):
    mean, mean_sq = global_bias_and_mean_square(diff, arctic=(region == "arctic"))
    rows.append(
        {
            "Model": model,
            "Residual": residual,
            "sky": sky,
            "region": region,
            "mean": mean,
            "mean_sq": mean_sq,
        }
    )


def summarize_feedback_metrics(rows):
    rows_df = pd.DataFrame(rows)
    table_rows = []
    for (model, residual), group_df in rows_df.groupby(
        ["Model", "Residual"], sort=False
    ):
        row = {"Model": model, "Residual": residual}
        for region in ["global", "arctic"]:
            for sky in ["all", "clear"]:
                subset = group_df[(group_df.region == region) & (group_df.sky == sky)]
                row[f"{region} {sky} MBE"] = float(subset["mean"].mean())
                row[f"{region} {sky} RMSE"] = float(np.sqrt(subset["mean_sq"].mean()))
        table_rows.append(row)
    return pd.DataFrame(table_rows).set_index(["Model", "Residual"])


# --------------------------------------------------------------------------- #
# TSR test: NN TSR predictions vs ERA5 TSR, 1990-2020, all-sky & clear-sky
# --------------------------------------------------------------------------- #


def collect_tsr_prediction_metrics(
    label, table_config, table_vc, table_model, table_preprocessor
):
    rows, validation_points = [], []
    for year in range(1990, 2021):
        raw_ds = xr.open_dataset(
            Path(table_config.dataset.era5.raw_path) / make_era5_filename(year)
        )
        for date in raw_ds.date.values:
            raw_date = raw_ds.sel(date=date)

            all_pred = (
                predict_tsr(raw_date, table_model, table_preprocessor, table_vc)
                / SECONDS_PER_DAY
            )
            clear_pred = (
                predict_tsr(
                    raw_date,
                    table_model,
                    table_preprocessor,
                    table_vc,
                    clear_sky=True,
                )
                / SECONDS_PER_DAY
            )

            all_truth = raw_date.tsr / SECONDS_PER_DAY
            clear_truth = raw_date.tsrc / SECONDS_PER_DAY

            append_prediction_metric(rows, label, "all", all_pred - all_truth)
            append_prediction_metric(rows, label, "clear", clear_pred - clear_truth)

            if len(validation_points) < MAX_VALIDATION_POINTS:
                validation_points.extend(
                    [
                        validation_point_dataset(
                            "tsr", label, date, "all", all_pred, all_truth
                        ),
                        validation_point_dataset(
                            "tsr", label, date, "clear", clear_pred, clear_truth
                        ),
                    ]
                )
        raw_ds.close()

    write_validation_points(
        validation_output_dir / f"tsr_{safe_name(label)}.nc", validation_points
    )
    return rows


def save_table(table, path):
    path.parent.mkdir(exist_ok=True, parents=True)
    table.to_csv(path)
    return table


def write_tsr_prediction_table():
    table = aggregate_seed_rows(
        summarize_prediction_metrics(
            collect_model_metric_rows(collect_tsr_prediction_metrics), include_std=True
        )
    )
    return save_table(table, tsr_table_path)


# --------------------------------------------------------------------------- #
# Kernel date test: 2015 held-out months, NN albedo kernel vs FAL kernel
# (autograd-derived and, optionally, finite-difference-derived)
# --------------------------------------------------------------------------- #


def collect_albedo_kernel_metrics(
    label,
    table_config,
    table_vc,
    table_model,
    table_preprocessor,
    finite_difference=False,
):
    rows, validation_points = [], []
    held_out_months = [month for month in range(1, 13) if month not in (3, 6, 9, 12)]
    dates = [f"2015-{month:02d}" for month in held_out_months]

    raw_ds = xr.open_dataset(
        Path(table_config.dataset.era5.raw_path) / make_era5_filename(2015)
    )
    kernels = xr.open_dataset(
        Path(table_config.dataset.kernels.raw_path) / make_kernel_filename(2015, "fal")
    )

    for date in dates:
        raw_date = raw_ds.sel(date=[date]).interp(
            latitude=kernels.latitude, longitude=kernels.longitude
        )
        true_kernel = kernels.sel(date=date).fal.squeeze(
            "date", drop=True
        ) * kernel_delta("fal")

        all_truth = true_kernel.sel(all_clr="all")
        clear_truth = true_kernel.sel(all_clr="clr")
        all_pred = predict_sensitivity(
            raw_date,
            table_preprocessor,
            table_model,
            table_vc,
            all_truth,
            variable="fal",
            autograd=not finite_difference,
        ).squeeze("date", drop=True)
        clear_pred = predict_sensitivity(
            raw_date,
            table_preprocessor,
            table_model,
            table_vc,
            clear_truth,
            variable="fal",
            clear_sky=True,
            autograd=not finite_difference,
        ).squeeze("date", drop=True)

        append_prediction_metric(rows, label, "all", all_pred - all_truth)
        append_prediction_metric(rows, label, "clear", clear_pred - clear_truth)

        if len(validation_points) < MAX_VALIDATION_POINTS:
            test_name = "kernel_finite_difference" if finite_difference else "kernel"
            validation_points.extend(
                [
                    validation_point_dataset(
                        test_name, label, date, "all", all_pred, all_truth
                    ),
                    validation_point_dataset(
                        test_name, label, date, "clear", clear_pred, clear_truth
                    ),
                ]
            )

    raw_ds.close()
    kernels.close()

    suffix = "kernel_finite_difference" if finite_difference else "kernel"
    write_validation_points(
        validation_output_dir / f"{suffix}_{safe_name(label)}.nc", validation_points
    )
    return rows


def write_albedo_kernel_table():
    table = aggregate_seed_rows(
        summarize_prediction_metrics(
            collect_model_metric_rows(collect_albedo_kernel_metrics), include_std=True
        )
    )
    return save_table(table, kernel_table_path)


def write_albedo_kernel_finite_difference_table():
    rows = collect_model_metric_rows(
        collect_albedo_kernel_metrics, finite_difference=True
    )
    table = aggregate_seed_rows(summarize_prediction_metrics(rows, include_std=True))
    return save_table(table, kernel_fd_table_path)


# --------------------------------------------------------------------------- #
# Feedback quantification test: context base_date -> perturbed_date response residuals
# --------------------------------------------------------------------------- #


def collect_feedback_response_metrics(ctx, label, config_path, checkpoint_path):
    rows = []
    responses = load_feedback_ablation_responses(
        ctx, label, config_path, "nn", checkpoint_path
    )
    for sky, response_sky in [("all", "all"), ("clear", "clr")]:
        terms = shakirova_terms(responses, response_sky)
        for residual, diff in [
            ("allcross", terms["allcross"][-1]),
            ("nn sum", terms["nn"][-1]),
        ]:
            for region in ["global", "arctic"]:
                append_feedback_metric(rows, label, residual, sky, region, diff)
    return rows


def collect_feedback_kernel_metrics(ctx):
    rows = []
    responses = load_feedback_ablation_responses(ctx, None, None, "kernel")
    for sky, response_sky in [("all", "all"), ("clear", "clr")]:
        terms = shakirova_terms(responses, response_sky)
        diff = terms["kernel"][-1]
        for region in ["global", "arctic"]:
            append_feedback_metric(rows, "Kernel", "kernel sum", sky, region, diff)
    return rows


def write_feedback_response_table(ctx):
    rows = []
    for label, config_path, checkpoint_path in expanded_table_model_specs():
        rows.extend(
            collect_feedback_response_metrics(ctx, label, config_path, checkpoint_path)
        )
    rows.extend(collect_feedback_kernel_metrics(ctx))

    table = aggregate_seed_rows(summarize_feedback_metrics(rows))
    return save_table(table, feedback_table_path)


def ensure_feedback_kernel_method(table, ctx):
    """Backfill the kernel-method row into a cached feedback table that predates it."""
    if (
        isinstance(table.index, pd.MultiIndex)
        and ("Kernel", "kernel sum") in table.index
    ):
        return table
    print("Feedback quantification test: recomputing for method-level mean/std rows")
    return write_feedback_response_table(ctx)


# --------------------------------------------------------------------------- #
# Cache loading and top-level table computation
# --------------------------------------------------------------------------- #


def load_or_compute_table(path, description, compute_table, index_col=0):
    """Load a cached CSV table if present and up to date, else compute and cache it."""
    if path.exists():
        table = pd.read_csv(path, index_col=index_col)
        if seed_aggregation_available() and not table_has_seed_aggregation(table):
            print(f"{description}: recomputing for seed mean/std")
            return compute_table()
        print(f"{description}: loaded {path}")
        return table
    print(f"{description}: computing")
    return compute_table()


def write_neurips_summary_tables(ctx=None):
    ctx = ctx or make_neurips_feedback_context()
    tsr_table = load_or_compute_table(
        tsr_table_path,
        "TSR test: monthly 1990-2020 NN TSR predictions vs ERA5 TSR, all-sky and clear-sky",
        write_tsr_prediction_table,
    )
    kernel_table = load_or_compute_table(
        kernel_table_path,
        "Kernel date test: 2015 held-out months except March, June, September, December; NN autograd albedo kernel vs FAL kernel",
        write_albedo_kernel_table,
    )
    kernel_fd_table = load_or_compute_table(
        kernel_fd_table_path,
        "Kernel finite-difference date test: same 2015 held-out months; NN finite-difference albedo kernel vs FAL kernel",
        write_albedo_kernel_finite_difference_table,
    )
    feedback_table = ensure_feedback_kernel_method(
        load_or_compute_table(
            feedback_table_path,
            "Feedback quantification test: context base_date to perturbed_date response residuals, global and Arctic",
            lambda: write_feedback_response_table(ctx),
            index_col=[0, 1],
        ),
        ctx,
    )
    return tsr_table, kernel_table, kernel_fd_table, feedback_table


# %%
import re

sobolev_lambda_tsr_raw_path = (
    table_output_dir / "sobolev_lambda_seed_metrics_tsr_test_1990_2020.csv"
)
sobolev_lambda_kernel_raw_path = (
    table_output_dir
    / "sobolev_lambda_seed_metrics_kernel_date_test_2015_heldout_months.csv"
)
sobolev_lambda_feedback_raw_path = (
    table_output_dir
    / "sobolev_lambda_seed_metrics_feedback_quantification_two_year.csv"
)


def sobolev_lambda_seed_paths(label, config_path, checkpoint_dir):
    seeds = seed_checkpoint_paths(config_path, checkpoint_dir)
    if len(seeds) != 5:
        raise FileNotFoundError(
            f"{label}: expected 5 seeded checkpoints, found {len(seeds)}"
        )
    return seeds


def sobolev_seed_labels(label):
    return [f"{label} [seed {i}]" for i in range(1, 6)]


def strip_sobolev_seed_label(value):
    return re.sub(r" \[seed \d+\](?=$| )", "", str(value))


def sobolev_seed_mean_std_table(table):
    table = table.astype(float).copy()
    if isinstance(table.index, pd.MultiIndex):
        tuples = [
            (strip_sobolev_seed_label(parts[0]), *parts[1:]) for parts in table.index
        ]
        table.index = pd.MultiIndex.from_tuples(tuples, names=table.index.names)
        grouped = table.groupby(level=list(range(table.index.nlevels)), sort=False)
    else:
        table.index = table.index.map(strip_sobolev_seed_label)
        grouped = table.groupby(level=0, sort=False)

    mean = grouped.mean()
    std = grouped.std(ddof=1)
    return mean.map(lambda x: f"{x:.3f}") + " ± " + std.map(lambda x: f"{x:.3f}")


def missing_sobolev_specs(table):
    existing = set(
        (
            table.index.get_level_values(0)
            if isinstance(table.index, pd.MultiIndex)
            else table.index
        ).astype(str)
    )
    return [
        (label, config_path, checkpoint_dir)
        for label, config_path, checkpoint_dir in SOBOLEV_LAMBDA_SPECS
        if not set(sobolev_seed_labels(label)).issubset(existing)
    ]


def sobolev_lambda_collect_model_metric_rows(row_fn, specs=None, *args, **kwargs):
    rows = []
    for label, config_path, checkpoint_dir in specs or SOBOLEV_LAMBDA_SPECS:
        for seed_label, checkpoint_path in zip(
            sobolev_seed_labels(label),
            sobolev_lambda_seed_paths(label, config_path, checkpoint_dir),
        ):
            table_config, table_vc, table_model, table_preprocessor = (
                load_evaluation_model(config_path, checkpoint_path)
            )
            rows.extend(
                row_fn(
                    seed_label,
                    table_config,
                    table_vc,
                    table_model,
                    table_preprocessor,
                    *args,
                    **kwargs,
                )
            )
    return rows


def load_or_update_raw_sobolev_table(path, description, compute_table, index_col=0):
    if not path.exists():
        print(f"{description}: computing")
        table = compute_table(SOBOLEV_LAMBDA_SPECS)
        table.to_csv(path)
        return table

    table = pd.read_csv(path, index_col=index_col)
    missing = missing_sobolev_specs(table)
    if not missing:
        print(f"{description}: loaded {path}")
        return table

    print(f"{description}: computing missing configs {[label for label, *_ in missing]}")
    table = pd.concat([table, compute_table(missing)])
    table = table[~table.index.duplicated(keep="last")]
    table.to_csv(path)
    return table


def compute_sobolev_lambda_tsr_raw_table(specs):
    return summarize_prediction_metrics(
        sobolev_lambda_collect_model_metric_rows(collect_tsr_prediction_metrics, specs),
        include_std=False,
    )


def compute_sobolev_lambda_kernel_raw_table(specs):
    return summarize_prediction_metrics(
        sobolev_lambda_collect_model_metric_rows(collect_albedo_kernel_metrics, specs),
        include_std=False,
    )


def compute_sobolev_lambda_feedback_raw_table(specs, ctx):
    rows = []
    for label, config_path, checkpoint_dir in specs:
        for seed_label, checkpoint_path in zip(
            sobolev_seed_labels(label),
            sobolev_lambda_seed_paths(label, config_path, checkpoint_dir),
        ):
            rows.extend(
                collect_feedback_response_metrics(
                    ctx, seed_label, config_path, checkpoint_path
                )
            )
    return summarize_feedback_metrics(rows)


def write_sobolev_lambda_summary_tables(ctx=None):
    ctx = ctx or make_neurips_feedback_context()
    sobolev_lambda_tsr_raw_table = load_or_update_raw_sobolev_table(
        sobolev_lambda_tsr_raw_path,
        "Sobolev lambda ablation TSR seed metrics",
        compute_sobolev_lambda_tsr_raw_table,
    )
    sobolev_lambda_kernel_raw_table = load_or_update_raw_sobolev_table(
        sobolev_lambda_kernel_raw_path,
        "Sobolev lambda ablation kernel sensitivity seed metrics",
        compute_sobolev_lambda_kernel_raw_table,
    )
    sobolev_lambda_feedback_raw_table = load_or_update_raw_sobolev_table(
        sobolev_lambda_feedback_raw_path,
        "Sobolev lambda ablation two-year feedback seed metrics",
        lambda specs: compute_sobolev_lambda_feedback_raw_table(specs, ctx),
        index_col=[0, 1],
    )

    return (
        sobolev_lambda_tsr_raw_table,
        sobolev_lambda_kernel_raw_table,
        sobolev_lambda_feedback_raw_table,
        sobolev_seed_mean_std_table(sobolev_lambda_tsr_raw_table),
        sobolev_seed_mean_std_table(sobolev_lambda_kernel_raw_table),
        sobolev_seed_mean_std_table(sobolev_lambda_feedback_raw_table),
    )
