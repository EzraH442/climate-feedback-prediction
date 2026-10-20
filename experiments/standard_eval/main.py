# Example: python experiments/standard_eval/main.py --config_file configs/model/fal/2011-2014_3,6,9,12_baseline.yaml
import sys
from pathlib import Path

import torch
import xarray as xr
from tap import Tap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config, variable_config_from_omegaconf
from experiments.bivariate_perturbation.main import (
    compute_kernel_contour,
    compute_tsr_contour,
)
from experiments.common import parse_args_and_confirm, write_netcdf
from experiments.kernel_difference.main import (
    compute_hybrid_kernel_difference,
    compute_kernel_difference,
)
from experiments.standard_eval.analysis import (
    eval_kernel_vars,
    plot_standard_eval_results,
)
from preprocessing import Preprocessor
from utils import (
    SECONDS_PER_DAY,
    compute_nn_kernel,
    compute_nn_kernel_autograd,
    generate_paths_yearly,
    interpolate_spatial_field,
    kernel_delta,
    load_model_and_preprocessor,
    make_era5_filename,
    make_kernel_filename,
    nn_pred,
    target_flux,
)


class StandardEvalArgs(Tap):
    config_file: Path
    checkpoint_path: Path | None = None
    output_dir: Path | None = None

    def process_args(self):
        self.config = load_config(self.config_file)
        if self.output_dir is None:
            checkpoint_path = self.checkpoint_path or (
                Path(self.config.train.checkpoint_dir) / "best_model.pt"
            )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            self.output_dir = (
                Path(self.config.train.checkpoint_dir)
                / "figures"
                / str(checkpoint["epoch"])
            )


def flux_summary(pred: xr.DataArray, truth: xr.DataArray, attrs: dict) -> xr.Dataset:
    pred = pred / SECONDS_PER_DAY
    truth = truth / SECONDS_PER_DAY
    diff = pred - truth
    return xr.Dataset(
        {
            "truth_mean": truth.mean("date").astype("float32"),
            "prediction_mean": pred.mean("date").astype("float32"),
            "mbe": diff.mean("date").astype("float32"),
            "rmse": ((diff**2).mean("date") ** 0.5).astype("float32"),
        },
        attrs=attrs,
    )


def compute_tsr_summary(processed_dataset, preprocessor, model, vc):
    pred = nn_pred(
        processed_dataset,
        model,
        preprocessor,
        vc,
        ["date", "latitude", "longitude"],
    )
    truth = target_flux(processed_dataset, preprocessor, vc.target_var)
    target_label = "TSR" if vc.target_var == "pal" else vc.target_var.upper()
    return flux_summary(
        pred,
        truth,
        {"sky": "all", "test_name": "tsr_test", "target_label": target_label},
    )


def compute_tsrc_summary(raw_dataset, preprocessor, model, vc):
    clear_processed = preprocessor.transform(vc.clear_sky_input(raw_dataset))
    pred = nn_pred(
        clear_processed,
        model,
        preprocessor,
        vc,
        ["date", "latitude", "longitude"],
        clear=True,
    )
    truth = target_flux(clear_processed, preprocessor, vc.clear_sky_target)
    target_label = "TSRC" if vc.clear_sky_target == "palc" else vc.clear_sky_target.upper()
    return flux_summary(
        pred,
        truth,
        {"sky": "clr", "test_name": "tsrc_test", "target_label": target_label},
    )


def as_field(values, lon, lat):
    return xr.DataArray(
        values,
        coords={"latitude": lat, "longitude": lon},
        dims=("latitude", "longitude"),
    )


def interpolate_to_kernel_grid(values, lon, lat, kernel_grid):
    field = values
    if values.shape != kernel_grid.shape:
        field = interpolate_spatial_field(
            values, lon, lat, kernel_grid.longitude, kernel_grid.latitude
        )
    return as_field(field, kernel_grid.longitude, kernel_grid.latitude)


def compute_kernel_date_result(ds, true_kernel, preprocessor, model, vc, date, kernel_name):
    nn_fd_all, lon, lat = compute_nn_kernel(
        ds, preprocessor, model, vc, perturbation_var=kernel_name
    )
    nn_fd_clear, _, _ = compute_nn_kernel(
        ds, preprocessor, model, vc, clear=True, perturbation_var=kernel_name
    )
    nn_grad_all, grad_lon, grad_lat = compute_nn_kernel_autograd(
        ds, preprocessor, model, vc, var=kernel_name
    )
    nn_grad_clear, _, _ = compute_nn_kernel_autograd(
        ds, preprocessor, model, vc, var=kernel_name, clear=True
    )

    rrtm_all = (
        true_kernel[kernel_name].sel(all_clr="all").squeeze(drop=True)
        * kernel_delta(kernel_name)
    )
    rrtm_clear = (
        true_kernel[kernel_name].sel(all_clr="clr").squeeze(drop=True)
        * kernel_delta(kernel_name)
    )

    fd_all = interpolate_to_kernel_grid(nn_fd_all, lon, lat, rrtm_all)
    fd_clear = interpolate_to_kernel_grid(nn_fd_clear, lon, lat, rrtm_clear)
    grad_all = interpolate_to_kernel_grid(nn_grad_all, grad_lon, grad_lat, rrtm_all)
    grad_clear = interpolate_to_kernel_grid(nn_grad_clear, grad_lon, grad_lat, rrtm_clear)

    return xr.Dataset(
        {
            "nn_fd_all": fd_all.astype("float32"),
            "nn_fd_clear": fd_clear.astype("float32"),
            "nn_grad_all": grad_all.astype("float32"),
            "nn_grad_clear": grad_clear.astype("float32"),
            "rrtm_all": rrtm_all.astype("float32"),
            "rrtm_clear": rrtm_clear.astype("float32"),
            "diff_fd_all": (fd_all - rrtm_all).astype("float32"),
            "diff_fd_clear": (fd_clear - rrtm_clear).astype("float32"),
            "diff_grad_all": (grad_all - rrtm_all).astype("float32"),
            "diff_grad_clear": (grad_clear - rrtm_clear).astype("float32"),
        },
        attrs={"date": date, "kernel_name": kernel_name},
    )


def fields_to_dataset(fields, attrs):
    return xr.Dataset(
        {
            key: value.astype("float32")
            for key, value in fields.items()
            if isinstance(value, xr.DataArray)
        },
        attrs=attrs,
    )


def align_hybrid_fields(fields):
    aligned = dict(fields)
    for name in ("delta_fal", "delta_ecod"):
        value = aligned.get(name)
        if value is None or value.shape == aligned["nn_base"].shape:
            continue
        aligned[name] = as_field(
            interpolate_spatial_field(
                value,
                value.longitude,
                value.latitude,
                aligned["lon"],
                aligned["lat"],
            ),
            aligned["lon"],
            aligned["lat"],
        )
    return aligned


def compute_standard_eval(args):
    config = args.config
    vc = variable_config_from_omegaconf(config)
    checkpoint_path = (
        args.checkpoint_path
        if args.checkpoint_path
        else Path(config.train.checkpoint_dir) / "best_model.pt"
    )
    model, preprocessor, _ = load_model_and_preprocessor(config, checkpoint_path)
    output_dir = args.output_dir
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    eval_years = [2012, 2013, 2015]
    raw_dataset = xr.open_mfdataset(
        generate_paths_yearly(config.dataset.era5.raw_path, eval_years, make_era5_filename),
        combine="nested",
        concat_dim="date",
    )
    raw_test_dataset = raw_dataset.sel(
        date=raw_dataset.date.dt.year.isin(config.dataset.test_years)
    )
    processed_dataset = xr.open_mfdataset(
        generate_paths_yearly(
            config.dataset.era5.path, config.dataset.test_years, make_era5_filename
        ),
        combine="nested",
        concat_dim="date",
    )

    preprocessor = Preprocessor(config)
    preprocessor.load(config.preprocess.params_dir)

    write_netcdf(
        results_dir / "tsr_test.nc",
        compute_tsr_summary(processed_dataset, preprocessor, model, vc),
    )
    write_netcdf(
        results_dir / "tsrc_test.nc",
        compute_tsrc_summary(raw_test_dataset, preprocessor, model, vc),
    )

    for kernel_name in eval_kernel_vars(config):
        kernels_dataset = xr.open_mfdataset(
            generate_paths_yearly(
                config.dataset.kernels.raw_path,
                eval_years,
                lambda year, var=kernel_name: make_kernel_filename(year, var),
            ),
            combine="nested",
            concat_dim="date",
        )
        date = "2015-09"
        write_netcdf(
            results_dir / f"kernel_date_{kernel_name}_{date}.nc",
            compute_kernel_date_result(
                raw_dataset.sel(date=[date]),
                kernels_dataset.sel(date=date),
                preprocessor,
                model,
                vc,
                date,
                kernel_name,
            ),
        )

        fields = compute_kernel_difference(
            raw_dataset,
            kernels_dataset,
            preprocessor,
            model,
            vc,
            "2012-09",
            "2013-09",
            kernel_name,
        )
        attrs = {
            "base_date": fields["base_date"],
            "perturbed_date": fields["perturbed_date"],
            "kernel_name": kernel_name,
        }
        write_netcdf(results_dir / f"second_order_{kernel_name}.nc", fields_to_dataset(fields, attrs))
        write_netcdf(
            results_dir / f"hybrid_second_order_{kernel_name}.nc",
            fields_to_dataset(
                align_hybrid_fields(
                    compute_hybrid_kernel_difference(
                        fields, preprocessor, model, vc, kernel_name
                    )
                ),
                attrs,
            ),
        )
        kernels_dataset.close()

    write_netcdf(
        results_dir / "kernel_contour_fal_ecod.nc",
        compute_kernel_contour(
            processed_dataset,
            preprocessor,
            model,
            vc,
            var_a="fal",
            var_b="ecod",
            date="2015-09",
        ),
    )
    write_netcdf(
        results_dir / "tsr_contour_fal_ecod.nc",
        compute_tsr_contour(
            processed_dataset,
            preprocessor,
            model,
            vc,
            var_a="fal",
            var_b="ecod",
            date="2015-09",
        ),
    )
    return output_dir


def main():
    args = parse_args_and_confirm(
        StandardEvalArgs(description="Test the trained model on validation data.")
    )
    print("=" * 20 + " computation " + "=" * 20)
    output_dir = compute_standard_eval(args)
    print("=" * 20 + " plotting " + "=" * 20)
    plot_standard_eval_results(output_dir)


if __name__ == "__main__":
    main()
