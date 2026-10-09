import sys
from pathlib import Path

import xarray as xr
from tap import Tap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config, variable_config_from_omegaconf
from experiments.common import (
    checkpoint_paths_for_args,
    model_label,
    output_path,
    parse_args_and_confirm,
    write_netcdf,
)
from utils import (
    compute_nn_kernel,
    compute_nn_kernel_autograd,
    generate_paths_yearly,
    kernel_delta,
    load_model_and_preprocessor,
    make_era5_filename,
    make_kernel_filename,
)


class SensitivityPredictionArgs(Tap):
    config_file: Path
    checkpoint_path: list[Path] | None = None
    seeds: list[str] | None = None
    output_dir: Path | None = None
    era5_data_path: Path = Path("data/era5")
    kernel_data_path: Path | None = None
    dates: list[str]
    variable: str = "fal"
    clear_sky: bool = False
    autograd: bool = False

    def process_args(self):
        if self.seeds and self.checkpoint_path:
            self.error("--seeds and --checkpoint_path are mutually exclusive")


def collect_sensitivity_input_data(data_path: Path, dates: list[str]) -> xr.Dataset:
    years = sorted({int(date[:4]) for date in dates})
    return xr.open_mfdataset(
        generate_paths_yearly(data_path, years, make_era5_filename),
        combine="nested",
        concat_dim="date",
    ).sel(date=dates)


def collect_sensitivity_truth_data(
    kernel_path: Path, dates: list[str], variable: str, clear_sky: bool
) -> xr.DataArray:
    years = sorted({int(date[:4]) for date in dates})
    kernels = xr.open_mfdataset(
        generate_paths_yearly(
            kernel_path, years, lambda year: make_kernel_filename(year, variable)
        ),
        combine="nested",
        concat_dim="date",
    )
    sky = "clr" if clear_sky else "all"
    return (
        kernels[variable]
        .sel(date=dates, all_clr=sky)
        * kernel_delta(variable)
    )


def predict_sensitivity(ds, preprocessor, model, vc, args, truth):
    raw = ds.interp(latitude=truth.latitude, longitude=truth.longitude)
    if args.autograd:
        field, lon, lat = compute_nn_kernel_autograd(
            raw,
            preprocessor,
            model,
            vc,
            var=args.variable,
            clear=args.clear_sky,
        )
        return sensitivity_field(field, raw.date, lon, lat)
    field, lon, lat = compute_nn_kernel(
        raw,
        preprocessor,
        model,
        vc,
        clear=args.clear_sky,
        perturbation_var=args.variable,
    )
    return sensitivity_field(field, raw.date, lon, lat)


def sensitivity_field(field, dates, lon, lat):
    if field.ndim == 2:
        return xr.DataArray(
            field,
            coords={"latitude": lat, "longitude": lon},
            dims=("latitude", "longitude"),
        ).expand_dims(date=dates)
    return xr.DataArray(
        field,
        coords={"date": dates, "latitude": lat, "longitude": lon},
        dims=("date", "latitude", "longitude"),
    )


def predict_sensitivity_for_models(ds, config, vc, args, truth, checkpoint_paths):
    predictions = []
    labels = []
    for checkpoint_path in checkpoint_paths:
        model, preprocessor, _ = load_model_and_preprocessor(
            config, checkpoint_path, downscaling=False
        )
        predictions.append(predict_sensitivity(ds, preprocessor, model, vc, args, truth))
        labels.append(model_label(checkpoint_path))
    if len(predictions) == 1:
        return predictions[0]
    return xr.concat(predictions, dim=xr.IndexVariable("model", labels))


def main():
    args = parse_args_and_confirm(SensitivityPredictionArgs())
    config = load_config(args.config_file)
    vc = variable_config_from_omegaconf(config)
    output_dir = output_path(args, "sensitivity_prediction")

    ds = collect_sensitivity_input_data(args.era5_data_path, args.dates)
    truth = collect_sensitivity_truth_data(
        Path(args.kernel_data_path or config.dataset.kernels.raw_path),
        args.dates,
        args.variable,
        args.clear_sky,
    )
    prediction = predict_sensitivity_for_models(
        ds, config, vc, args, truth, checkpoint_paths_for_args(args, config)
    )

    write_netcdf(
        output_dir / f"{args.variable}.nc",
        xr.Dataset({"prediction": prediction, "truth": truth}),
    )


if __name__ == "__main__":
    main()
