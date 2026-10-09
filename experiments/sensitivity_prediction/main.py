import sys
from pathlib import Path

import xarray as xr
from tap import Tap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config, variable_config_from_omegaconf
from experiments.common import (
    output_path,
    parse_args_and_confirm,
    write_netcdf,
)
from utils import (
    compute_nn_kernel,
    compute_nn_kernel_autograd,
    generate_paths_yearly,
    load_model_and_preprocessor,
    make_era5_filename,
)


class SensitivityPredictionArgs(Tap):
    config_file: Path
    checkpoint_path: Path | None = None
    output_dir: Path | None = None
    era5_data_path: Path = Path("data/era5")
    date: str
    variable: str = "fal"
    clear_sky: bool = False
    autograd: bool = False


def collect_sensitivity_input_data(data_path: Path, date: str) -> xr.Dataset:
    return xr.open_mfdataset(
        generate_paths_yearly(data_path, [int(date[:4])], make_era5_filename),
        combine="nested",
        concat_dim="date",
    ).sel(date=[date])


def main():
    args = parse_args_and_confirm(SensitivityPredictionArgs())
    config = load_config(args.config_file)
    vc = variable_config_from_omegaconf(config)
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else Path(config.train.checkpoint_dir) / "best_model.pt"
    )
    model, preprocessor, _ = load_model_and_preprocessor(
        config, checkpoint_path, downscaling=False
    )
    output_dir = output_path(args, "sensitivity_prediction")

    ds = collect_sensitivity_input_data(args.era5_data_path, args.date)
    if args.autograd:
        field, lon, lat = compute_nn_kernel_autograd(
            ds, preprocessor, model, vc, var=args.variable, clear=args.clear_sky
        )
    else:
        field, lon, lat = compute_nn_kernel(
            ds,
            preprocessor,
            model,
            vc,
            clear=args.clear_sky,
            perturbation_var=args.variable,
        )

    write_netcdf(
        output_dir / f"{args.date}_{args.variable}.nc",
        xr.DataArray(
            field,
            coords={"latitude": lat, "longitude": lon},
            dims=("latitude", "longitude"),
            name=f"dR_d{args.variable}",
        ),
    )


if __name__ == "__main__":
    main()
