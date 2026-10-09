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
    SECONDS_PER_DAY,
    generate_paths_yearly,
    kernel_delta,
    load_model_and_preprocessor,
    make_era5_filename,
    nn_pred,
)


class BivariatePerturbationArgs(Tap):
    config_file: Path
    checkpoint_path: Path | None = None
    output_dir: Path | None = None
    era5_data_path: Path = Path("data/era5")
    date: str
    var_a: str = "fal"
    var_b: str = "tcwv"
    clear_sky: bool = False


def collect_bivariate_input_data(data_path: Path, date: str) -> xr.Dataset:
    return xr.open_mfdataset(
        generate_paths_yearly(data_path, [int(date[:4])], make_era5_filename),
        combine="nested",
        concat_dim="date",
    ).sel(date=[date])


def main():
    args = parse_args_and_confirm(BivariatePerturbationArgs())
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
    output_dir = output_path(args, "bivariate_perturbation")

    base = collect_bivariate_input_data(args.era5_data_path, args.date)
    if args.clear_sky:
        base = vc.clear_sky_input(base)

    da = kernel_delta(args.var_a)
    db = kernel_delta(args.var_b)
    a = base.assign({args.var_a: base[args.var_a] + da})
    b = base.assign({args.var_b: base[args.var_b] + db})
    ab = a.assign({args.var_b: a[args.var_b] + db})

    y0 = nn_pred(
        preprocessor.transform(base),
        model,
        preprocessor,
        vc,
        ["date", "latitude", "longitude"],
        clear=args.clear_sky,
    )
    ya = nn_pred(
        preprocessor.transform(a),
        model,
        preprocessor,
        vc,
        ["date", "latitude", "longitude"],
        clear=args.clear_sky,
    )
    yb = nn_pred(
        preprocessor.transform(b),
        model,
        preprocessor,
        vc,
        ["date", "latitude", "longitude"],
        clear=args.clear_sky,
    )
    yab = nn_pred(
        preprocessor.transform(ab),
        model,
        preprocessor,
        vc,
        ["date", "latitude", "longitude"],
        clear=args.clear_sky,
    )

    write_netcdf(
        output_dir / f"{args.date}_{args.var_a}_{args.var_b}.nc",
        xr.Dataset(
            {
                "base": y0 / SECONDS_PER_DAY,
                f"d_{args.var_a}": (ya - y0) / SECONDS_PER_DAY,
                f"d_{args.var_b}": (yb - y0) / SECONDS_PER_DAY,
                "d_both": (yab - y0) / SECONDS_PER_DAY,
            }
        ),
    )


if __name__ == "__main__":
    main()
