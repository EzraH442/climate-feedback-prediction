import sys
from pathlib import Path

import xarray as xr
from omegaconf import DictConfig
from tap import Tap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config, variable_config_from_omegaconf
from experiments.common import parse_args_and_confirm, write_netcdf
from utils import (
    generate_paths_yearly,
    load_model_and_preprocessor,
    make_era5_filename,
    nn_pred,
)

DEFAULT_DATES = [
    f"{year}-{month:02d}" for year in range(1990, 2021) for month in range(1, 13)
]


class TSRTestArgs(Tap):
    config_file: Path
    checkpoint_path: Path
    output_base: str | None = None
    config: DictConfig | None = None

    era5_data_path: Path = Path("data/era5")
    dates: list[str] = DEFAULT_DATES

    clear_sky: bool = False

    def process_args(self):
        self.config = load_config(self.config_file)
        self.vc = variable_config_from_omegaconf(self.config)
        self.checkpoint_path = Path(self.checkpoint_path)
        self.era5_data_path = Path(self.era5_data_path)


def collect_tsr_input_data(data_path: Path, dates: list[str]) -> xr.Dataset:
    years = sorted({int(date[:4]) for date in dates})
    return xr.open_mfdataset(
        generate_paths_yearly(data_path, years, make_era5_filename),
        combine="nested",
        concat_dim="date",
    ).sel(date=dates)


def main():
    args = parse_args_and_confirm(TSRTestArgs())

    model, pp, _ = load_model_and_preprocessor(args.config, args.checkpoint_path)
    output_dir = Path(args.output_base or "tsr_prediction")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = collect_tsr_input_data(args.era5_data_path, args.dates)
    model_input = args.vc.clear_sky_input(data) if args.clear_sky else data
    pred = nn_pred(pp.transform(model_input), model, pp, args.vc, clear=args.clear_sky)
    true = data[args.vc.get_target_var(args.clear_sky)]
    write_netcdf(
        output_dir / "fields.nc",
        xr.Dataset({"prediction": pred, "truth": true}),
    )


if __name__ == "__main__":
    main()
