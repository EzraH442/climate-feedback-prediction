import sys
from pathlib import Path

import xarray as xr
from omegaconf import DictConfig
from tap import Tap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config, variable_config_from_omegaconf
from experiments.common import write_netcdf
from utils import load_model_and_preprocessor, nn_pred

dates = [f"{year}-{month}" for month in range(1, 13) for year in range(1990, 2021)]


class TSRTestArgs(Tap):
    config_file: Path
    checkpoint_path: Path
    output_base: str | None = None
    config: DictConfig | None = None

    era5_data_path: Path
    dates: list[str] = dates

    clear_sky: bool = False

    def process_args(self):
        self.config = load_config(self.config_file)
        self.vc = variable_config_from_omegaconf(self.config)
        self.checkpoint_path = Path(self.checkpoint_path)


def main():
    args = TSRTestArgs().parse_args()

    model, pp, _ = load_model_and_preprocessor(args.config, args.checkpoint_path)
    output_dir = Path(args.output_base or "tsr_prediction")
    output_dir.mkdir(parents=True, exist_ok=True)

    data = xr.open_dataset(args.era5_data_path).sel(date=args.dates)
    pred = nn_pred(data, model, pp, args.vc, clear=args.clear_sky)
    true = data[args.vc.get_target_var(args.clear_sky)]
    write_netcdf(
        output_dir / "fields.nc",
        xr.Dataset({"prediction": pred, "truth": true}),
    )


if __name__ == "__main__":
    main()
