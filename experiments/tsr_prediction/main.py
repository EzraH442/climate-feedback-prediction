# Example: python experiments/tsr_prediction/main.py --config_file configs/model/fal/2011-2014_3,6,9,12_baseline.yaml
import sys
from pathlib import Path

import xarray as xr
from omegaconf import DictConfig
from tap import Tap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config, variable_config_from_omegaconf
from experiments.common import (
    checkpoint_paths_for_args,
    model_label,
    parse_args_and_confirm,
    write_netcdf,
)
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
    checkpoint_path: list[Path] | None = None
    seeds: list[str] | None = None
    output_dir: Path | None = None
    config: DictConfig | None = None

    era5_data_path: Path = Path("data/era5")
    dates: list[str] = DEFAULT_DATES

    clear_sky: bool = False

    def process_args(self):
        if self.seeds and self.checkpoint_path:
            self.error("--seeds and --checkpoint_path are mutually exclusive")
        self.config = load_config(self.config_file)
        self.vc = variable_config_from_omegaconf(self.config)
        self.era5_data_path = Path(self.era5_data_path)
        self.output_dir = Path(self.output_dir or "tsr_prediction")
        self.output_dir.mkdir(parents=True, exist_ok=True)


def collect_tsr_input_data(data_path: Path, dates: list[str]) -> xr.Dataset:
    years = sorted({int(date[:4]) for date in dates})
    return xr.open_mfdataset(
        generate_paths_yearly(data_path, years, make_era5_filename),
        combine="nested",
        concat_dim="date",
    ).sel(date=dates)


def predict_tsr(
    data: xr.Dataset,
    model,
    preprocessor,
    vc,
    clear_sky: bool = False,
    processed_data: xr.Dataset | None = None,
) -> xr.DataArray:
    model_input = vc.clear_sky_input(data) if clear_sky else data
    dim_names = [dim for dim in ["date", "latitude", "longitude"] if dim in data.dims]
    return nn_pred(
        processed_data if processed_data is not None else preprocessor.transform(model_input),
        model,
        preprocessor,
        vc,
        dim_names,
        clear=clear_sky,
    )


def predict_tsr_for_models(
    args, data: xr.Dataset, checkpoint_paths: list[Path]
) -> xr.DataArray:
    loaded_models = [
        load_model_and_preprocessor(args.config, checkpoint_path)
        for checkpoint_path in checkpoint_paths
    ]
    model_input = args.vc.clear_sky_input(data) if args.clear_sky else data
    processed_data = loaded_models[0][1].transform(model_input)

    predictions = []
    labels = []
    for checkpoint_path, (model, pp, _) in zip(checkpoint_paths, loaded_models):
        predictions.append(
            predict_tsr(
                data,
                model,
                pp,
                args.vc,
                clear_sky=args.clear_sky,
                processed_data=processed_data,
            )
        )
        labels.append(model_label(checkpoint_path))
    if len(predictions) == 1:
        return predictions[0]
    return xr.concat(predictions, dim=xr.IndexVariable("model", labels))


def main():
    args: TSRTestArgs = parse_args_and_confirm(TSRTestArgs())

    data = collect_tsr_input_data(args.era5_data_path, args.dates)
    pred = predict_tsr_for_models(args, data, checkpoint_paths_for_args(args, args.config))
    true = data[args.vc.get_target_var(args.clear_sky)]
    write_netcdf(
        args.output_dir / "fields.nc",
        xr.Dataset({"prediction": pred, "truth": true}),
    )


if __name__ == "__main__":
    main()
