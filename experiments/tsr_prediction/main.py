# Example: python experiments/tsr_prediction/main.py --config_file configs/model/fal/2011-2014_3,6,9,12_baseline.yaml
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from omegaconf import DictConfig
from tap import Tap
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config, variable_config_from_omegaconf
from experiments.common import (
    checkpoint_paths_for_args,
    model_label,
    parse_args_and_confirm,
)
from utils import (
    SECONDS_PER_DAY,
    generate_paths_yearly,
    global_mean,
    load_model_and_preprocessor,
    make_era5_filename,
    nn_pred,
)

TSR_DATE_BATCH_SIZE = 12

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
    prediction = nn_pred(
        processed_data if processed_data is not None else preprocessor.transform(model_input),
        model,
        preprocessor,
        vc,
        dim_names,
        clear=clear_sky,
    )
    return prediction.interp(
        latitude=data.latitude,
        longitude=data.longitude,
        method="nearest",
        kwargs={"fill_value": "extrapolate"},
    )


def compute_tsr_metrics_for_models(
    args, data: xr.Dataset, checkpoint_paths: list[Path]
) -> tuple[xr.DataArray, xr.DataArray]:
    loaded_models = [
        load_model_and_preprocessor(args.config, checkpoint_path)
        for checkpoint_path in checkpoint_paths
    ]
    labels = [model_label(checkpoint_path) for checkpoint_path in checkpoint_paths]
    mbe_total = None
    mse_total = None
    n_dates = 0

    for start in tqdm(
        range(0, data.sizes["date"], TSR_DATE_BATCH_SIZE),
        desc="Evaluating TSR batches",
    ):
        batch = data.isel(date=slice(start, start + TSR_DATE_BATCH_SIZE))
        model_input = args.vc.clear_sky_input(batch) if args.clear_sky else batch
        processed_data = loaded_models[0][1].transform(model_input)
        predictions = [
            predict_tsr(
                batch,
                model,
                pp,
                args.vc,
                clear_sky=args.clear_sky,
                processed_data=processed_data,
            )
            for model, pp, _ in loaded_models
        ]
        if len(predictions) == 1:
            predictions_da = predictions[0].expand_dims(model=[labels[0]])
        else:
            predictions_da = xr.concat(
                predictions, dim=xr.IndexVariable("model", labels)
            )

        truth = batch[args.vc.get_target_var(args.clear_sky)] / SECONDS_PER_DAY
        residual = predictions_da / SECONDS_PER_DAY - truth
        batch_dates = batch.sizes["date"]
        batch_mbe = global_mean(residual.mean("date"))
        batch_mse = global_mean((residual**2).mean("date"))
        mbe_total = (
            batch_mbe * batch_dates
            if mbe_total is None
            else mbe_total + batch_mbe * batch_dates
        )
        mse_total = (
            batch_mse * batch_dates
            if mse_total is None
            else mse_total + batch_mse * batch_dates
        )
        n_dates += batch_dates

    mbe = mbe_total / n_dates
    rmse = np.sqrt(mse_total / n_dates)
    return mbe, rmse


def print_tsr_metrics(mbe: xr.DataArray, rmse: xr.DataArray) -> None:
    if mbe.sizes["model"] == 1:
        print(f"MBE: {float(mbe.squeeze('model')):.4f} W/m^2")
        print(f"RMSE: {float(rmse.squeeze('model')):.4f} W/m^2")
        return
    print(
        f"MBE: {float(mbe.mean('model')):.4f} +/- "
        f"{float(mbe.std('model')):.4f} W/m^2"
    )
    print(
        f"RMSE: {float(rmse.mean('model')):.4f} +/- "
        f"{float(rmse.std('model')):.4f} W/m^2"
    )


def main():
    args: TSRTestArgs = parse_args_and_confirm(TSRTestArgs())

    data = collect_tsr_input_data(args.era5_data_path, args.dates)
    mbe, rmse = compute_tsr_metrics_for_models(
        args, data, checkpoint_paths_for_args(args, args.config)
    )
    print_tsr_metrics(mbe, rmse)


if __name__ == "__main__":
    main()
