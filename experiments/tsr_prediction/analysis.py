from pathlib import Path

import numpy as np
import xarray as xr
from tap import Tap

from experiments.common import parse_args_and_confirm
from utils import SECONDS_PER_DAY, global_mean


class TSRPredictionAnalysisArgs(Tap):
    input_path: Path = Path("tsr_prediction/fields.nc")


def metrics(ds: xr.Dataset) -> tuple[float, float]:
    prediction = ds["prediction"] / SECONDS_PER_DAY
    truth = ds["truth"] / SECONDS_PER_DAY
    residual = prediction - truth
    mbe = float(global_mean(residual.mean("date")))
    rmse = float(np.sqrt(global_mean((residual**2).mean("date"))))
    return mbe, rmse


def main() -> None:
    args = parse_args_and_confirm(TSRPredictionAnalysisArgs())
    mbe, rmse = metrics(xr.open_dataset(args.input_path))
    print(f"MBE: {mbe:.4f} W/m^2")
    print(f"RMSE: {rmse:.4f} W/m^2")


if __name__ == "__main__":
    main()
