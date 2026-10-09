from pathlib import Path

import numpy as np
import xarray as xr
from tap import Tap

from experiments.common import parse_args_and_confirm
from utils import SECONDS_PER_DAY, global_mean


class TSRPredictionAnalysisArgs(Tap):
    input_path: Path = Path("tsr_prediction/fields.nc")


def metrics(ds: xr.Dataset):
    prediction = ds["prediction"] / SECONDS_PER_DAY
    truth = ds["truth"] / SECONDS_PER_DAY
    residual = prediction - truth
    mbe = global_mean(residual.mean("date"))
    rmse = np.sqrt(global_mean((residual**2).mean("date")))
    return mbe, rmse


def print_metrics(mbe, rmse) -> None:
    if "model" in mbe.dims:
        print(
            f"MBE: {float(mbe.mean('model')):.4f} +/- {float(mbe.std('model')):.4f} W/m^2"
        )
        print(
            f"RMSE: {float(rmse.mean('model')):.4f} +/- {float(rmse.std('model')):.4f} W/m^2"
        )
        return
    print(f"MBE: {float(mbe):.4f} W/m^2")
    print(f"RMSE: {float(rmse):.4f} W/m^2")


def main() -> None:
    args = parse_args_and_confirm(TSRPredictionAnalysisArgs())
    mbe, rmse = metrics(xr.open_dataset(args.input_path))
    print_metrics(mbe, rmse)


if __name__ == "__main__":
    main()
