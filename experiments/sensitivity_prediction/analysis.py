from pathlib import Path

import numpy as np
import xarray as xr
from tap import Tap

from experiments.common import parse_args_and_confirm
from utils import global_mean


class SensitivityPredictionAnalysisArgs(Tap):
    input_path: Path = Path("sensitivity_prediction/fal.nc")


def metrics(ds: xr.Dataset) -> tuple[float, float]:
    residual = ds["prediction"] - ds["truth"]
    mbe = float(global_mean(residual).mean("date"))
    rmse = float(np.sqrt(global_mean(residual**2).mean("date")))
    return mbe, rmse


def main() -> None:
    args = parse_args_and_confirm(SensitivityPredictionAnalysisArgs())
    mbe, rmse = metrics(xr.open_dataset(args.input_path))
    print(f"MBE: {mbe:.4f} W/m^2")
    print(f"RMSE: {rmse:.4f} W/m^2")


if __name__ == "__main__":
    main()
