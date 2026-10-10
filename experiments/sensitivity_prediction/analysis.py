import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import xarray as xr
from tap import Tap

from experiments.common import parse_args_and_confirm
from utils import global_mean


class SensitivityPredictionAnalysisArgs(Tap):
    input_path: Path = Path("sensitivity_prediction/fal.nc")


def metrics(ds: xr.Dataset):
    residual = ds["prediction"] - ds["truth"]
    mbe = global_mean(residual).mean("date")
    rmse = np.sqrt(global_mean(residual**2).mean("date"))
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
    args = parse_args_and_confirm(SensitivityPredictionAnalysisArgs())
    mbe, rmse = metrics(xr.open_dataset(args.input_path))
    print_metrics(mbe, rmse)


if __name__ == "__main__":
    main()
