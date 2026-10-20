# Example: python experiments/radiative_closure_single_date/analysis_table_kernel.py --response_path path/to/saved_responses_seed_1.nc path/to/saved_responses_seed_2.nc
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from tap import Tap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.common import parse_args_and_confirm
from utils import global_mean


class AnalysisTableKernelArgs(Tap):
    response_path: list[Path]
    arctic_boundary: float = 65.0


def load_responses(paths: list[Path]) -> xr.Dataset:
    datasets = []
    for path in paths:
        response = xr.load_dataset(path)
        if "model" not in response.dims:
            response = response.expand_dims(model=[str(path)])
        datasets.append(response)
    return xr.concat(datasets, dim="model")


def regional_metrics(
    prediction: xr.DataArray,
    truth: xr.DataArray,
    arctic_boundary: float,
) -> tuple[xr.DataArray, xr.DataArray]:
    residual = prediction - truth
    metrics = []
    for arctic in (True, False):
        field = residual.where(residual.latitude >= arctic_boundary, drop=True) if arctic else residual
        metrics.append((global_mean(field), np.sqrt(global_mean(field**2))))
    return tuple(zip(*metrics))


def format_metric(value: xr.DataArray, n_models: int) -> str:
    mean = float(value.mean("model"))
    if n_models == 1:
        return f"{mean:.3f}"
    std = float(value.std("model"))
    return f"{mean:.3f} +/- {std:.3f}"


def print_rows(rows: list[list[str]]) -> None:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    for row in rows:
        print("  ".join(value.ljust(width) for value, width in zip(row, widths)))


def main() -> None:
    args = parse_args_and_confirm(AnalysisTableKernelArgs())
    responses = load_responses(args.response_path)
    n_models = responses.sizes["model"]
    rows = [[
        "area",
        "prediction",
        "all-sky mbe",
        "clear-sky mbe",
        "all-sky rmse",
        "clear-sky rmse",
    ]]
    predictions = {
        "kernel": {
            "all": responses.dR_a_k_all + responses.dR_c_k_all + responses.dR_q_k_all,
            "clr": responses.dR_a_k_clr + responses.dR_q_k_clr,
        }
    }
    truths = {"all": responses.dR_era5_all, "clr": responses.dR_era5_clr}
    for area, index in (("arctic", 0), ("global", 1)):
        for name, prediction in predictions.items():
            all_mbe, all_rmse = regional_metrics(
                prediction["all"], truths["all"], args.arctic_boundary
            )
            clr_mbe, clr_rmse = regional_metrics(
                prediction["clr"], truths["clr"], args.arctic_boundary
            )
            rows.append([
                area,
                name,
                format_metric(all_mbe[index], n_models),
                format_metric(clr_mbe[index], n_models),
                format_metric(all_rmse[index], n_models),
                format_metric(clr_rmse[index], n_models),
            ])
    print_rows(rows)


if __name__ == "__main__":
    main()
