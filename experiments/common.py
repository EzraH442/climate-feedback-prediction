import json
import shlex
import sys
from pathlib import Path

import xarray as xr

from utils import SECONDS_PER_DAY


def parse_args_and_confirm(parser):
    args = parser.parse_args()
    print("Command:", shlex.join(sys.argv))
    print("Parsed args:")
    for key, value in vars(args).items():
        if key.startswith("_") or key in {"config", "vc"}:
            continue
        print(f"  {key}: {value!r}")
    if input("Continue? [y/N] ").strip().lower() not in {"y", "yes"}:
        raise SystemExit("Aborted.")
    return args


def output_path(args, default_name: str) -> Path:
    path = Path(args.output_dir or default_name)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def write_netcdf(path: Path, data: xr.Dataset | xr.DataArray) -> None:
    data.to_netcdf(path)


def with_flux_targets(ds: xr.Dataset) -> xr.Dataset:
    return ds.assign(
        tsr=ds.tsr / SECONDS_PER_DAY,
        tsrc=ds.tsrc / SECONDS_PER_DAY,
    )


def albedo_kernel_components(da, albedo_kernel) -> tuple[xr.DataArray, xr.DataArray]:
    dR_a_k = albedo_kernel.TOA_all * da * 100
    dR_a_k_clr = albedo_kernel.TOA_clr * da * 100
    return dR_a_k, dR_a_k_clr


def parse_seed_ranges(seed_ranges):
    seeds = []
    for seed_range in seed_ranges:
        if "-" in seed_range:
            start, end = map(int, seed_range.split("-", 1))
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(seed_range))
    return seeds


def checkpoint_paths_for_args(args, config) -> list[Path]:
    if args.seeds:
        return [
            Path(f"{config.train.checkpoint_dir}_seed_{seed}") / "best_model.pt"
            for seed in parse_seed_ranges(args.seeds)
        ]
    if args.checkpoint_path:
        return [Path(path) for path in args.checkpoint_path]
    return [Path(config.train.checkpoint_dir) / "best_model.pt"]


def model_label(checkpoint_path: Path) -> str:
    return checkpoint_path.parent.name
