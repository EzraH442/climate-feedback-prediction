import sys
from pathlib import Path

import xarray as xr
from tap import Tap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config, variable_config_from_omegaconf
from experiments.common import output_path, parse_args_and_confirm
from experiments.kernel_difference.analysis import (
    plot_hybrid_kernel_difference,
    plot_kernel_difference,
)
from preprocessing import Preprocessor
from utils import (
    compute_nn_kernel_autograd,
    generate_paths_yearly,
    interpolate_spatial_field,
    kernel_delta,
    load_model_and_preprocessor,
    make_era5_filename,
    make_kernel_filename,
)


class KernelDifferenceArgs(Tap):
    config_file: Path
    checkpoint_path: Path | None = None
    output_dir: Path | None = None
    era5_data_path: Path | None = None
    kernel_data_path: Path | None = None
    base_date: str = "2012-09"
    perturbed_date: str = "2013-09"
    variable: str = "fal"


def select_date_state(ds: xr.Dataset, date: str) -> xr.Dataset:
    selected = ds.sel(date=date)
    if "date" in selected.dims:
        selected = selected.squeeze("date", drop=True)
    return selected


def as_field(values, lon, lat):
    return xr.DataArray(
        values,
        coords={"longitude": lon, "latitude": lat},
        dims=("latitude", "longitude"),
    )


def compute_kernel_difference(
    raw: xr.Dataset,
    kernels: xr.Dataset,
    preprocessor: Preprocessor,
    model,
    vc,
    base_date: str,
    perturbed_date: str,
    variable: str,
) -> dict:
    base = select_date_state(raw, base_date)
    perturbed = select_date_state(raw, perturbed_date)
    nn_perturbed, lon, lat = compute_nn_kernel_autograd(
        perturbed.expand_dims(date=[perturbed_date]),
        preprocessor,
        model,
        vc,
        var=variable,
    )
    nn_base, _, _ = compute_nn_kernel_autograd(
        base.expand_dims(date=[base_date]),
        preprocessor,
        model,
        vc,
        var=variable,
    )

    rrtm_lat, rrtm_lon = kernels.latitude, kernels.longitude
    rrtm_perturbed = (
        kernels[variable].sel(all_clr="all", date=perturbed_date).squeeze(drop=True)
        * kernel_delta(variable)
    )
    rrtm_base = (
        kernels[variable].sel(all_clr="all", date=base_date).squeeze(drop=True)
        * kernel_delta(variable)
    )

    if nn_base.shape != rrtm_base.shape:
        nn_base = interpolate_spatial_field(nn_base, lon, lat, rrtm_lon, rrtm_lat)
        nn_perturbed = interpolate_spatial_field(
            nn_perturbed, lon, lat, rrtm_lon, rrtm_lat
        )
        lon, lat = rrtm_lon, rrtm_lat

    nn_base = as_field(nn_base, lon, lat)
    nn_perturbed = as_field(nn_perturbed, lon, lat)
    rrtm_base = rrtm_base.interp(latitude=lat, longitude=lon)
    rrtm_perturbed = rrtm_perturbed.interp(latitude=lat, longitude=lon)
    nn_diff = nn_perturbed - nn_base
    rrtm_diff = rrtm_perturbed - rrtm_base

    return {
        "base_date": base_date,
        "perturbed_date": perturbed_date,
        "base": base,
        "perturbed": perturbed,
        "lon": lon,
        "lat": lat,
        "nn_base": nn_base,
        "nn_perturbed": nn_perturbed,
        "nn_diff": nn_diff,
        "rrtm_base": rrtm_base,
        "rrtm_perturbed": rrtm_perturbed,
        "rrtm_diff": rrtm_diff,
        "bias": nn_diff - rrtm_diff,
    }


def compute_hybrid_kernel_difference(fields, preprocessor, model, vc, variable) -> dict:
    base = fields["base"]
    perturbed = fields["perturbed"]
    nn_base = fields["nn_base"]
    lon = fields["lon"]
    lat = fields["lat"]
    processed_base = preprocessor.transform(base)
    processed_perturbed = preprocessor.transform(perturbed)

    hybrid_fields = {}
    for label, variables in (
        ("hybrid_albedo", ["fal"]),
        ("hybrid_cloud", ["tcc", "tclw", "tciw", "hcc", "mcc", "lcc"]),
    ):
        available = [var for var in variables if var in perturbed]
        hybrid = base.assign({var: perturbed[var] for var in available})
        kernel, kernel_lon, kernel_lat = compute_nn_kernel_autograd(
            hybrid.expand_dims(date=[fields["base_date"]]),
            preprocessor,
            model,
            vc,
            var=variable,
        )
        if kernel.shape != nn_base.shape:
            kernel = interpolate_spatial_field(kernel, kernel_lon, kernel_lat, lon, lat)
        hybrid_fields[label] = as_field(kernel, lon, lat) - nn_base

    return {
        **fields,
        **hybrid_fields,
        "delta_fal": perturbed.fal - base.fal,
        "delta_ecod": processed_perturbed.ecod - processed_base.ecod,
    }


def collect_kernel_difference_input_data(data_path: Path, dates: list[str]) -> xr.Dataset:
    years = sorted({int(date[:4]) for date in dates})
    return xr.open_mfdataset(
        generate_paths_yearly(data_path, years, make_era5_filename),
        combine="nested",
        concat_dim="date",
    )


def collect_kernel_difference_kernel_data(
    kernel_path: Path, dates: list[str], variable: str
) -> xr.Dataset:
    years = sorted({int(date[:4]) for date in dates})
    return xr.open_mfdataset(
        generate_paths_yearly(
            kernel_path,
            years,
            lambda year: make_kernel_filename(year, variable),
        ),
        combine="nested",
        concat_dim="date",
    )


def run_kernel_difference(args) -> None:
    config = load_config(args.config_file)
    vc = variable_config_from_omegaconf(config)
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else Path(config.train.checkpoint_dir) / "best_model.pt"
    )
    model, preprocessor, _ = load_model_and_preprocessor(
        config, checkpoint_path, downscaling=False
    )

    dates = [args.base_date, args.perturbed_date]
    era5_path = args.era5_data_path or Path(config.dataset.era5.raw_path)
    kernel_path = args.kernel_data_path or Path(config.dataset.kernels.raw_path)
    raw = collect_kernel_difference_input_data(era5_path, dates)
    kernels = collect_kernel_difference_kernel_data(kernel_path, dates, args.variable)
    output_dir = output_path(args, "kernel_difference")

    fields = compute_kernel_difference(
        raw,
        kernels,
        preprocessor,
        model,
        vc,
        args.base_date,
        args.perturbed_date,
        args.variable,
    )
    plot_kernel_difference(fields, output_dir)
    plot_hybrid_kernel_difference(
        compute_hybrid_kernel_difference(fields, preprocessor, model, vc, args.variable),
        output_dir,
    )


def main():
    run_kernel_difference(parse_args_and_confirm(KernelDifferenceArgs()))


if __name__ == "__main__":
    main()
