import argparse
from pathlib import Path

import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create the downscaled monthly ERA5 pressure-level q/t file for closure tests."
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/era5"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/era5/era5_plev_qt_monthly_downscaled_neurips_1990_2020.nc"),
    )
    parser.add_argument(
        "--kernel",
        type=Path,
        default=Path("data/ERA5_kernels/ERA5_kernel_alb_TOA.nc"),
    )
    parser.add_argument("--start-year", type=int, default=1990)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-dtype", action="store_true")
    return parser.parse_args()


def make_qt_filename(year):
    return f"era5_plev_qt_monthly_{year}.nc"


def main():
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it.")

    years = range(args.start_year, args.end_year + 1)
    paths = [args.input_dir / make_qt_filename(year) for year in years]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing QT files: {missing[:5]}{' ...' if len(missing) > 5 else ''}")

    args.output.parent.mkdir(exist_ok=True, parents=True)
    kernel = xr.open_dataset(args.kernel)
    grid = {"latitude": kernel.latitude, "longitude": kernel.longitude}
    qt = xr.open_mfdataset(paths, combine="nested", concat_dim="date")
    downscaled = qt.interp(**grid).sortby("date")
    if not args.keep_dtype:
        downscaled = downscaled.astype({name: "float32" for name in downscaled.data_vars})

    downscaled.to_netcdf(args.output)
    qt.close()
    kernel.close()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()

"""
python make_neurips_downscaled_qt.py \
  --input-dir data/era5 \
  --output data/era5/era5_plev_qt_monthly_downscaled_neurips_1990_2020.nc \
  --kernel data/ERA5_kernels/ERA5_kernel_alb_TOA.nc \
  --start-year 1990 \
  --end-year 2020 \
  --overwrite

"""