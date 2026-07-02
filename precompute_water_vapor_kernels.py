from pathlib import Path
import argparse

import xarray as xr

from preprocess import make_era5_filename, make_water_vapor_kernel_filename
from utils import filter_by_months, integrate_over_pressure_levels


Rv = 461.52
Lv = 2260000


def make_qt_filename(year):
    return f"era5_plev_qt_monthly_{year}.nc"


def water_vapor_response(ds: xr.Dataset, ds_qt: xr.Dataset, kernel_field: xr.DataArray):
    qt_anomaly = ds_qt - ds_qt.mean("date")
    month_index = xr.DataArray(
        ds_qt.date.dt.month.data,
        dims="date",
        coords={"date": ds_qt.date},
    )
    kernel_for_date = kernel_field.sel(month=month_index)

    dT = (qt_anomaly.q / ds_qt.q) * (Rv / Lv) * (ds_qt.t**2)
    dR = integrate_over_pressure_levels(ds.sp, kernel_for_date * dT)
    return dR.compute()


def raw_era5_years(path: str) -> list[int]:
    prefix = "era5_single_levels_monthly_"
    return sorted(
        int(p.stem.removeprefix(prefix))
        for p in Path(path).glob(f"{prefix}*.nc")
    )


def build_water_vapor_kernel(
    era5_raw_path: str,
    era5_plev_path: str,
    wv_kernel_path: str,
    output_dirs: list[str],
    months=None,
):
    years = raw_era5_years(era5_raw_path)
    era5_paths = [Path(era5_raw_path) / make_era5_filename(y) for y in years]
    ds = xr.open_mfdataset(era5_paths, combine="nested", concat_dim="date")
    ds = filter_by_months(ds, months)

    kernel = xr.open_dataset(wv_kernel_path)
    ds = ds.interp(latitude=kernel.latitude, longitude=kernel.longitude)
    anomaly = ds - ds.mean("date")

    qt_paths = [Path(era5_plev_path) / make_qt_filename(y) for y in years]
    ds_qt = xr.open_mfdataset(qt_paths, combine="nested", concat_dim="date")
    ds_qt = filter_by_months(ds_qt, months)
    ds_qt = ds_qt.sel(date=ds.date.values)
    ds_qt = ds_qt.interp(latitude=kernel.latitude, longitude=kernel.longitude)
    dtcwv = anomaly.tcwv

    toa_cld = water_vapor_response(ds, ds_qt, kernel.TOA_all) / dtcwv.where(
        abs(dtcwv) > 1e-6
    )
    toa_clr = water_vapor_response(ds, ds_qt, kernel.TOA_clr) / dtcwv.where(
        abs(dtcwv) > 1e-6
    )
    # ponytail: zero near-constant tcwv anomalies; mask samples if this bias matters.
    out = xr.Dataset(
        {
            "TOA_cld": toa_cld.fillna(0),
            "TOA_clr": toa_clr.fillna(0),
        }
    )
    out["TOA_cld"].attrs.update(units="W m-2", description="All-sky dR/dtcwv")
    out["TOA_clr"].attrs.update(units="W m-2", description="Clear-sky dR/dtcwv")
    output_dirs = [Path(output_dir) for output_dir in output_dirs]
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        yearly = out.sel(date=str(year))
        for output_dir in output_dirs:
            output_path = output_dir / make_water_vapor_kernel_filename(year)
            print(f"Saving {output_path}")
            yearly.to_netcdf(output_path)


def main():
    parser = argparse.ArgumentParser(description="Precompute raw water-vapor Sobolev kernel")
    parser.add_argument("--era5-raw-path", default="data/era5")
    parser.add_argument(
        "--era5-plev-path",
        default="data/era5",
    )
    parser.add_argument(
        "--wv-kernel-path",
        default="data/ERA5_kernels/layer_specified_ta_wv_kernel/ERA5_kernel_wv_sw_nodp_TOA.nc",
    )
    parser.add_argument(
        "--output-dirs",
        nargs="+",
        default=["data/fal/kernels", "data/ts/kernels"],
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
    )
    parser.add_argument("--months", nargs="*", type=int)
    args = parser.parse_args()
    build_water_vapor_kernel(
        args.era5_raw_path,
        args.era5_plev_path,
        args.wv_kernel_path,
        [args.output_dir] if args.output_dir else args.output_dirs,
        args.months,
    )


if __name__ == "__main__":
    main()
