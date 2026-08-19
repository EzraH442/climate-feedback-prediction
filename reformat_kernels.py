from datetime import datetime
from itertools import product
from pathlib import Path

import xarray as xr

from utils_cartopy import make_era5_filename, make_kernel_filename


def make_kernel_path_alb(year: int, month: str, band: str="sw", sky: str="cld"):
    assert band in ["sw", "lw"], "band must be 'sw' or 'lw'"
    assert sky in ["cld", "clr"], "sky must be 'cld' or 'clr'"

    return (
        Path(f"{year}")
        / "monthly_kernel"
        / band
        / "alb"
        / f"RRTM_kernel_{sky}_alb_TOA_SFC_{month}.nc"
    )


def make_kernel_path_ts(year: int, month: str, band: str="lw", sky: str="cld"):
    assert band in ["sw", "lw"], "band must be 'sw' or 'lw'"
    assert sky in ["cld", "clr"], "sky must be 'cld' or 'clr'"

    return (
        Path(f"{year}")
        / "monthly_kernel"
        / band
        / "ts"
        / f"RRTM_kernel_{sky}_ts_TOA_SFC_{month}.nc"
    )


def make_kernel_path_wv(year: int, month: str, band: str="sw", sky: str="cld"):
    assert band in ["sw", "lw"], "band must be 'sw' or 'lw'"
    assert sky in ["cld", "clr"], "sky must be 'cld' or 'clr'"

    return (
        Path(f"{year}")
        / "monthly_kernel"
        / band
        / "wv_dt_xxtscheme"
        / f"RRTM_kernel_{sky}_wv_TOA_SFC_{month}.nc"
    )


def assign_date_coord(ds: xr.Dataset, year: int, month: str):
    return ds.assign_coords(date=datetime(int(year), int(month), 1)).expand_dims(
        dim="date"
    )


def select_toa_arr(ds: xr.Dataset):
    return ds["TOA"].sel(up_down_net=3, band=1)


def build_albedo_kernel_dataset(base: Path, years: list[int], months: list[str]):
    year_months = list(product(years, months))
    kernel_dataarrays_clr = [
        select_toa_arr(
            assign_date_coord(
                xr.open_dataset(
                    base / make_kernel_path_alb(year, month, band="sw", sky="clr")
                ),
                year,
                month,
            )
        )
        for year, month in year_months
    ]
    kernel_dataarrays_cld = [
        select_toa_arr(
            assign_date_coord(
                xr.open_dataset(
                    Path(base) / make_kernel_path_alb(year, month, band="sw", sky="cld")
                ),
                year,
                month,
            )
        )
        for year, month in year_months
    ]
    kernel_ds_clr = xr.combine_by_coords(kernel_dataarrays_clr)
    kernel_ds_cld = xr.combine_by_coords(kernel_dataarrays_cld)
    ds = xr.Dataset(
        data_vars={
            "fal": (
                ["all_clr", "date", "latitude", "longitude"],
                [kernel_ds_cld.TOA.data, kernel_ds_clr.TOA.data],
            ),
        },
        coords={
            "latitude": ("latitude", kernel_ds_cld.latitude.data),
            "longitude": ("longitude", kernel_ds_clr.longitude.data),
            "date": ("date", kernel_ds_clr.date.data),
            "all_clr": ("all_clr", ["all", "clr"]),
        },
    )
    ds = ds.transpose("date", "latitude", "longitude", "all_clr")
    return ds


def build_skt_kernel_dataset(base: Path, years: list[int], months: list[str]):
    year_months = list(product(years, months))
    cld_arrays = [
        select_toa_arr(
            assign_date_coord(
                xr.open_dataset(
                    Path(base) / make_kernel_path_ts(year, month, band="lw", sky="cld")
                ),
                year,
                month,
            )
        )
        for year, month in year_months
    ]
    clr_arrays = [
        select_toa_arr(
            assign_date_coord(
                xr.open_dataset(
                    Path(base) / make_kernel_path_ts(year, month, band="lw", sky="clr")
                ),
                year,
                month,
            )
        )
        for year, month in year_months
    ]
    kernel_ds_cld = xr.combine_by_coords(cld_arrays)
    kernel_ds_clr = xr.combine_by_coords(clr_arrays)
    kernel_ds_cld = kernel_ds_cld["TOA"]
    kernel_ds_clr = kernel_ds_clr["TOA"]
    ds = xr.Dataset(
        data_vars={
            "skt": (
                ["all_clr", "date", "latitude", "longitude"],
                [kernel_ds_cld.data, kernel_ds_clr.data],
            ),
        },
        coords={
            "latitude": ("latitude", kernel_ds_cld.latitude.data),
            "longitude": ("longitude", kernel_ds_clr.longitude.data),
            "date": ("date", kernel_ds_clr.date.data),
            "all_clr": ("all_clr", ["all", "clr"]),
        },
    )
    ds = ds.transpose("date", "latitude", "longitude", "all_clr")
    return ds


def build_wv_kernel_dataset(base: Path, years: list[int], months: list[str], dtcwv: xr.DataArray, band: str="sw"):
    year_months = list(product(years, months))
    cld_arrays = []
    clr_arrays = []

    for year, month in year_months:
        ds_cld = xr.open_dataset(
            base / make_kernel_path_wv(year, month, band=band, sky="cld")
        )
        ds_clr = xr.open_dataset(
            base / make_kernel_path_wv(year, month, band=band, sky="clr")
        )

        ds_cld = select_toa_arr(ds_cld)
        ds_clr = select_toa_arr(ds_clr)

        ds_cld = ds_cld.sum(dim='level')
        ds_clr = ds_clr.sum(dim='level')

        ds_cld = assign_date_coord(ds_cld, year, month)
        ds_clr = assign_date_coord(ds_clr, year, month)

        ds_cld = ds_cld / dtcwv.sel(date=ds_cld.date)
        ds_clr = ds_clr / dtcwv.sel(date=ds_clr.date)

        cld_arrays.append(ds_cld)
        clr_arrays.append(ds_clr)

    kernel_ds_cld = xr.combine_by_coords(cld_arrays)
    kernel_ds_clr = xr.combine_by_coords(clr_arrays)

    ds = xr.Dataset(
        data_vars={
            "tcwv": (
                ["all_clr", "date", "latitude", "longitude"],
                [kernel_ds_cld.data, kernel_ds_clr.data],
            ),
        },
        coords={
            "latitude": ("latitude", kernel_ds_cld.latitude.data),
            "longitude": ("longitude", kernel_ds_clr.longitude.data),
            "date": ("date", kernel_ds_clr.date.data),
            "all_clr": ("all_clr", ["all", "clr"]),
        },
    )
    ds = ds.transpose("date", "latitude", "longitude", "all_clr")
    return ds


def load_tcwv(years: list[int]):
    dss = []
    for year in years:
        t_path = 'data/era5/' + make_era5_filename(year)
        dss.append(xr.open_dataset(t_path))
    ds = xr.concat(dss, dim="date").tcwv
    return ds

def load_surface_pressure(years):
    dss = []
    for year in years:
        sp_path = 'data/era5/' + make_era5_filename(year)
        dss.append(xr.open_dataset(sp_path))
    ds = xr.concat(dss, dim="date")
    return ds

def write_yearly_kernels(ds, output_dir, years, var):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for year in years:
        path = output_dir / make_kernel_filename(year, var)
        kernel_subset = ds.sel(date=(ds.date.dt.year == year))
        kernel_subset.to_netcdf(path)

base = Path("./data/kernels_shared")
years = [2011, 2012, 2013, 2014, 2015]
months = [f"{m:02d}" for m in range(1,13)]
print(months, years)

tcwv = load_tcwv(years)
dtcwv = tcwv * 0.1

fal_kernel_ds = build_albedo_kernel_dataset(base, years, months)
skt_kernel_ds = build_skt_kernel_dataset(base, years, months)
sw_tcwv_kernel_ds = build_wv_kernel_dataset(base, years, months, dtcwv, band='sw')
lw_tcwv_kernel_ds = build_wv_kernel_dataset(base, years, months, dtcwv, band='lw')

write_yearly_kernels(fal_kernel_ds, "data/fal/kernels", years, 'fal')
write_yearly_kernels(sw_tcwv_kernel_ds, "data/fal/kernels", years, 'tcwv')
write_yearly_kernels(skt_kernel_ds, "data/ts/kernels", years, 'ts')
write_yearly_kernels(lw_tcwv_kernel_ds, "data/ts/kernels", years, 'tcwv')
