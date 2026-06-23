from pathlib import Path
import zipfile

import cdsapi
import cfgrib
import xarray as xr


ERA5_SINGLE_LEVELS_VARIABLES = [
    "skin_temperature",
    "forecast_albedo",
    "total_cloud_cover",
    "high_cloud_cover",
    "medium_cloud_cover",
    "low_cloud_cover",
    "surface_pressure",
    "total_column_cloud_ice_water",
    "total_column_cloud_liquid_water",
    "total_column_ozone",
    "total_column_water_vapour",
    "toa_incident_solar_radiation",
    "top_net_solar_radiation",
    "top_net_thermal_radiation",
]

MONTHS = [f"{m:02d}" for m in range(1, 13)]
THREE_HOUR_TIMES = [
    "00:00",
    "03:00",
    "06:00",
    "09:00",
    "12:00",
    "15:00",
    "18:00",
    "21:00",
]

c = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",
    key="3dbc229c-81f7-4fd5-be2f-c6f193729ef5",
)


def retrieve_era5_years_data(years=(2006, 2008), data_path="data/era5"):
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    for year in years:
        path = data_path / f"era5_single_levels_monthly_{year}.grib"
        if path.exists():
            print(f"found existing data for year {year}, skipping")
            continue

        c.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable": ERA5_SINGLE_LEVELS_VARIABLES,
                "year": year,
                "month": MONTHS,
                "time": "00:00",
            },
            str(path),
        )

def retrieve_era5_years_t2m(years=(2006, 2008), data_path="data/era5"):
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    for year in years:
        path = data_path / f"era5_t2m_monthly_{year}.grib"
        if path.exists():
            print(f"found existing data for year {year}, skipping")
            continue

        c.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable": ["2m_temperature"],
                "year": year,
                "month": MONTHS,
                "time": "00:00",
            },
            str(path),
        )

def retrieve_era5_years_tsrc(years=(2006, 2008), data_path="data/era5"):
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    for year in years:
        path = data_path / f"era5_tsrc_monthly_{year}.grib"
        if path.exists():
            print(f"found existing data for year {year}, skipping")
            continue

        c.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable": ["top_net_solar_radiation_clear_sky"],
                "year": year,
                "month": MONTHS,
                "time": "00:00",
            },
            str(path),
        )

def retrieve_era5_plevl_qt(years=(2006, 2008), data_path="data/era5"):
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)
    
    for year in years:
        path = data_path / f"era5_plev_qt_monthly_{year}.grib"
        if path.exists():
            print(f"found existing data for year {year}, skipping")
            continue

        c.retrieve(
            "reanalysis-era5-pressure-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "pressure_level": [
                    "1", "2", "3",
                    "5", "7", "10",
                    "20", "30", "50",
                    "70", "100", "125",
                    "150", "175", "200",
                    "225", "250", "300",
                    "350", "400", "450",
                    "500", "550", "600",
                    "650", "700", "750",
                    "775", "800", "825",
                    "850", "875", "900",
                    "925", "950", "975",
                    "1000"
                ],
                "variable": ["specific_humidity", "temperature"],
                "year": year,
                "month": MONTHS,
                "time": "00:00",
                "data_format": "grib",
                "download_format": "unarchived"
            },
            str(path),
        )

def load_all_era5_data(years):
    datasets = []
    for year in years:
        datasets_list = cfgrib.open_datasets(
            f"data/era5/era5_single_levels_monthly_{year}.grib"
        )
        datasets.append(
            xr.combine_by_coords(
                datasets_list,
                join="override",
                compat="override",
                combine_attrs="drop_conflicts",
            )
        )

    return xr.concat(datasets, dim="time")


def retrieve_era5_single_levels_3hr(years, data_path="data/era5_3hr"):
    data_path = Path(data_path)
    data_path.mkdir(parents=True, exist_ok=True)

    for year in years:
        for month in MONTHS:
            print(year, month)
            path = data_path / f"era5_single_levels_monthly_{year}_{month}.grib"
            if path.exists():
                print(f"found existing data for year {year} month {month}, skipping")
                continue

            c.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": ["reanalysis"],
                    "variable": ERA5_SINGLE_LEVELS_VARIABLES,
                    "year": year,
                    "month": month,
                    "day": [f"{d:02d}" for d in range(1, 32)],
                    "time": THREE_HOUR_TIMES,
                    "data_format": "grib",
                },
                str(path),
            )


def retrieve_era5_daily_stats(year, month, day):
    y = f"{year}"
    m = f"{month:02d}"
    d = f"{day:02d}"

    data_path = Path("data/era5_1hr_point")
    path = data_path / f"era5_single_levels_{year}_{m}_{d}.zip"
    extract_path = data_path / y / m / d

    data_path.mkdir(parents=True, exist_ok=True)
    extract_path.mkdir(parents=True, exist_ok=True)

    if path.exists():
        print(f"found existing data for {year}-{m}-{d}, skipping")
        return

    c.retrieve(
        "derived-era5-single-levels-daily-statistics",
        {
            "product_type": "reanalysis",
            "variable": ERA5_SINGLE_LEVELS_VARIABLES,
            "year": y,
            "month": [m],
            "day": [d],
            "daily_statistic": "daily_mean",
        },
        str(path),
    )

    with zipfile.ZipFile(path) as zip_ref:
        zip_ref.extractall(extract_path)
        

if __name__ == "__main__":
    retrieve_era5_years_data(range(1990, 2021))
    retrieve_era5_years_t2m(range(2006,2017))
    retrieve_era5_years_tsrc(range(2006,2017))
    retrieve_era5_plevl_qt(range(2006,2017))
    # retrieve_era5_single_levels_3hr([2015])
    retrieve_era5_daily_stats(2015, 3, 1)
    retrieve_era5_daily_stats(2015, 6, 1)
    retrieve_era5_daily_stats(2015, 9, 1)
    retrieve_era5_daily_stats(2015, 12, 1)
