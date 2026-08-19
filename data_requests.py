from pathlib import Path

import cdsapi

ERA5_SINGLE_LEVELS_VARIABLES = [
    "2m_temperature",
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
    "top_net_solar_radiation_clear_sky",
    "top_net_thermal_radiation",
]

MONTHS = [f"{m:02d}" for m in range(1, 13)]
# fmt: off
ALL_PRESSURE_LEVELS = (
    [
        "1",   "2",   "3",   "5",   "7",   "10",
        "20",  "30",  "50",  "70",  "100", "125",
        "150", "175", "200", "225", "250", "300",
        "350", "400", "450", "500", "550", "600",
        "650", "700", "750", "775", "800", "825",
        "850", "875", "900", "925", "950", "975",
        "1000",
    ],
)
# fmt: on

DATA_PATH_ERA5 = Path("data/era5")

c = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",
    key="3dbc229c-81f7-4fd5-be2f-c6f193729ef5",
)


def retrieve_era5_years_data(years: list[int], data_path: Path = DATA_PATH_ERA5):
    data_path.mkdir(parents=True, exist_ok=True)

    for year in years:
        path = data_path / f"era5_single_levels_monthly_{year}.grib"
        if path.exists():
            print(f"found existing data for year {year}, skipping")
            continue

        _ = c.retrieve(
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


def retrieve_era5_plevl_qt(years: list[int], data_path: Path = DATA_PATH_ERA5):
    data_path.mkdir(parents=True, exist_ok=True)

    for year in years:
        path = data_path / f"era5_plev_qt_monthly_{year}.grib"
        if path.exists():
            print(f"found existing data for year {year}, skipping")
            continue

        _ = c.retrieve(
            "reanalysis-era5-pressure-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "pressure_level": ALL_PRESSURE_LEVELS,
                "variable": ["specific_humidity", "temperature"],
                "year": year,
                "month": MONTHS,
                "time": "00:00",
                "data_format": "grib",
                "download_format": "unarchived",
            },
            str(path),
        )


if __name__ == "__main__":
    years = list(range(1990, 2021))
    retrieve_era5_years_data(years)
    retrieve_era5_plevl_qt(years)
