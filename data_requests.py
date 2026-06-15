import os
import cdsapi
import xarray as xr
import cfgrib
import zipfile
from pathlib import Path

c = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",
    key="3dbc229c-81f7-4fd5-be2f-c6f193729ef5",
)


def retrieve_era5_years_data(years=[2006, 2008], data_path="data/era5"):
    os.makedirs(data_path, exist_ok=True)

    for year in years:
        path = os.path.join(data_path, f"era5_single_levels_monthly_{year}.grib")

        if os.path.exists(path):
            print(f"found existing data for year {year}, skipping")
            continue

        c.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable": [
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
                ],
                "year": year,
                "month": [f"{m:02d}" for m in range(1, 13)],
                "time": "00:00",
            },
            path,
        )

    # c.retrieve(
    #     "reanalysis-era5-pressure-levels-monthly-means",
    #     {
    #         "product_type": "monthly_averaged_reanalysis",
    #         "variable": ["temperature", "specific_humidity"],
    #         "pressure_level": ["10", "200", "500", "700"],
    #         "year": "2005",
    #         "month": [f"{m:02d}" for m in range(1, 13)],
    #         "time": "00:00",
    #     },
    #     "era5_pressure_levels_monthly_2005.h5",
    # )



def load_all_era5_data(years):
    paths = [f"data/era5/era5_single_levels_monthly_{year}.grib" for year in years]
    datasets = []
    for path in paths:
        datasets_list = cfgrib.open_datasets(path)
        dataset = xr.combine_by_coords(
            datasets_list,
            join="override",
            compat="override",
            combine_attrs="drop_conflicts",
        )
        datasets.append(dataset)

    dataset = xr.concat(datasets, dim="time")
    return dataset

def retrieve_era5_single_levels_3hr(years, data_path="data/era5_3hr"):
    dataset = "reanalysis-era5-single-levels"
    
    os.makedirs(data_path, exist_ok=True)

    for year in years:
        for month in range(1,13):
            m = f"{month:02d}"
            print(year, m)
            path = os.path.join(data_path, f"era5_single_levels_monthly_{year}_{m}.grib")
        
            if os.path.exists(path):
                print(f"found existing data for year {year}, skipping")
                continue
        
            request = {
                "product_type": ["reanalysis"],
                "variable": [
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
                ],
                "year": year,
                "month": m,
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": ["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"],
                "data_format": "grib",
            }

            c.retrieve(dataset, request, path)

def retrieve_era5_daily_stats(year, month, day):
    y = f"{year}"
    m = f"{month:02d}"
    d = f"{day:02d}"
    
    data_path = Path("data/era5_1hr_point")
    dataset = "derived-era5-single-levels-daily-statistics"
    
    path = data_path / f"era5_single_levels_{year}_{m}_{d}.zip"
    extract_path = data_path / y / m / d
    
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(extract_path, exist_ok=True)
    
    if os.path.exists(path):
        print(f"found existing data for year {year}, skipping")
    
    request = {
        "product_type": "reanalysis",
        "variable": [
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
        ],
        "year": y,
        "month": [m],
        "day": [d],
        "daily_statistic": "daily_mean",
        #"time_zone": "utc+00:00",
        #"frequency": "1_hourly",
    }
    
    c.retrieve(dataset, request, path)
    
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

def retrieve_era5_years_data_v2(years=[2006, 2008], data_path="data/v2/era5"):
    ALL_MONTHS = ([f"{m:02d}" for m in range(1, 13)],)
    os.makedirs(data_path, exist_ok=True)

    for year in years:
        path = os.path.join(data_path, f"era5_single_levels_monthly_{year}.grib")

        if os.path.exists(path):
            print(f"found existing data for year {year}, skipping")
            continue

        c.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            {
                "product_type": "monthly_averaged_reanalysis",
                "variable": [
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
                ],
                "year": year,
                "month": ALL_MONTHS,
                "time": "00:00",
            },
            path,
        )


if __name__ == "__main__":
    #years = range(1990,2021)
    #retrieve_era5_years_data(years)
    #retrieve_era5_single_levels_3hr([2015])
    retrieve_era5_daily_stats(2015, 3, 1)
    retrieve_era5_daily_stats(2015, 6, 1)
    #retrieve_era5_daily_stats(2015, 9, 1)
    retrieve_era5_daily_stats(2015, 12, 1)