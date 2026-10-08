import argparse
from pathlib import Path

import xarray as xr

from config_utils import load_config
from constants import KERNEL_YEARS
from preprocessing import Preprocessor
from utils import (
    load_yearly_and_filter_by_months,
    make_cloud_profile_filename,
    make_combined_kernel_filename,
    make_era5_filename,
    make_kernel_filename,
)


def load_cloud_profiles(
    path: str, years, months, target_grid: xr.Dataset
) -> xr.Dataset:
    profiles = load_yearly_and_filter_by_months(
        path, years, months, make_cloud_profile_filename
    )
    profiles = profiles[["ciwc", "clwc"]]
    if profiles.latitude.equals(target_grid.latitude) and profiles.longitude.equals(
        target_grid.longitude
    ):
        return profiles
    return profiles.interp(
        latitude=target_grid.latitude,
        longitude=target_grid.longitude,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )


def interpolate_kernel_dataset(
    raw_kernel: xr.Dataset,
    target_latitude: xr.DataArray,
    target_longitude: xr.DataArray,
) -> xr.Dataset:
    if "latitude" not in raw_kernel.coords or "longitude" not in raw_kernel.coords:
        raise ValueError(
            "Kernel dataset must expose 'latitude' and 'longitude' coordinates before interpolation."
        )

    return raw_kernel.interp(
        latitude=target_latitude,
        longitude=target_longitude,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )


def convert_kernels_for_target(processed_kernel, target_var, train_data, val_data):
    if target_var not in ("pal", "palc"):
        return processed_kernel
    raw_tisr = xr.concat([train_data.tisr, val_data.tisr], dim="date").interp(
        latitude=processed_kernel.latitude,
        longitude=processed_kernel.longitude,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )
    return processed_kernel / raw_tisr

def preprocess(config_path):
    config = load_config(config_path)
    input_vars = config.dataset.input_vars
    target_var = config.dataset.target_var

    train_val_years = [config.dataset.train_years, config.dataset.val_years]
    train_data, val_data = [
        load_yearly_and_filter_by_months(
            config.dataset.era5.raw_path,
            years,
            config.dataset.months,
            make_era5_filename,
        )
        for years in train_val_years
    ]

    # ensure tsrc is present if clear sky training is enabled
    if "clear" in config.dataset.sky:
        assert config.dataset.clear_sky_var in train_data.data_vars

    ### load kernels data
    kernel_vars = config.dataset.kernel_vars
    all_years = set(config.dataset.train_years + config.dataset.val_years)
    all_kern_years = all_years.intersection(KERNEL_YEARS)
    raw_kern_path = config.dataset.kernels.raw_path
    kern_datasets = [
        load_yearly_and_filter_by_months(
            raw_kern_path,
            all_kern_years,
            config.dataset.months,
            lambda year: make_kernel_filename(year, var),
        )
        for var in kernel_vars
    ]  # list of datasets (date, lat, lon, all_clr), with field .<var> for each kernel variable

    # (date, lat, lon, all_clr) with fields for each kernel variable
    kern_ds_combined = xr.merge(kern_datasets)

    # --- preprocess era5 data ---
    preprocessor = Preprocessor(config)
    print("Fitting preprocessor on training data...")
    preprocessor.fit(train_data)

    print("Transforming training and validation data...")
    train_preprocessed = preprocessor.transform(train_data)
    val_preprocessed = preprocessor.transform(val_data)

    print("Saving preprocessor state...")
    preprocessor.save(config.preprocess.params_dir)

    Path(config.dataset.era5.path).mkdir(parents=True, exist_ok=True)

    for year in config.dataset.train_years:
        path = f"{config.dataset.era5.path}/era5_single_levels_monthly_{year}.nc"
        print(f"Saving scaled data for year {year} to {path}...")
        train_preprocessed.sel(date=str(year)).to_netcdf(path)

    for year in config.dataset.val_years:
        path = f"{config.dataset.era5.path}/era5_single_levels_monthly_{year}.nc"
        print(f"Saving scaled data for year {year} to {path}...")
        val_preprocessed.sel(date=str(year)).to_netcdf(path)

    if len(kernel_vars) == 0:
        return

    Path(config.dataset.kernels.path).mkdir(parents=True, exist_ok=True)

    # --- preprocess kernel data ---
    target_latitude = train_preprocessed["latitude"]
    target_longitude = train_preprocessed["longitude"]

    scalar = preprocessor.scalar
    ranges = scalar.get_data_max() - scalar.get_data_min()

    target_range = ranges[target_var]
    if len(config.dataset.sky) == 1 and config.dataset.sky[0] == "clear":
        target_range = ranges[config.dataset.clear_sky_var]

    processed_kernel = interpolate_kernel_dataset(
        kern_ds_combined,
        target_latitude=target_latitude,
        target_longitude=target_longitude,
    )
    processed_kernel = convert_kernels_for_target(
        processed_kernel, target_var, train_data, val_data
    )
    processed_kernel = processed_kernel * (ranges) / target_range * (3600 * 24)

    for year in all_kern_years:
        output_kernel_path = Path(
            config.dataset.kernels.path
        ) / make_combined_kernel_filename(year)
        yearly_kernel = processed_kernel.sel(date=processed_kernel.date.dt.year == year)
        if int(yearly_kernel.sizes.get("date", 0)) == 0:
            raise ValueError(
                f"No kernel dates found for year {year} in raw kernel dataset."
            )
        print(
            f"Saving interpolated kernel data for year {year} to {output_kernel_path}..."
        )
        yearly_kernel.to_netcdf(output_kernel_path)

    processed_kernel.close()


def main():
    parser = argparse.ArgumentParser(description="Preprocess ERA5 data")
    parser.add_argument(
        "--config_file",
        type=str,
        required=False,
        help="Path to config file",
        default="configs/preprocess/baseline.yaml",
    )
    args = parser.parse_args()

    preprocess(args.config_file)


if __name__ == "__main__":
    main()
