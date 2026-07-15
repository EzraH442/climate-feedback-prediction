from pathlib import Path

from preprocessing import fast_compute_cloud_optical_depth, DianaPreprocessor
import xarray as xr
import argparse

from config_utils import load_config
from ecod_calculation import ecod_from_profiles
from utils import (
    load_yearly_and_filter_by_months,
    make_era5_filename,
    make_combined_kernel_filename,
    make_ecod_filename,
    make_cloud_profile_filename,
    make_kernel_filename,
)

# def load_yearly_monthly_data(path: str, years, months, filename_fn):
#    paths = [Path(path) / filename_fn(year) for year in years]
#    print(f"Loading data from: {paths}")
#    return filter_by_months(
#        xr.open_mfdataset(paths, combine="nested", concat_dim="date"),
#        months,
#    )


def ecod_method_name(method=False):
    if isinstance(method, bool):
        return "fast" if method else "true"
    return method


def load_ecod(path: str, years, months, method=False) -> xr.DataArray:
    method = ecod_method_name(method)
    ecod_filename_fn = lambda year: make_ecod_filename(year, method)
    ds = load_yearly_and_filter_by_months(path, years, months, ecod_filename_fn)

    return ds.ecod


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


def cache_ecod(path: str, years, method=False) -> None:
    method = ecod_method_name(method)
    root = Path(path)
    for year in years:
        output_path = root / make_ecod_filename(year, method)
        if output_path.exists():
            continue
        raw = xr.open_dataset(root / make_era5_filename(year))
        if method == "fast":
            ecod = fast_compute_cloud_optical_depth(
                raw["tclw"], raw["tciw"], raw["tcc"]
            )
        else:
            profiles = load_cloud_profiles(path, [year], None, raw)
            ecod = ecod_from_profiles(
                profiles["ciwc"],
                profiles["clwc"],
                profiles["level"] * 100.0,
            )
            if method == "true_tcc":
                ecod = ecod * raw["tcc"]
            profiles.close()
        print(f"Saving cached ECOD for {year} to {output_path}...")
        ecod.to_dataset(name="ecod").to_netcdf(output_path)
        raw.close()


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


KERNEL_YEARS = set(range(2011, 2016))


def preprocess(config_path):
    config = load_config(config_path)
    input_vars = config.dataset.input_vars
    target_var = config.dataset.target_var

    ### precompute ecod
    ecod_enabled = config.preprocess.ecod.enabled
    ecod_method = config.preprocess.ecod.method
    if ecod_enabled:
        cache_ecod(
            config.dataset.era5.raw_path,
            range(1990, 2021),
            ecod_method,
        )

    ### load raw data
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

    # add ecod if enabled
    if ecod_enabled:
        train_ecod, val_ecod = [
            load_ecod(
                config.dataset.era5.raw_path, years, config.dataset.months, ecod_method
            )
            for years in train_val_years
        ]
        train_data = train_data.assign(ecod=train_ecod)
        val_data = val_data.assign(ecod=val_ecod)

    # ensure tsrc is present if clear sky training is enabled
    if config.dataset.clear_sky.enabled:
        assert config.dataset.clear_sky.var in train_data.data_vars

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
    preprocessor = DianaPreprocessor(config)
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

    processed_kernel = interpolate_kernel_dataset(
        kern_ds_combined,
        target_latitude=target_latitude,
        target_longitude=target_longitude,
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
