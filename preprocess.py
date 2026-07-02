from pathlib import Path

from preprocessing import create_2024_preprocessor
import xarray as xr
import argparse

from config_utils import load_config
from ecod_calculation import ecod_from_profiles
from utils import filter_by_months


def make_era5_filename(year):
    return f"era5_single_levels_monthly_{year}.nc"


def make_kernel_filename(year):
    return f"RRTM_kernel_monthly_{year}_alb_TOA_SFC.nc"


def make_cloud_profile_filename(year):
    return f"era5_plev_ciwc_clwc_monthly_{year}.nc"


def make_ecod_filename(year, fast_ecod=False):
    prefix = "era5_fast_ecod" if fast_ecod else "era5_ecod"
    return f"{prefix}_monthly_{year}.nc"


def compute_cloud_optical_depth(
    tclw,
    tciw,
    tcc,
    re_liquid=10e-6,
    re_ice=30e-6,
    rho_water=1000.0,
    rho_ice=917.0,
):
    tau_l = 1.5 * tclw / (rho_water * re_liquid)
    tau_i = 1.5 * tciw / (rho_ice * re_ice)
    return (tau_l + tau_i) * tcc


def open_years(path: str, years, filename_fn, months):
    paths = [Path(path) / filename_fn(year) for year in years]
    print(f"Loading data from: {paths}")
    return filter_by_months(
        xr.open_mfdataset(paths, combine="nested", concat_dim="date"),
        months,
    )


def load_cloud_profiles(path: str, years, months, target_grid: xr.Dataset) -> xr.Dataset:
    profiles = open_years(path, years, make_cloud_profile_filename, months)
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


def raw_era5_years(path: str) -> list[int]:
    prefix = "era5_single_levels_monthly_"
    return sorted(
        int(p.stem.removeprefix(prefix))
        for p in Path(path).glob(f"{prefix}*.nc")
    )


def cache_ecod(path: str, years, fast_ecod=False) -> None:
    root = Path(path)
    for year in years:
        output_path = root / make_ecod_filename(year, fast_ecod)
        if output_path.exists():
            continue
        raw = xr.open_dataset(root / make_era5_filename(year))
        if fast_ecod:
            ecod = compute_cloud_optical_depth(raw["tclw"], raw["tciw"], raw["tcc"])
        else:
            profiles = load_cloud_profiles(path, [year], None, raw)
            ecod = ecod_from_profiles(
                profiles["ciwc"],
                profiles["clwc"],
                profiles["level"] * 100.0,
            )
            profiles.close()
        print(f"Saving cached ECOD for {year} to {output_path}...")
        ecod.to_dataset(name="ecod").to_netcdf(output_path)
        raw.close()


def load_ecod(path: str, years, months, fast_ecod=False) -> xr.DataArray:
    return open_years(
        path,
        years,
        lambda year: make_ecod_filename(year, fast_ecod),
        months,
    )["ecod"]


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
        kwargs={"fill_value": "extrapolate"}
    )


def preprocess(config_path):
    conf = load_config(config_path)
    input_var = conf.dataset.input_var
    target_var = conf.dataset.target_var
    input_vars = conf.dataset.input_vars
    clear_sky = conf.dataset.clear_sky_training
    clear_sky_enabled = bool(clear_sky.enabled)
    ecod_enabled = conf.preprocess.ecod.enabled
    ecod_fast = conf.preprocess.ecod.method == "fast"
    if ecod_enabled:
        cache_ecod(
            conf.dataset.era5.raw_path,
            raw_era5_years(conf.dataset.era5.raw_path),
            ecod_fast,
        )

    all_kern_years = [
        y
        for y in sorted(conf.dataset.train_years + conf.dataset.val_years)
        if y in list(range(2011, 2016))
    ]

    train_data = open_years(
        conf.dataset.era5.raw_path,
        conf.dataset.train_years,
        make_era5_filename,
        conf.dataset.months,
    )
    val_data = open_years(
        conf.dataset.era5.raw_path,
        conf.dataset.val_years,
        make_era5_filename,
        conf.dataset.months,
    )
    if clear_sky_enabled:
        train_data = train_data.assign(
            {clear_sky.output_var: train_data[clear_sky.source_var]}
        )
        val_data = val_data.assign(
            {clear_sky.output_var: val_data[clear_sky.source_var]}
        )
    if ecod_enabled:
        train_data = train_data.assign(
            ecod=load_ecod(
                conf.dataset.era5.raw_path,
                conf.dataset.train_years,
                conf.dataset.months,
                ecod_fast,
            )
        )
        val_data = val_data.assign(
            ecod=load_ecod(
                conf.dataset.era5.raw_path,
                conf.dataset.val_years,
                conf.dataset.months,
                ecod_fast,
            )
        )

    kern_data = open_years(
        conf.dataset.kernels.raw_path,
        all_kern_years,
        make_kernel_filename,
        conf.dataset.months,
    )

    # --- preprocess era5 data ---
    preprocessor = create_2024_preprocessor(
        input_vars=input_vars,
        target_var=target_var,
        ecod=ecod_enabled,
    )
    print("Fitting preprocessor on training data...")
    preprocessor.fit(train_data)
    scaler = preprocessor.preprocessors[-1]
    if clear_sky_enabled:
        spatial_preprocessor = preprocessor.preprocessors[-2]
        scaler.update_shared_range(
            target_var,
            clear_sky.output_var,
            spatial_preprocessor.transform(train_data[[clear_sky.output_var]])[
                clear_sky.output_var
            ],
        )

    print("Transforming training and validation data...")
    train_preprocessed = preprocessor.transform(train_data)
    val_preprocessed = preprocessor.transform(val_data)
    if clear_sky_enabled:
        spatial_preprocessor = preprocessor.preprocessors[-2]
        for raw, processed in [
            (train_data, train_preprocessed),
            (val_data, val_preprocessed),
        ]:
            processed[clear_sky.output_var] = scaler.transform(
                spatial_preprocessor.transform(raw[[clear_sky.output_var]])
            )[clear_sky.output_var]

    print("Saving preprocessor state...")
    preprocessor.save(conf.preprocess.params_dir)

    Path(conf.dataset.era5.path).mkdir(parents=True, exist_ok=True)
    Path(conf.dataset.kernels.path).mkdir(parents=True, exist_ok=True)

    for year in conf.dataset.train_years:
        path = f"{conf.dataset.era5.path}/era5_single_levels_monthly_{year}.nc"
        print(f"Saving scaled data for year {year} to {path}...")
        train_preprocessed.sel(date=str(year)).to_netcdf(path)

    for year in conf.dataset.val_years:
        path = f"{conf.dataset.era5.path}/era5_single_levels_monthly_{year}.nc"
        print(f"Saving scaled data for year {year} to {path}...")
        val_preprocessed.sel(date=str(year)).to_netcdf(path)

    # --- preprocess kernel data ---
    target_latitude = train_preprocessed["latitude"]
    target_longitude = train_preprocessed["longitude"]

    scaler = preprocessor.preprocessors[-1]  # XarrayMinMaxScaler
    data_min = scaler.data_min_
    data_max = scaler.data_max_
    processed_kernel = interpolate_kernel_dataset(
        kern_data,
        target_latitude=target_latitude,
        target_longitude=target_longitude,
    )
    input_range = float(data_max[input_var] - data_min[input_var])
    target_range = float(data_max[target_var] - data_min[target_var])
    processed_kernel["TOA_clr"] = processed_kernel["TOA_clr"] * (input_range / target_range) * (3600 * 24)
    processed_kernel["TOA_cld"] = processed_kernel["TOA_cld"] * (input_range / target_range) * (3600 * 24)

    for year in all_kern_years:
        output_kernel_path = Path(conf.dataset.kernels.path) / make_kernel_filename(
            year
        )
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
