import os
from pathlib import Path

from preprocessing import create_2024_preprocessor, Downscaler, XarrayMinMaxScaler,SequentialPreprocessor
import xarray as xr
import argparse

from config_utils import load_config

def create_2024_preprocessor_no_ecod():
    return SequentialPreprocessor(
        preprocessors=[
            Downscaler(factor=[("latitude", 4), ("longitude", 4)]),
            XarrayMinMaxScaler(dim=("date", "latitude", "longitude")),
        ]
    )


def make_era5_filename(year):
    return f"era5_single_levels_monthly_{year}.nc"


def make_kernel_filename(year):
    return f"RRTM_kernel_monthly_{year}_cld_alb_TOA_SFC.nc"


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

    # --- make paths ---
    train_paths = [
        Path(conf.dataset.era5.raw_path) / make_era5_filename(year)
        for year in conf.dataset.train_years
    ]

    val_paths = [
        Path(conf.dataset.era5.raw_path) / make_era5_filename(year)
        for year in conf.dataset.val_years
    ]

    all_kern_years = [
        y for y in sorted(conf.dataset.train_years + conf.dataset.val_years) if y in list(range(2011, 2016))
    ]
    print(all_kern_years)
    kernel_paths = [
        Path(conf.dataset.kernels.raw_path) / make_kernel_filename(year)
        for year in all_kern_years
    ]

    # --- load data ---
    print(f"Loading ERA5 train data from: {train_paths}")
    train_data = xr.open_mfdataset(train_paths, combine="nested", concat_dim="date")

    print(f"Loading ERA5 val data from: {val_paths}")
    val_data = xr.open_mfdataset(val_paths, combine="nested", concat_dim="date")
    """
       tcc        (date, latitude, longitude) float64 100MB 
       fal        (date, latitude, longitude) float64 100MB 0.7555 0.7555 ... 0.85
       hcc        (date, latitude, longitude) float64 100MB 0.3654 0.3654 ... 0.085
       mcc        (date, latitude, longitude) float64 100MB 0.474 0.474 ... 0.1886
       lcc        (date, latitude, longitude) float64 100MB 0.8883 ... 0.1287
       sp         (date, latitude, longitude) float64 100MB 1.011e+05 ... 7.03e+04
       tciw       (date, latitude, longitude) float64 100MB 0.02062 ... 0.003697
       tclw       (date, latitude, longitude) float64 100MB 0.005785 ... 2.902e-05
       tco3       (date, latitude, longitude) float64 100MB 0.006788 ... 0.005639
       tcwv       (date, latitude, longitude) float64 100MB 2.86 2.86 ... 1.031
       tisr       (date, latitude, longitude) float64 100MB 33.78 33.78 ... 24.45
       tsr        (date, latitude, longitude) float64 100MB 0.0 0.0 ... 1.344e+07
    """

    print(f"Loading raw kernel data from: {kernel_paths}")
    kern_data = xr.open_mfdataset(kernel_paths, combine="nested", concat_dim="date")

    # --- preprocess era5 data ---
    preprocessor = create_2024_preprocessor()
    #preprocessor = create_2024_preprocessor_no_ecod()
    print("Fitting preprocessor on training data...")
    preprocessor.fit(train_data)

    print("Saving preprocessor state...")
    preprocessor.save(conf.preprocess.params_dir)

    print("Transforming training and validation data...")
    train_preprocessed = preprocessor.transform(train_data)
    val_preprocessed = preprocessor.transform(val_data)

    if not os.path.exists(conf.dataset.era5.path):
        os.makedirs(conf.dataset.era5.path)
    if not os.path.exists(conf.dataset.kernels.path):
        os.makedirs(conf.dataset.kernels.path)

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
    
    fal_range = float(data_max["fal"] - data_min["fal"])
    tsr_range = float(data_max["tsr"] - data_min["tsr"])
    processed_kernel["TOA"] = processed_kernel["TOA"] * (fal_range / tsr_range)

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
