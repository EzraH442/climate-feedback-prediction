import argparse
from pathlib import Path
import xarray as xr

from ecod_calculation import ecod_from_profiles
from preprocessing import fast_compute_cloud_optical_depth
from preprocess import load_cloud_profiles
from utils import make_ecod_filename, make_era5_filename


ECOD_METHODS = ("true", "true_tcc", "fast")


def write_ecod(path, ecod):
    print(f"Saving cached ECOD to {path}...")
    ecod.to_dataset(name="ecod").to_netcdf(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--era5-path", type=Path, default=Path("data/era5"))
    parser.add_argument("--years", type=int, nargs="+", default=list(range(1990, 2021)))
    parser.add_argument("--methods", nargs="+", choices=ECOD_METHODS, default=ECOD_METHODS)
    args = parser.parse_args()


    methods = tuple(dict.fromkeys(args.methods))
    for year in args.years:
        paths = {
            method: args.era5_path / make_ecod_filename(year, method)
            for method in methods
        }
        missing = [method for method, path in paths.items() if not path.exists()]
        if not missing:
            continue

        raw = xr.open_dataset(args.era5_path / make_era5_filename(year))
        try:
            if "fast" in missing:
                write_ecod(
                    paths["fast"],
                    fast_compute_cloud_optical_depth(raw["tclw"], raw["tciw"], raw["tcc"]),
                )

            profile_methods = {"true", "true_tcc"}.intersection(missing)
            if profile_methods:
                profiles = load_cloud_profiles(str(args.era5_path), [year], None, raw)
                try:
                    ecod = ecod_from_profiles(
                        profiles["ciwc"],
                        profiles["clwc"],
                        profiles["level"] * 100.0,
                    )
                    if "true" in missing:
                        write_ecod(paths["true"], ecod)
                    if "true_tcc" in missing:
                        write_ecod(paths["true_tcc"], ecod * raw["tcc"])
                finally:
                    profiles.close()
        finally:
            raw.close()


if __name__ == "__main__":
    main()

