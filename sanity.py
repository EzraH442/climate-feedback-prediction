from pathlib import Path

import xarray as xr

from dataloader import make_era5_filename

raw_root = Path("data/era5")
odd_years = list(range(1991, 2021, 2))
raw_paths = [raw_root / make_era5_filename(year) for year in odd_years]

missing_paths = [path for path in raw_paths if not path.exists()]
if missing_paths:
    missing = "\n".join(str(path) for path in missing_paths)
    raise FileNotFoundError(f"Missing ERA5 file(s):\n{missing}")

dataset = xr.open_mfdataset(raw_paths, combine="nested", concat_dim="date")
tsr_mean = dataset["tsr"].mean().item()

print(f"Mean tsr over odd years 1991-2019: {tsr_mean}")

