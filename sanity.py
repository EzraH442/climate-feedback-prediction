from pathlib import Path

import xarray as xr

from dataloader import make_era5_filename


raw_root = Path("data/era5")
odd_years = list(range(1991, 2021, 2))
target_months = {3, 9}
raw_paths = [raw_root / make_era5_filename(year) for year in odd_years]
dataset = xr.open_mfdataset(raw_paths, combine="nested", concat_dim="date")
month_mask = dataset["date"].dt.month.isin(target_months)
dataset = dataset.where(month_mask, drop=True)

n_dates = int(dataset.sizes.get("date", 0))
if n_dates == 0:
    raise ValueError("No March/September dates were selected from the ERA5 files.")

tsr = dataset["tsr"]
tsr_mean = tsr.mean(skipna=True).to_numpy().item()
valid_count = int(tsr.count().to_numpy().item())

print(f"Selected {n_dates} monthly slices")
print(f"Valid tsr values: {valid_count}")
print(f"Mean tsr over March and September of odd years 1991-2019: {tsr_mean / (3600 * 24)}")