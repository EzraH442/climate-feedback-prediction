import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from utils import (
    SECONDS_PER_DAY,
    interpolate_spatial_field,
    plot_global_field,
    plot_north_pole_field,
)


OLD_INPUT_ORDER = [
    "fal",
    "hcc",
    "lcc",
    "mcc",
    "sp",
    "tisr",
    "tciw",
    "tclw",
    "tco3",
    "tcwv",
    "ecod",
    "ecod_fal",
]


class OldFalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = len(OLD_INPUT_ORDER)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 11),
            torch.nn.Tanh(),
            torch.nn.Linear(11, 1),
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)


def load_old_model(weights_path: Path) -> OldFalModel:
    weights = np.load(weights_path)
    model = OldFalModel()
    with torch.no_grad():
        model.model[0].weight.copy_(
            torch.from_numpy(
                weights["layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE"]
            ).t()
        )
        model.model[0].bias.copy_(
            torch.from_numpy(
                weights["layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE"]
            )
        )
        model.model[2].weight.copy_(
            torch.from_numpy(
                weights["layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE"]
            ).t()
        )
        model.model[2].bias.copy_(
            torch.from_numpy(
                weights["layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE"]
            )
        )
    model.eval()
    return model


def old_scale(ds: xr.Dataset, scalar_dir: Path) -> xr.Dataset:
    data_min = xr.open_dataset(scalar_dir / "min.nc")
    data_max = xr.open_dataset(scalar_dir / "max.nc")
    denom = (data_max - data_min).where(data_max != data_min, 1.0)
    return ((ds - data_min) / denom) * 2 - 1


def old_scale_value(values, name: str, scalar_dir: Path):
    data_min = xr.open_dataset(scalar_dir / "min.nc")
    data_max = xr.open_dataset(scalar_dir / "max.nc")
    denom = float(data_max[name] - data_min[name]) or 1.0
    return ((values - float(data_min[name])) / denom) * 2 - 1


def old_inverse_value(values, name: str, scalar_dir: Path):
    data_min = xr.open_dataset(scalar_dir / "min.nc")
    data_max = xr.open_dataset(scalar_dir / "max.nc")
    return ((values + 1) / 2) * float(data_max[name] - data_min[name]) + float(
        data_min[name]
    )


def old_inverse_tsr(values: np.ndarray, scalar_dir: Path) -> np.ndarray:
    return old_inverse_value(values, "tsr", scalar_dir)


def load_dataset(path: Path, years: list[int] | None) -> xr.Dataset:
    ds = xr.open_dataset(path)
    rename = {"time": "date", "lat": "latitude", "lon": "longitude"}
    ds = ds.rename({old: new for old, new in rename.items() if old in ds})
    if years is not None:
        ds = ds.sel(date=ds.date.dt.year.isin(years))
    ds["ecod_fal"] = ds["ecod"] * ds["fal"]
    missing = [name for name in [*OLD_INPUT_ORDER, "tsr"] if name not in ds]
    if missing:
        raise ValueError(f"Dataset missing old model variables: {missing}")
    return ds[[*OLD_INPUT_ORDER, "tsr"]]


def predict(model: OldFalModel, inputs: np.ndarray, batch_size: int) -> np.ndarray:
    flat = inputs.reshape(-1, inputs.shape[-1])
    outputs = np.empty(flat.shape[0], dtype=np.float32)
    for start in range(0, len(flat), batch_size):
        end = min(start + batch_size, len(flat))
        with torch.no_grad():
            outputs[start:end] = model(torch.from_numpy(flat[start:end]).float()).numpy()
    return outputs.reshape(inputs.shape[:-1])


def inputs_from_scaled(scaled: xr.Dataset) -> np.ndarray:
    return (
        scaled[OLD_INPUT_ORDER]
        .to_dataarray()
        .transpose("date", "latitude", "longitude", "variable")
        .to_numpy()
    )


def predict_tsr(model: OldFalModel, ds: xr.Dataset, scalar_dir: Path, batch_size: int):
    pred_scaled = predict(model, inputs_from_scaled(old_scale(ds, scalar_dir)), batch_size)
    return old_inverse_tsr(pred_scaled, scalar_dir)


def clear_sky(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy(deep=True)
    for name in ["hcc", "mcc", "lcc", "tciw", "tclw", "ecod", "ecod_fal"]:
        if name in ds:
            ds[name][:] = 0
    return ds


def compute_old_kernel_autograd(
    model: OldFalModel,
    ds: xr.Dataset,
    scalar_dir: Path,
    batch_size: int,
):
    scaled = old_scale(ds, scalar_dir)
    inputs = inputs_from_scaled(scaled)
    flat = inputs.reshape(-1, inputs.shape[-1])
    outputs = np.empty(flat.shape[0], dtype=np.float32)
    fal_index = OLD_INPUT_ORDER.index("fal")

    target_range = float(
        xr.open_dataset(scalar_dir / "max.nc")["tsr"]
        - xr.open_dataset(scalar_dir / "min.nc")["tsr"]
    )
    fal_range = float(
        xr.open_dataset(scalar_dir / "max.nc")["fal"]
        - xr.open_dataset(scalar_dir / "min.nc")["fal"]
    )

    for start in range(0, len(flat), batch_size):
        end = min(start + batch_size, len(flat))
        batch = torch.from_numpy(flat[start:end]).float().requires_grad_(True)
        pred = model(batch)
        grad = torch.autograd.grad(pred.sum(), batch)[0][:, fal_index]
        outputs[start:end] = grad.detach().numpy()

    kernel = (
        outputs.reshape(inputs.shape[:-1])
        * (target_range / fal_range)
        / SECONDS_PER_DAY
        * 0.01
    )
    return np.squeeze(kernel), ds.longitude.to_numpy(), ds.latitude.to_numpy()


def write_outputs(ds: xr.Dataset, pred_tsr: np.ndarray, output_dir: Path):
    true_tsr = ds["tsr"].to_numpy() / SECONDS_PER_DAY
    pred_tsr = pred_tsr / SECONDS_PER_DAY
    diff = pred_tsr - true_tsr

    true_mean = np.nanmean(true_tsr, axis=0)
    pred_mean = np.nanmean(pred_tsr, axis=0)
    mbe_map = np.nanmean(diff, axis=0)
    rmse_map = np.sqrt(np.nanmean(diff**2, axis=0))

    lon = ds.longitude.to_numpy()
    lat = ds.latitude.to_numpy()
    max_tsr = max(float(np.nanmax(true_mean)), float(np.nanmax(pred_mean)))
    max_abs_mbe = float(np.nanmax(np.abs(mbe_map)))
    max_rmse = float(np.nanmax(rmse_map))

    plot_global_field(
        true_mean,
        lon,
        lat,
        "TSR (ERA5)",
        output_dir / "old_global_tsr_era5.png",
        cmap="Spectral",
        vmin=0,
        vmax=max_tsr,
        label="$W/m^2$",
        annotation=f"{float(np.nanmean(true_mean)):.2f}",
    )
    plot_global_field(
        pred_mean,
        lon,
        lat,
        "TSR (old NN)",
        output_dir / "old_global_tsr_nn.png",
        cmap="Spectral",
        vmin=0,
        vmax=max_tsr,
        label="$W/m^2$",
        annotation=f"{float(np.nanmean(pred_mean)):.2f}",
    )
    plot_global_field(
        mbe_map,
        lon,
        lat,
        "MBE (old NN - ERA5)",
        output_dir / "old_global_tsr_mbe.png",
        cmap="RdBu_r",
        vmin=-max_abs_mbe,
        vmax=max_abs_mbe,
        label="$W/m^2$",
        annotation=f"{float(np.nanmean(mbe_map)):.2f}",
    )
    plot_global_field(
        rmse_map,
        lon,
        lat,
        "RMSE",
        output_dir / "old_global_tsr_rmse.png",
        cmap="Blues",
        vmin=0,
        vmax=max_rmse,
        label="$W/m^2$",
        annotation=f"{float(np.sqrt(np.nanmean(rmse_map**2))):.2f}",
    )

    with (output_dir / "old_model_metrics.csv").open("w", encoding="ascii", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["global_tsr_mean_wm2", f"{float(np.nanmean(true_mean)):.10e}"])
        writer.writerow(["global_prediction_mean_wm2", f"{float(np.nanmean(pred_mean)):.10e}"])
        writer.writerow(["global_mbe_wm2", f"{float(np.nanmean(mbe_map)):.10e}"])
        writer.writerow(["global_rmse_wm2", f"{float(np.sqrt(np.nanmean(rmse_map**2))):.10e}"])
        writer.writerow(["max_abs_mbe_wm2", f"{max_abs_mbe:.10e}"])
        writer.writerow(["max_rmse_wm2", f"{max_rmse:.10e}"])


def global_tsr_test(ds: xr.Dataset, model: OldFalModel, scalar_dir: Path, output_dir: Path, batch_size: int):
    write_outputs(ds, predict_tsr(model, ds, scalar_dir, batch_size), output_dir)


def tsr_ecod_fal_contour_test(
    ds: xr.Dataset,
    model: OldFalModel,
    scalar_dir: Path,
    output_dir: Path,
    date="2015-09",
    latitude=84.0,
    longitude=17.0,
    scatter=True,
    extrapolate=False,
    n_fal=100,
    n_ecod=100,
):
    base = ds.sel(date=date, latitude=latitude, longitude=longitude, method="nearest")
    base_features = np.array([float(base[name]) for name in OLD_INPUT_ORDER], dtype=np.float32)
    data_max = xr.open_dataset(scalar_dir / "max.nc")

    fal_values = np.linspace(float(xr.open_dataset(scalar_dir / "min.nc")["fal"]), float(data_max["fal"]), n_fal)
    ecod_values = np.linspace(float(xr.open_dataset(scalar_dir / "min.nc")["ecod"]), float(data_max["ecod"]), n_ecod)
    if extrapolate:
        fal_values = np.linspace(0, float(data_max["fal"]) * 2, n_fal)
        ecod_values = np.linspace(0, float(data_max["ecod"]) * 2, n_ecod)
    fal_grid, ecod_grid = np.meshgrid(fal_values, ecod_values)

    inputs = np.broadcast_to(base_features, (n_ecod * n_fal, len(OLD_INPUT_ORDER))).copy()
    inputs[:, OLD_INPUT_ORDER.index("fal")] = fal_grid.ravel()
    inputs[:, OLD_INPUT_ORDER.index("ecod")] = ecod_grid.ravel()
    inputs[:, OLD_INPUT_ORDER.index("ecod_fal")] = (ecod_grid * fal_grid).ravel()
    scaled_inputs = np.column_stack(
        [old_scale_value(inputs[:, i], name, scalar_dir) for i, name in enumerate(OLD_INPUT_ORDER)]
    )
    pred = old_inverse_tsr(predict(model, scaled_inputs.reshape(n_ecod, n_fal, -1), n_ecod * n_fal), scalar_dir)
    pred = pred / SECONDS_PER_DAY

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    contour = ax.contourf(fal_grid, ecod_grid, pred, levels=31, cmap="Spectral")
    if scatter:
        ax.scatter(ds["fal"].to_numpy().ravel(), ds["ecod"].to_numpy().ravel(), s=1, c="black", alpha=0.05, linewidths=0)
    ax.scatter([float(base["fal"])], [float(base["ecod"])], s=20, c="white", edgecolors="black")
    ax.set_xlabel("fal")
    ax.set_ylabel("ecod")
    ax.set_title(f"Old NN TSR over fal/ecod; lat={float(base.latitude):.2f}, lon={float(base.longitude):.2f}, date={date}")
    fig.colorbar(contour, ax=ax).set_label("$W/m^2$")
    fig.tight_layout()
    fig.savefig(output_dir / "old_tsr_contour_fal_ecod.png")
    plt.close(fig)


def kernel_ecod_fal_contour_test(
    ds: xr.Dataset,
    model: OldFalModel,
    scalar_dir: Path,
    output_dir: Path,
    date="2015-09",
    latitude=84.0,
    longitude=17.0,
    scatter=True,
    extrapolate=False,
    n_fal=100,
    n_ecod=100,
):
    base = ds.sel(date=date, latitude=latitude, longitude=longitude, method="nearest")
    base_features = np.array([float(base[name]) for name in OLD_INPUT_ORDER], dtype=np.float32)
    data_min = xr.open_dataset(scalar_dir / "min.nc")
    data_max = xr.open_dataset(scalar_dir / "max.nc")

    fal_values = np.linspace(float(data_min["fal"]), float(data_max["fal"]), n_fal)
    ecod_values = np.linspace(float(data_min["ecod"]), float(data_max["ecod"]), n_ecod)
    if extrapolate:
        fal_values = np.linspace(0, float(data_max["fal"]) * 2, n_fal)
        ecod_values = np.linspace(0, float(data_max["ecod"]) * 2, n_ecod)
    fal_grid, ecod_grid = np.meshgrid(fal_values, ecod_values)

    inputs = np.broadcast_to(base_features, (n_ecod * n_fal, len(OLD_INPUT_ORDER))).copy()
    inputs[:, OLD_INPUT_ORDER.index("fal")] = fal_grid.ravel()
    inputs[:, OLD_INPUT_ORDER.index("ecod")] = ecod_grid.ravel()
    inputs[:, OLD_INPUT_ORDER.index("ecod_fal")] = (ecod_grid * fal_grid).ravel()
    scaled_inputs = np.column_stack(
        [old_scale_value(inputs[:, i], name, scalar_dir) for i, name in enumerate(OLD_INPUT_ORDER)]
    )

    batch = torch.from_numpy(scaled_inputs).float().requires_grad_(True)
    output = model(batch)
    grad = torch.autograd.grad(output.sum(), batch)[0][:, OLD_INPUT_ORDER.index("fal")]
    kernel = (
        grad.detach().numpy()
        * (float(data_max["tsr"] - data_min["tsr"]) / float(data_max["fal"] - data_min["fal"]))
        / SECONDS_PER_DAY
        * 0.01
    ).reshape(n_ecod, n_fal)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    max_abs = float(np.nanmax(np.abs(kernel)))
    levels = np.linspace(-max_abs, max_abs, 31) if max_abs > 0 else 31
    contour = ax.contourf(fal_grid, ecod_grid, kernel, levels=levels, cmap="RdBu_r", extend="both")
    if scatter:
        ax.scatter(ds["fal"].to_numpy().ravel(), ds["ecod"].to_numpy().ravel(), s=1, c="black", alpha=0.05, linewidths=0)
    ax.set_xlabel("fal")
    ax.set_ylabel("ecod")
    ax.set_title(f"Old NN surface albedo kernel over fal/ecod; lat={float(base.latitude):.2f}, lon={float(base.longitude):.2f}, date={date}")
    fig.colorbar(contour, ax=ax).set_label(r"$W/m^2 1\%$")
    fig.tight_layout()
    fig.savefig(output_dir / "old_kernel_contour_fal_ecod.png")
    plt.close(fig)


def load_rrtm_kernels(kernel_dir: Path) -> xr.Dataset:
    paths = sorted(kernel_dir.glob("RRTM_kernel_monthly_*_alb_TOA_SFC.nc"))
    if not paths:
        raise FileNotFoundError(f"No RRTM kernel files found in {kernel_dir}")
    return xr.open_mfdataset(paths, combine="nested", concat_dim="date")


def kernel_date_test(
    ds: xr.Dataset,
    model: OldFalModel,
    scalar_dir: Path,
    true_kernel: xr.Dataset,
    date: str,
    output_dir: Path,
    batch_size: int,
):
    nn_grad_cld, lon, lat = compute_old_kernel_autograd(model, ds.sel(date=date), scalar_dir, batch_size)
    nn_grad_clr, _, _ = compute_old_kernel_autograd(model, clear_sky(ds).sel(date=date), scalar_dir, batch_size)
    rrtm = true_kernel.sel(date=date)
    rrtm_kern_cld = rrtm["TOA_cld"].to_numpy()[0] * 0.01
    rrtm_kern_clr = rrtm["TOA_clr"].to_numpy()[0] * 0.01
    kern_lon = rrtm.longitude.to_numpy()
    kern_lat = rrtm.latitude.to_numpy()

    plot_global_field(nn_grad_cld, lon, lat, f"Old NN Surface Albedo Kernel via autograd (all)\n{date}", output_dir / f"old_kern_all_nn_grad_{date}.png", cmap="RdBu_r", vmin=-3, vmax=3, label=r"$W/m^2 1\%$", annotation=f"{np.nanmean(nn_grad_cld):.2f}")
    plot_global_field(nn_grad_clr, lon, lat, f"Old NN Surface Albedo Kernel via autograd (clear)\n{date}", output_dir / f"old_kern_clr_nn_grad_{date}.png", cmap="RdBu_r", vmin=-3, vmax=3, label=r"$W/m^2 1\%$", annotation=f"{np.nanmean(nn_grad_clr):.2f}")
    plot_north_pole_field(nn_grad_cld, lon, lat, f"Old NN Surface Albedo Kernel via autograd (all)\n{date}", output_dir / f"old_kern_all_nn_grad_np_{date}.png", cmap="RdBu_r", vmin=-3, vmax=3, label=r"$W/m^2 1\%$")
    plot_north_pole_field(nn_grad_clr, lon, lat, f"Old NN Surface Albedo Kernel via autograd (clear)\n{date}", output_dir / f"old_kern_clr_nn_grad_np_{date}.png", cmap="RdBu_r", vmin=-3, vmax=3, label=r"$W/m^2 1\%$")

    plot_global_field(rrtm_kern_cld, kern_lon, kern_lat, f"RRTM Surface Albedo Kernel (all)\n{date}", output_dir / f"old_kern_all_rrtm_{date}.png", cmap="RdBu_r", vmin=-3, vmax=3, label=r"$W/m^2 1\%$", annotation=f"{np.nanmean(rrtm_kern_cld):.2f}")
    plot_global_field(rrtm_kern_clr, kern_lon, kern_lat, f"RRTM Surface Albedo Kernel (clear)\n{date}", output_dir / f"old_kern_clr_rrtm_{date}.png", cmap="RdBu_r", vmin=-3, vmax=3, label=r"$W/m^2 1\%$", annotation=f"{np.nanmean(rrtm_kern_clr):.2f}")

    nn_cld_interp = interpolate_spatial_field(nn_grad_cld, lon, lat, kern_lon, kern_lat)
    nn_clr_interp = interpolate_spatial_field(nn_grad_clr, lon, lat, kern_lon, kern_lat)
    diff_cld = nn_cld_interp - rrtm_kern_cld
    diff_clr = nn_clr_interp - rrtm_kern_clr
    plot_global_field(diff_cld, kern_lon, kern_lat, f"Old NN autograd-RRTM Surface Albedo Kernel (all)\n{date}", output_dir / f"old_kern_all_nn_grad-rrtm_{date}.png", cmap="RdBu_r", vmin=-0.3, vmax=0.3, label=r"$W/m^2 1\%$", annotation=f"{np.nanmean(diff_cld):.2f}; {np.nanmean(np.abs(diff_cld)):.2f}")
    plot_global_field(diff_clr, kern_lon, kern_lat, f"Old NN autograd-RRTM Surface Albedo Kernel (clear)\n{date}", output_dir / f"old_kern_clr_nn_grad-rrtm_{date}.png", cmap="RdBu_r", vmin=-0.3, vmax=0.3, label=r"$W/m^2 1\%$", annotation=f"{np.nanmean(diff_clr):.2f}; {np.nanmean(np.abs(diff_clr)):.2f}")


def second_order_test(
    ds: xr.Dataset,
    model: OldFalModel,
    scalar_dir: Path,
    true_kernel: xr.Dataset,
    dates: list[str],
    output_dir: Path,
    batch_size: int,
):
    nn_1, lon, lat = compute_old_kernel_autograd(model, ds.sel(date=dates[1]), scalar_dir, batch_size)
    nn_0, _, _ = compute_old_kernel_autograd(model, ds.sel(date=dates[0]), scalar_dir, batch_size)
    delta_nn = nn_1 - nn_0
    rrtm_1 = true_kernel["TOA_cld"].sel(date=dates[1]).to_numpy()[0] * 0.01
    rrtm_0 = true_kernel["TOA_cld"].sel(date=dates[0]).to_numpy()[0] * 0.01
    delta_rrtm = rrtm_1 - rrtm_0
    kern_lon = true_kernel.longitude.to_numpy()
    kern_lat = true_kernel.latitude.to_numpy()
    delta_nn_interp = interpolate_spatial_field(delta_nn, lon, lat, kern_lon, kern_lat)
    delta_diff = delta_nn_interp - delta_rrtm
    plot_north_pole_field(delta_nn_interp, kern_lon, kern_lat, f"Old NN autograd kernel difference\n({dates[1]} minus {dates[0]})", output_dir / "old_delta_k_nn_grad_np.png", vmin=-1, vmax=1)
    plot_north_pole_field(delta_rrtm, kern_lon, kern_lat, f"ERA5 surface albedo kernel difference\n({dates[1]} minus {dates[0]})", output_dir / "old_delta_k_rrtm_np.png", vmin=-1, vmax=1)
    plot_north_pole_field(delta_diff, kern_lon, kern_lat, r"$K_{NN,\mathrm{grad}} - K_{ERA5}$" + f"\n({dates[1]} minus {dates[0]})", output_dir / "old_delta_k_nn_grad-rrtm_north_pole.png", vmin=-1, vmax=1)



def parse_years(value: str) -> list[int] | None:
    if value == "all":
        return None
    return [int(year) for year in value.split(",") if year]


def main():
    parser = argparse.ArgumentParser(description="Evaluate the old TensorFlow fal model.")
    parser.add_argument(
        "--data_path",
        type=Path,
        default=Path(
            "old/era5_1deg_monthly_avg_sl_1990-2020_tisr_tciw_tclw_tcwv_hcc_mcc_lcc_sp_tco3_fal_tsr_msl_ecod_noZeros.nc"
        ),
    )
    parser.add_argument("--weights_path", type=Path, default=Path("old/model_weights.npz"))
    parser.add_argument("--scalar_dir", type=Path, default=Path("old/old_scalar"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures/old_model"))
    parser.add_argument("--years", default="1991,1993,1995,1997,1999,2001,2003,2005,2007,2009,2011,2013,2015,2017,2019")
    parser.add_argument("--kernel_dir", type=Path, default=Path("data/fal/kernels"))
    parser.add_argument("--skip_kernel_tests", action="store_true")
    parser.add_argument("--run_feature_tests", action="store_true")
    parser.add_argument("--batch_size", type=int, default=262144)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(args.data_path, parse_years(args.years))
    scaled = old_scale(ds, args.scalar_dir)
    inputs = (
        scaled[OLD_INPUT_ORDER]
        .to_dataarray()
        .transpose("date", "latitude", "longitude", "variable")
        .to_numpy()
    )
    model = load_old_model(args.weights_path)
    pred_scaled = predict(model, inputs, args.batch_size)
    pred_tsr = old_inverse_tsr(pred_scaled, args.scalar_dir)
    write_outputs(ds, pred_tsr, args.output_dir)

    ds_all = load_dataset(args.data_path, None)
    tsr_ecod_fal_contour_test(
        ds_all,
        model,
        args.scalar_dir,
        args.output_dir,
        scatter=False,
        extrapolate=True,
    )
    kernel_ecod_fal_contour_test(
        ds_all,
        model,
        args.scalar_dir,
        args.output_dir,
        scatter=False,
        extrapolate=True,
    )

    kernels = load_rrtm_kernels(args.kernel_dir)
    available_dates = set(str(value)[:7] for value in ds_all.date.values)
    for date in ["2013-09", "2015-09", "2015-12"]:
        if date not in available_dates:
            print(f"Skipping old kernel date test for missing dataset date: {date}")
            continue
        kernel_date_test(
            ds_all,
            model,
            args.scalar_dir,
            kernels,
            date,
            args.output_dir,
            args.batch_size,
        )
    second_order_test(
        ds_all,
        model,
        args.scalar_dir,
        kernels,
        ["2012-09", "2013-09"],
        args.output_dir,
        args.batch_size,
    )

if __name__ == "__main__":
    main()
