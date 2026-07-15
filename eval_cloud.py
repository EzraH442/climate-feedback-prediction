import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from config_utils import load_config, variable_config_from_omegaconf
from preprocess import load_cloud_profiles
from ecod_calculation import ecod_from_profiles
from preprocessing import fast_compute_cloud_optical_depth
from utils import (
    SECONDS_PER_DAY,
    generate_paths_yearly,
    load_model_and_preprocessor,
    make_era5_filename,
)


CLOUD_VARS = ["lcc", "mcc", "hcc"]
CLOUD_LABELS = ["Low cloud cover", "Medium cloud cover", "High cloud cover"]


def load_point(config, date, latitude, longitude, data_path):
    year = int(date[:4])
    ds = xr.open_mfdataset(
        generate_paths_yearly(data_path, [year], make_era5_filename),
        combine="nested",
        concat_dim="date",
    ).sel(date=date, latitude=latitude, longitude=longitude, method="nearest")

    if config.preprocess.ecod.enabled:
        if config.preprocess.ecod.method == "fast":
            ecod = fast_compute_cloud_optical_depth(ds.tclw, ds.tciw, ds.tcc)
        else:
            point_grid = xr.Dataset(
                coords={
                    "latitude": [float(ds.latitude)],
                    "longitude": [float(ds.longitude)],
                }
            )
            profiles = load_cloud_profiles(
                data_path, [year], [int(date[5:7])], point_grid
            )
            ecod = ecod_from_profiles(
                profiles.ciwc,
                profiles.clwc,
                profiles.level * 100.0,
            ).sel(date=date, method="nearest")
        ds = ds.assign(ecod=ecod)

    return ds


def cloud_derivatives(ds, model, preprocessor, config, ecod_values):
    vconf = variable_config_from_omegaconf(config)
    feature_names = vconf.input_order()
    scaler = preprocessor.scalar
    vmin = scaler.get_data_min()
    vmax = scaler.get_data_max()
    target_range = float(vmax[vconf.target_var] - vmin[vconf.target_var])

    rows = []
    for cloud_var in CLOUD_VARS:
        varied = xr.concat(
            [ds.assign(ecod=float(ecod)) for ecod in ecod_values],
            dim=xr.IndexVariable("ecod_value", ecod_values),
        )
        processed = preprocessor.transform(varied)
        inputs_np = vconf.inputs_np(processed, ["ecod_value", "variable"])
        inputs = torch.from_numpy(inputs_np).float().requires_grad_(True)
        outputs = model(inputs)
        grads = torch.autograd.grad(outputs.sum(), inputs)[0]
        input_range = float(vmax[cloud_var] - vmin[cloud_var])
        rows.append(
            grads[:, feature_names.index(cloud_var)].detach().numpy()
            * target_range
            / input_range
            / SECONDS_PER_DAY
        )
    return np.vstack(rows)


def plot_cloud_derivatives(ecod_values, values, output_path, title):
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=300)
    vmax = float(np.nanmax(np.abs(values)))
    if vmax == 0:
        vmax = 1.0
    image = ax.imshow(
        values,
        aspect="auto",
        origin="lower",
        extent=[ecod_values[0], ecod_values[-1], -0.5, 2.5],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_yticks(range(3), CLOUD_LABELS)
    ax.set_xlabel("Effective cloud optical depth")
    ax.set_xscale("log")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label=r"$dR/dy$ [$W/m^2$]")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--date", required=True, help="YYYY-MM")
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--output_path", default="cloud_derivatives.png")
    parser.add_argument("--n_ecod", type=int, default=100)
    parser.add_argument("--min_ecod", type=float, default=1e-3)
    args = parser.parse_args()

    config = load_config(args.config_file)
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else Path(config.train.checkpoint_dir) / "best_model.pt"
    )
    model, preprocessor, _ = load_model_and_preprocessor(
        config, checkpoint_path, downscaling=False
    )

    ds = load_point(
        config,
        args.date,
        args.latitude,
        args.longitude,
        Path(args.data_path or config.dataset.era5.raw_path),
    )
    ecod_max = float(preprocessor.scalar.get_data_max()["ecod"])
    ecod_values = np.geomspace(args.min_ecod, ecod_max, args.n_ecod)
    values = cloud_derivatives(ds, model, preprocessor, config, ecod_values)
    plot_cloud_derivatives(
        ecod_values,
        values,
        Path(args.output_path),
        f"{args.date}; lat={float(ds.latitude):.2f}, lon={float(ds.longitude):.2f}",
    )


if __name__ == "__main__":
    main()
