import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr

from config_utils import load_config, variable_config_from_omegaconf
from utils import (
    generate_paths_yearly,
    load_model_and_preprocessor,
    make_era5_filename,
    plot_global_field,
)


def load_year(config, date, data_path):
    year = int(date[:4])
    ds = xr.open_mfdataset(
        generate_paths_yearly(data_path, [year], make_era5_filename),
        combine="nested",
        concat_dim="date",
    )

    return ds


def first_layer_pca(ds, model, preprocessor, config, date):
    vconf = variable_config_from_omegaconf(config)
    processed = preprocessor.transform(ds)
    selected = processed.sel(date=slice(date, date))
    if selected.sizes["date"] != 1:
        raise ValueError(f"Expected one date matching {date}, got {selected.sizes['date']}.")
    date_idx = int(np.where(processed.date.values == selected.date.values[0])[0][0])

    inputs_np = vconf.inputs_np(
        processed, ["date", "latitude", "longitude", "variable"]
    )
    date_size = inputs_np.shape[0]
    lat_size = inputs_np.shape[1]
    lon_size = inputs_np.shape[2]
    flat_inputs = inputs_np.reshape(-1, inputs_np.shape[-1])

    first_layer = model.model[:2]
    with torch.no_grad():
        hidden = first_layer(torch.from_numpy(flat_inputs).float()).numpy()

    if hidden.shape[1] < 2:
        raise ValueError("First hidden layer must have at least 2 dimensions for PCA.")

    centered = hidden - hidden.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    all_scores = (centered @ vt[:2].T).reshape(date_size, lat_size, lon_size, 2)
    scores = all_scores[date_idx].reshape(-1, 2)
    pc1 = all_scores[date_idx, :, :, 0]
    pc2 = all_scores[date_idx, :, :, 1]
    return scores, pc1, pc2, processed.latitude.values, processed.longitude.values


def plot_scatter(scores, output_path):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    ax.scatter(scores[:, 0], scores[:, 1], s=2, alpha=0.35, linewidths=0)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("First-layer activations")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--date", required=True, help="YYYY-MM")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--output_dir", default="autoencoder_plots")
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
    ds = load_year(
        config, args.date, Path(args.data_path or config.dataset.era5.raw_path)
    )
    scores, pc1, pc2, lat, lon = first_layer_pca(
        ds, model, preprocessor, config, args.date
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_scatter(scores, output_dir / "scatter.png")
    vmax = max(float(np.max(np.abs(pc1))), float(np.max(np.abs(pc2))))
    plot_global_field(
        pc1,
        lon,
        lat,
        f"First-layer PC1; {args.date}",
        output_dir / "pc1_map.png",
        vmin=-vmax,
        vmax=vmax,
        label="PC score",
    )
    plot_global_field(
        pc2,
        lon,
        lat,
        f"First-layer PC2; {args.date}",
        output_dir / "pc2_map.png",
        vmin=-vmax,
        vmax=vmax,
        label="PC score",
    )


if __name__ == "__main__":
    main()
