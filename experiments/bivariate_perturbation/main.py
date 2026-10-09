# Example: python experiments/bivariate_perturbation/main.py --config_file configs/model/fal/2011-2014_3,6,9,12_baseline.yaml --date 2015-09
import sys
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from tap import Tap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config, variable_config_from_omegaconf
from experiments.common import (
    parse_args_and_confirm,
    write_netcdf,
)
from utils import (
    SECONDS_PER_DAY,
    generate_paths_yearly,
    kernel_delta,
    load_model_and_preprocessor,
    make_era5_filename,
    nn_pred,
    scale_minmax_value,
)


class BivariatePerturbationArgs(Tap):
    config_file: Path
    checkpoint_path: Path | None = None
    output_dir: Path | None = None
    era5_data_path: Path = Path("data/era5")
    date: str
    var_a: str = "fal"
    var_b: str = "tcwv"
    clear_sky: bool = False

    def process_args(self):
        self.output_dir = Path(self.output_dir or "bivariate_perturbation")
        self.output_dir.mkdir(parents=True, exist_ok=True)


def collect_bivariate_input_data(data_path: Path, date: str) -> xr.Dataset:
    return xr.open_mfdataset(
        generate_paths_yearly(data_path, [int(date[:4])], make_era5_filename),
        combine="nested",
        concat_dim="date",
    ).sel(date=[date])


def contour_inputs(
    processed_ds: xr.Dataset,
    preprocessor,
    vc,
    var_a: str,
    var_b: str,
    date: str,
    latitude: float,
    longitude: float,
    n: int,
    extrapolate: bool,
):
    scaler = preprocessor.scalar
    feature_names = vc.input_order()
    ordered = vc.inputs(processed_ds)
    base_point = ordered.sel(date=date, latitude=latitude, longitude=longitude)
    observed_base_point = scaler.inverse_transform(base_point)
    base_features = np.array(
        [float(base_point[name]) for name in feature_names],
        dtype=np.float32,
    )

    vmin, vmax = scaler.get_data_min(), scaler.get_data_max()
    a_values = np.linspace(float(vmin[var_a]), float(vmax[var_a]), n)
    b_values = np.linspace(float(vmin[var_b]), float(vmax[var_b]), n)
    if extrapolate:
        a_values = np.linspace(0, float(vmax[var_a]), n)
        b_values = np.linspace(0, float(vmax[var_b]), n)

    a_grid, b_grid = np.meshgrid(a_values, b_values)
    inputs_np = np.broadcast_to(base_features, (n * n, len(feature_names))).copy()
    inputs_np[:, feature_names.index(var_a)] = scale_minmax_value(
        scaler, var_a, a_grid.ravel()
    )
    inputs_np[:, feature_names.index(var_b)] = scale_minmax_value(
        scaler, var_b, b_grid.ravel()
    )
    if "ecod_fal" in feature_names:
        inputs_np[:, feature_names.index("ecod_fal")] = scale_minmax_value(
            scaler, "ecod_fal", (b_grid * a_grid).ravel()
        )

    return {
        "scaler": scaler,
        "feature_names": feature_names,
        "inputs_np": inputs_np,
        "a_values": a_values,
        "b_values": b_values,
        "base_a": float(observed_base_point[var_a]),
        "base_b": float(observed_base_point[var_b]),
    }


def contour_dataset(values, context, var_a, var_b, field_name, attrs):
    return xr.Dataset(
        {field_name: (("var_b", "var_a"), values.astype("float32"))},
        coords={
            "var_a": ("var_a", context["a_values"].astype("float32")),
            "var_b": ("var_b", context["b_values"].astype("float32")),
        },
        attrs={
            **attrs,
            "var_a": var_a,
            "var_b": var_b,
            "base_var_a": context["base_a"],
            "base_var_b": context["base_b"],
        },
    )


def compute_kernel_contour(
    processed_ds: xr.Dataset,
    preprocessor,
    model,
    vc,
    var_a: str = "fal",
    var_b: str = "ecod",
    date: str = "2015-09",
    latitude: float = 83.625,
    longitude: float = 17.375,
    extrapolate: bool = False,
    n: int = 100,
) -> xr.Dataset:
    context = contour_inputs(
        processed_ds, preprocessor, vc, var_a, var_b, date, latitude, longitude, n, extrapolate
    )
    inputs = torch.from_numpy(context["inputs_np"]).float().requires_grad_(True)
    outputs = model(inputs)
    grads = torch.autograd.grad(outputs=outputs.sum(), inputs=inputs)[0][
        :, context["feature_names"].index(var_a)
    ]

    vmin, vmax = context["scaler"].get_data_min(), context["scaler"].get_data_max()
    target_range = float(vmax[vc.target_var] - vmin[vc.target_var])
    input_range = float(vmax[var_a] - vmin[var_a])
    kernel = (
        grads.detach().cpu().numpy() * (target_range / input_range) / SECONDS_PER_DAY
    ).reshape(n, n) * kernel_delta(var_a)

    return contour_dataset(
        kernel,
        context,
        var_a,
        var_b,
        "kernel",
        {
            "date": date,
            "latitude": latitude,
            "longitude": longitude,
            "target": vc.target_var,
        },
    )


def compute_tsr_contour(
    processed_ds: xr.Dataset,
    preprocessor,
    model,
    vc,
    var_a: str = "fal",
    var_b: str = "ecod",
    date: str = "2015-09",
    latitude: float = 83.625,
    longitude: float = 17.375,
    extrapolate: bool = False,
    n: int = 100,
) -> xr.Dataset:
    context = contour_inputs(
        processed_ds, preprocessor, vc, var_a, var_b, date, latitude, longitude, n, extrapolate
    )
    with torch.no_grad():
        outputs = model(torch.from_numpy(context["inputs_np"]).float()).numpy().reshape(n, n)

    scaler = context["scaler"]
    vmin, vmax = scaler.get_data_min(), scaler.get_data_max()
    target_min = float(vmin[vc.target_var])
    target_range = float(vmax[vc.target_var] - vmin[vc.target_var])
    target_scaled = (outputs - scaler.min_val) / (scaler.max_val - scaler.min_val)
    target_values = (target_scaled * target_range + target_min) / SECONDS_PER_DAY

    return contour_dataset(
        target_values,
        context,
        var_a,
        var_b,
        "flux",
        {
            "date": date,
            "latitude": latitude,
            "longitude": longitude,
            "target": vc.target_var,
        },
    )


def main():
    args = parse_args_and_confirm(BivariatePerturbationArgs())
    config = load_config(args.config_file)
    vc = variable_config_from_omegaconf(config)
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else Path(config.train.checkpoint_dir) / "best_model.pt"
    )
    model, preprocessor, _ = load_model_and_preprocessor(
        config, checkpoint_path, downscaling=False
    )
    output_dir = args.output_dir

    base = collect_bivariate_input_data(args.era5_data_path, args.date)
    if args.clear_sky:
        base = vc.clear_sky_input(base)

    da = kernel_delta(args.var_a)
    db = kernel_delta(args.var_b)
    a = base.assign({args.var_a: base[args.var_a] + da})
    b = base.assign({args.var_b: base[args.var_b] + db})
    ab = a.assign({args.var_b: a[args.var_b] + db})

    y0 = nn_pred(
        preprocessor.transform(base),
        model,
        preprocessor,
        vc,
        ["date", "latitude", "longitude"],
        clear=args.clear_sky,
    )
    ya = nn_pred(
        preprocessor.transform(a),
        model,
        preprocessor,
        vc,
        ["date", "latitude", "longitude"],
        clear=args.clear_sky,
    )
    yb = nn_pred(
        preprocessor.transform(b),
        model,
        preprocessor,
        vc,
        ["date", "latitude", "longitude"],
        clear=args.clear_sky,
    )
    yab = nn_pred(
        preprocessor.transform(ab),
        model,
        preprocessor,
        vc,
        ["date", "latitude", "longitude"],
        clear=args.clear_sky,
    )

    write_netcdf(
        output_dir / f"{args.date}_{args.var_a}_{args.var_b}.nc",
        xr.Dataset(
            {
                "base": y0 / SECONDS_PER_DAY,
                f"d_{args.var_a}": (ya - y0) / SECONDS_PER_DAY,
                f"d_{args.var_b}": (yb - y0) / SECONDS_PER_DAY,
                "d_both": (yab - y0) / SECONDS_PER_DAY,
            }
        ),
    )


if __name__ == "__main__":
    main()
