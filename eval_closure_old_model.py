import argparse
from pathlib import Path

import numpy as np
import xarray as xr

import eval_closure as closure
from eval_old_model import (
    OLD_INPUT_ORDER,
    fit_old_scaler,
    load_dataset,
    load_old_model,
    old_inverse_tsr,
    old_scale,
    predict,
)
from utils import SECONDS_PER_DAY, to_monthly


OLD_MODEL_DATA_PATH = Path(
    "old/era5_1deg_monthly_avg_sl_1990-2020_tisr_tciw_tclw_tcwv_hcc_mcc_lcc_sp_tco3_fal_tsr_msl_ecod_noZeros.nc"
)
OLD_MODEL_WEIGHTS_PATH = Path("model_weights.npz")
ERA5_DATA_PATH = Path("data/era5")
OLD_CLOUD_VARS = ["hcc", "mcc", "lcc", "tciw", "tclw", "ecod"]


class IdentityPreprocessor:
    def transform(self, ds):
        return ds


def monthly_old_inputs(scaled: xr.Dataset) -> np.ndarray:
    return (
        scaled[OLD_INPUT_ORDER]
        .to_dataarray()
        .transpose("year", "month", "latitude", "longitude", "variable")
        .to_numpy()
    )


def old_predict_monthly_tsr(model, ds, scaler, batch_size: int) -> xr.DataArray:
    pred_scaled = predict(model, monthly_old_inputs(old_scale(ds, scaler)), batch_size)
    pred = old_inverse_tsr(pred_scaled, scaler)
    return xr.DataArray(
        pred,
        dims=("year", "month", "latitude", "longitude"),
        coords={
            "year": ds.year,
            "month": ds.month,
            "latitude": ds.latitude,
            "longitude": ds.longitude,
        },
    )


def refresh_ecod_fal(ds: xr.Dataset) -> xr.Dataset:
    if "ecod" in ds and "fal" in ds:
        ds = ds.assign(ecod_fal=ds["ecod"] * ds["fal"])
    return ds


def old_clear_sky(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.copy(deep=True)
    for name in OLD_CLOUD_VARS:
        if name in ds:
            ds[name][:] = 0
    return refresh_ecod_fal(ds)


def old_nn_radiative_response(
    ds: xr.Dataset,
    anomaly: xr.Dataset,
    model,
    scaler,
    variables,
    batch_size: int,
) -> list[xr.DataArray]:
    pred_original = old_predict_monthly_tsr(model, ds, scaler, batch_size)
    results = []

    for var_list in variables:
        if not isinstance(var_list, list):
            var_list = [var_list]
        perturbed = ds.assign({name: ds[name] + anomaly[name] for name in var_list})
        perturbed = refresh_ecod_fal(perturbed)
        pred_perturbed = old_predict_monthly_tsr(model, perturbed, scaler, batch_size)
        results.append((pred_perturbed - pred_original) / SECONDS_PER_DAY)

    return results


def load_clear_sky_response(data_path: Path, ds: xr.Dataset) -> xr.DataArray:
    paths = sorted(data_path.glob("era5_tsrc_monthly_*.nc"))
    if not paths:
        raise FileNotFoundError(f"No era5_tsrc_monthly_*.nc files found in {data_path}")

    ds_tsrc = xr.open_mfdataset(paths).tsrc
    ds_tsrc = ds_tsrc.sel(date=slice(ds.date.min().values, ds.date.max().values))
    ds_tsrc = ds_tsrc.interp(latitude=ds.latitude, longitude=ds.longitude).compute()
    ds_tsrc_monthly = to_monthly(ds_tsrc / SECONDS_PER_DAY)
    return ds_tsrc_monthly - ds_tsrc_monthly.mean("year")


def build_old_response_dataset(
    ds_monthly: xr.Dataset,
    ds_monthly_clr: xr.Dataset,
    anomaly: xr.Dataset,
    dR: xr.DataArray,
    dR_clr: xr.DataArray,
    model,
    scaler,
    batch_size: int,
    skip_cross: bool,
) -> xr.Dataset:
    dR_a_nn, dR_c_nn, dR_q_nn, dR_co3_nn = old_nn_radiative_response(
        ds_monthly,
        anomaly,
        model,
        scaler,
        ["fal", OLD_CLOUD_VARS, "tcwv", "tco3"],
        batch_size,
    )
    dR_a_nn_clr, dR_q_nn_clr, dR_co3_nn_clr = old_nn_radiative_response(
        ds_monthly_clr,
        anomaly,
        model,
        scaler,
        ["fal", "tcwv", "tco3"],
        batch_size,
    )

    dR_a_k, dR_a_k_clr = closure.albedo_kernel_components(
        anomaly, xr.open_dataset(closure.ALBEDO_KERNEL_PATH).sel(month=[3,9])
    )
    ds_qt = xr.load_dataset(closure.QT_PATH)
    ds_qt = ds_qt.sel(date=ds_qt.date.dt.month.isin([3,9]))
    dR_q_k, dR_q_k_clr = closure.water_vapor_kernel_components(
        ds_monthly, ds_qt, xr.open_dataset(closure.WATER_VAPOR_KERNEL_PATH).sel(month=[3,9])
    )
    dR_c_k = (dR - dR_clr) - (dR_a_k - dR_a_k_clr) - (dR_q_k - dR_q_k_clr)

    data_vars = {
        "dR_era5_all": dR.assign_attrs(
            plot_label="ERA5 all", filename="old_dR_era5_all.png"
        ),
        "dR_era5_clr": dR_clr.assign_attrs(
            plot_label="ERA5 clear", filename="old_dR_era5_clr.png"
        ),
        "dR_a_nn_all": dR_a_nn.assign_attrs(
            plot_label="a", filename="old_dR_a.png", vmax=40
        ),
        "dR_c_nn_all": dR_c_nn.assign_attrs(
            plot_label="c", filename="old_dR_c.png", vmax=60
        ),
        "dR_q_nn_all": dR_q_nn.assign_attrs(
            plot_label="q", filename="old_dR_q.png", vmax=7
        ),
        "dR_a_nn_clr": dR_a_nn_clr.assign_attrs(
            plot_label="a,clr", filename="old_dR_a,clr.png", vmax=60
        ),
        "dR_q_nn_clr": dR_q_nn_clr.assign_attrs(
            plot_label="q,clr", filename="old_dR_q,clr.png", vmax=4
        ),
        "dR_a_k_all": dR_a_k.assign_attrs(
            plot_label="a", filename="old_k_dR_a.png", vmax=40
        ),
        "dR_c_k_all": dR_c_k.assign_attrs(
            plot_label="c", filename="old_k_dR_c.png", vmax=60
        ),
        "dR_q_k_all": dR_q_k.assign_attrs(
            plot_label="q", filename="old_k_dR_q.png", vmax=7
        ),
        "dR_a_k_clr": dR_a_k_clr.assign_attrs(
            plot_label="a,clr", filename="old_k_dR_a,clr.png", vmax=40
        ),
        "dR_q_k_clr": dR_q_k_clr.assign_attrs(
            plot_label="q,clr", filename="old_k_dR_q,clr.png", vmax=7
        ),
    }

    if not skip_cross:
        dR_aq_nn, dR_ac_nn, dR_qc_nn = old_nn_radiative_response(
            ds_monthly,
            anomaly,
            model,
            scaler,
            [
                ["fal", "tcwv"],
                ["fal", *OLD_CLOUD_VARS],
                ["tcwv", *OLD_CLOUD_VARS],
            ],
            batch_size,
        )
        dR_aq_nn = dR_aq_nn - dR_a_nn - dR_q_nn
        dR_ac_nn = dR_ac_nn - dR_a_nn - dR_c_nn
        dR_qc_nn = dR_qc_nn - dR_q_nn - dR_c_nn
        data_vars.update(
            {
                "dR_aq_nn_all": dR_aq_nn.assign_attrs(
                    plot_label="a,q", filename="old_cross_dR_a,q.png"
                ),
                "dR_ac_nn_all": dR_ac_nn.assign_attrs(
                    plot_label="a,c", filename="old_cross_dR_a,c.png"
                ),
                "dR_qc_nn_all": dR_qc_nn.assign_attrs(
                    plot_label="q,c", filename="old_cross_dR_q,c.png"
                ),
            }
        )

    responses = xr.Dataset(data_vars)
    responses["dR_co3_nn_all"] = dR_co3_nn
    responses["dR_co3_nn_clr"] = dR_co3_nn_clr
    return responses


def main():
    parser = argparse.ArgumentParser(description="Run old-model closure analysis.")
    parser.add_argument("--data_path", type=Path, default=OLD_MODEL_DATA_PATH)
    parser.add_argument("--weights_path", type=Path, default=OLD_MODEL_WEIGHTS_PATH)
    parser.add_argument("--era5_data_path", type=Path, default=ERA5_DATA_PATH)
    parser.add_argument("--output_dir", type=Path, default=Path("closure_test_2_old_model"))
    parser.add_argument("--year", type=int, default=2012)
    parser.add_argument("--month", type=int, default=9)
    parser.add_argument("--north_boundary", type=float, default=closure.NORTH_BOUNDARY)
    parser.add_argument("--residual_samples", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=262144)
    parser.add_argument("--skip_cross", action="store_true")
    args = parser.parse_args()

    old_ds_all = load_dataset(args.data_path, None).sel(date=slice('2007-01', '2016-12'))
    scaler = fit_old_scaler(old_ds_all)
    model = load_old_model(args.weights_path)

    albedo_kernel = xr.open_dataset(closure.ALBEDO_KERNEL_PATH)
    kernel_grid = {"latitude": albedo_kernel.latitude, "longitude": albedo_kernel.longitude}
    ds = refresh_ecod_fal(old_ds_all.interp(**kernel_grid))

    closure.NORTH_BOUNDARY = args.north_boundary
    closure.NORTH_MASK = ds.latitude.values > closure.NORTH_BOUNDARY

    ds_monthly = to_monthly(ds.copy(deep=True))
    ds_monthly_means = ds_monthly.mean("year")
    anomaly = ds_monthly - ds_monthly_means
    anomaly["tsr"] = anomaly["tsr"] / SECONDS_PER_DAY
    dR = anomaly["tsr"].compute()
    dR_clr = load_clear_sky_response(args.era5_data_path, ds)
    ds_monthly_clr = old_clear_sky(ds_monthly)

    print(ds_monthly)
    responses = build_old_response_dataset(
        ds_monthly=ds_monthly,
        ds_monthly_clr=ds_monthly_clr,
        anomaly=anomaly,
        dR=dR,
        dR_clr=dR_clr,
        model=model,
        scaler=scaler,
        batch_size=args.batch_size,
        skip_cross=args.skip_cross,
    )

    closure.date_closure_test(
        anomaly,
        ds_monthly,
        ds_monthly_means,
        IdentityPreprocessor(),
        responses,
        args.output_dir,
        args.year,
        args.month,
        dR_co3_nn=responses.get("dR_co3_nn_all"),
        dR_co3_nn_clr=responses.get("dR_co3_nn_clr"),
    )
    closure.timeseries_test(
        responses,
        args.output_dir,
        args.residual_samples,
    )


if __name__ == "__main__":
    main()
