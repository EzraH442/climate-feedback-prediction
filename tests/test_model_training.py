import sys
import warnings
from pathlib import Path

import torch
import xarray as xr
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings(
    "ignore", message="urllib3 .* doesn't match a supported version!"
)

from model import (
    SimpleModelSobolevTrainer,
)
from preprocessing import PlanetaryAlbedoCalculator, XarrayMinMaxScaler


def test_sobolev_var_weights_scale_matching_gradient_columns():
    config = OmegaConf.create(
        {
            "dataset": {"input_vars": ["fal", "tcwv"], "kernel_vars": ["fal", "tcwv"]},
            "model": {
                "input_dim": 2,
                "hidden_dim_sizes": [2],
                "activation": "tanh",
            },
            "optimizer": {"learning_rate": 0.001},
            "train": {
                "sobolev_alpha": 0.5,
                "sobolev_vars": ["fal", "tcwv"],
                "sobolev_var_weights": {"tcwv": 3.0},
            },
        }
    )
    trainer = SimpleModelSobolevTrainer(config, experiment=None)
    for param in trainer.model.parameters():
        param.data.zero_()

    _, loss_1, total_loss = trainer._run_epoch(
        [(torch.ones(1, 2), torch.zeros(1), torch.tensor([[1.0, 2.0]]))],
        train=False,
    )
    assert loss_1 == 3.25
    assert total_loss == 3.25


def test_planetary_albedo_calculator_adds_all_and_clear_sky_fields():
    ds = xr.Dataset(
        {
            "tsr": ("x", [2.0, 6.0]),
            "tsrc": ("x", [1.0, 3.0]),
            "tisr": ("x", [4.0, 12.0]),
        }
    )
    out = PlanetaryAlbedoCalculator().transform(ds)
    assert out["pal"].to_numpy().tolist() == [0.5, 0.5]
    assert out["palc"].to_numpy().tolist() == [0.25, 0.25]


def test_minmax_scaler_warns_when_transform_exceeds_fit_range():
    scaler = XarrayMinMaxScaler(dim="x")
    scaler.fit(xr.Dataset({"a": ("x", [0.0, 1.0])}))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scaler.transform(xr.Dataset({"a": ("x", [2.0])}))

    assert any("outside [-1, 1]" in str(w.message) for w in caught)


if __name__ == "__main__":
    test_lm_optimizer_uses_lbfgs()
    test_sobolev_var_weights_scale_matching_gradient_columns()
    test_planetary_albedo_calculator_adds_all_and_clear_sky_fields()
    test_minmax_scaler_warns_when_transform_exceeds_fit_range()
