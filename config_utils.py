from pathlib import Path

from omegaconf import DictConfig, OmegaConf

import xarray as xr
import numpy as np

def load_config(config_path: str | Path) -> DictConfig:
    config = _load_config_recursive(Path(config_path).resolve(), seen=set())
    validate_config(config)
    return config


def validate_config(config: DictConfig) -> None:
    if "dataset" not in config:
        return

    if "preprocess" in config and "ecod" in config.preprocess:
        if config.preprocess.ecod.method not in ("true", "true_tcc", "fast"):
            raise ValueError(
                "preprocess.ecod.method must be 'true', 'true_tcc', or 'fast'."
            )

    if "model" in config and "input_vars" in config.dataset:
        input_vars = config.dataset.input_vars
        if input_vars and config.model.input_dim != len(input_vars):
            raise ValueError(
                f"model.input_dim={config.model.input_dim} but dataset.input_vars has {len(input_vars)} fields."
            )

    if "train" in config and "sobolev" in config.train:
        sobolev_vars = config.train.sobolev_vars
        input_vars = config.dataset.input_vars
        if config.train.sobolev and not sobolev_vars:
            raise ValueError("train.sobolev=True requires train.sobolev_vars.")
        missing = [var for var in sobolev_vars if var not in input_vars]
        if missing:
            raise ValueError(f"train.sobolev_vars not in dataset.input_vars: {missing}")
        
        if config.train.sobolev:
            missing = [
                var for var in sobolev_vars if var not in config.dataset.kernel_vars
            ]
            if missing:
                raise ValueError(
                    f"train.sobolev_vars not in dataset.kernel_vars: {missing}"
                )


def _load_config_recursive(config_path: Path, seen: set[Path]) -> DictConfig:
    if config_path in seen:
        chain = " -> ".join(str(path) for path in [*seen, config_path])
        raise ValueError(f"Cyclic config base reference detected: {chain}")

    seen = set(seen)
    seen.add(config_path)

    loaded = OmegaConf.load(config_path)
    assert isinstance(loaded, DictConfig)

    base_refs = loaded.pop("base", None)
    if base_refs is None:
        return loaded

    if isinstance(base_refs, str):
        base_refs = [base_refs]

    merged = OmegaConf.create()
    for base_ref in base_refs:
        base_path = (config_path.parent / str(base_ref)).resolve()
        merged = OmegaConf.merge(merged, _load_config_recursive(base_path, seen))

    return OmegaConf.merge(merged, loaded)

from dataclasses import dataclass

@dataclass
class VariableConfig:
    input_vars: list[str]
    target_var: str
    clear_sky_zero_vars: list[str]
    clear_sky_target: str
    kernel_vars: list[str]


    def all_vars(self) -> list[str]:
        return list(
            dict.fromkeys(
                [
                    *self.kernel_vars,
                    *self.input_vars,
                    self.target_var,
                    *self.clear_sky_zero_vars,
                    self.clear_sky_target,
                ]
            )
        )

    def input_order(self):
        return list(dict.fromkeys([*self.kernel_vars, *self.input_vars]))

    def kern_input_order(self):
        return list(dict.fromkeys(self.kernel_vars))

    def clear_sky_input(self, ds: xr.Dataset) -> xr.Dataset:
        zero_vars = list(self.clear_sky_zero_vars)
        assign_dict = {var: xr.zeros_like(ds[var]) for var in zero_vars if var in ds}
        return ds.assign(assign_dict)

    def inputs(self, ds: xr.Dataset) -> xr.Dataset:
        return ds[self.input_order()]
    
    def kern_inputs(self, ds: xr.Dataset, clear=False) -> xr.Dataset:
        out = ds[self.kern_input_order()]
        if clear:
            return out.sel(all_clr='clr')
        else:
            return out.sel(all_clr='all')
            
    def outputs(self, ds: xr.Dataset, clear=False) -> xr.DataArray:
        if clear:
            return ds[self.clear_sky_target]
        else:
            return ds[self.target_var]

    def outputs_np(self, ds: xr.Dataset, clear=False) -> np.ndarray:
        return self.outputs(ds, clear).to_numpy()

    def inputs_np(self, ds: xr.Dataset, dim_order=None) -> np.ndarray:
        inputs = self.inputs(ds)
        da = inputs.to_dataarray()
        if dim_order is not None:
            da = da.transpose(*dim_order)
        return da.values

    def kern_inputs_np(self, ds: xr.Dataset, dim_order=None, clear=False) -> np.ndarray:
        inputs = self.kern_inputs(ds, clear)
        da = inputs.to_dataarray()
        if dim_order is not None:
            da = da.transpose(*dim_order)
        return da.values


def variable_config_from_omegaconf(config: DictConfig) -> VariableConfig:
    return VariableConfig(
        input_vars=list(config.dataset.input_vars),
        target_var=config.dataset.target_var,
        clear_sky_zero_vars=list(config.dataset.clear_sky_zero_vars),
        clear_sky_target=config.dataset.clear_sky_var,
        kernel_vars=list(config.dataset.kernel_vars),
    )
