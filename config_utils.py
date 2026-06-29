from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path) -> DictConfig:
    config = _load_config_recursive(Path(config_path).resolve(), seen=set())
    validate_config(config)
    return config


def validate_config(config: DictConfig) -> None:
    if "dataset" not in config:
        return

    if "model" in config and "input_vars" in config.dataset:
        input_vars = config.dataset.input_vars
        if input_vars and config.model.input_dim != len(input_vars):
            raise ValueError(
                f"model.input_dim={config.model.input_dim} but dataset.input_vars has {len(input_vars)} fields."
            )

    if "clear_sky_zero_vars" in config.dataset and "input_vars" in config.dataset:
        input_vars = config.dataset.input_vars
        missing = [
            var for var in config.dataset.clear_sky_zero_vars
            if var not in input_vars
        ]
        if missing:
            raise ValueError(f"clear_sky_zero_vars not in dataset.input_vars: {missing}")

    if (
        "clear_sky_training" in config.dataset
        and config.dataset.clear_sky_training.enabled
        and not config.dataset.clear_sky_zero_vars
    ):
        raise ValueError("clear_sky_training requires dataset.clear_sky_zero_vars.")

    if "train" in config and "sobolev" in config.train:
        sobolev_vars = config.train.sobolev_vars
        input_vars = config.dataset.input_vars
        if config.train.sobolev and not sobolev_vars:
            raise ValueError("train.sobolev=True requires train.sobolev_vars.")
        missing = [var for var in sobolev_vars if var not in input_vars]
        if missing:
            raise ValueError(f"train.sobolev_vars not in dataset.input_vars: {missing}")

        if config.dataset.clear_sky_training.enabled and config.train.sobolev:
            if len(sobolev_vars) != 1:
                raise ValueError("clear_sky_training with Sobolev requires exactly one sobolev var.")
            if "TOA_clr" not in config.dataset.kernels.vars:
                raise ValueError("clear_sky_training with Sobolev requires TOA_clr in dataset.kernels.vars.")


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
