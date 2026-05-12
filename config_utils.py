from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(config_path: str | Path) -> DictConfig:
    return _load_config_recursive(Path(config_path).resolve(), seen=set())


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
