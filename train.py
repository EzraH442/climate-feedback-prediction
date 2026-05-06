import argparse
import os
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from dataloader import ClimateTorchDataset, make_era5_filename
from model import SimpleModelTrainer


def stage_training_data(config_path: str) -> str:
    config = OmegaConf.load(config_path)
    slurm_tmpdir = Path(
        os.environ.get("SLURM_TMPDIR", Path(config.dataset.era5.path).resolve())
    )
    if not slurm_tmpdir.exists():
        raise FileNotFoundError(f"SLURM_TMPDIR does not exist: {slurm_tmpdir}.")
    staged_data_dir = slurm_tmpdir / "era5_processed"
    staged_data_dir.mkdir(parents=True, exist_ok=True)

    years_to_stage = sorted(
        set([*config.dataset.train_years, *config.dataset.val_years])
    )

    for year in years_to_stage:
        filename = make_era5_filename(year)
        source = Path(config.dataset.era5.path) / filename
        destination = staged_data_dir / filename
        if not source.exists():
            raise FileNotFoundError(f"Required training file not found: {source}")
        if not destination.exists():
            print(f"Copying {source} -> {destination}")
            shutil.copy2(source, destination)

    config.dataset.era5.path = str(staged_data_dir)
    runtime_config_path = slurm_tmpdir / "train_runtime_config.yaml"
    OmegaConf.save(config, runtime_config_path)
    return str(runtime_config_path)


def train(config_path: str):

    config = OmegaConf.load(config_path)

    train_dataset = ClimateTorchDataset(
        config_path=config_path,
        data_type="train",
    )
    val_dataset = ClimateTorchDataset(
        config_path=config_path,
        data_type="val",
    )

    torch.manual_seed(config.seed)
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = False
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        pin_memory = True

    print(f"Number of workers for DataLoader: {num_workers}")
    print(f"Using device: {device}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    checkpoint_path = None  # Set to a valid path to resume from checkpoint
    if (Path(config.train.checkpoint_dir) / "best_model.pt").exists():
        checkpoint_path = str(Path(config.train.checkpoint_dir) / "best_model.pt")

    trainer = SimpleModelTrainer(
        config=config, device=device, checkpoint_path=checkpoint_path
    )
    trainer.train_model(train_dataloader, val_dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train model on ERA5 data")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to config file",
        default="config_train.yaml",
    )
    args = parser.parse_args()

    train(args.config_file)


if __name__ == "__main__":
    main()
