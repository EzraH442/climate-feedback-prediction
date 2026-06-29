from comet_ml import ExistingExperiment, Experiment
import argparse
import os
import shutil
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import omegaconf
from omegaconf import OmegaConf

from config_utils import load_config
from dataloader import ClimateTorchDataset, KernelDataset, make_era5_filename
from model import (
    SimpleModelTrainer,
    SimpleModelSobolevTrainer,
)


def comet_experiment_key_path(checkpoint_dir: str) -> Path:
    return Path(checkpoint_dir) / "comet_experiment_key.txt"


def stage_training_data(config_path: str) -> str:
    config = load_config(config_path)
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


def create_comet_experiment(
    config: omegaconf.DictConfig, checkpoint_path: str | None = None
):
    api_key = os.environ.get("COMET_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "COMET_API_KEY is not set. Export it before running train.py."
        )

    key_path = comet_experiment_key_path(config.train.checkpoint_dir)
    experiment_key = None
    if checkpoint_path is not None and key_path.exists():
        experiment_key = key_path.read_text(encoding="ascii").strip() or None

    if checkpoint_path and experiment_key:
        experiment = ExistingExperiment(
            api_key=api_key,
            previous_experiment=experiment_key,
            log_env_details=False,
            log_env_gpu=True,
            log_env_cpu=True,
        )
    else:
        experiment = Experiment(
            api_key=api_key,
            project_name=config.wandb.project,
            workspace=config.wandb.entity,
            log_env_details=False,
            log_env_gpu=True,
            log_env_cpu=True,
        )
        experiment.log_parameters(OmegaConf.to_container(config, resolve=True))

    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(f"{experiment.get_key()}\n", encoding="ascii")
    return experiment


def train(config_path: str, resume: bool = True):
    staged_config_path = stage_training_data(config_path)
    config = load_config(staged_config_path)
    assert isinstance(config, omegaconf.DictConfig), ""

    if config.train.sobolev:
        train_dataset = KernelDataset(
            config_path=staged_config_path,
            data_type="train",
        )
        val_dataset = KernelDataset(
            config_path=staged_config_path,
            data_type="val",
        )
    else:
        train_dataset = ClimateTorchDataset(
            config_path=staged_config_path,
            data_type="train",
        )
        val_dataset = ClimateTorchDataset(
            config_path=staged_config_path,
            data_type="val",
        )

    torch.manual_seed(config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory = False
    num_workers = max(0, int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) - 1)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        pin_memory = True
        num_workers += 1

    print(f"Number of workers for DataLoader: {num_workers}")
    print(f"Using device: {device}")

    train_sampler = None
    train_shuffle = True
    if config.train.area_weighted_sampling:
        train_sampler = WeightedRandomSampler(
            weights=train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=torch.Generator().manual_seed(config.seed),
        )
        train_shuffle = False

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    checkpoint_path = None  # Set to a valid path to resume from checkpoint
    if resume:
        if not Path(config.train.checkpoint_dir).exists():
            raise FileNotFoundError(
                f"Checkpoint directory does not exist: {config.train.checkpoint_dir}"
            )
        checkpoint_files = list(
            Path(config.train.checkpoint_dir).glob("model_epoch_*.pt")
        )
        if not checkpoint_files:
            raise FileNotFoundError(
                f"No checkpoint files found in {config.train.checkpoint_dir} to resume from."
            )

        max_epoch = 0
        for file in checkpoint_files:
            epoch = int(file.stem.split("_")[-1])
            max_epoch = max(max_epoch, epoch)
        checkpoint_path = str(
            Path(config.train.checkpoint_dir) / f"model_epoch_{max_epoch}.pt"
        )
        print(f"Resuming training from checkpoint: {checkpoint_path}")

    experiment = create_comet_experiment(config, checkpoint_path=checkpoint_path)

    if config.train.sobolev:
        trainer = SimpleModelSobolevTrainer(
            config=config,
            experiment=experiment,
            device=device,
            checkpoint_path=checkpoint_path,
        )
    else:
        trainer = SimpleModelTrainer(
            config=config,
            experiment=experiment,
            device=device,
            checkpoint_path=checkpoint_path,
        )
    trainer.train_model(train_dataloader, val_dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train model on ERA5 data")
    parser.add_argument(
        "--config_file",
        type=str,
        required=False,
        help="Path to config file",
        default="configs/model/baseline.yaml",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to resume training from checkpoint if available",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Whether to start training from scratch",
    )
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    train(args.config_file, args.resume)


if __name__ == "__main__":
    main()
