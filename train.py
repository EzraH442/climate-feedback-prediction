import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from dataloader import ClimateTorchDataset
from model import SimpleModelTrainer

config_path = "config.yaml"
config = OmegaConf.load(config_path)
torch.manual_seed(config.seed)

train_dataset = ClimateTorchDataset(config_path=config_path, data_type="train")
val_dataset = ClimateTorchDataset(config_path=config_path, data_type="val")

train_dataloader = DataLoader(
    train_dataset,
    batch_size=config.train.batch_size,
    shuffle=True,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=config.train.batch_size,
    shuffle=False,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

trainer = SimpleModelTrainer(config_file=config_path, device=device)
trainer.train_model(train_dataloader, val_dataloader)
