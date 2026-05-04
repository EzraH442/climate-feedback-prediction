import torch
from omegaconf import OmegaConf
from dataloader import ClimateTorchDataset, prepare_data
from model import SimpleModelTrainer
from preprocessing import XarrayMinMaxScaler

config_path="config.yaml"
config = OmegaConf.load(config_path)
torch.manual_seed(config.seed)

dataset = ClimateTorchDataset(config_path=config_path)
train_dataloader, val_dataloader = prepare_data(config_path, dataset)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

trainer=SimpleModelTrainer(
    config_file=config_path,
    device=device
)

trainer.train_model(train_dataloader, val_dataloader)