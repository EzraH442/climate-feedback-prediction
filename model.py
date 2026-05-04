import os
import torch
import torch.nn as nn
import wandb
import omegaconf

class SimpleModel(nn.Module):
    def __init__(self, config):
        super(SimpleModel, self).__init__()
        self.config = config
        self.model = nn.Sequential(
            nn.Linear(config.model.input_dim, config.model.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.model.hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        return self.model(x)

class SimpleModelTrainer(nn.Module):
    def __init__(
        self, config_file: str = "config.yaml", checkpoint_path: str | None = None, device: str = "cpu"
    ):
        super(SimpleModelTrainer, self).__init__()
        self.device = torch.device(device)
        self.loss_fn = nn.MSELoss()
        self.best_val_loss = float('inf') # Track for saving the "best" model

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.config = checkpoint["config"]
            self.model = SimpleModel(self.config).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config.optimizer.learning_rate
            )
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.epoch = checkpoint["epoch"] + 1
        else:
            self.config = omegaconf.OmegaConf.load(config_file)
            self.model = SimpleModel(self.config).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config.optimizer.learning_rate
            )
            self.epoch = 0

        # --- WandB Setup ---
        wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            config=omegaconf.OmegaConf.to_container(self.config, resolve=True),
            resume="allow" if checkpoint_path else None
        )

    def checkpoint(self, epoch, val_loss, is_best=False):
        if not os.path.exists(self.config.train.checkpoint_dir):
            os.makedirs(self.config.train.checkpoint_dir)
            
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "val_loss": val_loss,
        }
        
        # Save regular checkpoint
        if epoch % 10 == 0:
            torch.save(checkpoint_data, f"{self.config.train.checkpoint_dir}/model_epoch_{epoch}.pt")
        
        # Save "best" model separately
        if is_best:
            torch.save(checkpoint_data, f"{self.config.train.checkpoint_dir}/best_model.pt")

    def train_model(self, train_loader, val_loader):
        for epoch in range(self.epoch, self.config.train.epochs):
            # --- TRAINING PHASE ---
            self.model.train()
            train_loss = 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)

            # --- VALIDATION PHASE ---
            self.model.eval()
            val_loss = 0
            with torch.no_grad(): # Disable gradient calculation to save memory/time
                for x_val, y_val in val_loader:
                    x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                    val_pred = self.model(x_val)
                    v_loss = self.loss_fn(val_pred, y_val)
                    val_loss += v_loss.item()

            avg_val_loss = val_loss / len(val_loader)

            # --- LOGGING & CHECKPOINTING ---
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_train_loss,
                "val/loss": avg_val_loss
            })

            is_best = avg_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = avg_val_loss

            if epoch % 10 == 0 or is_best:
                print(f"Epoch {epoch:04d} | Train: {avg_train_loss:.4e} | Val: {avg_val_loss:.4e}")
                self.checkpoint(epoch, avg_val_loss, is_best=is_best)
        
        wandb.finish()