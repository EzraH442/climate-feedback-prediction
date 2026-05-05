import os
import torch
import torch.nn as nn
import wandb
import omegaconf
import time


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


class SimpleModelTrainer:
    def __init__(
        self,
        config,
        checkpoint_path: str | None = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.loss_fn = nn.MSELoss()
        self.best_val_loss = float("inf")  # Track for saving the "best" model
        self.epoch = 0  # Track current epoch for checkpointing
        self.best_epoch = 0  # Track epoch of the best model
        self.total_training_time = 0.0  # Track total training time across epochs

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
            self.best_val_loss = checkpoint["best_val_loss"]
            self.epoch = checkpoint["epoch"] + 1
            self.best_epoch = checkpoint["best_epoch"]
            self.total_training_time = checkpoint["total_training_time"]
        else:
            self.config = config
            self.model = SimpleModel(self.config).to(self.device)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config.optimizer.learning_rate
            )

        # --- WandB Setup ---
        wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            config=omegaconf.OmegaConf.to_container(self.config, resolve=True),
            resume="allow" if checkpoint_path else None,
        )

    def checkpoint(self, epoch, is_best=False):
        if not os.path.exists(self.config.train.checkpoint_dir):
            os.makedirs(self.config.train.checkpoint_dir)

        checkpoint_data = {
            "epoch": epoch,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "total_training_time": self.total_training_time,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }

        # Save regular checkpoint
        if epoch % 10 == 0:
            torch.save(
                checkpoint_data,
                f"{self.config.train.checkpoint_dir}/model_epoch_{epoch}.pt",
            )

        # Save "best" model separately
        if is_best:
            torch.save(
                checkpoint_data, f"{self.config.train.checkpoint_dir}/best_model.pt"
            )

    def train_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ):
        for epoch in range(self.epoch, self.config.train.epochs):
            epoch_start_time = time.perf_counter()
            print(f"Epoch {epoch} started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

            # --- TRAINING PHASE ---
            self.model.train()
            train_loss = 0

            for i, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                y_pred = self.model(x)
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                if i % 100 == 0:
                    print(f"Batch {i}/{len(train_loader)} | Loss: {loss.item():.4e}")

            avg_train_loss = train_loss / len(train_loader)
            print(f"avg_train_loss: {avg_train_loss:.4e}")

            # --- VALIDATION PHASE ---
            self.model.eval()
            val_loss = 0
            with torch.no_grad():  # Disable gradient calculation to save memory/time
                for i, (x_val, y_val) in enumerate(val_loader):
                    x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                    val_pred = self.model(x_val)
                    v_loss = self.loss_fn(val_pred, y_val)
                    val_loss += v_loss.item()
                    if i % 100 == 0:
                        print(
                            f"Val Batch {i}/{len(val_loader)} | Val Loss: {v_loss.item():.4e}"
                        )

            avg_val_loss = val_loss / len(val_loader)
            print(f"avg_val_loss: {avg_val_loss:.4e}")

            # --- TIME TRACKING ---
            epoch_end_time = time.perf_counter()
            epoch_duration = epoch_end_time - epoch_start_time
            self.total_training_time += epoch_duration
            print(
                f"Epoch {epoch} completed in {epoch_duration:.2f} seconds. Total training time: {self.total_training_time:.2f} seconds."
            )

            # --- LOGGING & CHECKPOINTING ---
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss": avg_train_loss,
                    "val/loss": avg_val_loss,
                    "time/epoch_duration": epoch_duration,
                    "time/total_training_time_hours": self.total_training_time / 3600,
                }
            )

            is_best = avg_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = avg_val_loss
                self.best_epoch = epoch

            if epoch % 10 == 0 or is_best:
                print(
                    f"Epoch {epoch:04d} | Train: {avg_train_loss:.4e} | Val: {avg_val_loss:.4e}"
                )
                self.checkpoint(epoch, is_best=is_best)

            # --- EARLY STOPPING ---
            if avg_val_loss < self.config.train.early_stopping_threshold:
                print(
                    f"Early stopping at epoch {epoch} with val loss {avg_val_loss:.4e}"
                )
                break

            elif (
                epoch - self.best_epoch
                >= self.config.train.max_epochs_without_improvement
            ):
                print(
                    f"Early stopping at epoch {epoch} due to no improvement for {self.config.train.max_epochs_without_improvement} epochs"
                )
                break

        print(f"Training complete. Total time: {self.total_training_time/3600:.2f}h")
        wandb.finish()
