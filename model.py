import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from comet_ml import Experiment
import omegaconf


def build_activation(name: str) -> nn.Module:
    normalized = name.lower()
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "relu":
        return nn.ReLU()
    raise ValueError(f"Unsupported activation: {name}")


class SimpleModel(nn.Module):
    def __init__(self, config):
        super(SimpleModel, self).__init__()
        self.config = config
        self.input_dim = config.model.input_dim
        hidden_dim_sizes = list(config.model.hidden_dim_sizes)
        if len(hidden_dim_sizes) == 0:
            raise ValueError("config.model.hidden_dim_sizes must not be empty")

        self.hidden_dim_sizes = hidden_dim_sizes
        self.activation_name = config.model.activation

        layers = []
        in_dim = self.input_dim
        for hidden_dim in hidden_dim_sizes:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(build_activation(self.activation_name))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.model(x).squeeze(-1)


class SimpleModelTrainer:
    def __init__(
        self,
        config: omegaconf.DictConfig,
        experiment: Experiment,
        checkpoint_path: str | None = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.loss_fn = nn.MSELoss()
        self.best_val_loss = float("inf")  # Track for saving the "best" model
        self.epoch = 0  # Track current epoch for checkpointing
        self.best_epoch = 0  # Track epoch of the best model
        self.total_training_time = 0.0  # Track total training time across epochs
        self.experiment = experiment
        self.scheduler = None

        if checkpoint_path is not None:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.config = checkpoint["config"]
            self.model = SimpleModel(self.config).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config.optimizer.learning_rate
            )
            self.scheduler = self._build_scheduler()
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
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
            self.scheduler = self._build_scheduler()

    def _build_scheduler(self):
        scheduler_config = self.config.optimizer.scheduler
        if scheduler_config is None:
            return None

        scheduler_type = scheduler_config.type
        if scheduler_type == "cosine_annealing":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.t_max,
                eta_min=scheduler_config.eta_min,
            )

        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def checkpoint(self, epoch, is_best=False):
        if not os.path.exists(self.config.train.checkpoint_dir):
            os.makedirs(self.config.train.checkpoint_dir, exist_ok=True)

        checkpoint_data = {
            "epoch": epoch,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "total_training_time": self.total_training_time,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        if self.scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save regular checkpoint
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
            with torch.no_grad():
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
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch} completed in {epoch_duration:.2f} seconds. Total training time: {self.total_training_time:.2f} seconds. Learning rate: {current_lr:.4e}"
            )

            # --- LOGGING & CHECKPOINTING ---
            self.experiment.log_metrics(
                {
                    "train/loss": avg_train_loss,
                    "val/loss": avg_val_loss,
                    "optimizer/learning_rate": current_lr,
                    "time/epoch_duration": epoch_duration,
                    "time/total_training_time_hours": self.total_training_time / 3600,
                },
                epoch=epoch,
            )

            is_best = avg_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = avg_val_loss
                self.best_epoch = epoch

            self.checkpoint(epoch, is_best=is_best)
            if self.scheduler is not None:
                self.scheduler.step()

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
        self.experiment.end()


class SimpleModelSobolevTrainer:
    def __init__(
        self,
        config: omegaconf.DictConfig,
        experiment: Experiment,
        checkpoint_path: str | None = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.best_val_loss = float("inf")  # Track for saving the "best" model
        self.epoch = 0  # Track current epoch for checkpointing
        self.best_epoch = 0  # Track epoch of the best model
        self.total_training_time = 0.0  # Track total training time across epochs
        self.experiment = experiment
        self.scheduler = None

        if checkpoint_path is not None:
            checkpoint = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
            self.config = checkpoint["config"]
            self.model = SimpleModel(self.config).to(self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config.optimizer.learning_rate
            )
            self.scheduler = self._build_scheduler()
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
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
            self.scheduler = self._build_scheduler()

        self.sobolev_alpha = self.config.train.sobolev_alpha
        sobolev_vars = self.config.train.sobolev_vars
        input_vars = self.config.dataset.input_vars
        self.sobolev_input_indices = [input_vars.index(var) for var in sobolev_vars]

    def _build_scheduler(self):
        scheduler_config = self.config.optimizer.scheduler
        if scheduler_config is None:
            return None

        scheduler_type = scheduler_config.type
        if scheduler_type == "cosine_annealing":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.t_max,
                eta_min=scheduler_config.eta_min,
            )

        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    def checkpoint(self, epoch, is_best=False):
        if not os.path.exists(self.config.train.checkpoint_dir):
            os.makedirs(self.config.train.checkpoint_dir, exist_ok=True)

        checkpoint_data = {
            "epoch": epoch,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "total_training_time": self.total_training_time,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }
        if self.scheduler is not None:
            checkpoint_data["scheduler_state_dict"] = self.scheduler.state_dict()

        # Save regular checkpoint
        torch.save(
            checkpoint_data,
            f"{self.config.train.checkpoint_dir}/model_epoch_{epoch}.pt",
        )

        # Save "best" model separately
        if is_best:
            torch.save(
                checkpoint_data, f"{self.config.train.checkpoint_dir}/best_model.pt"
            )

    def _run_epoch(self, loader, train: bool):
        self.model.train(mode=train)
        total_loss_0 = 0
        total_loss_1 = 0

        for i, (x, y, kernels) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            kernels = kernels.to(self.device)
            x.requires_grad_(True)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            y_pred = self.model(x)
            grads = torch.autograd.grad(
                outputs=y_pred.sum(),
                inputs=x,
                create_graph=train,
            )[0][..., self.sobolev_input_indices]

            if grads.shape != kernels.shape:
                raise ValueError(
                    f"Gradient shape {tuple(grads.shape)} does not match kernel shape {tuple(kernels.shape)}."
                )

            loss_0 = F.mse_loss(y_pred, y)
            loss_1 = self.sobolev_alpha * F.mse_loss(grads, kernels)
            loss = loss_0 + loss_1

            if train:
                loss.backward()
                self.optimizer.step()

            total_loss_0 += loss_0.item()
            total_loss_1 += loss_1.item()

            label = "Batch" if train else "Val Batch"
            if i % 100 == 0:
                print(f"{label} {i}/{len(loader)} | Loss: {loss.item():.4e}")

        avg_loss_0 = total_loss_0 / len(loader)
        avg_loss_1 = total_loss_1 / len(loader)
        return avg_loss_0, avg_loss_1, avg_loss_0 + avg_loss_1

    def train_model(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ):
        for epoch in range(self.epoch, self.config.train.epochs):
            epoch_start_time = time.perf_counter()
            print(f"Epoch {epoch} started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

            avg_train_loss_0, avg_train_loss_1, avg_train_loss = self._run_epoch(
                train_loader, train=True
            )
            print(f"avg_train_loss_0: {avg_train_loss_0:.4e}")
            print(f"avg_train_loss_1: {avg_train_loss_1:.4e}")
            print(f"avg_train_loss: {avg_train_loss:.4e}")

            avg_val_loss_0, avg_val_loss_1, avg_val_loss = self._run_epoch(
                val_loader, train=False
            )
            print(f"avg_val_loss_0: {avg_val_loss_0:.4e}")
            print(f"avg_val_loss_1: {avg_val_loss_1:.4e}")
            print(f"avg_val_loss: {avg_val_loss:.4e}")

            epoch_duration = time.perf_counter() - epoch_start_time
            self.total_training_time += epoch_duration
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch} completed in {epoch_duration:.2f} seconds. Total training time: {self.total_training_time:.2f} seconds. Learning rate: {current_lr:.4e}"
            )

            self.experiment.log_metrics(
                {
                    "train/loss_0": avg_train_loss_0,
                    "train/loss_1": avg_train_loss_1,
                    "train/loss": avg_train_loss,
                    "val/loss_0": avg_val_loss_0,
                    "val/loss_1": avg_val_loss_1,
                    "val/loss": avg_val_loss,
                    "optimizer/learning_rate": current_lr,
                    "time/epoch_duration": epoch_duration,
                    "time/total_training_time_hours": self.total_training_time / 3600,
                },
                epoch=epoch,
            )

            is_best = avg_val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = avg_val_loss
                self.best_epoch = epoch

            self.checkpoint(epoch, is_best=is_best)
            if self.scheduler is not None:
                self.scheduler.step()

            if avg_val_loss < self.config.train.early_stopping_threshold:
                print(
                    f"Early stopping at epoch {epoch} with val loss {avg_val_loss:.4e}"
                )
                break

            if epoch - self.best_epoch >= self.config.train.max_epochs_without_improvement:
                print(
                    f"Early stopping at epoch {epoch} due to no improvement for {self.config.train.max_epochs_without_improvement} epochs"
                )
                break

        print(f"Training complete. Total time: {self.total_training_time/3600:.2f}h")
        self.experiment.end()
