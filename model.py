import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from comet_ml import Experiment
import omegaconf


class SimpleModel(nn.Module):
    def __init__(self, config):
        super(SimpleModel, self).__init__()
        self.config = config
        self.input_dim = config.model.input_dim
        self.hidden_dim = config.model.hidden_dim
        self.model = nn.Sequential(
            nn.Linear(config.model.input_dim, config.model.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.model.hidden_dim, 1),
        )

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
        scheduler_config = getattr(self.config.optimizer, "scheduler", None)
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

        self.sobolev_alpha = config.train.sobolev_alpha

    def _build_scheduler(self):
        scheduler_config = getattr(self.config.optimizer, "scheduler", None)
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
            train_loss_0 = 0
            train_loss_1 = 0

            for i, (x, y, alb_kern) in enumerate(train_loader):
                x.requires_grad_(True)
                x, y = x.to(self.device), y.to(self.device)
                alb_kern = alb_kern.to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                y_pred = self.model(x)

                # compute \partial y_pred / \partial x at the input point x
                g_pred = torch.autograd.grad(
                    outputs=y_pred.sum(),
                    inputs=x,
                    create_graph=True,
                )[0]

                # \partial y_pred / \partial x_0 at x, assume x_0 = albedo dim (fal)
                g_pred_alb = g_pred[..., 0]  # (..., 1)

                loss_0 = F.mse_loss(y_pred, y)
                loss_1 = self.sobolev_alpha * F.mse_loss(g_pred_alb, alb_kern)
                loss = loss_0 + loss_1
                loss.backward()
                self.optimizer.step()
                train_loss_0 += loss_0.item()
                train_loss_1 += loss_1.item()
                if i % 100 == 0:
                    print(f"Batch {i}/{len(train_loader)} | Loss: {loss.item():.4e}")

            avg_train_loss_0 = train_loss_0 / len(train_loader)
            avg_train_loss_1 = train_loss_1 / len(train_loader)
            avg_train_loss = avg_train_loss_0 + avg_train_loss_1
            print(f"avg_train_loss_0: {avg_train_loss_0:.4e}")
            print(f"avg_train_loss_1: {avg_train_loss_1:.4e}")
            print(f"avg_train_loss: {avg_train_loss:.4e}")

            # --- VALIDATION PHASE ---
            self.model.eval()
            val_loss_0 = 0
            val_loss_1 = 0
            for i, (x_val, y_val, alb_kern) in enumerate(val_loader):
                x_val.requires_grad_(True)
                x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                alb_kern = alb_kern.to(self.device)
                val_pred = self.model(x_val)
                g_val = torch.autograd.grad(
                    outputs=val_pred.sum(),
                    inputs=x_val,
                    create_graph=False,
                )[0]
                g_val_alb = g_val[..., 0]

                loss_0 = F.mse_loss(val_pred, y_val)
                loss_1 = self.sobolev_alpha * F.mse_loss(g_val_alb, alb_kern)
                loss = loss_0 + loss_1

                val_loss_0 += loss_0.item()
                val_loss_1 += loss_1.item()
                if i % 100 == 0:
                    print(
                        f"Val Batch {i}/{len(val_loader)} | Val Loss: {loss.item():.4e}"
                    )

            avg_val_loss_0 = val_loss_0 / len(val_loader)
            avg_val_loss_1 = val_loss_1 / len(val_loader)
            avg_val_loss = avg_val_loss_0 + avg_val_loss_1
            print(f"avg_val_loss_0: {avg_val_loss_0:.4e}")
            print(f"avg_val_loss_1: {avg_val_loss_1:.4e}")
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
