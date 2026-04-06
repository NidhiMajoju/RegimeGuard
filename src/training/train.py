import math
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from src.models.regime_guard import RegimeGuard  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - placeholder for future model location
    RegimeGuard = None  # noqa: N816

from src.losses.composite_loss import CompositeRegimeLoss


class RegimeGuardTrainer:
    """
    Minimal training wrapper with early stopping and checkpointing.

    Expects model(X) -> (pred, uncertainty).
    """

    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        # Store references
        self.model = model
        self.config = config

        # Device management
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Optimizer + scheduler (ReduceLROnPlateau halves LR when val loss stalls)
        lr = config.get("learning_rate", 1e-3)
        weight_decay = config.get("weight_decay", 0.0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            verbose=True,
        )

        # Early stopping helpers
        self.best_val_loss = math.inf
        self.patience_counter = 0
        self.early_stopping_patience = config.get("early_stopping_patience", 10)

    def _move_batch(
        self, batch: Tuple[torch.Tensor, ...]
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Support (X, y) or (X, y, regime_labels).
        """
        if len(batch) == 2:
            X, y = batch
            regimes = None
        elif len(batch) == 3:
            X, y, regimes = batch
        else:
            raise ValueError("Expected batch of (X, y) or (X, y, regimes)")

        X = X.to(self.device)
        y = y.to(self.device)
        regimes = regimes.to(self.device) if regimes is not None else None
        return X, y, regimes

    def train_epoch(self, train_loader: DataLoader, loss_fn: CompositeRegimeLoss) -> float:
        self.model.train()
        running_loss = 0.0
        total_batches = 0

        for batch in train_loader:
            X, y, regimes = self._move_batch(batch)

            self.optimizer.zero_grad()

            pred, uncertainty = self.model(X)

            # Composite loss combines MSE, reliability, optional IRM
            loss_dict = loss_fn(
                pred=pred,
                target=y,
                uncertainty=uncertainty,
                regime_indicator=regimes,
                model_params=self.model.parameters(),
            )

            loss = loss_dict["total"]
            loss.backward()
            # Keep gradients stable on spiky series
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item()
            total_batches += 1

        return running_loss / max(total_batches, 1)

    @torch.no_grad()
    def validate(
        self, val_loader: DataLoader, loss_fn: CompositeRegimeLoss
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        self.model.eval()

        running_loss = 0.0
        total_batches = 0
        preds = []
        targets = []

        for batch in val_loader:
            X, y, regimes = self._move_batch(batch)

            pred, uncertainty = self.model(X)

            # Re-use same loss for consistency with train
            loss_dict = loss_fn(
                pred=pred,
                target=y,
                uncertainty=uncertainty,
                regime_indicator=regimes,
                model_params=self.model.parameters(),
            )

            running_loss += loss_dict["total"].item()
            total_batches += 1

            preds.append(pred.detach().cpu())
            targets.append(y.detach().cpu())

        avg_loss = running_loss / max(total_batches, 1)
        preds_tensor = torch.cat(preds, dim=0) if preds else torch.tensor([])
        targets_tensor = torch.cat(targets, dim=0) if targets else torch.tensor([])

        return avg_loss, preds_tensor, targets_tensor

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: CompositeRegimeLoss,
        num_epochs: int,
    ) -> None:

        checkpoint_path = self.config.get("checkpoint_path", "checkpoints/regime_guard.pt")
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(train_loader, loss_fn)
            val_loss, _, _ = self.validate(val_loader, loss_fn)

            # Step scheduler on validation signal
            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(checkpoint_path)
                print(f"  ↳ New best model saved to {checkpoint_path}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print("Early stopping triggered.")
                    break

    def save_checkpoint(self, path: str) -> None:
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        # Keep optimizer state for exact LR/bias correction on resume
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "best_val_loss": self.best_val_loss,
            },
            ckpt_path,
        )

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.best_val_loss = ckpt.get("best_val_loss", math.inf)
        self.model.to(self.device)


if __name__ == "__main__":
    # Example wiring; swap in your data loaders and model implementation.
    config = {
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "early_stopping_patience": 10,
        "num_epochs": 50,
        "checkpoint_path": "checkpoints/regime_guard.pt",
    }

    if RegimeGuard is None:
        raise ImportError("Please implement RegimeGuard in src/models/regime_guard.py")

    model = RegimeGuard(config)
    loss_fn = CompositeRegimeLoss(w_mse=0.7, w_reliability=0.2, w_irm=0.1)

    # Plug in real loaders here
    train_loader: DataLoader
    val_loader: DataLoader
    raise SystemExit("Instantiate train_loader/val_loader, then call trainer.train(...)")

    # Example:
    # trainer = RegimeGuardTrainer(model, config)
    # trainer.train(train_loader, val_loader, loss_fn, config["num_epochs"])
