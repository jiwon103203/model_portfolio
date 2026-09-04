"""Training / evaluation loop for FactorGCL.

Follows the "Implementation Details" of the supplementary material: Adam with a
learning rate of 1e-3, 100 epochs, early stopping after 20 epochs without
improvement, random seed 0.
"""

from __future__ import annotations

import copy
import random
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DataConfig, ModelConfig, TrainConfig
from .data.dataset import Batch, StockDailyDataset, build_dataloader
from .losses import FactorGCLLoss
from .metrics import PeriodMetrics, evaluate_all_periods
from .models.factorgcl import FactorGCL


def set_seed(seed: int) -> None:
    """Seed every source of randomness used by the training loop."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


@dataclass
class Predictions:
    """Per-day model outputs, kept in the shape the metrics and backtest expect."""

    dates: np.ndarray
    #: one ``(N_t, L)`` array per trading day
    predictions: List[np.ndarray] = field(default_factory=list)
    labels: List[np.ndarray] = field(default_factory=list)
    raw_returns: List[np.ndarray] = field(default_factory=list)
    stock_index: List[np.ndarray] = field(default_factory=list)
    #: hidden factor exposures ``beta_h`` per day, ``(N_t, M)``
    hidden_exposure: List[np.ndarray] = field(default_factory=list)


@dataclass
class EarlyStopping:
    """Stop after ``patience`` epochs without improvement of the monitored score."""

    patience: int = 20
    mode: str = "max"
    best_score: float = field(init=False)
    counter: int = field(init=False, default=0)
    best_epoch: int = field(init=False, default=-1)

    def __post_init__(self) -> None:
        self.best_score = -np.inf if self.mode == "max" else np.inf

    def step(self, score: float, epoch: int) -> Tuple[bool, bool]:
        """Returns ``(improved, should_stop)``."""
        if not np.isfinite(score):
            self.counter += 1
            return False, self.counter >= self.patience
        improved = score > self.best_score if self.mode == "max" else score < self.best_score
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        return improved, self.counter >= self.patience


class Trainer:
    """Trains a :class:`FactorGCL` model on daily cross-sections."""

    def __init__(
        self,
        model: FactorGCL,
        train_config: TrainConfig,
        data_config: DataConfig,
    ) -> None:
        self.config = train_config
        self.data_config = data_config
        self.device = resolve_device(train_config.device)
        self.model = model.to(self.device)
        if train_config.use_contrastive and not model.config.use_alpha:
            # Eq. (11) contrasts the alpha embeddings, which are identically zero
            # without the individual alpha module -- the paper's corresponding
            # ablation row is "-wo Alpha&CL", i.e. both are removed together.
            warnings.warn(
                "use_contrastive=True with use_alpha=False: the contrastive loss "
                "would be computed on zero alpha embeddings and carries no signal; "
                "disable it (the paper's '-wo Alpha&CL' variant).",
                RuntimeWarning,
                stacklevel=2,
            )
        self.criterion = FactorGCLLoss(
            gamma=train_config.gamma,
            temperature=train_config.temperature,
            use_contrastive=train_config.use_contrastive,
        )
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay
        )
        self.history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    def _forward(self, batch: Batch, with_future: bool) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        output = self.model(
            batch.x,
            batch.prior_exposure,
            future_x=batch.future_x if with_future else None,
            node_mask=batch.node_mask,
        )
        extras = {
            "hidden_exposure": output.hidden_exposure,
            "alpha_embedding": output.alpha_embedding,
        }
        if output.future_alpha_embedding is not None:
            extras["future_alpha_embedding"] = output.future_alpha_embedding
        return output.prediction, extras

    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        totals = {"loss": 0.0, "mse": 0.0, "contrastive": 0.0}
        num_batches = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            use_cl = self.config.use_contrastive
            prediction, extras = self._forward(batch, with_future=use_cl)

            anchor = positive = None
            if use_cl and "future_alpha_embedding" in extras:
                anchor = self.model.project(extras["alpha_embedding"])
                positive = self.model.project(extras["future_alpha_embedding"])

            loss = self.criterion(
                prediction,
                batch.label,
                anchor=anchor,
                positive=positive,
                node_mask=batch.node_mask,
            )
            loss.total.backward()
            if self.config.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            totals["loss"] += float(loss.total.detach())
            totals["mse"] += float(loss.mse.detach())
            totals["contrastive"] += float(loss.contrastive.detach())
            num_batches += 1

        return {k: v / max(num_batches, 1) for k, v in totals.items()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, loader: DataLoader) -> Predictions:
        """Run the model over a loader, unpadding every cross-section."""
        self.model.eval()
        dates: List[np.datetime64] = []
        result = Predictions(dates=np.asarray([]))

        for batch in loader:
            batch = batch.to(self.device)
            prediction, extras = self._forward(batch, with_future=False)
            prediction = prediction.cpu().numpy()
            hidden_exposure = extras["hidden_exposure"].cpu().numpy()
            mask = batch.node_mask.cpu().numpy().astype(bool)

            for b in range(mask.shape[0]):
                valid = mask[b]
                result.predictions.append(prediction[b][valid])
                result.labels.append(batch.label[b].cpu().numpy()[valid])
                result.raw_returns.append(batch.raw_return[b].cpu().numpy()[valid])
                result.stock_index.append(batch.stock_index[b].cpu().numpy()[valid])
                result.hidden_exposure.append(hidden_exposure[b][valid])
                dates.append(batch.dates[b])

        result.dates = np.asarray(dates)
        return result

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[Dict[int, PeriodMetrics], Predictions]:
        predictions = self.predict(loader)
        metrics = evaluate_all_periods(
            predictions.predictions, predictions.labels, self.data_config.forward_periods
        )
        return metrics, predictions

    @torch.no_grad()
    def validation_loss(self, loader: DataLoader) -> float:
        self.model.eval()
        total, count = 0.0, 0
        for batch in loader:
            batch = batch.to(self.device)
            prediction, _ = self._forward(batch, with_future=False)
            loss = self.criterion(prediction, batch.label, node_mask=batch.node_mask)
            total += float(loss.mse)
            count += 1
        return total / max(count, 1)

    # ------------------------------------------------------------------
    def fit(
        self,
        train_dataset: StockDailyDataset,
        valid_dataset: Optional[StockDailyDataset] = None,
        checkpoint_path: Optional[str | Path] = None,
    ) -> Dict[str, float]:
        """Train with early stopping on the validation IC of ``monitor_period``."""
        set_seed(self.config.seed)
        train_loader = build_dataloader(
            train_dataset,
            batch_days=self.config.batch_days,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
        )
        valid_loader = (
            build_dataloader(valid_dataset, batch_days=self.config.batch_days, shuffle=False)
            if valid_dataset is not None and len(valid_dataset) > 0
            else None
        )

        stopper = EarlyStopping(patience=self.config.early_stopping_patience, mode="max")
        best_state = copy.deepcopy(self.model.state_dict())
        last_epoch = -1

        for epoch in range(self.config.max_epochs):
            train_stats = self.train_epoch(train_loader)
            record = {"epoch": float(epoch), **train_stats}

            if valid_loader is None:
                # without a validation set there is nothing to early-stop on,
                # so the last epoch is kept
                best_state = copy.deepcopy(self.model.state_dict())
                if checkpoint_path is not None:
                    torch.save(best_state, checkpoint_path)
                last_epoch = epoch
                self.history.append(record)
                if self.config.verbose:
                    self._log(record)
                continue

            metrics, _ = self.evaluate(valid_loader)
            monitor = self.config.monitor_period
            if monitor not in metrics:
                monitor = list(metrics)[-1]
            score = metrics[monitor].ic
            record["valid_ic"] = float(score)
            record["valid_icir"] = float(metrics[monitor].icir)

            improved, should_stop = stopper.step(score, epoch)
            if improved:
                best_state = copy.deepcopy(self.model.state_dict())
                if checkpoint_path is not None:
                    torch.save(best_state, checkpoint_path)
            record["best_valid_ic"] = float(stopper.best_score)

            self.history.append(record)
            if self.config.verbose:
                self._log(record)
            if should_stop:
                if self.config.verbose:
                    print(
                        f"[early stopping] no improvement for "
                        f"{self.config.early_stopping_patience} epochs "
                        f"(best epoch {stopper.best_epoch}, IC {stopper.best_score:.4f})"
                    )
                break

        self.model.load_state_dict(best_state)
        if valid_loader is None:
            return {"best_epoch": float(last_epoch), "best_valid_ic": float("nan")}
        return {"best_epoch": float(stopper.best_epoch), "best_valid_ic": float(stopper.best_score)}

    @staticmethod
    def _log(record: Dict[str, float]) -> None:
        parts = [f"epoch {int(record['epoch']):3d}"]
        for key in ("loss", "mse", "contrastive", "valid_ic", "valid_icir"):
            if key in record:
                parts.append(f"{key} {record[key]:.4f}")
        print(" | ".join(parts), flush=True)


def build_model(model_config: ModelConfig, seed: int = 0) -> FactorGCL:
    """Instantiate FactorGCL with a fixed seed, as in the paper's setup."""
    set_seed(seed)
    return FactorGCL(model_config)
