"""Rolling train / validation / test protocol.

    "We follow the temporal order to split the dataset into training set,
    validation set and test set, where the time length is 5 years : 1 year :
    2 years, and adopt a rolling method for training and testing.  The overall
    test period is from 01/01/2020 to 06/30/2023."
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import ExperimentConfig, RollingConfig
from .data.dataset import build_dataloader, split_by_date
from .data.preprocess import MarketPanel
from .engine import Predictions, Trainer, build_model
from .metrics import PeriodMetrics, evaluate_all_periods


@dataclass
class RollingWindow:
    """One (train, valid, test) split of the timeline."""

    train: Tuple[str, str]
    valid: Tuple[str, str]
    test: Tuple[str, str]

    def __str__(self) -> str:  # pragma: no cover - reporting helper
        return (
            f"train {self.train[0]}~{self.train[1]} | "
            f"valid {self.valid[0]}~{self.valid[1]} | "
            f"test {self.test[0]}~{self.test[1]}"
        )


def _shift_years(day: date, years: int) -> date:
    try:
        return day.replace(year=day.year + years)
    except ValueError:  # 29 February
        return day.replace(year=day.year + years, day=28)


def _to_date(value) -> date:
    return np.datetime64(value, "D").astype("O")


def _fmt(day: date) -> str:
    return day.isoformat()


def _day() -> timedelta:
    return timedelta(days=1)


def generate_rolling_windows(
    panel_start,
    panel_end,
    config: Optional[RollingConfig] = None,
) -> List[RollingWindow]:
    """Build the rolling windows covering ``[test_start, test_end]``.

    Each window keeps the paper's 5:1:2 year proportion; consecutive windows
    advance by ``step_years`` (the test length by default), so the test periods
    tile the overall test range without overlapping.
    """
    cfg = config or RollingConfig()
    panel_start_date, panel_end_date = _to_date(panel_start), _to_date(panel_end)

    test_start = _to_date(cfg.test_start) if cfg.test_start else _shift_years(
        panel_start_date, cfg.train_years + cfg.valid_years
    )
    test_end = _to_date(cfg.test_end) if cfg.test_end else panel_end_date
    test_end = min(test_end, panel_end_date)

    windows: List[RollingWindow] = []
    current_test_start = test_start
    while current_test_start <= test_end:
        current_test_end = min(
            _shift_years(current_test_start, cfg.test_years) - _day(), test_end
        )
        valid_start = _shift_years(current_test_start, -cfg.valid_years)
        train_start = _shift_years(valid_start, -cfg.train_years)
        if train_start < panel_start_date:
            # not enough history for a full training window: use what is there
            train_start = panel_start_date
        if valid_start <= train_start:
            break

        windows.append(
            RollingWindow(
                train=(_fmt(train_start), _fmt(valid_start - _day())),
                valid=(_fmt(valid_start), _fmt(current_test_start - _day())),
                test=(_fmt(current_test_start), _fmt(current_test_end)),
            )
        )
        current_test_start = _shift_years(current_test_start, cfg.step_years)
    return windows


@dataclass
class RollingResult:
    """Test-set predictions of every window, plus the pooled metrics."""

    windows: List[RollingWindow]
    per_window_metrics: List[Dict[int, PeriodMetrics]]
    pooled_metrics: Dict[int, PeriodMetrics]
    predictions: Predictions
    fit_stats: List[Dict[str, float]]


def run_rolling_experiment(
    panel: MarketPanel,
    config: ExperimentConfig,
    windows: Optional[Sequence[RollingWindow]] = None,
    cache_samples: bool = False,
) -> RollingResult:
    """Train one model per rolling window and concatenate the test predictions."""
    config = config.sync()
    if windows is None:
        windows = generate_rolling_windows(panel.dates[0], panel.dates[-1], config.rolling)
    windows = list(windows)
    if not windows:
        raise ValueError("no rolling window fits inside the panel's date range")

    all_dates: List[np.datetime64] = []
    pooled = Predictions(dates=np.asarray([]))
    per_window_metrics: List[Dict[int, PeriodMetrics]] = []
    fit_stats: List[Dict[str, float]] = []

    for window in windows:
        if config.train.verbose:
            print(f"\n=== rolling window: {window} ===", flush=True)
        train_set, valid_set, test_set = split_by_date(
            panel,
            config.data,
            [window.train, window.valid, window.test],
            cache=cache_samples,
        )
        if len(train_set) == 0 or len(test_set) == 0:
            if config.train.verbose:
                print("  skipped: empty train or test split", flush=True)
            continue

        model = build_model(config.model, seed=config.train.seed)
        trainer = Trainer(model, config.train, config.data)
        fit_stats.append(trainer.fit(train_set, valid_set))

        test_loader = build_dataloader(test_set, batch_days=config.train.batch_days, shuffle=False)
        metrics, predictions = trainer.evaluate(test_loader)
        per_window_metrics.append(metrics)

        pooled.predictions.extend(predictions.predictions)
        pooled.labels.extend(predictions.labels)
        pooled.raw_returns.extend(predictions.raw_returns)
        pooled.stock_index.extend(predictions.stock_index)
        pooled.hidden_exposure.extend(predictions.hidden_exposure)
        all_dates.extend(list(predictions.dates))

    if not all_dates:
        raise ValueError(
            "every rolling window produced an empty test split; check that the "
            "panel's date range covers RollingConfig.test_start / test_end"
        )

    pooled.dates = np.asarray(all_dates)
    pooled_metrics = evaluate_all_periods(
        pooled.predictions, pooled.labels, config.data.forward_periods
    )
    return RollingResult(
        windows=windows,
        per_window_metrics=per_window_metrics,
        pooled_metrics=pooled_metrics,
        predictions=pooled,
        fit_stats=fit_stats,
    )
