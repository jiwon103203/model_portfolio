"""FactorGCL - a hypergraph-based factor model with temporal residual
contrastive learning for stock returns prediction.

Reference implementation of:

    Yitong Duan, Weiran Wang, Jian Li.
    "FactorGCL: A Hypergraph-Based Factor Model with Temporal Residual
    Contrastive Learning for Stock Returns Prediction."
    AAAI-25, pp. 173-180.
"""

from .backtest import BacktestResult, TopKBacktester, backtest_predictions, daily_returns_from_panel
from .config import (
    BacktestConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RollingConfig,
    TrainConfig,
)
from .data import (
    Batch,
    DailySample,
    MarketPanel,
    SampleBuilder,
    StockDailyDataset,
    SyntheticConfig,
    build_dataloader,
    collate_days,
    generate_market_panel,
    split_by_date,
)
from .engine import Predictions, Trainer, build_model, set_seed
from .losses import FactorGCLLoss, info_nce_loss, multi_period_mse
from .metrics import PeriodMetrics, evaluate_all_periods, evaluate_period, format_metrics_table
from .models import FactorGCL, FactorGCLOutput, HypergraphConv
from .rolling import RollingResult, RollingWindow, generate_rolling_windows, run_rolling_experiment

__version__ = "0.1.0"

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Batch",
    "DailySample",
    "DataConfig",
    "ExperimentConfig",
    "FactorGCL",
    "FactorGCLLoss",
    "FactorGCLOutput",
    "HypergraphConv",
    "MarketPanel",
    "ModelConfig",
    "PeriodMetrics",
    "Predictions",
    "RollingConfig",
    "RollingResult",
    "RollingWindow",
    "SampleBuilder",
    "StockDailyDataset",
    "SyntheticConfig",
    "TopKBacktester",
    "TrainConfig",
    "Trainer",
    "backtest_predictions",
    "build_dataloader",
    "build_model",
    "collate_days",
    "daily_returns_from_panel",
    "evaluate_all_periods",
    "evaluate_period",
    "format_metrics_table",
    "generate_market_panel",
    "generate_rolling_windows",
    "info_nce_loss",
    "multi_period_mse",
    "run_rolling_experiment",
    "set_seed",
    "split_by_date",
]
