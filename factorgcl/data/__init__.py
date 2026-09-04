"""Data pipeline: raw market panel -> daily cross-sectional samples."""

from .dataset import Batch, StockDailyDataset, build_dataloader, collate_days, split_by_date
from .preprocess import DailySample, MarketPanel, SampleBuilder
from .synthetic import MARKET_FIELDS, SyntheticConfig, generate_market_panel

__all__ = [
    "Batch",
    "DailySample",
    "MARKET_FIELDS",
    "MarketPanel",
    "SampleBuilder",
    "StockDailyDataset",
    "SyntheticConfig",
    "build_dataloader",
    "collate_days",
    "generate_market_panel",
    "split_by_date",
]
