"""Torch dataset and collation for the daily cross-sections.

One training example is a whole trading day: the hypergraph of FactorGCL is
built over the cross-section, so stocks cannot be split across mini-batches.
When several days are batched together the cross-sections are zero-padded to the
largest one and a ``node_mask`` marks the real stocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from ..config import DataConfig
from .preprocess import DailySample, MarketPanel, SampleBuilder


@dataclass
class Batch:
    """A padded batch of trading days."""

    x: torch.Tensor                # (B, N, T, D)
    future_x: torch.Tensor         # (B, N, T', D)
    prior_exposure: torch.Tensor   # (B, N, K)
    label: torch.Tensor            # (B, N, L)
    raw_return: torch.Tensor       # (B, N, L)
    node_mask: torch.Tensor        # (B, N)
    stock_index: torch.Tensor      # (B, N), -1 where padded
    dates: np.ndarray              # (B,)

    def to(self, device: torch.device | str) -> "Batch":
        return Batch(
            x=self.x.to(device),
            future_x=self.future_x.to(device),
            prior_exposure=self.prior_exposure.to(device),
            label=self.label.to(device),
            raw_return=self.raw_return.to(device),
            node_mask=self.node_mask.to(device),
            stock_index=self.stock_index.to(device),
            dates=self.dates,
        )

    def __len__(self) -> int:
        return int(self.x.shape[0])


class StockDailyDataset(Dataset):
    """Sequence of daily cross-sections built from a :class:`MarketPanel`."""

    def __init__(
        self,
        panel: MarketPanel,
        config: DataConfig,
        date_indices: Optional[Sequence[int]] = None,
        cache: bool = False,
        builder: Optional[SampleBuilder] = None,
    ) -> None:
        self.panel = panel
        self.config = config
        self.builder = builder if builder is not None else SampleBuilder(panel, config)
        if date_indices is None:
            date_indices = self.builder.valid_date_indices()
        self.date_indices = np.asarray(date_indices, dtype=np.int64)
        self.cache = cache
        self._cache: Dict[int, DailySample] = {}
        self._drop_unusable()

    def _drop_unusable(self) -> None:
        """Remove days whose cross-section is too small to be a valid sample."""
        keep: List[int] = []
        for date_index in self.date_indices:
            date_index = int(date_index)
            if self.cache:
                sample = self.builder.build(date_index)
                if sample is None:
                    continue
                self._cache[date_index] = sample
            elif not self.builder.is_usable(date_index):
                continue
            keep.append(date_index)
        self.date_indices = np.asarray(keep, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.date_indices.shape[0])

    def __getitem__(self, index: int) -> DailySample:
        date_index = int(self.date_indices[index])
        if date_index in self._cache:
            return self._cache[date_index]
        sample = self.builder.build(date_index)
        if sample is None:  # pragma: no cover - filtered out by _drop_unusable
            raise IndexError(f"sample for date index {date_index} is unavailable")
        if self.cache:
            self._cache[date_index] = sample
        return sample

    @property
    def dates(self) -> np.ndarray:
        return self.panel.dates[self.date_indices]


def collate_days(samples: Sequence[DailySample]) -> Batch:
    """Pad a list of cross-sections into a single batch."""
    batch_size = len(samples)
    max_stocks = max(len(s) for s in samples)
    seq_len, num_features = samples[0].x.shape[1], samples[0].x.shape[2]
    future_len = samples[0].future_x.shape[1]
    num_factors = samples[0].prior_exposure.shape[1]
    num_periods = samples[0].label.shape[1]

    x = np.zeros((batch_size, max_stocks, seq_len, num_features), dtype=np.float32)
    future_x = np.zeros((batch_size, max_stocks, future_len, num_features), dtype=np.float32)
    prior = np.zeros((batch_size, max_stocks, num_factors), dtype=np.float32)
    label = np.zeros((batch_size, max_stocks, num_periods), dtype=np.float32)
    raw_return = np.zeros((batch_size, max_stocks, num_periods), dtype=np.float32)
    node_mask = np.zeros((batch_size, max_stocks), dtype=np.float32)
    stock_index = np.full((batch_size, max_stocks), -1, dtype=np.int64)

    for b, sample in enumerate(samples):
        n = len(sample)
        x[b, :n] = sample.x
        future_x[b, :n] = sample.future_x
        prior[b, :n] = sample.prior_exposure
        label[b, :n] = sample.label
        raw_return[b, :n] = sample.raw_return
        node_mask[b, :n] = 1.0
        stock_index[b, :n] = sample.stock_index

    return Batch(
        x=torch.from_numpy(x),
        future_x=torch.from_numpy(future_x),
        prior_exposure=torch.from_numpy(prior),
        label=torch.from_numpy(label),
        raw_return=torch.from_numpy(raw_return),
        node_mask=torch.from_numpy(node_mask),
        stock_index=torch.from_numpy(stock_index),
        dates=np.asarray([s.date for s in samples]),
    )


def build_dataloader(
    dataset: StockDailyDataset,
    batch_days: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_days,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_days,
        drop_last=False,
    )


def split_by_date(
    panel: MarketPanel,
    config: DataConfig,
    boundaries: Sequence[Tuple[str, str]],
    cache: bool = False,
    purge_days: int = 0,
) -> List[StockDailyDataset]:
    """Split the usable trading days into consecutive ``[start, end]`` windows.

    The split is on the *prediction* day ``t``, which is what the paper's
    "5 years : 1 year : 2 years" protocol describes.  Note that a sample also
    reaches ``max(T', dt_max + 1)`` days forward for its future window and its
    longest label, so the tail of one window overlaps the head of the next in
    calendar time even though no prediction day is shared.  The paper does not
    purge that overlap, so ``purge_days`` defaults to 0; set it to
    ``max(future_seq_len, max(forward_periods) + 1)`` to drop the affected days
    from the end of every window.

    Args:
        boundaries: one ``(start, end)`` pair per split, in chronological order.
        cache: keep the built samples in memory (only for small universes).
        purge_days: number of trailing prediction days to drop from each window.
    """
    builder = SampleBuilder(panel, config)
    usable = builder.valid_date_indices()
    usable_dates = panel.dates[usable]

    datasets: List[StockDailyDataset] = []
    for start, end in boundaries:
        start_date = np.datetime64(start, "D")
        end_date = np.datetime64(end, "D")
        selected = usable[(usable_dates >= start_date) & (usable_dates <= end_date)]
        if purge_days > 0 and selected.size > purge_days:
            selected = selected[:-purge_days]
        datasets.append(
            StockDailyDataset(panel, config, date_indices=selected, cache=cache, builder=builder)
        )
    return datasets
