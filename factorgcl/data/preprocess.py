"""Market panel container and the preprocessing of the appendix.

    "We first look back at the past T = 60 days of data to obtain a sequence of
    60 x 6, then we preprocess the data by dimensionless processing.
    Specifically, for the price dimension data, we divide by the current closing
    price, to obtain the relative price, and for the trading volume dimension
    data, we divide by the average trading volume, to obtain the relative
    trading volume.  The construction of future sequence data is similar.  For
    the label, we use the future stock returns with multiple periods
    (dt = 1, 5, 10, 20), and obtain the label after cross-sectional
    standardization.

        y~_t = (price_{t+dt+1} - price_{t+1}) / price_{t+1}
        y_t  = (y~_t - mu_t) / sigma_t

    Finally, we drop samples with too many missing values, fill missing values
    with 0, and clip extreme values in the dataset."
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..config import DataConfig


@dataclass
class DailySample:
    """One cross-section, i.e. one trading day of the stock market."""

    date: np.datetime64
    #: positions of the surviving stocks inside ``MarketPanel.stock_ids``
    stock_index: np.ndarray
    #: historical sequences ``x``, ``(N, T, D)``
    x: np.ndarray
    #: future sequences ``x'``, ``(N, T', D)``
    future_x: np.ndarray
    #: prior factor exposures ``beta``, ``(N, K)``
    prior_exposure: np.ndarray
    #: cross-sectionally standardised multi-period labels, ``(N, L)``
    label: np.ndarray
    #: raw (non standardised) forward returns, ``(N, L)``; used by the backtest
    raw_return: np.ndarray

    def __len__(self) -> int:
        return int(self.stock_index.shape[0])


class MarketPanel:
    """Day-level price/volume data of a stock universe, stored as a dense cube.

    Args:
        dates: sorted ``datetime64[D]`` array of trading days, ``(T_total,)``.
        stock_ids: identifiers of the universe, ``(S,)``.
        values: ``(T_total, S, D)`` cube of raw market fields.  ``NaN`` marks a
            missing observation (suspended or not yet listed stock).
        fields: names of the ``D`` fields, in the order of the last axis.
        industry: prior factor membership.  Either static ``(S,)`` integer codes
            or time varying ``(T_total, S)`` codes; ``-1`` means "unknown".
        num_industries: number of prior factors ``K``.
    """

    def __init__(
        self,
        dates: np.ndarray,
        stock_ids: np.ndarray,
        values: np.ndarray,
        fields: Sequence[str],
        industry: np.ndarray,
        num_industries: Optional[int] = None,
    ) -> None:
        self.dates = np.asarray(dates)
        self.stock_ids = np.asarray(stock_ids)
        self.values = np.asarray(values, dtype=np.float32)
        self.fields = list(fields)
        self.industry = np.asarray(industry, dtype=np.int64)

        if self.values.shape != (self.dates.shape[0], self.stock_ids.shape[0], len(self.fields)):
            raise ValueError(
                f"values shape {self.values.shape} does not match "
                f"({self.dates.shape[0]}, {self.stock_ids.shape[0]}, {len(self.fields)})"
            )
        if self.industry.ndim not in (1, 2):
            raise ValueError("industry must be a (S,) or (T_total, S) integer array")

        self.num_industries = (
            int(num_industries)
            if num_industries is not None
            else int(self.industry.max()) + 1
        )
        self._field_index = {name: i for i, name in enumerate(self.fields)}

    # ------------------------------------------------------------------
    def field(self, name: str) -> int:
        if name not in self._field_index:
            raise KeyError(f"unknown market field {name!r}; available: {self.fields}")
        return self._field_index[name]

    def industry_at(self, date_index: int) -> np.ndarray:
        """Industry codes of every stock on a given day, ``(S,)``."""
        return self.industry if self.industry.ndim == 1 else self.industry[date_index]

    @property
    def num_dates(self) -> int:
        return int(self.dates.shape[0])

    @property
    def num_stocks(self) -> int:
        return int(self.stock_ids.shape[0])

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            dates=self.dates,
            stock_ids=self.stock_ids,
            values=self.values,
            fields=np.asarray(self.fields),
            industry=self.industry,
            num_industries=np.asarray(self.num_industries),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MarketPanel":
        payload = np.load(path, allow_pickle=False)
        return cls(
            dates=payload["dates"],
            stock_ids=payload["stock_ids"],
            values=payload["values"],
            fields=[str(f) for f in payload["fields"]],
            industry=payload["industry"],
            num_industries=int(payload["num_industries"]),
        )

    # ------------------------------------------------------------------
    @classmethod
    def from_long_dataframe(
        cls,
        frame,
        fields: Sequence[str],
        date_column: str = "date",
        stock_column: str = "stock_id",
        industry_column: str = "industry",
        num_industries: Optional[int] = None,
    ) -> "MarketPanel":
        """Build a panel from a long ``(date, stock_id, fields..., industry)`` table.

        Any ``(date, stock)`` pair absent from the table becomes ``NaN``, i.e. a
        missing observation.
        """
        import pandas as pd  # imported lazily: the core package only needs numpy

        frame = pd.DataFrame(frame)
        dates = np.sort(frame[date_column].unique())
        stock_ids = np.sort(frame[stock_column].unique())
        date_pos = {d: i for i, d in enumerate(dates)}
        stock_pos = {s: i for i, s in enumerate(stock_ids)}

        values = np.full((len(dates), len(stock_ids), len(fields)), np.nan, dtype=np.float32)
        row = frame[date_column].map(date_pos).to_numpy()
        col = frame[stock_column].map(stock_pos).to_numpy()
        for k, name in enumerate(fields):
            values[row, col, k] = frame[name].to_numpy(dtype=np.float32)

        industry = np.full((len(dates), len(stock_ids)), -1, dtype=np.int64)
        if industry_column in frame.columns:
            industry[row, col] = frame[industry_column].to_numpy(dtype=np.int64)
            # carry the last known industry forward, then backward
            for t in range(1, len(dates)):
                gap = industry[t] < 0
                industry[t, gap] = industry[t - 1, gap]
            for t in range(len(dates) - 2, -1, -1):
                gap = industry[t] < 0
                industry[t, gap] = industry[t + 1, gap]
            if bool((industry == industry[0]).all()):
                industry = industry[0]
        else:
            industry = np.zeros(len(stock_ids), dtype=np.int64)

        return cls(
            dates=dates,
            stock_ids=stock_ids,
            values=values,
            fields=list(fields),
            industry=industry,
            num_industries=num_industries,
        )


@dataclass
class _Validity:
    """Intermediate result of the per-day validity check."""

    stock_index: np.ndarray
    anchor_close: np.ndarray
    volume_scale: np.ndarray
    labels: np.ndarray
    raw_returns: np.ndarray
    industry: np.ndarray


class SampleBuilder:
    """Turns a :class:`MarketPanel` into per-day :class:`DailySample` objects.

    Samples are built lazily so that a universe of thousands of stocks over ten
    years never has to be materialised as sequences.
    """

    def __init__(self, panel: MarketPanel, config: DataConfig) -> None:
        self.panel = panel
        self.config = config

        self.price_index = np.asarray([panel.field(f) for f in config.price_fields], dtype=np.int64)
        self.volume_index = np.asarray([panel.field(f) for f in config.volume_fields], dtype=np.int64)
        self.feature_index = np.concatenate([self.price_index, self.volume_index])
        self.close_index = panel.field("close")
        self.label_price_index = panel.field(config.label_price_field)
        self.num_prior_factors = max(config.num_prior_factors, panel.num_industries)

        self._label_cache: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    @property
    def max_forward(self) -> int:
        """Furthest future index a sample needs: labels and the future window."""
        label_horizon = max(self.config.forward_periods) + 1
        return max(label_horizon, self.config.future_seq_len)

    def valid_date_indices(self) -> np.ndarray:
        """Trading days for which a full sample can be constructed."""
        first = self.config.seq_len - 1
        last = self.panel.num_dates - 1 - self.max_forward
        if last < first:
            return np.empty(0, dtype=np.int64)
        return np.arange(first, last + 1, dtype=np.int64)

    # ------------------------------------------------------------------
    def _raw_labels(self, date_index: int) -> np.ndarray:
        """``y~ = (P_{t+dt+1} - P_{t+1}) / P_{t+1}`` for every horizon, ``(S, L)``."""
        prices = self.panel.values[:, :, self.label_price_index]
        base = prices[date_index + 1]
        out = np.full((self.panel.num_stocks, self.config.num_periods), np.nan, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            for k, period in enumerate(self.config.forward_periods):
                future = prices[date_index + period + 1]
                out[:, k] = (future - base) / base
        out[~np.isfinite(out)] = np.nan
        return out

    def _standardised_labels(self, date_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Cross-sectionally standardised labels and the raw returns they come from."""
        if date_index in self._label_cache:
            return self._label_cache[date_index]

        raw = self._raw_labels(date_index)
        standardised = np.full_like(raw, np.nan)
        for k in range(raw.shape[1]):
            column = raw[:, k]
            finite = np.isfinite(column)
            if finite.sum() < 2:
                continue
            mean = column[finite].mean()
            std = column[finite].std()
            if std < 1e-12:
                continue
            standardised[finite, k] = (column[finite] - mean) / std
        np.clip(standardised, -self.config.label_clip, self.config.label_clip, out=standardised)
        self._label_cache[date_index] = (standardised, raw)
        return standardised, raw

    # ------------------------------------------------------------------
    def _dimensionless(
        self,
        window: np.ndarray,
        anchor_close: np.ndarray,
        volume_scale: np.ndarray,
    ) -> np.ndarray:
        """Divide prices by the anchor close and volumes by their mean.

        Args:
            window: ``(S, T, D_raw)`` slice of the panel.
            anchor_close: ``(S,)`` closing price used as the price unit.
            volume_scale: ``(S, n_volume)`` average trading volume per field.
        """
        prices = window[:, :, self.price_index]
        volumes = window[:, :, self.volume_index]
        with np.errstate(invalid="ignore", divide="ignore"):
            prices = prices / anchor_close[:, None, None]
            volumes = volumes / volume_scale[:, None, :]
        features = np.concatenate([prices, volumes], axis=2)
        features[~np.isfinite(features)] = np.nan
        return features

    def _validity(self, date_index: int) -> Optional[_Validity]:
        """Decide which stocks survive on a day, without building any sequence.

        Kept separate from :meth:`build` so that scanning a decade of a
        thousands-of-stocks universe for usable days stays cheap.
        """
        cfg = self.config
        start = date_index - cfg.seq_len + 1
        if start < 0 or date_index + self.max_forward >= self.panel.num_dates:
            return None

        labels, raw_returns = self._standardised_labels(date_index)

        anchor_close = self.panel.values[date_index, :, self.close_index].astype(np.float64)
        anchor_close = np.where(np.isfinite(anchor_close) & (anchor_close > 0), anchor_close, np.nan)

        history = self.panel.values[start : date_index + 1]  # (T, S, D_raw)
        with warnings.catch_warnings():
            # an entirely suspended stock gives an all-NaN slice; the NaN it
            # produces is exactly what the validity filter below looks for
            warnings.simplefilter("ignore", RuntimeWarning)
            volume_scale = np.nanmean(
                history[:, :, self.volume_index].astype(np.float64), axis=0
            )  # (S, n_volume)
        volume_scale = np.where(np.isfinite(volume_scale) & (volume_scale > 0), volume_scale, np.nan)

        # "drop samples with too many missing values"
        missing_ratio = (~np.isfinite(history[:, :, self.feature_index])).mean(axis=(0, 2))
        industry = self.panel.industry_at(date_index)
        valid = (
            (missing_ratio <= cfg.max_missing_ratio)
            & np.isfinite(anchor_close)
            & np.isfinite(volume_scale).all(axis=1)
            & np.isfinite(labels).all(axis=1)
            & (industry >= 0)
        )
        if valid.sum() < cfg.min_stocks_per_day:
            return None

        return _Validity(
            stock_index=np.flatnonzero(valid),
            anchor_close=anchor_close,
            volume_scale=volume_scale,
            labels=labels,
            raw_returns=raw_returns,
            industry=industry,
        )

    def is_usable(self, date_index: int) -> bool:
        """Whether a full sample can be built for this day."""
        return self._validity(date_index) is not None

    def build(self, date_index: int) -> Optional[DailySample]:
        """Build the sample of one trading day, or ``None`` if it is unusable."""
        cfg = self.config
        validity = self._validity(date_index)
        if validity is None:
            return None

        start = date_index - cfg.seq_len + 1
        stock_index = validity.stock_index
        anchor_close = validity.anchor_close
        volume_scale = validity.volume_scale
        labels, raw_returns = validity.labels, validity.raw_returns
        industry = validity.industry

        # slice the surviving stocks first, so the transpose only copies what is used
        history = np.transpose(
            self.panel.values[start : date_index + 1][:, stock_index, :], (1, 0, 2)
        ).astype(np.float64)  # (N, T, D_raw)
        future = np.transpose(
            self.panel.values[date_index + 1 : date_index + 1 + cfg.future_seq_len][
                :, stock_index, :
            ],
            (1, 0, 2),
        ).astype(np.float64)  # (N, T', D_raw)

        x = self._dimensionless(history, anchor_close[stock_index], volume_scale[stock_index])

        if cfg.future_anchor == "current":
            future_anchor = anchor_close[stock_index]
        elif cfg.future_anchor == "window":
            future_close = future[:, -1, self.close_index]
            future_anchor = np.where(
                np.isfinite(future_close) & (future_close > 0), future_close, anchor_close[stock_index]
            )
        else:
            raise ValueError(f"unknown future_anchor {cfg.future_anchor!r}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            future_volume_scale = np.nanmean(future[:, :, self.volume_index], axis=1)
        future_volume_scale = np.where(
            np.isfinite(future_volume_scale) & (future_volume_scale > 0),
            future_volume_scale,
            volume_scale[stock_index],
        )
        future_x = self._dimensionless(future, future_anchor, future_volume_scale)

        # --- clip extreme values, then fill the remaining gaps with 0 --------
        for array in (x, future_x):
            np.clip(array, -cfg.feature_clip, cfg.feature_clip, out=array)
            np.nan_to_num(array, copy=False, nan=cfg.fill_value, posinf=cfg.fill_value, neginf=cfg.fill_value)

        prior_exposure = np.zeros((stock_index.shape[0], self.num_prior_factors), dtype=np.float32)
        prior_exposure[np.arange(stock_index.shape[0]), industry[stock_index]] = 1.0

        return DailySample(
            date=self.panel.dates[date_index],
            stock_index=stock_index,
            x=x.astype(np.float32),
            future_x=future_x.astype(np.float32),
            prior_exposure=prior_exposure,
            label=labels[stock_index].astype(np.float32),
            raw_return=raw_returns[stock_index].astype(np.float32),
        )

    def build_all(self, date_indices: Optional[Sequence[int]] = None) -> List[DailySample]:
        """Eagerly build every usable sample (only for small universes)."""
        indices = self.valid_date_indices() if date_indices is None else np.asarray(date_indices)
        samples = [self.build(int(i)) for i in indices]
        return [s for s in samples if s is not None]
