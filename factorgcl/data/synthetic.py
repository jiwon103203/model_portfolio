"""A synthetic stock market with a known factor structure.

The experiments of the paper use a proprietary China A-share dataset (and a Hong
Kong one in the appendix) that cannot be redistributed.  This generator produces
a market whose returns are driven by

* ``K`` industry (prior) factors with autocorrelated factor returns,
* ``M_true`` hidden factors whose exposures are *not* implied by the industry
  membership -- exactly the situation FactorGCL is designed for,
* an autocorrelated idiosyncratic (alpha) component,
* white noise that sets the signal-to-noise ratio,

so that the whole pipeline can be run end to end and the model's ability to
recover hidden factors can actually be checked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .preprocess import MarketPanel

MARKET_FIELDS = ("high", "open", "low", "close", "vwap", "volume")


@dataclass
class SyntheticConfig:
    num_stocks: int = 300
    num_days: int = 1500
    num_industries: int = 20
    num_hidden_factors: int = 8
    start_date: str = "2014-01-01"

    #: autocorrelation of the factor returns -- this is what makes the future
    #: predictable from the past at all
    factor_autocorr: float = 0.6
    alpha_autocorr: float = 0.3

    industry_factor_vol: float = 0.010
    hidden_factor_vol: float = 0.008
    alpha_vol: float = 0.006
    noise_vol: float = 0.014

    #: probability that a stock is suspended (missing) on a given day
    suspension_rate: float = 0.01
    #: fraction of the universe that is not listed at the very start
    late_listing_rate: float = 0.15
    seed: int = 0


def _ar1(num_steps: int, num_series: int, rho: float, vol: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary AR(1) paths, ``(num_steps, num_series)``."""
    innovation_vol = vol * np.sqrt(max(1.0 - rho**2, 1e-8))
    out = np.empty((num_steps, num_series), dtype=np.float64)
    out[0] = rng.normal(0.0, vol, size=num_series)
    for t in range(1, num_steps):
        out[t] = rho * out[t - 1] + rng.normal(0.0, innovation_vol, size=num_series)
    return out


def generate_market_panel(config: Optional[SyntheticConfig] = None) -> MarketPanel:
    """Generate a :class:`MarketPanel` with a known latent factor structure."""
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    num_days, num_stocks = cfg.num_days, cfg.num_stocks

    # --- exposures ----------------------------------------------------------
    industry = rng.integers(0, cfg.num_industries, size=num_stocks)
    industry_exposure = np.zeros((num_stocks, cfg.num_industries))
    industry_exposure[np.arange(num_stocks), industry] = 1.0

    # hidden exposures are dense and unrelated to the industry membership
    hidden_exposure = rng.normal(0.0, 1.0, size=(num_stocks, cfg.num_hidden_factors))
    hidden_exposure /= np.linalg.norm(hidden_exposure, axis=1, keepdims=True) + 1e-8
    hidden_exposure *= rng.uniform(0.5, 1.5, size=(num_stocks, 1))

    # --- factor and idiosyncratic returns -----------------------------------
    industry_returns = _ar1(num_days, cfg.num_industries, cfg.factor_autocorr, cfg.industry_factor_vol, rng)
    hidden_returns = _ar1(num_days, cfg.num_hidden_factors, cfg.factor_autocorr, cfg.hidden_factor_vol, rng)
    alpha = _ar1(num_days, num_stocks, cfg.alpha_autocorr, cfg.alpha_vol, rng)
    noise = rng.normal(0.0, cfg.noise_vol, size=(num_days, num_stocks))

    returns = (
        industry_returns @ industry_exposure.T
        + hidden_returns @ hidden_exposure.T
        + alpha
        + noise
    )

    # --- prices -------------------------------------------------------------
    close = 10.0 * np.exp(np.cumsum(returns, axis=0)) * rng.uniform(0.5, 5.0, size=(1, num_stocks))
    previous_close = np.vstack([close[:1], close[:-1]])

    intraday = np.abs(rng.normal(0.0, 0.01, size=(num_days, num_stocks)))
    open_price = previous_close * (1.0 + rng.normal(0.0, 0.005, size=(num_days, num_stocks)))
    high = np.maximum(open_price, close) * (1.0 + intraday)
    low = np.minimum(open_price, close) * (1.0 - intraday)
    vwap = (open_price + close + high + low) / 4.0

    log_volume = _ar1(num_days, num_stocks, 0.8, 0.5, rng) + rng.uniform(11.0, 15.0, size=(1, num_stocks))
    volume = np.exp(log_volume)

    values = np.stack([high, open_price, low, close, vwap, volume], axis=2).astype(np.float32)

    # --- missing observations ----------------------------------------------
    suspended = rng.random((num_days, num_stocks)) < cfg.suspension_rate
    values[suspended] = np.nan

    num_late = int(cfg.late_listing_rate * num_stocks)
    if num_late > 0:
        late = rng.choice(num_stocks, size=num_late, replace=False)
        listing_day = rng.integers(1, max(2, num_days // 3), size=num_late)
        for stock, day in zip(late, listing_day):
            values[:day, stock] = np.nan

    dates = np.arange(
        np.datetime64(cfg.start_date, "D"),
        np.datetime64(cfg.start_date, "D") + np.timedelta64(num_days * 2, "D"),
    )
    # keep weekdays only, so the calendar looks like a real trading calendar
    weekday = dates.astype("datetime64[D]").astype(int) % 7
    dates = dates[(weekday != 2) & (weekday != 3)][:num_days]

    stock_ids = np.asarray([f"S{i:05d}" for i in range(num_stocks)])

    panel = MarketPanel(
        dates=dates,
        stock_ids=stock_ids,
        values=values,
        fields=MARKET_FIELDS,
        industry=industry,
        num_industries=cfg.num_industries,
    )
    # ground truth kept around so that experiments can score factor recovery
    panel.ground_truth = {  # type: ignore[attr-defined]
        "industry_exposure": industry_exposure,
        "hidden_exposure": hidden_exposure,
        "industry_returns": industry_returns,
        "hidden_returns": hidden_returns,
        "alpha": alpha,
        "returns": returns,
    }
    return panel
