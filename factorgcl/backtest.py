"""TopK investment simulation and its metrics.

    "This strategy involves investing in the TopK stocks with the highest
    predicted scores each trading day and selling them after holding for dt
    days [...] We use the equally weighted CSI300 and CSI500 portfolio as the
    benchmark, and set TopK = 30, dt = 10, and the transaction cost to 0.3%."

Because a position is held for ``dt`` days while a new one is opened every day,
the capital is split into ``dt`` equally sized tranches; one tranche is rolled
over per trading day.  Metric definitions follow the supplementary material::

    CR(t)   = sum_{i<=t} r(i)
    CER(t)  = sum_{i<=t} (r(i) - r_b(i))
    AR      = CER(T_test) / T * 252
    IR      = E[r - r_b] / std(r - r_b) * sqrt(252)
    RoMaD   = AR / MaxDrawdown
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import BacktestConfig
from .data.preprocess import MarketPanel


@dataclass
class BacktestResult:
    """Daily series and summary statistics of one simulation."""

    dates: np.ndarray
    strategy_returns: np.ndarray
    benchmark_returns: np.ndarray
    excess_returns: np.ndarray
    cumulative_return: np.ndarray
    cumulative_excess_return: np.ndarray
    annualized_return: float
    information_ratio: float
    max_drawdown: float
    romad: float
    turnover: float

    def summary(self) -> Dict[str, float]:
        return {
            "AR": self.annualized_return,
            "IR": self.information_ratio,
            "RoMaD": self.romad,
            "MaxDrawdown": self.max_drawdown,
            "CR": float(self.cumulative_return[-1]) if self.cumulative_return.size else 0.0,
            "CER": float(self.cumulative_excess_return[-1])
            if self.cumulative_excess_return.size
            else 0.0,
            "Turnover": self.turnover,
            "num_days": float(self.dates.shape[0]),
        }

    def __str__(self) -> str:  # pragma: no cover - reporting helper
        s = self.summary()
        return (
            f"AR {s['AR']:.4f} | IR {s['IR']:.4f} | RoMaD {s['RoMaD']:.4f} | "
            f"MaxDD {s['MaxDrawdown']:.4f} | CR {s['CR']:.4f} | CER {s['CER']:.4f}"
        )


def daily_returns_from_panel(panel: MarketPanel, price_field: str = "vwap") -> np.ndarray:
    """Simple daily returns ``(P_{t+1} - P_t) / P_t``, shape ``(T_total, S)``.

    Row ``t`` holds the return earned between the close of day ``t`` and day
    ``t + 1``; the last row is ``NaN``.
    """
    prices = panel.values[:, :, panel.field(price_field)].astype(np.float64)
    returns = np.full_like(prices, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        returns[:-1] = (prices[1:] - prices[:-1]) / prices[:-1]
    returns[~np.isfinite(returns)] = np.nan
    return returns


def max_drawdown(curve: np.ndarray) -> float:
    """Largest peak-to-trough loss of a cumulative (additive) return curve."""
    if curve.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(curve)
    return float(np.max(running_peak - curve))


@dataclass
class _Tranche:
    """One dated slice of capital: an equally weighted basket held for dt days."""

    stocks: np.ndarray
    remaining_days: int


class TopKBacktester:
    """Simulates the TopK strategy on a set of daily prediction scores."""

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()

    def run(
        self,
        dates: Sequence[np.datetime64],
        scores: Sequence[np.ndarray],
        stock_index: Sequence[np.ndarray],
        daily_returns: np.ndarray,
        panel_dates: np.ndarray,
        universe: Optional[Dict[np.datetime64, np.ndarray]] = None,
    ) -> BacktestResult:
        """Run the simulation.

        Args:
            dates: prediction days ``t``, in chronological order.
            scores: ``(N_t,)`` predicted score per stock for each prediction day.
            stock_index: ``(N_t,)`` positions of those stocks inside the panel.
            daily_returns: ``(T_total, S)`` matrix from :func:`daily_returns_from_panel`.
            panel_dates: the panel's full date axis, used to align the returns.
            universe: optional per-day investable subset (e.g. CSI300 members);
                defaults to all stocks with a prediction on that day.
        """
        cfg = self.config
        order = np.argsort(np.asarray(dates))
        dates = [np.asarray(dates)[i] for i in order]
        scores = [np.asarray(scores[i]).ravel() for i in order]
        stock_index = [np.asarray(stock_index[i]).ravel() for i in order]

        date_position = {d: i for i, d in enumerate(panel_dates)}
        cost_per_tranche = cfg.transaction_cost / max(cfg.holding_days, 1)

        active: deque[_Tranche] = deque()
        strategy: List[float] = []
        benchmark: List[float] = []
        used_dates: List[np.datetime64] = []
        traded_notional = 0.0

        for day, score, index in zip(dates, scores, stock_index):
            position = date_position.get(day)
            if position is None or position + 1 >= daily_returns.shape[0]:
                continue

            candidates = index
            candidate_scores = score
            if universe is not None:
                members = universe.get(day)
                if members is not None:
                    keep = np.isin(candidates, members)
                    candidates, candidate_scores = candidates[keep], candidate_scores[keep]
            if candidates.size == 0:
                continue

            # --- open the tranche of day t; it earns returns from t+1 onwards
            k = min(cfg.top_k, candidates.size)
            top = candidates[np.argsort(-candidate_scores, kind="stable")[:k]]
            active.append(_Tranche(stocks=top, remaining_days=cfg.holding_days))
            while len(active) > cfg.holding_days:
                active.popleft()
            traded_notional += 1.0 / max(cfg.holding_days, 1)

            # --- return earned over [t, t+1) by every active tranche
            row = daily_returns[position]
            tranche_returns = []
            for tranche in active:
                stock_returns = row[tranche.stocks]
                valid = np.isfinite(stock_returns)
                tranche_returns.append(float(stock_returns[valid].mean()) if valid.any() else 0.0)
            weight = 1.0 / max(cfg.holding_days, 1)
            portfolio_return = float(np.sum(tranche_returns)) * weight
            # the newly opened tranche pays its round-trip transaction cost
            portfolio_return -= cost_per_tranche

            universe_returns = row[candidates]
            valid_universe = np.isfinite(universe_returns)
            benchmark_return = (
                float(universe_returns[valid_universe].mean()) if valid_universe.any() else 0.0
            )

            strategy.append(portfolio_return)
            benchmark.append(benchmark_return)
            used_dates.append(day)

            for tranche in active:
                tranche.remaining_days -= 1
            while active and active[0].remaining_days <= 0:
                active.popleft()

        strategy_returns = np.asarray(strategy, dtype=np.float64)
        benchmark_returns = np.asarray(benchmark, dtype=np.float64)
        excess = strategy_returns - benchmark_returns

        cumulative_return = np.cumsum(strategy_returns)
        cumulative_excess = np.cumsum(excess)
        num_days = strategy_returns.shape[0]

        if num_days == 0:
            return BacktestResult(
                dates=np.asarray(used_dates),
                strategy_returns=strategy_returns,
                benchmark_returns=benchmark_returns,
                excess_returns=excess,
                cumulative_return=cumulative_return,
                cumulative_excess_return=cumulative_excess,
                annualized_return=0.0,
                information_ratio=0.0,
                max_drawdown=0.0,
                romad=0.0,
                turnover=0.0,
            )

        annualized_return = float(cumulative_excess[-1] / num_days * cfg.trading_days_per_year)
        excess_std = float(excess.std())
        information_ratio = (
            float(excess.mean() / excess_std * np.sqrt(cfg.trading_days_per_year))
            if excess_std > 1e-12
            else 0.0
        )
        drawdown = max_drawdown(cumulative_excess)
        romad = float(annualized_return / drawdown) if drawdown > 1e-12 else 0.0

        return BacktestResult(
            dates=np.asarray(used_dates),
            strategy_returns=strategy_returns,
            benchmark_returns=benchmark_returns,
            excess_returns=excess,
            cumulative_return=cumulative_return,
            cumulative_excess_return=cumulative_excess,
            annualized_return=annualized_return,
            information_ratio=information_ratio,
            max_drawdown=drawdown,
            romad=romad,
            turnover=float(traded_notional / num_days),
        )


def backtest_predictions(
    predictions,
    panel: MarketPanel,
    config: Optional[BacktestConfig] = None,
    period_index: Optional[int] = None,
    forward_periods: Sequence[int] = (1, 5, 10, 20),
    price_field: str = "vwap",
    universe: Optional[Dict[np.datetime64, np.ndarray]] = None,
) -> BacktestResult:
    """Convenience wrapper that backtests a :class:`~factorgcl.engine.Predictions`.

    The score used for stock selection is the predicted return of the horizon
    that matches the holding period (paper: ``dt = 10``).
    """
    cfg = config or BacktestConfig()
    if period_index is None:
        try:
            period_index = list(forward_periods).index(cfg.holding_days)
        except ValueError:
            period_index = len(forward_periods) - 1

    scores = [np.asarray(p)[:, period_index] for p in predictions.predictions]
    returns = daily_returns_from_panel(panel, price_field=price_field)
    return TopKBacktester(cfg).run(
        dates=predictions.dates,
        scores=scores,
        stock_index=predictions.stock_index,
        daily_returns=returns,
        panel_dates=panel.dates,
        universe=universe,
    )
