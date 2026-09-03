"""TopK simulation and the CR / CER / AR / IR / RoMaD definitions."""

import numpy as np

from factorgcl.backtest import (
    BacktestConfig,
    TopKBacktester,
    daily_returns_from_panel,
    max_drawdown,
)
from factorgcl.data import SyntheticConfig, generate_market_panel


def build_inputs(num_days=60, num_stocks=20, holding_days=1):
    """A market where the first stock always earns 10% and the rest earn 0%."""
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-01") + num_days)
    daily_returns = np.zeros((num_days, num_stocks))
    daily_returns[:, 0] = 0.10
    scores = [np.arange(num_stocks)[::-1].astype(float) for _ in range(num_days - 1)]
    stock_index = [np.arange(num_stocks) for _ in range(num_days - 1)]
    return dates, scores, stock_index, daily_returns


def test_topk_picks_the_highest_scores():
    dates, scores, stock_index, daily_returns = build_inputs()
    config = BacktestConfig(top_k=1, holding_days=1, transaction_cost=0.0)
    result = TopKBacktester(config).run(
        dates[:-1], scores, stock_index, daily_returns, dates
    )
    # stock 0 has the top score every day and returns 10% per day
    np.testing.assert_allclose(result.strategy_returns, 0.10, rtol=1e-9)


def test_transaction_cost_is_charged_on_the_rebalanced_tranche():
    dates, scores, stock_index, daily_returns = build_inputs()
    free = TopKBacktester(BacktestConfig(top_k=1, holding_days=5, transaction_cost=0.0)).run(
        dates[:-1], scores, stock_index, daily_returns, dates
    )
    costly = TopKBacktester(BacktestConfig(top_k=1, holding_days=5, transaction_cost=0.005)).run(
        dates[:-1], scores, stock_index, daily_returns, dates
    )
    # one fifth of the book turns over per day, so the daily drag is 0.005 / 5
    np.testing.assert_allclose(free.strategy_returns - costly.strategy_returns, 0.001, rtol=1e-9)


def test_holding_period_splits_the_capital_into_tranches():
    """With dt tranches, only 1/dt of the book is invested in the newest pick."""
    num_days, num_stocks = 30, 10
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-01") + num_days)
    daily_returns = np.zeros((num_days, num_stocks))
    daily_returns[:, 0] = 0.10  # only the stock picked on day 0 earns anything
    # after day 0 the ranking flips, so later tranches hold zero-return stocks
    scores = [np.arange(num_stocks)[::-1].astype(float)] + [
        np.arange(num_stocks).astype(float) for _ in range(num_days - 2)
    ]
    stock_index = [np.arange(num_stocks) for _ in range(num_days - 1)]

    config = BacktestConfig(top_k=1, holding_days=4, transaction_cost=0.0)
    result = TopKBacktester(config).run(dates[:-1], scores, stock_index, daily_returns, dates)
    # the day-0 tranche is 1/4 of the book and lives for 4 days
    np.testing.assert_allclose(result.strategy_returns[:4], 0.10 / 4, rtol=1e-9)
    np.testing.assert_allclose(result.strategy_returns[4:], 0.0, atol=1e-12)


def test_metric_definitions():
    dates, scores, stock_index, daily_returns = build_inputs()
    config = BacktestConfig(top_k=1, holding_days=1, transaction_cost=0.0)
    result = TopKBacktester(config).run(dates[:-1], scores, stock_index, daily_returns, dates)

    excess = result.strategy_returns - result.benchmark_returns
    num_days = result.strategy_returns.size

    np.testing.assert_allclose(result.cumulative_return, np.cumsum(result.strategy_returns))
    np.testing.assert_allclose(result.cumulative_excess_return, np.cumsum(excess))
    assert abs(result.annualized_return - excess.sum() / num_days * 252) < 1e-12
    if excess.std() > 1e-12:
        assert abs(result.information_ratio - excess.mean() / excess.std() * np.sqrt(252)) < 1e-9
    else:
        # a constant excess return has no risk to divide by
        assert result.information_ratio == 0.0


def test_max_drawdown():
    curve = np.array([0.0, 0.1, 0.3, 0.05, 0.2, 0.5])
    assert abs(max_drawdown(curve) - 0.25) < 1e-12
    assert max_drawdown(np.array([0.0, 1.0, 2.0])) == 0.0


def test_romad_is_annualised_return_over_max_drawdown():
    rng = np.random.default_rng(0)
    num_days, num_stocks = 120, 15
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-01") + num_days)
    daily_returns = rng.normal(0.0, 0.02, size=(num_days, num_stocks))
    scores = [rng.normal(size=num_stocks) for _ in range(num_days - 1)]
    stock_index = [np.arange(num_stocks) for _ in range(num_days - 1)]

    result = TopKBacktester(BacktestConfig(top_k=5, holding_days=10)).run(
        dates[:-1], scores, stock_index, daily_returns, dates
    )
    drawdown = max_drawdown(result.cumulative_excess_return)
    if drawdown > 1e-12:
        assert abs(result.romad - result.annualized_return / drawdown) < 1e-9


def test_benchmark_is_the_equally_weighted_universe():
    num_days, num_stocks = 20, 8
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-01") + num_days)
    daily_returns = np.tile(np.linspace(-0.02, 0.02, num_stocks), (num_days, 1))
    scores = [np.arange(num_stocks).astype(float) for _ in range(num_days - 1)]
    stock_index = [np.arange(num_stocks) for _ in range(num_days - 1)]

    result = TopKBacktester(BacktestConfig(top_k=3, holding_days=1, transaction_cost=0.0)).run(
        dates[:-1], scores, stock_index, daily_returns, dates
    )
    np.testing.assert_allclose(result.benchmark_returns, daily_returns[0].mean(), atol=1e-12)


def test_daily_returns_from_panel():
    panel = generate_market_panel(SyntheticConfig(num_stocks=10, num_days=40, num_industries=2))
    returns = daily_returns_from_panel(panel, "vwap")
    vwap = panel.values[:, :, panel.field("vwap")].astype(np.float64)
    expected = (vwap[1] - vwap[0]) / vwap[0]
    finite = np.isfinite(expected)
    np.testing.assert_allclose(returns[0][finite], expected[finite], rtol=1e-6)
    assert np.all(np.isnan(returns[-1]))


def test_universe_restriction():
    num_days, num_stocks = 20, 10
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-01") + num_days)
    daily_returns = np.zeros((num_days, num_stocks))
    daily_returns[:, 0] = 1.0
    daily_returns[:, 5] = 0.5
    scores = [np.arange(num_stocks)[::-1].astype(float) for _ in range(num_days - 1)]
    stock_index = [np.arange(num_stocks) for _ in range(num_days - 1)]
    members = {d: np.arange(5, num_stocks) for d in dates}

    result = TopKBacktester(BacktestConfig(top_k=1, holding_days=1, transaction_cost=0.0)).run(
        dates[:-1], scores, stock_index, daily_returns, dates, universe=members
    )
    # stock 0 is excluded, so the best available pick is stock 5
    np.testing.assert_allclose(result.strategy_returns, 0.5, rtol=1e-9)
