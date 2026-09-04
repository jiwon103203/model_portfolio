"""Preprocessing rules of the appendix."""

import numpy as np
import pytest

from factorgcl.config import DataConfig
from factorgcl.data import SyntheticConfig, generate_market_panel
from factorgcl.data.dataset import StockDailyDataset, collate_days, split_by_date
from factorgcl.data.preprocess import MarketPanel, SampleBuilder


@pytest.fixture(scope="module")
def panel():
    return generate_market_panel(
        SyntheticConfig(num_stocks=80, num_days=320, num_industries=6, seed=0)
    )


@pytest.fixture(scope="module")
def config():
    return DataConfig(num_prior_factors=6)


def test_sample_shapes(panel, config):
    builder = SampleBuilder(panel, config)
    sample = builder.build(int(builder.valid_date_indices()[5]))
    num_stocks = len(sample)
    assert sample.x.shape == (num_stocks, 60, 6)
    assert sample.future_x.shape == (num_stocks, 20, 6)
    assert sample.prior_exposure.shape == (num_stocks, 6)
    assert sample.label.shape == (num_stocks, 4)


def test_prices_are_divided_by_the_current_close(panel, config):
    """"for the price dimension data, we divide by the current closing price"."""
    builder = SampleBuilder(panel, config)
    date_index = int(builder.valid_date_indices()[10])
    sample = builder.build(date_index)

    close_column = config.price_fields.index("close")
    # the last row of the window is day t itself, so close_t / close_t == 1
    np.testing.assert_allclose(sample.x[:, -1, close_column], 1.0, rtol=1e-5)


def test_volumes_are_divided_by_the_average_volume(panel, config):
    """"for the trading volume dimension data, we divide by the average trading volume"."""
    builder = SampleBuilder(panel, config)
    sample = builder.build(int(builder.valid_date_indices()[10]))
    volume_column = len(config.price_fields)
    # a stock with no missing volume in its window averages exactly 1
    means = np.nanmean(sample.x[:, :, volume_column], axis=1)
    assert abs(np.median(means) - 1.0) < 0.05


def test_labels_are_cross_sectionally_standardised(panel, config):
    builder = SampleBuilder(panel, config)
    date_index = int(builder.valid_date_indices()[20])
    standardised, raw = builder._standardised_labels(date_index)
    finite = np.isfinite(standardised[:, 0])
    assert abs(standardised[finite, 0].mean()) < 1e-6
    assert abs(standardised[finite, 0].std() - 1.0) < 1e-6


def test_label_formula_matches_the_appendix(panel, config):
    """y~ = (price_{t+dt+1} - price_{t+1}) / price_{t+1} on the vwap."""
    builder = SampleBuilder(panel, config)
    date_index = int(builder.valid_date_indices()[15])
    raw = builder._raw_labels(date_index)
    vwap = panel.values[:, :, panel.field("vwap")].astype(np.float64)

    for k, period in enumerate(config.forward_periods):
        expected = (vwap[date_index + period + 1] - vwap[date_index + 1]) / vwap[date_index + 1]
        finite = np.isfinite(expected) & np.isfinite(raw[:, k])
        np.testing.assert_allclose(raw[finite, k], expected[finite], rtol=1e-6)


def test_no_nan_or_inf_survives_preprocessing(panel, config):
    builder = SampleBuilder(panel, config)
    for date_index in builder.valid_date_indices()[:20]:
        sample = builder.build(int(date_index))
        assert np.isfinite(sample.x).all()
        assert np.isfinite(sample.future_x).all()
        assert np.isfinite(sample.label).all()


def test_features_are_clipped(panel):
    config = DataConfig(num_prior_factors=6, feature_clip=1.5)
    builder = SampleBuilder(panel, config)
    sample = builder.build(int(builder.valid_date_indices()[0]))
    assert sample.x.max() <= 1.5 + 1e-6 and sample.x.min() >= -1.5 - 1e-6


def test_stocks_with_too_many_missing_values_are_dropped(config):
    """A stock that is suspended for most of the window must not enter the sample."""
    dates = np.arange(np.datetime64("2020-01-01"), np.datetime64("2020-01-01") + 200)
    rng = np.random.default_rng(0)
    # give every stock its own price path, otherwise the cross-sectional
    # standardisation of the labels would divide by a zero dispersion
    prices = np.exp(np.cumsum(rng.normal(0.0, 0.02, size=(200, 30)), axis=0)) * 10.0
    values = np.repeat(prices[:, :, None], 6, axis=2).astype(np.float32)
    values[:, :, 5] = 1000.0
    values[:, 0, :] = np.nan  # stock 0 never trades

    panel = MarketPanel(
        dates=dates,
        stock_ids=np.asarray([f"S{i}" for i in range(30)]),
        values=values,
        fields=("high", "open", "low", "close", "vwap", "volume"),
        industry=np.zeros(30, dtype=np.int64),
        num_industries=6,
    )
    builder = SampleBuilder(panel, config)
    sample = builder.build(int(builder.valid_date_indices()[0]))
    assert sample is not None
    assert 0 not in sample.stock_index


def test_prior_exposure_is_the_industry_one_hot(panel, config):
    builder = SampleBuilder(panel, config)
    date_index = int(builder.valid_date_indices()[3])
    sample = builder.build(date_index)
    industry = panel.industry_at(date_index)[sample.stock_index]
    np.testing.assert_array_equal(sample.prior_exposure.sum(axis=1), np.ones(len(sample)))
    np.testing.assert_array_equal(sample.prior_exposure.argmax(axis=1), industry)


def test_splits_are_in_strict_temporal_order(panel, config):
    boundaries = [("2014-01-01", "2014-08-31"), ("2014-09-01", "2014-11-30"), ("2014-12-01", "2015-12-31")]
    train, valid, test = split_by_date(panel, config, boundaries)
    assert len(train) > 0 and len(valid) > 0 and len(test) > 0
    assert train.dates.max() < valid.dates.min() < test.dates.min()


def test_purging_drops_the_overlapping_tail(panel, config):
    """A training sample's label reaches forward, so its horizon overlaps the
    next window in calendar time; ``purge_days`` removes those days."""
    boundaries = [("2014-01-01", "2014-08-31"), ("2014-09-01", "2014-11-30"), ("2014-12-01", "2015-12-31")]
    horizon = max(config.future_seq_len, max(config.forward_periods) + 1)

    train, _, _ = split_by_date(panel, config, boundaries)
    purged_train, _, _ = split_by_date(panel, config, boundaries, purge_days=horizon)

    assert len(purged_train) == len(train) - horizon
    np.testing.assert_array_equal(purged_train.dates, train.dates[:-horizon])


def test_collate_pads_and_masks(panel, config):
    dataset = StockDailyDataset(panel, config)
    batch = collate_days([dataset[0], dataset[1], dataset[2]])
    assert batch.x.shape[0] == 3
    for b, index in enumerate(range(3)):
        num_stocks = len(dataset[index])
        assert float(batch.node_mask[b].sum()) == num_stocks
        assert bool((batch.node_mask[b, num_stocks:] == 0).all())
        np.testing.assert_allclose(batch.x[b, :num_stocks].numpy(), dataset[index].x, rtol=1e-6)


def test_usable_days_leave_room_for_labels_and_future_window(panel, config):
    builder = SampleBuilder(panel, config)
    usable = builder.valid_date_indices()
    assert usable[0] == config.seq_len - 1
    horizon = max(max(config.forward_periods) + 1, config.future_seq_len)
    assert usable[-1] + horizon <= panel.num_dates - 1


def test_panel_round_trip(tmp_path, panel):
    path = tmp_path / "panel.npz"
    panel.save(path)
    restored = MarketPanel.load(path)
    np.testing.assert_array_equal(restored.dates, panel.dates)
    np.testing.assert_array_equal(np.nan_to_num(restored.values), np.nan_to_num(panel.values))
    assert restored.num_industries == panel.num_industries
