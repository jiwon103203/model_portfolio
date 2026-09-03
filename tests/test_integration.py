"""End-to-end checks: the pipeline runs and the model actually learns."""

import numpy as np
import pytest
import torch

from factorgcl import (
    ExperimentConfig,
    Trainer,
    backtest_predictions,
    build_dataloader,
    build_model,
    generate_market_panel,
    split_by_date,
)
from factorgcl.data import SyntheticConfig
from factorgcl.rolling import generate_rolling_windows, run_rolling_experiment


@pytest.fixture(scope="module")
def panel():
    return generate_market_panel(
        SyntheticConfig(num_stocks=60, num_days=1100, num_industries=6, num_hidden_factors=4, seed=0)
    )


def small_config(panel, epochs=3):
    config = ExperimentConfig()
    config.data.num_prior_factors = panel.num_industries
    config.model.hidden_dim = 16
    config.model.num_hidden_factors = 8
    config.train.max_epochs = epochs
    config.train.batch_days = 8
    config.train.device = "cpu"
    config.train.verbose = False
    config.backtest.top_k = 10
    return config.sync()


def test_training_reduces_the_loss_and_beats_a_random_model(panel):
    config = small_config(panel, epochs=6)
    train_set, valid_set, test_set = split_by_date(
        panel,
        config.data,
        [("2014-01-01", "2015-03-31"), ("2015-04-01", "2015-08-31"), ("2015-09-01", "2016-12-31")],
    )
    assert len(train_set) > 0 and len(valid_set) > 0 and len(test_set) > 0

    model = build_model(config.model, seed=0)
    trainer = Trainer(model, config.train, config.data)

    test_loader = build_dataloader(test_set, batch_days=config.train.batch_days)
    untrained_metrics, _ = trainer.evaluate(test_loader)

    trainer.fit(train_set, valid_set)
    trained_metrics, predictions = trainer.evaluate(test_loader)

    assert trainer.history[-1]["loss"] < trainer.history[0]["loss"]
    # the synthetic market has a real factor structure, so training must help
    assert trained_metrics[1].ic > untrained_metrics[1].ic
    assert trained_metrics[1].ic > 0.02

    for daily in predictions.predictions:
        assert np.isfinite(daily).all()


def test_contrastive_loss_decreases_during_training(panel):
    config = small_config(panel, epochs=5)
    train_set, valid_set, _ = split_by_date(
        panel, config.data, [("2014-01-01", "2015-06-30"), ("2015-07-01", "2015-12-31"), ("2016-01-01", "2016-01-31")]
    )
    trainer = Trainer(build_model(config.model, seed=0), config.train, config.data)
    trainer.fit(train_set, valid_set)
    assert trainer.history[-1]["contrastive"] < trainer.history[0]["contrastive"]


def test_disabling_the_contrastive_loss_zeroes_that_term(panel):
    config = small_config(panel, epochs=2)
    config.train.use_contrastive = False
    train_set, _, _ = split_by_date(
        panel, config.data, [("2014-01-01", "2015-06-30"), ("2015-07-01", "2015-07-31"), ("2015-08-01", "2015-08-31")]
    )
    trainer = Trainer(build_model(config.model, seed=0), config.train, config.data)
    stats = trainer.fit(train_set, None)
    assert all(record["contrastive"] == 0.0 for record in trainer.history)
    # without a validation set there is nothing to early-stop on: the last
    # epoch is kept and no IC is reported
    assert stats["best_epoch"] == config.train.max_epochs - 1
    assert np.isnan(stats["best_valid_ic"])


def test_backtest_runs_on_model_predictions(panel):
    config = small_config(panel, epochs=2)
    config.backtest.holding_days = 10
    train_set, valid_set, test_set = split_by_date(
        panel, config.data, [("2014-01-01", "2015-12-31"), ("2016-01-01", "2016-06-30"), ("2016-07-01", "2017-12-31")]
    )
    trainer = Trainer(build_model(config.model, seed=0), config.train, config.data)
    trainer.fit(train_set, valid_set)
    _, predictions = trainer.evaluate(build_dataloader(test_set, batch_days=8))

    result = backtest_predictions(
        predictions, panel, config.backtest, forward_periods=config.data.forward_periods
    )
    assert result.dates.size > 0
    assert result.strategy_returns.shape == result.benchmark_returns.shape
    assert np.isfinite(result.cumulative_excess_return).all()
    assert np.isfinite(result.annualized_return)


def test_early_stopping_restores_the_best_checkpoint(panel, tmp_path):
    config = small_config(panel, epochs=4)
    config.train.early_stopping_patience = 1
    train_set, valid_set, _ = split_by_date(
        panel, config.data, [("2014-01-01", "2015-06-30"), ("2015-07-01", "2015-12-31"), ("2016-01-01", "2016-01-31")]
    )
    trainer = Trainer(build_model(config.model, seed=0), config.train, config.data)
    stats = trainer.fit(train_set, valid_set, checkpoint_path=tmp_path / "best.pt")

    saved = torch.load(tmp_path / "best.pt", map_location="cpu")
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value.cpu(), saved[key].cpu())
    assert stats["best_epoch"] >= 0


def test_rolling_experiment(panel):
    config = small_config(panel, epochs=2)
    config.rolling.train_years = 1
    config.rolling.valid_years = 1
    config.rolling.test_years = 1
    config.rolling.step_years = 1
    config.rolling.test_start = "2016-01-01"
    config.rolling.test_end = "2017-06-30"

    windows = generate_rolling_windows(panel.dates[0], panel.dates[-1], config.rolling)
    assert len(windows) >= 1
    for window in windows:
        assert window.train[1] < window.valid[0] < window.test[0]

    result = run_rolling_experiment(panel, config)
    assert result.predictions.dates.size > 0
    assert set(result.pooled_metrics) == set(config.data.forward_periods)
    # the pooled test days must be strictly increasing across windows
    assert np.all(np.diff(result.predictions.dates.astype("datetime64[D]").astype(int)) > 0)


def test_seed_makes_training_reproducible(panel):
    config = small_config(panel, epochs=2)
    train_set, valid_set, test_set = split_by_date(
        panel, config.data, [("2014-01-01", "2015-12-31"), ("2016-01-01", "2016-06-30"), ("2016-07-01", "2016-12-31")]
    )

    def run():
        trainer = Trainer(build_model(config.model, seed=0), config.train, config.data)
        trainer.fit(train_set, valid_set)
        metrics, _ = trainer.evaluate(build_dataloader(test_set, batch_days=8))
        return metrics[1].ic

    assert abs(run() - run()) < 1e-9


def test_ablation_variants_all_train(panel):
    """The four variants of Table 2, including "-wo Alpha&CL" and "-wo CL"."""
    variants = [
        ({"use_prior": False}, True),
        ({"use_hidden": False}, True),
        ({"use_alpha": False}, False),  # "-wo Alpha&CL"
        ({}, False),                    # "-wo CL"
    ]
    for overrides, use_contrastive in variants:
        config = small_config(panel, epochs=2)
        config.train.use_contrastive = use_contrastive
        for key, value in overrides.items():
            setattr(config.model, key, value)
        train_set, valid_set, _ = split_by_date(
            panel,
            config.data,
            [("2014-01-01", "2015-06-30"), ("2015-07-01", "2015-12-31"), ("2016-01-01", "2016-01-31")],
        )
        trainer = Trainer(build_model(config.model, seed=0), config.train, config.data)
        trainer.fit(train_set, valid_set)
        assert np.isfinite(trainer.history[-1]["loss"])


def test_contrastive_without_alpha_module_warns(panel):
    """The alpha embeddings are identically zero without the alpha module, so
    contrasting them carries no signal; the paper removes both together."""
    config = small_config(panel, epochs=1)
    config.model.use_alpha = False
    config.train.use_contrastive = True
    with pytest.warns(RuntimeWarning, match="use_alpha=False"):
        Trainer(build_model(config.model, seed=0), config.train, config.data)
