"""Metric definitions of the supplementary material."""

import numpy as np

from factorgcl.metrics import daily_ic, evaluate_all_periods, evaluate_period, sign_f1


def test_ic_equals_the_pearson_correlation():
    rng = np.random.default_rng(0)
    pred, target = rng.normal(size=200), rng.normal(size=200)
    got = daily_ic([pred], [target])[0]
    assert abs(got - np.corrcoef(pred, target)[0, 1]) < 1e-10


def test_ic_of_a_perfect_prediction_is_one():
    target = np.arange(50, dtype=float)
    assert abs(daily_ic([target], [target])[0] - 1.0) < 1e-10
    assert abs(daily_ic([-target], [target])[0] + 1.0) < 1e-10


def test_icir_is_mean_ic_over_std_ic():
    rng = np.random.default_rng(1)
    predictions = [rng.normal(size=60) for _ in range(40)]
    targets = [rng.normal(size=60) for _ in range(40)]
    metrics = evaluate_period(predictions, targets)
    ics = daily_ic(predictions, targets)
    assert abs(metrics.ic - ics.mean()) < 1e-12
    assert abs(metrics.icir - ics.mean() / ics.std()) < 1e-12


def test_ic_is_computed_per_day_then_averaged():
    """A model that is right every day must score well even if the pooled
    correlation across days is weak."""
    day_one_target = np.array([1.0, 2.0, 3.0])
    day_two_target = np.array([101.0, 102.0, 103.0])
    predictions = [np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])]
    metrics = evaluate_period(predictions, [day_one_target, day_two_target])
    assert abs(metrics.ic - 1.0) < 1e-9


def test_sign_f1_matches_the_appendix_counts():
    target = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    pred = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    # TP = 2 (i=0,4), FP = 1 (i=2), FN = 1 (i=1)
    scores = sign_f1(pred, target)
    assert abs(scores["precision"] - 2 / 3) < 1e-12
    assert abs(scores["recall"] - 2 / 3) < 1e-12
    assert abs(scores["f1"] - 2 / 3) < 1e-12


def test_mse_matches_the_definition():
    predictions = [np.array([1.0, 2.0])]
    targets = [np.array([0.0, 0.0])]
    assert abs(evaluate_period(predictions, targets).mse - 2.5) < 1e-12


def test_rank_ic_is_invariant_to_a_monotone_transform():
    rng = np.random.default_rng(2)
    target = rng.normal(size=100)
    pred = rng.normal(size=100)
    baseline = evaluate_period([pred], [target]).rank_ic
    transformed = evaluate_period([np.exp(pred)], [target]).rank_ic
    assert abs(baseline - transformed) < 1e-9


def test_evaluate_all_periods_splits_the_horizons():
    rng = np.random.default_rng(3)
    predictions = [rng.normal(size=(40, 4)) for _ in range(10)]
    targets = [rng.normal(size=(40, 4)) for _ in range(10)]
    results = evaluate_all_periods(predictions, targets, (1, 5, 10, 20))
    assert set(results) == {1, 5, 10, 20}
    for period_index, period in enumerate((1, 5, 10, 20)):
        expected = evaluate_period(
            [p[:, period_index] for p in predictions], [t[:, period_index] for t in targets]
        )
        assert abs(results[period].ic - expected.ic) < 1e-12


def test_non_finite_days_are_skipped():
    good = np.array([1.0, 2.0, 3.0])
    constant = np.array([1.0, 1.0, 1.0])  # zero std -> undefined IC
    ics = daily_ic([good, constant], [good, good])
    assert ics.size == 2 and np.isnan(ics[1]) and abs(ics[0] - 1.0) < 1e-9
