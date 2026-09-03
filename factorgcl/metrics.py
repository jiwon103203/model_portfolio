"""Evaluation metrics defined in the supplementary material.

    IC_t  = 1/N * (y_hat_t - mean(y_hat_t))^T (y_t - mean(y_t)) / (std(y_hat_t) std(y_t))
    IC    = mean(IC_t)
    ICIR  = mean(IC_t) / std(IC_t)
    MSE   = 1/N sum_i (y_i - y_hat_i)^2
    F1    = 2 * Precision * Recall / (Precision + Recall)

with the sign-based confusion counts of the appendix::

    TP = #{y_i > 0 and y_hat_i > 0}
    FP = #{y_i < 0 and y_hat_i > 0}
    FN = #{y_i > 0 and y_hat_i < 0}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np


@dataclass
class PeriodMetrics:
    """Metrics for a single forward prediction period."""

    ic: float
    icir: float
    rank_ic: float
    rank_icir: float
    mse: float
    f1: float
    precision: float
    recall: float
    num_days: int

    def as_dict(self) -> Dict[str, float]:
        return {
            "IC": self.ic,
            "ICIR": self.icir,
            "RankIC": self.rank_ic,
            "RankICIR": self.rank_icir,
            "MSE": self.mse,
            "F1": self.f1,
            "Precision": self.precision,
            "Recall": self.recall,
            "num_days": float(self.num_days),
        }


def _pearson(pred: np.ndarray, target: np.ndarray) -> float:
    """Cross-sectional correlation, exactly the IC_t formula of the appendix."""
    if pred.size < 2:
        return np.nan
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    # population standard deviations, matching 1/N * <.,.> / (std * std)
    denominator = pred.std() * target.std()
    if not np.isfinite(denominator) or denominator < 1e-12:
        return np.nan
    return float(np.dot(pred_centered, target_centered) / pred.size / denominator)


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks, so that ties do not bias the rank correlation."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    ranks[order] = np.arange(values.shape[0], dtype=np.float64)
    # average the ranks of tied values
    sorted_values = values[order]
    start = 0
    for i in range(1, values.shape[0] + 1):
        if i == values.shape[0] or sorted_values[i] != sorted_values[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return ranks


def daily_ic(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    rank: bool = False,
) -> np.ndarray:
    """IC per trading day."""
    values = []
    for pred, target in zip(predictions, targets):
        pred = np.asarray(pred, dtype=np.float64).ravel()
        target = np.asarray(target, dtype=np.float64).ravel()
        valid = np.isfinite(pred) & np.isfinite(target)
        if valid.sum() < 2:
            continue
        pred, target = pred[valid], target[valid]
        if rank:
            pred, target = _rank(pred), _rank(target)
        values.append(_pearson(pred, target))
    return np.asarray(values, dtype=np.float64)


def sign_f1(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Precision / recall / F1 from the sign agreement rules of the appendix."""
    true_positive = float(np.sum((target > 0) & (pred > 0)))
    false_positive = float(np.sum((target < 0) & (pred > 0)))
    false_negative = float(np.sum((target > 0) & (pred < 0)))
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_period(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
) -> PeriodMetrics:
    """All metrics for one forward period, given one array per trading day."""
    if len(predictions) == 0:
        return PeriodMetrics(
            ic=np.nan,
            icir=np.nan,
            rank_ic=np.nan,
            rank_icir=np.nan,
            mse=np.nan,
            f1=np.nan,
            precision=np.nan,
            recall=np.nan,
            num_days=0,
        )

    ics = daily_ic(predictions, targets, rank=False)
    rank_ics = daily_ic(predictions, targets, rank=True)

    flat_pred = np.concatenate([np.asarray(p, dtype=np.float64).ravel() for p in predictions])
    flat_target = np.concatenate([np.asarray(t, dtype=np.float64).ravel() for t in targets])
    valid = np.isfinite(flat_pred) & np.isfinite(flat_target)
    flat_pred, flat_target = flat_pred[valid], flat_target[valid]

    classification = sign_f1(flat_pred, flat_target)

    def _mean_and_std(values: np.ndarray) -> tuple:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.nan, np.nan
        return float(finite.mean()), float(finite.std())

    ic_mean, ic_std = _mean_and_std(ics)
    rank_ic_mean, rank_ic_std = _mean_and_std(rank_ics)

    return PeriodMetrics(
        ic=ic_mean,
        icir=ic_mean / ic_std if np.isfinite(ic_std) and ic_std > 0 else np.nan,
        rank_ic=rank_ic_mean,
        rank_icir=rank_ic_mean / rank_ic_std
        if np.isfinite(rank_ic_std) and rank_ic_std > 0
        else np.nan,
        mse=float(np.mean((flat_pred - flat_target) ** 2)) if flat_pred.size else np.nan,
        f1=classification["f1"],
        precision=classification["precision"],
        recall=classification["recall"],
        num_days=int(ics.size),
    )


def evaluate_all_periods(
    predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
    forward_periods: Sequence[int],
) -> Dict[int, PeriodMetrics]:
    """Evaluate every forward period.

    Args:
        predictions / targets: one ``(N_t, L)`` array per trading day.
        forward_periods: the ``L`` horizons, e.g. ``(1, 5, 10, 20)``.
    """
    results: Dict[int, PeriodMetrics] = {}
    for index, period in enumerate(forward_periods):
        period_pred = [np.asarray(p)[:, index] for p in predictions]
        period_target = [np.asarray(t)[:, index] for t in targets]
        results[period] = evaluate_period(period_pred, period_target)
    return results


def format_metrics_table(results: Dict[int, PeriodMetrics]) -> str:
    """Render a Table-1-style summary."""
    header = f"{'period':>8} {'IC':>9} {'ICIR':>9} {'RankIC':>9} {'MSE':>9} {'F1':>9}"
    lines = [header, "-" * len(header)]
    for period, metrics in sorted(results.items()):
        lines.append(
            f"{'dt=' + str(period):>8} {metrics.ic:9.4f} {metrics.icir:9.4f} "
            f"{metrics.rank_ic:9.4f} {metrics.mse:9.4f} {metrics.f1:9.4f}"
        )
    return "\n".join(lines)
