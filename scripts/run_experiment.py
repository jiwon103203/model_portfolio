#!/usr/bin/env python3
"""Train FactorGCL, evaluate it, and run the TopK investment simulation.

    # single split
    python scripts/run_experiment.py --panel data/synthetic_panel.npz

    # the paper's rolling protocol (5y train : 1y valid : 2y test)
    python scripts/run_experiment.py --panel data/synthetic_panel.npz --rolling

    # ablations of Table 2
    python scripts/run_experiment.py --panel data/... --ablation wo_hidden
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

import numpy as np

from factorgcl import (
    ExperimentConfig,
    MarketPanel,
    Trainer,
    backtest_predictions,
    build_dataloader,
    build_model,
    format_metrics_table,
    generate_rolling_windows,
    run_rolling_experiment,
    split_by_date,
)

ABLATIONS = {
    "full": {},
    "wo_prior": {"use_prior": False},
    "wo_hidden": {"use_hidden": False},
    "wo_alpha": {"use_alpha": False},
    "wo_alpha_cl": {"use_alpha": False, "_no_cl": True},
    "wo_cl": {"_no_cl": True},
}


def build_config(args: argparse.Namespace, panel: MarketPanel) -> ExperimentConfig:
    config = ExperimentConfig()
    config.data.num_prior_factors = panel.num_industries
    config.model.num_hidden_factors = args.num_hidden_factors
    config.model.hidden_dim = args.hidden_dim
    config.train.max_epochs = args.epochs
    config.train.batch_days = args.batch_days
    config.train.seed = args.seed
    config.train.device = args.device
    config.backtest.top_k = args.top_k
    config.backtest.holding_days = args.holding_days
    config.rolling.train_years = args.rolling_train_years
    config.rolling.valid_years = args.rolling_valid_years
    config.rolling.test_years = args.rolling_test_years
    config.rolling.step_years = args.rolling_test_years
    if args.rolling and args.test_range is not None:
        config.rolling.test_start, config.rolling.test_end = args.test_range

    overrides = dict(ABLATIONS[args.ablation])
    if overrides.pop("_no_cl", False):
        config.train.use_contrastive = False
    for key, value in overrides.items():
        setattr(config.model, key, value)
    return config.sync()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, required=True, help="npz written by make_synthetic_data.py")
    parser.add_argument("--rolling", action="store_true", help="use the paper's rolling protocol")
    parser.add_argument("--rolling-train-years", type=int, default=5)
    parser.add_argument("--rolling-valid-years", type=int, default=1)
    parser.add_argument("--rolling-test-years", type=int, default=2)
    parser.add_argument("--ablation", choices=sorted(ABLATIONS), default="full")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-hidden-factors", type=int, default=32)
    parser.add_argument("--batch-days", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--holding-days", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--train-range", nargs=2, metavar=("START", "END"), default=None)
    parser.add_argument("--valid-range", nargs=2, metavar=("START", "END"), default=None)
    parser.add_argument(
        "--test-range",
        nargs=2,
        metavar=("START", "END"),
        default=None,
        help="single split: the test window. With --rolling: the overall test span "
        "the rolling windows must tile.",
    )
    parser.add_argument("--out", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    panel = MarketPanel.load(args.panel)
    config = build_config(args, panel)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.rolling:
        result = run_rolling_experiment(panel, config)
        metrics, predictions = result.pooled_metrics, result.predictions
        print("\nrolling windows:")
        for window in result.windows:
            print(f"  {window}")
    else:
        if args.train_range is None:
            windows = generate_rolling_windows(panel.dates[0], panel.dates[-1], config.rolling)
            window = windows[0]
            train_range, valid_range, test_range = window.train, window.valid, window.test
        else:
            train_range = tuple(args.train_range)
            valid_range = tuple(args.valid_range)
            test_range = tuple(args.test_range)
        print(f"train {train_range} | valid {valid_range} | test {test_range}")

        train_set, valid_set, test_set = split_by_date(
            panel, config.data, [train_range, valid_range, test_range]
        )
        print(f"days: train {len(train_set)} | valid {len(valid_set)} | test {len(test_set)}")

        model = build_model(config.model, seed=config.train.seed)
        trainer = Trainer(model, config.train, config.data)
        trainer.fit(train_set, valid_set, checkpoint_path=args.out / f"factorgcl_{args.ablation}.pt")

        test_loader = build_dataloader(test_set, batch_days=config.train.batch_days, shuffle=False)
        metrics, predictions = trainer.evaluate(test_loader)

    print("\n=== stock return prediction (test set) ===")
    print(format_metrics_table(metrics))

    result = backtest_predictions(
        predictions,
        panel,
        config=config.backtest,
        forward_periods=config.data.forward_periods,
    )
    print("\n=== TopK investment simulation ===")
    print(result)

    payload = {
        "config": config.to_dict(),
        "metrics": {str(k): v.as_dict() for k, v in metrics.items()},
        "backtest": result.summary(),
    }
    (args.out / f"results_{args.ablation}.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(
        args.out / f"backtest_{args.ablation}.npz",
        dates=result.dates,
        strategy_returns=result.strategy_returns,
        benchmark_returns=result.benchmark_returns,
        cumulative_excess_return=result.cumulative_excess_return,
    )
    print(f"\nwrote {args.out / f'results_{args.ablation}.json'}")


if __name__ == "__main__":
    main()
