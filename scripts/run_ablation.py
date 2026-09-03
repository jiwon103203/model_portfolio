#!/usr/bin/env python3
"""Reproduce the ablation study of Table 2 on a given market panel.

Builds the four variants of the paper -- ``-wo Prior``, ``-wo Hidden``,
``-wo Alpha&CL``, ``-wo CL`` -- plus the full model, trains each one with the
same seed and split, and prints the resulting IC per forward period.

    python scripts/run_ablation.py --panel data/synthetic_panel.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path
from typing import Dict

from factorgcl import (
    ExperimentConfig,
    MarketPanel,
    Trainer,
    build_dataloader,
    build_model,
    generate_rolling_windows,
    split_by_date,
)
from factorgcl.metrics import PeriodMetrics

# name -> (model overrides, use_contrastive)
VARIANTS = {
    "-wo Prior": ({"use_prior": False}, True),
    "-wo Hidden": ({"use_hidden": False}, True),
    "-wo Alpha&CL": ({"use_alpha": False}, False),
    "-wo CL": ({}, False),
    "FactorGCL": ({}, True),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-hidden-factors", type=int, default=32)
    parser.add_argument("--batch-days", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--train-range", nargs=2, metavar=("START", "END"), default=None)
    parser.add_argument("--valid-range", nargs=2, metavar=("START", "END"), default=None)
    parser.add_argument("--test-range", nargs=2, metavar=("START", "END"), default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("outputs/ablation.json"))
    args = parser.parse_args()

    panel = MarketPanel.load(args.panel)
    base = ExperimentConfig()
    base.data.num_prior_factors = panel.num_industries
    if args.train_range is None:
        window = generate_rolling_windows(panel.dates[0], panel.dates[-1], base.rolling)[0]
        train_range, valid_range, test_range = window.train, window.valid, window.test
    else:
        train_range = tuple(args.train_range)
        valid_range = tuple(args.valid_range)
        test_range = tuple(args.test_range)
    print(f"split: train {train_range} | valid {valid_range} | test {test_range}\n")

    results: Dict[str, Dict[int, PeriodMetrics]] = {}
    for name, (overrides, use_contrastive) in VARIANTS.items():
        config = ExperimentConfig()
        config.data.num_prior_factors = panel.num_industries
        config.model.hidden_dim = args.hidden_dim
        config.model.num_hidden_factors = args.num_hidden_factors
        config.train.max_epochs = args.epochs
        config.train.batch_days = args.batch_days
        config.train.seed = args.seed
        config.train.device = args.device
        config.train.verbose = not args.quiet
        config.train.use_contrastive = use_contrastive
        for key, value in overrides.items():
            setattr(config.model, key, value)
        config.sync()

        print(f"--- {name} ---", flush=True)
        train_set, valid_set, test_set = split_by_date(
            panel, config.data, [train_range, valid_range, test_range]
        )
        trainer = Trainer(build_model(config.model, seed=config.train.seed), config.train, config.data)
        trainer.fit(train_set, valid_set)
        metrics, _ = trainer.evaluate(
            build_dataloader(test_set, batch_days=config.train.batch_days)
        )
        results[name] = metrics

    periods = list(base.data.forward_periods)
    header = f"{'':>14}" + "".join(f"{'IC dt=' + str(p):>12}" for p in periods)
    print("\n" + header)
    print("-" * len(header))
    for name in VARIANTS:
        row = f"{name:>14}" + "".join(f"{results[name][p].ic:12.4f}" for p in periods)
        print(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {name: {str(p): m.as_dict() for p, m in metrics.items()} for name, metrics in results.items()},
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
