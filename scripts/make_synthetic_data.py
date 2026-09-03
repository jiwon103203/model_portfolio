#!/usr/bin/env python3
"""Generate a synthetic market panel so the pipeline can be run end to end.

The China A-share / Hong Kong datasets of the paper are proprietary; this
produces a market with the same shape (day-level high/open/low/close/vwap/volume
plus an industry membership) and a known latent factor structure.

    python scripts/make_synthetic_data.py --out data/synthetic_panel.npz
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from pathlib import Path

import numpy as np

from factorgcl.data import SyntheticConfig, generate_market_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic_panel.npz"))
    parser.add_argument("--num-stocks", type=int, default=500)
    parser.add_argument("--num-days", type=int, default=2400)
    parser.add_argument("--num-industries", type=int, default=83)
    parser.add_argument("--num-hidden-factors", type=int, default=8)
    parser.add_argument("--start-date", type=str, default="2014-01-01")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = SyntheticConfig(
        num_stocks=args.num_stocks,
        num_days=args.num_days,
        num_industries=args.num_industries,
        num_hidden_factors=args.num_hidden_factors,
        start_date=args.start_date,
        seed=args.seed,
    )
    panel = generate_market_panel(config)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    panel.save(args.out)

    print(f"wrote {args.out}")
    print(f"  dates    : {panel.num_dates} ({panel.dates[0]} .. {panel.dates[-1]})")
    print(f"  stocks   : {panel.num_stocks}")
    print(f"  fields   : {panel.fields}")
    print(f"  industries: {panel.num_industries}")
    print(f"  missing  : {float(np.isnan(panel.values).mean()):.2%}")


if __name__ == "__main__":
    main()
