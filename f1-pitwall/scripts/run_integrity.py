#!/usr/bin/env python3
"""Phase 0 error metric: race-state integrity rate (target ≥ 99%)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.integrity import run_integrity_check
from shared.replay import RaceReplay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2021, 2022, 2023, 2024, 2025],
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset",
    )
    parser.add_argument("--min-integrity", type=float, default=0.99)
    args = parser.parse_args()

    print(f"Loading dataset from {args.dataset} ...")
    replay = RaceReplay(args.dataset).load()
    report = run_integrity_check(
        replay,
        n_samples=args.samples,
        seed=args.seed,
        years=tuple(args.years),
    )
    print(report.summary())
    if report.errors:
        print("\nexamples:")
        for e in report.errors[:10]:
            print(" ", e)

    if report.integrity_rate < args.min_integrity:
        raise SystemExit(
            f"FAIL: integrity_rate {report.integrity_rate:.4%} "
            f"< target {args.min_integrity:.2%}"
        )
    print(f"\nPASS: integrity_rate ≥ {args.min_integrity:.2%}")


if __name__ == "__main__":
    main()
