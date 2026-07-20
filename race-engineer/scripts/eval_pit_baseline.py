#!/usr/bin/env python3
"""Evaluate a saved Phase 1 pit baseline on a year split (default: test)."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from race_engineer.pit_baseline import (
    build_samples,
    compute_metrics,
    samples_to_xy,
    score_model,
    write_predictions_csv,
)
from race_engineer.replay import RaceReplay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=ROOT / "artifacts" / "pit_baseline" / "models.pkl",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val", "test"),
        default="test",
    )
    parser.add_argument(
        "--model-name",
        choices=("majority", "logistic", "hgb", "primary"),
        default="primary",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional predictions CSV path",
    )
    args = parser.parse_args()

    with args.models.open("rb") as f:
        bundle = pickle.load(f)

    name = bundle["primary"] if args.model_name == "primary" else args.model_name
    model = bundle["models"][name]
    threshold = bundle["thresholds"][name]
    years = bundle["splits"][f"{args.split}_years"]

    replay = RaceReplay(args.dataset).load()
    samples = build_samples(replay, years)
    x, y = samples_to_xy(samples)
    scores = score_model(model, x)
    preds = (scores >= threshold).astype(int)
    metrics = compute_metrics(y, scores, threshold)

    print(json.dumps({"model": name, "split": args.split, "metrics": metrics}, indent=2))

    if args.out_csv:
        write_predictions_csv(args.out_csv, samples, scores, preds)
        print(f"Wrote {args.out_csv}")


if __name__ == "__main__":
    main()
