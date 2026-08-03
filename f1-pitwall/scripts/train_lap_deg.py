#!/usr/bin/env python3
"""Train Phase 4 lap-degradation model (next-lap ms) and write metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategy_engineer.lap_deg import (  # noqa: E402
    FEATURE_NAMES,
    LapDegModel,
    build_lap_deg_samples,
    metrics_by_circuit,
    predict_ms,
    regression_metrics,
    save_metrics,
    split_by_year,
    train_hgb,
    _xy,
)
from shared.replay import RaceReplay  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "lap_deg",
    )
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.dataset} ...")
    replay = RaceReplay(args.dataset).load()
    years = list(range(2018, 2026))
    print("Building stint-relative samples ...")
    samples = build_lap_deg_samples(replay, years)
    train, val, test = split_by_year(
        samples,
        train_years=range(2018, 2023),
        val_years=[2023],
        test_years=[2024, 2025],
    )
    print(f"n_train={len(train)} n_val={len(val)} n_test={len(test)}")

    print("Training HGB ...")
    model = train_hgb(train)
    bundle = LapDegModel(
        model=model,
        feature_names=list(FEATURE_NAMES),
        train_years=list(range(2018, 2023)),
        val_years=[2023],
        test_years=[2024, 2025],
    )
    model_path = out / "model.pkl"
    bundle.save(model_path)
    print(f"Wrote {model_path}")

    splits = {"train": train, "val": val, "test": test}
    payload: dict = {
        "phase": 4,
        "label_definition": "next_lap_ms",
        "feature_names": list(FEATURE_NAMES),
        "model": "hgb",
        "splits": {},
    }
    for name, rows in splits.items():
        if not rows:
            continue
        x, y = _xy(rows)
        pred = predict_ms(model, x)
        m = regression_metrics(y, pred)
        payload["splits"][name] = {
            **m,
            "by_circuit": metrics_by_circuit(rows, pred),
        }
        print(
            f"{name}: MAE={m['mae_ms']:.1f} ms  "
            f"MAPE={m['mape_pct']:.2f}%  n={m['n']}"
        )

    metrics_path = out / "metrics.json"
    save_metrics(metrics_path, payload)
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
