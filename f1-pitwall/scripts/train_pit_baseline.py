#!/usr/bin/env python3
"""Train Phase 1 pit-next-lap baselines and write metrics + predictions."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategy_engineer.pit_baseline import (
    DEFAULT_SPLITS,
    LABEL_DEFINITION,
    build_samples,
    compute_metrics,
    metrics_payload,
    samples_to_xy,
    save_metrics,
    score_model,
    summarize_split,
    train_models,
    write_predictions_csv,
)
from shared.replay import RaceReplay


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
        default=ROOT / "artifacts" / "pit_baseline",
    )
    parser.add_argument(
        "--primary",
        choices=("majority", "logistic", "hgb"),
        default="hgb",
        help="Primary model for predictions.csv and README metrics",
    )
    args = parser.parse_args()
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.dataset} ...")
    replay = RaceReplay(args.dataset).load()

    splits = {
        "train_years": list(DEFAULT_SPLITS["train_years"]),
        "val_years": list(DEFAULT_SPLITS["val_years"]),
        "test_years": list(DEFAULT_SPLITS["test_years"]),
    }

    print("Building samples (train/val/test) ...")
    train = build_samples(replay, splits["train_years"])
    val = build_samples(replay, splits["val_years"])
    test = build_samples(replay, splits["test_years"])

    split_stats = {
        "train": summarize_split("train", train),
        "val": summarize_split("val", val),
        "test": summarize_split("test", test),
    }
    for s in split_stats.values():
        print(
            f"  {s['split']}: n={s['n']} pos={s['n_positive']} "
            f"({s['positive_rate']:.3%}) races={s['n_races']}"
        )

    x_train, y_train = samples_to_xy(train)
    x_val, y_val = samples_to_xy(val)
    x_test, y_test = samples_to_xy(test)

    print("Training models ...")
    fitted = train_models(x_train, y_train, x_val, y_val)

    results: dict = {}
    for name, bundle in fitted.items():
        model = bundle["model"]
        thr = bundle["threshold"]
        results[name] = {
            "threshold": thr,
            "val": compute_metrics(y_val, score_model(model, x_val), thr),
            "test": compute_metrics(y_test, score_model(model, x_test), thr),
        }
        vt = results[name]["test"]
        print(
            f"  {name}: test F1={vt['f1']:.4f} AUROC={vt['auroc']:.4f} "
            f"P={vt['precision']:.4f} R={vt['recall']:.4f} thr={thr:.4f}"
        )

    primary = args.primary
    primary_model = fitted[primary]["model"]
    primary_thr = fitted[primary]["threshold"]
    test_scores = score_model(primary_model, x_test)
    test_pred = (test_scores >= primary_thr).astype(int)

    write_predictions_csv(out / "predictions_test.csv", test, test_scores, test_pred)
    print(f"Wrote {out / 'predictions_test.csv'}")

    payload = metrics_payload(
        label_definition=LABEL_DEFINITION,
        splits=splits,
        split_stats=split_stats,
        results=results,
        primary_model=primary,
    )
    save_metrics(out / "metrics.json", payload)
    print(f"Wrote {out / 'metrics.json'}")

    with (out / "models.pkl").open("wb") as f:
        pickle.dump(
            {
                "label_definition": LABEL_DEFINITION,
                "primary": primary,
                "models": {k: v["model"] for k, v in fitted.items()},
                "thresholds": {k: v["threshold"] for k, v in fitted.items()},
                "splits": splits,
            },
            f,
        )
    print(f"Wrote {out / 'models.pkl'}")

    primary_test = results[primary]["test"]
    print(
        f"\nPRIMARY ({primary}) test: "
        f"F1={primary_test['f1']:.4f} AUROC={primary_test['auroc']:.4f}"
    )


if __name__ == "__main__":
    main()
