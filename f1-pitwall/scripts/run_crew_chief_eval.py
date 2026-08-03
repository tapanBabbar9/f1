#!/usr/bin/env python3
"""Phase 2: evaluate crew chief on frozen races; compare to Phase 1 baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategy_engineer.crew_chief_eval import (
    build_eval_points,
    load_frozen_race_ids,
    run_crew_chief_eval,
)
from strategy_engineer.board import get_backend
from strategy_engineer.pit_baseline import (
    FEATURE_NAMES,
    build_samples as build_baseline_samples,
    compute_metrics,
    load_safety_car_periods,
    load_tyre_laps,
    score_model,
)
from shared.replay import RaceReplay


def _sanitize(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _phase1_on_points(replay, points, models_path: Path) -> dict | None:
    if not models_path.exists():
        return None
    with models_path.open("rb") as f:
        bundle = pickle.load(f)
    primary = bundle["primary"]
    model = bundle["models"][primary]
    thr = bundle["thresholds"][primary]

    # Build a lookup of features for needed (race, driver, lap) via build_samples
    years = sorted({int(replay._races[p.race_id]["year"]) for p in points})
    samples = build_baseline_samples(
        replay,
        years,
        sc_by_race=load_safety_car_periods(replay),
        tyre_by_key=load_tyre_laps(replay),
    )
    by_key = {(s.race_id, s.driver_id, s.lap): s for s in samples}

    y_true = []
    scores = []
    missing = 0
    for p in points:
        s = by_key.get((p.race_id, p.driver_id, p.lap))
        if s is None:
            missing += 1
            continue
        y_true.append(p.y_true)
        import numpy as np

        x = np.asarray([s.features], dtype=np.float64)
        scores.append(float(score_model(model, x)[0]))

    if not y_true:
        return None
    import numpy as np

    metrics = compute_metrics(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(scores, dtype=np.float64),
        thr,
    )
    return {
        "primary_model": primary,
        "threshold": thr,
        "n_scored": len(y_true),
        "n_missing_features": missing,
        "feature_names": list(FEATURE_NAMES),
        "pit_metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "openai", "heuristic"),
        default="auto",
    )
    parser.add_argument("--samples", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--frozen",
        type=Path,
        default=ROOT / "artifacts" / "eval" / "frozen_races.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "crew_chief",
    )
    parser.add_argument(
        "--phase1-models",
        type=Path,
        default=ROOT / "artifacts" / "pit_baseline" / "models.pkl",
    )
    parser.add_argument(
        "--min-schema",
        type=float,
        default=0.99,
        help="Fail if schema validity below this (LLM/heuristic)",
    )
    args = parser.parse_args()
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.dataset} ...")
    replay = RaceReplay(args.dataset).load()
    race_ids = load_frozen_race_ids(args.frozen)
    points = build_eval_points(
        replay, race_ids, n_samples=args.samples, seed=args.seed
    )
    print(
        f"Eval points: {len(points)} "
        f"(positives={sum(p.y_true for p in points)}) "
        f"backend={args.backend}"
    )

    backend = get_backend(args.backend)
    print(f"Using backend: {backend.name}")
    result = run_crew_chief_eval(replay, backend, points)

    phase1 = _phase1_on_points(replay, points, args.phase1_models)
    payload = {
        "phase": 2,
        "label_definition": "pit_on_lap_plus_1",
        "frozen_races": args.frozen.name,
        "n_samples": args.samples,
        "seed": args.seed,
        "crew_chief": {
            "backend": result["backend"],
            "schema_validity": result["schema_validity"],
            "schema_ok": result["schema_ok"],
            "pit_metrics": result["pit_metrics"],
            "errors": result["errors"],
        },
        "phase1_baseline_on_same_points": phase1,
    }

    metrics_path = out / "metrics.json"
    metrics_path.write_text(
        json.dumps(_sanitize(payload), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {metrics_path}")

    preds_path = out / "predictions.csv"
    with preds_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "race_id",
            "driver_id",
            "lap",
            "y_true",
            "y_pred",
            "score",
            "action",
            "tyre",
            "push",
            "reason",
            "driver_message",
            "rationale",
            "schema_valid",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in result["decisions"]:
            w.writerow(row)
    print(f"Wrote {preds_path}")

    cc = payload["crew_chief"]
    pm = cc["pit_metrics"]
    print(
        f"\nCrew chief ({cc['backend']}): "
        f"schema={cc['schema_validity']:.2%} "
        f"F1={pm['f1']:.4f} P={pm['precision']:.4f} R={pm['recall']:.4f}"
    )
    if phase1:
        p1m = phase1["pit_metrics"]
        print(
            f"Phase 1 ({phase1['primary_model']}) same points: "
            f"F1={p1m['f1']:.4f} AUROC={p1m.get('auroc')}"
        )

    if cc["schema_validity"] < args.min_schema:
        raise SystemExit(
            f"FAIL: schema_validity {cc['schema_validity']:.2%} "
            f"< {args.min_schema:.2%}"
        )
    print(f"\nPASS: schema_validity ≥ {args.min_schema:.2%}")


if __name__ == "__main__":
    main()
