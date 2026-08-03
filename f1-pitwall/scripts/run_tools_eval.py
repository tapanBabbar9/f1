#!/usr/bin/env python3
"""Phase 3: evaluate tool-calling crew chief (faithfulness + pit F1)."""

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

from strategy_engineer.crew_chief_eval import (  # noqa: E402
    build_eval_points,
    load_frozen_race_ids,
)
from strategy_engineer.pit_baseline import (  # noqa: E402
    FEATURE_NAMES,
    build_samples as build_baseline_samples,
    compute_metrics,
    load_safety_car_periods,
    load_tyre_laps,
    score_model,
)
from shared.replay import RaceReplay  # noqa: E402
from strategy_engineer.tools_agent import get_tool_backend  # noqa: E402


def _sanitize(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def _phase1_on_points(replay, points, models_path: Path):
    if not models_path.exists():
        return None
    with models_path.open("rb") as f:
        bundle = pickle.load(f)
    primary = bundle["primary"]
    model = bundle["models"][primary]
    thr = bundle["thresholds"][primary]
    years = sorted({int(replay._races[p.race_id]["year"]) for p in points})
    samples = build_baseline_samples(
        replay,
        years,
        sc_by_race=load_safety_car_periods(replay),
        tyre_by_key=load_tyre_laps(replay),
    )
    by_key = {(s.race_id, s.driver_id, s.lap): s for s in samples}
    import numpy as np

    y_true, scores = [], []
    for p in points:
        s = by_key.get((p.race_id, p.driver_id, p.lap))
        if s is None:
            continue
        y_true.append(p.y_true)
        x = np.asarray([s.features], dtype=np.float64)
        scores.append(float(score_model(model, x)[0]))
    if not y_true:
        return None
    return {
        "primary_model": primary,
        "threshold": thr,
        "n_scored": len(y_true),
        "feature_names": list(FEATURE_NAMES),
        "pit_metrics": compute_metrics(
            np.asarray(y_true, dtype=np.int64),
            np.asarray(scores, dtype=np.float64),
            thr,
        ),
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
        choices=("auto", "openai_tools", "heuristic_tools"),
        default="heuristic_tools",
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
        default=ROOT / "artifacts" / "tools",
    )
    parser.add_argument(
        "--phase1-models",
        type=Path,
        default=ROOT / "artifacts" / "pit_baseline" / "models.pkl",
    )
    parser.add_argument("--min-schema", type=float, default=0.99)
    parser.add_argument("--min-faithfulness", type=float, default=0.90)
    args = parser.parse_args()
    out: Path = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from {args.dataset} ...")
    replay = RaceReplay(args.dataset).load()
    race_ids = load_frozen_race_ids(args.frozen)
    points = build_eval_points(
        replay, race_ids, n_samples=args.samples, seed=args.seed
    )
    backend = get_tool_backend(args.backend, replay)
    print(
        f"Eval points: {len(points)} "
        f"(positives={sum(p.y_true for p in points)}) "
        f"backend={backend.name}"
    )

    import numpy as np

    y_true: list[int] = []
    y_score: list[float] = []
    schema_ok = 0
    faith_scores: list[float] = []
    tools_used_n = 0
    errors: list[str] = []
    rows: list[dict] = []

    for pt in points:
        state = replay.get_state(pt.race_id, pt.driver_id, pt.lap)
        try:
            td = backend.decide_with_tools(state)
            schema_ok += 1
            pred = td.decision.pit_next
            faith = float(td.faithfulness.get("faithfulness", 0.0))
            faith_scores.append(faith)
            if td.tool_results:
                tools_used_n += 1
            row = {
                "race_id": pt.race_id,
                "driver_id": pt.driver_id,
                "lap": pt.lap,
                "y_true": pt.y_true,
                "y_pred": pred,
                "score": float(pred),
                "action": td.decision.action,
                "tyre": td.decision.tyre,
                "push": td.decision.push,
                "rationale": td.decision.rationale,
                "schema_valid": True,
                "faithfulness": faith,
                "tools_used": "|".join(t.name for t in td.tool_results),
                "n_tools": len(td.tool_results),
            }
        except Exception as exc:  # noqa: BLE001
            pred = 0
            if len(errors) < 20:
                errors.append(
                    f"race={pt.race_id} driver={pt.driver_id} lap={pt.lap}: {exc}"
                )
            faith_scores.append(0.0)
            row = {
                "race_id": pt.race_id,
                "driver_id": pt.driver_id,
                "lap": pt.lap,
                "y_true": pt.y_true,
                "y_pred": pred,
                "score": 0.0,
                "action": None,
                "tyre": None,
                "push": None,
                "rationale": None,
                "schema_valid": False,
                "faithfulness": 0.0,
                "tools_used": "",
                "n_tools": 0,
                "error": str(exc),
            }
        y_true.append(pt.y_true)
        y_score.append(float(pred))
        rows.append(row)

    n = len(points)
    schema_validity = schema_ok / n if n else 0.0
    mean_faith = float(np.mean(faith_scores)) if faith_scores else 0.0
    pit_metrics = compute_metrics(
        np.asarray(y_true, dtype=np.int64),
        np.asarray(y_score, dtype=np.float64),
        threshold=0.5,
    )
    phase1 = _phase1_on_points(replay, points, args.phase1_models)

    payload = {
        "phase": 3,
        "label_definition": "pit_on_lap_plus_1",
        "frozen_races": args.frozen.name,
        "n_samples": args.samples,
        "seed": args.seed,
        "crew_chief_tools": {
            "backend": backend.name,
            "schema_validity": schema_validity,
            "schema_ok": schema_ok,
            "mean_faithfulness": mean_faith,
            "tool_use_rate": tools_used_n / n if n else 0.0,
            "pit_metrics": pit_metrics,
            "errors": errors,
        },
        "phase1_baseline_on_same_points": phase1,
    }
    metrics_path = out / "metrics.json"
    metrics_path.write_text(
        json.dumps(_sanitize(payload), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {metrics_path}")

    preds_path = out / "predictions.csv"
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
        "rationale",
        "schema_valid",
        "faithfulness",
        "tools_used",
        "n_tools",
    ]
    with preds_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Wrote {preds_path}")

    cc = payload["crew_chief_tools"]
    pm = cc["pit_metrics"]
    print(
        f"\nTools ({cc['backend']}): "
        f"schema={cc['schema_validity']:.2%} "
        f"faith={cc['mean_faithfulness']:.2%} "
        f"tool_use={cc['tool_use_rate']:.2%} "
        f"F1={pm['f1']:.4f}"
    )
    if phase1:
        print(
            f"Phase 1 (hgb) same points: "
            f"F1={phase1['pit_metrics']['f1']:.4f}"
        )

    ok = True
    if schema_validity < args.min_schema:
        print(f"FAIL: schema_validity < {args.min_schema:.2%}")
        ok = False
    if mean_faith < args.min_faithfulness:
        print(f"FAIL: mean_faithfulness < {args.min_faithfulness:.2%}")
        ok = False
    if ok:
        print(f"\nPASS: schema ≥ {args.min_schema:.2%} and faith ≥ {args.min_faithfulness:.2%}")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
