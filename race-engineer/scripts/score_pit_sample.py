#!/usr/bin/env python3
"""Score one RaceState with the Phase 1 pit baseline (CLI demo)."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from race_engineer.pit_baseline import (  # noqa: E402
    FEATURE_NAMES,
    build_samples,
    load_safety_car_periods,
    load_tyre_laps,
    score_model,
)
from race_engineer.replay import RaceReplay  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--name-contains", default="British")
    parser.add_argument("--driver-id", type=int, default=1)
    parser.add_argument("--lap", type=int, default=22)
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
    args = parser.parse_args()

    if not args.models.exists():
        raise SystemExit(
            f"Missing {args.models}. Train first:\n"
            "  race-engineer/.venv/bin/python race-engineer/scripts/train_pit_baseline.py"
        )

    replay = RaceReplay(args.dataset).load()
    matches = [
        r
        for r in replay.list_races(year=args.year)
        if args.name_contains.lower() in r["name"].lower()
    ]
    if not matches:
        raise SystemExit("race not found")
    race_id = matches[0]["race_id"]
    state = replay.get_state(race_id, args.driver_id, args.lap)

    samples = build_samples(
        replay,
        [args.year],
        sc_by_race=load_safety_car_periods(replay),
        tyre_by_key=load_tyre_laps(replay),
    )
    hit = next(
        (
            s
            for s in samples
            if s.race_id == race_id
            and s.driver_id == args.driver_id
            and s.lap == args.lap
        ),
        None,
    )
    if hit is None:
        raise SystemExit(
            "No labeled sample for this (race, driver, lap) "
            "(final lap has no next-lap label)."
        )

    with args.models.open("rb") as f:
        bundle = pickle.load(f)
    primary = bundle["primary"]
    model = bundle["models"][primary]
    thr = float(bundle["thresholds"][primary])
    score = float(score_model(model, np.asarray([hit.features], dtype=np.float64))[0])
    pred = int(score >= thr)

    print(state.pit_wall_view())
    print()
    print(
        json.dumps(
            {
                "label": "pit_on_lap_plus_1",
                "y_true": int(hit.y),
                "model": primary,
                "score": round(score, 4),
                "threshold": round(thr, 4),
                "y_pred": pred,
                "features": {
                    name: (None if val != val else round(float(val), 4))
                    for name, val in zip(FEATURE_NAMES, hit.features)
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
