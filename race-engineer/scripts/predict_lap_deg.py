#!/usr/bin/env python3
"""Predict next-lap ms for one RaceState (Phase 4 deg CLI demo)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from race_engineer.lap_deg import LapDegModel
from race_engineer.replay import RaceReplay


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
        "--model",
        type=Path,
        default=ROOT / "artifacts" / "lap_deg" / "model.pkl",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(
            f"Missing {args.model}. Train first:\n"
            "  race-engineer/.venv/bin/python race-engineer/scripts/train_lap_deg.py"
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
    deg = LapDegModel.load(args.model)
    pred_ms = deg.predict_next_lap_ms(state)

    # Actual next lap if present.
    actual = None
    laps = replay._laps.get(race_id, {}).get(args.driver_id, {})
    if args.lap + 1 in laps:
        actual = laps[args.lap + 1]["milliseconds"]

    print(state.pit_wall_view())
    print()
    payload = {
        "predicted_next_lap_ms": round(pred_ms, 1),
        "predicted_next_lap_s": round(pred_ms / 1000.0, 3),
        "actual_next_lap_ms": actual,
        "error_ms": None if actual is None else round(pred_ms - actual, 1),
        "last_lap_ms": state.last_lap_time_ms,
        "stint_age_laps": state.stint_age_laps,
        "tyre_compound": state.tyre_compound,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
