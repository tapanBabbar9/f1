#!/usr/bin/env python3
"""Single-shot crew chief decision for one RaceState (CLI demo)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategy_engineer.board import get_backend
from shared.replay import RaceReplay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--name-contains", default="British")
    parser.add_argument("--driver-id", type=int, default=1)
    parser.add_argument("--lap", type=int, default=22)
    parser.add_argument(
        "--backend",
        choices=("auto", "openai", "heuristic"),
        default="auto",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset",
    )
    args = parser.parse_args()

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
    print(state.pit_wall_view())
    print()
    backend = get_backend(args.backend)
    print(f"backend: {backend.name}")
    decision = backend.decide(state)
    print(json.dumps(decision.to_dict(), indent=2))


if __name__ == "__main__":
    main()
