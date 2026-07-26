#!/usr/bin/env python3
"""Single-shot Phase 6 sim-conditioned decision (CLI demo)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from race_engineer.replay import RaceReplay
from race_engineer.sim_agent import get_sim_backend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--name-contains", default="British")
    parser.add_argument("--driver-id", type=int, default=1)
    parser.add_argument("--lap", type=int, default=22)
    parser.add_argument(
        "--backend",
        choices=("auto", "openai_sim", "heuristic_sim", "pit_next_sim"),
        default="heuristic_sim",
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
    backend = get_sim_backend(args.backend, replay)
    print(f"backend: {backend.name}")
    sd = backend.decide_with_sims(state)
    print(json.dumps(sd.to_dict(), indent=2))


if __name__ == "__main__":
    main()
