#!/usr/bin/env python3
"""Print a sample RaceState (pit-wall view) for a known race."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.replay import RaceReplay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race-id", type=int, default=None)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--name-contains", default="Abu Dhabi")
    parser.add_argument("--driver-id", type=int, default=None)
    parser.add_argument("--lap", type=int, default=34)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset",
    )
    args = parser.parse_args()

    replay = RaceReplay(args.dataset).load()

    race_id = args.race_id
    if race_id is None:
        matches = [
            r
            for r in replay.list_races(year=args.year)
            if args.name_contains.lower() in r["name"].lower()
        ]
        if not matches:
            raise SystemExit(f"No race matched year={args.year} name~={args.name_contains}")
        race_id = matches[0]["race_id"]

    driver_id = args.driver_id
    if driver_id is None:
        # Prefer a non-pit lap near P4 for a readable pit-wall demo.
        candidates = []
        for did in replay.drivers_in_race(race_id):
            if args.lap not in replay.available_laps(race_id, did):
                continue
            st = replay.get_state(race_id, did, args.lap)
            if st.pit_this_lap:
                continue
            candidates.append(st)
        if not candidates:
            raise SystemExit(f"No driver has lap {args.lap} in race {race_id}")
        candidates.sort(key=lambda s: (abs(s.position - 4), s.position))
        driver_id = candidates[0].driver_id

    state = replay.get_state(race_id, driver_id, args.lap)
    print(state.pit_wall_view())
    print()
    print("--- raw ---")
    for k, v in state.to_dict().items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
