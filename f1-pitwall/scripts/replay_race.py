#!/usr/bin/env python3
"""Phase 7: full-race lap-by-lap replay with optional pit-wall memory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from strategy_engineer.crew_chief_eval import pick_drivers_for_race  # noqa: E402
from strategy_engineer.memory import RaceMemoryStore  # noqa: E402
from shared.replay import RaceReplay  # noqa: E402
from shared.pipeline import get_sim_backend
from strategy_engineer.strategy import replay_race_decisions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--name-contains", default="British")
    parser.add_argument("--driver-id", type=int, default=None)
    parser.add_argument("--lap-from", type=int, default=1)
    parser.add_argument("--lap-to", type=int, default=None)
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Persist pit-wall instructions lap-to-lap (Phase 7)",
    )
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
    driver_id = args.driver_id
    if driver_id is None:
        drivers = pick_drivers_for_race(replay, race_id)
        driver_id = drivers[0] if drivers else replay.drivers_in_race(race_id)[0]

    backend = get_sim_backend(args.backend, replay)
    memory = RaceMemoryStore() if args.memory else None
    result = replay_race_decisions(
        backend,
        replay,
        race_id,
        driver_id,
        memory=memory,
        lap_from=args.lap_from,
        lap_to=args.lap_to,
    )

    print(
        f"race_id={race_id} driver_id={driver_id} "
        f"laps={result.laps[0]}..{result.laps[-1]} "
        f"backend={backend.name} memory={'on' if args.memory else 'off'}"
    )
    print(f"mean_regret={result.mean_regret:.4f}" if result.mean_regret else "")
    if args.memory:
        ff = result.flip_flop_rate(replay)
        print(f"flip_flop_rate={ff:.4f}" if ff is not None else "flip_flop_rate=n/a")

    rows = []
    for lap, sd in zip(result.laps, result.decisions):
        rows.append(
            {
                "lap": lap,
                "action": sd.decision.action,
                "chosen_option_id": sd.chosen_option_id,
                "oracle_option_id": sd.oracle_option_id,
                "regret": sd.regret,
            }
        )
    print(json.dumps(rows[-5:], indent=2))
    if len(rows) > 5:
        print(f"... ({len(rows)} laps total, showing last 5)")


if __name__ == "__main__":
    main()
