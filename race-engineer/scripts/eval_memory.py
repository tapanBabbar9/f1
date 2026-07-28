#!/usr/bin/env python3
"""Phase 7: memory-on vs memory-off full-race eval (regret delta + flip-flop rate)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from race_engineer.crew_chief_eval import (  # noqa: E402
    load_frozen_race_ids,
    pick_drivers_for_race,
)
from race_engineer.memory import RaceMemoryStore  # noqa: E402
from race_engineer.replay import RaceReplay  # noqa: E402
from race_engineer.sim import save_json  # noqa: E402
from race_engineer.sim_agent import get_sim_backend, replay_race_decisions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "openai_sim", "heuristic_sim", "pit_next_sim"),
        default="heuristic_sim",
    )
    parser.add_argument(
        "--max-races",
        type=int,
        default=0,
        help="Limit frozen races (0 = all)",
    )
    parser.add_argument(
        "--lap-from",
        type=int,
        default=1,
        help="First lap in replay window",
    )
    parser.add_argument(
        "--lap-to",
        type=int,
        default=None,
        help="Last lap (default: each driver's final lap)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "memory",
    )
    args = parser.parse_args()

    replay = RaceReplay(args.dataset).load()
    race_ids = load_frozen_race_ids()
    if args.max_races > 0:
        race_ids = race_ids[: args.max_races]

    backend = get_sim_backend(args.backend, replay)
    print(f"backend={backend.name} races={len(race_ids)}")

    off_regrets: list[float] = []
    on_regrets: list[float] = []
    flip_rates: list[float] = []
    n_laps = 0
    rows: list[dict] = []

    for rid in race_ids:
        for did in pick_drivers_for_race(replay, rid):
            off = replay_race_decisions(
                backend,
                replay,
                rid,
                did,
                memory=None,
                lap_from=args.lap_from,
                lap_to=args.lap_to,
            )
            mem = RaceMemoryStore()
            on = replay_race_decisions(
                backend,
                replay,
                rid,
                did,
                memory=mem,
                lap_from=args.lap_from,
                lap_to=args.lap_to,
            )
            if not off.decisions:
                continue
            off_mean = off.mean_regret or 0.0
            on_mean = on.mean_regret or 0.0
            ff = on.flip_flop_rate(replay)
            off_regrets.extend(d.regret for d in off.decisions)
            on_regrets.extend(d.regret for d in on.decisions)
            n_laps += len(off.decisions)
            if ff is not None:
                flip_rates.append(ff)
            rows.append(
                {
                    "race_id": rid,
                    "driver_id": did,
                    "n_laps": len(off.decisions),
                    "mean_regret_off": round(off_mean, 4),
                    "mean_regret_on": round(on_mean, 4),
                    "regret_delta": round(on_mean - off_mean, 4),
                    "flip_flop_rate": round(ff, 4) if ff is not None else None,
                }
            )

    def _mean(xs: list[float]) -> float | None:
        return round(sum(xs) / len(xs), 4) if xs else None

    payload = {
        "phase": 7,
        "backend": backend.name,
        "n_race_drivers": len(rows),
        "n_laps": n_laps,
        "lap_from": args.lap_from,
        "lap_to": args.lap_to,
        "mean_position_regret_memory_off": _mean(off_regrets),
        "mean_position_regret_memory_on": _mean(on_regrets),
        "mean_regret_delta_on_minus_off": (
            round(
                (_mean(on_regrets) or 0.0) - (_mean(off_regrets) or 0.0),
                4,
            )
            if off_regrets
            else None
        ),
        "mean_flip_flop_rate": _mean(flip_rates),
        "note": (
            "Full-race replay per frozen (race, driver). Flip-flop = consecutive "
            "laps where pit/stay flipped but evidence fingerprint unchanged "
            "(position/gap buckets, compound, pit count, SC, sim oracle). "
            "heuristic_sim follows oracle each lap — expect ~0 regret delta and "
            "low flip-flops unless the board is static while oracle action toggles."
        ),
        "by_race_driver": rows,
    }
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "metrics.json", payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "by_race_driver"}, indent=2))
    print(f"Wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
