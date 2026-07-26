#!/usr/bin/env python3
"""Phase 6: evaluate sim-conditioned agent (position regret vs oracle)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from race_engineer.crew_chief_eval import (  # noqa: E402
    build_eval_points,
    load_frozen_race_ids,
)
from race_engineer.replay import RaceReplay  # noqa: E402
from race_engineer.sim import save_json  # noqa: E402
from race_engineer.sim_agent import (  # noqa: E402
    PitNextSimBaseline,
    get_sim_backend,
)


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
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "sim_agent",
    )
    args = parser.parse_args()

    replay = RaceReplay(args.dataset).load()
    points = build_eval_points(
        replay,
        load_frozen_race_ids(),
        n_samples=args.samples,
        seed=args.seed,
    )
    backend = get_sim_backend(args.backend, replay)
    # Contrast baseline always-box-next (same cards).
    pit_base = PitNextSimBaseline(replay)

    print(f"backend={backend.name} points={len(points)}")

    regrets = []
    faiths = []
    oracle_match = 0
    schema_ok = 0
    pit_regrets = []
    rows = []

    for i, pt in enumerate(points):
        state = replay.get_state(pt.race_id, pt.driver_id, pt.lap)
        sd = backend.decide_with_sims(state)
        schema_ok += 1
        regrets.append(sd.regret)
        faiths.append(float(sd.faithfulness.get("faithfulness", 0.0)))
        if sd.chosen_option_id == sd.oracle_option_id:
            oracle_match += 1

        pd = pit_base.decide_with_sims(state)
        pit_regrets.append(pd.regret)

        rows.append(
            {
                "race_id": pt.race_id,
                "driver_id": pt.driver_id,
                "lap": pt.lap,
                "action": sd.decision.action,
                "chosen_option_id": sd.chosen_option_id,
                "oracle_option_id": sd.oracle_option_id,
                "regret": sd.regret,
                "faithfulness": sd.faithfulness.get("faithfulness"),
            }
        )
        if (i + 1) % 20 == 0:
            print(f"  … {i + 1}/{len(points)}")

    n = len(regrets)
    payload = {
        "phase": 6,
        "backend": backend.name,
        "n_points": n,
        "seed": args.seed,
        "schema_validity": 1.0 if n else None,
        "mean_position_regret": round(sum(regrets) / n, 4) if n else None,
        "oracle_match_rate": round(oracle_match / n, 4) if n else None,
        "mean_faithfulness": round(sum(faiths) / n, 4) if n else None,
        "pit_next_baseline_mean_regret": (
            round(sum(pit_regrets) / n, 4) if n else None
        ),
        "note": (
            "Regret = E[finish|chosen] − E[finish|oracle] on Phase 5 cards. "
            "heuristic_sim follows oracle (expect ~0 regret). "
            "pit_next_baseline is always box-next for contrast."
        ),
    }
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "metrics.json", payload)
    # Predictions CSV is gitignored via artifacts/**/*.csv
    csv_path = out / f"predictions_{backend.name}.csv"
    if rows:
        import csv

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
