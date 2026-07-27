#!/usr/bin/env python3
"""Evaluate Phase 5 sim calibration (Brier) on frozen-race sample points."""

from __future__ import annotations

import argparse
import csv
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
from race_engineer.sim import (  # noqa: E402
    brier_binary,
    oracle_best,
    save_json,
    simulate_strategy_cards,
)


def _load_finish_positions(dataset: Path) -> dict[tuple[int, int], int]:
    """(race_id, driver_id) -> finishing positionOrder."""
    path = dataset / "results.csv"
    out: dict[tuple[int, int], int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rid = int(row["raceId"])
                did = int(row["driverId"])
                pos = int(row["positionOrder"])
            except ValueError:
                continue
            out[(rid, did)] = pos
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset",
    )
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-rolls", type=int, default=64)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "sim",
    )
    args = parser.parse_args()

    replay = RaceReplay(args.dataset).load()
    finishes = _load_finish_positions(args.dataset)
    points = build_eval_points(
        replay,
        load_frozen_race_ids(),
        n_samples=args.samples,
        seed=args.seed,
    )
    print(f"points={len(points)} n_rolls={args.n_rolls}")

    probs3, y3 = [], []
    probs5, y5 = [], []
    probs10, y10 = [], []
    mean_pos_err = []
    rows = []

    for i, pt in enumerate(points):
        actual = finishes.get((pt.race_id, pt.driver_id))
        if actual is None:
            continue
        state = replay.get_state(pt.race_id, pt.driver_id, pt.lap)
        cards = simulate_strategy_cards(
            replay,
            state,
            n_rolls=args.n_rolls,
            seed=args.seed + i,
        )
        stay = next((c for c in cards if c.label == "stay_to_finish"), None)
        if stay is not None:
            card = stay
            card_name = "stay_to_finish"
        else:
            # Mandatory pit pending: use longest legal stay-then-pit trajectory.
            stay_opts = [
                c for c in cards if c.label.startswith("stay_") and c.label.endswith("_then_pit")
            ]
            if stay_opts:
                card = max(stay_opts, key=lambda c: c.pit_after_laps)
                card_name = card.label
            else:
                card = oracle_best(cards)
                card_name = card.label
        probs3.append(card.p_finish_le_3)
        probs5.append(card.p_finish_le_5)
        probs10.append(card.p_finish_le_10)
        y3.append(1 if actual <= 3 else 0)
        y5.append(1 if actual <= 5 else 0)
        y10.append(1 if actual <= 10 else 0)
        mean_pos_err.append(abs(card.mean_finish_pos - actual))
        rows.append(
            {
                "race_id": pt.race_id,
                "driver_id": pt.driver_id,
                "lap": pt.lap,
                "actual_finish": actual,
                "card": card_name,
                "mean_finish_pos": card.mean_finish_pos,
                "P_le_3": card.p_finish_le_3,
                "P_le_5": card.p_finish_le_5,
                "P_le_10": card.p_finish_le_10,
            }
        )
        if (i + 1) % 20 == 0:
            print(f"  … {i + 1}/{len(points)}")

    payload = {
        "phase": 5,
        "n_points": len(rows),
        "n_rolls": args.n_rolls,
        "seed": args.seed,
        "calibration_card": "stay_to_finish_if_legal_else_longest_stay_then_pit",
        "brier": {
            "finish_le_3": round(brier_binary(probs3, y3), 4) if probs3 else None,
            "finish_le_5": round(brier_binary(probs5, y5), 4) if probs5 else None,
            "finish_le_10": round(brier_binary(probs10, y10), 4) if probs10 else None,
        },
        "mean_abs_position_error": (
            round(float(sum(mean_pos_err) / len(mean_pos_err)), 3) if mean_pos_err else None
        ),
        "note": (
            "v0 static-rival field; dry-race pit constraints applied in sim. "
            "Brier on stay-to-finish when legal, else longest stay-then-pit "
            "trajectory vs actual finish. Lower Brier is better (0=perfect)."
        ),
    }
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    save_json(metrics_path, payload)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
