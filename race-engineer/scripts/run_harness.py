#!/usr/bin/env python3
"""Phase 8: evaluation harness 1.0.0 — full-race model comparison."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from race_engineer.crew_chief_eval import (  # noqa: E402
    load_frozen_race_ids,
    pick_drivers_for_race,
)
from race_engineer.harness import (  # noqa: E402
    HARNESS_VERSION,
    render_report,
    run_harness,
)
from race_engineer.replay import RaceReplay  # noqa: E402
from race_engineer.sim import save_json  # noqa: E402


def _write_lap_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset",
    )
    parser.add_argument(
        "--models",
        default="hgb,heuristic_sim,heuristic_crew",
        help="Comma-separated: hgb, heuristic_crew, heuristic_sim, openai_sim, pit_next_sim",
    )
    parser.add_argument(
        "--max-races",
        type=int,
        default=0,
        help="Limit frozen races (0 = all)",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=2,
        help="Pit-lap error within-N-laps tolerance",
    )
    parser.add_argument(
        "--memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pit-wall memory for sim agents (default: on)",
    )
    parser.add_argument("--lap-from", type=int, default=1)
    parser.add_argument("--lap-to", type=int, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "artifacts" / "eval",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip markdown report (gitignored)",
    )
    args = parser.parse_args()

    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    replay = RaceReplay(args.dataset).load()
    race_ids = load_frozen_race_ids()
    if args.max_races > 0:
        race_ids = race_ids[: args.max_races]

    print(
        f"harness={HARNESS_VERSION} models={model_ids} "
        f"races={len(race_ids)} tolerance=±{args.tolerance} memory={args.memory}"
    )

    payload = run_harness(
        replay,
        race_ids,
        model_ids,
        pick_drivers_fn=pick_drivers_for_race,
        tolerance_laps=args.tolerance,
        memory=args.memory,
        lap_from=args.lap_from,
        lap_to=args.lap_to,
    )

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    lap_records = payload.pop("lap_records", [])
    save_json(out / "metrics.json", payload)

    manifest = {
        "harness_version": HARNESS_VERSION,
        "schema": "eval_harness_v1",
        "leaderboards": ["leaderboard_pit_next", "leaderboard_sim_plan"],
        "models": model_ids,
        "frozen_races_id": payload["frozen_races_id"],
    }
    save_json(out / "harness_manifest.json", manifest)

    if not args.no_report:
        report = render_report(payload)
        (out / "report.md").write_text(report, encoding="utf-8")
        print(f"Wrote {out / 'report.md'}")

    preds_dir = out / "predictions"
    for mid in model_ids:
        rows = [r for r in lap_records if r.get("model_id") == mid]
        _write_lap_csv(preds_dir / f"{mid}_laps.csv", rows)

    print(
        json.dumps(
            {
                "leaderboard_pit_next": payload["leaderboard_pit_next"],
                "leaderboard_sim_plan": payload["leaderboard_sim_plan"],
            },
            indent=2,
        )
    )
    print(f"Wrote {out / 'metrics.json'}")
    print(f"Wrote {out / 'harness_manifest.json'}")


if __name__ == "__main__":
    main()
