#!/usr/bin/env python3
"""
Build dataset/tyre_laps.csv from FastF1 (Phase 1 tyre compound).

Joins to Ergast-style tables via:
  year + race name → raceId
  drivers.code → driverId
  LapNumber → lap

Compound values: SOFT, MEDIUM, HARD, INTERMEDIATE, WET, UNKNOWN, ...
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from race_engineer.replay import RaceReplay

COMPOUND_FILE = "tyre_laps.csv"
CACHE_DIR_NAME = "fastf1_cache"


def _enable_cache(cache_dir: Path) -> None:
    import fastf1

    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))


def _driver_code_map(replay: RaceReplay) -> dict[str, int]:
    """Map 3-letter code → driverId (prefer latest non-empty code)."""
    out: dict[str, int] = {}
    for did, row in replay._drivers.items():
        code = (row.get("code") or "").strip()
        if code and code != "\\N":
            out[code.upper()] = did
    return out


def _load_existing(path: Path) -> set[tuple[int, int, int]]:
    if not path.exists():
        return set()
    keys: set[tuple[int, int, int]] = set()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            keys.add((int(row["raceId"]), int(row["driverId"]), int(row["lap"])))
    return keys


def fetch_race_tyres(
    year: int,
    race_name: str,
    race_id: int,
    code_to_id: dict[str, int],
    *,
    retries: int = 4,
) -> list[dict]:
    import fastf1

    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            session = fastf1.get_session(year, race_name, "R")
            session.load(laps=True, telemetry=False, weather=False, messages=False)
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc).lower()
            if "500 calls" in msg or "rate" in msg:
                wait = 300 * (attempt + 1)
                print(f"  rate-limited; sleeping {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            # Transient FastF1 load glitches (e.g. empty session after partial fail)
            if "not been loaded yet" in msg and attempt < retries - 1:
                wait = 15 * (attempt + 1)
                print(f"  load glitch; retrying in {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            raise
    else:
        assert last_exc is not None
        raise last_exc

    laps = session.laps
    if laps is None or laps.empty:
        return []

    rows: list[dict] = []
    for _, lap in laps.iterrows():
        code = str(lap.get("Driver") or "").strip().upper()
        if not code or code not in code_to_id:
            continue
        lap_no = lap.get("LapNumber")
        if lap_no is None or (isinstance(lap_no, float) and lap_no != lap_no):
            continue
        compound = str(lap.get("Compound") or "UNKNOWN").strip().upper()
        if compound in ("", "NAN", "NONE"):
            compound = "UNKNOWN"
        stint = lap.get("Stint")
        tyre_life = lap.get("TyreLife")
        try:
            stint_v = int(float(stint)) if stint == stint and stint is not None else ""
        except (TypeError, ValueError):
            stint_v = ""
        try:
            life_v = (
                int(float(tyre_life))
                if tyre_life == tyre_life and tyre_life is not None
                else ""
            )
        except (TypeError, ValueError):
            life_v = ""
        rows.append(
            {
                "raceId": race_id,
                "driverId": code_to_id[code],
                "lap": int(float(lap_no)),
                "compound": compound,
                "stint": stint_v,
                "tyreLife": life_v,
                "driverCode": code,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(range(2018, 2026)),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "artifacts" / CACHE_DIR_NAME,
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=2.0,
        help="Pause between race downloads to be polite to the API",
    )
    parser.add_argument(
        "--limit-races",
        type=int,
        default=None,
        help="Optional cap for smoke tests",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch races even if rows already exist for that raceId",
    )
    args = parser.parse_args()

    _enable_cache(args.cache_dir)
    replay = RaceReplay(args.dataset).load()
    code_to_id = _driver_code_map(replay)
    out_path = args.dataset / COMPOUND_FILE

    fieldnames = [
        "raceId",
        "driverId",
        "lap",
        "compound",
        "stint",
        "tyreLife",
        "driverCode",
    ]

    if args.force or not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    existing_keys = _load_existing(out_path)
    existing_race_ids = {k[0] for k in existing_keys}

    races = [
        r for r in replay.list_races() if r["year"] in set(args.years)
    ]
    if args.limit_races is not None:
        races = races[: args.limit_races]

    failures: list[str] = []
    fetched = 0

    for race in races:
        rid = race["race_id"]
        if not args.force and rid in existing_race_ids:
            continue
        label = f"{race['year']} {race['name']}"
        print(f"Fetching {label} (raceId={rid}) ...", flush=True)
        try:
            rows = fetch_race_tyres(
                race["year"], race["name"], rid, code_to_id
            )
        except Exception as exc:  # noqa: BLE001 - want full ingest to continue
            failures.append(f"{label}: {exc}")
            print(f"  FAIL: {exc}", flush=True)
            time.sleep(args.sleep)
            continue
        print(f"  {len(rows)} lap-compound rows", flush=True)
        with out_path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerows(rows)
        existing_race_ids.add(rid)
        fetched += 1
        time.sleep(args.sleep)

    # Final de-dupe pass for safety.
    all_rows = []
    with out_path.open(newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    dedup: dict[tuple[int, int, int], dict] = {}
    for row in all_rows:
        key = (int(row["raceId"]), int(row["driverId"]), int(row["lap"]))
        dedup[key] = row
    final = [dedup[k] for k in sorted(dedup)]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(final)

    print(
        f"\nWrote {out_path} ({len(final)} rows). "
        f"Fetched {fetched} races this run. Failures={len(failures)}",
        flush=True,
    )
    for fail in failures[:20]:
        print(" ", fail, flush=True)


if __name__ == "__main__":
    main()
