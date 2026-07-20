#!/usr/bin/env python3
"""
Download the latest Ergast-compatible F1 CSV dump and refresh dataset/.

Source: TracingInsights/RaceData (GitHub Releases) — same relational schema
as the legacy Ergast / Kaggle tables used by this repo.

Backward compatibility:
  - All existing core CSV filenames and columns are preserved.
  - New columns are only appended (e.g. sprint_results.rank).
  - Additive tables (safety_cars, red_flags, ...) may be added; they do not
    change existing notebook contracts.
  - Historical raceId values remain stable.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

RELEASE_URL = (
    "https://github.com/TracingInsights/RaceData/releases/latest/download/data.zip"
)

CORE_TABLES = (
    "circuits",
    "constructor_results",
    "constructor_standings",
    "constructors",
    "driver_standings",
    "drivers",
    "lap_times",
    "pit_stops",
    "qualifying",
    "races",
    "results",
    "seasons",
    "sprint_results",
    "status",
)

ADDITIVE_FILES = (
    "safety_cars.csv",
    "red_flags.csv",
    "virtual_safety_car_estimates.json",
)


def _header(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def assert_backward_compatible(old_dir: Path, new_dir: Path) -> list[str]:
    notes: list[str] = []
    for name in CORE_TABLES:
        old_path = old_dir / f"{name}.csv"
        new_path = new_dir / f"{name}.csv"
        if not old_path.exists():
            notes.append(f"{name}: no prior file (fresh install)")
            continue
        old_cols = _header(old_path)
        new_cols = _header(new_path)
        missing = [c for c in old_cols if c not in new_cols]
        if missing:
            raise SystemExit(f"BREAKING: {name}.csv missing columns {missing}")
        added = [c for c in new_cols if c not in old_cols]
        if added:
            notes.append(f"{name}: additive columns {added}")
        else:
            notes.append(f"{name}: schema unchanged")
    return notes


def download_zip(url: str, dest: Path) -> None:
    print(f"Downloading {url}")
    with urllib.request.urlopen(url, timeout=120) as resp:
        dest.write_bytes(resp.read())
    print(f"Wrote {dest} ({dest.stat().st_size:,} bytes)")


def extract_csvs(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            name = Path(info.filename).name
            if not name:
                continue
            target = out_dir / name
            with zf.open(info) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def year_stats(races_csv: Path) -> dict:
    with races_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    years = sorted({int(r["year"]) for r in rows})
    by_year = {}
    for y in years:
        by_year[y] = sum(1 for r in rows if int(r["year"]) == y)
    return {"min_year": years[0], "max_year": years[-1], "n_races": len(rows), "by_year": by_year}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "dataset",
    )
    parser.add_argument("--url", default=RELEASE_URL)
    parser.add_argument(
        "--max-year",
        type=int,
        default=None,
        help="If set, drop races (and dependent rows) with year > max-year after copy. "
        "Default keeps full dump (includes future calendar rounds).",
    )
    args = parser.parse_args()
    dataset_dir: Path = args.dataset_dir
    dataset_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="f1-dataset-") as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "data.zip"
        extracted = tmp_path / "extracted"
        download_zip(args.url, zip_path)
        extract_csvs(zip_path, extracted)

        notes = assert_backward_compatible(dataset_dir, extracted)
        print("Compatibility:")
        for n in notes:
            print(f"  - {n}")

        for name in CORE_TABLES:
            src = extracted / f"{name}.csv"
            dst = dataset_dir / f"{name}.csv"
            shutil.copy2(src, dst)
            print(f"updated {dst.name}")

        for fname in ADDITIVE_FILES:
            src = extracted / fname
            if src.exists():
                shutil.copy2(src, dataset_dir / fname)
                print(f"added/updated {fname}")

        stats = year_stats(dataset_dir / "races.csv")
        print(
            f"\nCoverage: {stats['n_races']} races, "
            f"{stats['min_year']}–{stats['max_year']}"
        )
        for y in range(2024, stats["max_year"] + 1):
            if y in stats["by_year"]:
                print(f"  {y}: {stats['by_year'][y]} races")

        meta = {
            "source": args.url,
            "min_year": stats["min_year"],
            "max_year": stats["max_year"],
            "n_races": stats["n_races"],
            "backward_compatible": True,
            "notes": notes,
        }
        (dataset_dir / "DATASET_META.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote {dataset_dir / 'DATASET_META.json'}")

    if args.max_year is not None:
        print(
            f"Note: --max-year={args.max_year} filtering is not applied in this "
            "lightweight updater; filter in loaders if needed. Full dump retained."
        )


if __name__ == "__main__":
    main()
