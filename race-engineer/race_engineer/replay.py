from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from race_engineer.state import RaceState

# Default: repo-root dataset/ relative to race-engineer/
_DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "dataset"


def _parse_int(value: str, default: int | None = None) -> int | None:
    if value is None or value == "" or value == "\\N":
        return default
    return int(value)


class RaceReplay:
    """Seekable historical race state from Ergast-style CSVs."""

    def __init__(self, dataset_dir: Path | str | None = None):
        self.dataset_dir = Path(dataset_dir) if dataset_dir else _DEFAULT_DATASET
        self._races: dict[int, dict] = {}
        self._drivers: dict[int, dict] = {}
        self._circuits: dict[int, dict] = {}
        # race_id -> driver_id -> lap -> {position, milliseconds, time}
        self._laps: dict[int, dict[int, dict[int, dict]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        # race_id -> driver_id -> sorted pit laps
        self._pits: dict[int, dict[int, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # race_id -> max lap observed
        self._total_laps: dict[int, int] = {}
        # (race_id, driver_id, lap) -> {compound, tyreLife}
        self._tyres: dict[tuple[int, int, int], dict] = {}
        # race_id -> list[(deployed, retreated_or_None)]
        self._sc_periods: dict[int, list[tuple[int, int | None]]] = {}
        self._loaded = False

    def load(self) -> RaceReplay:
        self._load_races()
        self._load_drivers()
        self._load_circuits()
        self._load_lap_times()
        self._load_pit_stops()
        self._load_tyre_laps()
        self._load_safety_cars()
        self._loaded = True
        return self

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Call RaceReplay.load() before querying state")

    def _load_races(self) -> None:
        path = self.dataset_dir / "races.csv"
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rid = int(row["raceId"])
                self._races[rid] = row

    def _load_drivers(self) -> None:
        path = self.dataset_dir / "drivers.csv"
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self._drivers[int(row["driverId"])] = row

    def _load_circuits(self) -> None:
        path = self.dataset_dir / "circuits.csv"
        if not path.exists():
            return
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self._circuits[int(row["circuitId"])] = row

    def _load_lap_times(self) -> None:
        path = self.dataset_dir / "lap_times.csv"
        max_lap: dict[int, int] = defaultdict(int)
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rid = int(row["raceId"])
                did = int(row["driverId"])
                lap = int(row["lap"])
                self._laps[rid][did][lap] = {
                    "position": int(row["position"]),
                    "milliseconds": int(row["milliseconds"]),
                    "time": row["time"],
                }
                if lap > max_lap[rid]:
                    max_lap[rid] = lap
        self._total_laps = dict(max_lap)

    def _load_pit_stops(self) -> None:
        path = self.dataset_dir / "pit_stops.csv"
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rid = int(row["raceId"])
                did = int(row["driverId"])
                self._pits[rid][did].append(int(row["lap"]))
        for rid in self._pits:
            for did in self._pits[rid]:
                self._pits[rid][did].sort()

    def _load_tyre_laps(self) -> None:
        path = self.dataset_dir / "tyre_laps.csv"
        if not path.exists():
            return
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rid = int(row["raceId"])
                did = int(row["driverId"])
                lap = int(row["lap"])
                compound = (row.get("compound") or "").strip().upper()
                if compound in ("", "UNKNOWN", "TEST_UNKNOWN"):
                    compound_v: str | None = None
                else:
                    compound_v = compound
                life_raw = (row.get("tyreLife") or "").strip()
                life_v: int | None
                try:
                    life_v = int(float(life_raw)) if life_raw else None
                except ValueError:
                    life_v = None
                self._tyres[(rid, did, lap)] = {
                    "compound": compound_v,
                    "tyreLife": life_v,
                }

    def _load_safety_cars(self) -> None:
        path = self.dataset_dir / "safety_cars.csv"
        if not path.exists():
            return
        by_key = {
            f"{int(row['year'])} {row['name']}": rid
            for rid, row in self._races.items()
        }
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rid = by_key.get(row["Race"])
                if rid is None:
                    continue
                deployed = int(float(row["Deployed"]))
                retreated_raw = (row.get("Retreated") or "").strip()
                retreated: int | None
                if retreated_raw == "":
                    retreated = None
                else:
                    retreated = int(float(retreated_raw))
                self._sc_periods.setdefault(rid, []).append((deployed, retreated))

    def _sc_flags(
        self, race_id: int, lap: int, total_laps: int
    ) -> tuple[bool, int, bool, bool]:
        periods = self._sc_periods.get(race_id, [])
        sc_active = False
        laps_since = 0
        for deployed, retreated in periods:
            end = total_laps if retreated is None else retreated
            if deployed <= lap <= end:
                sc_active = True
                laps_since = lap - deployed
                break
        deployed_laps = {p[0] for p in periods}
        return (
            sc_active,
            laps_since if sc_active else 0,
            lap in deployed_laps,
            (lap - 1) in deployed_laps,
        )

    def list_races(self, year: int | None = None) -> list[dict]:
        self._require_loaded()
        rows = []
        for rid, row in sorted(self._races.items()):
            if year is not None and int(row["year"]) != year:
                continue
            rows.append(
                {
                    "race_id": rid,
                    "year": int(row["year"]),
                    "round": int(row["round"]),
                    "name": row["name"],
                    "date": row["date"],
                    "circuit_id": int(row["circuitId"]),
                }
            )
        return rows

    def drivers_in_race(self, race_id: int) -> list[int]:
        self._require_loaded()
        return sorted(self._laps.get(race_id, {}).keys())

    def available_laps(self, race_id: int, driver_id: int) -> list[int]:
        self._require_loaded()
        return sorted(self._laps.get(race_id, {}).get(driver_id, {}).keys())

    def pit_laps(self, race_id: int, driver_id: int) -> frozenset[int]:
        """Historical pit-stop lap numbers for a driver in a race."""
        self._require_loaded()
        return frozenset(self._pits.get(race_id, {}).get(driver_id, []))

    def _cumulative_times(self, race_id: int, lap: int) -> dict[int, int]:
        """Elapsed race time (ms) for each driver that has completed `lap`."""
        out: dict[int, int] = {}
        for did, driver_laps in self._laps.get(race_id, {}).items():
            if lap not in driver_laps:
                continue
            total = 0
            missing = False
            for L in range(1, lap + 1):
                if L not in driver_laps:
                    missing = True
                    break
                total += driver_laps[L]["milliseconds"]
            if not missing:
                out[did] = total
        return out

    def _stint_age(self, race_id: int, driver_id: int, lap: int) -> tuple[int, int, bool]:
        pit_laps = self._pits.get(race_id, {}).get(driver_id, [])
        pits_so_far = [p for p in pit_laps if p <= lap]
        pit_this_lap = lap in pit_laps
        last_pit = pits_so_far[-1] if pits_so_far else 0
        # New tyres from the pit lap onward → age 0 on the stop lap.
        stint_age = lap - last_pit
        return stint_age, len(pits_so_far), pit_this_lap

    def get_state(self, race_id: int, driver_id: int, lap: int) -> RaceState:
        self._require_loaded()
        if race_id not in self._races:
            raise KeyError(f"Unknown race_id={race_id}")
        driver_laps = self._laps.get(race_id, {}).get(driver_id)
        if not driver_laps or lap not in driver_laps:
            raise KeyError(
                f"No lap data for race_id={race_id} driver_id={driver_id} lap={lap}"
            )

        race = self._races[race_id]
        driver = self._drivers.get(driver_id, {})
        lap_row = driver_laps[lap]
        position = lap_row["position"]

        cumulatives = self._cumulative_times(race_id, lap)
        # Position board at this lap from cumulative time (lower = ahead).
        ordered = sorted(cumulatives.items(), key=lambda kv: kv[1])
        pos_by_driver = {did: i + 1 for i, (did, _) in enumerate(ordered)}

        gap_ahead_ms: int | None = None
        gap_behind_ms: int | None = None
        if driver_id in cumulatives:
            board_pos = pos_by_driver[driver_id]
            my_time = cumulatives[driver_id]
            if board_pos > 1:
                ahead_id = ordered[board_pos - 2][0]
                gap_ahead_ms = my_time - cumulatives[ahead_id]
            if board_pos < len(ordered):
                behind_id = ordered[board_pos][0]
                gap_behind_ms = cumulatives[behind_id] - my_time

        stint_age, pit_count, pit_this_lap = self._stint_age(race_id, driver_id, lap)

        window = []
        for L in range(max(1, lap - 4), lap + 1):
            if L in driver_laps:
                window.append(driver_laps[L]["milliseconds"])

        code = driver.get("code")
        if code in (None, "", "\\N"):
            code = None

        tyre = self._tyres.get((race_id, driver_id, lap), {})
        total_laps = self._total_laps.get(race_id, lap)
        sc_active, laps_since_sc, sc_this, sc_prev = self._sc_flags(
            race_id, lap, total_laps
        )
        circuit_id = int(race["circuitId"])
        circuit = self._circuits.get(circuit_id, {})
        circuit_name = (circuit.get("name") or circuit.get("circuitRef") or "").strip()
        if not circuit_name:
            circuit_name = f"circuit_{circuit_id}"

        return RaceState(
            race_id=race_id,
            year=int(race["year"]),
            race_name=race["name"],
            circuit_id=circuit_id,
            circuit_name=circuit_name,
            driver_id=driver_id,
            driver_ref=driver.get("driverRef", str(driver_id)),
            driver_code=code,
            lap=lap,
            total_laps=total_laps,
            position=position,
            gap_ahead_ms=gap_ahead_ms,
            gap_behind_ms=gap_behind_ms,
            stint_age_laps=stint_age,
            pit_this_lap=pit_this_lap,
            pit_count=pit_count,
            last_lap_time_ms=lap_row["milliseconds"],
            last_lap_times_ms=tuple(window),
            cumulative_time_ms=cumulatives.get(driver_id, 0),
            drivers_on_track=len(cumulatives),
            tyre_compound=tyre.get("compound"),
            tyre_life=tyre.get("tyreLife"),
            sc_active=sc_active,
            laps_since_sc_deploy=laps_since_sc,
            sc_deployed_this_lap=sc_this,
            sc_deployed_prev_lap=sc_prev,
        )
