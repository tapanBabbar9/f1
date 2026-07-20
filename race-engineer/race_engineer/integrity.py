from __future__ import annotations

import random
from dataclasses import dataclass

from race_engineer.replay import RaceReplay


@dataclass
class IntegrityReport:
    samples: int
    position_ok: int
    pit_flag_ok: int
    stint_age_ok: int
    errors: list[str]

    @property
    def position_rate(self) -> float:
        return self.position_ok / self.samples if self.samples else 0.0

    @property
    def pit_flag_rate(self) -> float:
        return self.pit_flag_ok / self.samples if self.samples else 0.0

    @property
    def stint_age_rate(self) -> float:
        return self.stint_age_ok / self.samples if self.samples else 0.0

    @property
    def integrity_rate(self) -> float:
        """Fraction of samples where position, pit flag, and stint age all match."""
        if not self.samples:
            return 0.0
        # Conservative: average of the three checks (all must be high for Phase 0).
        return min(self.position_rate, self.pit_flag_rate, self.stint_age_rate)

    def summary(self) -> str:
        return (
            f"samples={self.samples}\n"
            f"position_match={self.position_rate:.4%} ({self.position_ok}/{self.samples})\n"
            f"pit_flag_match={self.pit_flag_rate:.4%} ({self.pit_flag_ok}/{self.samples})\n"
            f"stint_age_match={self.stint_age_rate:.4%} ({self.stint_age_ok}/{self.samples})\n"
            f"integrity_rate={self.integrity_rate:.4%}  (min of the three; target ≥ 99%)\n"
            f"error_examples={len(self.errors)}"
        )


def run_integrity_check(
    replay: RaceReplay,
    *,
    n_samples: int = 2000,
    seed: int = 42,
    years: tuple[int, ...] = (2021, 2022, 2023, 2024, 2025),
) -> IntegrityReport:
    """
    Error metric for Phase 0.

    Sample random (race, driver, lap) tuples and verify:
      1. state.position == lap_times.position
      2. state.pit_this_lap iff a pit_stops row exists for that lap
      3. stint_age_laps == lap - last_pit_lap (0 if no pit yet → age == lap)
    """
    rng = random.Random(seed)
    candidates: list[tuple[int, int, int]] = []

    for race in replay.list_races():
        if race["year"] not in years:
            continue
        rid = race["race_id"]
        for did in replay.drivers_in_race(rid):
            for lap in replay.available_laps(rid, did):
                candidates.append((rid, did, lap))

    if not candidates:
        return IntegrityReport(0, 0, 0, 0, ["no candidates in selected years"])

    sample_n = min(n_samples, len(candidates))
    samples = rng.sample(candidates, sample_n)

    position_ok = pit_ok = stint_ok = 0
    errors: list[str] = []

    for rid, did, lap in samples:
        state = replay.get_state(rid, did, lap)
        raw = replay._laps[rid][did][lap]
        expected_pos = raw["position"]
        expected_pit = lap in replay._pits.get(rid, {}).get(did, [])
        pit_laps = [p for p in replay._pits.get(rid, {}).get(did, []) if p <= lap]
        last_pit = pit_laps[-1] if pit_laps else 0
        expected_age = lap - last_pit

        pos_match = state.position == expected_pos
        pit_match = state.pit_this_lap == expected_pit
        age_match = state.stint_age_laps == expected_age

        if pos_match:
            position_ok += 1
        if pit_match:
            pit_ok += 1
        if age_match:
            stint_ok += 1

        if not (pos_match and pit_match and age_match) and len(errors) < 20:
            errors.append(
                f"race={rid} driver={did} lap={lap} "
                f"pos={state.position}/{expected_pos} "
                f"pit={state.pit_this_lap}/{expected_pit} "
                f"age={state.stint_age_laps}/{expected_age}"
            )

    return IntegrityReport(
        samples=sample_n,
        position_ok=position_ok,
        pit_flag_ok=pit_ok,
        stint_age_ok=stint_ok,
        errors=errors,
    )
