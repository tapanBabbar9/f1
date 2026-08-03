from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class RaceState:
    """Unified pit-wall snapshot for a single (race, driver, lap)."""

    race_id: int
    year: int
    race_name: str
    circuit_id: int
    circuit_name: str
    driver_id: int
    driver_ref: str
    driver_code: str | None
    lap: int
    total_laps: int
    position: int
    gap_ahead_ms: int | None
    gap_behind_ms: int | None
    stint_age_laps: int
    pit_this_lap: bool
    pit_count: int
    last_lap_time_ms: int
    last_lap_times_ms: tuple[int, ...] = field(default_factory=tuple)
    cumulative_time_ms: int = 0
    drivers_on_track: int = 0
    tyre_compound: str | None = None
    tyre_life: int | None = None
    sc_active: bool = False
    laps_since_sc_deploy: int = 0
    sc_deployed_this_lap: bool = False
    sc_deployed_prev_lap: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def pit_wall_view(self, *, anonymize: bool = False) -> str:
        """Human-readable snapshot shaped like a race-engineer feed.

        When ``anonymize=True`` (LLM prompts), omit race name, year, and driver
        identity so the model cannot recall the historical result by fingerprint.
        """
        gap_ahead = (
            f"+{self.gap_ahead_ms / 1000:.3f}s"
            if self.gap_ahead_ms is not None
            else "—"
        )
        gap_behind = (
            f"-{self.gap_behind_ms / 1000:.3f}s"
            if self.gap_behind_ms is not None
            else "—"
        )
        compound = self.tyre_compound or "unknown"
        age = (
            f"{self.tyre_life} laps (set)"
            if self.tyre_life is not None
            else f"{self.stint_age_laps} laps"
        )
        if self.sc_active:
            sc_line = f"Yes (deployed {self.laps_since_sc_deploy} laps ago)"
        else:
            sc_line = "No"
        # Circuit stays even when anonymized — pit-loss / undercut timing depends on it.
        circuit = self.circuit_name or f"circuit_{self.circuit_id}"
        header: list[str] = []
        if anonymize:
            header.append("Race: (identity withheld)")
            header.append("Driver: (identity withheld)")
        else:
            code = self.driver_code or self.driver_ref
            header.append(f"Race: {self.race_name} ({self.year})")
            header.append(f"Driver: {code}")
        header.append(f"Circuit: {circuit}")
        body = (
            f"Lap: {self.lap} / {self.total_laps}\n"
            f"Current Position: P{self.position}\n"
            f"Gap Ahead: {gap_ahead}\n"
            f"Gap Behind: {gap_behind}\n"
            f"Tyres:\n"
            f"  Compound: {compound}\n"
            f"  Age: {age}\n"
            f"Safety Car: {sc_line}\n"
            f"Pit this lap: {'Yes' if self.pit_this_lap else 'No'}\n"
            f"Pit stops so far: {self.pit_count}\n"
            f"Last lap: {self.last_lap_time_ms} ms\n"
            f"Cars on track: {self.drivers_on_track}"
        )
        return "\n".join(header) + "\n" + body
