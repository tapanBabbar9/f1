"""Phase 7: lap-to-lap pit-wall memory keyed by (race, driver)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from race_engineer.state import RaceState

# Coarse gap buckets for evidence fingerprint (ms).
_GAP_BUCKET_MS = 500


def _bucket_gap_ms(value: int | None) -> str:
    if value is None:
        return "none"
    sign = "p" if value >= 0 else "n"
    mag = abs(value)
    bucket = (mag // _GAP_BUCKET_MS) * _GAP_BUCKET_MS
    return f"{sign}{bucket}"


def evidence_fingerprint(
    state: RaceState,
    sim: dict[str, Any] | None = None,
) -> str:
    """Stable board + sim signature for flip-flop detection.

    Stint age is intentionally omitted — advancing one lap is not "new evidence".
    """
    oracle = "?"
    if sim and sim.get("available"):
        oracle = str(sim.get("oracle_option_id", "?"))
    compound = (state.tyre_compound or "UNK").strip().upper()
    return (
        f"P{state.position}|ga{_bucket_gap_ms(state.gap_ahead_ms)}|"
        f"gb{_bucket_gap_ms(state.gap_behind_ms)}|{compound}|"
        f"pc{state.pit_count}|sc{int(state.sc_active)}|o{oracle}"
    )


@dataclass
class MemoryEntry:
    lap: int
    action: str  # engineer advice: pit | stay
    chosen_option_id: str
    chosen_label: str
    oracle_option_id: str
    rationale: str
    evidence: str
    mean_finish_pos: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lap": self.lap,
            "action": self.action,
            "chosen_option_id": self.chosen_option_id,
            "chosen_label": self.chosen_label,
            "oracle_option_id": self.oracle_option_id,
            "rationale": self.rationale,
            "evidence": self.evidence,
            "mean_finish_pos": self.mean_finish_pos,
        }

    def _option_suffix(self) -> str:
        mf = (
            f", E[finish]={self.mean_finish_pos:.2f}"
            if self.mean_finish_pos is not None
            else ""
        )
        return (
            f"(option {self.chosen_option_id}, "
            f"{self.chosen_label}{mf})"
        )

    def prompt_line(self, *, pit_laps: frozenset[int] | None = None) -> str:
        """One line for the LLM memory block.

        When ``pit_laps`` is set, reconcile engineer advice with historical
        execution on the upcoming lap (lap+1). Trajectory / stored fields keep
        raw advice only.
        """
        meta = self._option_suffix()
        rationale = self.rationale[:120]
        upcoming = self.lap + 1

        if pit_laps is None:
            return f"L{self.lap}: {self.action} {meta} — {rationale}"

        pitted = upcoming in pit_laps
        if self.action == "pit":
            if pitted:
                head = f"L{self.lap}: pit (executed lap {upcoming})"
            else:
                head = (
                    f"L{self.lap}: advised box lap {upcoming}; driver stayed out"
                )
        elif self.action == "stay" and pitted:
            head = f"L{self.lap}: advised stay; driver pitted lap {upcoming}"
        else:
            head = f"L{self.lap}: stay"

        return f"{head} {meta} — {rationale}"

    def one_line(self) -> str:
        return self.prompt_line()


@dataclass
class RaceMemoryStore:
    """In-memory pit-wall log for one or more (race_id, driver_id) keys."""

    _entries: dict[tuple[int, int], list[MemoryEntry]] = field(
        default_factory=dict
    )

    def key(self, race_id: int, driver_id: int) -> tuple[int, int]:
        return (race_id, driver_id)

    def clear(self, race_id: int | None = None, driver_id: int | None = None) -> None:
        if race_id is None:
            self._entries.clear()
            return
        if driver_id is None:
            to_drop = [k for k in self._entries if k[0] == race_id]
            for k in to_drop:
                del self._entries[k]
            return
        self._entries.pop((race_id, driver_id), None)

    def entries(
        self, race_id: int, driver_id: int, *, before_lap: int | None = None
    ) -> list[MemoryEntry]:
        rows = list(self._entries.get((race_id, driver_id), ()))
        if before_lap is not None:
            rows = [e for e in rows if e.lap < before_lap]
        return rows

    def last_entry(
        self, race_id: int, driver_id: int
    ) -> MemoryEntry | None:
        rows = self._entries.get((race_id, driver_id))
        if not rows:
            return None
        return rows[-1]

    def record_sim_decision(
        self,
        state: RaceState,
        *,
        action: str,
        chosen_option_id: str,
        chosen_label: str,
        oracle_option_id: str,
        rationale: str,
        sim: dict[str, Any] | None,
        mean_finish_pos: float | None = None,
    ) -> MemoryEntry:
        entry = MemoryEntry(
            lap=state.lap,
            action=action,
            chosen_option_id=chosen_option_id,
            chosen_label=chosen_label,
            oracle_option_id=oracle_option_id,
            rationale=rationale,
            evidence=evidence_fingerprint(state, sim),
            mean_finish_pos=mean_finish_pos,
        )
        k = self.key(state.race_id, state.driver_id)
        self._entries.setdefault(k, []).append(entry)
        return entry

    def format_prompt_block(
        self,
        race_id: int,
        driver_id: int,
        *,
        before_lap: int,
        max_entries: int = 8,
        pit_laps: frozenset[int] | None = None,
    ) -> str:
        """Prior pit-wall instructions for the LLM user prompt."""
        prior = self.entries(race_id, driver_id, before_lap=before_lap)
        if not prior:
            return ""
        tail = prior[-max_entries:]
        lines = [e.prompt_line(pit_laps=pit_laps) for e in tail]
        return (
            "Pit-wall memory (your prior advice this race; notes when the driver "
            "did not follow a box call — stay consistent unless the board or sim "
            "cards materially change):\n"
            + "\n".join(f"- {ln}" for ln in lines)
        )

    @staticmethod
    def entry_advisory_mismatch(
        entry: MemoryEntry, pit_laps: frozenset[int]
    ) -> bool:
        """Engineer advised box on lap+1 but history shows the driver stayed out."""
        return entry.action == "pit" and (entry.lap + 1) not in pit_laps

    def flip_flops(
        self,
        race_id: int,
        driver_id: int,
        *,
        pit_laps: frozenset[int] | None = None,
        exclude_advisory_mismatch: bool = False,
    ) -> list[tuple[MemoryEntry, MemoryEntry]]:
        """Consecutive laps where advised action flipped but evidence unchanged."""
        rows = self.entries(race_id, driver_id)
        out: list[tuple[MemoryEntry, MemoryEntry]] = []
        for prev, cur in zip(rows, rows[1:]):
            if exclude_advisory_mismatch and pit_laps is not None:
                if self.entry_advisory_mismatch(prev, pit_laps):
                    continue
            if prev.action == cur.action:
                continue
            if prev.evidence == cur.evidence:
                out.append((prev, cur))
        return out

    def flip_flop_rate(
        self,
        race_id: int,
        driver_id: int,
        *,
        pit_laps: frozenset[int] | None = None,
        exclude_advisory_mismatch: bool = False,
    ) -> float | None:
        rows = self.entries(race_id, driver_id)
        if len(rows) < 2:
            return None
        flips = self.flip_flops(
            race_id,
            driver_id,
            pit_laps=pit_laps,
            exclude_advisory_mismatch=exclude_advisory_mismatch,
        )
        eligible = 0
        for prev, _cur in zip(rows, rows[1:]):
            if exclude_advisory_mismatch and pit_laps is not None:
                if self.entry_advisory_mismatch(prev, pit_laps):
                    continue
            eligible += 1
        if eligible == 0:
            return None
        return len(flips) / eligible
