"""Phase 7: lap-to-lap pit-wall memory keyed by (race, driver)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from shared.state import RaceState

# Coarse gap buckets for evidence fingerprint (ms).
_GAP_BUCKET_MS = 500
# Flag large E[finish] moves when evidence fingerprint is unchanged.
DEFAULT_FINISH_SWING_THRESHOLD = 5.0


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
    planned_pit_lap: int | None = None

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
            "planned_pit_lap": self.planned_pit_lap,
        }

    def _plan_meta(self) -> str:
        """Plan as an absolute lap: a label alone cannot show the stop sliding."""
        if self.planned_pit_lap is not None:
            return f"(option {self.chosen_option_id}, target stop L{self.planned_pit_lap})"
        return f"(option {self.chosen_option_id}, {self.chosen_label})"

    def _head(self, *, pit_laps: frozenset[int] | None = None) -> str:
        upcoming = self.lap + 1
        if pit_laps is None:
            return f"{self.action}"

        pitted = upcoming in pit_laps
        if self.action == "pit":
            if pitted:
                return f"pit (executed lap {upcoming})"
            return f"advised box lap {upcoming}; driver stayed out"
        if self.action == "stay" and pitted:
            return f"advised stay; driver pitted lap {upcoming}"
        return "stay"

    def prompt_line(self, *, pit_laps: frozenset[int] | None = None) -> str:
        """One line for the LLM memory block (plan only — no stale sim numbers)."""
        return f"L{self.lap}: {self._head(pit_laps=pit_laps)} {self._plan_meta()}"

    def one_line(self) -> str:
        return self.prompt_line()


def _execution_mismatch(
    entry: MemoryEntry, pit_laps: frozenset[int]
) -> bool:
    """Advice on lap L vs whether the driver pitted on lap L+1."""
    pitted = (entry.lap + 1) in pit_laps
    if entry.action == "pit":
        return not pitted
    return pitted


def _memory_group_key(
    entry: MemoryEntry, pit_laps: frozenset[int] | None
) -> tuple[str, str, str, bool, int | None]:
    mismatch = False
    if pit_laps is not None:
        mismatch = _execution_mismatch(entry, pit_laps)
    # planned_pit_lap is part of the key so a drifting target breaks the run of
    # laps instead of collapsing into one line that reads as plan continuity.
    return (
        entry.action,
        entry.chosen_option_id,
        entry.chosen_label,
        mismatch,
        entry.planned_pit_lap,
    )


def _summarize_prompt_lines(
    entries: list[MemoryEntry],
    *,
    pit_laps: frozenset[int] | None,
) -> list[str]:
    """Merge consecutive laps with the same plan/evidence into one line."""
    if not entries:
        return []
    groups: list[
        tuple[tuple[str, str, str, bool, int | None], MemoryEntry, MemoryEntry]
    ] = []
    for entry in entries:
        key = _memory_group_key(entry, pit_laps)
        if groups and groups[-1][0] == key and groups[-1][2].lap + 1 == entry.lap:
            groups[-1] = (key, groups[-1][1], entry)
        else:
            groups.append((key, entry, entry))

    lines: list[str] = []
    for _key, start, end in groups:
        lap_part = f"L{start.lap}" if start.lap == end.lap else f"L{start.lap}-{end.lap}"
        lines.append(
            f"{lap_part}: {start._head(pit_laps=pit_laps)} {start._plan_meta()}"
        )
    return lines


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

    def committed_pit_lap(
        self,
        race_id: int,
        driver_id: int,
        *,
        current_lap: int,
        pit_laps: frozenset[int] | None = None,
    ) -> int | None:
        """Absolute stop lap already promised, if it is still ahead of us.

        Dropped once the target passes or the driver has since pitted, so a stale
        plan cannot pin the option menu to a lap that no longer means anything.
        """
        rows = [e for e in self.entries(race_id, driver_id) if e.lap < current_lap]
        for entry in reversed(rows):
            target = entry.planned_pit_lap
            if target is None:
                continue
            if target <= current_lap:
                return None
            if pit_laps is not None and any(
                entry.lap < p <= current_lap for p in pit_laps
            ):
                return None
            return target
        return None

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
        planned_pit_lap: int | None = None,
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
            planned_pit_lap=planned_pit_lap,
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
        lines = _summarize_prompt_lines(tail, pit_laps=pit_laps)
        return (
            "Pit-wall memory (your prior advice this race; plan history only — "
            "cite E_finish / P_finish from current simulate_strategies, not "
            "from these lines):\n"
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

    def finish_pos_swings(
        self,
        race_id: int,
        driver_id: int,
        *,
        threshold: float = DEFAULT_FINISH_SWING_THRESHOLD,
    ) -> list[tuple[MemoryEntry, MemoryEntry, float]]:
        """Consecutive laps with same evidence but large E[finish] move."""
        rows = self.entries(race_id, driver_id)
        out: list[tuple[MemoryEntry, MemoryEntry, float]] = []
        for prev, cur in zip(rows, rows[1:]):
            if prev.evidence != cur.evidence:
                continue
            if prev.mean_finish_pos is None or cur.mean_finish_pos is None:
                continue
            delta = abs(prev.mean_finish_pos - cur.mean_finish_pos)
            if delta >= threshold:
                out.append((prev, cur, delta))
        return out

    def finish_pos_swing_rate(
        self,
        race_id: int,
        driver_id: int,
        *,
        threshold: float = DEFAULT_FINISH_SWING_THRESHOLD,
    ) -> float | None:
        rows = self.entries(race_id, driver_id)
        if len(rows) < 2:
            return None
        swings = self.finish_pos_swings(
            race_id, driver_id, threshold=threshold
        )
        eligible = sum(
            1
            for prev, cur in zip(rows, rows[1:])
            if prev.evidence == cur.evidence
            and prev.mean_finish_pos is not None
            and cur.mean_finish_pos is not None
        )
        if eligible == 0:
            return None
        return len(swings) / eligible
