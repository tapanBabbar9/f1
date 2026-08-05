"""Objective race context for Race Engineer radio (presentation only).

This module is a sensor, not a strategist. It exposes current/prior board facts
and recent pit events; the radio LLM decides what they mean and how to say it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from shared.state import RaceState

if TYPE_CHECKING:
    from shared.replay import RaceReplay


@dataclass(frozen=True)
class RadioSituation:
    """Objective facts available to the radio agent."""

    lines: tuple[str, ...] = ()

    def prompt_block(self) -> str:
        if not self.lines:
            return ""
        return "Recent race context (objective facts):\n" + "\n".join(
            f"- {line}" for line in self.lines
        )


def _fmt_gap_s(ms: int | None) -> str:
    if ms is None:
        return "unknown"
    return f"{ms / 1000.0:.1f}s"


def _neighbor_ids(
    replay: RaceReplay, state: RaceState
) -> tuple[int | None, int | None]:
    """Ahead / behind driver ids from cumulative board at this lap."""
    cumul = replay._cumulative_times(state.race_id, state.lap)
    if state.driver_id not in cumul:
        return None, None
    ordered = sorted(cumul.items(), key=lambda kv: kv[1])
    idx = next(
        (i for i, (did, _) in enumerate(ordered) if did == state.driver_id),
        None,
    )
    if idx is None:
        return None, None
    ahead = ordered[idx - 1][0] if idx > 0 else None
    behind = ordered[idx + 1][0] if idx + 1 < len(ordered) else None
    return ahead, behind


def _recent_pit_lap(
    replay: RaceReplay,
    race_id: int,
    driver_id: int | None,
    current_lap: int,
) -> int | None:
    if driver_id is None:
        return None
    recent = [
        lap
        for lap in replay.pit_laps(race_id, driver_id)
        if current_lap - 1 <= lap <= current_lap
    ]
    return max(recent) if recent else None


def build_radio_situation(
    state: RaceState,
    replay: RaceReplay | None = None,
) -> RadioSituation:
    """Build raw current/prior facts without interpreting race strategy."""
    lines: list[str] = []

    remaining = max(0, state.total_laps - state.lap)
    tyre_age = (
        state.tyre_life
        if state.tyre_life is not None
        else state.stint_age_laps
    )
    lines.append(
        f"{remaining} laps remain; current P{state.position}; "
        f"{state.tyre_compound or 'unknown'} tyres, age {tyre_age}; "
        f"{state.pit_count} stops completed."
    )
    lines.append(
        f"Current gaps: ahead {_fmt_gap_s(state.gap_ahead_ms)}, "
        f"behind {_fmt_gap_s(state.gap_behind_ms)}."
    )
    if state.sc_active:
        lines.append(
            f"Safety car active, deployed {state.laps_since_sc_deploy} laps ago."
        )

    prev: RaceState | None = None
    if replay is not None and state.lap > 1:
        try:
            prev = replay.get_state(state.race_id, state.driver_id, state.lap - 1)
        except KeyError:
            prev = None

    if prev is not None:
        lines.append(
            f"Previous lap: P{prev.position}; gaps ahead "
            f"{_fmt_gap_s(prev.gap_ahead_ms)}, behind "
            f"{_fmt_gap_s(prev.gap_behind_ms)}."
        )
        if state.position != prev.position:
            lines.append(
                f"Position changed P{prev.position} -> P{state.position}."
            )

    if state.pit_this_lap:
        lines.append("Driver pitted on the current completed lap.")
    elif prev is not None and prev.pit_this_lap:
        lines.append("Driver pitted on the previous lap.")

    if replay is not None:
        ahead_id, behind_id = _neighbor_ids(replay, state)
        ahead_pit = _recent_pit_lap(
            replay, state.race_id, ahead_id, state.lap
        )
        behind_pit = _recent_pit_lap(
            replay, state.race_id, behind_id, state.lap
        )
        if ahead_pit is not None:
            lines.append(f"Current nearest car ahead pitted on lap {ahead_pit}.")
        if behind_pit is not None:
            lines.append(
                f"Current nearest car behind pitted on lap {behind_pit}."
            )

    return RadioSituation(lines=tuple(lines))
