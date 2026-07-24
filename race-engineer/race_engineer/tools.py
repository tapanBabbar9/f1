"""Phase 3 tools: bound to a RaceState so the LLM never sees race/driver IDs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from statistics import median
from typing import Any, Callable

from race_engineer.replay import RaceReplay
from race_engineer.state import RaceState

# OpenAI-compatible tool schemas (no identity args — binding supplies context).
OPENAI_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_gaps",
            "description": (
                "Gap ahead / behind (seconds) and current position for the "
                "bound driver at the current lap."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stint_age",
            "description": (
                "Current stint age, tyre compound/life, and pit-stop count "
                "for the bound driver."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_remaining_laps",
            "description": "Current lap, total race laps, and laps remaining.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_circuit_undercut_stats",
            "description": (
                "Historical first-stop timing for this circuit (excludes the "
                "current race). Useful for typical pit windows / undercut timing."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


@dataclass
class ToolResult:
    name: str
    payload: dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(self.payload, sort_keys=True)


class ToolBelt:
    """Execute Phase 3 tools against a fixed (replay, state) binding."""

    def __init__(self, replay: RaceReplay, state: RaceState):
        self.replay = replay
        self.state = state
        self._handlers: dict[str, Callable[[], dict[str, Any]]] = {
            "get_gaps": self.get_gaps,
            "get_stint_age": self.get_stint_age,
            "get_remaining_laps": self.get_remaining_laps,
            "lookup_circuit_undercut_stats": self.lookup_circuit_undercut_stats,
        }

    def names(self) -> list[str]:
        return list(self._handlers.keys())

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        if name not in self._handlers:
            raise KeyError(f"unknown tool: {name}")
        # Arguments ignored — binding supplies identity (anti-leakage).
        _ = arguments
        return ToolResult(name=name, payload=self._handlers[name]())

    def get_gaps(self) -> dict[str, Any]:
        s = self.state
        return {
            "position": s.position,
            "gap_ahead_s": None
            if s.gap_ahead_ms is None
            else round(s.gap_ahead_ms / 1000.0, 3),
            "gap_behind_s": None
            if s.gap_behind_ms is None
            else round(s.gap_behind_ms / 1000.0, 3),
            "drivers_on_track": s.drivers_on_track,
        }

    def get_stint_age(self) -> dict[str, Any]:
        s = self.state
        return {
            "stint_age_laps": s.stint_age_laps,
            "tyre_compound": s.tyre_compound,
            "tyre_life": s.tyre_life,
            "pit_count": s.pit_count,
            "pit_this_lap": s.pit_this_lap,
            "sc_active": s.sc_active,
            "laps_since_sc_deploy": s.laps_since_sc_deploy,
        }

    def get_remaining_laps(self) -> dict[str, Any]:
        s = self.state
        remaining = max(0, s.total_laps - s.lap)
        return {
            "lap": s.lap,
            "total_laps": s.total_laps,
            "remaining_laps": remaining,
            "lap_fraction": round(s.lap / s.total_laps, 4) if s.total_laps else None,
        }

    def lookup_circuit_undercut_stats(self) -> dict[str, Any]:
        """Median first-stop lap fraction on this circuit (other races only)."""
        s = self.state
        circuit_id = s.circuit_id
        fractions: list[float] = []
        stops_per_driver: list[int] = []

        for rid, race in self.replay._races.items():
            if int(race["circuitId"]) != circuit_id:
                continue
            if rid == s.race_id:
                continue
            total = self.replay._total_laps.get(rid)
            if not total or total < 5:
                continue
            for did, pit_laps in self.replay._pits.get(rid, {}).items():
                if not pit_laps:
                    continue
                stops_per_driver.append(len(pit_laps))
                first = min(pit_laps)
                fractions.append(first / float(total))

        n = len(fractions)
        if n == 0:
            return {
                "circuit_id": circuit_id,
                "circuit_name": s.circuit_name,
                "n_first_stops": 0,
                "median_first_stop_lap_fraction": None,
                "median_stops_per_driver": None,
                "note": "insufficient historical pit data for this circuit",
            }
        return {
            "circuit_id": circuit_id,
            "circuit_name": s.circuit_name,
            "n_first_stops": n,
            "median_first_stop_lap_fraction": round(float(median(fractions)), 4),
            "median_stops_per_driver": round(float(median(stops_per_driver)), 2),
        }

    def call_all(self) -> list[ToolResult]:
        return [self.call(name) for name in self.names()]


def flatten_numbers(obj: Any) -> list[float]:
    """Collect finite floats from nested tool payloads."""
    out: list[float] = []
    if isinstance(obj, bool) or obj is None:
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
        return out
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(flatten_numbers(v))
        return out
    if isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(flatten_numbers(v))
        return out
    return out
