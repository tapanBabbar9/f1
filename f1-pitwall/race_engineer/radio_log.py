"""Short-term memory of what the race engineer already said.

The radio agent is otherwise stateless per lap, so whenever the board looks
similar it re-sends the same call. Keeping the last few transmissions lets the
agent see its own side of the conversation instead of being told, rule by rule,
what not to say in each situation.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

_WORD_RE = re.compile(r"[a-z0-9]+")

DEFAULT_RECALL = 4
DEFAULT_SIMILARITY = 0.6


@dataclass(frozen=True)
class RadioCall:
    lap: int
    message: str


def _tokens(message: str) -> frozenset[str]:
    return frozenset(_WORD_RE.findall(message.lower()))


def similarity(a: str, b: str) -> float:
    """Word-overlap of two calls; 1.0 means the same words in any order."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_repetitive(
    message: str,
    recent: Iterable[str],
    *,
    threshold: float = DEFAULT_SIMILARITY,
) -> bool:
    """True when the call says what a recent one already said."""
    return any(similarity(message, prev) >= threshold for prev in recent)


def format_recent_calls(calls: Sequence[RadioCall]) -> str:
    if not calls:
        return ""
    lines = [f"- lap {c.lap}: {c.message}" for c in calls]
    return "Radio you already sent (oldest first):\n" + "\n".join(lines)


class RadioLog:
    """Per-driver ring buffer of recent transmissions."""

    def __init__(self, *, recall: int = DEFAULT_RECALL):
        self.recall = recall
        self._calls: dict[tuple[int, int], deque[RadioCall]] = {}

    def calls(self, race_id: int, driver_id: int) -> tuple[RadioCall, ...]:
        return tuple(self._calls.get((race_id, driver_id), ()))

    def messages(self, race_id: int, driver_id: int) -> tuple[str, ...]:
        return tuple(c.message for c in self.calls(race_id, driver_id))

    def record(
        self, race_id: int, driver_id: int, lap: int, message: str
    ) -> None:
        msg = (message or "").strip()
        if not msg:
            return
        buf = self._calls.setdefault(
            (race_id, driver_id), deque(maxlen=self.recall)
        )
        buf.append(RadioCall(lap=lap, message=msg))
