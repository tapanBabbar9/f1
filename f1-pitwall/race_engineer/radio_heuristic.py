"""Offline / fallback radio compose (no LLM).

Keep this thin — situational interpretation belongs to the radio LLM.
"""

from __future__ import annotations

from typing import Any

from race_engineer.situation import RadioSituation, build_radio_situation
from shared.decision import (
    Action,
    CrewChiefDecision,
    PushLevel,
    TyreChoice,
)
from shared.state import RaceState


def _gap_words(gap_ms: int) -> str:
    """Radio-friendly gap: 'eight tenths' or 'one point six'."""
    sec = abs(gap_ms) / 1000.0
    if sec < 0.95:
        tenths = max(1, round(sec * 10))
        ones = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six",
            7: "seven",
            8: "eight",
            9: "nine",
        }
        word = ones.get(tenths, str(tenths))
        return f"{word} tenth" if tenths == 1 else f"{word} tenths"
    whole = int(sec)
    frac = int(round((sec - whole) * 10))
    if frac >= 10:
        whole += 1
        frac = 0
    ones_whole = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
        13: "thirteen",
        14: "fourteen",
        15: "fifteen",
    }
    whole_w = ones_whole.get(whole, str(whole))
    if frac <= 0:
        return "one second" if whole == 1 else f"{whole_w} seconds"
    frac_w = ones_whole.get(frac, str(frac))
    return f"{whole_w} point {frac_w}"


def compose_situation_radio(
    state: RaceState,
    *,
    action: Action,
    tyre: TyreChoice,
    push: PushLevel,
    situation: RadioSituation | None = None,
) -> str:
    """Minimal offline fallback; the LLM owns situational interpretation."""
    _ = situation
    if action == "pit":
        compound = tyre or "medium"
        if state.sc_active:
            return f"Box box, safety car window, {compound}."[:120]
        return f"Box box, box this lap, {compound}."[:120]

    if state.sc_active:
        return "Stay out under safety car."[:120]
    if push == "high":
        if state.gap_ahead_ms is not None:
            return f"Push now, gap ahead {_gap_words(state.gap_ahead_ms)}."[:120]
        return "Push now."[:120]
    if push == "low":
        return "Manage the tyres."[:120]
    if state.gap_behind_ms is not None and state.gap_behind_ms < 1_500:
        return f"Car behind {_gap_words(state.gap_behind_ms)}."[:120]
    return "Keep this pace."[:120]


class HeuristicRadioBackend:
    """Minimal deterministic fallback when no radio LLM is available."""

    name = "heuristic_radio"

    def compose(
        self,
        state: RaceState,
        decision: CrewChiefDecision,
        *,
        replay: Any | None = None,
    ) -> str:
        situation = build_radio_situation(state, replay)
        return compose_situation_radio(
            state,
            action=decision.action,
            tyre=decision.tyre,
            push=decision.push,
            situation=situation,
        )[:120]


class PassthroughRadioBackend:
    """A/B: keep pre-filled radio if present; else heuristic compose."""

    name = "passthrough"

    def compose(
        self,
        state: RaceState,
        decision: CrewChiefDecision,
        *,
        replay: Any | None = None,
    ) -> str:
        msg = (decision.driver_message or "").strip()
        if msg:
            return msg[:120]
        return HeuristicRadioBackend().compose(state, decision, replay=replay)
