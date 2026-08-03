"""Shared decision schema for Strategy / Race Engineer agents."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from shared.state import RaceState

Action = Literal["pit", "stay"]
TyreChoice = Literal["soft", "medium", "hard"] | None
PushLevel = Literal["low", "med", "high"]

# Pit decision at lap L is post-lap; target is L+1. "Box this lap" = that upcoming lap.
UPCOMING_LAP_DECISION = (
    "Given the current race state, decide whether the driver should pit on the "
    "upcoming lap or stay out. The board shows the lap just completed (L); the "
    "upcoming lap is L+1. A pit call targets that upcoming lap, not the lap "
    "already finished."
)

UPCOMING_LAP_USER_QUESTION = (
    "Decide: pit on the upcoming lap, or stay out?"
)


@dataclass(frozen=True)
class CrewChiefDecision:
    action: Action
    tyre: TyreChoice
    push: PushLevel
    reason: str
    driver_message: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def pit_next(self) -> int:
        return 1 if self.action == "pit" else 0


def _fmt_gap_seconds(gap_ms: int) -> str:
    """Radio-friendly gap: 'eight tenths' or 'one point six'."""
    sec = gap_ms / 1000.0
    if sec < 1.05:
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
            10: "ten",
        }
        word = ones.get(tenths, str(tenths))
        return f"{word} tenth" if tenths == 1 else f"{word} tenths"
    whole = int(sec)
    frac = round((sec - whole) * 10)
    if frac == 0:
        return "one second" if whole == 1 else f"{whole} seconds"
    return f"{whole} point {frac}"


def compose_reason(
    *,
    action: Action,
    tyre: TyreChoice,
    push: PushLevel,
    rationale: str,
) -> str:
    """One-line pit-wall reason from rationale or action."""
    chunk = rationale.split(";")[0].split(".")[0].strip()
    if chunk:
        return chunk[:100]
    if action == "pit":
        compound = tyre or "medium"
        return f"Box this lap on {compound}."
    if push == "high":
        return "Stay out and push on pace."
    if push == "low":
        return "Stay out and manage the tyre."
    return "Stay out this lap."


def compose_driver_message(
    state: RaceState | None,
    *,
    action: Action,
    tyre: TyreChoice,
    push: PushLevel,
) -> str:
    """Short engineer radio call (box / stay / gap / push / manage)."""
    if action == "pit":
        compound = tyre or "medium"
        if state and state.sc_active:
            return f"Box box, safety car window, {compound}."
        return f"Box box, box this lap, {compound}."

    parts = ["Stay out, stay out"]
    if state:
        if push == "high" and state.gap_ahead_ms is not None:
            parts.append(f"Push, gap ahead {_fmt_gap_seconds(state.gap_ahead_ms)}")
        elif push == "high":
            parts.append("Push now")
        elif state.gap_behind_ms is not None and state.gap_behind_ms < 1500:
            parts.append(f"Car behind {_fmt_gap_seconds(state.gap_behind_ms)}")
        elif state.gap_ahead_ms is not None:
            parts.append(f"Gap ahead {_fmt_gap_seconds(state.gap_ahead_ms)}")
        age = state.tyre_life if state.tyre_life is not None else state.stint_age_laps
        if age >= 18:
            parts.append("manage the tyre")
        elif push == "low":
            parts.append("manage temps")
    elif push == "high":
        parts.append("Push")
    elif push == "low":
        parts.append("manage the tyre")

    msg = ". ".join(parts)
    if not msg.endswith("."):
        msg += "."
    return msg[:120]


def finalize_decision(
    state: RaceState | None, decision: CrewChiefDecision
) -> CrewChiefDecision:
    """Fill reason / driver_message when the model omitted them."""
    reason = decision.reason.strip() or compose_reason(
        action=decision.action,
        tyre=decision.tyre,
        push=decision.push,
        rationale=decision.rationale,
    )
    driver_message = decision.driver_message.strip() or compose_driver_message(
        state,
        action=decision.action,
        tyre=decision.tyre,
        push=decision.push,
    )
    if reason == decision.reason and driver_message == decision.driver_message:
        return decision
    return CrewChiefDecision(
        action=decision.action,
        tyre=decision.tyre,
        push=decision.push,
        reason=reason,
        driver_message=driver_message,
        rationale=decision.rationale,
    )


def build_user_prompt(state: RaceState) -> str:
    """LLM-facing board: anonymized to reduce historical-result leakage."""
    return (
        "Race state (pit-wall feed):\n\n"
        f"{state.pit_wall_view(anonymize=True)}\n\n"
        f"{UPCOMING_LAP_USER_QUESTION}"
    )


def _normalize_tyre(value: Any) -> TyreChoice:
    if value is None or value == "" or str(value).lower() in {"null", "none"}:
        return None
    t = str(value).strip().lower()
    # Collapse legacy soft names if the model emits them.
    if t in {"soft", "supersoft", "ultrasoft", "hypersoft"}:
        return "soft"
    if t in {"medium", "hard"}:
        return t  # type: ignore[return-value]
    raise ValueError(f"invalid tyre: {value!r}")


def _normalize_action(value: Any) -> Action:
    a = str(value).strip().lower()
    if a in {"pit", "stay"}:
        return a  # type: ignore[return-value]
    raise ValueError(f"invalid action: {value!r}")


def _normalize_push(value: Any) -> PushLevel:
    p = str(value).strip().lower()
    if p in {"low", "med", "medium", "high"}:
        return "med" if p == "medium" else p  # type: ignore[return-value]
    raise ValueError(f"invalid push: {value!r}")


def parse_decision(raw: str, *, state: RaceState | None = None) -> CrewChiefDecision:
    """Parse model text into a validated CrewChiefDecision."""
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("decision must be a JSON object")

    action = _normalize_action(data.get("action"))
    tyre = _normalize_tyre(data.get("tyre"))
    push = _normalize_push(data.get("push"))
    rationale = str(data.get("rationale") or "").strip()
    if not rationale:
        raise ValueError("rationale is required")
    reason = str(data.get("reason") or data.get("one_line_reason") or "").strip()
    driver_message = str(
        data.get("driver_message") or data.get("radio") or ""
    ).strip()

    if action == "stay" and tyre is not None:
        # Soft-correct common model slip: stay with a tyre suggestion.
        tyre = None
    if action == "pit" and tyre is None:
        raise ValueError("pit action requires a tyre recommendation")

    draft = CrewChiefDecision(
        action=action,
        tyre=tyre,
        push=push,
        reason=reason,
        driver_message=driver_message,
        rationale=rationale,
    )
    return finalize_decision(state, draft)


def validate_decision_dict(data: dict[str, Any]) -> CrewChiefDecision:
    return parse_decision(json.dumps(data))
