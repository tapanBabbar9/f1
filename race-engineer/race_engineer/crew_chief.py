from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from race_engineer.racing_rules import DRY_MANDATORY_PIT_RULES
from race_engineer.state import RaceState

Action = Literal["pit", "stay"]
TyreChoice = Literal["soft", "medium", "hard"] | None
PushLevel = Literal["low", "med", "high"]

# Pit decision at lap L is post-lap; target is L+1. Radio "box this lap" = that upcoming lap.
UPCOMING_LAP_DECISION = (
    "Given the current race state, decide whether the driver should pit on the "
    "upcoming lap or stay out. The board shows the lap just completed (L); the "
    "upcoming lap is L+1. On team radio a pit call is \"box this lap\" — that "
    "is the upcoming lap, not the lap already finished."
)

UPCOMING_LAP_USER_QUESTION = (
    'Decide: pit on the upcoming lap (radio: "box this lap"), or stay out?'
)

# Authentic pit-wall radio patterns (gap, box, push, manage) — no driver names.
RADIO_STYLE_GUIDE = """
Radio style for driver_message (≤120 chars, engineer-to-driver):
- Pit: "Box, box." / "Box this lap, hard." / "Box this lap, medium." ("this lap" = upcoming lap, board lap+1).
- Stay: "Stay out." / "Stay out, push." / "Stay out, gap behind one point six."
- Gap ahead: "Push, gap ahead eight tenths." / "Gap ahead one point two." / "Push now."
- Gap behind / undercut: "Stay out, car behind one point six." / "Push, undercut threat."
- Tyres: "Manage the tyres." / "Tyres are hot." / "Fronts are overheating." / "Lift and coast."
- Safety Car window: "Box, safety car window, medium." / "Stay out under safety car."

Use real gap values from the board, spoken naturally:
- 0.8 → "eight tenths"
- 1.2 → "one point two"
- 2.0 → "two seconds"

Keep messages short, calm, and operational. No driver or team names. No explanations.
"""

SYSTEM_PROMPT = """You are an F1 race engineer on the pit wall.
""" + UPCOMING_LAP_DECISION + """

The feed withholds race event, year, and driver identity on purpose. Circuit is kept
because pit-loss and strategy windows depend on the track. Decide from the board shown;
do not guess which real-world race weekend or driver this is.

Reply with ONLY a JSON object (no markdown) matching this schema:
{
  "action": "pit" | "stay",
  "tyre": "soft" | "medium" | "hard" | null,
  "push": "low" | "med" | "high",
  "reason": "<one line: primary pit-wall justification, ≤100 chars>",
  "driver_message": "<radio call to the driver, ≤120 chars>",
  "rationale": "<1-3 sentences citing the race state (pit-wall log)>"
}

Rules:
- If action is "stay", tyre must be null.
- If action is "pit", tyre must be soft, medium, or hard (recommended compound for the stop).
- reason = single concise line; rationale = fuller explanation with numbers from the board.
- driver_message = what you say over the team radio (box/stay/push/gap/tyre).
- Do not invent numbers that are not in the race state.
- Do not name drivers, teams, or race events.

""" + RADIO_STYLE_GUIDE + "\n" + DRY_MANDATORY_PIT_RULES


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
