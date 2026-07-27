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

SYSTEM_PROMPT = """You are an F1 race engineer on the pit wall.
Given the current race state, decide whether the driver should pit on the NEXT lap or stay out.

The feed withholds race event, year, and driver identity on purpose. Circuit is kept
because pit-loss and strategy windows depend on the track. Decide from the board shown;
do not guess which real-world race weekend or driver this is.

Reply with ONLY a JSON object (no markdown) matching this schema:
{
  "action": "pit" | "stay",
  "tyre": "soft" | "medium" | "hard" | null,
  "push": "low" | "med" | "high",
  "rationale": "<short explanation citing the race state>"
}

Rules:
- If action is "stay", tyre must be null.
- If action is "pit", tyre must be soft, medium, or hard (recommended compound for the stop).
- Keep rationale to 1-3 sentences. Do not invent numbers that are not in the race state.
- Do not name drivers, teams, or race events in the rationale.

""" + DRY_MANDATORY_PIT_RULES


@dataclass(frozen=True)
class CrewChiefDecision:
    action: Action
    tyre: TyreChoice
    push: PushLevel
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def pit_next(self) -> int:
        return 1 if self.action == "pit" else 0


def build_user_prompt(state: RaceState) -> str:
    """LLM-facing board: anonymized to reduce historical-result leakage."""
    return (
        "Race state (pit-wall feed):\n\n"
        f"{state.pit_wall_view(anonymize=True)}\n\n"
        "Decide: pit on the NEXT lap, or stay out?"
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


def parse_decision(raw: str) -> CrewChiefDecision:
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

    if action == "stay" and tyre is not None:
        # Soft-correct common model slip: stay with a tyre suggestion.
        tyre = None
    if action == "pit" and tyre is None:
        raise ValueError("pit action requires a tyre recommendation")

    return CrewChiefDecision(
        action=action, tyre=tyre, push=push, rationale=rationale
    )


def validate_decision_dict(data: dict[str, Any]) -> CrewChiefDecision:
    return parse_decision(json.dumps(data))
