"""Phase 9: Race Engineer radio agent (presentation only).

Takes a Strategy decision + board state and returns driver_message.
Must never change action / tyre / push / rationale or write memory.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

from shared.decision import (
    Action,
    CrewChiefDecision,
    PushLevel,
    TyreChoice,
    compose_driver_message,
)
from shared.messages import RadioResult
from shared.state import RaceState

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
Vary phrasing lap to lap — same decision can use different radio wording.
"""

RADIO_SYSTEM_PROMPT = """You are the F1 race engineer on team radio.
Strategy has already decided pit/stay, tyre, and push. Your only job is the
driver radio call — concise, calm, operational.

Reply with ONLY a JSON object (no markdown):
{
  "driver_message": "<radio to driver, ≤120 chars>"
}

Rules:
- Reflect the given action / tyre / push. Do not invent a different plan.
- Do not name drivers, teams, or race events.
- No option letters, no sim dumps, no explanations.

""" + RADIO_STYLE_GUIDE


def build_radio_user_prompt(
    state: RaceState,
    *,
    action: Action,
    tyre: TyreChoice,
    push: PushLevel,
    reason: str = "",
) -> str:
    board = state.pit_wall_view(anonymize=True)
    tyre_s = tyre if tyre is not None else "null"
    lines = [
        "Race state (pit-wall feed):",
        "",
        board,
        "",
        "Strategy decision (binding — do not override):",
        f"  action: {action}",
        f"  tyre: {tyre_s}",
        f"  push: {push}",
    ]
    if reason.strip():
        lines.append(f"  reason: {reason.strip()[:100]}")
    lines.append("")
    lines.append("Compose driver_message only.")
    return "\n".join(lines)


def parse_radio_message(raw: str | dict[str, Any]) -> str:
    if isinstance(raw, dict):
        msg = str(raw.get("driver_message") or raw.get("radio") or "").strip()
    else:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("radio reply must be a JSON object")
        msg = str(data.get("driver_message") or data.get("radio") or "").strip()
    if not msg:
        raise ValueError("driver_message is required")
    return msg[:120]


class RadioBackend(Protocol):
    name: str

    def compose(
        self,
        state: RaceState,
        decision: CrewChiefDecision,
    ) -> str: ...


class HeuristicRadioBackend:
    """Deterministic radio from action / tyre / push + board gaps."""

    name = "heuristic_radio"

    def compose(
        self,
        state: RaceState,
        decision: CrewChiefDecision,
    ) -> str:
        return compose_driver_message(
            state,
            action=decision.action,
            tyre=decision.tyre,
            push=decision.push,
        )[:120]


class PassthroughRadioBackend:
    """A/B: keep pre-filled radio if present; else heuristic compose."""

    name = "passthrough"

    def compose(
        self,
        state: RaceState,
        decision: CrewChiefDecision,
    ) -> str:
        msg = (decision.driver_message or "").strip()
        if msg:
            return msg[:120]
        return HeuristicRadioBackend().compose(state, decision)


class OpenAIRadioBackend:
    """LLM radio-only call; never changes strategy fields."""

    name = "openai_radio"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
        fallback: RadioBackend | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "openai package required: pip install openai"
            ) from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        kwargs: dict[str, Any] = {"api_key": key}
        url = base_url or os.environ.get("OPENAI_BASE_URL")
        if url:
            kwargs["base_url"] = url
        if default_headers:
            kwargs["default_headers"] = default_headers
        self._client = OpenAI(**kwargs)
        self.model = model or os.environ.get(
            "RACE_ENGINEER_RADIO_MODEL",
            os.environ.get("CREW_CHIEF_MODEL", "gpt-4o-mini"),
        )
        self.max_retries = max_retries
        self.fallback = fallback or HeuristicRadioBackend()

    def compose(
        self,
        state: RaceState,
        decision: CrewChiefDecision,
    ) -> str:
        user = build_radio_user_prompt(
            state,
            action=decision.action,
            tyre=decision.tyre,
            push=decision.push,
            reason=decision.reason,
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": RADIO_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        last_err: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                raw = resp.choices[0].message.content or ""
                return parse_radio_message(raw)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Previous reply was invalid ({exc}). "
                            "Return ONLY "
                            '{"driver_message":"..."} matching the schema.'
                        ),
                    }
                )
        return self.fallback.compose(state, decision)


def apply_radio(
    state: RaceState,
    decision: CrewChiefDecision,
    radio: RadioBackend,
) -> tuple[CrewChiefDecision, RadioResult]:
    """Overwrite driver_message only; strategy fields stay identical."""
    msg = radio.compose(state, decision).strip()[:120]
    if not msg:
        msg = HeuristicRadioBackend().compose(state, decision)
    result = RadioResult(
        driver_message=msg,
        radio_backend=radio.name,
    )
    if msg == decision.driver_message:
        return decision, result
    rewritten = CrewChiefDecision(
        action=decision.action,
        tyre=decision.tyre,
        push=decision.push,
        reason=decision.reason,
        rationale=decision.rationale,
        driver_message=msg,
    )
    # Guard: radio must not mutate strategy fields.
    assert rewritten.action == decision.action
    assert rewritten.tyre == decision.tyre
    assert rewritten.push == decision.push
    assert rewritten.rationale == decision.rationale
    assert rewritten.reason == decision.reason
    return rewritten, result


def get_radio_backend(
    name: str | None = None,
    *,
    default_headers: dict[str, str] | None = None,
) -> RadioBackend:
    raw = (name or os.environ.get("RACE_ENGINEER_RADIO", "heuristic")).strip().lower()
    if raw in {"heuristic", "heuristic_radio", "compose"}:
        return HeuristicRadioBackend()
    if raw in {"passthrough", "none", "off", "strategy"}:
        return PassthroughRadioBackend()
    if raw in {"openai", "openai_radio", "llm"}:
        if not os.environ.get("OPENAI_API_KEY"):
            return HeuristicRadioBackend()
        return OpenAIRadioBackend(default_headers=default_headers)
    if raw == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAIRadioBackend(default_headers=default_headers)
        return HeuristicRadioBackend()
    raise ValueError(f"unknown radio backend: {name!r}")
