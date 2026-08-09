"""Phase 9: Race Engineer radio LLM (presentation only).

Takes a Strategy decision + board state and returns driver_message.
Must never change action / tyre / push / rationale or write memory.

Heuristic / offline fallback lives in ``radio_heuristic.py``.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol, Sequence

from race_engineer.radio_heuristic import (
    HeuristicRadioBackend,
    PassthroughRadioBackend,
)
from race_engineer.radio_log import (
    RadioCall,
    format_recent_calls,
    is_repetitive,
)
from race_engineer.situation import RadioSituation, build_radio_situation
from shared.decision import (
    Action,
    CrewChiefDecision,
    PushLevel,
    TyreChoice,
)
from shared.messages import RadioResult
from shared.state import RaceState

# Minimal style guidance; the model already knows modern F1 radio conventions.
RADIO_STYLE_GUIDE = """
Sound like a modern F1 race engineer: brief, calm, operational, and aware of
the unfolding race rather than narrating a snapshot.

Use your racing knowledge to infer the useful message from objective context:
chasing or being chased, position change, a recent stop, an undercut/cover
sequence, tyre management, or late-race defense. Mention at most one main idea.

"Stay out" is a pit-window instruction, not a greeting. Use it only when a stop
is genuinely live from the context (for example safety car, recent rival stop,
old tyres, or an active strategy window). Otherwise give the useful status or
instruction directly.

You are mid-conversation, not starting fresh each lap. Radio you already sent
is listed for you. The driver has heard it, so do not send it again in new
words — say what changed since then, or say less.

Speak gaps naturally ("eight tenths", "one point two"). Do not invent facts.
"""

RADIO_SYSTEM_PROMPT = """You are the F1 race engineer on team radio.
Strategy has already decided pit/stay, tyre, and push. Your only job is the
driver radio call. Use your knowledge of modern F1 strategy and radio.

Reply with ONLY a JSON object (no markdown):
{
  "driver_message": "<radio to driver, ≤120 chars>"
}

Rules:
- Reflect the given action / tyre / push. Do not invent a different plan.
- Ground every factual claim in the board or recent context.
- Infer what matters; do not mechanically repeat every supplied fact.
- Keep it to one short radio transmission (≤120 chars).
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
    situation: RadioSituation | None = None,
    recent_calls: Sequence[RadioCall] = (),
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
    sit = situation or RadioSituation()
    block = sit.prompt_block()
    if block:
        lines.extend(["", block])
    recent = format_recent_calls(recent_calls)
    if recent:
        lines.extend(["", recent])
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
        *,
        replay: Any | None = None,
        recent_calls: Sequence[RadioCall] = (),
    ) -> str: ...


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
        *,
        replay: Any | None = None,
        recent_calls: Sequence[RadioCall] = (),
    ) -> str:
        situation = build_radio_situation(state, replay)
        user = build_radio_user_prompt(
            state,
            action=decision.action,
            tyre=decision.tyre,
            push=decision.push,
            reason=decision.reason,
            situation=situation,
            recent_calls=recent_calls,
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": RADIO_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        recent = [c.message for c in recent_calls]
        repeat: str | None = None
        for _ in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                raw = resp.choices[0].message.content or ""
                msg = parse_radio_message(raw)
            except Exception as exc:  # noqa: BLE001
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
                continue
            # One nudge only; a second near-duplicate is better than no call.
            if repeat is not None or not is_repetitive(msg, recent):
                return msg
            repeat = msg
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": json.dumps({"driver_message": msg}),
                    },
                    {
                        "role": "user",
                        "content": (
                            "That is the call you already made. Send only what "
                            "has changed since then, or a shorter call."
                        ),
                    },
                ]
            )
        if repeat is not None:
            return repeat
        return self.fallback.compose(
            state, decision, replay=replay, recent_calls=recent_calls
        )


def apply_radio(
    state: RaceState,
    decision: CrewChiefDecision,
    radio: RadioBackend,
    *,
    replay: Any | None = None,
    recent_calls: Sequence[RadioCall] = (),
) -> tuple[CrewChiefDecision, RadioResult]:
    """Overwrite driver_message only; strategy fields stay identical."""
    msg = radio.compose(
        state, decision, replay=replay, recent_calls=recent_calls
    ).strip()[:120]
    if not msg:
        msg = HeuristicRadioBackend().compose(state, decision, replay=replay)
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
    raw = (name or os.environ.get("RACE_ENGINEER_RADIO", "auto")).strip().lower()
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
