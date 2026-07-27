"""LLM backends for the Phase 2 crew chief."""

from __future__ import annotations

import os
from typing import Protocol

from race_engineer.crew_chief import (
    SYSTEM_PROMPT,
    CrewChiefDecision,
    build_user_prompt,
    compose_driver_message,
    compose_reason,
    finalize_decision,
    parse_decision,
)
from race_engineer.state import RaceState


class CrewChiefBackend(Protocol):
    name: str

    def decide(self, state: RaceState) -> CrewChiefDecision: ...


class HeuristicBackend:
    """
    Offline stand-in (no API key).

    Rough pit-wall rules for schema-valid decisions so eval/CI can run
    without an LLM. Not a substitute for the OpenAI backend.
    """

    name = "heuristic"

    def decide(self, state: RaceState) -> CrewChiefDecision:
        age = state.tyre_life if state.tyre_life is not None else state.stint_age_laps
        compound = (state.tyre_compound or "").upper()
        remaining = state.total_laps - state.lap
        sc = bool(getattr(state, "sc_active", False))
        sc_deploy = bool(getattr(state, "sc_deployed_this_lap", False)) or bool(
            getattr(state, "sc_deployed_prev_lap", False)
        )

        pit = False
        tyre = None
        push = "med"
        reasons: list[str] = []

        if sc_deploy or (sc and age >= 8):
            pit = True
            reasons.append("Safety car window makes a pit stop relatively cheap")
        elif age >= 25 and remaining > 8:
            pit = True
            reasons.append(f"Tyre age is high ({age} laps) with race still left")
        elif compound in {"SOFT", "SUPERSOFT", "ULTRASOFT", "HYPERSOFT"} and age >= 15:
            pit = True
            reasons.append(f"Soft compound at {age} laps is typically past the cliff")
        elif remaining <= 3:
            pit = False
            reasons.append("Too few laps remaining to benefit from a stop")
        else:
            pit = False
            reasons.append("Current stint still viable; stay out")

        if pit:
            if remaining > 25:
                tyre = "hard"
            elif remaining > 12:
                tyre = "medium"
            else:
                tyre = "soft"
            push = "low"
        else:
            if state.gap_ahead_ms is not None and state.gap_ahead_ms < 800:
                push = "high"
                reasons.append("Close gap ahead — push to apply pressure")
            else:
                push = "med"

        rationale = "; ".join(reasons) + "."
        action = "pit" if pit else "stay"
        draft = CrewChiefDecision(
            action=action,  # type: ignore[arg-type]
            tyre=tyre,  # type: ignore[arg-type]
            push=push,  # type: ignore[arg-type]
            reason=compose_reason(
                action=action,  # type: ignore[arg-type]
                tyre=tyre,
                push=push,  # type: ignore[arg-type]
                rationale=rationale,
            ),
            driver_message=compose_driver_message(
                state,
                action=action,  # type: ignore[arg-type]
                tyre=tyre,
                push=push,  # type: ignore[arg-type]
            ),
            rationale=rationale,
        )
        return finalize_decision(state, draft)


class OpenAIBackend:
    """OpenAI Chat Completions with JSON object response format."""

    name = "openai"

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 2,
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
        kwargs = {"api_key": key}
        url = base_url or os.environ.get("OPENAI_BASE_URL")
        if url:
            kwargs["base_url"] = url
        self._client = OpenAI(**kwargs)
        self.model = model or os.environ.get("CREW_CHIEF_MODEL", "gpt-4o-mini")
        self.max_retries = max_retries

    def decide(self, state: RaceState) -> CrewChiefDecision:
        user = build_user_prompt(state)
        last_err: Exception | None = None
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ]
        for _ in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                raw = resp.choices[0].message.content or ""
                return parse_decision(raw, state=state)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Previous reply was invalid ({exc}). "
                            "Return ONLY valid JSON matching the schema."
                        ),
                    }
                )
        assert last_err is not None
        raise last_err


def get_backend(name: str = "auto") -> CrewChiefBackend:
    if name == "heuristic":
        return HeuristicBackend()
    if name == "openai":
        return OpenAIBackend()
    if name == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAIBackend()
        return HeuristicBackend()
    raise ValueError(f"unknown backend: {name}")
