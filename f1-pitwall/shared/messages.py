"""A2A messages between Strategy and Race Engineer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared.decision import Action, CrewChiefDecision, PushLevel, TyreChoice


@dataclass(frozen=True)
class StrategyBrief:
    """Immutable plan owned by Strategy. Race Engineer may only read this."""

    action: Action
    tyre: TyreChoice
    push: PushLevel
    reason: str
    rationale: str
    chosen_option_id: str | None = None
    chosen_label: str | None = None

    def plan_key(self) -> tuple[Any, ...]:
        return (
            self.action,
            self.tyre,
            self.push,
            self.reason,
            self.rationale,
            self.chosen_option_id,
            self.chosen_label,
        )


@dataclass(frozen=True)
class RadioResult:
    driver_message: str
    radio_backend: str


def brief_from_decision(
    decision: CrewChiefDecision,
    *,
    chosen_option_id: str | None = None,
    chosen_label: str | None = None,
) -> StrategyBrief:
    return StrategyBrief(
        action=decision.action,
        tyre=decision.tyre,
        push=decision.push,
        reason=decision.reason,
        rationale=decision.rationale,
        chosen_option_id=chosen_option_id,
        chosen_label=chosen_label,
    )


def plan_key_from_decision(decision: CrewChiefDecision) -> tuple[Any, ...]:
    return (
        decision.action,
        decision.tyre,
        decision.push,
        decision.reason,
        decision.rationale,
    )
