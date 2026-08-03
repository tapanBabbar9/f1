"""A2A orchestrator: Strategy owns the plan; Race Engineer reads it and speaks."""

from __future__ import annotations

from typing import Any

from race_engineer.radio import RadioBackend, apply_radio, get_radio_backend
from shared.decision import CrewChiefDecision
from shared.messages import RadioResult, brief_from_decision, plan_key_from_decision
from shared.state import RaceState
from strategy_engineer.memory import RaceMemoryStore
from strategy_engineer.strategy import (
    SimAgentDecision,
    SimAwareBackend,
    get_strategy_backend,
)


def apply_radio_decision(
    state: RaceState,
    decision: CrewChiefDecision,
    radio: RadioBackend | None = None,
) -> tuple[CrewChiefDecision, RadioResult]:
    """Hand a read-only Strategy plan to Race Engineer; assert plan unchanged."""
    before = plan_key_from_decision(decision)
    _ = brief_from_decision(decision)  # frozen A2A payload
    radio = radio or get_radio_backend()
    rewritten, result = apply_radio(state, decision, radio)
    after = plan_key_from_decision(rewritten)
    if after != before:
        raise RuntimeError(
            "Race Engineer must not mutate Strategy plan fields "
            f"(before={before!r} after={after!r})"
        )
    return rewritten, result


def attach_radio(
    state: RaceState,
    sd: SimAgentDecision,
    radio: RadioBackend,
) -> SimAgentDecision:
    """Post-pass after Strategy (+ memory) are final."""
    new_decision, radio_result = apply_radio_decision(state, sd.decision, radio)
    traj = sd.trajectory
    if traj is not None:
        traj = {
            **traj,
            "radio": {
                "backend": radio_result.radio_backend,
                "driver_message": radio_result.driver_message,
            },
        }
    return SimAgentDecision(
        decision=new_decision,
        chosen_option_id=sd.chosen_option_id,
        chosen_label=sd.chosen_label,
        chosen_mean_finish_pos=sd.chosen_mean_finish_pos,
        oracle_option_id=sd.oracle_option_id,
        oracle_label=sd.oracle_label,
        oracle_mean_finish_pos=sd.oracle_mean_finish_pos,
        regret=sd.regret,
        tool_results=sd.tool_results,
        faithfulness=sd.faithfulness,
        trajectory=traj,
        radio=radio_result,
    )


class MultiAgentSimBackend(SimAwareBackend):
    """
    A2A lap loop: Strategy decides (owns memory), then Race Engineer radio.

    Harness model_id stays the Strategy backend name.
    """

    def __init__(
        self,
        strategy: SimAwareBackend,
        radio: RadioBackend | None = None,
        *,
        default_headers: dict[str, str] | None = None,
    ):
        self.strategy = strategy
        self.radio = radio or get_radio_backend(default_headers=default_headers)
        self.replay = strategy.replay

    @property
    def name(self) -> str:
        return self.strategy.name

    @property
    def model(self) -> str | None:
        return getattr(self.strategy, "model", None)

    def decide_with_sims(
        self,
        state: RaceState,
        *,
        memory: RaceMemoryStore | None = None,
    ) -> SimAgentDecision:
        sd = self.strategy.decide_with_sims(state, memory=memory)
        return attach_radio(state, sd, self.radio)


def get_sim_backend(
    name: str,
    replay: Any,
    *,
    radio: RadioBackend | str | None = None,
    multi_agent: bool = True,
    default_headers: dict[str, str] | None = None,
) -> SimAwareBackend:
    """Strategy backend, optionally wrapped with Race Engineer radio (default on)."""
    strategy = get_strategy_backend(
        name, replay, default_headers=default_headers
    )
    if not multi_agent:
        return strategy
    if isinstance(radio, str) or radio is None:
        radio_backend = get_radio_backend(
            radio if isinstance(radio, str) else None,
            default_headers=default_headers,
        )
    else:
        radio_backend = radio
    return MultiAgentSimBackend(
        strategy,
        radio_backend,
        default_headers=default_headers,
    )
