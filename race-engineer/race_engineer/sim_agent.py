"""Phase 6: agent chooses among sim option cards and cites sim numbers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from race_engineer.crew_chief import (
    CrewChiefDecision,
    RADIO_STYLE_GUIDE,
    UPCOMING_LAP_DECISION,
    build_user_prompt,
    compose_driver_message,
    compose_reason,
    finalize_decision,
    parse_decision,
)
from race_engineer.faithfulness import faithfulness_score
from race_engineer.memory import RaceMemoryStore
from race_engineer.racing_rules import DRY_MANDATORY_PIT_RULES, mandatory_dry_pit_pending
from race_engineer.replay import RaceReplay
from race_engineer.sim import OptionCard, oracle_best
from race_engineer.state import RaceState
from race_engineer.tools import OPENAI_TOOL_SCHEMAS, ToolBelt, ToolResult

SIM_SYSTEM_PROMPT = """You are an F1 race engineer on the pit wall.
""" + UPCOMING_LAP_DECISION + """

You MUST call simulate_strategies (and may call other tools). Choose among the
returned option cards. Prefer lower expected finish position.

Reply with ONLY a JSON object (no markdown):
{
  "action": "pit" | "stay",
  "tyre": "soft" | "medium" | "hard" | null,
  "push": "low" | "med" | "high",
  "reason": "<one line, ≤100 chars>",
  "driver_message": "<radio to driver, ≤120 chars>",
  "chosen_option_id": "<option_id from simulate_strategies, e.g. A>",
  "rationale": "<1-3 sentences citing option E_finish / P_finish numbers>"
}

Rules:
- If action is "stay", tyre must be null.
- If action is "pit", tyre must be soft, medium, or hard.
- Map pit_next_lap → action pit (upcoming lap; radio: "box this lap"); any stay_* option → action stay.
- Cite at least one sim number (mean_finish_pos or P_finish_le_*).
- Do not name drivers, teams, or race events.
- The feed withholds race/year/driver; circuit is kept for pit-loss context.

""" + RADIO_STYLE_GUIDE + "\n" + DRY_MANDATORY_PIT_RULES

MEMORY_PROMPT_NOTE = """
When pit-wall memory is present, treat it as your prior advice. Notes like
"driver stayed out" reflect historical execution, not a retraction of your call.
Revise only when gaps, position, safety car, compound, or sim option rankings
materially change.
"""


def build_sim_user_prompt(
    state: RaceState,
    memory: RaceMemoryStore | None = None,
    *,
    replay: RaceReplay | None = None,
) -> str:
    base = build_user_prompt(state)
    if memory is None:
        return base
    pit_laps = replay.pit_laps(state.race_id, state.driver_id) if replay else None
    block = memory.format_prompt_block(
        state.race_id,
        state.driver_id,
        before_lap=state.lap,
        pit_laps=pit_laps,
    )
    if not block:
        return base
    return f"{base}\n\n{block}\n{MEMORY_PROMPT_NOTE.strip()}"


def _sim_payload_from_results(tool_results: list[ToolResult]) -> dict[str, Any] | None:
    for tr in reversed(tool_results):
        if tr.name == "simulate_strategies":
            return tr.payload
    return None


def _record_sim_decision(
    state: RaceState,
    sd: SimAgentDecision,
    memory: RaceMemoryStore | None,
    sim: dict[str, Any] | None,
) -> SimAgentDecision:
    if memory is not None:
        memory.record_sim_decision(
            state,
            action=sd.decision.action,
            chosen_option_id=sd.chosen_option_id,
            chosen_label=sd.chosen_label,
            oracle_option_id=sd.oracle_option_id,
            rationale=sd.decision.rationale,
            sim=sim,
            mean_finish_pos=sd.chosen_mean_finish_pos,
        )
    return sd


def option_to_action(label: str) -> str:
    return "pit" if label == "pit_next_lap" else "stay"


def option_to_tyre(label: str, *, current_compound: str | None = None) -> str | None:
    if label != "pit_next_lap":
        return None
    cur = (current_compound or "UNKNOWN").strip().upper()
    if cur == "MEDIUM":
        return "hard"
    if cur == "HARD":
        return "medium"
    if cur == "SOFT":
        return "hard"
    return "hard"


def find_option(
    options: list[dict[str, Any]], option_id: str
) -> dict[str, Any] | None:
    for o in options:
        if o.get("option_id") == option_id:
            return o
    return None


def position_regret(chosen_mean: float, oracle_mean: float) -> float:
    """E[finish|agent] − E[finish|oracle]; ≥0 when agent is worse or equal."""
    return float(chosen_mean - oracle_mean)


def parse_sim_decision(
    raw: str | dict[str, Any], *, state: RaceState | None = None
) -> tuple[CrewChiefDecision, str]:
    """Parse crew-chief JSON plus required chosen_option_id."""
    if isinstance(raw, dict):
        decision = parse_decision(json.dumps(raw), state=state)
        payload = raw
    else:
        decision = parse_decision(raw, state=state)
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines)
        payload = json.loads(text)

    oid = payload.get("chosen_option_id")
    if oid is None or str(oid).strip() == "":
        raise ValueError("chosen_option_id is required")
    return decision, str(oid).strip()


@dataclass
class SimAgentDecision:
    decision: CrewChiefDecision
    chosen_option_id: str
    chosen_label: str
    chosen_mean_finish_pos: float
    oracle_option_id: str
    oracle_label: str
    oracle_mean_finish_pos: float
    regret: float
    tool_results: list[ToolResult] = field(default_factory=list)
    faithfulness: dict[str, Any] = field(default_factory=dict)
    trajectory: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            **self.decision.to_dict(),
            "chosen_option_id": self.chosen_option_id,
            "chosen_label": self.chosen_label,
            "chosen_mean_finish_pos": round(self.chosen_mean_finish_pos, 3),
            "oracle_option_id": self.oracle_option_id,
            "oracle_label": self.oracle_label,
            "oracle_mean_finish_pos": round(self.oracle_mean_finish_pos, 3),
            "regret": round(self.regret, 4),
            "tools_used": [t.name for t in self.tool_results],
            "faithfulness": self.faithfulness,
        }
        if self.trajectory is not None:
            out["trajectory"] = self.trajectory
        return out


def _cards_from_payload(sim: dict[str, Any]) -> list[OptionCard]:
    out: list[OptionCard] = []
    for o in sim.get("options") or []:
        out.append(
            OptionCard(
                option_id=str(o["option_id"]),
                label=str(o["label"]),
                pit_after_laps=int(o["pit_after_laps"]),
                n_rolls=int(o.get("n_rolls", 0)),
                mean_finish_pos=float(o["mean_finish_pos"]),
                median_finish_pos=float(
                    o.get("median_finish_pos", o["mean_finish_pos"])
                ),
                p_finish_le_3=float(o.get("P_finish_le_3", 0.0)),
                p_finish_le_5=float(o.get("P_finish_le_5", 0.0)),
                p_finish_le_10=float(o.get("P_finish_le_10", 0.0)),
                mean_race_time_ms=float(o.get("mean_race_time_ms", 0.0)),
            )
        )
    return out


def build_sim_decision_from_choice(
    *,
    sim: dict[str, Any],
    chosen_option_id: str,
    push: str = "med",
    tool_results: list[ToolResult],
    rationale: str | None = None,
    reason: str | None = None,
    driver_message: str | None = None,
    trajectory: dict[str, Any] | None = None,
    current_compound: str | None = None,
    state: RaceState | None = None,
) -> SimAgentDecision:
    if not sim.get("available"):
        raise RuntimeError(f"simulate_strategies unavailable: {sim}")
    options = sim.get("options") or []
    chosen = find_option(options, chosen_option_id)
    if chosen is None:
        raise ValueError(f"unknown chosen_option_id={chosen_option_id!r}")

    oracle_id = str(sim["oracle_option_id"])
    oracle = find_option(options, oracle_id)
    if oracle is None:
        best = oracle_best(_cards_from_payload(sim))
        oracle_id = best.option_id
        oracle = {
            "option_id": best.option_id,
            "label": best.label,
            "mean_finish_pos": best.mean_finish_pos,
        }

    label = str(chosen["label"])
    action = option_to_action(label)
    tyre = option_to_tyre(label, current_compound=current_compound)
    c_mean = float(chosen["mean_finish_pos"])
    o_mean = float(oracle["mean_finish_pos"])
    if rationale is None:
        p3 = chosen.get("P_finish_le_3")
        rationale = (
            f"Selecting option {chosen_option_id} ({label}) with "
            f"mean_finish_pos={c_mean}; P_finish_le_3={p3}; "
            f"oracle={oracle_id} mean_finish_pos={o_mean}."
        )
    draft = CrewChiefDecision(
        action=action,  # type: ignore[arg-type]
        tyre=tyre,  # type: ignore[arg-type]
        push=push,  # type: ignore[arg-type]
        reason=reason or compose_reason(
            action=action,  # type: ignore[arg-type]
            tyre=tyre,
            push=push,  # type: ignore[arg-type]
            rationale=rationale,
        ),
        driver_message=driver_message or compose_driver_message(
            state,
            action=action,  # type: ignore[arg-type]
            tyre=tyre,
            push=push,  # type: ignore[arg-type]
        ),
        rationale=rationale,
    )
    decision = finalize_decision(state, draft)
    faith = faithfulness_score(decision.rationale, tool_results)
    return SimAgentDecision(
        decision=decision,
        chosen_option_id=chosen_option_id,
        chosen_label=label,
        chosen_mean_finish_pos=c_mean,
        oracle_option_id=oracle_id,
        oracle_label=str(oracle["label"]),
        oracle_mean_finish_pos=o_mean,
        regret=position_regret(c_mean, o_mean),
        tool_results=tool_results,
        faithfulness=faith,
        trajectory=trajectory,
    )


class SimAwareBackend:
    name: str
    replay: RaceReplay

    def decide_with_sims(
        self,
        state: RaceState,
        *,
        memory: RaceMemoryStore | None = None,
    ) -> SimAgentDecision: ...


class HeuristicSimBackend(SimAwareBackend):
    """
    Offline Phase 6 stand-in: call simulate_strategies, pick the oracle card,
    map to pit/stay, cite sim numbers (high faithfulness / zero regret).
    """

    name = "heuristic_sim"

    def __init__(self, replay: RaceReplay):
        self.replay = replay

    def decide_with_sims(
        self,
        state: RaceState,
        *,
        memory: RaceMemoryStore | None = None,
    ) -> SimAgentDecision:
        belt = ToolBelt(self.replay, state)
        results = [
            belt.call("get_gaps"),
            belt.call("get_remaining_laps"),
            belt.call("simulate_strategies"),
        ]
        sim = results[-1].payload
        oid = str(sim["oracle_option_id"])
        sd = build_sim_decision_from_choice(
            sim=sim,
            chosen_option_id=oid,
            tool_results=results,
            current_compound=state.tyre_compound,
            state=state,
        )
        return _record_sim_decision(state, sd, memory, sim)


class PitNextSimBaseline(SimAwareBackend):
    """Always pick pit_next_lap when present — positive-regret contrast baseline."""

    name = "pit_next_sim"

    def __init__(self, replay: RaceReplay):
        self.replay = replay

    def decide_with_sims(
        self,
        state: RaceState,
        *,
        memory: RaceMemoryStore | None = None,
    ) -> SimAgentDecision:
        belt = ToolBelt(self.replay, state)
        results = [belt.call("simulate_strategies")]
        sim = results[0].payload
        options = sim.get("options") or []
        pit = next((o for o in options if o.get("label") == "pit_next_lap"), None)
        oid = str(pit["option_id"]) if pit else str(sim["oracle_option_id"])
        sd = build_sim_decision_from_choice(
            sim=sim, chosen_option_id=oid, tool_results=results,
            current_compound=state.tyre_compound,
            state=state,
        )
        return _record_sim_decision(state, sd, memory, sim)


class OpenAISimBackend(SimAwareBackend):
    """OpenAI-compatible chat: tools then JSON with chosen_option_id."""

    name = "openai_sim"

    def __init__(
        self,
        replay: RaceReplay,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        max_tool_rounds: int = 4,
        max_retries: int = 2,
        default_headers: dict[str, str] | None = None,
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
        self.replay = replay
        self.model = model or os.environ.get("CREW_CHIEF_MODEL", "gpt-4o-mini")
        self.max_tool_rounds = max_tool_rounds
        self.max_retries = max_retries

    def decide_with_sims(
        self,
        state: RaceState,
        *,
        memory: RaceMemoryStore | None = None,
    ) -> SimAgentDecision:
        belt = ToolBelt(self.replay, state)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SIM_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_sim_user_prompt(state, memory, replay=self.replay),
            },
        ]
        last_err: Exception | None = None
        attempt_logs: list[dict[str, Any]] = []

        for attempt_i in range(self.max_retries + 1):
            collected: list[ToolResult] = []
            round_messages = list(messages)
            try:
                for round_i in range(self.max_tool_rounds):
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        messages=round_messages,
                        tools=OPENAI_TOOL_SCHEMAS,
                        tool_choice="auto",
                        temperature=0.0,
                    )
                    msg = resp.choices[0].message
                    tool_calls = msg.tool_calls or []
                    if not tool_calls:
                        content = msg.content or ""
                        round_messages.append(
                            {"role": "assistant", "content": content}
                        )
                        decision, oid = parse_sim_decision(content, state=state)
                        forced_sim = False
                        sim_payload = next(
                            (
                                t.payload
                                for t in collected
                                if t.name == "simulate_strategies"
                            ),
                            None,
                        )
                        if sim_payload is None:
                            sim_tr = belt.call("simulate_strategies")
                            collected.append(sim_tr)
                            sim_payload = sim_tr.payload
                            forced_sim = True
                        trajectory = {
                            "backend": self.name,
                            "model": self.model,
                            "attempts": attempt_logs
                            + [
                                {
                                    "attempt": attempt_i,
                                    "rounds": round_i + 1,
                                    "messages": round_messages,
                                    "tools_called": [t.name for t in collected],
                                    "tool_payloads": {
                                        t.name: t.payload for t in collected
                                    },
                                    "forced_simulate_strategies": forced_sim,
                                    "error": None,
                                }
                            ],
                        }
                        sd = build_sim_decision_from_choice(
                            sim=sim_payload,
                            chosen_option_id=oid,
                            push=decision.push,
                            tool_results=collected,
                            rationale=decision.rationale,
                            reason=decision.reason,
                            driver_message=decision.driver_message,
                            trajectory=trajectory,
                            current_compound=state.tyre_compound,
                            state=state,
                        )
                        return _record_sim_decision(
                            state, sd, memory, sim_payload
                        )

                    round_messages.append(
                        {
                            "role": "assistant",
                            "content": msg.content,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments or "{}",
                                    },
                                }
                                for tc in tool_calls
                            ],
                        }
                    )
                    for tc in tool_calls:
                        args = json.loads(tc.function.arguments or "{}")
                        tr = belt.call(tc.function.name, args)
                        collected.append(tr)
                        round_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "name": tc.function.name,
                                "content": tr.to_json(),
                            }
                        )
                raise RuntimeError("max tool rounds exceeded without decision")
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                attempt_logs.append(
                    {
                        "attempt": attempt_i,
                        "messages": round_messages,
                        "tools_called": [t.name for t in collected],
                        "tool_payloads": {t.name: t.payload for t in collected},
                        "error": str(exc),
                    }
                )
                continue
        raise RuntimeError(f"openai_sim failed: {last_err}")


@dataclass
class RaceReplayResult:
    race_id: int
    driver_id: int
    laps: list[int]
    decisions: list[SimAgentDecision]
    memory: RaceMemoryStore | None = None

    @property
    def mean_regret(self) -> float | None:
        if not self.decisions:
            return None
        return sum(d.regret for d in self.decisions) / len(self.decisions)

    def flip_flop_rate(
        self,
        replay: RaceReplay,
        *,
        exclude_advisory_mismatch: bool = False,
    ) -> float | None:
        if self.memory is None:
            return None
        pit_laps = replay.pit_laps(self.race_id, self.driver_id)
        return self.memory.flip_flop_rate(
            self.race_id,
            self.driver_id,
            pit_laps=pit_laps,
            exclude_advisory_mismatch=exclude_advisory_mismatch,
        )


def replay_race_decisions(
    backend: SimAwareBackend,
    replay: RaceReplay,
    race_id: int,
    driver_id: int,
    *,
    memory: RaceMemoryStore | None = None,
    lap_from: int = 1,
    lap_to: int | None = None,
) -> RaceReplayResult:
    """Full-race (or window) lap-by-lap Phase 6 loop with optional memory."""
    laps_avail = replay.available_laps(race_id, driver_id)
    if not laps_avail:
        raise KeyError(f"No laps for race_id={race_id} driver_id={driver_id}")
    hi = lap_to if lap_to is not None else laps_avail[-1]
    lap_range = [L for L in laps_avail if lap_from <= L <= hi]
    store = memory if memory is not None else RaceMemoryStore()
    use_memory = memory is not None
    decisions: list[SimAgentDecision] = []
    for lap in lap_range:
        state = replay.get_state(race_id, driver_id, lap)
        sd = backend.decide_with_sims(
            state, memory=store if use_memory else None
        )
        decisions.append(sd)
    return RaceReplayResult(
        race_id=race_id,
        driver_id=driver_id,
        laps=lap_range,
        decisions=decisions,
        memory=store if use_memory else None,
    )


def get_sim_backend(name: str, replay: RaceReplay) -> SimAwareBackend:
    if name in {"heuristic_sim", "heuristic"}:
        return HeuristicSimBackend(replay)
    if name in {"pit_next_sim", "pit_next"}:
        return PitNextSimBaseline(replay)
    if name in {"openai_sim", "openai"}:
        return OpenAISimBackend(replay)
    if name == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAISimBackend(replay)
        return HeuristicSimBackend(replay)
    raise ValueError(f"unknown sim backend: {name}")
