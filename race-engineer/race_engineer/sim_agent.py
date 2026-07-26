"""Phase 6: agent chooses among sim option cards and cites sim numbers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from race_engineer.crew_chief import CrewChiefDecision, build_user_prompt, parse_decision
from race_engineer.faithfulness import faithfulness_score
from race_engineer.replay import RaceReplay
from race_engineer.sim import OptionCard, oracle_best
from race_engineer.state import RaceState
from race_engineer.tools import OPENAI_TOOL_SCHEMAS, ToolBelt, ToolResult

SIM_SYSTEM_PROMPT = """You are an F1 race engineer on the pit wall.
Decide whether the driver should pit on the NEXT lap or stay out.

You MUST call simulate_strategies (and may call other tools). Choose among the
returned option cards. Prefer lower expected finish position.

Reply with ONLY a JSON object (no markdown):
{
  "action": "pit" | "stay",
  "tyre": "soft" | "medium" | "hard" | null,
  "push": "low" | "med" | "high",
  "chosen_option_id": "<option_id from simulate_strategies, e.g. A>",
  "rationale": "<1-3 sentences citing option E_finish / P_finish numbers>"
}

Rules:
- If action is "stay", tyre must be null.
- If action is "pit", tyre must be soft, medium, or hard.
- Map pit_next_lap → action pit; any stay_* option → action stay.
- Cite at least one sim number (mean_finish_pos or P_finish_le_*).
- Do not name drivers, teams, or race events.
- The feed withholds race/year/driver; circuit is kept for pit-loss context.
"""


def option_to_action(label: str) -> str:
    return "pit" if label == "pit_next_lap" else "stay"


def option_to_tyre(label: str) -> str | None:
    if label == "pit_next_lap":
        return "hard"
    return None


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


def parse_sim_decision(raw: str | dict[str, Any]) -> tuple[CrewChiefDecision, str]:
    """Parse crew-chief JSON plus required chosen_option_id."""
    if isinstance(raw, dict):
        decision = parse_decision(json.dumps(raw))
        payload = raw
    else:
        decision = parse_decision(raw)
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

    def to_dict(self) -> dict[str, Any]:
        return {
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
    tyre = option_to_tyre(label)
    c_mean = float(chosen["mean_finish_pos"])
    o_mean = float(oracle["mean_finish_pos"])
    if rationale is None:
        p3 = chosen.get("P_finish_le_3")
        rationale = (
            f"Selecting option {chosen_option_id} ({label}) with "
            f"mean_finish_pos={c_mean}; P_finish_le_3={p3}; "
            f"oracle={oracle_id} mean_finish_pos={o_mean}."
        )
    decision = CrewChiefDecision(
        action=action,  # type: ignore[arg-type]
        tyre=tyre,  # type: ignore[arg-type]
        push=push,  # type: ignore[arg-type]
        rationale=rationale,
    )
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
    )


class SimAwareBackend:
    name: str
    replay: RaceReplay

    def decide_with_sims(self, state: RaceState) -> SimAgentDecision: ...


class HeuristicSimBackend(SimAwareBackend):
    """
    Offline Phase 6 stand-in: call simulate_strategies, pick the oracle card,
    map to pit/stay, cite sim numbers (high faithfulness / zero regret).
    """

    name = "heuristic_sim"

    def __init__(self, replay: RaceReplay):
        self.replay = replay

    def decide_with_sims(self, state: RaceState) -> SimAgentDecision:
        belt = ToolBelt(self.replay, state)
        results = [
            belt.call("get_gaps"),
            belt.call("get_remaining_laps"),
            belt.call("simulate_strategies"),
        ]
        sim = results[-1].payload
        oid = str(sim["oracle_option_id"])
        return build_sim_decision_from_choice(
            sim=sim,
            chosen_option_id=oid,
            tool_results=results,
        )


class PitNextSimBaseline(SimAwareBackend):
    """Always pick pit_next_lap when present — positive-regret contrast baseline."""

    name = "pit_next_sim"

    def __init__(self, replay: RaceReplay):
        self.replay = replay

    def decide_with_sims(self, state: RaceState) -> SimAgentDecision:
        belt = ToolBelt(self.replay, state)
        results = [belt.call("simulate_strategies")]
        sim = results[0].payload
        options = sim.get("options") or []
        pit = next((o for o in options if o.get("label") == "pit_next_lap"), None)
        oid = str(pit["option_id"]) if pit else str(sim["oracle_option_id"])
        return build_sim_decision_from_choice(
            sim=sim, chosen_option_id=oid, tool_results=results
        )


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

    def decide_with_sims(self, state: RaceState) -> SimAgentDecision:
        belt = ToolBelt(self.replay, state)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SIM_SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(state)},
        ]
        last_err: Exception | None = None
        for _attempt in range(self.max_retries + 1):
            try:
                collected: list[ToolResult] = []
                round_messages = list(messages)
                for _ in range(self.max_tool_rounds):
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
                        decision, oid = parse_sim_decision(content)
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
                        return build_sim_decision_from_choice(
                            sim=sim_payload,
                            chosen_option_id=oid,
                            push=decision.push,
                            tool_results=collected,
                            rationale=decision.rationale,
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
                                "content": tr.to_json(),
                            }
                        )
                raise RuntimeError("max tool rounds exceeded without decision")
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                continue
        raise RuntimeError(f"openai_sim failed: {last_err}")


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
