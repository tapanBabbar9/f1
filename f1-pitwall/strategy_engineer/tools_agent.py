"""Phase 3 tool-calling crew chief backends."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from strategy_engineer.board import HeuristicBackend
from shared.decision import (
    CrewChiefDecision,
    UPCOMING_LAP_DECISION,
    UPCOMING_LAP_USER_QUESTION,
    parse_decision,
)
from shared.faithfulness import faithfulness_score
from shared.racing_rules import DRY_MANDATORY_PIT_RULES
from shared.replay import RaceReplay
from shared.state import RaceState
from shared.tools import OPENAI_TOOL_SCHEMAS, ToolBelt, ToolResult

TOOL_SYSTEM_PROMPT = """You are an F1 strategy engineer on the pit wall.
""" + UPCOMING_LAP_DECISION + """

You MUST call the provided tools before deciding. Do not invent timing numbers —
only cite figures returned by tools (or present on the anonymized board).

The feed withholds race event, year, and driver identity. Circuit is kept for
pit-loss context. Do not guess which real-world race or driver this is.

After tools, reply with ONLY a JSON object (no markdown):
{
  "action": "pit" | "stay",
  "tyre": "soft" | "medium" | "hard" | null,
  "push": "low" | "med" | "high",
  "reason": "<one line, ≤100 chars>",
  "rationale": "<1-3 sentences citing tool numbers>"
}

Rules:
- If action is "stay", tyre must be null.
- If action is "pit", tyre must be soft, medium, or hard.
- Do not name drivers, teams, or race events.
- Do not write driver radio copy — a separate Race Engineer agent handles that.

""" + DRY_MANDATORY_PIT_RULES



@dataclass
class ToolDecision:
    decision: CrewChiefDecision
    tool_results: list[ToolResult] = field(default_factory=list)
    faithfulness: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.decision.to_dict(),
            "tools_used": [t.name for t in self.tool_results],
            "faithfulness": self.faithfulness,
        }


class ToolAwareBackend:
    """Protocol-ish: decide_with_tools returns ToolDecision."""

    name: str
    replay: RaceReplay

    def decide(self, state: RaceState) -> CrewChiefDecision:
        return self.decide_with_tools(state).decision

    def decide_with_tools(self, state: RaceState) -> ToolDecision: ...


class HeuristicToolBackend(ToolAwareBackend):
    """
    Offline Phase 3 stand-in: call all tools, then apply heuristic rules,
    citing tool numbers in the rationale (high faithfulness for CI).
    """

    name = "heuristic_tools"

    def __init__(self, replay: RaceReplay):
        self.replay = replay
        self._base = HeuristicBackend()

    def decide_with_tools(self, state: RaceState) -> ToolDecision:
        belt = ToolBelt(self.replay, state)
        results = belt.call_all()
        by_name = {t.name: t.payload for t in results}
        base = self._base.decide(state)

        gaps = by_name["get_gaps"]
        stint = by_name["get_stint_age"]
        rem = by_name["get_remaining_laps"]
        und = by_name["lookup_circuit_undercut_stats"]
        deg = by_name.get("predict_lap_time", {})

        bits = [
            f"stint_age_laps={stint.get('stint_age_laps')}",
            f"remaining_laps={rem.get('remaining_laps')}",
        ]
        if gaps.get("gap_ahead_s") is not None:
            bits.append(f"gap_ahead_s={gaps['gap_ahead_s']}")
        if gaps.get("gap_behind_s") is not None:
            bits.append(f"gap_behind_s={gaps['gap_behind_s']}")
        med_f = und.get("median_first_stop_lap_fraction")
        if med_f is not None:
            bits.append(f"circuit_median_first_stop_frac={med_f}")
        if deg.get("available") and deg.get("predicted_next_lap_s") is not None:
            bits.append(f"predicted_next_lap_s={deg['predicted_next_lap_s']}")
            if deg.get("delta_vs_last_ms") is not None:
                bits.append(f"pace_delta_vs_last_ms={deg['delta_vs_last_ms']}")
        sim = by_name.get("simulate_strategies", {})
        if sim.get("available") and sim.get("oracle_option_id") is not None:
            bits.append(
                f"sim_oracle={sim['oracle_option_id']}:{sim.get('oracle_label')}"
                f"(E_finish={sim.get('oracle_mean_finish_pos')})"
            )

        rationale = f"{base.rationale.rstrip('.')} ({'; '.join(bits)})."
        decision = CrewChiefDecision(
            action=base.action,
            tyre=base.tyre,
            push=base.push,
            reason=base.reason,
            rationale=rationale,
            driver_message=base.driver_message,
        )
        faith = faithfulness_score(decision.rationale, results)
        return ToolDecision(
            decision=decision, tool_results=results, faithfulness=faith
        )


class OpenAIToolBackend(ToolAwareBackend):
    """OpenAI-compatible chat with tool calls, then JSON decision."""

    name = "openai_tools"

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

    def decide_with_tools(self, state: RaceState) -> ToolDecision:
        belt = ToolBelt(self.replay, state)
        collected: list[ToolResult] = []
        board = state.pit_wall_view(anonymize=True)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": TOOL_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Anonymized pit-wall feed:\n\n"
                    f"{board}\n\n"
                    f"Call tools as needed, then {UPCOMING_LAP_USER_QUESTION}"
                ),
            },
        ]

        # Round 1: require at least one tool call.
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=OPENAI_TOOL_SCHEMAS,
            tool_choice="required",
            temperature=0.2,
        )
        msg = resp.choices[0].message
        messages.append(self._message_to_dict(msg))
        collected.extend(self._run_tool_calls(belt, msg, messages))

        # Further rounds: auto tools until model stops or cap.
        for _ in range(self.max_tool_rounds - 1):
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=OPENAI_TOOL_SCHEMAS,
                tool_choice="auto",
                temperature=0.2,
            )
            msg = resp.choices[0].message
            messages.append(self._message_to_dict(msg))
            if not getattr(msg, "tool_calls", None):
                break
            collected.extend(self._run_tool_calls(belt, msg, messages))

        # Final JSON decision (no tools).
        messages.append(
            {
                "role": "user",
                "content": (
                    "Using only the tool results and board, return ONLY the "
                    "decision JSON object now."
                ),
            }
        )
        last_err: Exception | None = None
        decision: CrewChiefDecision | None = None
        for _ in range(self.max_retries + 1):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.2,
                )
                raw = resp.choices[0].message.content or ""
                decision = parse_decision(raw, state=state)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Previous reply was invalid ({exc}). "
                            "Return ONLY valid decision JSON."
                        ),
                    }
                )
        if decision is None:
            assert last_err is not None
            raise last_err

        faith = faithfulness_score(decision.rationale, collected)
        return ToolDecision(
            decision=decision, tool_results=collected, faithfulness=faith
        )

    @staticmethod
    def _message_to_dict(msg: Any) -> dict[str, Any]:
        d: dict[str, Any] = {"role": "assistant", "content": msg.content}
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or "{}",
                    },
                }
                for tc in tool_calls
            ]
        return d

    def _run_tool_calls(
        self,
        belt: ToolBelt,
        msg: Any,
        messages: list[dict[str, Any]],
    ) -> list[ToolResult]:
        out: list[ToolResult] = []
        for tc in getattr(msg, "tool_calls", None) or []:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            try:
                result = belt.call(name, args if isinstance(args, dict) else {})
                payload = result.payload
            except Exception as exc:  # noqa: BLE001
                result = ToolResult(name=name, payload={"error": str(exc)})
                payload = result.payload
            out.append(result)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(payload),
                }
            )
        return out


def get_tool_backend(
    name: str,
    replay: RaceReplay,
    *,
    default_headers: dict[str, str] | None = None,
) -> ToolAwareBackend:
    if name in ("heuristic_tools", "heuristic"):
        return HeuristicToolBackend(replay)
    if name in ("openai_tools", "openai", "tools"):
        return OpenAIToolBackend(replay, default_headers=default_headers)
    if name == "auto":
        if os.environ.get("OPENAI_API_KEY"):
            return OpenAIToolBackend(replay, default_headers=default_headers)
        return HeuristicToolBackend(replay)
    raise ValueError(f"unknown tool backend: {name}")
