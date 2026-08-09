"""Phase 9: Strategy + Race Engineer radio isolation tests."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from race_engineer.radio import (
    HeuristicRadioBackend,
    OpenAIRadioBackend,
    PassthroughRadioBackend,
    apply_radio,
    parse_radio_message,
)
from race_engineer.radio_log import RadioCall, RadioLog, is_repetitive, similarity
from shared.decision import CrewChiefDecision
from shared.pipeline import MultiAgentSimBackend, attach_radio, get_sim_backend
from shared.replay import RaceReplay
from strategy_engineer.memory import RaceMemoryStore
from strategy_engineer.strategy import HeuristicSimBackend

DATASET = REPO / "dataset"


class _StubClient:
    """Replays canned driver_message replies and records what was sent."""

    def __init__(self, replies):
        self._replies = list(replies)
        self.sent: list[list[dict]] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs):
        self.sent.append(kwargs["messages"])
        idx = min(len(self.sent) - 1, len(self._replies) - 1)
        content = json.dumps({"driver_message": self._replies[idx]})
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


def _stub_radio_backend(replies):
    backend = OpenAIRadioBackend.__new__(OpenAIRadioBackend)
    client = _StubClient(replies)
    backend._client = client
    backend.model = "stub"
    backend.max_retries = 2
    backend.fallback = HeuristicRadioBackend()
    return backend, client


class TestRadioAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = RaceReplay(DATASET).load()
        cls.state = cls.replay.get_state(1132, 1, 22)

    def test_parse_radio_message(self):
        msg = parse_radio_message('{"driver_message":"Box box, box this lap, hard."}')
        self.assertEqual(msg, "Box box, box this lap, hard.")
        self.assertLessEqual(len(msg), 120)

    def test_apply_radio_preserves_strategy_fields(self):
        decision = CrewChiefDecision(
            action="stay",
            tyre=None,
            push="high",
            reason="Stay and push.",
            rationale="Gap ahead is close; mean_finish_pos=2.1.",
        )
        radio = HeuristicRadioBackend()
        rewritten, result = apply_radio(self.state, decision, radio)
        self.assertEqual(rewritten.action, decision.action)
        self.assertEqual(rewritten.tyre, decision.tyre)
        self.assertEqual(rewritten.push, decision.push)
        self.assertEqual(rewritten.reason, decision.reason)
        self.assertEqual(rewritten.rationale, decision.rationale)
        self.assertEqual(result.radio_backend, "heuristic_radio")
        self.assertTrue(rewritten.driver_message)
        self.assertIn("Push", rewritten.driver_message)
        self.assertIn("driver_message", rewritten.to_dict())

    def test_passthrough_keeps_strategy_radio(self):
        decision = CrewChiefDecision(
            action="pit",
            tyre="hard",
            push="low",
            reason="Box now.",
            rationale="SC window.",
            driver_message="Box, safety car window, hard.",
        )
        rewritten, _ = apply_radio(
            self.state, decision, PassthroughRadioBackend()
        )
        self.assertEqual(rewritten.driver_message, decision.driver_message)

    def test_strategy_omits_radio_until_multi_agent(self):
        strategy = HeuristicSimBackend(self.replay)
        bare = strategy.decide_with_sims(self.state)
        self.assertIsNone(bare.decision.driver_message)
        self.assertNotIn("driver_message", bare.to_dict())
        multi = MultiAgentSimBackend(strategy, HeuristicRadioBackend())
        with_radio = multi.decide_with_sims(self.state)
        self.assertTrue(with_radio.decision.driver_message)
        self.assertIn("driver_message", with_radio.to_dict())
        self.assertEqual(with_radio.radio.radio_backend, "heuristic_radio")

    def test_multi_agent_does_not_change_card_or_memory(self):
        strategy = HeuristicSimBackend(self.replay)
        multi = MultiAgentSimBackend(strategy, HeuristicRadioBackend())
        memory = RaceMemoryStore()
        sd = multi.decide_with_sims(self.state, memory=memory)
        self.assertEqual(sd.chosen_option_id, sd.oracle_option_id)
        self.assertAlmostEqual(sd.regret, 0.0)
        self.assertIsNotNone(sd.radio)
        self.assertEqual(sd.radio.radio_backend, "heuristic_radio")
        entries = memory.entries(self.state.race_id, self.state.driver_id)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].action, sd.decision.action)
        self.assertEqual(entries[0].chosen_option_id, sd.chosen_option_id)
        self.assertEqual(entries[0].rationale, sd.decision.rationale)
        self.assertFalse(hasattr(entries[0], "driver_message"))

    def test_attach_radio_leaves_faithfulness_and_regret(self):
        strategy = HeuristicSimBackend(self.replay)
        sd = strategy.decide_with_sims(self.state)
        before = (
            sd.chosen_option_id,
            sd.regret,
            sd.decision.action,
            sd.decision.rationale,
            sd.faithfulness.get("faithfulness"),
        )
        after = attach_radio(self.state, sd, HeuristicRadioBackend())
        self.assertEqual(
            (
                after.chosen_option_id,
                after.regret,
                after.decision.action,
                after.decision.rationale,
                after.faithfulness.get("faithfulness"),
            ),
            before,
        )

    def test_get_sim_backend_wraps_by_default(self):
        backend = get_sim_backend("heuristic_sim", self.replay)
        self.assertIsInstance(backend, MultiAgentSimBackend)
        bare = get_sim_backend(
            "heuristic_sim", self.replay, multi_agent=False
        )
        self.assertIsInstance(bare, HeuristicSimBackend)

    def test_strategy_prompts_exclude_radio(self):
        from strategy_engineer.board import SYSTEM_PROMPT
        from strategy_engineer.strategy import SIM_SYSTEM_PROMPT
        from strategy_engineer.tools_agent import TOOL_SYSTEM_PROMPT

        for name, prompt in (
            ("SYSTEM", SYSTEM_PROMPT),
            ("SIM", SIM_SYSTEM_PROMPT),
            ("TOOL", TOOL_SYSTEM_PROMPT),
        ):
            with self.subTest(prompt=name):
                self.assertNotIn("driver_message", prompt)
                self.assertNotIn("Radio style for driver_message", prompt)
                self.assertNotIn("Box, box.", prompt)

    def test_agents_do_not_import_each_other(self):
        import race_engineer.radio as radio_mod
        import race_engineer.radio_heuristic as radio_heur_mod
        import strategy_engineer.strategy as strat_mod

        radio_src = Path(radio_mod.__file__).read_text(encoding="utf-8")
        radio_heur_src = Path(radio_heur_mod.__file__).read_text(encoding="utf-8")
        strat_src = Path(strat_mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("strategy_engineer", radio_src)
        self.assertNotIn("strategy_engineer", radio_heur_src)
        self.assertNotIn("race_engineer", strat_src)
        # LLM module should not embed heuristic compose logic.
        self.assertNotIn("def compose_situation_radio", radio_src)
        self.assertNotIn("class HeuristicRadioBackend", radio_src)

    def test_situation_block_in_prompt(self):
        from race_engineer.radio import build_radio_user_prompt
        from race_engineer.situation import build_radio_situation

        if 1052 not in self.replay._laps:
            self.skipTest("race 1052 not in dataset")
        state = self.replay.get_state(1052, 1, 14)
        sit = build_radio_situation(state, self.replay)
        prompt = build_radio_user_prompt(
            state,
            action="stay",
            tyre=None,
            push="high",
            situation=sit,
        )
        self.assertIn("Recent race context", prompt)
        self.assertTrue(sit.lines)

    def test_situation_detects_stop_and_pressure(self):
        from race_engineer.situation import build_radio_situation

        if 1052 not in self.replay._laps:
            self.skipTest("race 1052 not in dataset")
        # Context reports objective stop/board facts; it does not interpret them.
        state = self.replay.get_state(1052, 1, 13)
        sit = build_radio_situation(state, self.replay)
        joined = " ".join(sit.lines)
        self.assertIn("pitted", joined)
        self.assertIn("Current gaps", joined)

        late = self.replay.get_state(1052, 1, 55)
        late_sit = build_radio_situation(late, self.replay)
        late_joined = " ".join(late_sit.lines)
        self.assertIn("current P1", late_joined)
        self.assertIn("Previous lap", late_joined)

    def test_heuristic_radio_is_minimal_fallback(self):
        from race_engineer.radio import HeuristicRadioBackend

        if 1052 not in self.replay._laps:
            self.skipTest("race 1052 not in dataset")
        decision = CrewChiefDecision(
            action="stay",
            tyre=None,
            push="high",
            reason="Stay.",
            rationale="mean_finish_pos=3.0",
        )
        radio = HeuristicRadioBackend()
        state = self.replay.get_state(1052, 1, 15)
        msg = radio.compose(state, decision, replay=self.replay)
        self.assertIn("Push", msg)
        self.assertNotIn("Stay out", msg)
        self.assertLessEqual(len(msg), 120)

    def test_prompt_gives_llm_raw_deltas_and_pit_events(self):
        from race_engineer.radio import build_radio_user_prompt
        from race_engineer.situation import build_radio_situation

        if 1052 not in self.replay._laps:
            self.skipTest("race 1052 not in dataset")
        state = self.replay.get_state(1052, 1, 14)
        situation = build_radio_situation(state, self.replay)
        prompt = build_radio_user_prompt(
            state,
            action="stay",
            tyre=None,
            push="high",
            situation=situation,
        )
        self.assertIn("Previous lap", prompt)
        self.assertIn("Position changed", prompt)
        self.assertIn("pitted", prompt)
        self.assertNotIn("Pit window: OPEN", prompt)
        self.assertNotIn("Undercut threat", prompt)

    def test_attach_radio_keeps_planned_pit_lap(self):
        strategy = HeuristicSimBackend(self.replay)
        sd = strategy.decide_with_sims(self.state)
        sd.chosen_planned_pit_lap = 34
        after = attach_radio(self.state, sd, HeuristicRadioBackend())
        self.assertEqual(after.chosen_planned_pit_lap, 34)
        self.assertEqual(after.to_dict()["planned_pit_lap"], 34)

    def test_attach_radio_carries_every_strategy_field(self):
        # replace() is what stops a newly added field from being dropped here.
        import dataclasses

        from strategy_engineer.strategy import SimAgentDecision

        strategy = HeuristicSimBackend(self.replay)
        sd = strategy.decide_with_sims(self.state)
        after = attach_radio(self.state, sd, HeuristicRadioBackend())
        rewritten = {"decision", "trajectory", "radio"}
        for f in dataclasses.fields(SimAgentDecision):
            if f.name in rewritten:
                continue
            with self.subTest(field=f.name):
                self.assertEqual(
                    getattr(after, f.name), getattr(sd, f.name)
                )

    def test_similarity_flags_reworded_repeat(self):
        a = "Stay out. Safety Car pace, keep the tyres under control."
        b = "Stay out, stay out. Safety car pace, keep the tyres in the window."
        c = "Box box, box this lap, hard."
        self.assertGreater(similarity(a, b), 0.6)
        self.assertTrue(is_repetitive(b, [a]))
        self.assertFalse(is_repetitive(c, [a, b]))

    def test_radio_log_keeps_last_calls_per_driver(self):
        log = RadioLog(recall=2)
        for lap, msg in ((1, "one"), (2, "two"), (3, "three")):
            log.record(1052, 1, lap, msg)
        log.record(1052, 2, 1, "other driver")
        self.assertEqual(log.messages(1052, 1), ("two", "three"))
        self.assertEqual(log.messages(1052, 2), ("other driver",))
        log.record(1052, 1, 4, "   ")
        self.assertEqual(log.messages(1052, 1), ("two", "three"))

    def test_recent_calls_reach_the_prompt(self):
        from race_engineer.radio import build_radio_user_prompt

        prompt = build_radio_user_prompt(
            self.state,
            action="stay",
            tyre=None,
            push="med",
            recent_calls=(RadioCall(lap=21, message="Stay out, safety car."),),
        )
        self.assertIn("Radio you already sent", prompt)
        self.assertIn("lap 21: Stay out, safety car.", prompt)

    def test_repeated_call_is_nudged_once(self):
        decision = CrewChiefDecision(
            action="stay",
            tyre=None,
            push="med",
            reason="Stay.",
            rationale="mean_finish_pos=2.0",
        )
        prior = "Stay out. Safety Car pace, keep the tyres under control."
        backend, client = _stub_radio_backend(
            [
                "Stay out, stay out. Safety car pace, keep tyres in the window.",
                "Track clear in two corners, we go again.",
            ]
        )
        msg = backend.compose(
            self.state,
            decision,
            recent_calls=(RadioCall(lap=21, message=prior),),
        )
        self.assertEqual(msg, "Track clear in two corners, we go again.")
        self.assertEqual(len(client.sent), 2)
        self.assertIn(
            "already made", client.sent[1][-1]["content"]
        )

    def test_fresh_call_costs_one_request(self):
        decision = CrewChiefDecision(
            action="stay",
            tyre=None,
            push="med",
            reason="Stay.",
            rationale="mean_finish_pos=2.0",
        )
        backend, client = _stub_radio_backend(["Car behind is closing, two tenths."])
        msg = backend.compose(
            self.state,
            decision,
            recent_calls=(RadioCall(lap=21, message="Stay out, safety car."),),
        )
        self.assertEqual(msg, "Car behind is closing, two tenths.")
        self.assertEqual(len(client.sent), 1)

    def test_second_repeat_is_sent_rather_than_dropped(self):
        decision = CrewChiefDecision(
            action="stay",
            tyre=None,
            push="med",
            reason="Stay.",
            rationale="mean_finish_pos=2.0",
        )
        prior = "Stay out. Safety Car pace, keep the tyres under control."
        backend, client = _stub_radio_backend(
            ["Stay out. Safety car pace, keep the tyres under control."]
        )
        msg = backend.compose(
            self.state,
            decision,
            recent_calls=(RadioCall(lap=21, message=prior),),
        )
        self.assertEqual(
            msg, "Stay out. Safety car pace, keep the tyres under control."
        )
        self.assertEqual(len(client.sent), 2)

    def test_multi_agent_backend_logs_its_own_radio(self):
        strategy = HeuristicSimBackend(self.replay)
        multi = MultiAgentSimBackend(strategy, HeuristicRadioBackend())
        sd = multi.decide_with_sims(self.state)
        self.assertEqual(
            multi.radio_log.messages(self.state.race_id, self.state.driver_id),
            (sd.decision.driver_message,),
        )

    def test_apply_radio_passes_replay(self):
        decision = CrewChiefDecision(
            action="stay",
            tyre=None,
            push="med",
            reason="Stay.",
            rationale="mean_finish_pos=2.0",
        )
        rewritten, _ = apply_radio(
            self.state, decision, HeuristicRadioBackend(), replay=self.replay
        )
        self.assertEqual(rewritten.action, "stay")
        self.assertTrue(rewritten.driver_message)


if __name__ == "__main__":
    unittest.main()
