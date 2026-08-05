"""Phase 9: Strategy + Race Engineer radio isolation tests."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from race_engineer.radio import (
    HeuristicRadioBackend,
    PassthroughRadioBackend,
    apply_radio,
    parse_radio_message,
)
from shared.decision import CrewChiefDecision
from shared.pipeline import MultiAgentSimBackend, attach_radio, get_sim_backend
from shared.replay import RaceReplay
from strategy_engineer.memory import RaceMemoryStore
from strategy_engineer.strategy import HeuristicSimBackend

DATASET = REPO / "dataset"


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
        self.assertIn("Stay out", rewritten.driver_message)
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
        import strategy_engineer.strategy as strat_mod

        radio_src = Path(radio_mod.__file__).read_text(encoding="utf-8")
        strat_src = Path(strat_mod.__file__).read_text(encoding="utf-8")
        self.assertNotIn("strategy_engineer", radio_src)
        self.assertNotIn("race_engineer", strat_src)


if __name__ == "__main__":
    unittest.main()
