from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from race_engineer.crew_chief_eval import build_eval_points, load_frozen_race_ids
from race_engineer.replay import RaceReplay
from race_engineer.sim_agent import (
    HeuristicSimBackend,
    PitNextSimBaseline,
    build_sim_user_prompt,
    option_to_action,
    parse_sim_decision,
    position_regret,
)
from race_engineer.memory import RaceMemoryStore

DATASET = REPO / "dataset"


class TestSimAgent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = RaceReplay(DATASET).load()
        cls.state = cls.replay.get_state(1132, 1, 22)

    def test_option_to_action(self):
        self.assertEqual(option_to_action("pit_next_lap"), "pit")
        self.assertEqual(option_to_action("stay_to_finish"), "stay")
        self.assertEqual(option_to_action("stay_3_then_pit"), "stay")

    def test_option_to_tyre(self):
        from race_engineer.sim_agent import option_to_tyre

        self.assertEqual(option_to_tyre("pit_next_lap", current_compound="MEDIUM"), "hard")
        self.assertEqual(option_to_tyre("stay_3_then_pit"), None)

    def test_position_regret(self):
        self.assertAlmostEqual(position_regret(4.0, 2.0), 2.0)
        self.assertAlmostEqual(position_regret(1.0, 1.0), 0.0)

    def test_build_sim_user_prompt_advisory_context(self):
        store = RaceMemoryStore()
        s14 = self.replay.get_state(1052, 1, 14) if 1052 in self.replay._laps else self.state
        store.record_sim_decision(
            s14,
            action="pit",
            chosen_option_id="A",
            chosen_label="pit_next_lap",
            oracle_option_id="A",
            rationale="Box.",
            sim={"available": True, "oracle_option_id": "A"},
        )
        s15 = self.replay.get_state(1052, 1, 15) if 1052 in self.replay._laps else self.state
        prompt = build_sim_user_prompt(
            s15, store, replay=self.replay if 1052 in self.replay._laps else None
        )
        self.assertIn("Strategy context:", prompt)
        if 1052 in self.replay._laps:
            self.assertIn("Execution note:", prompt)
            self.assertIn("advised box lap 15; driver stayed out", prompt)

    def test_parse_requires_option_id(self):
        with self.assertRaises(ValueError):
            parse_sim_decision(
                '{"action":"stay","tyre":null,"push":"med","rationale":"x"}'
            )
        d, oid = parse_sim_decision(
            {
                "action": "pit",
                "tyre": "hard",
                "push": "med",
                "chosen_option_id": "A",
                "rationale": "mean_finish_pos=2.0",
            }
        )
        self.assertEqual(d.action, "pit")
        self.assertEqual(oid, "A")

    def test_heuristic_follows_oracle_zero_regret(self):
        backend = HeuristicSimBackend(self.replay)
        sd = backend.decide_with_sims(self.state)
        self.assertEqual(sd.chosen_option_id, sd.oracle_option_id)
        self.assertAlmostEqual(sd.regret, 0.0)
        self.assertIn("simulate_strategies", sd.to_dict()["tools_used"])
        self.assertGreaterEqual(sd.faithfulness["faithfulness"], 0.9)
        self.assertIn(sd.decision.action, ("pit", "stay"))

    def test_pit_next_baseline_nonnegative_regret(self):
        backend = PitNextSimBaseline(self.replay)
        sd = backend.decide_with_sims(self.state)
        self.assertGreaterEqual(sd.regret, 0.0)

    def test_eval_smoke(self):
        race_ids = load_frozen_race_ids()[:2]
        points = build_eval_points(self.replay, race_ids, n_samples=3, seed=0)
        backend = HeuristicSimBackend(self.replay)
        for pt in points:
            state = self.replay.get_state(pt.race_id, pt.driver_id, pt.lap)
            sd = backend.decide_with_sims(state)
            self.assertAlmostEqual(sd.regret, 0.0)


if __name__ == "__main__":
    unittest.main()
