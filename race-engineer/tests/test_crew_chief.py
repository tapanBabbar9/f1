from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from race_engineer.crew_chief import build_user_prompt, parse_decision
from race_engineer.crew_chief_eval import (
    build_eval_points,
    load_frozen_race_ids,
    run_crew_chief_eval,
)
from race_engineer.llm import HeuristicBackend
from race_engineer.replay import RaceReplay

DATASET = REPO / "dataset"


class TestCrewChief(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = RaceReplay(DATASET).load()

    def test_parse_valid_pit(self):
        d = parse_decision(
            '{"action":"pit","tyre":"medium","push":"low","rationale":"Tyres old."}'
        )
        self.assertEqual(d.action, "pit")
        self.assertEqual(d.tyre, "medium")
        self.assertEqual(d.pit_next, 1)

    def test_parse_stay_null_tyre(self):
        d = parse_decision(
            '{"action":"stay","tyre":null,"push":"high","rationale":"Gap manageable."}'
        )
        self.assertEqual(d.action, "stay")
        self.assertIsNone(d.tyre)

    def test_parse_rejects_pit_without_tyre(self):
        with self.assertRaises(ValueError):
            parse_decision(
                '{"action":"pit","tyre":null,"push":"low","rationale":"Stop now."}'
            )

    def test_frozen_races_file(self):
        ids = load_frozen_race_ids()
        self.assertEqual(len(ids), 10)

    def test_heuristic_eval_schema(self):
        race_ids = load_frozen_race_ids()[:3]
        points = build_eval_points(
            self.replay, race_ids, n_samples=40, seed=0
        )
        self.assertGreater(len(points), 10)
        result = run_crew_chief_eval(
            self.replay, HeuristicBackend(), points
        )
        self.assertGreaterEqual(result["schema_validity"], 0.99)
        self.assertIn("f1", result["pit_metrics"])

    def test_sc_on_race_state(self):
        miami = next(
            r
            for r in self.replay.list_races(year=2024)
            if "Miami" in r["name"]
        )
        did = self.replay.drivers_in_race(miami["race_id"])[0]
        state = self.replay.get_state(miami["race_id"], did, 28)
        self.assertTrue(state.sc_active)
        self.assertTrue(state.sc_deployed_this_lap)
        self.assertIn("Safety Car:", state.pit_wall_view())

    def test_llm_prompt_withholds_identity(self):
        state = self.replay.get_state(1089, 830, 47)
        full = state.pit_wall_view()
        anon = state.pit_wall_view(anonymize=True)
        prompt = build_user_prompt(state)
        self.assertIn("Italian Grand Prix", full)
        self.assertIn("VER", full)
        self.assertIn("Circuit:", full)
        self.assertIn("Circuit:", anon)
        self.assertIn("Monza", anon)  # track kept for pit-loss context
        self.assertNotIn("Italian Grand Prix", anon)
        self.assertNotIn("VER", anon)
        self.assertNotIn("2022", anon)
        self.assertNotIn("Italian Grand Prix", prompt)
        self.assertNotIn("VER", prompt)
        self.assertIn("identity withheld", prompt)
        self.assertIn("Monza", prompt)


if __name__ == "__main__":
    unittest.main()
