from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from race_engineer.crew_chief_eval import build_eval_points, load_frozen_race_ids
from race_engineer.faithfulness import extract_rationale_numbers, faithfulness_score
from race_engineer.replay import RaceReplay
from race_engineer.tool_agent import HeuristicToolBackend
from race_engineer.tools import ToolBelt, ToolResult

DATASET = REPO / "dataset"


class TestTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = RaceReplay(DATASET).load()
        cls.state = cls.replay.get_state(1132, 1, 22)  # HAM British 2024 L22

    def test_tool_belt_gaps_and_remaining(self):
        belt = ToolBelt(self.replay, self.state)
        gaps = belt.get_gaps()
        self.assertEqual(gaps["position"], 3)
        self.assertAlmostEqual(gaps["gap_ahead_s"], 1.001, places=3)
        rem = belt.get_remaining_laps()
        self.assertEqual(rem["lap"], 22)
        self.assertEqual(rem["total_laps"], 52)
        self.assertEqual(rem["remaining_laps"], 30)

    def test_undercut_stats_excludes_current_race(self):
        belt = ToolBelt(self.replay, self.state)
        und = belt.lookup_circuit_undercut_stats()
        self.assertEqual(und["circuit_id"], 9)
        self.assertGreater(und["n_first_stops"], 0)
        self.assertIsNotNone(und["median_first_stop_lap_fraction"])

    def test_faithfulness_matches_tool_numbers(self):
        tools = [
            ToolResult("get_gaps", {"gap_ahead_s": 1.001, "position": 3}),
        ]
        good = faithfulness_score("Gap ahead is 1.001s from P3.", tools)
        self.assertEqual(good["faithfulness"], 1.0)
        bad = faithfulness_score("Gap ahead is 9.999s.", tools)
        self.assertLess(bad["faithfulness"], 1.0)
        self.assertIn(9.999, bad["unmatched"])

    def test_extract_numbers(self):
        self.assertEqual(extract_rationale_numbers("age 22, gap +1.001"), [22.0, 1.001])

    def test_heuristic_tools_schema_and_faith(self):
        backend = HeuristicToolBackend(self.replay)
        td = backend.decide_with_tools(self.state)
        self.assertIn(td.decision.action, ("pit", "stay"))
        self.assertEqual(len(td.tool_results), 6)
        self.assertGreaterEqual(td.faithfulness["faithfulness"], 0.9)

    def test_heuristic_tools_eval_smoke(self):
        race_ids = load_frozen_race_ids()[:2]
        points = build_eval_points(self.replay, race_ids, n_samples=5, seed=0)
        backend = HeuristicToolBackend(self.replay)
        ok = 0
        faith = []
        for pt in points:
            state = self.replay.get_state(pt.race_id, pt.driver_id, pt.lap)
            td = backend.decide_with_tools(state)
            ok += 1
            faith.append(td.faithfulness["faithfulness"])
        self.assertEqual(ok, len(points))
        self.assertGreaterEqual(sum(faith) / len(faith), 0.9)


if __name__ == "__main__":
    unittest.main()
