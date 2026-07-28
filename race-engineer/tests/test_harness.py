from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from race_engineer.crew_chief_eval import pick_drivers_for_race
from race_engineer.harness import (
    HARNESS_VERSION,
    HarnessLapDecision,
    planned_pit_lap,
    run_harness,
    score_stints_pit_next,
    score_stints_sim_plan,
)
from race_engineer.replay import RaceReplay

DATASET = REPO / "dataset"


class TestHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = RaceReplay(DATASET).load()

    def test_planned_pit_lap_labels(self):
        self.assertEqual(planned_pit_lap(22, action="pit"), 23)
        self.assertEqual(
            planned_pit_lap(22, action="stay", label="stay_8_then_pit"),
            30,
        )

    def test_pit_next_scores_first_box(self):
        rid, did = 1132, 1
        pits = sorted(self.replay._pits.get(rid, {}).get(did, []))
        actual = pits[0]
        decisions = [
            HarnessLapDecision(
                lap=actual - 1,
                action="pit",
                pit_next=1,
                planned_pit_lap=actual,
                y_true_next_lap=1,
            )
        ]
        stints = score_stints_pit_next(self.replay, rid, did, decisions, tolerance_laps=2)
        self.assertEqual(stints[0].abs_error, 0)
        self.assertEqual(stints[0].score_kind, "first_box_stop")

    def test_sim_plan_scores_before_stop(self):
        rid, did = 1132, 1
        pits = sorted(self.replay._pits.get(rid, {}).get(did, []))
        actual = pits[0]
        decisions = [
            HarnessLapDecision(
                lap=actual - 1,
                action="stay",
                pit_next=0,
                planned_pit_lap=actual,
                chosen_label="stay_1_then_pit",
                y_true_next_lap=1,
            )
        ]
        stints = score_stints_sim_plan(self.replay, rid, did, decisions, tolerance_laps=2)
        self.assertEqual(stints[0].abs_error, 0)
        self.assertEqual(stints[0].score_kind, "plan_before_stop")

    def test_run_harness_smoke(self):
        payload = run_harness(
            self.replay,
            [1052],
            ["heuristic_sim", "heuristic_crew"],
            pick_drivers_fn=pick_drivers_for_race,
            tolerance_laps=2,
            memory=True,
            lap_from=1,
            lap_to=15,
        )
        self.assertEqual(payload["harness_version"], HARNESS_VERSION)
        self.assertIn("leaderboard_pit_next", payload)
        self.assertIn("leaderboard_sim_plan", payload)
        self.assertEqual(len(payload["leaderboard_pit_next"]), 1)
        self.assertEqual(len(payload["leaderboard_sim_plan"]), 1)


if __name__ == "__main__":
    unittest.main()
