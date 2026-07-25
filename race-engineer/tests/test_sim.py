from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from race_engineer.replay import RaceReplay
from race_engineer.sim import (
    brier_binary,
    build_default_options,
    finish_position,
    oracle_best,
    simulate_ego_finish_ms,
    simulate_strategy_cards,
)
from race_engineer.tools import ToolBelt

DATASET = REPO / "dataset"


class TestSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = RaceReplay(DATASET).load()
        cls.state = cls.replay.get_state(1132, 1, 22)  # HAM British 2024 L22

    def test_build_default_options(self):
        opts = build_default_options(30)
        labels = [o.label for o in opts]
        self.assertIn("pit_next_lap", labels)
        self.assertIn("stay_to_finish", labels)
        self.assertEqual(len({o.pit_after_laps for o in opts}), len(opts))

    def test_finish_position(self):
        self.assertEqual(finish_position(100.0, [90.0, 110.0, 120.0]), 2)

    def test_brier_perfect(self):
        self.assertAlmostEqual(brier_binary([1.0, 0.0], [1, 0]), 0.0)

    def test_simulate_cards_smoke(self):
        cards = simulate_strategy_cards(
            self.replay, self.state, n_rolls=8, seed=0, noise_ms=0.0
        )
        self.assertGreaterEqual(len(cards), 2)
        best = oracle_best(cards)
        self.assertIn(best, cards)
        for c in cards:
            self.assertGreaterEqual(c.mean_finish_pos, 1.0)
            self.assertLessEqual(c.p_finish_le_3, 1.0)

    def test_stay_longer_than_remaining_never_pits(self):
        """pit_after_laps > remaining → finish without boxing."""
        from race_engineer.sim import SimSnapshot, build_pace_schedule, _default_deg
        import numpy as np

        cumul = self.replay._cumulative_times(self.state.race_id, self.state.lap)
        ego = float(cumul.get(self.state.driver_id, self.state.cumulative_time_ms))
        snap = SimSnapshot.from_state(self.state, ego)
        remaining = snap.total_laps - snap.lap
        rng = np.random.default_rng(0)
        pace = build_pace_schedule(_default_deg(), snap)
        finish = simulate_ego_finish_ms(
            snap,
            pit_after_laps=remaining + 5,
            pace=pace,
            pit_loss_ms=22_000.0,
            rng=rng,
            noise_ms=0.0,
        )
        self.assertGreater(finish, ego)

    def test_tool_simulate_strategies(self):
        belt = ToolBelt(self.replay, self.state)
        out = belt.simulate_strategies()
        self.assertTrue(out["available"])
        self.assertGreaterEqual(out["n_options"], 2)
        self.assertIn("oracle_option_id", out)
        self.assertEqual(len(out["options"]), out["n_options"])


if __name__ == "__main__":
    unittest.main()
