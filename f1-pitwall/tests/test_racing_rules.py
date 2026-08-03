from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from shared.racing_rules import (
    alternate_dry_compound_id,
    mandatory_dry_pit_pending,
)
from shared.replay import RaceReplay
from strategy_engineer.sim import build_default_options, simulate_strategy_cards

DATASET = REPO / "dataset"


class TestRacingRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = RaceReplay(DATASET).load()
        cls.ham_l22 = cls.replay.get_state(1132, 1, 22)

    def test_mandatory_pending_before_first_stop(self):
        self.assertEqual(self.ham_l22.pit_count, 0)
        self.assertTrue(mandatory_dry_pit_pending(self.ham_l22))

    def test_alternate_compound_from_medium(self):
        self.assertEqual(alternate_dry_compound_id(2.0), 3.0)

    def test_build_options_omit_stay_to_finish_when_must_pit(self):
        opts = build_default_options(30, mandatory_pit_pending=True)
        labels = [o.label for o in opts]
        self.assertNotIn("stay_to_finish", labels)
        self.assertIn("pit_next_lap", labels)

    def test_sim_cards_omit_stay_to_finish_ham_l22(self):
        cards = simulate_strategy_cards(
            self.replay, self.ham_l22, n_rolls=4, seed=0, noise_ms=0.0
        )
        labels = [c.label for c in cards]
        self.assertNotIn("stay_to_finish", labels)


if __name__ == "__main__":
    unittest.main()
