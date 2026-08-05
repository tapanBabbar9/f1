from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from shared.replay import RaceReplay
from strategy_engineer.sim import (
    brier_binary,
    build_default_options,
    finish_position,
    finish_position_board_sanity,
    oracle_best,
    projected_finish_position,
    simulate_ego_finish_ms,
    simulate_strategy_cards,
)
from shared.tools import ToolBelt

DATASET = REPO / "dataset"


class TestSim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = RaceReplay(DATASET).load()
        cls.state = cls.replay.get_state(1132, 1, 22)  # HAM British 2024 L22

    def test_build_default_options(self):
        opts = build_default_options(30, mandatory_pit_pending=False)
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
        labels = [c.label for c in cards]
        self.assertNotIn("stay_to_finish", labels)
        best = oracle_best(cards)
        self.assertIn(best, cards)
        for c in cards:
            self.assertGreaterEqual(c.mean_finish_pos, 1.0)
            self.assertLessEqual(c.p_finish_le_3, 1.0)

    def test_stay_longer_than_remaining_never_pits(self):
        """pit_after_laps > remaining → finish without boxing."""
        from strategy_engineer.sim import SimSnapshot, build_pace_schedule, _default_deg
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
        self.assertEqual(out["options"][0]["n_rolls"], 64)

    def test_sim_finish_stable_when_evidence_unchanged(self):
        """E[finish] should not swing wildly across +1 lap at same board bucket."""
        replay = self.replay
        race_id = 1052
        driver_id = 1
        if race_id not in replay._laps:
            self.skipTest("race 1052 not in dataset")
        cards_by_lap = {}
        for lap in (6, 7):
            try:
                state = replay.get_state(race_id, driver_id, lap)
            except KeyError:
                self.skipTest(f"lap {lap} missing for 1052/1")
            cards = simulate_strategy_cards(replay, state, n_rolls=64, seed=42)
            best = oracle_best(cards)
            cards_by_lap[lap] = best.mean_finish_pos
        if 6 in cards_by_lap and 7 in cards_by_lap:
            self.assertLess(
                abs(cards_by_lap[7] - cards_by_lap[6]),
                5.0,
                msg=f"L6 E={cards_by_lap[6]} L7 E={cards_by_lap[7]}",
            )

    def test_sim_realistic_near_board_position(self):
        """At P2 with healthy gaps, E[finish] should stay in plausible range."""
        if 1052 not in self.replay._laps:
            self.skipTest("race 1052 not in dataset")
        state = self.replay.get_state(1052, 1, 6)
        cards = simulate_strategy_cards(self.replay, state, n_rolls=64, seed=42)
        best = oracle_best(cards)
        self.assertLessEqual(best.mean_finish_pos, state.position + 6)
        self.assertGreaterEqual(best.mean_finish_pos, 1.0)
        pit_card = next(c for c in cards if c.label == "pit_next_lap")
        stay_card = next(c for c in cards if c.label == "stay_8_then_pit")
        self.assertLessEqual(pit_card.planned_pit_lap or 0, stay_card.planned_pit_lap or 99)

    def test_board_sanity_large_gap_ahead(self):
        """P3 with large gap ahead should not project finish better than board."""
        if 1052 not in self.replay._laps:
            self.skipTest("race 1052 not in dataset")
        state = self.replay.get_state(1052, 1, 15)
        self.assertGreater(state.gap_ahead_ms or 0, 5_000)
        held = finish_position_board_sanity(1.0, state)
        self.assertAlmostEqual(held, float(state.position), delta=0.25)

    def test_board_sanity_is_monotonic(self):
        """Bounds must not flatten distinct projections into one number."""
        if 1052 not in self.replay._laps:
            self.skipTest("race 1052 not in dataset")
        state = self.replay.get_state(1052, 1, 15)
        outs = [finish_position_board_sanity(float(p), state) for p in range(1, 12)]
        for lo, hi in zip(outs, outs[1:]):
            self.assertLess(lo, hi, msg=f"non-monotonic: {outs}")

    def test_sim_options_are_rankable(self):
        """A wear/pace asymmetry once collapsed every option to one E[finish]."""
        if 1052 not in self.replay._laps:
            self.skipTest("race 1052 not in dataset")
        ties = 0
        laps = [6, 10, 20, 25, 30, 35, 40]
        for lap in laps:
            cards = simulate_strategy_cards(
                self.replay, self.replay.get_state(1052, 1, lap), n_rolls=64, seed=42
            )
            if len({round(c.mean_finish_pos, 3) for c in cards}) == 1:
                ties += 1
        self.assertLessEqual(ties, 1, msg=f"{ties}/{len(laps)} laps had all options tied")

    def test_sc_lap_does_not_anchor_green_pace(self):
        """Under SC the pace anchor must come off a green lap, not the slow lap."""
        if 1052 not in self.replay._laps:
            self.skipTest("race 1052 not in dataset")
        from strategy_engineer.sim import (
            SimSnapshot,
            _default_deg,
            _green_flag_anchor_ms,
        )

        state = self.replay.get_state(1052, 1, 2)
        self.assertTrue(state.sc_active)
        snap = SimSnapshot.from_state(state, 0.0)
        anchor = _green_flag_anchor_ms(_default_deg(), snap, green_reference_ms=95_000.0)
        self.assertLess(anchor, state.last_lap_time_ms)
        self.assertEqual(anchor, 95_000.0)

    def test_sim_not_overoptimistic_mid_race(self):
        """At P3 with large gap ahead, E[finish] should stay near board."""
        if 1052 not in self.replay._laps:
            self.skipTest("race 1052 not in dataset")
        state = self.replay.get_state(1052, 1, 15)
        cards = simulate_strategy_cards(self.replay, state, n_rolls=64, seed=42)
        stay = next(c for c in cards if c.label == "stay_to_finish")
        self.assertGreaterEqual(stay.mean_finish_pos, state.position - 0.5)
        self.assertLessEqual(stay.mean_finish_pos, state.position + 2.5)

    def test_sim_stay_preferred_over_pit_after_late_stop(self):
        """After a recent stop with large gaps, stay should beat a third box."""
        if 1052 not in self.replay._laps:
            self.skipTest("race 1052 not in dataset")
        state = self.replay.get_state(1052, 1, 29)
        cards = simulate_strategy_cards(self.replay, state, n_rolls=64, seed=42)
        stay = next(c for c in cards if c.label == "stay_to_finish")
        pit = next(c for c in cards if c.label == "pit_next_lap")
        self.assertLessEqual(stay.mean_finish_pos, pit.mean_finish_pos + 0.1)
        best = oracle_best(cards)
        self.assertEqual(best.label, "stay_to_finish")

    def test_projected_finish_position_chain(self):
        state = self.replay.get_state(1052, 1, 6) if 1052 in self.replay._laps else self.state
        out = projected_finish_position(10, state)
        self.assertGreaterEqual(out, 1.0)
        self.assertLessEqual(out, float(state.drivers_on_track))


if __name__ == "__main__":
    unittest.main()
