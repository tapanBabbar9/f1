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
        self.assertIn("live_tyre_deg", out)
        self.assertIn(out["live_tyre_deg"]["source"], {"compound_prior", "live_stint"})
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
        """A wear/pace asymmetry once collapsed every option to one race time.

        Equal finish positions are legitimate — ten seconds of race time does not
        change where you finish when the gaps are large — so the discrimination
        that has to survive is in race time, which oracle_best tie-breaks on.
        """
        if 1052 not in self.replay._laps:
            self.skipTest("race 1052 not in dataset")
        for lap in (6, 10, 20, 25, 30, 35, 40):
            cards = simulate_strategy_cards(
                self.replay, self.replay.get_state(1052, 1, lap), n_rolls=64, seed=42
            )
            times = {round(c.mean_race_time_ms, 1) for c in cards}
            self.assertEqual(
                len(times), len(cards), msg=f"L{lap} race times not distinct: {times}"
            )
            spread = max(times) - min(times)
            self.assertGreater(spread, 500.0, msg=f"L{lap} spread only {spread}ms")

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

    def test_wear_curve_is_monotonic_and_saturating(self):
        """Waiting must never be free, and marginal wear must ease with age."""
        from strategy_engineer.sim import _wear_ms, compound_deg_total_ms

        total = compound_deg_total_ms(2.0)
        wear = [_wear_ms(total, life) for life in range(0, 40)]
        for lo, hi in zip(wear, wear[1:]):
            self.assertGreater(hi, lo, msg=f"non-monotonic wear: {wear[:12]}")
        early = wear[5] - wear[4]
        late = wear[30] - wear[29]
        self.assertGreater(early, late)
        self.assertLess(wear[-1], total)

    def test_softer_compound_wears_faster(self):
        from strategy_engineer.sim import compound_deg_total_ms

        soft, medium, hard = (compound_deg_total_ms(c) for c in (1.0, 2.0, 3.0))
        self.assertGreater(soft, medium)
        self.assertGreater(medium, hard)

    @staticmethod
    def _live_deg_replay(lap_times: list[float], *, sc_laps=()) -> RaceReplay:
        replay = RaceReplay(DATASET)
        race_id, driver_id = 9999, 1
        replay._loaded = True
        replay._total_laps[race_id] = 50
        replay._sc_periods[race_id] = [
            (lap, lap) for lap in sc_laps
        ]
        for lap, milliseconds in enumerate(lap_times, start=2):
            replay._laps[race_id][driver_id][lap] = {
                "position": 1,
                "milliseconds": int(milliseconds),
                "time": "",
            }
            replay._tyres[(race_id, driver_id, lap)] = {
                "compound": "MEDIUM",
                "tyreLife": lap - 1,
            }
        return replay

    def test_live_deg_falls_back_until_enough_clean_laps(self):
        from strategy_engineer.sim import estimate_live_stint_deg

        replay = self._live_deg_replay([90_000, 90_100, 90_200, 90_300])
        estimate = estimate_live_stint_deg(
            replay, 9999, 1, 5, 50, compound_id=2.0, tyre_life=4.0
        )
        self.assertEqual(estimate.source, "compound_prior")
        self.assertEqual(estimate.sample_count, 4)

    def test_live_deg_learns_current_stint_slope(self):
        from strategy_engineer.sim import estimate_live_stint_deg

        # +250ms raw/lap; after fuel correction this is clearly above the prior.
        replay = self._live_deg_replay(
            [90_000 + 250 * i for i in range(8)]
        )
        estimate = estimate_live_stint_deg(
            replay, 9999, 1, 9, 50, compound_id=2.0, tyre_life=8.0
        )
        self.assertEqual(estimate.source, "live_stint")
        self.assertEqual(estimate.sample_count, 8)
        self.assertGreater(
            estimate.slope_ms_per_lap, estimate.prior_slope_ms_per_lap
        )

    def test_live_deg_ignores_sc_and_slow_laps(self):
        from strategy_engineer.sim import estimate_live_stint_deg

        replay = self._live_deg_replay(
            [90_000, 90_100, 120_000, 90_300, 90_400, 90_500],
            sc_laps={4},
        )
        estimate = estimate_live_stint_deg(
            replay, 9999, 1, 7, 50, compound_id=2.0, tyre_life=6.0
        )
        self.assertEqual(estimate.source, "live_stint")
        self.assertEqual(estimate.sample_count, 5)
        self.assertLess(estimate.slope_ms_per_lap, 250.0)

    def test_live_deg_detects_sustained_cliff(self):
        from strategy_engineer.sim import estimate_live_stint_deg

        lap_times = [90_000 + 40 * i for i in range(7)]
        lap_times.extend([90_700, 91_200, 91_800])
        replay = self._live_deg_replay(lap_times)
        estimate = estimate_live_stint_deg(
            replay, 9999, 1, 11, 50, compound_id=2.0, tyre_life=10.0
        )
        self.assertTrue(estimate.cliff_detected)
        self.assertGreaterEqual(estimate.cliff_extra_ms_per_lap, 180.0)

    def test_race_time_has_interior_optimum(self):
        """A monotone curve means "later is always better" and the stop never lands."""
        import numpy as np

        from strategy_engineer.sim import (
            SimSnapshot,
            _default_deg,
            _recent_green_pace_ms,
            build_pace_schedule,
        )

        if 1052 not in self.replay._laps:
            self.skipTest("race 1052 not in dataset")
        state = self.replay.get_state(1052, 1, 6)
        cumul = self.replay._cumulative_times(1052, 6)
        snap = SimSnapshot.from_state(state, float(cumul[1]))
        green = _recent_green_pace_ms(
            self.replay, 1052, 1, 6, state.total_laps, board_sc_active=False
        )
        pace = build_pace_schedule(_default_deg(), snap, green_reference_ms=green)
        remaining = state.total_laps - state.lap
        times = [
            simulate_ego_finish_ms(
                snap,
                pit_after_laps=n,
                pace=pace,
                pit_loss_ms=22_000.0,
                rng=np.random.default_rng(1),
                noise_ms=0.0,
            )
            for n in range(remaining)
        ]
        best_n = min(range(len(times)), key=lambda i: times[i])
        self.assertGreater(best_n, 0)
        self.assertLess(best_n, remaining - 1, msg="optimum sits at the grid edge")

    def test_optimal_stop_lap_is_time_consistent(self):
        """Re-asking a lap later must not push the stop a lap further away."""
        import numpy as np

        from strategy_engineer.sim import (
            SimSnapshot,
            _default_deg,
            _recent_green_pace_ms,
            build_pace_schedule,
        )

        if 1052 not in self.replay._laps:
            self.skipTest("race 1052 not in dataset")

        def best_stop_lap(lap: int) -> int:
            state = self.replay.get_state(1052, 1, lap)
            cumul = self.replay._cumulative_times(1052, lap)
            snap = SimSnapshot.from_state(state, float(cumul[1]))
            green = _recent_green_pace_ms(
                self.replay, 1052, 1, lap, state.total_laps, board_sc_active=False
            )
            pace = build_pace_schedule(_default_deg(), snap, green_reference_ms=green)
            n = min(
                range(state.total_laps - lap),
                key=lambda k: simulate_ego_finish_ms(
                    snap,
                    pit_after_laps=k,
                    pace=pace,
                    pit_loss_ms=22_000.0,
                    rng=np.random.default_rng(1),
                    noise_ms=0.0,
                ),
            )
            return lap + max(n, 1)

        # Laps 15-19 are one fresh stint, so the ideal stop lap is the same lap
        # whichever of them we ask from. It used to march forward ~2 laps per lap.
        targets = [best_stop_lap(lap) for lap in range(15, 20)]
        self.assertLessEqual(
            max(targets) - min(targets), 2, msg=f"stop lap drifts: {targets}"
        )

    def test_hold_plan_option_targets_committed_lap(self):
        opts = build_default_options(30, plan_offset=6)
        hold = [o for o in opts if o.label == "hold_plan"]
        self.assertEqual(len(hold), 1)
        self.assertEqual(hold[0].pit_after_laps, 6)
        # Offsets of 0/1 are already covered by pit_next_lap, which carries the
        # correct pit action.
        self.assertNotIn(
            "hold_plan", [o.label for o in build_default_options(30, plan_offset=1)]
        )

    def test_hold_plan_card_wins_near_ties(self):
        """Without a switching cost the winner is re-drawn from noise every lap."""
        if 1052 not in self.replay._laps:
            self.skipTest("race 1052 not in dataset")
        state = self.replay.get_state(1052, 1, 18)
        cards = simulate_strategy_cards(
            self.replay, state, n_rolls=64, seed=42, plan_target_lap=32
        )
        hold = next(c for c in cards if c.label == "hold_plan")
        self.assertEqual(hold.planned_pit_lap, 32)
        self.assertEqual(oracle_best(cards).label, "hold_plan")

    def test_committed_plan_does_not_slide(self):
        """Replaying with memory, the target stop must hold, not advance each lap."""
        from strategy_engineer.memory import RaceMemoryStore

        if 1052 not in self.replay._laps:
            self.skipTest("race 1052 not in dataset")
        pit_laps = self.replay.pit_laps(1052, 1)
        memory = RaceMemoryStore()
        targets: list[tuple[int, int]] = []
        for lap in range(14, 25):
            state = self.replay.get_state(1052, 1, lap)
            target = memory.committed_pit_lap(
                1052, 1, current_lap=lap, pit_laps=pit_laps
            )
            best = oracle_best(
                simulate_strategy_cards(
                    self.replay, state, n_rolls=64, seed=42, plan_target_lap=target
                )
            )
            memory.record_sim_decision(
                state,
                action="pit" if best.label == "pit_next_lap" else "stay",
                chosen_option_id=best.option_id,
                chosen_label=best.label,
                oracle_option_id=best.option_id,
                rationale="",
                sim=None,
                mean_finish_pos=best.mean_finish_pos,
                planned_pit_lap=best.planned_pit_lap,
            )
            targets.append((lap, best.planned_pit_lap or 0))
        moves = sum(1 for (_, a), (_, b) in zip(targets, targets[1:]) if b > a)
        self.assertLessEqual(moves, 1, msg=f"stop kept sliding: {targets}")

    def test_projected_finish_position_chain(self):
        state = self.replay.get_state(1052, 1, 6) if 1052 in self.replay._laps else self.state
        out = projected_finish_position(10, state)
        self.assertGreaterEqual(out, 1.0)
        self.assertLessEqual(out, float(state.drivers_on_track))


if __name__ == "__main__":
    unittest.main()
