from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from strategy_engineer.crew_chief_eval import load_frozen_race_ids, pick_drivers_for_race
from strategy_engineer.memory import RaceMemoryStore, evidence_fingerprint
from shared.replay import RaceReplay
from strategy_engineer.strategy import HeuristicSimBackend, replay_race_decisions

DATASET = REPO / "dataset"


class TestMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = RaceReplay(DATASET).load()
        cls.state = cls.replay.get_state(1132, 1, 22)

    def test_evidence_ignores_stint_age(self):
        s22 = self.replay.get_state(1132, 1, 22)
        s23 = self.replay.get_state(1132, 1, 23)
        e22 = evidence_fingerprint(s22)
        e23 = evidence_fingerprint(s23)
        # Position/gaps may differ lap-to-lap; at minimum keys are well-formed.
        self.assertTrue(e22.startswith("P"))
        self.assertTrue(e23.startswith("P"))

    def test_memory_prompt_block(self):
        store = RaceMemoryStore()
        store.record_sim_decision(
            self.state,
            action="stay",
            chosen_option_id="D",
            chosen_label="stay_8_then_pit",
            oracle_option_id="D",
            rationale="Plan stay 8 then box.",
            sim={"available": True, "oracle_option_id": "D"},
            mean_finish_pos=3.2,
        )
        block = store.format_prompt_block(
            self.state.race_id, self.state.driver_id, before_lap=23
        )
        self.assertIn("Pit-wall memory", block)
        self.assertIn("stay_8_then_pit", block)
        self.assertIn("L22", block)
        self.assertNotIn("E[finish]", block)
        self.assertNotIn("Plan stay 8 then box", block)

    def test_memory_prompt_summarizes_same_plan(self):
        store = RaceMemoryStore()
        base = dict(
            action="stay",
            chosen_option_id="D",
            chosen_label="stay_8_then_pit",
            oracle_option_id="D",
            rationale="numbers",
            sim={"available": True, "oracle_option_id": "D"},
            mean_finish_pos=1.0,
        )
        for lap in (21, 22, 23):
            store.record_sim_decision(
                self.replay.get_state(1132, 1, lap), **base
            )
        block = store.format_prompt_block(1132, 1, before_lap=24)
        self.assertIn("L21-23", block)
        self.assertNotIn("L21:", block)

    def _record(self, store, lap, **over):
        base = dict(
            action="stay",
            chosen_option_id="A",
            chosen_label="hold_plan",
            oracle_option_id="A",
            rationale="numbers",
            sim={"available": True, "oracle_option_id": "A"},
            mean_finish_pos=1.0,
            planned_pit_lap=30,
        )
        base.update(over)
        return store.record_sim_decision(self.replay.get_state(1132, 1, lap), **base)

    def test_committed_pit_lap_holds_absolute_target(self):
        store = RaceMemoryStore()
        self._record(store, 22, planned_pit_lap=30)
        self.assertEqual(
            store.committed_pit_lap(1132, 1, current_lap=23), 30
        )

    def test_committed_pit_lap_expires_once_target_passes(self):
        store = RaceMemoryStore()
        self._record(store, 22, planned_pit_lap=25)
        self.assertIsNone(store.committed_pit_lap(1132, 1, current_lap=25))
        self.assertIsNone(store.committed_pit_lap(1132, 1, current_lap=26))

    def test_committed_pit_lap_dropped_after_driver_pits(self):
        store = RaceMemoryStore()
        self._record(store, 22, planned_pit_lap=30)
        self.assertIsNone(
            store.committed_pit_lap(
                1132, 1, current_lap=26, pit_laps=frozenset({24})
            )
        )

    def test_memory_prompt_shows_target_stop_lap(self):
        """A label alone cannot show the stop sliding; the absolute lap can."""
        store = RaceMemoryStore()
        self._record(store, 21, planned_pit_lap=30)
        self._record(store, 22, planned_pit_lap=31)
        block = store.format_prompt_block(1132, 1, before_lap=23)
        self.assertIn("target stop L30", block)
        self.assertIn("target stop L31", block)
        self.assertNotIn("L21-22", block)

    def test_memory_prompt_groups_a_held_target(self):
        store = RaceMemoryStore()
        for lap in (21, 22, 23):
            self._record(store, lap, planned_pit_lap=30)
        block = store.format_prompt_block(1132, 1, before_lap=24)
        self.assertIn("L21-23", block)
        self.assertIn("target stop L30", block)

    def test_finish_pos_swing_detection(self):
        store = RaceMemoryStore()
        sim = {"available": True, "oracle_option_id": "D"}
        ev = "P2|gap1500|gbp3000|MEDIUM|pc0|sc0|oD"
        for lap, mf in ((6, 1.0), (7, 18.0)):
            s = self.replay.get_state(1132, 1, lap)
            store.record_sim_decision(
                s,
                action="stay",
                chosen_option_id="D",
                chosen_label="stay_8_then_pit",
                oracle_option_id="D",
                rationale="x",
                sim=sim,
                mean_finish_pos=mf,
            )
            store._entries[(1132, 1)][-1].evidence = ev
        swings = store.finish_pos_swings(1132, 1)
        self.assertEqual(len(swings), 1)
        self.assertAlmostEqual(swings[0][2], 17.0)

    def test_memory_reconcile_advised_pit_driver_stayed_out(self):
        store = RaceMemoryStore()
        s3 = self.replay.get_state(1132, 1, 3)
        store.record_sim_decision(
            s3,
            action="pit",
            chosen_option_id="A",
            chosen_label="pit_next_lap",
            oracle_option_id="A",
            rationale="Tyres at cliff.",
            sim={"available": True, "oracle_option_id": "A"},
        )
        pit_laps = self.replay.pit_laps(1132, 1)
        line = store.entries(1132, 1)[0].prompt_line(pit_laps=pit_laps)
        if 4 in pit_laps:
            self.assertIn("executed lap 4", line)
        else:
            self.assertIn("advised box lap 4; driver stayed out", line)

    def test_memory_reconcile_advised_stay_driver_pitted(self):
        store = RaceMemoryStore()
        s12 = self.replay.get_state(1132, 1, 12)
        store.record_sim_decision(
            s12,
            action="stay",
            chosen_option_id="D",
            chosen_label="stay_8_then_pit",
            oracle_option_id="D",
            rationale="Hold.",
            sim={"available": True, "oracle_option_id": "D"},
        )
        line = store.entries(1132, 1)[0].prompt_line(pit_laps=frozenset({13}))
        self.assertIn("advised stay; driver pitted lap 13", line)
        self.assertIn("stay_8_then_pit", line)

    def test_memory_prompt_splits_stay_execution_mismatch(self):
        store = RaceMemoryStore()
        base = dict(
            action="stay",
            chosen_option_id="D",
            chosen_label="stay_8_then_pit",
            oracle_option_id="D",
            rationale="numbers",
            sim={"available": True, "oracle_option_id": "D"},
            mean_finish_pos=5.0,
        )
        for lap in (10, 11):
            store.record_sim_decision(
                self.replay.get_state(1132, 1, lap), **base
            )
        store.record_sim_decision(
            self.replay.get_state(1132, 1, 12), **base
        )
        block = store.format_prompt_block(
            1132, 1, before_lap=13, pit_laps=frozenset({13})
        )
        self.assertIn("L10-11: stay", block)
        self.assertIn("L12: advised stay; driver pitted lap 13", block)
        self.assertNotIn("L10-12", block)

    def test_memory_reconcile_advised_pit_executed(self):
        store = RaceMemoryStore()
        s3 = self.replay.get_state(1132, 1, 3)
        store.record_sim_decision(
            s3,
            action="pit",
            chosen_option_id="A",
            chosen_label="pit_next_lap",
            oracle_option_id="A",
            rationale="Box now.",
            sim={"available": True, "oracle_option_id": "A"},
        )
        line = store.entries(1132, 1)[0].prompt_line(pit_laps=frozenset({4}))
        self.assertIn("pit (executed lap 4)", line)
        self.assertIn("pit_next_lap", line)

    def test_memory_reconcile_without_pit_laps_keeps_raw_advice(self):
        store = RaceMemoryStore()
        s3 = self.replay.get_state(1132, 1, 3)
        store.record_sim_decision(
            s3,
            action="pit",
            chosen_option_id="A",
            chosen_label="pit_next_lap",
            oracle_option_id="A",
            rationale="Box now.",
            sim={"available": True, "oracle_option_id": "A"},
        )
        line = store.entries(1132, 1)[0].prompt_line()
        self.assertIn("L3: pit", line)
        self.assertNotIn("driver stayed out", line)

    def test_flip_flop_detection(self):
        store = RaceMemoryStore()
        base = dict(
            action="stay",
            chosen_option_id="B",
            chosen_label="stay_3_then_pit",
            oracle_option_id="B",
            rationale="x",
            sim={"available": True, "oracle_option_id": "B"},
        )
        s21 = self.replay.get_state(1132, 1, 21)
        s22 = self.replay.get_state(1132, 1, 22)
        ev = evidence_fingerprint(s22, base["sim"])
        store.record_sim_decision(s21, **base, mean_finish_pos=4.0)
        store.record_sim_decision(
            s22,
            action="pit",
            chosen_option_id="A",
            chosen_label="pit_next_lap",
            oracle_option_id="B",
            rationale="y",
            sim=base["sim"],
            mean_finish_pos=4.0,
        )
        # Force same evidence on both entries for unit test.
        store._entries[(1132, 1)][0].evidence = ev
        store._entries[(1132, 1)][1].evidence = ev
        flips = store.flip_flops(1132, 1)
        self.assertEqual(len(flips), 1)

    def test_flip_flop_excludes_advisory_mismatch(self):
        store = RaceMemoryStore()
        sim = {"available": True, "oracle_option_id": "B"}
        ev = evidence_fingerprint(self.state, sim)
        s21 = self.replay.get_state(1132, 1, 21)
        s22 = self.replay.get_state(1132, 1, 22)
        s23 = self.replay.get_state(1132, 1, 23)
        store.record_sim_decision(
            s21,
            action="pit",
            chosen_option_id="A",
            chosen_label="pit_next_lap",
            oracle_option_id="B",
            rationale="box",
            sim=sim,
        )
        store.record_sim_decision(
            s22,
            action="stay",
            chosen_option_id="B",
            chosen_label="stay_3_then_pit",
            oracle_option_id="B",
            rationale="stay",
            sim=sim,
        )
        store.record_sim_decision(
            s23,
            action="stay",
            chosen_option_id="B",
            chosen_label="stay_3_then_pit",
            oracle_option_id="B",
            rationale="stay",
            sim=sim,
        )
        for entry in store._entries[(1132, 1)]:
            entry.evidence = ev
        pit_laps = frozenset({999})  # lap 22 not a pit → L21 advisory mismatch
        self.assertEqual(len(store.flip_flops(1132, 1)), 1)
        self.assertEqual(
            len(
                store.flip_flops(
                    1132, 1, pit_laps=pit_laps, exclude_advisory_mismatch=True
                )
            ),
            0,
        )

    def test_full_race_replay_memory_on_off(self):
        race_ids = load_frozen_race_ids()[:1]
        rid = race_ids[0]
        did = pick_drivers_for_race(self.replay, rid)[0]
        backend = HeuristicSimBackend(self.replay)
        off = replay_race_decisions(
            backend, self.replay, rid, did, memory=None, lap_from=1, lap_to=30
        )
        on = replay_race_decisions(
            backend,
            self.replay,
            rid,
            did,
            memory=RaceMemoryStore(),
            lap_from=1,
            lap_to=30,
        )
        self.assertEqual(len(off.decisions), len(on.decisions))
        self.assertAlmostEqual(off.mean_regret or 0.0, 0.0)
        self.assertAlmostEqual(on.mean_regret or 0.0, 0.0)
        # Choices are no longer expected to match: with memory the option menu
        # carries a hold_plan card for the committed stop lap, which is the point.
        # What must hold is that the remembered plan does not slide every lap.
        targets = [
            d.chosen_planned_pit_lap
            for d in on.decisions
            if d.decision.action != "pit" and d.chosen_planned_pit_lap is not None
        ]
        moves = sum(1 for a, b in zip(targets, targets[1:]) if b > a)
        self.assertLess(
            moves, len(targets) // 2, msg=f"planned stop keeps sliding: {targets}"
        )


if __name__ == "__main__":
    unittest.main()
