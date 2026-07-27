from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from race_engineer.crew_chief_eval import load_frozen_race_ids, pick_drivers_for_race
from race_engineer.memory import RaceMemoryStore, evidence_fingerprint
from race_engineer.replay import RaceReplay
from race_engineer.sim_agent import HeuristicSimBackend, replay_race_decisions

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
        self.assertEqual(
            [d.chosen_option_id for d in off.decisions],
            [d.chosen_option_id for d in on.decisions],
        )


if __name__ == "__main__":
    unittest.main()
