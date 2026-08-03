from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from shared.integrity import run_integrity_check
from shared.replay import RaceReplay

DATASET = REPO / "dataset"


class TestRaceReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = RaceReplay(DATASET).load()

    def test_2025_season_present(self):
        races_2025 = self.replay.list_races(year=2025)
        self.assertEqual(len(races_2025), 24)

    def test_known_2024_abu_dhabi_state(self):
        matches = [
            r
            for r in self.replay.list_races(year=2024)
            if "Abu Dhabi" in r["name"]
        ]
        self.assertTrue(matches)
        rid = matches[0]["race_id"]
        drivers = self.replay.drivers_in_race(rid)
        self.assertTrue(drivers)
        did = drivers[0]
        lap = 10
        self.assertIn(lap, self.replay.available_laps(rid, did))
        state = self.replay.get_state(rid, did, lap)
        self.assertEqual(state.race_id, rid)
        self.assertEqual(state.lap, lap)
        self.assertGreaterEqual(state.position, 1)
        self.assertEqual(state.year, 2024)

    def test_pit_flag_matches_table(self):
        # Find any known pit and assert flag.
        rid = next(
            r["race_id"]
            for r in self.replay.list_races(year=2024)
            if self.replay.drivers_in_race(r["race_id"])
        )
        found = False
        for did in self.replay.drivers_in_race(rid):
            pit_laps = self.replay._pits.get(rid, {}).get(did, [])
            if not pit_laps:
                continue
            lap = pit_laps[0]
            state = self.replay.get_state(rid, did, lap)
            self.assertTrue(state.pit_this_lap)
            self.assertEqual(state.stint_age_laps, 0)
            found = True
            break
        self.assertTrue(found, "expected at least one pit stop in first 2024 race")

    def test_integrity_rate_baseline(self):
        report = run_integrity_check(
            self.replay,
            n_samples=500,
            seed=0,
            years=(2023, 2024, 2025),
        )
        self.assertGreaterEqual(report.integrity_rate, 0.99, report.summary())


if __name__ == "__main__":
    unittest.main()
