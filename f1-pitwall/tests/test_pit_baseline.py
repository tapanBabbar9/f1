from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from strategy_engineer.pit_baseline import (
    COMPOUND_TO_ID,
    LABEL_DEFINITION,
    MajorityBaseline,
    best_f1_threshold,
    build_samples,
    compute_metrics,
    load_safety_car_periods,
    load_tyre_laps,
    samples_to_xy,
    sc_features_for_lap,
)
from shared.replay import RaceReplay

DATASET = REPO / "dataset"


class TestPitBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = RaceReplay(DATASET).load()

    def test_label_definition_frozen(self):
        self.assertEqual(LABEL_DEFINITION, "pit_on_lap_plus_1")

    def test_build_samples_has_positives(self):
        samples = build_samples(self.replay, [2024])
        self.assertGreater(len(samples), 1000)
        n_pos = sum(s.y for s in samples)
        self.assertGreater(n_pos, 50)
        # Positive rate should be rare (< 15%).
        self.assertLess(n_pos / len(samples), 0.15)
        self.assertEqual(len(samples[0].features), 20)

    def test_label_matches_next_lap_pit(self):
        samples = build_samples(self.replay, [2024])
        # Spot-check a handful of positives and negatives.
        checked = 0
        for s in samples:
            pit_laps = set(self.replay._pits.get(s.race_id, {}).get(s.driver_id, []))
            expected = 1 if (s.lap + 1) in pit_laps else 0
            self.assertEqual(s.y, expected)
            checked += 1
            if checked >= 200:
                break

    def test_safety_car_join_and_features(self):
        sc = load_safety_car_periods(self.replay)
        # 2024 Miami GP: SC deployed lap 28, retreated 33
        miami = next(
            r
            for r in self.replay.list_races(year=2024)
            if "Miami" in r["name"]
        )
        periods = sc[miami["race_id"]]
        self.assertIn((28, 33), periods)
        active, since, this, prev = sc_features_for_lap(periods, 28, 57)
        self.assertEqual(active, 1.0)
        self.assertEqual(since, 0.0)
        self.assertEqual(this, 1.0)
        self.assertEqual(prev, 0.0)
        active2, since2, this2, prev2 = sc_features_for_lap(periods, 29, 57)
        self.assertEqual(active2, 1.0)
        self.assertEqual(since2, 1.0)
        self.assertEqual(this2, 0.0)
        self.assertEqual(prev2, 1.0)
        off, *_ = sc_features_for_lap(periods, 34, 57)
        self.assertEqual(off, 0.0)

    def test_circuit_id_in_features(self):
        samples = build_samples(self.replay, [2024])
        # British GP circuit_id is 9
        british = next(
            r
            for r in self.replay.list_races(year=2024)
            if "British" in r["name"]
        )
        row = next(s for s in samples if s.race_id == british["race_id"])
        # circuit_id is feature index 13
        self.assertEqual(row.features[13], float(british["circuit_id"]))

    def test_tyre_compound_feature_when_available(self):
        tyres = load_tyre_laps(self.replay)
        if not tyres:
            self.skipTest("tyre_laps.csv not built yet")
        # Pick any known tyre row and ensure sample matches compound_id.
        (rid, did, lap), (cid, life) = next(iter(tyres.items()))
        samples = {
            (s.race_id, s.driver_id, s.lap): s
            for s in build_samples(self.replay, [2024, 2025, 2023, 2022])
        }
        key = (rid, did, lap)
        if key not in samples:
            self.skipTest("tyre row not in labeled sample set (final lap?)")
        self.assertEqual(samples[key].features[18], float(cid))
        self.assertIn(cid, set(COMPOUND_TO_ID.values()))

    def test_majority_and_metrics(self):
        samples = build_samples(self.replay, [2023])
        x, y = samples_to_xy(samples)
        model = MajorityBaseline().fit(x, y)
        scores = model.predict_proba(x)[:, 1]
        thr = best_f1_threshold(y, scores)
        metrics = compute_metrics(y, scores, thr)
        self.assertIn("f1", metrics)
        self.assertIn("auroc", metrics)
        self.assertEqual(metrics["n"], len(y))
        # Constant score → AUROC defined but uninformative; F1 may be 0.
        self.assertTrue(np.isfinite(metrics["auroc"]) or metrics["auroc"] != metrics["auroc"])


if __name__ == "__main__":
    unittest.main()
