from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from race_engineer.lap_deg import (
    FEATURE_NAMES,
    LapDegModel,
    build_lap_deg_samples,
    features_from_state,
    regression_metrics,
    split_by_year,
    train_hgb,
    _xy,
)
from race_engineer.replay import RaceReplay
from race_engineer.tools import ToolBelt

DATASET = REPO / "dataset"
MODEL = ROOT / "artifacts" / "lap_deg" / "model.pkl"


class TestLapDeg(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.replay = RaceReplay(DATASET).load()

    def test_build_samples_nonempty(self):
        samples = build_lap_deg_samples(self.replay, [2024])
        self.assertGreater(len(samples), 1000)
        self.assertEqual(len(samples[0].features), len(FEATURE_NAMES))

    def test_train_smoke_and_metrics(self):
        samples = build_lap_deg_samples(self.replay, [2023, 2024])
        train, val, test = split_by_year(
            samples,
            train_years=[2023],
            val_years=[],
            test_years=[2024],
        )
        self.assertGreater(len(train), 100)
        model = train_hgb(train[:5000])  # cap for speed
        x, y = _xy(test[:2000])
        pred = model.predict(x)
        m = regression_metrics(y, pred)
        self.assertIn("mae_ms", m)
        self.assertLess(m["mae_ms"], 15_000)  # sanity: <15s MAE

    def test_features_from_state_shape(self):
        state = self.replay.get_state(1132, 1, 22)
        x = features_from_state(state)
        self.assertEqual(x.shape, (1, len(FEATURE_NAMES)))

    @unittest.skipUnless(MODEL.exists(), "train_lap_deg.py not run yet")
    def test_predict_tool_available(self):
        # Reset cached model loader
        import race_engineer.tools as tools_mod

        tools_mod._DEG_MODEL = None
        state = self.replay.get_state(1132, 1, 22)
        belt = ToolBelt(self.replay, state)
        out = belt.predict_lap_time()
        self.assertTrue(out["available"])
        self.assertIn("predicted_next_lap_ms", out)
        deg = LapDegModel.load(MODEL)
        pred = deg.predict_next_lap_ms(state)
        self.assertAlmostEqual(out["predicted_next_lap_ms"], round(pred, 1), places=0)


if __name__ == "__main__":
    unittest.main()
