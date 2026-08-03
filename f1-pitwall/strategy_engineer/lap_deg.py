"""Phase 4 (lap degradation): predict next-lap pace from stint context."""

from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from strategy_engineer.pit_baseline import COMPOUND_TO_ID, load_tyre_laps
from shared.replay import RaceReplay
from shared.state import RaceState

FEATURE_NAMES = [
    "circuit_id",
    "stint_age",
    "tyre_life",
    "compound_id",
    "lap_fraction",
    "remaining_laps",
    "last_lap_s",
    "prev_lap_s",
    "prev2_lap_s",
    "lap_delta_s",
    "sc_active",
]

_DEFAULT_ARTIFACT = (
    Path(__file__).resolve().parents[1] / "artifacts" / "lap_deg" / "model.pkl"
)


@dataclass(frozen=True)
class LapDegSample:
    race_id: int
    driver_id: int
    lap: int  # current lap (features from this lap; target = next lap)
    year: int
    circuit_id: int
    y_next_ms: float
    features: tuple[float, ...]


def _pit_set(replay: RaceReplay, race_id: int, driver_id: int) -> set[int]:
    return set(replay._pits.get(race_id, {}).get(driver_id, []))


def build_lap_deg_samples(
    replay: RaceReplay,
    years: list[int],
    *,
    tyre_by_key: dict[tuple[int, int, int], tuple[int, float]] | None = None,
) -> list[LapDegSample]:
    """
    One sample per (race, driver, lap) where lap+1 exists and is not a pit lap.

    Predict y = milliseconds of lap+1 from features at `lap`.
    Skip current or next lap if either is a pit stop (in/out distortion).
    """
    if tyre_by_key is None:
        tyre_by_key = load_tyre_laps(replay)

    year_set = set(years)
    out: list[LapDegSample] = []

    for race_id, race in replay._races.items():
        year = int(race["year"])
        if year not in year_set:
            continue
        circuit_id = int(race["circuitId"])
        total = replay._total_laps.get(race_id)
        if not total or total < 5:
            continue

        for driver_id, laps in replay._laps.get(race_id, {}).items():
            pits = _pit_set(replay, race_id, driver_id)
            sorted_laps = sorted(laps.keys())
            for i, lap in enumerate(sorted_laps):
                if i + 1 >= len(sorted_laps):
                    continue
                next_lap = sorted_laps[i + 1]
                if next_lap != lap + 1:
                    continue
                if lap in pits or next_lap in pits:
                    continue
                # Need previous laps for lags when available.
                row = laps[lap]
                next_ms = float(laps[next_lap]["milliseconds"])
                # Filter extreme outliers (safety / red-flag weirdness).
                if next_ms < 50_000 or next_ms > 200_000:
                    continue
                if row["milliseconds"] < 50_000 or row["milliseconds"] > 200_000:
                    continue

                stint_age, _, _ = replay._stint_age(race_id, driver_id, lap)
                compound_id, tyre_life = tyre_by_key.get(
                    (race_id, driver_id, lap), (0, float(stint_age))
                )
                prev_ms = float(laps[lap - 1]["milliseconds"]) if (lap - 1) in laps else float(
                    row["milliseconds"]
                )
                prev2_ms = (
                    float(laps[lap - 2]["milliseconds"])
                    if (lap - 2) in laps
                    else prev_ms
                )
                last_s = row["milliseconds"] / 1000.0
                prev_s = prev_ms / 1000.0
                prev2_s = prev2_ms / 1000.0
                sc_active, _, _, _ = replay._sc_flags(race_id, lap, total)

                feats = (
                    float(circuit_id),
                    float(stint_age),
                    float(tyre_life),
                    float(compound_id),
                    float(lap / total),
                    float(total - lap),
                    last_s,
                    prev_s,
                    prev2_s,
                    last_s - prev_s,
                    1.0 if sc_active else 0.0,
                )
                out.append(
                    LapDegSample(
                        race_id=race_id,
                        driver_id=driver_id,
                        lap=lap,
                        year=year,
                        circuit_id=circuit_id,
                        y_next_ms=next_ms,
                        features=feats,
                    )
                )
    return out


def split_by_year(
    samples: list[LapDegSample],
    *,
    train_years: range | list[int],
    val_years: range | list[int],
    test_years: range | list[int],
) -> tuple[list[LapDegSample], list[LapDegSample], list[LapDegSample]]:
    tr = set(train_years)
    va = set(val_years)
    te = set(test_years)
    train, val, test = [], [], []
    for s in samples:
        if s.year in tr:
            train.append(s)
        elif s.year in va:
            val.append(s)
        elif s.year in te:
            test.append(s)
    return train, val, test


def _xy(samples: list[LapDegSample]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([s.features for s in samples], dtype=np.float64)
    y = np.asarray([s.y_next_ms for s in samples], dtype=np.float64)
    return x, y


def train_hgb(train: list[LapDegSample]) -> HistGradientBoostingRegressor:
    x, y = _xy(train)
    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.08,
        max_iter=200,
        random_state=42,
    )
    model.fit(x, y)
    return model


def predict_ms(model: HistGradientBoostingRegressor, x: np.ndarray) -> np.ndarray:
    return model.predict(x)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    err = y_pred - y_true
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    # Avoid div by zero.
    mape = float(np.mean(abs_err / np.maximum(y_true, 1.0)) * 100.0)
    rmse = float(np.sqrt(np.mean(err**2)))
    return {
        "n": int(len(y_true)),
        "mae_ms": mae,
        "mape_pct": mape,
        "rmse_ms": rmse,
        "median_ae_ms": float(np.median(abs_err)),
    }


def metrics_by_circuit(
    samples: list[LapDegSample],
    y_pred: np.ndarray,
) -> dict[str, Any]:
    by: dict[int, list[tuple[float, float]]] = {}
    for s, pred in zip(samples, y_pred):
        by.setdefault(s.circuit_id, []).append((s.y_next_ms, float(pred)))
    out: dict[str, Any] = {}
    for cid, pairs in sorted(by.items()):
        yt = np.asarray([p[0] for p in pairs], dtype=np.float64)
        yp = np.asarray([p[1] for p in pairs], dtype=np.float64)
        out[str(cid)] = regression_metrics(yt, yp)
    return out


def features_from_state(state: RaceState) -> np.ndarray:
    """Build the feature vector for the current RaceState (predict next lap)."""
    compound = (state.tyre_compound or "UNKNOWN").strip().upper()
    compound_id = float(COMPOUND_TO_ID.get(compound, 0))
    tyre_life = float(
        state.tyre_life if state.tyre_life is not None else state.stint_age_laps
    )
    window = list(state.last_lap_times_ms) if state.last_lap_times_ms else []
    last_ms = float(state.last_lap_time_ms)
    prev_ms = float(window[-2]) if len(window) >= 2 else last_ms
    prev2_ms = float(window[-3]) if len(window) >= 3 else prev_ms
    last_s = last_ms / 1000.0
    prev_s = prev_ms / 1000.0
    prev2_s = prev2_ms / 1000.0
    total = max(state.total_laps, 1)
    feats = np.asarray(
        [
            [
                float(state.circuit_id),
                float(state.stint_age_laps),
                tyre_life,
                compound_id,
                float(state.lap / total),
                float(total - state.lap),
                last_s,
                prev_s,
                prev2_s,
                last_s - prev_s,
                1.0 if state.sc_active else 0.0,
            ]
        ],
        dtype=np.float64,
    )
    return feats


@dataclass
class LapDegModel:
    model: HistGradientBoostingRegressor
    feature_names: list[str]
    train_years: list[int]
    val_years: list[int]
    test_years: list[int]

    def predict_next_lap_ms(self, state: RaceState) -> float:
        x = features_from_state(state)
        return float(predict_ms(self.model, x)[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "feature_names": self.feature_names,
                    "train_years": self.train_years,
                    "val_years": self.val_years,
                    "test_years": self.test_years,
                },
                f,
            )

    @classmethod
    def load(cls, path: Path | None = None) -> LapDegModel:
        p = path or _DEFAULT_ARTIFACT
        with p.open("rb") as f:
            bundle = pickle.load(f)
        return cls(
            model=bundle["model"],
            feature_names=list(bundle.get("feature_names", FEATURE_NAMES)),
            train_years=list(bundle.get("train_years", [])),
            val_years=list(bundle.get("val_years", [])),
            test_years=list(bundle.get("test_years", [])),
        )


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def save_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(payload), indent=2) + "\n", encoding="utf-8")
