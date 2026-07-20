"""Phase 1: baseline pit-next-lap classifier (no LLM)."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from race_engineer.replay import RaceReplay

# Frozen label definition for Phase 1:
# y=1 iff the driver pits on the *next* lap (lap + 1).
LABEL_DEFINITION = "pit_on_lap_plus_1"

FEATURE_NAMES: tuple[str, ...] = (
    "lap",
    "total_laps",
    "lap_fraction",
    "remaining_laps",
    "position",
    "stint_age",
    "pit_count",
    "pit_this_lap",
    "gap_ahead_s",
    "gap_behind_s",
    "last_lap_s",
    "lap_delta_s",
    "drivers_on_track",
    "circuit_id",
    "sc_active",
    "laps_since_sc_deploy",
    "sc_deployed_this_lap",
    "sc_deployed_prev_lap",
    "compound_id",
    "tyre_life",
)

# Sentinel seconds when gap is undefined (leader / last).
_GAP_SENTINEL_S = 999.0

# FastF1 Compound → categorical id (0 = unknown / missing).
# Pre-2019 named softs kept distinct (strategy differed by colour).
COMPOUND_TO_ID: dict[str, int] = {
    "UNKNOWN": 0,
    "TEST_UNKNOWN": 0,
    "SOFT": 1,
    "MEDIUM": 2,
    "HARD": 3,
    "INTERMEDIATE": 4,
    "WET": 5,
    "SUPERSOFT": 6,
    "ULTRASOFT": 7,
    "HYPERSOFT": 8,
}

DEFAULT_SPLITS = {
    "train_years": (2018, 2019, 2020, 2021, 2022),
    "val_years": (2023,),
    "test_years": (2024, 2025),
}

# (deployed_lap, retreated_lap_inclusive_or_None_if_open_ended)
ScPeriod = tuple[int, int | None]
# (compound_id, tyre_life) keyed later by (race_id, driver_id, lap)
TyreKey = tuple[int, int, int]


@dataclass
class Sample:
    race_id: int
    driver_id: int
    lap: int
    year: int
    y: int
    features: tuple[float, ...]


def _stint_age_and_pits(
    pit_laps: Sequence[int], lap: int
) -> tuple[int, int, bool]:
    so_far = [p for p in pit_laps if p <= lap]
    pit_this = lap in pit_laps
    last_pit = so_far[-1] if so_far else 0
    return lap - last_pit, len(so_far), pit_this


def load_safety_car_periods(replay: RaceReplay) -> dict[int, list[ScPeriod]]:
    """
    Map race_id -> SC periods from dataset/safety_cars.csv.

    Rows join via "{year} {name}" matching races.csv. Retreated may be blank
    (open-ended / ended after last recorded lap).
    """
    path = replay.dataset_dir / "safety_cars.csv"
    if not path.exists():
        return {}

    by_key: dict[str, int] = {}
    for race in replay.list_races():
        by_key[f"{race['year']} {race['name']}"] = race["race_id"]

    out: dict[int, list[ScPeriod]] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rid = by_key.get(row["Race"])
            if rid is None:
                continue
            deployed = int(float(row["Deployed"]))
            retreated_raw = (row.get("Retreated") or "").strip()
            retreated: int | None
            if retreated_raw == "":
                retreated = None
            else:
                retreated = int(float(retreated_raw))
            out.setdefault(rid, []).append((deployed, retreated))
    return out


def load_tyre_laps(replay: RaceReplay) -> dict[TyreKey, tuple[int, float]]:
    """
    Load dataset/tyre_laps.csv → {(raceId, driverId, lap): (compound_id, tyre_life)}.

    Missing file or row → caller should treat as UNKNOWN / tyre_life=-1.
    """
    path = replay.dataset_dir / "tyre_laps.csv"
    out: dict[TyreKey, tuple[int, float]] = {}
    if not path.exists():
        return out
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            compound = (row.get("compound") or "UNKNOWN").strip().upper()
            cid = COMPOUND_TO_ID.get(compound, 0)
            life_raw = (row.get("tyreLife") or "").strip()
            try:
                life = float(life_raw) if life_raw != "" else -1.0
            except ValueError:
                life = -1.0
            key = (int(row["raceId"]), int(row["driverId"]), int(row["lap"]))
            out[key] = (cid, life)
    return out


def sc_features_for_lap(
    periods: Sequence[ScPeriod], lap: int, total_laps: int
) -> tuple[float, float, float, float]:
    """
    Returns:
      sc_active, laps_since_sc_deploy, sc_deployed_this_lap, sc_deployed_prev_lap

    laps_since_sc_deploy is 0 on the deploy lap while active; 0 when not under SC.
    """
    sc_active = False
    laps_since = 0
    for deployed, retreated in periods:
        end = total_laps if retreated is None else retreated
        if deployed <= lap <= end:
            sc_active = True
            laps_since = lap - deployed
            break

    deployed_laps = {p[0] for p in periods}
    return (
        1.0 if sc_active else 0.0,
        float(laps_since) if sc_active else 0.0,
        1.0 if lap in deployed_laps else 0.0,
        1.0 if (lap - 1) in deployed_laps else 0.0,
    )


def build_samples(
    replay: RaceReplay,
    years: Iterable[int],
    sc_by_race: dict[int, list[ScPeriod]] | None = None,
    tyre_by_key: dict[TyreKey, tuple[int, float]] | None = None,
) -> list[Sample]:
    """
    Build labeled feature rows for the given seasons.

    Skips the final lap of each driver (no lap+1 label).
    """
    year_set = set(years)
    samples: list[Sample] = []
    if sc_by_race is None:
        sc_by_race = load_safety_car_periods(replay)
    if tyre_by_key is None:
        tyre_by_key = load_tyre_laps(replay)

    for race in replay.list_races():
        year = race["year"]
        if year not in year_set:
            continue
        rid = race["race_id"]
        circuit_id = float(race["circuit_id"])
        total_laps = replay._total_laps.get(rid, 0)
        if total_laps < 2:
            continue

        driver_laps = replay._laps.get(rid, {})
        pit_map = replay._pits.get(rid, {})
        sc_periods = sc_by_race.get(rid, [])

        # Precompute cumulative times per lap for gap features.
        # lap -> driver_id -> cumulative ms
        cumul_by_lap: dict[int, dict[int, int]] = {}
        running: dict[int, int] = {did: 0 for did in driver_laps}
        for lap in range(1, total_laps + 1):
            for did, laps in driver_laps.items():
                if lap in laps:
                    running[did] = running.get(did, 0) + laps[lap]["milliseconds"]
            cumul_by_lap[lap] = {
                did: running[did]
                for did, laps in driver_laps.items()
                if lap in laps and did in running
            }

        for did, laps in driver_laps.items():
            available = sorted(laps.keys())
            if len(available) < 2:
                continue
            pit_laps = pit_map.get(did, [])
            max_driver_lap = available[-1]

            for lap in available:
                if lap >= max_driver_lap:
                    continue  # no lap+1
                y = 1 if (lap + 1) in pit_laps else 0

                row = laps[lap]
                position = float(row["position"])
                last_ms = float(row["milliseconds"])
                prev_ms = float(laps[lap - 1]["milliseconds"]) if lap - 1 in laps else last_ms
                stint_age, pit_count, pit_this = _stint_age_and_pits(pit_laps, lap)
                sc_active, laps_since_sc, sc_this, sc_prev = sc_features_for_lap(
                    sc_periods, lap, total_laps
                )
                compound_id, tyre_life = tyre_by_key.get(
                    (rid, did, lap), (0, -1.0)
                )

                board = cumul_by_lap.get(lap, {})
                gap_ahead_s = _GAP_SENTINEL_S
                gap_behind_s = _GAP_SENTINEL_S
                if did in board:
                    ordered = sorted(board.items(), key=lambda kv: kv[1])
                    pos_idx = next(i for i, (d, _) in enumerate(ordered) if d == did)
                    my_t = board[did]
                    if pos_idx > 0:
                        gap_ahead_s = (my_t - ordered[pos_idx - 1][1]) / 1000.0
                    if pos_idx < len(ordered) - 1:
                        gap_behind_s = (ordered[pos_idx + 1][1] - my_t) / 1000.0

                feats = (
                    float(lap),
                    float(total_laps),
                    lap / total_laps,
                    float(total_laps - lap),
                    position,
                    float(stint_age),
                    float(pit_count),
                    1.0 if pit_this else 0.0,
                    gap_ahead_s,
                    gap_behind_s,
                    last_ms / 1000.0,
                    (last_ms - prev_ms) / 1000.0,
                    float(len(board)),
                    circuit_id,
                    sc_active,
                    laps_since_sc,
                    sc_this,
                    sc_prev,
                    float(compound_id),
                    float(tyre_life),
                )
                samples.append(
                    Sample(
                        race_id=rid,
                        driver_id=did,
                        lap=lap,
                        year=year,
                        y=y,
                        features=feats,
                    )
                )
    return samples


def samples_to_xy(samples: Sequence[Sample]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([s.features for s in samples], dtype=np.float64)
    y = np.asarray([s.y for s in samples], dtype=np.int64)
    return x, y


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, Any]:
    y_pred = (y_score >= threshold).astype(np.int64)
    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "n_positive": int(y_true.sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["average_precision"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["auroc"] = float("nan")
        metrics["average_precision"] = float("nan")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return metrics


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Sweep thresholds on validation scores; return threshold maximizing F1."""
    if len(y_true) == 0 or y_true.sum() == 0:
        return 0.5
    candidates = np.unique(np.concatenate([[0.01, 0.05, 0.1, 0.2, 0.3, 0.5], y_score]))
    best_t, best_f1 = 0.5, -1.0
    for t in candidates:
        f1 = f1_score(y_true, (y_score >= t).astype(np.int64), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


class MajorityBaseline:
    """Always predict the majority class (almost always stay / 0)."""

    name = "majority"

    def __init__(self) -> None:
        self.majority_ = 0
        self.pos_rate_ = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> MajorityBaseline:
        self.pos_rate_ = float(y.mean()) if len(y) else 0.0
        self.majority_ = int(self.pos_rate_ >= 0.5)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # Score = train positive rate (constant); thresholding still applies.
        p1 = np.full(len(x), self.pos_rate_, dtype=np.float64)
        return np.column_stack([1.0 - p1, p1])


def make_logistic() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def make_hgb() -> HistGradientBoostingClassifier:
    # Nominal ids, not ordinal magnitudes.
    cat = [
        FEATURE_NAMES.index("circuit_id"),
        FEATURE_NAMES.index("compound_id"),
    ]
    return HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.08,
        max_iter=200,
        class_weight="balanced",
        random_state=42,
        categorical_features=cat,
    )


def train_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> dict[str, Any]:
    """Fit majority / logistic / HGB; tune threshold on val for each."""
    models: dict[str, Any] = {}

    majority = MajorityBaseline().fit(x_train, y_train)
    models["majority"] = {
        "model": majority,
        "threshold": best_f1_threshold(y_val, majority.predict_proba(x_val)[:, 1]),
    }

    logistic = make_logistic()
    logistic.fit(x_train, y_train)
    models["logistic"] = {
        "model": logistic,
        "threshold": best_f1_threshold(y_val, logistic.predict_proba(x_val)[:, 1]),
    }

    hgb = make_hgb()
    hgb.fit(x_train, y_train)
    models["hgb"] = {
        "model": hgb,
        "threshold": best_f1_threshold(y_val, hgb.predict_proba(x_val)[:, 1]),
    }
    return models


def score_model(model: Any, x: np.ndarray) -> np.ndarray:
    return model.predict_proba(x)[:, 1]


def write_predictions_csv(
    path: Path,
    samples: Sequence[Sample],
    y_score: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            ["race_id", "driver_id", "lap", "year", "y_true", "y_pred", "score"]
        )
        for s, score, pred in zip(samples, y_score, y_pred):
            w.writerow(
                [s.race_id, s.driver_id, s.lap, s.year, s.y, int(pred), f"{score:.6f}"]
            )


def summarize_split(name: str, samples: Sequence[Sample]) -> dict[str, Any]:
    n = len(samples)
    n_pos = sum(s.y for s in samples)
    return {
        "split": name,
        "n": n,
        "n_positive": n_pos,
        "positive_rate": (n_pos / n) if n else 0.0,
        "n_races": len({s.race_id for s in samples}),
    }


def metrics_payload(
    *,
    label_definition: str,
    splits: dict[str, Any],
    split_stats: dict[str, Any],
    results: dict[str, Any],
    primary_model: str,
) -> dict[str, Any]:
    return {
        "phase": 1,
        "label_definition": label_definition,
        "feature_set": "timing+circuit+sc+compound",
        "feature_names": list(FEATURE_NAMES),
        "splits": splits,
        "split_stats": split_stats,
        "primary_model": primary_model,
        "models": results,
    }


def save_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _sanitize(obj: Any) -> Any:
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: _sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize(v) for v in obj]
        return obj

    path.write_text(json.dumps(_sanitize(payload), indent=2) + "\n", encoding="utf-8")
