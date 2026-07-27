"""Phase 2 frozen-eval helpers: sample decision points + metrics."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from race_engineer.crew_chief import CrewChiefDecision
from race_engineer.llm import CrewChiefBackend
from race_engineer.pit_baseline import compute_metrics
from race_engineer.replay import RaceReplay
from race_engineer.state import RaceState

_DEFAULT_FROZEN = (
    Path(__file__).resolve().parents[1] / "artifacts" / "eval" / "frozen_races.json"
)
_DEFAULT_RESULTS = Path(__file__).resolve().parents[2] / "dataset" / "results.csv"


@dataclass(frozen=True)
class EvalPoint:
    race_id: int
    driver_id: int
    lap: int
    y_true: int


def load_frozen_race_ids(path: Path | None = None) -> list[int]:
    p = path or _DEFAULT_FROZEN
    data = json.loads(p.read_text(encoding="utf-8"))
    return [int(r["race_id"]) for r in data["races"]]


def pick_drivers_for_race(
    replay: RaceReplay,
    race_id: int,
    results_path: Path | None = None,
) -> list[int]:
    """Winner (positionOrder=1) + a midfield finisher (~P10) with lap data."""
    path = results_path or _DEFAULT_RESULTS
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["raceId"]) != race_id:
                continue
            try:
                order = int(row["positionOrder"])
            except ValueError:
                continue
            rows.append((order, int(row["driverId"])))
    rows.sort()
    if not rows:
        return replay.drivers_in_race(race_id)[:2]

    winner = rows[0][1]
    mid = None
    for order, did in rows:
        if 8 <= order <= 12 and did != winner:
            mid = did
            break
    if mid is None:
        for order, did in rows:
            if did != winner:
                mid = did
                break
    drivers = [winner]
    if mid is not None:
        drivers.append(mid)
    # Keep only drivers with lap data.
    return [d for d in drivers if replay.available_laps(race_id, d)]


def build_eval_points(
    replay: RaceReplay,
    race_ids: Sequence[int],
    *,
    n_samples: int = 150,
    seed: int = 42,
) -> list[EvalPoint]:
    """
    Stratified sample: keep all pit-next positives from frozen drivers,
    fill remainder with negatives.
    """
    rng = random.Random(seed)
    positives: list[EvalPoint] = []
    negatives: list[EvalPoint] = []

    for rid in race_ids:
        for did in pick_drivers_for_race(replay, rid):
            pit_laps = set(replay._pits.get(rid, {}).get(did, []))
            laps = replay.available_laps(rid, did)
            if len(laps) < 2:
                continue
            max_lap = laps[-1]
            for lap in laps:
                if lap >= max_lap:
                    continue
                y = 1 if (lap + 1) in pit_laps else 0
                point = EvalPoint(rid, did, lap, y)
                if y == 1:
                    positives.append(point)
                else:
                    negatives.append(point)

    points = list(positives)
    n_neg = max(0, n_samples - len(points))
    if negatives:
        points.extend(rng.sample(negatives, min(n_neg, len(negatives))))
    rng.shuffle(points)
    return points[:n_samples] if len(points) > n_samples else points


def run_crew_chief_eval(
    replay: RaceReplay,
    backend: CrewChiefBackend,
    points: Sequence[EvalPoint],
) -> dict[str, Any]:
    y_true: list[int] = []
    y_pred: list[int] = []
    y_score: list[float] = []
    schema_ok = 0
    errors: list[str] = []
    decisions: list[dict[str, Any]] = []

    for i, pt in enumerate(points):
        state = replay.get_state(pt.race_id, pt.driver_id, pt.lap)
        try:
            decision = backend.decide(state)
            schema_ok += 1
            pred = decision.pit_next
            # Soft score: 1.0 for pit, 0.0 for stay (no calibrated probs from LLM).
            score = float(pred)
            row = {
                "race_id": pt.race_id,
                "driver_id": pt.driver_id,
                "lap": pt.lap,
                "y_true": pt.y_true,
                "y_pred": pred,
                "score": score,
                "action": decision.action,
                "tyre": decision.tyre,
                "push": decision.push,
                "reason": decision.reason,
                "driver_message": decision.driver_message,
                "rationale": decision.rationale,
                "schema_valid": True,
            }
        except Exception as exc:  # noqa: BLE001
            pred = 0
            score = 0.0
            if len(errors) < 20:
                errors.append(
                    f"race={pt.race_id} driver={pt.driver_id} lap={pt.lap}: {exc}"
                )
            row = {
                "race_id": pt.race_id,
                "driver_id": pt.driver_id,
                "lap": pt.lap,
                "y_true": pt.y_true,
                "y_pred": pred,
                "score": score,
                "action": None,
                "tyre": None,
                "push": None,
                "rationale": None,
                "schema_valid": False,
                "error": str(exc),
            }
        y_true.append(pt.y_true)
        y_pred.append(pred)
        y_score.append(score)
        decisions.append(row)

    import numpy as np

    yt = np.asarray(y_true, dtype=np.int64)
    ys = np.asarray(y_score, dtype=np.float64)
    # Threshold 0.5 maps stay/pit scores.
    metrics = compute_metrics(yt, ys, threshold=0.5)
    n = len(points)
    return {
        "backend": getattr(backend, "name", type(backend).__name__),
        "n": n,
        "schema_validity": schema_ok / n if n else 0.0,
        "schema_ok": schema_ok,
        "pit_metrics": metrics,
        "errors": errors,
        "decisions": decisions,
    }
