"""Phase 8: evaluation harness — full-race replay + model comparison."""

from __future__ import annotations

import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

import numpy as np

from race_engineer.crew_chief import CrewChiefDecision
from race_engineer.llm import CrewChiefBackend, HeuristicBackend
from race_engineer.memory import RaceMemoryStore
from race_engineer.pit_baseline import (
    LABEL_DEFINITION,
    build_samples,
    load_safety_car_periods,
    load_tyre_laps,
    score_model,
)
from race_engineer.replay import RaceReplay
from race_engineer.sim_agent import (
    HeuristicSimBackend,
    SimAgentDecision,
    SimAwareBackend,
    get_sim_backend,
    replay_race_decisions,
)

HARNESS_VERSION = "1.0.0"
FROZEN_RACES_ID = "frozen_v1"
DEFAULT_TOLERANCE_LAPS = 2

ModelFamily = Literal["pit_next", "sim_plan"]

MODEL_FAMILIES: dict[str, ModelFamily] = {
    "hgb": "pit_next",
    "heuristic_crew": "pit_next",
    "pit_next_sim": "pit_next",
    "heuristic_sim": "sim_plan",
    "openai_sim": "sim_plan",
}

_DEFAULT_MODELS = Path(__file__).resolve().parents[1] / "artifacts" / "pit_baseline" / "models.pkl"
_STAY_N_RE = re.compile(r"stay_(\d+)_then_pit$")


def model_family(model_id: str) -> ModelFamily:
    fam = MODEL_FAMILIES.get(model_id.strip().lower())
    if fam is None:
        raise ValueError(f"unknown model family for {model_id!r}")
    return fam


def planned_pit_lap(
    lap: int,
    *,
    action: str,
    label: str | None = None,
) -> int | None:
    """Map a lap decision to the lap the agent plans to box (if known)."""
    if action == "pit":
        return lap + 1
    if label:
        m = _STAY_N_RE.match(label.strip())
        if m:
            return lap + int(m.group(1))
    return None


@dataclass(frozen=True)
class HarnessLapDecision:
    lap: int
    action: str
    pit_next: int
    planned_pit_lap: int | None
    chosen_label: str | None = None
    regret: float | None = None
    y_true_next_lap: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lap": self.lap,
            "action": self.action,
            "pit_next": self.pit_next,
            "planned_pit_lap": self.planned_pit_lap,
            "chosen_label": self.chosen_label,
            "regret": self.regret,
            "y_true_next_lap": self.y_true_next_lap,
        }


@dataclass
class StintScore:
    race_id: int
    driver_id: int
    stint_index: int
    stint_start_lap: int
    actual_pit_lap: int
    predicted_pit_lap: int | None
    abs_error: int | None
    within_tolerance: bool | None
    score_kind: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "race_id": self.race_id,
            "driver_id": self.driver_id,
            "stint_index": self.stint_index,
            "stint_start_lap": self.stint_start_lap,
            "actual_pit_lap": self.actual_pit_lap,
            "predicted_pit_lap": self.predicted_pit_lap,
            "abs_error": self.abs_error,
            "within_tolerance": self.within_tolerance,
            "score_kind": self.score_kind,
        }


@dataclass
class RaceDriverResult:
    model_id: str
    race_id: int
    driver_id: int
    laps: list[int]
    decisions: list[HarnessLapDecision]
    stints: list[StintScore] = field(default_factory=list)
    flip_flop_rate: float | None = None
    mean_regret: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "race_id": self.race_id,
            "driver_id": self.driver_id,
            "n_laps": len(self.laps),
            "flip_flop_rate": self.flip_flop_rate,
            "mean_regret": self.mean_regret,
            "stints": [s.to_dict() for s in self.stints],
        }


def _pit_laps(replay: RaceReplay, race_id: int, driver_id: int) -> list[int]:
    return sorted(replay._pits.get(race_id, {}).get(driver_id, []))


def _y_true_next_lap(replay: RaceReplay, race_id: int, driver_id: int, lap: int) -> int:
    return 1 if (lap + 1) in set(_pit_laps(replay, race_id, driver_id)) else 0


def _first_box_stop_lap(
    by_lap: dict[int, HarnessLapDecision],
    stint_start: int,
    before_lap: int,
) -> int | None:
    """First 'box next lap' commit in stint → predicted stop lap."""
    for lap in range(stint_start, before_lap):
        d = by_lap.get(lap)
        if d is not None and d.action == "pit":
            return lap + 1
    return None


def _plan_before_stop(
    by_lap: dict[int, HarnessLapDecision],
    stint_start: int,
    actual_pit: int,
) -> tuple[int | None, int | None]:
    """Planned stop from the lap before the real stop (fallback: latest plan in stint)."""
    ref_lap = actual_pit - 1
    if ref_lap >= stint_start:
        d = by_lap.get(ref_lap)
        if d is not None and d.planned_pit_lap is not None:
            return d.planned_pit_lap, ref_lap
    for lap in range(actual_pit - 2, stint_start - 1, -1):
        d = by_lap.get(lap)
        if d is not None and d.planned_pit_lap is not None:
            return d.planned_pit_lap, lap
    return None, None


def score_stints_pit_next(
    replay: RaceReplay,
    race_id: int,
    driver_id: int,
    decisions: Sequence[HarnessLapDecision],
    *,
    tolerance_laps: int = DEFAULT_TOLERANCE_LAPS,
) -> list[StintScore]:
    """Pit-next models: |first box-next commit stop − actual pit| per stint."""
    pits = _pit_laps(replay, race_id, driver_id)
    if not pits:
        return []
    by_lap = {d.lap: d for d in decisions}
    out: list[StintScore] = []
    stint_start = 1
    for idx, actual in enumerate(pits):
        predicted = _first_box_stop_lap(by_lap, stint_start, actual)
        abs_err = abs(predicted - actual) if predicted is not None else None
        within = abs_err is not None and abs_err <= tolerance_laps
        out.append(
            StintScore(
                race_id=race_id,
                driver_id=driver_id,
                stint_index=idx,
                stint_start_lap=stint_start,
                actual_pit_lap=actual,
                predicted_pit_lap=predicted,
                abs_error=abs_err,
                within_tolerance=within if predicted is not None else None,
                score_kind="first_box_stop",
            )
        )
        stint_start = actual + 1
    return out


def score_stints_sim_plan(
    replay: RaceReplay,
    race_id: int,
    driver_id: int,
    decisions: Sequence[HarnessLapDecision],
    *,
    tolerance_laps: int = DEFAULT_TOLERANCE_LAPS,
) -> list[StintScore]:
    """Sim-plan models: planned stop at lap before actual pit vs actual pit."""
    pits = _pit_laps(replay, race_id, driver_id)
    if not pits:
        return []
    by_lap = {d.lap: d for d in decisions}
    out: list[StintScore] = []
    stint_start = 1
    for idx, actual in enumerate(pits):
        predicted, _ref = _plan_before_stop(by_lap, stint_start, actual)
        abs_err = abs(predicted - actual) if predicted is not None else None
        within = abs_err is not None and abs_err <= tolerance_laps
        out.append(
            StintScore(
                race_id=race_id,
                driver_id=driver_id,
                stint_index=idx,
                stint_start_lap=stint_start,
                actual_pit_lap=actual,
                predicted_pit_lap=predicted,
                abs_error=abs_err,
                within_tolerance=within if predicted is not None else None,
                score_kind="plan_before_stop",
            )
        )
        stint_start = actual + 1
    return out


def score_stints_for_model(
    model_id: str,
    replay: RaceReplay,
    race_id: int,
    driver_id: int,
    decisions: Sequence[HarnessLapDecision],
    *,
    tolerance_laps: int = DEFAULT_TOLERANCE_LAPS,
) -> list[StintScore]:
    fam = model_family(model_id)
    if fam == "pit_next":
        return score_stints_pit_next(
            replay, race_id, driver_id, decisions, tolerance_laps=tolerance_laps
        )
    return score_stints_sim_plan(
        replay, race_id, driver_id, decisions, tolerance_laps=tolerance_laps
    )


def _timing_summary(stints: Sequence[StintScore]) -> dict[str, Any]:
    scored = [s for s in stints if s.abs_error is not None]
    within = [s for s in scored if s.within_tolerance]
    missed = [s for s in stints if s.predicted_pit_lap is None]
    return {
        "n_stints": len(stints),
        "n_scored": len(scored),
        "n_missed": len(missed),
        "timing_mae": (
            round(sum(s.abs_error for s in scored) / len(scored), 3) if scored else None
        ),
        "pct_within_tolerance": (
            round(len(within) / len(scored), 4) if scored else None
        ),
    }


def aggregate_model_summary(
    model_id: str,
    rows: Sequence[RaceDriverResult],
) -> dict[str, Any]:
    fam = model_family(model_id)
    stints: list[StintScore] = []
    next_matches: list[float] = []
    regrets: list[float] = []
    flip_rates: list[float] = []
    n_laps = 0

    for row in rows:
        n_laps += len(row.laps)
        stints.extend(row.stints)
        for d in row.decisions:
            if d.y_true_next_lap is not None:
                next_matches.append(float(d.pit_next == d.y_true_next_lap))
            if d.regret is not None:
                regrets.append(d.regret)
        if row.flip_flop_rate is not None:
            flip_rates.append(row.flip_flop_rate)

    timing = _timing_summary(stints)
    out: dict[str, Any] = {
        "model_id": model_id,
        "family": fam,
        "n_race_drivers": len(rows),
        "n_laps": n_laps,
        "next_lap_match": (
            round(sum(next_matches) / len(next_matches), 4) if next_matches else None
        ),
        **timing,
    }
    if fam == "sim_plan":
        out["mean_regret"] = round(sum(regrets) / len(regrets), 4) if regrets else None
        out["flip_flop_rate"] = (
            round(sum(flip_rates) / len(flip_rates), 4) if flip_rates else None
        )
    return out


def _from_crew_decision(
    replay: RaceReplay,
    race_id: int,
    driver_id: int,
    lap: int,
    decision: CrewChiefDecision,
    *,
    label: str | None = None,
    regret: float | None = None,
) -> HarnessLapDecision:
    plan = planned_pit_lap(lap, action=decision.action, label=label)
    return HarnessLapDecision(
        lap=lap,
        action=decision.action,
        pit_next=decision.pit_next,
        planned_pit_lap=plan,
        chosen_label=label,
        regret=regret,
        y_true_next_lap=_y_true_next_lap(replay, race_id, driver_id, lap),
    )


def _from_sim(
    replay: RaceReplay,
    race_id: int,
    driver_id: int,
    sd: SimAgentDecision,
    lap: int,
) -> HarnessLapDecision:
    return _from_crew_decision(
        replay,
        race_id,
        driver_id,
        lap,
        sd.decision,
        label=sd.chosen_label,
        regret=sd.regret,
    )


class HarnessModel(Protocol):
    model_id: str
    supports_memory: bool

    def replay_race(
        self,
        replay: RaceReplay,
        race_id: int,
        driver_id: int,
        *,
        memory: bool,
        lap_from: int = 1,
        lap_to: int | None = None,
    ) -> RaceDriverResult: ...


class HgbHarnessModel:
    model_id = "hgb"
    supports_memory = False

    def __init__(self, replay: RaceReplay, *, models_path: Path | None = None):
        path = models_path or _DEFAULT_MODELS
        if not path.exists():
            raise FileNotFoundError(f"HGB model not found: {path}")
        with path.open("rb") as f:
            bundle = pickle.load(f)
        self._model = bundle["models"][bundle["primary"]]
        self._threshold = float(bundle["thresholds"][bundle["primary"]])
        years = sorted({int(replay._races[r]["year"]) for r in replay._races})
        samples = build_samples(
            replay,
            years,
            sc_by_race=load_safety_car_periods(replay),
            tyre_by_key=load_tyre_laps(replay),
        )
        self._features = {
            (s.race_id, s.driver_id, s.lap): s.features for s in samples
        }

    def replay_race(
        self,
        replay: RaceReplay,
        race_id: int,
        driver_id: int,
        *,
        memory: bool,
        lap_from: int = 1,
        lap_to: int | None = None,
    ) -> RaceDriverResult:
        laps_avail = replay.available_laps(race_id, driver_id)
        hi = lap_to if lap_to is not None else laps_avail[-1]
        laps = [L for L in laps_avail if lap_from <= L <= hi]
        decisions: list[HarnessLapDecision] = []
        for lap in laps:
            feats = self._features.get((race_id, driver_id, lap))
            if feats is None:
                continue
            score = float(score_model(self._model, np.asarray([feats], dtype=np.float64))[0])
            pit_next = int(score >= self._threshold)
            action = "pit" if pit_next else "stay"
            decisions.append(
                HarnessLapDecision(
                    lap=lap,
                    action=action,
                    pit_next=pit_next,
                    planned_pit_lap=planned_pit_lap(lap, action=action),
                    y_true_next_lap=_y_true_next_lap(replay, race_id, driver_id, lap),
                )
            )
        return RaceDriverResult(
            model_id=self.model_id,
            race_id=race_id,
            driver_id=driver_id,
            laps=[d.lap for d in decisions],
            decisions=decisions,
        )


class CrewChiefHarnessModel:
    supports_memory = False

    def __init__(self, backend: CrewChiefBackend, *, model_id: str | None = None):
        self._backend = backend
        self.model_id = model_id or backend.name

    def replay_race(
        self,
        replay: RaceReplay,
        race_id: int,
        driver_id: int,
        *,
        memory: bool,
        lap_from: int = 1,
        lap_to: int | None = None,
    ) -> RaceDriverResult:
        laps_avail = replay.available_laps(race_id, driver_id)
        hi = lap_to if lap_to is not None else laps_avail[-1]
        laps = [L for L in laps_avail if lap_from <= L <= hi]
        decisions = [
            _from_crew_decision(
                replay,
                race_id,
                driver_id,
                lap,
                self._backend.decide(replay.get_state(race_id, driver_id, lap)),
            )
            for lap in laps
        ]
        return RaceDriverResult(
            model_id=self.model_id,
            race_id=race_id,
            driver_id=driver_id,
            laps=[d.lap for d in decisions],
            decisions=decisions,
        )


class SimHarnessModel:
    supports_memory = True

    def __init__(self, backend: SimAwareBackend, *, model_id: str | None = None):
        self._backend = backend
        self.model_id = model_id or backend.name

    def replay_race(
        self,
        replay: RaceReplay,
        race_id: int,
        driver_id: int,
        *,
        memory: bool,
        lap_from: int = 1,
        lap_to: int | None = None,
    ) -> RaceDriverResult:
        mem = RaceMemoryStore() if memory else None
        raw = replay_race_decisions(
            self._backend,
            replay,
            race_id,
            driver_id,
            memory=mem,
            lap_from=lap_from,
            lap_to=lap_to,
        )
        decisions = [
            _from_sim(replay, race_id, driver_id, sd, lap)
            for lap, sd in zip(raw.laps, raw.decisions)
        ]
        return RaceDriverResult(
            model_id=self.model_id,
            race_id=race_id,
            driver_id=driver_id,
            laps=raw.laps,
            decisions=decisions,
            flip_flop_rate=raw.flip_flop_rate() if memory else None,
            mean_regret=raw.mean_regret,
        )


def get_harness_model(model_id: str, replay: RaceReplay, **kwargs: Any) -> HarnessModel:
    mid = model_id.strip().lower()
    if mid in {"hgb", "pit_baseline", "phase1"}:
        return HgbHarnessModel(replay, models_path=kwargs.get("models_path"))
    if mid in {"heuristic_crew", "heuristic", "crew_chief"}:
        return CrewChiefHarnessModel(HeuristicBackend(), model_id="heuristic_crew")
    if mid in {"heuristic_sim", "sim"}:
        return SimHarnessModel(HeuristicSimBackend(replay), model_id="heuristic_sim")
    if mid in {"openai_sim", "openai"}:
        return SimHarnessModel(get_sim_backend("openai_sim", replay), model_id="openai_sim")
    if mid == "pit_next_sim":
        from race_engineer.sim_agent import PitNextSimBaseline

        return SimHarnessModel(PitNextSimBaseline(replay), model_id="pit_next_sim")
    raise ValueError(f"unknown harness model: {model_id}")


def _lap_records(all_rows: dict[str, list[RaceDriverResult]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for mid, rows in all_rows.items():
        for row in rows:
            for d in row.decisions:
                rec = d.to_dict()
                rec["model_id"] = mid
                rec["race_id"] = row.race_id
                rec["driver_id"] = row.driver_id
                out.append(rec)
    return out


def _compact_stint_stats(row: RaceDriverResult) -> dict[str, Any]:
    t = _timing_summary(row.stints)
    out = {
        "n_laps": len(row.laps),
        "timing_mae": t["timing_mae"],
        "pct_within_tolerance": t["pct_within_tolerance"],
        "n_stints_scored": t["n_scored"],
        "n_stints_missed": t["n_missed"],
    }
    if row.mean_regret is not None:
        out["mean_regret"] = row.mean_regret
    if row.flip_flop_rate is not None:
        out["flip_flop_rate"] = row.flip_flop_rate
    return out


def run_harness(
    replay: RaceReplay,
    race_ids: Sequence[int],
    model_ids: Sequence[str],
    *,
    pick_drivers_fn,
    tolerance_laps: int = DEFAULT_TOLERANCE_LAPS,
    memory: bool = True,
    lap_from: int = 1,
    lap_to: int | None = None,
    models_path: Path | None = None,
) -> dict[str, Any]:
    models: dict[str, HarnessModel] = {
        mid: get_harness_model(mid, replay, models_path=models_path) for mid in model_ids
    }
    all_rows: dict[str, list[RaceDriverResult]] = {mid: [] for mid in model_ids}
    by_race_driver: list[dict[str, Any]] = []

    for rid in race_ids:
        for did in pick_drivers_fn(replay, rid):
            entry: dict[str, Any] = {"race_id": rid, "driver_id": did, "models": {}}
            for mid, model in models.items():
                row = model.replay_race(
                    replay,
                    rid,
                    did,
                    memory=memory and model.supports_memory,
                    lap_from=lap_from,
                    lap_to=lap_to,
                )
                row.stints = score_stints_for_model(
                    mid,
                    replay,
                    rid,
                    did,
                    row.decisions,
                    tolerance_laps=tolerance_laps,
                )
                all_rows[mid].append(row)
                entry["models"][mid] = _compact_stint_stats(row)
            by_race_driver.append(entry)

    summaries = [aggregate_model_summary(mid, all_rows[mid]) for mid in model_ids]
    pit_next_board = [s for s in summaries if s["family"] == "pit_next"]
    sim_plan_board = [s for s in summaries if s["family"] == "sim_plan"]

    return {
        "harness_version": HARNESS_VERSION,
        "frozen_races_id": FROZEN_RACES_ID,
        "tolerance_laps": tolerance_laps,
        "memory_default": memory,
        "lap_from": lap_from,
        "lap_to": lap_to,
        "leaderboard_pit_next": pit_next_board,
        "leaderboard_sim_plan": sim_plan_board,
        "by_race_driver": by_race_driver,
        "lap_records": _lap_records(all_rows),
        "metric_definitions": {
            "pit_next": (
                "Timing MAE = |first box-next-lap stop − actual pit| per stint; "
                f"within ±{tolerance_laps} laps."
            ),
            "sim_plan": (
                "Timing MAE = |planned stop (from lap before actual pit) − actual pit| "
                f"per stint; within ±{tolerance_laps} laps. "
                "Uses stay_N_then_pit / pit_next card labels."
            ),
            "next_lap_match": (
                "Fraction of laps where pit/stay for lap+1 matches history."
            ),
            "mean_regret": "Sim only: E[finish|chosen] − E[finish|oracle] on option cards.",
            "flip_flop_rate": (
                "Sim only: pit/stay reversals without evidence change (memory on)."
            ),
        },
    }


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{100 * value:.1f}%"


def render_report(payload: dict[str, Any]) -> str:
    tol = payload.get("tolerance_laps", 2)
    defs = payload.get("metric_definitions") or {}
    lines = [
        f"# Eval harness {payload.get('harness_version')}",
        "",
        f"Frozen set `{payload.get('frozen_races_id')}` · tolerance ±{tol} laps · "
        f"memory (sim)={payload.get('memory_default')}",
        "",
        "## Pit-next models (HGB, crew chief, pit-next sim)",
        "",
        defs.get("pit_next", ""),
        "",
        "| Model | Timing MAE | Within ±tol | Next-lap match | Stints scored |",
        "|-------|------------|-------------|----------------|---------------|",
    ]
    for row in payload.get("leaderboard_pit_next") or []:
        lines.append(
            f"| {row['model_id']} "
            f"| {row.get('timing_mae', '—')} "
            f"| {_fmt_pct(row.get('pct_within_tolerance'))} "
            f"| {row.get('next_lap_match', '—')} "
            f"| {row.get('n_scored', 0)}/{row.get('n_stints', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Sim-plan models (option cards + memory)",
            "",
            defs.get("sim_plan", ""),
            "",
            "| Model | Timing MAE | Within ±tol | Next-lap match | Mean regret | Flip-flop |",
            "|-------|------------|-------------|----------------|-------------|-----------|",
        ]
    )
    for row in payload.get("leaderboard_sim_plan") or []:
        lines.append(
            f"| {row['model_id']} "
            f"| {row.get('timing_mae', '—')} "
            f"| {_fmt_pct(row.get('pct_within_tolerance'))} "
            f"| {row.get('next_lap_match', '—')} "
            f"| {row.get('mean_regret', '—')} "
            f"| {row.get('flip_flop_rate', '—')} |"
        )

    lines.extend(["", "### Shared definitions", ""])
    lines.append(f"- **Next-lap match:** {defs.get('next_lap_match', '')}")
    lines.append(f"- **Mean regret:** {defs.get('mean_regret', '')}")
    lines.append(f"- **Flip-flop:** {defs.get('flip_flop_rate', '')}")
    lines.append("")
    return "\n".join(lines)
