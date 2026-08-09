"""Phase 5 (simulation): pit-now vs stay-N Monte Carlo option cards."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from strategy_engineer.lap_deg import LapDegModel
from strategy_engineer.pit_baseline import COMPOUND_TO_ID
from shared.racing_rules import (
    alternate_dry_compound_id,
    mandatory_dry_pit_pending,
)
from shared.replay import RaceReplay
from shared.state import RaceState

# Default green-flag pit loss (station + in/out). Circuit-specific later.
DEFAULT_PIT_LOSS_MS = 22_000.0
DEFAULT_SC_PIT_LOSS_MS = 12_000.0  # lower effective loss under SC
DEFAULT_NOISE_MS = 400.0  # per-lap Gaussian noise in MC
DEFAULT_N_ROLLS = 64
# 0 = pit next lap. The grid must reach past a realistic stop window: if the best
# stop lies beyond the largest offset, the argmax is always the last card, and
# re-picking it every lap walks the stop forward one lap at a time.
DEFAULT_STAY_NS = (0, 3, 5, 8, 12, 18, 25)
# Lap times above this (ms) are SC-inflated and must not anchor green-flag rollouts.
_SC_LAP_MS_THRESHOLD = 100_000.0
# Fresh-stint pace gain vs each driver's own green-flag lap (ms).
_FRESH_TYRE_GAIN_MS = 600.0
# Forward tyre wear is NOT taken from the deg model. Probed against tyre_life it
# returns a step between life 0 and ~4 and is then flat, identical for every
# compound — an artifact of fuel load and out-laps, not wear. Reading a slope off
# it makes a fresh tyre look slower than the worn one it replaced, so every pit
# option loses and the longest stay always wins. The deg model is kept for the
# pace *level* (what it is trained for); forward wear uses this explicit curve:
# wear(life) = total * (1 - exp(-life / tau)), monotonic and saturating.
# Calibrated by sweeping against harness stop-timing error over the frozen set,
# not fitted per race. tau is long enough that wear is near-linear across a normal
# stint (the usual working assumption) while still saturating, which is what keeps
# a 45-lap extrapolation from reaching absurd lap times.
_STINT_DEG_TOTAL_MS = 3_500.0
_STINT_DEG_TAU_LAPS = 30.0
# Relative wear rate by compound (softer degrades faster). Ordering only.
_COMPOUND_DEG_SCALE = {1.0: 1.35, 2.0: 1.0, 3.0: 0.75}
# Live degradation fit. Raw lap times improve as fuel burns off, so add back a
# conservative per-lap fuel effect before interpreting the trend as tyre wear.
# The estimate is robustly fitted over the current stint and shrunk toward the
# generic prior; these limits prevent traffic or one lock-up becoming a "cliff".
_LIVE_DEG_MIN_LAPS = 5
_LIVE_DEG_WINDOW_LAPS = 12
_FUEL_EFFECT_MS_PER_LAP = 35.0
_LIVE_DEG_MAX_SLOPE_MS = 450.0
_LIVE_DEG_MAX_WEIGHT = 0.65
_LIVE_DEG_OUTLIER_FLOOR_MS = 500.0
_CLIFF_MIN_EXTRA_SLOPE_MS = 180.0
_CLIFF_MIN_RESIDUAL_MS = 300.0
_CLIFF_MAX_EXTRA_SLOPE_MS = 600.0
# Cap rival jitter so remaining-lap scaling does not swamp ranking.
_RIVAL_NOISE_LAP_CAP = 12
# Blend sim E[finish] toward board P when gaps are healthy (front-running).
_BOARD_BLEND_WEIGHT = 0.35
_BOARD_BLEND_MAX_GAP_MS = 5_000
# Gap-based sanity: cannot gain places without closing a gap ahead; large
# cushion behind limits positions lost (not race-specific tuning).
_SANITY_GAP_AHEAD_FLOOR_MS = 1_500
_SANITY_GAP_BEHIND_CAP_MS = 4_000
# Board bounds are calibration, not truth: compress the part of a projection that
# runs past them rather than dropping it, or every option clamps to one number and
# the oracle has nothing left to rank on.
_BOARD_CLAMP_COMPRESSION = 0.15
# Finish positions a rival lap must be worth before a called stop is moved.
_PLAN_CONTINUITY_CREDIT = 0.15


@dataclass
class StrategyOption:
    option_id: str
    pit_after_laps: int  # 0 = pit on the very next lap
    label: str


@dataclass
class OptionCard:
    option_id: str
    label: str
    pit_after_laps: int
    n_rolls: int
    mean_finish_pos: float
    median_finish_pos: float
    p_finish_le_3: float
    p_finish_le_5: float
    p_finish_le_10: float
    mean_race_time_ms: float
    planned_pit_lap: int | None = None

    def to_dict(self) -> dict[str, Any]:
        out = {
            "option_id": self.option_id,
            "label": self.label,
            "pit_after_laps": self.pit_after_laps,
            "n_rolls": self.n_rolls,
            "mean_finish_pos": round(self.mean_finish_pos, 3),
            "median_finish_pos": round(self.median_finish_pos, 2),
            "P_finish_le_3": round(self.p_finish_le_3, 4),
            "P_finish_le_5": round(self.p_finish_le_5, 4),
            "P_finish_le_10": round(self.p_finish_le_10, 4),
            "mean_race_time_ms": round(self.mean_race_time_ms, 1),
        }
        if self.planned_pit_lap is not None:
            out["planned_pit_lap"] = self.planned_pit_lap
        return out


@dataclass
class SimSnapshot:
    """Minimal ego car state for rolling the race forward."""

    circuit_id: float
    lap: int
    total_laps: int
    stint_age: float
    tyre_life: float
    compound_id: float
    last_lap_ms: float
    prev_lap_ms: float
    prev2_lap_ms: float
    sc_active: float
    cumulative_ms: float

    @classmethod
    def from_state(cls, state: RaceState, cumulative_ms: float) -> SimSnapshot:
        compound = (state.tyre_compound or "UNKNOWN").strip().upper()
        window = list(state.last_lap_times_ms) if state.last_lap_times_ms else []
        last = float(state.last_lap_time_ms)
        prev = float(window[-2]) if len(window) >= 2 else last
        prev2 = float(window[-3]) if len(window) >= 3 else prev
        return cls(
            circuit_id=float(state.circuit_id),
            lap=state.lap,
            total_laps=state.total_laps,
            stint_age=float(state.stint_age_laps),
            tyre_life=float(
                state.tyre_life if state.tyre_life is not None else state.stint_age_laps
            ),
            compound_id=float(COMPOUND_TO_ID.get(compound, 0)),
            last_lap_ms=last,
            prev_lap_ms=prev,
            prev2_lap_ms=prev2,
            sc_active=1.0 if state.sc_active else 0.0,
            cumulative_ms=float(cumulative_ms),
        )

    def feature_row(self) -> np.ndarray:
        last_s = self.last_lap_ms / 1000.0
        prev_s = self.prev_lap_ms / 1000.0
        prev2_s = self.prev2_lap_ms / 1000.0
        total = max(self.total_laps, 1)
        return np.asarray(
            [
                [
                    self.circuit_id,
                    self.stint_age,
                    self.tyre_life,
                    self.compound_id,
                    float(self.lap / total),
                    float(total - self.lap),
                    last_s,
                    prev_s,
                    prev2_s,
                    last_s - prev_s,
                    self.sc_active,
                ]
            ],
            dtype=np.float64,
        )


def _default_deg() -> LapDegModel | None:
    path = Path(__file__).resolve().parents[1] / "artifacts" / "lap_deg" / "model.pkl"
    if not path.exists():
        return None
    return LapDegModel.load(path)


def compound_deg_total_ms(compound_id: float) -> float:
    """Lap-time loss a fully worn set of this compound carries vs fresh."""
    return _STINT_DEG_TOTAL_MS * _COMPOUND_DEG_SCALE.get(compound_id, 1.0)


def _wear_ms(deg_total_ms: float, tyre_life: float) -> float:
    """Saturating tyre wear: rises quickly early, then plateaus."""
    life = max(0.0, tyre_life)
    return max(0.0, deg_total_ms) * (1.0 - math.exp(-life / _STINT_DEG_TAU_LAPS))


def _wear_delta_ms(deg_total_ms: float, life_from: float, life_to: float) -> float:
    """Extra wear between two tyre ages (never negative)."""
    return max(
        0.0,
        _wear_ms(deg_total_ms, life_to) - _wear_ms(deg_total_ms, life_from),
    )


@dataclass(frozen=True)
class LiveDegEstimate:
    """Current-stint degradation inferred from clean laps available so far."""

    deg_total_ms: float
    slope_ms_per_lap: float
    prior_slope_ms_per_lap: float
    sample_count: int
    source: str
    cliff_start_life: float | None = None
    cliff_extra_ms_per_lap: float = 0.0

    @property
    def cliff_detected(self) -> bool:
        return self.cliff_start_life is not None and self.cliff_extra_ms_per_lap > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "sample_count": self.sample_count,
            "slope_ms_per_lap": round(self.slope_ms_per_lap, 1),
            "prior_slope_ms_per_lap": round(self.prior_slope_ms_per_lap, 1),
            "cliff_detected": self.cliff_detected,
            "cliff_start_life": self.cliff_start_life,
            "cliff_extra_ms_per_lap": round(self.cliff_extra_ms_per_lap, 1),
        }


def _prior_wear_slope_ms(deg_total_ms: float, tyre_life: float) -> float:
    """Derivative of the saturating prior at the current tyre age."""
    life = max(0.0, tyre_life)
    return (
        max(0.0, deg_total_ms)
        / _STINT_DEG_TAU_LAPS
        * math.exp(-life / _STINT_DEG_TAU_LAPS)
    )


def _theil_sen_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Median pairwise slope: robust to a small number of traffic/lock-up laps."""
    slopes = [
        (ys[j] - ys[i]) / (xs[j] - xs[i])
        for i in range(len(xs))
        for j in range(i + 1, len(xs))
        if xs[j] != xs[i]
    ]
    return float(np.median(slopes)) if slopes else 0.0


def _driver_compound_id(
    replay: RaceReplay, race_id: int, driver_id: int, lap: int
) -> float:
    tyre = replay._tyres.get((race_id, driver_id, lap), {})
    compound = (tyre.get("compound") or "UNKNOWN").strip().upper()
    return float(COMPOUND_TO_ID.get(compound, 0))


def _clean_current_stint_laps(
    replay: RaceReplay,
    race_id: int,
    driver_id: int,
    lap: int,
    total_laps: int,
    *,
    compound_id: float,
) -> tuple[list[float], list[float]]:
    """Return (tyre age, lap ms) for usable laps in the current stint."""
    driver_laps = replay._laps.get(race_id, {}).get(driver_id, {})
    pits = [p for p in replay._pits.get(race_id, {}).get(driver_id, []) if p <= lap]
    # The pit lap contains stationary time; lap 1 contains the race start.
    first_usable = (max(pits) + 1) if pits else 2
    points: list[tuple[float, float]] = []
    for current_lap in range(first_usable, lap + 1):
        row = driver_laps.get(current_lap)
        if row is None:
            continue
        ms = float(row["milliseconds"])
        sc_active, _, _, _ = replay._sc_flags(race_id, current_lap, total_laps)
        if sc_active or ms >= _SC_LAP_MS_THRESHOLD:
            continue
        lap_compound_id = _driver_compound_id(
            replay, race_id, driver_id, current_lap
        )
        if (
            compound_id > 0
            and lap_compound_id > 0
            and lap_compound_id != compound_id
        ):
            continue
        tyre = replay._tyres.get((race_id, driver_id, current_lap), {})
        life = tyre.get("tyreLife")
        if life is None:
            life = current_lap - (max(pits) if pits else 0)
        points.append((float(life), ms))
    points = points[-_LIVE_DEG_WINDOW_LAPS:]
    return [p[0] for p in points], [p[1] for p in points]


def estimate_live_stint_deg(
    replay: RaceReplay,
    race_id: int,
    driver_id: int,
    lap: int,
    total_laps: int,
    *,
    compound_id: float,
    tyre_life: float,
) -> LiveDegEstimate:
    """Fit live wear from this driver's current green stint.

    The robust lap-time slope is fuel-corrected and partially pooled with the
    compound prior. A separate recent-vs-earlier test detects a sustained cliff.
    """
    prior_total = compound_deg_total_ms(compound_id)
    prior_slope = _prior_wear_slope_ms(prior_total, tyre_life)
    xs, ys = _clean_current_stint_laps(
        replay,
        race_id,
        driver_id,
        lap,
        total_laps,
        compound_id=compound_id,
    )
    if len(xs) < _LIVE_DEG_MIN_LAPS:
        return LiveDegEstimate(
            deg_total_ms=prior_total,
            slope_ms_per_lap=prior_slope,
            prior_slope_ms_per_lap=prior_slope,
            sample_count=len(xs),
            source="compound_prior",
        )

    raw_slope = _theil_sen_slope(xs, ys)
    intercept = float(np.median([y - raw_slope * x for x, y in zip(xs, ys)]))
    residuals = np.asarray(
        [y - (intercept + raw_slope * x) for x, y in zip(xs, ys)],
        dtype=np.float64,
    )
    residual_median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - residual_median)))
    threshold = max(_LIVE_DEG_OUTLIER_FLOOR_MS, 3.0 * 1.4826 * mad)
    kept = [
        (x, y)
        for x, y, residual in zip(xs, ys, residuals)
        if abs(float(residual) - residual_median) <= threshold
    ]
    if len(kept) >= _LIVE_DEG_MIN_LAPS:
        fit_xs = [p[0] for p in kept]
        fit_ys = [p[1] for p in kept]
        raw_slope = _theil_sen_slope(fit_xs, fit_ys)
    else:
        fit_xs, fit_ys = xs, ys

    measured_slope = min(
        _LIVE_DEG_MAX_SLOPE_MS,
        max(0.0, raw_slope + _FUEL_EFFECT_MS_PER_LAP),
    )
    live_weight = min(
        _LIVE_DEG_MAX_WEIGHT,
        len(fit_xs) / (len(fit_xs) + 8.0),
    )
    fitted_slope = (
        (1.0 - live_weight) * prior_slope + live_weight * measured_slope
    )
    # Choose a curve total whose derivative at the current age matches the
    # pooled live slope. Keep broad physical limits for noisy/short samples.
    fitted_total = fitted_slope * _STINT_DEG_TAU_LAPS * math.exp(
        max(0.0, tyre_life) / _STINT_DEG_TAU_LAPS
    )
    fitted_total = min(9_000.0, max(500.0, fitted_total))

    cliff_start: float | None = None
    cliff_extra = 0.0
    if len(xs) >= 7:
        earlier_xs, earlier_ys = xs[:-3], ys[:-3]
        recent_xs, recent_ys = xs[-3:], ys[-3:]
        baseline_slope = _theil_sen_slope(earlier_xs, earlier_ys)
        baseline_intercept = float(
            np.median(
                [
                    y - baseline_slope * x
                    for x, y in zip(earlier_xs, earlier_ys)
                ]
            )
        )
        recent_slope = _theil_sen_slope(recent_xs, recent_ys)
        recent_residual = float(
            np.median(
                [
                    y - (baseline_intercept + baseline_slope * x)
                    for x, y in zip(recent_xs, recent_ys)
                ]
            )
        )
        extra = recent_slope - baseline_slope
        if (
            extra >= _CLIFF_MIN_EXTRA_SLOPE_MS
            and recent_residual >= _CLIFF_MIN_RESIDUAL_MS
        ):
            cliff_start = recent_xs[0]
            cliff_extra = min(_CLIFF_MAX_EXTRA_SLOPE_MS, extra)

    return LiveDegEstimate(
        deg_total_ms=fitted_total,
        slope_ms_per_lap=fitted_slope,
        prior_slope_ms_per_lap=prior_slope,
        sample_count=len(fit_xs),
        source="live_stint",
        cliff_start_life=cliff_start,
        cliff_extra_ms_per_lap=cliff_extra,
    )


def live_deg_for_state(replay: RaceReplay, state: RaceState) -> LiveDegEstimate:
    """Convenience wrapper used by both the simulator and its tool response."""
    compound = (state.tyre_compound or "UNKNOWN").strip().upper()
    compound_id = float(COMPOUND_TO_ID.get(compound, 0))
    tyre_life = float(
        state.tyre_life
        if state.tyre_life is not None
        else state.stint_age_laps
    )
    return estimate_live_stint_deg(
        replay,
        state.race_id,
        state.driver_id,
        state.lap,
        state.total_laps,
        compound_id=compound_id,
        tyre_life=tyre_life,
    )


def _cliff_wear_ms(
    tyre_life: float,
    cliff_start_life: float | None,
    cliff_extra_ms_per_lap: float,
) -> float:
    if cliff_start_life is None:
        return 0.0
    return max(0.0, tyre_life - cliff_start_life) * max(
        0.0, cliff_extra_ms_per_lap
    )


@dataclass
class PaceSchedule:
    """
    Fast MC pace: deg-model probe for the pace level, explicit wear curve for
    forward degradation. Avoids calling sklearn on every simulated lap.
    """

    # compound_id -> (intercept_ms at life=0, total wear_ms as life grows)
    by_compound: dict[float, tuple[float, float]]
    fallback_ms: float
    # Only the current set can have a live detected cliff; a future set uses prior.
    cliff_by_compound: dict[float, tuple[float | None, float]] = field(
        default_factory=dict
    )

    def deg_total_ms(self, compound_id: float) -> float:
        pair = self.by_compound.get(compound_id)
        if pair is not None:
            return pair[1]
        return compound_deg_total_ms(compound_id)

    def wear_ms(self, compound_id: float, tyre_life: float) -> float:
        pair = self.by_compound.get(compound_id)
        deg_total = pair[1] if pair is not None else compound_deg_total_ms(compound_id)
        cliff_start, cliff_extra = self.cliff_by_compound.get(
            compound_id, (None, 0.0)
        )
        return _wear_ms(deg_total, tyre_life) + _cliff_wear_ms(
            tyre_life, cliff_start, cliff_extra
        )

    def lap_ms(self, snap: SimSnapshot, rng: np.random.Generator, noise_ms: float) -> float:
        if snap.sc_active > 0.5:
            base = max(snap.last_lap_ms, 100_000.0)
        else:
            pair = self.by_compound.get(snap.compound_id)
            if pair is None:
                base = self.fallback_ms
            else:
                intercept, deg_total = pair
                cliff_start, cliff_extra = self.cliff_by_compound.get(
                    snap.compound_id, (None, 0.0)
                )
                base = (
                    intercept
                    + _wear_ms(deg_total, snap.tyre_life)
                    + _cliff_wear_ms(snap.tyre_life, cliff_start, cliff_extra)
                )
        if noise_ms > 0:
            base = base + float(rng.normal(0.0, noise_ms))
        return max(base, 40_000.0)


def _green_flag_anchor_ms(
    deg: LapDegModel | None,
    snap0: SimSnapshot,
    *,
    green_reference_ms: float | None = None,
) -> float:
    """Pace anchor for MC rollouts; SC / slow laps must not skew green projection."""
    if (
        snap0.sc_active < 0.5
        and snap0.last_lap_ms < _SC_LAP_MS_THRESHOLD
    ):
        return snap0.last_lap_ms
    # The board lap is SC-inflated, and the deg model reads last/prev lap times
    # straight off its feature row, so it extrapolates from the slow lap. A real
    # green lap from the replay keeps the ego on the same footing as the rivals.
    if green_reference_ms is not None and green_reference_ms < _SC_LAP_MS_THRESHOLD:
        return green_reference_ms
    if deg is not None:
        green = SimSnapshot(
            circuit_id=snap0.circuit_id,
            lap=snap0.lap,
            total_laps=snap0.total_laps,
            stint_age=snap0.stint_age,
            tyre_life=snap0.tyre_life,
            compound_id=snap0.compound_id,
            last_lap_ms=snap0.last_lap_ms,
            prev_lap_ms=snap0.prev_lap_ms,
            prev2_lap_ms=snap0.prev2_lap_ms,
            sc_active=0.0,
            cumulative_ms=snap0.cumulative_ms,
        )
        return float(deg.model.predict(green.feature_row())[0])
    for ms in (snap0.prev_lap_ms, snap0.prev2_lap_ms, snap0.last_lap_ms):
        if ms < _SC_LAP_MS_THRESHOLD:
            return ms
    return 95_000.0


def build_pace_schedule(
    deg: LapDegModel | None,
    snap0: SimSnapshot,
    *,
    green_reference_ms: float | None = None,
    live_deg: LiveDegEstimate | None = None,
) -> PaceSchedule:
    anchor_ms = _green_flag_anchor_ms(deg, snap0, green_reference_ms=green_reference_ms)
    current_total = (
        live_deg.deg_total_ms
        if live_deg is not None
        else compound_deg_total_ms(snap0.compound_id)
    )
    cliff_start = live_deg.cliff_start_life if live_deg is not None else None
    cliff_extra = live_deg.cliff_extra_ms_per_lap if live_deg is not None else 0.0
    # Pace this car would show on a fresh set of its current compound. The anchor
    # is a worn lap, so the wear on it must come off before the fresh-tyre gain is
    # applied: deriving post-pit pace from the worn anchor makes a new tyre look
    # slower the longer you wait, which is what pushes the stop over the horizon.
    unworn_ms = (
        anchor_ms
        - _wear_ms(current_total, snap0.tyre_life)
        - _cliff_wear_ms(snap0.tyre_life, cliff_start, cliff_extra)
    )
    by_c: dict[float, tuple[float, float]] = {}
    cliffs: dict[float, tuple[float | None, float]] = {}
    post_pit_id = alternate_dry_compound_id(snap0.compound_id)
    for cid in {snap0.compound_id, post_pit_id}:
        if cid == snap0.compound_id:
            deg_total = current_total
            intercept = unworn_ms
            if cliff_start is not None:
                cliffs[cid] = (cliff_start, cliff_extra)
        else:
            deg_total = compound_deg_total_ms(cid)
            intercept = _fresh_tyre_pace_ms(unworn_ms)
        by_c[cid] = (intercept, deg_total)
    return PaceSchedule(
        by_compound=by_c,
        fallback_ms=anchor_ms,
        cliff_by_compound=cliffs,
    )


def _laps_until_sc_end(replay: RaceReplay, race_id: int, lap: int, total_laps: int) -> int:
    for deployed, retreated in replay._sc_periods.get(race_id, []):
        end = total_laps if retreated is None else retreated
        if deployed <= lap <= end:
            return max(0, end - lap)
    return 0


def _apply_lap(snap: SimSnapshot, lap_ms: float, *, sc_after: float = 0.0) -> SimSnapshot:
    return SimSnapshot(
        circuit_id=snap.circuit_id,
        lap=snap.lap + 1,
        total_laps=snap.total_laps,
        stint_age=snap.stint_age + 1.0,
        tyre_life=snap.tyre_life + 1.0,
        compound_id=snap.compound_id,
        last_lap_ms=lap_ms,
        prev_lap_ms=snap.last_lap_ms,
        prev2_lap_ms=snap.prev_lap_ms,
        sc_active=sc_after,
        cumulative_ms=snap.cumulative_ms + lap_ms,
    )


def _pit_stop(
    snap: SimSnapshot,
    pit_loss_ms: float,
    new_compound_id: float | None = None,
) -> SimSnapshot:
    """HARD/MEDIUM alternate default for second stint if unknown."""
    if new_compound_id is None:
        new_compound_id = alternate_dry_compound_id(snap.compound_id)
    return SimSnapshot(
        circuit_id=snap.circuit_id,
        lap=snap.lap,
        total_laps=snap.total_laps,
        stint_age=0.0,
        tyre_life=0.0,
        compound_id=new_compound_id,
        last_lap_ms=snap.last_lap_ms,
        prev_lap_ms=snap.prev_lap_ms,
        prev2_lap_ms=snap.prev2_lap_ms,
        sc_active=snap.sc_active,
        cumulative_ms=snap.cumulative_ms + pit_loss_ms,
    )


def _recent_green_pace_ms(
    replay: RaceReplay,
    race_id: int,
    driver_id: int,
    lap: int,
    total_laps: int,
    *,
    board_sc_active: bool,
) -> float:
    laps = replay._laps.get(race_id, {}).get(driver_id, {})
    recent: list[float] = []
    for L in range(lap, max(0, lap - 2) - 1, -1):
        if L not in laps:
            continue
        ms = float(laps[L]["milliseconds"])
        if not board_sc_active:
            sc_at_l, _, _, _ = replay._sc_flags(race_id, L, total_laps)
            if sc_at_l or ms >= _SC_LAP_MS_THRESHOLD:
                continue
        recent.append(ms)
    if not recent and not board_sc_active:
        for L in range(lap, max(0, lap - 8) - 1, -1):
            if L not in laps:
                continue
            ms = float(laps[L]["milliseconds"])
            sc_at_l, _, _, _ = replay._sc_flags(race_id, L, total_laps)
            if not sc_at_l and ms < _SC_LAP_MS_THRESHOLD:
                return ms
    return float(np.mean(recent)) if recent else 90_000.0


def _stint_time_ms(
    laps: int,
    pace_ms: float,
    deg_total_ms: float,
    *,
    life_from: float = 0.0,
    cliff_start_life: float | None = None,
    cliff_extra_ms_per_lap: float = 0.0,
) -> float:
    """Time for `laps` laps, adding wear accrued beyond the current tyre age.

    `pace_ms` already reflects wear at `life_from`, so only the increment counts.
    """
    if laps <= 0:
        return 0.0
    total = 0.0
    for i in range(laps):
        life_to = life_from + i
        curve_delta = _wear_delta_ms(deg_total_ms, life_from, life_to)
        cliff_delta = max(
            0.0,
            _cliff_wear_ms(
                life_to, cliff_start_life, cliff_extra_ms_per_lap
            )
            - _cliff_wear_ms(
                life_from, cliff_start_life, cliff_extra_ms_per_lap
            ),
        )
        total += pace_ms + curve_delta + cliff_delta
    return total


def _driver_tyre_life(
    replay: RaceReplay, race_id: int, driver_id: int, lap: int
) -> float:
    """Laps on the current set, inferred from this driver's pit history."""
    tyre_life = replay._tyres.get((race_id, driver_id, lap), {}).get("tyreLife")
    if tyre_life is not None:
        return float(tyre_life)
    pits = [p for p in replay._pits.get(race_id, {}).get(driver_id, []) if p <= lap]
    if not pits:
        return float(lap)
    return float(lap - max(pits))


def _project_driver_finish_ms(
    replay: RaceReplay,
    state: RaceState,
    driver_id: int,
    cum: float,
    *,
    remaining: int,
    pace_ms: float,
    pit_loss_ms: float,
    deg_total_ms: float = 0.0,
    post_pit_deg_total_ms: float | None = None,
    tyre_life: float = 0.0,
    cliff_start_life: float | None = None,
    cliff_extra_ms_per_lap: float = 0.0,
    sc_surplus_ms: float = 0.0,
) -> float:
    """Historical replay projection: rivals follow known future pit laps."""
    if remaining <= 0:
        return cum
    # Strip the wear already on this set to recover fresh-set pace, then apply
    # the fresh-tyre gain. Using worn pace here would hide the benefit of pitting.
    unworn_ms = (
        pace_ms
        - _wear_ms(deg_total_ms, tyre_life)
        - _cliff_wear_ms(
            tyre_life, cliff_start_life, cliff_extra_ms_per_lap
        )
    )
    fresh_ms = _fresh_tyre_pace_ms(unworn_ms)
    if post_pit_deg_total_ms is None:
        post_pit_deg_total_ms = deg_total_ms
    pit_laps = sorted(
        p for p in replay._pits.get(state.race_id, {}).get(driver_id, [])
        if p > state.lap
    )
    if pit_laps:
        first_pit = pit_laps[0]
        laps_before = max(0, first_pit - state.lap)
        laps_after = max(0, remaining - laps_before)
        return (
            float(cum)
            + _stint_time_ms(
                laps_before,
                pace_ms,
                deg_total_ms,
                life_from=tyre_life,
                cliff_start_life=cliff_start_life,
                cliff_extra_ms_per_lap=cliff_extra_ms_per_lap,
            )
            + pit_loss_ms
            + _stint_time_ms(laps_after, fresh_ms, post_pit_deg_total_ms)
            + sc_surplus_ms
        )
    return (
        float(cum)
        + _stint_time_ms(
            remaining,
            pace_ms,
            deg_total_ms,
            life_from=tyre_life,
            cliff_start_life=cliff_start_life,
            cliff_extra_ms_per_lap=cliff_extra_ms_per_lap,
        )
        + sc_surplus_ms
    )


def _fresh_tyre_pace_ms(pre_pit_pace_ms: float) -> float:
    return max(pre_pit_pace_ms - _FRESH_TYRE_GAIN_MS, 85_000.0)


def rival_finish_times_ms(
    replay: RaceReplay,
    state: RaceState,
    *,
    pit_loss_ms: float = DEFAULT_PIT_LOSS_MS,
    sc_surplus_ms: float = 0.0,
) -> list[float]:
    """
    Rival finish estimates: each driver's green-flag pace plus known future pit
    laps and a driver-specific fresh-tyre offset. Excludes the ego driver.

    Each rival gets the same estimator as the ego, fitted to that rival's current
    stint. This keeps treatment symmetric without pretending every car has the
    same live degradation.
    """
    cumul = replay._cumulative_times(state.race_id, state.lap)
    remaining = max(0, state.total_laps - state.lap)
    total = state.total_laps
    out: list[float] = []
    for did, cum in cumul.items():
        if did == state.driver_id:
            continue
        pace = _recent_green_pace_ms(
            replay,
            state.race_id,
            did,
            state.lap,
            total,
            board_sc_active=False,
        )
        tyre_life = _driver_tyre_life(
            replay, state.race_id, did, state.lap
        )
        compound_id = _driver_compound_id(
            replay, state.race_id, did, state.lap
        )
        live_deg = estimate_live_stint_deg(
            replay,
            state.race_id,
            did,
            state.lap,
            total,
            compound_id=compound_id,
            tyre_life=tyre_life,
        )
        post_pit_id = alternate_dry_compound_id(compound_id)
        out.append(
            _project_driver_finish_ms(
                replay,
                state,
                did,
                float(cum),
                remaining=remaining,
                pace_ms=pace,
                pit_loss_ms=pit_loss_ms,
                deg_total_ms=live_deg.deg_total_ms,
                post_pit_deg_total_ms=compound_deg_total_ms(post_pit_id),
                tyre_life=tyre_life,
                cliff_start_life=live_deg.cliff_start_life,
                cliff_extra_ms_per_lap=live_deg.cliff_extra_ms_per_lap,
                sc_surplus_ms=sc_surplus_ms,
            )
        )
    return out


def finish_position(ego_finish_ms: float, rival_finishes: Sequence[float]) -> int:
    worse = sum(1 for t in rival_finishes if t < ego_finish_ms)
    return worse + 1


def _soft_clamp(
    value: float,
    *,
    floor: float | None = None,
    ceiling: float | None = None,
) -> float:
    """Bound a projection while keeping it monotonic in the raw input."""
    out = value
    if ceiling is not None and out > ceiling:
        out = ceiling + (out - ceiling) * _BOARD_CLAMP_COMPRESSION
    if floor is not None and out < floor:
        out = floor - (floor - out) * _BOARD_CLAMP_COMPRESSION
    return out


def finish_position_board_blend(raw_pos: int, state: RaceState) -> float:
    """Pull projected finish toward board P when gaps support a front-running read."""
    if state.position > 5:
        return float(raw_pos)
    if state.gap_ahead_ms is not None and state.gap_ahead_ms > _BOARD_BLEND_MAX_GAP_MS:
        return float(raw_pos)
    # Do not project a front-runner to the back unless tyres truly cliff.
    max_drop = 3 + (state.stint_age_laps // 3)
    capped = _soft_clamp(float(raw_pos), ceiling=state.position + max(3, max_drop))
    w = _BOARD_BLEND_WEIGHT
    if capped > state.position + 4:
        w = min(0.55, w + 0.15)
    blended = w * float(state.position) + (1.0 - w) * float(capped)
    return max(1.0, min(float(state.drivers_on_track), blended))


def finish_position_board_sanity(pos: float, state: RaceState) -> float:
    """Clamp projected finish using board gaps (can't pass cars without closing gaps)."""
    p = float(state.position)
    out = float(pos)
    if state.gap_ahead_ms is not None and state.gap_ahead_ms > _SANITY_GAP_AHEAD_FLOOR_MS:
        out = _soft_clamp(out, floor=p)
    if state.gap_behind_ms is not None and state.gap_behind_ms > _SANITY_GAP_BEHIND_CAP_MS:
        laps_left = max(1, state.total_laps - state.lap)
        max_drop = min(
            3.0,
            1.0 + state.gap_behind_ms / 10_000.0 + laps_left / 40.0,
        )
        out = _soft_clamp(out, ceiling=p + max_drop)
    # Stable train: large gaps both sides → finish near current position.
    if (
        state.gap_ahead_ms is not None
        and state.gap_ahead_ms > 5_000
        and state.gap_behind_ms is not None
        and state.gap_behind_ms > 4_000
    ):
        out = _soft_clamp(out, floor=p, ceiling=p + 2.0)
    return max(1.0, min(float(state.drivers_on_track), out))


def projected_finish_position(raw_pos: int, state: RaceState) -> float:
    """MC rank → board blend → gap sanity."""
    blended = finish_position_board_blend(raw_pos, state)
    return finish_position_board_sanity(blended, state)


def _extra_stop_mean_penalty(
    state: RaceState,
    label: str,
    *,
    pit_after_laps: int,
    remaining: int,
) -> float:
    """Discourage further stops when already on 2+ stops and not chasing."""
    if label == "stay_to_finish" or state.pit_count < 2:
        return 0.0
    if pit_after_laps > remaining:
        return 0.0
    if state.gap_ahead_ms is not None and state.gap_ahead_ms > 5_000:
        return 0.5
    return 0.0


def _plan_continuity_credit(label: str) -> float:
    """Switching cost for abandoning a stop lap already called.

    Options sit within Monte Carlo noise of each other around the optimum, so
    without a switching cost the winner is re-drawn every lap and the stop wanders.
    A pit wall does not move a called stop for a hundredth of a position either.
    """
    return _PLAN_CONTINUITY_CREDIT if label == "hold_plan" else 0.0


def simulate_ego_finish_ms(
    snap0: SimSnapshot,
    *,
    pit_after_laps: int,
    pace: PaceSchedule,
    pit_loss_ms: float,
    rng: np.random.Generator,
    noise_ms: float,
    laps_until_sc_end: int = 0,
) -> float:
    """
    Roll from current lap to total_laps.
    pit_after_laps=0 → pit before completing any more racing laps (box next).
    pit_after_laps=N → complete N laps, then pit, then finish.
    If N >= remaining, never pit.
    """
    snap = snap0
    remaining = snap.total_laps - snap.lap
    if remaining <= 0:
        return snap.cumulative_ms

    pitted = False
    target_pit = pit_after_laps
    if target_pit < 0:
        target_pit = remaining + 1  # never

    laps_done = 0
    sc_left = laps_until_sc_end
    while snap.lap < snap.total_laps:
        if (not pitted) and laps_done >= target_pit and target_pit <= remaining:
            snap = _pit_stop(snap, pit_loss_ms)
            pitted = True
        sc_now = 1.0 if sc_left > 0 else 0.0
        roll = SimSnapshot(
            circuit_id=snap.circuit_id,
            lap=snap.lap,
            total_laps=snap.total_laps,
            stint_age=snap.stint_age,
            tyre_life=snap.tyre_life,
            compound_id=snap.compound_id,
            last_lap_ms=snap.last_lap_ms,
            prev_lap_ms=snap.prev_lap_ms,
            prev2_lap_ms=snap.prev2_lap_ms,
            sc_active=sc_now,
            cumulative_ms=snap.cumulative_ms,
        )
        lap_ms = pace.lap_ms(roll, rng, noise_ms)
        sc_after = 1.0 if sc_left > 1 else 0.0
        snap = _apply_lap(snap, lap_ms, sc_after=sc_after)
        if sc_left > 0:
            sc_left -= 1
        laps_done += 1
    return snap.cumulative_ms


def _planned_pit_lap(state_lap: int, pit_after_laps: int) -> int:
    if pit_after_laps <= 0:
        return state_lap + 1
    return state_lap + pit_after_laps


def build_default_options(
    remaining_laps: int,
    *,
    mandatory_pit_pending: bool = False,
    plan_offset: int | None = None,
) -> list[StrategyOption]:
    """Option menu for one lap.

    `plan_offset` is laps from now to an already-committed stop lap. Without it
    the menu is purely relative, so re-picking the same label each lap walks the
    absolute stop forward — a plan that never arrives. The hold_plan card makes
    "keep the stop where we said" a real, scorable choice whose offset shrinks.
    """
    opts: list[StrategyOption] = []
    # At offset <= 1 the committed lap IS the next lap, and pit_next_lap already
    # covers it with the correct pit action — so the plan firms into a box call.
    if plan_offset is not None and 1 < plan_offset <= remaining_laps:
        opts.append(StrategyOption("A", plan_offset, "hold_plan"))
    for n in DEFAULT_STAY_NS:
        if n == 0:
            opts.append(
                StrategyOption(chr(ord("A") + len(opts)), 0, "pit_next_lap")
            )
        elif n < remaining_laps:
            opts.append(
                StrategyOption(
                    chr(ord("A") + len(opts)),
                    n,
                    f"stay_{n}_then_pit",
                )
            )
    # Stay-to-flag only legal once mandatory dry pit is satisfied.
    if remaining_laps > 0 and not mandatory_pit_pending:
        opts.append(
            StrategyOption(
                chr(ord("A") + len(opts)),
                remaining_laps + 1,
                "stay_to_finish",
            )
        )
    # Deduplicate by pit_after_laps.
    seen: set[int] = set()
    uniq: list[StrategyOption] = []
    for o in opts:
        if o.pit_after_laps in seen:
            continue
        seen.add(o.pit_after_laps)
        uniq.append(o)
    # Relabel ids A,B,C...
    out = []
    for i, o in enumerate(uniq):
        out.append(
            StrategyOption(chr(ord("A") + i), o.pit_after_laps, o.label)
        )
    return out


def simulate_strategy_cards(
    replay: RaceReplay,
    state: RaceState,
    *,
    deg: LapDegModel | None = None,
    n_rolls: int = DEFAULT_N_ROLLS,
    pit_loss_ms: float = DEFAULT_PIT_LOSS_MS,
    noise_ms: float = DEFAULT_NOISE_MS,
    seed: int = 42,
    options: Sequence[StrategyOption] | None = None,
    plan_target_lap: int | None = None,
) -> list[OptionCard]:
    if deg is None:
        deg = _default_deg()
    cumul = replay._cumulative_times(state.race_id, state.lap)
    ego_cum = float(cumul.get(state.driver_id, state.cumulative_time_ms))
    snap0 = SimSnapshot.from_state(state, ego_cum)
    ego_green_ms = _recent_green_pace_ms(
        replay,
        state.race_id,
        state.driver_id,
        state.lap,
        state.total_laps,
        board_sc_active=False,
    )
    ego_live_deg = live_deg_for_state(replay, state)
    pace = build_pace_schedule(
        deg,
        snap0,
        green_reference_ms=ego_green_ms,
        live_deg=ego_live_deg,
    )
    remaining = max(0, state.total_laps - state.lap)
    effective_pit_loss = (
        DEFAULT_SC_PIT_LOSS_MS if state.sc_active else pit_loss_ms
    )
    sc_left = _laps_until_sc_end(replay, state.race_id, state.lap, state.total_laps)
    sc_lap_ms = max(snap0.last_lap_ms, _SC_LAP_MS_THRESHOLD) if sc_left > 0 else 0.0
    rivals = rival_finish_times_ms(
        replay,
        state,
        pit_loss_ms=effective_pit_loss,
        sc_surplus_ms=min(sc_left, remaining)
        * max(0.0, sc_lap_ms - pace.fallback_ms),
    )
    rival_noise = noise_ms * math.sqrt(min(remaining, _RIVAL_NOISE_LAP_CAP))
    must_pit = mandatory_dry_pit_pending(state)
    if options is not None:
        opts = list(options)
        if must_pit:
            opts = [o for o in opts if o.pit_after_laps <= remaining]
    else:
        opts = build_default_options(
            remaining,
            mandatory_pit_pending=must_pit,
            plan_offset=(
                plan_target_lap - state.lap if plan_target_lap is not None else None
            ),
        )
    if not opts:
        raise ValueError("no legal strategy options after dry-race pit constraints")
    rng = np.random.default_rng(seed)

    cards: list[OptionCard] = []
    for opt in opts:
        positions: list[int] = []
        times: list[float] = []
        for _ in range(n_rolls):
            finish_ms = simulate_ego_finish_ms(
                snap0,
                pit_after_laps=opt.pit_after_laps,
                pace=pace,
                pit_loss_ms=effective_pit_loss,
                rng=rng,
                noise_ms=noise_ms,
                laps_until_sc_end=sc_left,
            )
            noisy_rivals = [
                r + float(rng.normal(0.0, rival_noise))
                for r in rivals
            ]
            raw_pos = finish_position(finish_ms, noisy_rivals)
            positions.append(projected_finish_position(raw_pos, state))
            times.append(finish_ms)
        pos_a = np.asarray(positions, dtype=np.float64)
        mean_pos = (
            float(np.mean(pos_a))
            + _extra_stop_mean_penalty(
                state, opt.label, pit_after_laps=opt.pit_after_laps, remaining=remaining
            )
            - _plan_continuity_credit(opt.label)
        )
        cards.append(
            OptionCard(
                option_id=opt.option_id,
                label=opt.label,
                pit_after_laps=opt.pit_after_laps,
                n_rolls=n_rolls,
                mean_finish_pos=mean_pos,
                median_finish_pos=float(np.median(pos_a)),
                p_finish_le_3=float(np.mean(pos_a <= 3)),
                p_finish_le_5=float(np.mean(pos_a <= 5)),
                p_finish_le_10=float(np.mean(pos_a <= 10)),
                mean_race_time_ms=float(np.mean(times)),
                planned_pit_lap=_planned_pit_lap(state.lap, opt.pit_after_laps),
            )
        )
    return cards


def oracle_best(cards: Sequence[OptionCard]) -> OptionCard:
    """Lowest mean finish position wins; tie-break on race time."""
    return min(cards, key=lambda c: (c.mean_finish_pos, c.mean_race_time_ms))


def brier_binary(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(payload), indent=2) + "\n", encoding="utf-8")
