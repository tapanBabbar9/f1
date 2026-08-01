"""Phase 5 (simulation): pit-now vs stay-N Monte Carlo option cards."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from race_engineer.lap_deg import LapDegModel
from race_engineer.pit_baseline import COMPOUND_TO_ID
from race_engineer.racing_rules import (
    alternate_dry_compound_id,
    mandatory_dry_pit_pending,
)
from race_engineer.replay import RaceReplay
from race_engineer.state import RaceState

# Default green-flag pit loss (station + in/out). Circuit-specific later.
DEFAULT_PIT_LOSS_MS = 22_000.0
DEFAULT_SC_PIT_LOSS_MS = 12_000.0  # lower effective loss under SC
DEFAULT_NOISE_MS = 400.0  # per-lap Gaussian noise in MC
DEFAULT_N_ROLLS = 64
DEFAULT_STAY_NS = (0, 3, 5, 8)  # 0 = pit next lap
# Lap times above this (ms) are SC-inflated and must not anchor green-flag rollouts.
_SC_LAP_MS_THRESHOLD = 100_000.0
# Fresh-stint pace gain vs each driver's own green-flag lap (ms).
_FRESH_TYRE_GAIN_MS = 600.0
# Cap rival jitter so remaining-lap scaling does not swamp ranking.
_RIVAL_NOISE_LAP_CAP = 12
# Blend sim E[finish] toward board P when gaps are healthy (front-running).
_BOARD_BLEND_WEIGHT = 0.35
_BOARD_BLEND_MAX_GAP_MS = 5_000
# Gap-based sanity: cannot gain places without closing a gap ahead; large
# cushion behind limits positions lost (not race-specific tuning).
_SANITY_GAP_AHEAD_FLOOR_MS = 1_500
_SANITY_GAP_BEHIND_CAP_MS = 4_000


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


@dataclass
class PaceSchedule:
    """
    Fast MC pace: a few HGB probes → linear tyre-life slope per compound.
    Avoids calling the sklearn model on every simulated lap.
    """

    # compound_id -> (intercept_ms at life=0, slope_ms_per_lap)
    by_compound: dict[float, tuple[float, float]]
    fallback_ms: float

    def lap_ms(self, snap: SimSnapshot, rng: np.random.Generator, noise_ms: float) -> float:
        if snap.sc_active > 0.5:
            base = max(snap.last_lap_ms, 100_000.0)
        else:
            pair = self.by_compound.get(snap.compound_id)
            if pair is None:
                base = self.fallback_ms
            else:
                intercept, slope = pair
                base = intercept + slope * snap.tyre_life
        if noise_ms > 0:
            base = base + float(rng.normal(0.0, noise_ms))
        return max(base, 40_000.0)


def _probe_ms(deg: LapDegModel, snap: SimSnapshot, tyre_life: float, compound_id: float) -> float:
    probe = SimSnapshot(
        circuit_id=snap.circuit_id,
        lap=snap.lap,
        total_laps=snap.total_laps,
        stint_age=tyre_life,
        tyre_life=tyre_life,
        compound_id=compound_id,
        last_lap_ms=snap.last_lap_ms,
        prev_lap_ms=snap.prev_lap_ms,
        prev2_lap_ms=snap.prev2_lap_ms,
        sc_active=snap.sc_active,
        cumulative_ms=snap.cumulative_ms,
    )
    return float(deg.model.predict(probe.feature_row())[0])


def _green_flag_anchor_ms(deg: LapDegModel | None, snap0: SimSnapshot) -> float:
    """Pace anchor for MC rollouts; SC / slow laps must not skew green projection."""
    if (
        snap0.sc_active < 0.5
        and snap0.last_lap_ms < _SC_LAP_MS_THRESHOLD
    ):
        return snap0.last_lap_ms
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


def build_pace_schedule(deg: LapDegModel | None, snap0: SimSnapshot) -> PaceSchedule:
    anchor_ms = _green_flag_anchor_ms(deg, snap0)
    if deg is None:
        return PaceSchedule(by_compound={}, fallback_ms=anchor_ms)

    by_c: dict[float, tuple[float, float]] = {}
    compounds = {snap0.compound_id, 3.0}  # current + default post-pit HARD
    green_snap = SimSnapshot(
        circuit_id=snap0.circuit_id,
        lap=snap0.lap,
        total_laps=snap0.total_laps,
        stint_age=snap0.stint_age,
        tyre_life=snap0.tyre_life,
        compound_id=snap0.compound_id,
        last_lap_ms=anchor_ms,
        prev_lap_ms=snap0.prev_lap_ms,
        prev2_lap_ms=snap0.prev2_lap_ms,
        sc_active=0.0,
        cumulative_ms=snap0.cumulative_ms,
    )
    for cid in compounds:
        life0 = snap0.tyre_life if cid == snap0.compound_id else 0.0
        probe = green_snap if cid == snap0.compound_id else green_snap
        y0 = _probe_ms(deg, probe, life0, cid)
        y1 = _probe_ms(deg, probe, life0 + 8.0, cid)
        # Wear-only: never allow "faster with age" in the MC schedule.
        slope = max(0.0, (y1 - y0) / 8.0)
        # Express as intercept at life=0: y = intercept + slope * life
        intercept = y0 - slope * life0
        # Anchor current compound so first predicted lap stays near board pace.
        if cid == snap0.compound_id:
            predicted_now = intercept + slope * snap0.tyre_life
            shift = anchor_ms - predicted_now
            intercept = intercept + shift
        elif cid == 3.0:
            # Post-pit HARD: align with fresh-tyre pace used for rival projection.
            fresh = _fresh_tyre_pace_ms(anchor_ms)
            intercept = fresh
            slope = max(slope, 0.0)
        by_c[cid] = (intercept, slope)
    return PaceSchedule(by_compound=by_c, fallback_ms=anchor_ms)


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


def _project_driver_finish_ms(
    replay: RaceReplay,
    state: RaceState,
    driver_id: int,
    cum: float,
    *,
    remaining: int,
    pace_ms: float,
    post_pit_pace_ms: float,
    pit_loss_ms: float,
) -> float:
    """Historical replay projection: rivals follow known future pit laps."""
    if remaining <= 0:
        return cum
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
            + laps_before * pace_ms
            + pit_loss_ms
            + laps_after * post_pit_pace_ms
        )
    return float(cum) + remaining * pace_ms


def _fresh_tyre_pace_ms(pre_pit_pace_ms: float) -> float:
    return max(pre_pit_pace_ms - _FRESH_TYRE_GAIN_MS, 85_000.0)


def rival_finish_times_ms(
    replay: RaceReplay,
    state: RaceState,
    *,
    pit_loss_ms: float = DEFAULT_PIT_LOSS_MS,
) -> list[float]:
    """
    Rival finish estimates: each driver's green-flag pace plus known future pit
    laps and a driver-specific fresh-tyre offset. Excludes the ego driver.
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
            board_sc_active=state.sc_active,
        )
        post_pit = _fresh_tyre_pace_ms(pace)
        out.append(
            _project_driver_finish_ms(
                replay,
                state,
                did,
                float(cum),
                remaining=remaining,
                pace_ms=pace,
                post_pit_pace_ms=post_pit,
                pit_loss_ms=pit_loss_ms,
            )
        )
    return out


def finish_position(ego_finish_ms: float, rival_finishes: Sequence[float]) -> int:
    worse = sum(1 for t in rival_finishes if t < ego_finish_ms)
    return worse + 1


def finish_position_board_blend(raw_pos: int, state: RaceState) -> float:
    """Pull projected finish toward board P when gaps support a front-running read."""
    if state.position > 5:
        return float(raw_pos)
    if state.gap_ahead_ms is not None and state.gap_ahead_ms > _BOARD_BLEND_MAX_GAP_MS:
        return float(raw_pos)
    # Do not project a front-runner to the back unless tyres truly cliff.
    max_drop = 3 + (state.stint_age_laps // 3)
    capped = min(raw_pos, state.position + max(3, max_drop))
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
        out = max(out, p)
    if state.gap_behind_ms is not None and state.gap_behind_ms > _SANITY_GAP_BEHIND_CAP_MS:
        laps_left = max(1, state.total_laps - state.lap)
        max_drop = min(
            3.0,
            1.0 + state.gap_behind_ms / 10_000.0 + laps_left / 40.0,
        )
        out = min(out, p + max_drop)
    # Stable train: large gaps both sides → finish near current position.
    if (
        state.gap_ahead_ms is not None
        and state.gap_ahead_ms > 5_000
        and state.gap_behind_ms is not None
        and state.gap_behind_ms > 4_000
    ):
        out = min(out, p + 2.0)
        out = max(out, p)
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
) -> list[StrategyOption]:
    opts: list[StrategyOption] = []
    for n in DEFAULT_STAY_NS:
        if n == 0:
            opts.append(
                StrategyOption("A", 0, "pit_next_lap")
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
) -> list[OptionCard]:
    if deg is None:
        deg = _default_deg()
    cumul = replay._cumulative_times(state.race_id, state.lap)
    ego_cum = float(cumul.get(state.driver_id, state.cumulative_time_ms))
    snap0 = SimSnapshot.from_state(state, ego_cum)
    pace = build_pace_schedule(deg, snap0)
    remaining = max(0, state.total_laps - state.lap)
    effective_pit_loss = (
        DEFAULT_SC_PIT_LOSS_MS if state.sc_active else pit_loss_ms
    )
    rivals = rival_finish_times_ms(
        replay,
        state,
        pit_loss_ms=effective_pit_loss,
    )
    sc_left = _laps_until_sc_end(replay, state.race_id, state.lap, state.total_laps)
    rival_noise = noise_ms * math.sqrt(min(remaining, _RIVAL_NOISE_LAP_CAP))
    must_pit = mandatory_dry_pit_pending(state)
    if options is not None:
        opts = list(options)
        if must_pit:
            opts = [o for o in opts if o.pit_after_laps <= remaining]
    else:
        opts = build_default_options(remaining, mandatory_pit_pending=must_pit)
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
        mean_pos = float(np.mean(pos_a)) + _extra_stop_mean_penalty(
            state, opt.label, pit_after_laps=opt.pit_after_laps, remaining=remaining
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
