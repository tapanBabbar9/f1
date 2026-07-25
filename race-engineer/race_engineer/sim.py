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
from race_engineer.replay import RaceReplay
from race_engineer.state import RaceState

# Default green-flag pit loss (station + in/out). Circuit-specific later.
DEFAULT_PIT_LOSS_MS = 22_000.0
DEFAULT_NOISE_MS = 400.0  # per-lap Gaussian noise in MC
DEFAULT_N_ROLLS = 64
DEFAULT_STAY_NS = (0, 3, 5, 8)  # 0 = pit next lap


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

    def to_dict(self) -> dict[str, Any]:
        return {
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


def build_pace_schedule(deg: LapDegModel | None, snap0: SimSnapshot) -> PaceSchedule:
    fallback = snap0.last_lap_ms
    if deg is None:
        return PaceSchedule(by_compound={}, fallback_ms=fallback)

    by_c: dict[float, tuple[float, float]] = {}
    compounds = {snap0.compound_id, 3.0}  # current + default post-pit HARD
    for cid in compounds:
        life0 = snap0.tyre_life if cid == snap0.compound_id else 0.0
        y0 = _probe_ms(deg, snap0, life0, cid)
        y1 = _probe_ms(deg, snap0, life0 + 8.0, cid)
        # Wear-only: never allow "faster with age" in the MC schedule.
        slope = max(0.0, (y1 - y0) / 8.0)
        # Express as intercept at life=0: y = intercept + slope * life
        intercept = y0 - slope * life0
        # Anchor current compound so first predicted lap stays near board pace.
        if cid == snap0.compound_id:
            predicted_now = intercept + slope * snap0.tyre_life
            shift = snap0.last_lap_ms - predicted_now
            intercept = intercept + shift
        by_c[cid] = (intercept, slope)
    return PaceSchedule(by_compound=by_c, fallback_ms=fallback)


def _apply_lap(snap: SimSnapshot, lap_ms: float) -> SimSnapshot:
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
        sc_active=0.0,  # v0: no SC evolution mid-sim
        cumulative_ms=snap.cumulative_ms + lap_ms,
    )


def _pit_stop(snap: SimSnapshot, pit_loss_ms: float, new_compound_id: float = 3.0) -> SimSnapshot:
    """HARD=3 default for second stint if unknown."""
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


def rival_finish_times_ms(
    replay: RaceReplay,
    state: RaceState,
) -> list[float]:
    """
    Static-rival finish estimates: current cumulative + remaining * recent pace.
    Excludes the ego driver.
    """
    cumul = replay._cumulative_times(state.race_id, state.lap)
    remaining = max(0, state.total_laps - state.lap)
    out: list[float] = []
    for did, cum in cumul.items():
        if did == state.driver_id:
            continue
        laps = replay._laps.get(state.race_id, {}).get(did, {})
        recent = []
        for L in range(state.lap, max(0, state.lap - 2) - 1, -1):
            if L in laps:
                recent.append(laps[L]["milliseconds"])
        pace = float(np.mean(recent)) if recent else 90_000.0
        out.append(float(cum) + remaining * pace)
    return out


def simulate_ego_finish_ms(
    snap0: SimSnapshot,
    *,
    pit_after_laps: int,
    pace: PaceSchedule,
    pit_loss_ms: float,
    rng: np.random.Generator,
    noise_ms: float,
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
    while snap.lap < snap.total_laps:
        # Pit at the start of this iteration if scheduled.
        if (not pitted) and laps_done >= target_pit and target_pit <= remaining:
            snap = _pit_stop(snap, pit_loss_ms)
            pitted = True
        lap_ms = pace.lap_ms(snap, rng, noise_ms)
        snap = _apply_lap(snap, lap_ms)
        laps_done += 1
    return snap.cumulative_ms


def finish_position(ego_finish_ms: float, rival_finishes: Sequence[float]) -> int:
    worse = sum(1 for t in rival_finishes if t < ego_finish_ms)
    return worse + 1


def build_default_options(remaining_laps: int) -> list[StrategyOption]:
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
    # Always include stay-to-flag if not already covered.
    if remaining_laps > 0:
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
    rivals = rival_finish_times_ms(replay, state)
    remaining = max(0, state.total_laps - state.lap)
    opts = list(options) if options is not None else build_default_options(remaining)
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
                pit_loss_ms=pit_loss_ms,
                rng=rng,
                noise_ms=noise_ms,
            )
            # Light noise on rivals so ranking isn't fully deterministic.
            noisy_rivals = [
                r + float(rng.normal(0.0, noise_ms * math.sqrt(max(remaining, 1))))
                for r in rivals
            ]
            pos = finish_position(finish_ms, noisy_rivals)
            positions.append(pos)
            times.append(finish_ms)
        pos_a = np.asarray(positions, dtype=np.float64)
        cards.append(
            OptionCard(
                option_id=opt.option_id,
                label=opt.label,
                pit_after_laps=opt.pit_after_laps,
                n_rolls=n_rolls,
                mean_finish_pos=float(np.mean(pos_a)),
                median_finish_pos=float(np.median(pos_a)),
                p_finish_le_3=float(np.mean(pos_a <= 3)),
                p_finish_le_5=float(np.mean(pos_a <= 5)),
                p_finish_le_10=float(np.mean(pos_a <= 10)),
                mean_race_time_ms=float(np.mean(times)),
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
