import type { LogLap, PlaybackLog, RaceState } from "./types";

export type LapPoint = {
  lap: number;
  last_lap_time_ms: number;
  position: number;
  gap_ahead_ms: number | null;
  gap_behind_ms: number | null;
  tyre_compound: string | null;
  pit_this_lap: boolean;
  sc_active: boolean;
  stint_age_laps: number;
};

export function visibleLaps(log: PlaybackLog, upToLapIndex: number): LogLap[] {
  return log.laps.slice(0, Math.max(0, upToLapIndex + 1));
}

export function toPoints(laps: LogLap[]): LapPoint[] {
  return laps.map((l) => {
    const s: RaceState = l.state;
    return {
      lap: s.lap,
      last_lap_time_ms: s.last_lap_time_ms,
      position: s.position,
      gap_ahead_ms: s.gap_ahead_ms,
      gap_behind_ms: s.gap_behind_ms,
      tyre_compound: s.tyre_compound,
      pit_this_lap: s.pit_this_lap,
      sc_active: s.sc_active,
      stint_age_laps: s.stint_age_laps,
    };
  });
}

/** Racing laps only for pace charts — drop SC / huge outliers that squash the scale. */
export function pacePoints(points: LapPoint[]): LapPoint[] {
  if (points.length === 0) return points;
  const racing = points.filter((p) => !p.sc_active && p.last_lap_time_ms < 130_000);
  if (racing.length >= 3) return racing;
  // Fallback: drop top 15% slowest so SC opens don't dominate
  const sorted = [...points].sort(
    (a, b) => a.last_lap_time_ms - b.last_lap_time_ms,
  );
  const keep = Math.max(3, Math.ceil(sorted.length * 0.85));
  const cutoff = sorted[keep - 1]?.last_lap_time_ms ?? Infinity;
  return points.filter((p) => p.last_lap_time_ms <= cutoff);
}
