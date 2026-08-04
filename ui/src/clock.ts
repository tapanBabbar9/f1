import type { PlaybackLog, TickView } from "./types";

/** Resolve which lap owns race-clock second t. */
export function lapIndexAt(log: PlaybackLog, t_s: number): number {
  const t = Math.max(0, Math.min(t_s, log.meta.total_duration_s - 1));
  let lo = 0;
  let hi = log.laps.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (log.laps[mid].t_start_s <= t) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

/**
 * 1Hz decide-loop mock for one race-clock second.
 * FAST is provisional at lap start; SLOW commits after available_at_ms; radio only then.
 */
export function tickAt(log: PlaybackLog, t_s: number): TickView {
  const lapIndex = lapIndexAt(log, t_s);
  const lap = log.laps[lapIndex];
  const msIntoLap = Math.max(0, (t_s - lap.t_start_s) * 1000);
  const fast = lap.fast;
  const slow = lap.slow;
  const provisional =
    fast && msIntoLap >= fast.available_at_ms ? fast : null;
  const committed = Boolean(slow && msIntoLap >= slow.available_at_ms);
  const radio =
    committed && slow?.driver_message ? slow.driver_message : null;

  return {
    t_s,
    lapIndex,
    msIntoLap,
    state: lap.state,
    fast,
    slow,
    committed,
    radio,
    provisional,
  };
}
