/** Canonical RaceState — mirrors f1-pitwall/shared/state.py to_dict(). */
export type RaceState = {
  race_id: number;
  year: number;
  race_name: string;
  circuit_id: number;
  circuit_name: string;
  driver_id: number;
  driver_ref: string;
  driver_code: string | null;
  lap: number;
  total_laps: number;
  position: number;
  gap_ahead_ms: number | null;
  gap_behind_ms: number | null;
  stint_age_laps: number;
  pit_this_lap: boolean;
  pit_count: number;
  last_lap_time_ms: number;
  last_lap_times_ms: number[];
  cumulative_time_ms: number;
  drivers_on_track: number;
  tyre_compound: string | null;
  tyre_life: number | null;
  sc_active: boolean;
  laps_since_sc_deploy: number;
  sc_deployed_this_lap: boolean;
  sc_deployed_prev_lap: boolean;
};

export type FastDecision = {
  available_at_ms: number;
  action: "pit" | "stay";
  pit_next: number;
  planned_pit_lap: number | null;
  source: "hgb";
};

export type SlowDecision = {
  available_at_ms: number;
  action: "pit" | "stay";
  tyre: string | null;
  push: string | null;
  planned_pit_lap: number | null;
  chosen_label: string | null;
  reason: string | null;
  rationale: string | null;
  driver_message: string | null;
  regret: number | null;
  source: "openai_sim";
};

export type LogLap = {
  t_start_s: number;
  duration_s: number;
  state: RaceState;
  fast: FastDecision | null;
  slow: SlowDecision | null;
};

export type PlaybackLog = {
  sample_id: string;
  hz: number;
  race_id: number;
  driver_id: number;
  meta: {
    year: number;
    race_name: string;
    circuit_name: string;
    driver_code: string;
    driver_ref: string;
    total_laps: number;
    total_duration_s: number;
  };
  laps: LogLap[];
};

export type TickView = {
  t_s: number;
  lapIndex: number;
  msIntoLap: number;
  state: RaceState;
  fast: FastDecision | null;
  slow: SlowDecision | null;
  /** SLOW has arrived for this lap. */
  committed: boolean;
  /** Radio unlocked only after SLOW commit. */
  radio: string | null;
  provisional: FastDecision | null;
};
