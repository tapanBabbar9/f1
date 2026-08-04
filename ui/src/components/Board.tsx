import type { RaceState } from "../types";
import { formatGap, formatLapMs } from "../format";

type Props = {
  state: RaceState;
  flashKey: number;
};

export function Board({ state, flashKey }: Props) {
  const compound = state.tyre_compound ?? "unknown";
  const age =
    state.tyre_life != null
      ? `${state.tyre_life} laps (set)`
      : `${state.stint_age_laps} laps`;
  const sc = state.sc_active
    ? `Yes (deployed ${state.laps_since_sc_deploy} laps ago)`
    : "No";

  return (
    <section className="board" key={flashKey} aria-label="Pit wall board">
      <div className="board-grid">
        <div className="board-cell board-cell--hero">
          <span className="board-label">Lap</span>
          <span className="board-value board-value--lg">
            {state.lap}
            <span className="board-muted"> / {state.total_laps}</span>
          </span>
        </div>
        <div className="board-cell board-cell--hero">
          <span className="board-label">Position</span>
          <span className="board-value board-value--lg">P{state.position}</span>
        </div>
        <div className="board-cell">
          <span className="board-label">Gap Ahead</span>
          <span className="board-value">
            {formatGap(state.gap_ahead_ms, "+")}
          </span>
        </div>
        <div className="board-cell">
          <span className="board-label">Gap Behind</span>
          <span className="board-value">
            {formatGap(state.gap_behind_ms, "-")}
          </span>
        </div>
        <div className="board-cell">
          <span className="board-label">Compound</span>
          <span
            className={`board-value tyre tyre--${compound.toLowerCase()}`}
          >
            {compound}
          </span>
        </div>
        <div className="board-cell">
          <span className="board-label">Tyre Age</span>
          <span className="board-value">{age}</span>
        </div>
        <div className="board-cell">
          <span className="board-label">Safety Car</span>
          <span
            className={`board-value ${state.sc_active ? "flag-sc" : ""}`}
          >
            {sc}
          </span>
        </div>
        <div className="board-cell">
          <span className="board-label">Pit this lap</span>
          <span
            className={`board-value ${state.pit_this_lap ? "flag-pit" : ""}`}
          >
            {state.pit_this_lap ? "Yes" : "No"}
          </span>
        </div>
        <div className="board-cell">
          <span className="board-label">Pit stops</span>
          <span className="board-value">{state.pit_count}</span>
        </div>
        <div className="board-cell">
          <span className="board-label">Last lap</span>
          <span className="board-value">
            {formatLapMs(state.last_lap_time_ms)}
          </span>
        </div>
        <div className="board-cell">
          <span className="board-label">Cars on track</span>
          <span className="board-value">{state.drivers_on_track}</span>
        </div>
        <div className="board-cell">
          <span className="board-label">Stint age</span>
          <span className="board-value">{state.stint_age_laps} laps</span>
        </div>
      </div>
    </section>
  );
}
