import { useMemo } from "react";
import { formatGap, formatLapMs } from "../format";
import { pacePoints, toPoints, visibleLaps, type LapPoint } from "../series";
import type { PlaybackLog } from "../types";
import { LineChart } from "./LineChart";

type Props = {
  log: PlaybackLog;
  lapIndex: number;
};

function markersFrom(points: LapPoint[]) {
  const out: { index: number; kind: "pit" | "sc" }[] = [];
  points.forEach((p, i) => {
    if (p.pit_this_lap) out.push({ index: i, kind: "pit" });
    else if (p.sc_active) out.push({ index: i, kind: "sc" });
  });
  return out;
}

export function Telemetry({ log, lapIndex }: Props) {
  const points = useMemo(
    () => toPoints(visibleLaps(log, lapIndex)),
    [log, lapIndex],
  );
  const pace = useMemo(() => pacePoints(points), [points]);
  const recent = points.slice(-12).reverse();

  const lapLabels = points.map((p) => p.lap);
  const paceLabels = pace.map((p) => p.lap);

  return (
    <section className="telemetry" aria-label="Telemetry graphs">
      <div className="telemetry-charts">
        <LineChart
          title="Lap time"
          labels={paceLabels}
          series={[
            {
              id: "lap",
              color: "var(--fast)",
              values: pace.map((p) => p.last_lap_time_ms / 1000),
            },
          ]}
          markers={markersFrom(pace)}
          formatY={(v) => `${v.toFixed(1)}`}
        />
        <LineChart
          title="Position"
          labels={lapLabels}
          series={[
            {
              id: "pos",
              color: "var(--warn)",
              values: points.map((p) => p.position),
            },
          ]}
          markers={markersFrom(points)}
          formatY={(v) => `P${Math.round(v)}`}
          invertY
          height={140}
        />
        <LineChart
          title="Gaps"
          labels={lapLabels}
          series={[
            {
              id: "ahead",
              color: "var(--accent)",
              values: points.map((p) =>
                p.gap_ahead_ms == null ? null : p.gap_ahead_ms / 1000,
              ),
            },
            {
              id: "behind",
              color: "var(--slow)",
              values: points.map((p) =>
                p.gap_behind_ms == null ? null : p.gap_behind_ms / 1000,
              ),
            },
          ]}
          markers={markersFrom(points)}
          formatY={(v) => `${v.toFixed(1)}s`}
          height={140}
        />
      </div>

      <div className="lap-table-wrap">
        <header className="lap-table-head">
          <span className="radio-tag">LAP TIMES</span>
          <span className="radio-sub">last {recent.length} · RaceState</span>
        </header>
        <table className="lap-table">
          <thead>
            <tr>
              <th>Lap</th>
              <th>Time</th>
              <th>Pos</th>
              <th>+Ahead</th>
              <th>-Behind</th>
              <th>Tyre</th>
              <th>Flags</th>
            </tr>
          </thead>
          <tbody>
            {recent.map((p) => (
              <tr
                key={p.lap}
                className={
                  p.lap === points[points.length - 1]?.lap ? "is-current" : ""
                }
              >
                <td>{p.lap}</td>
                <td className="mono">{formatLapMs(p.last_lap_time_ms)}</td>
                <td>P{p.position}</td>
                <td className="mono">{formatGap(p.gap_ahead_ms, "+")}</td>
                <td className="mono">{formatGap(p.gap_behind_ms, "-")}</td>
                <td>
                  <span
                    className={`tyre tyre--${(p.tyre_compound ?? "unknown").toLowerCase()}`}
                  >
                    {p.tyre_compound ?? "—"}
                  </span>
                  <span className="lap-age"> {p.stint_age_laps}L</span>
                </td>
                <td className="flags">
                  {p.pit_this_lap ? <span className="pill pill--pit">PIT</span> : null}
                  {p.sc_active ? <span className="pill pill--sc">SC</span> : null}
                  {!p.pit_this_lap && !p.sc_active ? "—" : null}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="chart-hint">
          Vertical markers: <span className="hint-pit">pit</span> ·{" "}
          <span className="hint-sc">SC</span>. Lap-time scale excludes SC /
          outliers.
        </p>
      </div>
    </section>
  );
}
