type Series = {
  id: string;
  color: string;
  values: (number | null)[];
};

type Marker = {
  index: number;
  kind: "pit" | "sc";
};

type Props = {
  title: string;
  labels: number[];
  series: Series[];
  markers?: Marker[];
  formatY: (v: number) => string;
  invertY?: boolean;
  height?: number;
};

function extent(series: Series[]): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series) {
    for (const v of s.values) {
      if (v == null || Number.isNaN(v)) continue;
      lo = Math.min(lo, v);
      hi = Math.max(hi, v);
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  if (lo === hi) return [lo - 1, hi + 1];
  const pad = (hi - lo) * 0.08;
  return [lo - pad, hi + pad];
}

function polyline(
  values: (number | null)[],
  xAt: (i: number) => number,
  yAt: (v: number) => number,
): string {
  const parts: string[] = [];
  let drawing = false;
  values.forEach((v, i) => {
    if (v == null || Number.isNaN(v)) {
      drawing = false;
      return;
    }
    const cmd = drawing ? "L" : "M";
    parts.push(`${cmd}${xAt(i).toFixed(1)} ${yAt(v).toFixed(1)}`);
    drawing = true;
  });
  return parts.join(" ");
}

export function LineChart({
  title,
  labels,
  series,
  markers = [],
  formatY,
  invertY = false,
  height = 160,
}: Props) {
  const w = 640;
  const h = height;
  const padL = 48;
  const padR = 12;
  const padT = 12;
  const padB = 28;
  const n = Math.max(labels.length, 1);
  const [yMin, yMax] = extent(series);
  const xAt = (i: number) =>
    padL + (n <= 1 ? 0 : (i / (n - 1)) * (w - padL - padR));
  const yAt = (v: number) => {
    const t = (v - yMin) / (yMax - yMin);
    const u = invertY ? t : 1 - t;
    return padT + u * (h - padT - padB);
  };

  const yTicks = [yMin, (yMin + yMax) / 2, yMax];
  const xLabelIdx =
    n <= 6
      ? labels.map((_, i) => i)
      : [0, Math.floor((n - 1) / 2), n - 1].filter(
          (v, i, a) => a.indexOf(v) === i,
        );

  return (
    <figure className="chart">
      <figcaption className="chart-title">{title}</figcaption>
      <svg
        className="chart-svg"
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        aria-label={title}
      >
        {yTicks.map((v, i) => (
          <g key={i}>
            <line
              className="chart-grid"
              x1={padL}
              x2={w - padR}
              y1={yAt(v)}
              y2={yAt(v)}
            />
            <text className="chart-axis" x={padL - 6} y={yAt(v) + 3} textAnchor="end">
              {formatY(v)}
            </text>
          </g>
        ))}

        {markers.map((m) => (
          <line
            key={`${m.kind}-${m.index}`}
            className={`chart-marker chart-marker--${m.kind}`}
            x1={xAt(m.index)}
            x2={xAt(m.index)}
            y1={padT}
            y2={h - padB}
          />
        ))}

        {series.map((s) => (
          <path
            key={s.id}
            className="chart-line"
            d={polyline(s.values, xAt, yAt)}
            style={{ stroke: s.color }}
            fill="none"
          />
        ))}

        {series.map((s) => {
          const last = [...s.values]
            .map((v, i) => ({ v, i }))
            .reverse()
            .find((p) => p.v != null);
          if (!last || last.v == null) return null;
          return (
            <circle
              key={`${s.id}-dot`}
              cx={xAt(last.i)}
              cy={yAt(last.v)}
              r={3.5}
              style={{ fill: s.color }}
            />
          );
        })}

        {xLabelIdx.map((i) => (
          <text
            key={i}
            className="chart-axis"
            x={xAt(i)}
            y={h - 8}
            textAnchor="middle"
          >
            L{labels[i]}
          </text>
        ))}
      </svg>
      {series.length > 1 ? (
        <ul className="chart-legend">
          {series.map((s) => (
            <li key={s.id}>
              <span style={{ background: s.color }} />
              {s.id}
            </li>
          ))}
        </ul>
      ) : null}
    </figure>
  );
}
