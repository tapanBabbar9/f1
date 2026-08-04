import { formatClock } from "../format";

const SPEEDS = [1, 4, 16] as const;

type Props = {
  playing: boolean;
  speed: number;
  t_s: number;
  total_s: number;
  onToggle: () => void;
  onSpeed: (s: number) => void;
  onSeek: (t: number) => void;
};

export function Transport({
  playing,
  speed,
  t_s,
  total_s,
  onToggle,
  onSpeed,
  onSeek,
}: Props) {
  return (
    <section className="transport" aria-label="Playback transport">
      <button type="button" className="btn" onClick={onToggle}>
        {playing ? "Pause" : "Play"}
      </button>
      <div className="speed-group" role="group" aria-label="Playback speed">
        {SPEEDS.map((s) => (
          <button
            key={s}
            type="button"
            className={`btn btn--sm ${speed === s ? "is-active" : ""}`}
            onClick={() => onSpeed(s)}
          >
            {s}×
          </button>
        ))}
      </div>
      <label className="scrub">
        <span className="scrub-clock">
          {formatClock(t_s)} / {formatClock(total_s)}
        </span>
        <input
          type="range"
          min={0}
          max={Math.max(0, total_s - 1)}
          value={Math.min(t_s, Math.max(0, total_s - 1))}
          onChange={(e) => onSeek(Number(e.target.value))}
        />
      </label>
    </section>
  );
}
