function pad(n: number, w = 2): string {
  return String(n).padStart(w, "0");
}

/** Format race-clock seconds as H:MM:SS or M:SS. */
export function formatClock(t_s: number): string {
  const t = Math.max(0, Math.floor(t_s));
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = t % 60;
  if (h > 0) return `${h}:${pad(m)}:${pad(s)}`;
  return `${m}:${pad(s)}`;
}

export function formatGap(ms: number | null, sign: "+" | "-"): string {
  if (ms == null) return "—";
  return `${sign}${(ms / 1000).toFixed(3)}s`;
}

export function formatLapMs(ms: number): string {
  const total = ms / 1000;
  const m = Math.floor(total / 60);
  const s = total - m * 60;
  if (m > 0) return `${m}:${s.toFixed(3).padStart(6, "0")}`;
  return `${s.toFixed(3)}s`;
}
