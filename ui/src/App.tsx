import { useEffect, useMemo, useRef, useState } from "react";
import { tickAt } from "./clock";
import { Board } from "./components/Board";
import { DualPath } from "./components/DualPath";
import { Radio } from "./components/Radio";
import { Telemetry } from "./components/Telemetry";
import { Transport } from "./components/Transport";
import type { PlaybackLog } from "./types";

export default function App() {
  const [log, setLog] = useState<PlaybackLog | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(16);
  const accRef = useRef(0);
  const lastRef = useRef<number | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/logs/1052_1.json")
      .then((r) => {
        if (!r.ok) throw new Error(`Failed to load log (${r.status})`);
        return r.json() as Promise<PlaybackLog>;
      })
      .then((data) => {
        if (!cancelled) setLog(data);
      })
      .catch((e: unknown) => {
        if (!cancelled)
          setError(e instanceof Error ? e.message : "Failed to load log");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!log || !playing) {
      lastRef.current = null;
      return;
    }
    let raf = 0;
    const total = log.meta.total_duration_s;
    const step = (now: number) => {
      if (lastRef.current == null) lastRef.current = now;
      const dt = (now - lastRef.current) / 1000;
      lastRef.current = now;
      accRef.current += dt * speed;
      if (accRef.current >= 1) {
        const advance = Math.floor(accRef.current);
        accRef.current -= advance;
        setT((prev) => {
          const next = prev + advance;
          if (next >= total) {
            setPlaying(false);
            return total - 1;
          }
          return next;
        });
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [log, playing, speed]);

  const view = useMemo(() => (log ? tickAt(log, t) : null), [log, t]);

  if (error) {
    return (
      <main className="shell">
        <p className="error">{error}</p>
      </main>
    );
  }

  if (!log || !view) {
    return (
      <main className="shell">
        <p className="loading">Loading pit-wall feed…</p>
      </main>
    );
  }

  const { meta } = log;

  return (
    <main className="shell">
      <div className="atmosphere" aria-hidden="true" />
      <header className="hero">
        <p className="hero-kicker">Pit wall · live loop mock</p>
        <h1 className="hero-title">
          <span className="hero-driver">{meta.driver_code}</span>
          <span className="hero-race">
            {meta.race_name} {meta.year}
          </span>
        </h1>
        <p className="hero-sub">{meta.circuit_name}</p>
      </header>

      <Transport
        playing={playing}
        speed={speed}
        t_s={t}
        total_s={meta.total_duration_s}
        onToggle={() => setPlaying((p) => !p)}
        onSpeed={setSpeed}
        onSeek={(next) => {
          setT(next);
          accRef.current = 0;
        }}
      />

      <Board state={view.state} flashKey={view.state.lap} />

      <DualPath
        provisional={view.provisional}
        slow={view.slow}
        committed={view.committed}
        msIntoLap={view.msIntoLap}
      />

      <Radio radio={view.radio} committed={view.committed} />

      <Telemetry log={log} lapIndex={view.lapIndex} />
    </main>
  );
}
