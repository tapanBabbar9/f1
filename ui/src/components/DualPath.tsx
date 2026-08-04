import type { FastDecision, SlowDecision } from "../types";

type Props = {
  provisional: FastDecision | null;
  slow: SlowDecision | null;
  committed: boolean;
  msIntoLap: number;
};

export function DualPath({ provisional, slow, committed, msIntoLap }: Props) {
  const slowEta =
    slow && !committed
      ? Math.max(0, Math.ceil((slow.available_at_ms - msIntoLap) / 1000))
      : 0;

  return (
    <section className="dual" aria-label="Dual path decide loop">
      <article className="path path--fast">
        <header className="path-head">
          <span className="path-tag">FAST</span>
          <span className="path-sub">HGB · provisional</span>
        </header>
        {provisional ? (
          <div className="path-body">
            <p className={`call call--${provisional.action}`}>
              {provisional.action.toUpperCase()}
            </p>
            <p className="path-meta">
              plan L
              {provisional.planned_pit_lap != null
                ? provisional.planned_pit_lap
                : "—"}
              <span className="path-latency">0 ms</span>
            </p>
          </div>
        ) : (
          <p className="path-empty">Waiting for lap tick…</p>
        )}
      </article>

      <article className={`path path--slow ${committed ? "is-committed" : ""}`}>
        <header className="path-head">
          <span className="path-tag">SLOW</span>
          <span className="path-sub">
            {committed ? "openai_sim · committed" : "openai_sim · pending"}
          </span>
        </header>
        {committed && slow ? (
          <div className="path-body">
            <p className={`call call--${slow.action}`}>
              {slow.action.toUpperCase()}
              {slow.tyre ? ` · ${slow.tyre}` : ""}
            </p>
            <p className="path-meta">
              {slow.chosen_label ?? `plan L${slow.planned_pit_lap ?? "—"}`}
              <span className="path-latency">{slow.available_at_ms} ms</span>
            </p>
            {slow.reason ? <p className="path-reason">{slow.reason}</p> : null}
          </div>
        ) : (
          <p className="path-empty">
            {slow
              ? `Sim + strategy in flight… ${slowEta}s`
              : "No SLOW decision for this lap"}
          </p>
        )}
      </article>
    </section>
  );
}
