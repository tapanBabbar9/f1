type Props = {
  radio: string | null;
  committed: boolean;
};

export function Radio({ radio, committed }: Props) {
  return (
    <section className="radio" aria-label="Race engineer radio">
      <header className="radio-head">
        <span className="radio-tag">RADIO</span>
        <span className="radio-sub">
          {committed ? "on air" : "gated — waits for SLOW commit"}
        </span>
      </header>
      {radio ? (
        <p className="radio-msg" key={radio}>
          “{radio}”
        </p>
      ) : (
        <p className="radio-muted">—</p>
      )}
    </section>
  );
}
