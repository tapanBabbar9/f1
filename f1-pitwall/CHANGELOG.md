# Changelog

Dated notes for behaviour fixes. The README stays onboarding; this file is for what broke and what changed.

## 2026-08-07

### Null `planned_pit_lap` in trajectory export

Harness trajectories showed top-level `planned_pit_lap: null` even when the chosen option card had a stop lap. Cause: `attach_radio` rebuilt `SimAgentDecision` field-by-field and omitted `chosen_planned_pit_lap`, so it fell back to `None` after the radio pass. Fix: use `dataclasses.replace(...)` so only `decision` / `trajectory` / `radio` are updated. Regression tests assert every strategy field survives the radio attach.

### Radio repetition under similar boards

The radio agent was stateless per lap, so a stable board (e.g. safety-car stay) produced near-identical calls — lap 4 of race 1052 matched lap 2 byte-for-byte. Fix: `race_engineer/radio_log.py` keeps the last four transmissions per driver, feeds them into the prompt, and the OpenAI backend re-asks once on heavy word overlap with a recent call. No per-situation phrasing rules; the model decides what changed. Heuristic radio remains a thin offline fallback and does not vary on history.

## 2026-08-06

### Sliding planned stop

Re-evaluating each lap walked the absolute stop forward (`L9 → L10 → …`) even when the strategy intent was unchanged. Three causes:

1. **Wear from the lap-deg probe** was flat after a few laps and identical across compounds, so a fresh tyre looked no faster than the worn one and waiting always won. Forward wear now uses a saturating compound prior `wear(life) = total · (1 − e^(−life/τ))`, with live stint fit when enough clean laps exist (see below).
2. **Post-pit pace** was derived from the worn current lap (`_fresh_tyre_pace_ms(anchor)`), so the ideal stop receded as the old set aged. Wear is stripped from the anchor before applying the fresh-tyre gain; the optimum is time-consistent across re-asks.
3. **Option grid** capped stay offsets at 8, so the argmax was often the last card. Grid extends to 25; `hold_plan` scores the committed lap with a small continuity credit so Monte Carlo noise does not redraw a called stop.

Memory stored only the option label (`stay_8_then_pit`), so prompts looked continuous while the target drifted. `MemoryEntry` now carries absolute `planned_pit_lap`; prompts render `target stop LN`; the summarisation key includes the target so drift breaks the run. `committed_pit_lap()` feeds `hold_plan` and a "Committed plan: stop on lap N" line.

Harness calibration on 8 races / 29 stints: scorable-plan stints 11 → 24, timing MAE 8.1 → 5.0, within ±2 laps 0% → 46%.

### Live stint degradation

Generic wear alone is not circuit- or driver-specific. `estimate_live_stint_deg` fits deg from this driver's recent green laps in the current stint (Theil–Sen slope, fuel-corrected, SC/pit/slow outliers dropped), pools with the compound prior, and detects a late-stint cliff from sustained residual acceleration. Rivals are fitted independently. `simulate_strategies.live_tyre_deg` exposes source, sample count, slope, and cliff state. Falls back to the prior until enough clean laps exist.
