# race-engineer

AI Race Engineer package.

Same moment walks through every live phase: **Hamilton, British GP 2024, lap 22**.

```text
race-engineer/
  race_engineer/
    state.py / replay.py / integrity.py
    pit_baseline.py      # Phase 1
    crew_chief.py        # Phase 2 schema + prompts
    llm.py               # openai + heuristic backends
    crew_chief_eval.py
    tools.py / tool_agent.py / faithfulness.py   # Phase 3
    lap_deg.py           # Phase 4 next-lap pace
    sim.py               # Phase 5 Monte Carlo option cards
    sim_agent.py         # Phase 6 reasons over sims
    memory.py            # Phase 7 lap-to-lap pit-wall memory
    harness.py           # Phase 8 eval harness 1.0.0
    racing_rules.py      # v0 dry-race mandatory pit (prompt + sim filter)
  scripts/
    print_sample_state.py / run_integrity.py / update_dataset.py
    build_tyre_laps.py / train_pit_baseline.py / eval_pit_baseline.py
    score_pit_sample.py / decide_once.py / run_crew_chief_eval.py
    decide_once_tools.py / run_tools_eval.py
    train_lap_deg.py / predict_lap_deg.py
    simulate_once.py / eval_sim.py
    decide_once_sim.py / eval_sim_agent.py
    replay_race.py / eval_memory.py
    run_harness.py
  artifacts/
    pit_baseline/metrics.json
    crew_chief/metrics.json
    tools/metrics.json
    lap_deg/metrics.json
    sim/metrics.json
    sim_agent/metrics.json
    memory/metrics.json
    eval/frozen_races.json
    eval/metrics.json
    eval/harness_manifest.json
  requirements.txt
```

```bash
python3 -m venv race-engineer/.venv
race-engineer/.venv/bin/pip install -r race-engineer/requirements.txt
```



## Phase 0 — race state (world model)

**What it does:** turn historical timing into a pit-wall snapshot for one `(race, driver, lap)`.

```bash
python3 race-engineer/scripts/print_sample_state.py --year 2024 --name-contains British --driver-id 1 --lap 22
```

```text
Race: British Grand Prix (2024)
Driver: HAM
Circuit: Silverstone Circuit
Lap: 22 / 52
Current Position: P3
Gap Ahead: +1.001s
Gap Behind: -1.688s
Tyres:
  Compound: MEDIUM
  Age: 22 laps (set)
Safety Car: No
Pit this lap: No
Pit stops so far: 0
Last lap: 91095 ms
Cars on track: 19

--- raw ---
race_id: 1132
year: 2024
race_name: British Grand Prix
circuit_id: 9
circuit_name: Silverstone Circuit
driver_id: 1
driver_ref: hamilton
driver_code: HAM
lap: 22
total_laps: 52
position: 3
gap_ahead_ms: 1001
gap_behind_ms: 1688
stint_age_laps: 22
pit_this_lap: False
pit_count: 0
last_lap_time_ms: 91095
last_lap_times_ms: (95071, 103711, 100021, 92156, 91095)
cumulative_time_ms: 2046904
drivers_on_track: 19
tyre_compound: MEDIUM
tyre_life: 22
sc_active: False
laps_since_sc_deploy: 0
sc_deployed_this_lap: False
sc_deployed_prev_lap: False
```

Integrity: `python3 race-engineer/scripts/run_integrity.py --samples 2000` → **100%** on 2k samples (2021–2025).

## Phase 1 — pit baseline (no LLM)

**What it does:** from that board, predict `y = 1` iff the driver pits on **lap + 1**. Pure ML over timing / circuit / SC / compound features.

```bash
# one-time train (writes models.pkl locally; gitignored)
race-engineer/.venv/bin/python race-engineer/scripts/train_pit_baseline.py

# same Hamilton board as Phase 0
race-engineer/.venv/bin/python race-engineer/scripts/score_pit_sample.py --year 2024 --name-contains British --driver-id 1 --lap 22
```

```text
(same pit-wall view as Phase 0)

{
  "label": "pit_on_lap_plus_1",
  "y_true": 0,
  "model": "hgb",
  "score": 0.8321,
  "threshold": 0.5561,
  "y_pred": 1,
  ...
}
```

**Intuition:** history says he **stayed** (`y_true=0`), but HGB still fires **pit** here (false positive). Mid-stint boards look “pit-ish” to a crude classifier — that is why held-out **F1≈0.164** even with **AUROC≈0.756**.

Held-out scoreboard: `artifacts/pit_baseline/metrics.json`.

## Phase 2 — crew chief (structured decision)

**What it does:** same pit/stay question, but return JSON `{action, tyre, push, rationale}`. Offline `heuristic` for CI; `openai` when keyed. LLM prompts **withhold race/year/driver** (circuit kept for pit-loss context).

```bash
race-engineer/.venv/bin/python race-engineer/scripts/decide_once.py --backend heuristic --year 2024 --name-contains British --driver-id 1 --lap 22
```

```text
(same pit-wall view as Phase 0)

backend: heuristic
{
  "action": "stay",
  "tyre": null,
  "push": "med",
  "rationale": "Current stint still viable; stay out."
}
```

**Intuition:** on this board the heuristic **stay** matches history; Phase 1’s HGB did not. Frozen eval (150 pts, anonymized prompt): **Sol F1=0.50**, heuristic **0.306**, Phase 1 HGB **0.651** (schema **100%**). Sol trails HGB here because it is zero-shot on a thin pit-wall snapshot, while HGB was trained on historical pit labels.

```bash
race-engineer/.venv/bin/python race-engineer/scripts/run_crew_chief_eval.py --backend heuristic --samples 150

# LLM (requires OPENAI_API_KEY)
export OPENAI_API_KEY=...
race-engineer/.venv/bin/python race-engineer/scripts/run_crew_chief_eval.py --backend openai --samples 150
```

Committed scoreboard: `artifacts/crew_chief/metrics.json` (heuristic). Predictions CSV is gitignored.

## Phase 3 — tool calling

**What it does:** before deciding, the agent must call bound tools (`get_gaps`, `get_stint_age`, `get_remaining_laps`, `lookup_circuit_undercut_stats`). Race/driver IDs stay server-side. Rationale numbers are checked against tool returns (**faithfulness**).

```bash
race-engineer/.venv/bin/python race-engineer/scripts/decide_once_tools.py --backend heuristic_tools --year 2024 --name-contains British --driver-id 1 --lap 22
```

```text
(same pit-wall view as Phase 0)

backend: heuristic_tools
{
  "action": "stay",
  "tyre": null,
  "push": "med",
  "rationale": "Current stint still viable; stay out (stint_age_laps=22; remaining_laps=30; gap_ahead_s=1.001; ...).",
  "tools_used": ["get_gaps", "get_stint_age", "get_remaining_laps", "lookup_circuit_undercut_stats"],
  "faithfulness": {"faithfulness": 1.0, ...}
}
```

**Intuition:** same pit call as Phase 2 heuristic, but every cited number comes from tools. Frozen eval (150 pts): **faithfulness 100%**, **F1=0.306** (heuristic_tools); Phase 1 HGB **0.651**. Tool use alone does not raise F1 until the LLM/policy improves — it stops invented numbers.

```bash
race-engineer/.venv/bin/python race-engineer/scripts/run_tools_eval.py --backend heuristic_tools --samples 150

# LLM tools (requires OPENAI_API_KEY)
export OPENAI_API_KEY=...
race-engineer/.venv/bin/python race-engineer/scripts/run_tools_eval.py --backend openai_tools --samples 150
```

Committed scoreboard: `artifacts/tools/metrics.json` (heuristic_tools).

## Phase 4 — lap degradation

**What it does:** predict the driver's **next lap time** from stint age, compound, circuit, and recent pace. Exposed as tool `predict_lap_time` (used by Phase 3 tool belt).

```bash
race-engineer/.venv/bin/python race-engineer/scripts/train_lap_deg.py
race-engineer/.venv/bin/python race-engineer/scripts/predict_lap_deg.py --year 2024 --name-contains British --driver-id 1 --lap 22
```

```text
(same pit-wall view as Phase 0)

{
  "predicted_next_lap_ms": 91363.0,
  "predicted_next_lap_s": 91.363,
  "actual_next_lap_ms": 91107,
  "error_ms": 256.0,
  ...
}
```

**Intuition:** at lap 22 on mediums, the model expects ~~91.4s next; actual was 91.1s (~~0.26s error here). Hold-out **2024–25 test: MAE≈1.79s, MAPE≈1.77%**.

Committed scoreboard: `artifacts/lap_deg/metrics.json`.

## Phase 5 — simulation (option cards)

**What it does:** roll the race forward under pit-next vs stay-N options (lap deg + pit loss + pace noise). Returns Monte Carlo cards with mean finish position and `P(finish ≤ 3/5/10)`. Exposed as tool `simulate_strategies`.

When `pit_stops so far = 0` on a dry compound, `stay_to_finish` **is omitted** from the option menu (mandatory pit still owed).

```bash
race-engineer/.venv/bin/python race-engineer/scripts/simulate_once.py --year 2024 --name-contains British --driver-id 1 --lap 22
race-engineer/.venv/bin/python race-engineer/scripts/eval_sim.py --samples 80
```

Committed scoreboard: `artifacts/sim/metrics.json` — **Brier P(finish≤3/5/10)≈0.09/0.17/0.12**, mean |E[pos]−actual|≈**2.1** (80 pts; static-rival v0).

## Phase 6 — reasons over sims

**What it does:** call `simulate_strategies`, pick an option card, map to pit/stay for the **next lap**, and cite sim numbers. Offline `heuristic_sim` follows the oracle card; `pit_next_sim` always boxes next as a contrast baseline.

**Dry-race sporting (v0):** shared rules in `racing_rules.py` — appended to Phases 2–6 LLM prompts; sim drops illegal no-pit plans and switches compound on pit. Example: British L22 with 0 stops → oracle moves from illegal `stay_to_finish` to `stay_8_then_pit` (still **stay** next lap, but plan includes a stop).

```bash
race-engineer/.venv/bin/python race-engineer/scripts/decide_once_sim.py --backend heuristic_sim --year 2024 --name-contains British --driver-id 1 --lap 22
race-engineer/.venv/bin/python race-engineer/scripts/eval_sim_agent.py --backend heuristic_sim --samples 80
```

Committed scoreboard: `artifacts/sim_agent/metrics.json` — **mean regret=0**, oracle match **100%**, faith **≈0.994** (80 pts).

## Phase 7 — memory across laps

**What it does:** persist pit-wall instructions keyed by `(race, driver)` each lap; inject prior plan into the Phase 6 user prompt; support full-race replay (not just single-lap eval samples). Metrics: flip-flop rate (pit/stay reversal without material board/sim change) and regret delta vs memory-off.

```bash
race-engineer/.venv/bin/python race-engineer/scripts/replay_race.py --memory --backend heuristic_sim --year 2024 --name-contains British --driver-id 1
race-engineer/.venv/bin/python race-engineer/scripts/eval_memory.py --backend heuristic_sim
```

Committed scoreboard: `artifacts/memory/metrics.json` — **regret delta=0**, flip-flop rate **0** (20 race-drivers, 1087 laps; heuristic_sim).

## Phase 8 — evaluation harness 1.0.0

**Eval contract:** historical board + advisory engineer (+ pit-wall memory when enabled). The harness does **not** rewrite lap times or gaps if the driver ignores a box call; Monte Carlo sim cards are the only forward model.

**What it does:** full-race replay on the frozen set (10 races × winner + midfield). Models are split into two families with **separate leaderboards** so timing metrics are comparable:


| Family       | Models                                  | Timing MAE                                                                            |
| ------------ | --------------------------------------- | ------------------------------------------------------------------------------------- |
| **Pit-next** | `hgb`, `heuristic_crew`, `pit_next_sim` | \|first box-next-lap stop − actual pit\| per stint                                      |
| **Sim-plan** | `heuristic_sim`, `openai_sim`           | \|planned stop from lap before actual pit − actual pit\| (uses `stay_N_then_pit` cards) |


**Shared:** `next_lap_match` (pit/stay for lap+1 vs history), `advisory_mismatch_rate` (advised box but driver stayed out). **Sim-only:** mean regret, flip-flop rate (+ clean subsample excluding unexecuted box pairs). **Memory:** on by default for sim-plan models and `pit_next_sim`; off for HGB and heuristic crew (frozen per-lap).

```bash
race-engineer/.venv/bin/python race-engineer/scripts/train_pit_baseline.py   # once, for HGB
race-engineer/.venv/bin/python race-engineer/scripts/run_harness.py \
  --models hgb,heuristic_sim,heuristic_crew,pit_next_sim --tolerance 2
```

Outputs: `artifacts/eval/metrics.json` (`leaderboard_pit_next`, `leaderboard_sim_plan`, `metric_definitions`), `report.md`, `harness_manifest.json` (schema **eval_harness_v1**).