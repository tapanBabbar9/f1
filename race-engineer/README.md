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
  scripts/
    print_sample_state.py / run_integrity.py / update_dataset.py
    build_tyre_laps.py / train_pit_baseline.py / eval_pit_baseline.py
    score_pit_sample.py / decide_once.py / run_crew_chief_eval.py
    decide_once_tools.py / run_tools_eval.py
  artifacts/
    pit_baseline/metrics.json
    crew_chief/metrics.json
    tools/metrics.json
    eval/frozen_races.json
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
