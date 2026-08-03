# F1 Lab

Historical Formula 1 data plus an **AI Race Engineer** built in small, measurable phases.

## Dataset

Ergast-compatible CSVs under [`dataset/`](dataset/) (1950–present).

| Table | Role |
|-------|------|
| `races`, `circuits`, `seasons` | Calendar |
| `drivers`, `constructors` | Participants |
| `results`, `sprint_results`, `qualifying` | Session outcomes |
| `lap_times`, `pit_stops` | Timing (race engineer core) |
| `tyre_laps` | Per-lap compound / tyre life (FastF1) |
| `*_standings`, `status` | Championships / status codes |
| `safety_cars`, `red_flags` | Full SC / red-flag periods |
| `virtual_safety_car_estimates.json` | VSC lap estimates (additive) |

**Backward compatibility:** core filenames and columns match the legacy Kaggle/Ergast schema. New fields are append-only (e.g. `sprint_results.rank`). `raceId` values for historical races are stable.

Refresh Ergast-style tables:

```bash
python3 f1-pitwall/scripts/update_dataset.py
```

Refresh tyre compounds (separate FastF1 ingest; keeps local cache):

```bash
f1-pitwall/.venv/bin/python f1-pitwall/scripts/build_tyre_laps.py --years 2018 2019 2020 2021 2022 2023 2024 2025
```

## Live phases

| Phase | Status | Deliverable | Error metric | Latest |
|-------|--------|-------------|--------------|--------|
| **0 — Race Replay Loader** | **done** | `RaceReplay.get_state()` → `RaceState` | State integrity rate (target ≥ 99%) | **100%** on 2k samples (2021–2025) |
| **1 — Baseline Pit Classifier** | **done** | Pit/stay predictions on held-out races | Pit-lap F1 / AUROC | **HGB test F1=0.164, AUROC=0.756** |
| **2 — Crew Chief Agent** | **done** | Structured strategy JSON + rationale | Schema validity + pit F1 | **schema 100%; Sol F1=0.50, heuristic 0.306, P1 0.651** (same 150 pts; Sol trails HGB — zero-shot on a sparse board vs a model trained on pit labels) |
| **3 — Tool Calling** | **done** | Gaps/stint/remaining/undercut tools before decide | Tool faithfulness + pit F1 | **faith 100%; heuristic_tools F1=0.306 vs P1 0.651** (same 150 pts) |
| **4 — Lap Degradation** | **done** | Next-lap pace tool (`predict_lap_time`) | MAE / MAPE | **test MAE≈1.79s, MAPE≈1.77%** (2024–25) |
| **5 — Simulation** | **done** | Monte Carlo pit-now vs stay-N option cards | Finish-position Brier | **Brier P(≤3/5/10)≈0.09/0.17/0.12; MAE≈2.1 pos** (80 pts) |
| **6 — Reasons Over Sims** | **done** | Choose among option cards + cite sim numbers; dry-race pit rules in prompt + sim | Position regret vs oracle | **regret=0; oracle match 100%; faith≈99%** (80 pts) |
| **7 — Memory Across Laps** | **done** | Pit-wall memory store + full-race replay wired into Phase 6 | Flip-flop rate + regret delta vs memory-off | **regret delta=0; flip-flop=0** (1087 laps, heuristic_sim) |
| **8 — Evaluation Harness** | **done** | Full-race model compare (pit-lap MAE ±2) | Pit-lap MAE + % within tolerance | **see `artifacts/eval/metrics.json`** |
| **9 — Multi-Agent Setup** | **done** | Strategy agent + Race Engineer radio post-pass | Action identity under radio swap | **radio cannot change card/memory** |
| 10 — Historian RAG | next | Similar-stint retrieval tool | Relevance@5 + regret delta | — |

### Phase 0 — how to run

Worked example (Hamilton, British GP 2024, lap 22) in [`f1-pitwall/README.md`](f1-pitwall/README.md).

```bash
python3 f1-pitwall/scripts/print_sample_state.py --year 2024 --name-contains British --driver-id 1 --lap 22
python3 f1-pitwall/scripts/run_integrity.py --samples 2000
```

### Phase 1 — how to run

Same board → pit-next-lap score (see package README for sample output).

```bash
# one-time: python3 -m venv f1-pitwall/.venv && f1-pitwall/.venv/bin/pip install -r f1-pitwall/requirements.txt
f1-pitwall/.venv/bin/python f1-pitwall/scripts/train_pit_baseline.py
f1-pitwall/.venv/bin/python f1-pitwall/scripts/score_pit_sample.py --year 2024 --name-contains British --driver-id 1 --lap 22
f1-pitwall/.venv/bin/python f1-pitwall/scripts/eval_pit_baseline.py
```

### Phase 2 — how to run

Same board → structured pit/stay JSON (see package README).

```bash
# offline / CI (rule-based stand-in)
f1-pitwall/.venv/bin/python f1-pitwall/scripts/decide_once.py --backend heuristic --year 2024 --name-contains British --driver-id 1 --lap 22
f1-pitwall/.venv/bin/python f1-pitwall/scripts/run_crew_chief_eval.py --backend heuristic --samples 150

# LLM (requires OPENAI_API_KEY)
export OPENAI_API_KEY=...
f1-pitwall/.venv/bin/python f1-pitwall/scripts/run_crew_chief_eval.py --backend openai --samples 150

# tests
f1-pitwall/.venv/bin/python -m unittest discover -s f1-pitwall/tests -v
```

### Phase 3 — how to run

Same board → tools then decision (see package README).

```bash
f1-pitwall/.venv/bin/python f1-pitwall/scripts/decide_once_tools.py --backend heuristic_tools --year 2024 --name-contains British --driver-id 1 --lap 22
f1-pitwall/.venv/bin/python f1-pitwall/scripts/run_tools_eval.py --backend heuristic_tools --samples 150

# LLM tools (requires OPENAI_API_KEY)
export OPENAI_API_KEY=...
f1-pitwall/.venv/bin/python f1-pitwall/scripts/run_tools_eval.py --backend openai_tools --samples 150
```

### Phase 4 — how to run

Next-lap pace model (Hamilton board) in [`f1-pitwall/README.md`](f1-pitwall/README.md).

```bash
f1-pitwall/.venv/bin/python f1-pitwall/scripts/train_lap_deg.py
f1-pitwall/.venv/bin/python f1-pitwall/scripts/predict_lap_deg.py --year 2024 --name-contains British --driver-id 1 --lap 22
```

### Phase 5 — how to run

Monte Carlo strategy cards (same Hamilton board):

```bash
f1-pitwall/.venv/bin/python f1-pitwall/scripts/simulate_once.py --year 2024 --name-contains British --driver-id 1 --lap 22
f1-pitwall/.venv/bin/python f1-pitwall/scripts/eval_sim.py --samples 80
```

### Phase 6 — how to run

Sim-conditioned pit/stay (pick an option card, cite sim numbers). Dry-race
mandatory pit: `stay_to_finish` dropped from sim when `pit_stops=0`; prompts
include the same rule (`race_engineer/racing_rules.py`).

```bash
f1-pitwall/.venv/bin/python f1-pitwall/scripts/decide_once_sim.py --backend heuristic_sim --year 2024 --name-contains British --driver-id 1 --lap 22
f1-pitwall/.venv/bin/python f1-pitwall/scripts/eval_sim_agent.py --backend heuristic_sim --samples 80
```

### Phase 7 — how to run

Full-race lap-by-lap replay with pit-wall memory surfaced in the Phase 6 prompt:

```bash
f1-pitwall/.venv/bin/python f1-pitwall/scripts/replay_race.py --memory --backend heuristic_sim --year 2024 --name-contains British --driver-id 1
f1-pitwall/.venv/bin/python f1-pitwall/scripts/eval_memory.py --backend heuristic_sim
```

### Phase 8 — how to run

Full-race harness on the frozen set with **two leaderboards** (pit-next vs sim-plan timing). **Contract:** historical board + advisory engineer — no counterfactual lap physics if the driver ignores a box call. See [`f1-pitwall/README.md`](f1-pitwall/README.md).

```bash
f1-pitwall/.venv/bin/python f1-pitwall/scripts/run_harness.py \
  --models hgb,heuristic_sim,heuristic_crew,pit_next_sim --tolerance 2
```

### Phase 9 — how to run

Strategy (sim card choice + memory) then Race Engineer radio (`driver_message` only).
Radio cannot override action or write memory. Default radio is deterministic
(`RACE_ENGINEER_RADIO=heuristic`); use `openai` to iterate radio independently.

```bash
f1-pitwall/.venv/bin/python f1-pitwall/scripts/decide_once_sim.py \
  --backend heuristic_sim --radio heuristic --year 2024 --name-contains British --driver-id 1 --lap 22
```

Metrics: [`pit_baseline/metrics.json`](f1-pitwall/artifacts/pit_baseline/metrics.json), [`crew_chief/metrics.json`](f1-pitwall/artifacts/crew_chief/metrics.json), [`tools/metrics.json`](f1-pitwall/artifacts/tools/metrics.json), [`lap_deg/metrics.json`](f1-pitwall/artifacts/lap_deg/metrics.json), [`sim/metrics.json`](f1-pitwall/artifacts/sim/metrics.json), [`sim_agent/metrics.json`](f1-pitwall/artifacts/sim_agent/metrics.json), [`memory/metrics.json`](f1-pitwall/artifacts/memory/metrics.json), [`eval/metrics.json`](f1-pitwall/artifacts/eval/metrics.json).

Code: [`f1-pitwall/`](f1-pitwall/).

## Legacy notebooks

- [`lap-times/`](lap-times/) — lap-time ML experiments  
- [`radio/`](radio/) — radio message sentiment  

These remain independent of the f1-pitwall project.
