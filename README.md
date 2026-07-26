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
python3 race-engineer/scripts/update_dataset.py
```

Refresh tyre compounds (separate FastF1 ingest; keeps local cache):

```bash
race-engineer/.venv/bin/python race-engineer/scripts/build_tyre_laps.py --years 2018 2019 2020 2021 2022 2023 2024 2025
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
| **6 — Reasons Over Sims** | **done** | Choose among option cards + cite sim numbers | Position regret vs oracle | **regret=0; oracle match 100%; faith≈99%; pit-next baseline regret≈5.5** (80 pts) |
| 7 — Evaluation Harness | next | Historical replay reports | Decision match + pit-lap MAE | — |

### Phase 0 — how to run

Worked example (Hamilton, British GP 2024, lap 22) in [`race-engineer/README.md`](race-engineer/README.md).

```bash
python3 race-engineer/scripts/print_sample_state.py --year 2024 --name-contains British --driver-id 1 --lap 22
python3 race-engineer/scripts/run_integrity.py --samples 2000
```

### Phase 1 — how to run

Same board → pit-next-lap score (see package README for sample output).

```bash
# one-time: python3 -m venv race-engineer/.venv && race-engineer/.venv/bin/pip install -r race-engineer/requirements.txt
race-engineer/.venv/bin/python race-engineer/scripts/train_pit_baseline.py
race-engineer/.venv/bin/python race-engineer/scripts/score_pit_sample.py --year 2024 --name-contains British --driver-id 1 --lap 22
race-engineer/.venv/bin/python race-engineer/scripts/eval_pit_baseline.py
```

### Phase 2 — how to run

Same board → structured pit/stay JSON (see package README).

```bash
# offline / CI (rule-based stand-in)
race-engineer/.venv/bin/python race-engineer/scripts/decide_once.py --backend heuristic --year 2024 --name-contains British --driver-id 1 --lap 22
race-engineer/.venv/bin/python race-engineer/scripts/run_crew_chief_eval.py --backend heuristic --samples 150

# LLM (requires OPENAI_API_KEY)
export OPENAI_API_KEY=...
race-engineer/.venv/bin/python race-engineer/scripts/run_crew_chief_eval.py --backend openai --samples 150

# tests
race-engineer/.venv/bin/python -m unittest discover -s race-engineer/tests -v
```

### Phase 3 — how to run

Same board → tools then decision (see package README).

```bash
race-engineer/.venv/bin/python race-engineer/scripts/decide_once_tools.py --backend heuristic_tools --year 2024 --name-contains British --driver-id 1 --lap 22
race-engineer/.venv/bin/python race-engineer/scripts/run_tools_eval.py --backend heuristic_tools --samples 150

# LLM tools (requires OPENAI_API_KEY)
export OPENAI_API_KEY=...
race-engineer/.venv/bin/python race-engineer/scripts/run_tools_eval.py --backend openai_tools --samples 150
```

### Phase 4 — how to run

Next-lap pace model (Hamilton board) in [`race-engineer/README.md`](race-engineer/README.md).

```bash
race-engineer/.venv/bin/python race-engineer/scripts/train_lap_deg.py
race-engineer/.venv/bin/python race-engineer/scripts/predict_lap_deg.py --year 2024 --name-contains British --driver-id 1 --lap 22
```

### Phase 5 — how to run

Monte Carlo strategy cards (same Hamilton board):

```bash
race-engineer/.venv/bin/python race-engineer/scripts/simulate_once.py --year 2024 --name-contains British --driver-id 1 --lap 22
race-engineer/.venv/bin/python race-engineer/scripts/eval_sim.py --samples 80
```

### Phase 6 — how to run

Sim-conditioned pit/stay (pick an option card, cite sim numbers):

```bash
race-engineer/.venv/bin/python race-engineer/scripts/decide_once_sim.py --backend heuristic_sim --year 2024 --name-contains British --driver-id 1 --lap 22
race-engineer/.venv/bin/python race-engineer/scripts/eval_sim_agent.py --backend heuristic_sim --samples 80
```

Metrics: [`pit_baseline/metrics.json`](race-engineer/artifacts/pit_baseline/metrics.json), [`crew_chief/metrics.json`](race-engineer/artifacts/crew_chief/metrics.json), [`tools/metrics.json`](race-engineer/artifacts/tools/metrics.json), [`lap_deg/metrics.json`](race-engineer/artifacts/lap_deg/metrics.json), [`sim/metrics.json`](race-engineer/artifacts/sim/metrics.json), [`sim_agent/metrics.json`](race-engineer/artifacts/sim_agent/metrics.json).

Code: [`race-engineer/`](race-engineer/).

## Legacy notebooks

- [`lap-times/`](lap-times/) — lap-time ML experiments  
- [`radio/`](radio/) — radio message sentiment  

These remain independent of the race-engineer package.
