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
| `tyre_laps` | Per-lap compound / tyre life (FastF1; Phase 1) |
| `*_standings`, `status` | Championships / status codes |
| `safety_cars`, `red_flags` | Additive safety metadata (optional) |

**Backward compatibility:** core filenames and columns match the legacy Kaggle/Ergast schema. New fields are append-only (e.g. `sprint_results.rank`). `raceId` values for historical races are stable.

Refresh from the latest public dump:

```bash
python3 race-engineer/scripts/update_dataset.py
```

## Live phases

| Phase | Status | Deliverable | Error metric | Latest |
|-------|--------|-------------|--------------|--------|
| **0 — Race Replay Loader** | **done** | `RaceReplay.get_state()` → `RaceState` | State integrity rate (target ≥ 99%) | **100%** on 2k samples (2021–2025) |
| **1 — Baseline Pit Classifier** | **done** | Pit/stay predictions on held-out races | Pit-lap F1 / AUROC | **HGB test F1=0.164, AUROC=0.756** |
| 2 — Crew Chief Agent | next | Structured strategy decisions | Schema validity + pit F1 | — |

### Phase 0 — how to run

```bash
python3 race-engineer/scripts/print_sample_state.py --year 2024 --name-contains "Abu Dhabi" --lap 34
python3 race-engineer/scripts/run_integrity.py --samples 2000
```

### Phase 1 — how to run

```bash
# one-time: python3 -m venv race-engineer/.venv && race-engineer/.venv/bin/pip install -r race-engineer/requirements.txt
# tyre compounds (FastF1 → dataset/tyre_laps.csv; cache kept under race-engineer/artifacts/fastf1_cache/)
race-engineer/.venv/bin/python race-engineer/scripts/build_tyre_laps.py --years 2018 2019 2020 2021 2022 2023 2024 2025
race-engineer/.venv/bin/python race-engineer/scripts/train_pit_baseline.py
race-engineer/.venv/bin/python race-engineer/scripts/eval_pit_baseline.py

# tests
race-engineer/.venv/bin/python -m unittest discover -s race-engineer/tests -v
```

Metrics: [`race-engineer/artifacts/pit_baseline/metrics.json`](race-engineer/artifacts/pit_baseline/metrics.json).

Code: [`race-engineer/`](race-engineer/).

## Legacy notebooks

- [`lap-times/`](lap-times/) — lap-time ML experiments  
- [`radio/`](radio/) — radio message sentiment  

These remain independent of the race-engineer package.
