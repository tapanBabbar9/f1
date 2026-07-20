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
| 1 — Baseline Pit Classifier | next | Pit/stay predictions on held-out races | Pit-lap F1 / AUROC | — |

### Phase 0 — how to run

```bash
# Sample pit-wall snapshot
python3 race-engineer/scripts/print_sample_state.py --year 2024 --name-contains "Abu Dhabi" --lap 34

# Integrity metric (must be ≥ 99%)
python3 race-engineer/scripts/run_integrity.py --samples 2000

# Unit tests
python3 -m unittest discover -s race-engineer/tests -v
```

Code: [`race-engineer/`](race-engineer/).

## Legacy notebooks

- [`lap-times/`](lap-times/) — lap-time ML experiments  
- [`radio/`](radio/) — radio message sentiment  

These remain independent of the race-engineer package.
