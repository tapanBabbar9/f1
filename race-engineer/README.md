# race-engineer

AI Race Engineer package.

```text
race-engineer/
  race_engineer/
    state.py         # RaceState dataclass + pit-wall view
    replay.py        # RaceReplay loader (Ergast CSVs)
    integrity.py     # Phase 0 error metric
    pit_baseline.py  # Phase 1 pit-next-lap classifier
  scripts/
    print_sample_state.py
    run_integrity.py
    update_dataset.py
    train_pit_baseline.py
    eval_pit_baseline.py
  artifacts/pit_baseline/
    metrics.json     # committed baseline scores
  tests/
  requirements.txt   # scikit-learn (Phase 1+)
```

Phase 0 is stdlib-only. Phase 1 needs the venv:

```bash
python3 -m venv race-engineer/.venv
race-engineer/.venv/bin/pip install -r race-engineer/requirements.txt
```

## Phase 0 example

Hamilton, British GP 2024, lap 22 (run from repo root):

```bash
python3 race-engineer/scripts/print_sample_state.py --year 2024 --name-contains British --driver-id 1 --lap 22
```

```text
Race: British Grand Prix (2024)
Driver: HAM
Lap: 22 / 52
Current Position: P3
Gap Ahead: +1.001s
Gap Behind: -1.688s
Tyres:
  Compound: unknown
  Age: 22 laps
Pit this lap: No
Pit stops so far: 0
Last lap: 91095 ms
Cars on track: 19

--- raw ---
race_id: 1132
year: 2024
race_name: British Grand Prix
circuit_id: 9
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
```

## Phase 1 — pit baseline

Label: pit on **next** lap (`lap + 1`). Split: train 2018–2022 / val 2023 / test 2024–2025.

```bash
race-engineer/.venv/bin/python race-engineer/scripts/train_pit_baseline.py
race-engineer/.venv/bin/python race-engineer/scripts/eval_pit_baseline.py
```

Primary model (**hgb**) on test: **F1=0.164**, **AUROC=0.756** (see `artifacts/pit_baseline/metrics.json`).

Feature set `timing+circuit+sc+compound`: timing + `circuit_id` + SC flags + FastF1 `compound_id` / `tyre_life` from `dataset/tyre_laps.csv` (2018–2025; FastF1 cache under `artifacts/fastf1_cache/`, not deleted on re-run).

```bash
race-engineer/.venv/bin/python race-engineer/scripts/build_tyre_laps.py --years 2018 2019 2020 2021 2022 2023 2024 2025
```
