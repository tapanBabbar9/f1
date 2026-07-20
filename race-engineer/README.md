# race-engineer

AI Race Engineer package. **Phase 0** ships a seekable historical race state.

```text
race-engineer/
  race_engineer/
    state.py       # RaceState dataclass + pit-wall view
    replay.py      # RaceReplay loader (Ergast CSVs)
    integrity.py   # Phase 0 error metric
  scripts/
    print_sample_state.py
    run_integrity.py
    update_dataset.py
  tests/
    test_replay_integrity.py
```

Stdlib only for Phase 0 (no pandas required).

## Example

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
