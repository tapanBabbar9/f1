# race-engineer

AI Race Engineer package.

```text
race-engineer/
  race_engineer/
    state.py / replay.py / integrity.py
    pit_baseline.py      # Phase 1
    crew_chief.py        # Phase 2 schema + prompts
    llm.py               # openai + heuristic backends
    crew_chief_eval.py
  scripts/
    print_sample_state.py / run_integrity.py / update_dataset.py
    build_tyre_laps.py / train_pit_baseline.py / eval_pit_baseline.py
    decide_once.py / run_crew_chief_eval.py
  artifacts/
    pit_baseline/metrics.json
    crew_chief/metrics.json
    eval/frozen_races.json
  requirements.txt
```

```bash
python3 -m venv race-engineer/.venv
race-engineer/.venv/bin/pip install -r race-engineer/requirements.txt
```

## Phase 0 example

```bash
python3 race-engineer/scripts/print_sample_state.py --year 2024 --name-contains British --driver-id 1 --lap 22
```

## Phase 1 — pit baseline

```bash
race-engineer/.venv/bin/python race-engineer/scripts/train_pit_baseline.py
```

Primary **hgb** test: **F1=0.164**, **AUROC=0.756**.

## Phase 2 — crew chief

```bash
race-engineer/.venv/bin/python race-engineer/scripts/decide_once.py --backend heuristic --year 2024 --name-contains British --driver-id 1 --lap 22
race-engineer/.venv/bin/python race-engineer/scripts/run_crew_chief_eval.py --backend heuristic --samples 150

# LLM
export OPENAI_API_KEY=...
race-engineer/.venv/bin/python race-engineer/scripts/run_crew_chief_eval.py --backend openai --samples 150
```

Frozen eval (150 pts): heuristic **schema=100%**, **F1=0.306**; Phase 1 HGB on same points **F1=0.651**.
Committed scoreboard: `artifacts/crew_chief/metrics.json` (heuristic). Predictions CSV is gitignored.
