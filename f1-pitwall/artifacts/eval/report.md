# Eval harness 1.0.0

Frozen set `frozen_v1` · tolerance ±2 laps · memory (sim)=True

## Pit-next models (HGB, crew chief, pit-next sim)

Timing MAE = |first box-next-lap stop − actual pit| per stint; within ±2 laps.

| Model | Timing MAE | Within ±tol | Next-lap match | Stints scored |
|-------|------------|-------------|----------------|---------------|
| hgb | 6.857 | 35.7% | 0.8266 | 28/34 |
| heuristic_crew | 11.188 | 25.0% | 0.7792 | 16/34 |

## Sim-plan models (option cards + memory)

Timing MAE = |planned stop (from lap before actual pit) − actual pit| per stint; within ±2 laps. Uses stay_N_then_pit / pit_next card labels.

| Model | Timing MAE | Within ±tol | Next-lap match | Mean regret | Flip-flop |
|-------|------------|-------------|----------------|-------------|-----------|
| heuristic_sim | 8.906 | 18.8% | 0.8464 | 0.0 | 0.0 |

### Shared definitions

- **Next-lap match:** Fraction of laps where pit/stay for lap+1 matches history.
- **Mean regret:** Sim only: E[finish|chosen] − E[finish|oracle] on option cards.
- **Flip-flop:** Sim only: pit/stay reversals without evidence change (memory on).
