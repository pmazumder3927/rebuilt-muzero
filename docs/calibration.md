# Calibration

`data/massive_results.json` (≈10k rows) is a reference dataset for one robot's
fuel-scoring performance, generated from a separate physics-aware cycle-time
study. We use it to calibrate the macro simulator's robot-throughput
sensitivity — not to match every row exactly, but to make sure the sim
responds to robot parameters in the right direction with reasonable
magnitudes.

## Dataset shape

Each row is a single trial for a blue-robot parameter set:

- Inputs: `capacity`, `shooting_rate`, `intake_rate`, `accuracy`, `max_speed`,
  `acceleration`, `align_time`, `dump_time`, `cycle_variance`,
  `defense_penalty`, plus turret/feature flags.
- Outputs: `fuel`, `blue_pts`, `red_pts`.

Notable observations from the dataset itself:

- `blue_pts == fuel` for all rows (fuel is the only point source represented).
- `cycles` is always 0 (not usable directly).
- Median-ish robot parameters cluster near:
  capacity ≈ 60 · shooting_rate ≈ 20 · intake_rate ≈ 8 · accuracy ≈ 0.92 ·
  max_speed ≈ 15.5 · acceleration ≈ 22 · align_time ≈ 0.5 · dump_time ≈ 0.3 ·
  cycle_variance ≈ 1 · defense_penalty ≈ 0.15

## Inspecting the dataset

```bash
python scripts/analyze_massive_results.py
```

Prints column list + row count, numeric min/median/p90/max, mean fuel by
`turret_type` / `shoot_on_move` / `shoot_while_intake`, and a representative
"median-ish" row you can use as an anchor spec.

## Calibration harness

A rough "does the sim respond to robot parameters the right way?" check:

```bash
python scripts/calibrate_against_massive_results.py \
  --samples 200 --episodes-per-sample 3 --hub-mode rebuilt
```

You can tune the effective travel scale used for the comparison:

```bash
python scripts/calibrate_against_massive_results.py --distance-scale 2.0
```

Outputs MAE, RMSE, correlation between sim-predicted fuel scored and the
dataset's `fuel` column, plus the 10 worst-fit rows.
