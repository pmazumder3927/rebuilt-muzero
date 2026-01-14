# Calibration: `massive_results.json`

This repo includes `massive_results.json` (≈10k rows) as a reference dataset for **one robot’s fuel-scoring performance**.

Each row is a single simulated/estimated outcome for a particular blue-robot parameter set:

- Inputs: `capacity`, `shooting_rate`, `intake_rate`, `accuracy`, `max_speed`, `acceleration`, `align_time`, `dump_time`, `cycle_variance`, `defense_penalty`, plus turret/feature flags.
- Outputs: `fuel`, `blue_pts`, `red_pts`.

Key observations (from the dataset itself):

- `blue_pts == fuel` for all rows (fuel is the only point source represented here).
- `cycles` is always `0` (not usable for calibration directly).
- Median-ish robot parameters are very close to:
  - capacity ≈ 60
  - shooting_rate ≈ 20
  - intake_rate ≈ 8
  - accuracy ≈ 0.92
  - max_speed ≈ 15.5
  - acceleration ≈ 22
  - align_time ≈ 0.5
  - dump_time ≈ 0.3
  - cycle_variance ≈ 1
  - defense_penalty ≈ 0.15

## How to inspect the dataset

Run:

`python3 scripts/analyze_massive_results.py --path massive_results.json`

This prints:

- key list + row count
- numeric min/median/p90/max
- mean fuel by `turret_type`, `shoot_on_move`, `shoot_while_intake`
- a “median-ish” row you can use as an anchor spec

## How we use this for sim calibration

The macro-simulator will be extended to:

- accept these robot parameters in `RobotSpec` (including `max_speed`, `acceleration`, `align_time`, `dump_time`, feature flags)
- optionally run a small calibration harness that compares sim output (fuel scored) against `fuel` in the dataset for sampled rows

The goal is not to match every row exactly (this dataset has one sample per config and includes stochasticity),
but to set *reasonable default scalings* and ensure the sim’s throughput sensitivity matches reality trends.

