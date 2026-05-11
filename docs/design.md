# Design: REBUILT macro decision coach

This document describes the modelling choices behind the macro simulator and the
MuZero training stack. It is the reference for *why* things are shaped the way
they are; the README has the user-facing summary.

## Problem framing

We want a **macro decision coach** for the FIRST Robotics 2026 game **REBUILT**.
The coach recommends a high-level intent every ~1s (not joystick control),
optimizing either:

- **Playoffs**: win probability / score margin
- **Qualifications**: ranking points + match points

To make MuZero tractable on commodity hardware, we model matches as a sequence
of **timed macros** instead of simulating physics or driver control.

## What is modelled (and what isn't)

The macro simulator captures the strategy-critical mechanics of REBUILT:

### Match phases + HUB activation windows

- **AUTO** 20s → **TRANSITION** 10s → **4 × Alliance Shifts** (25s each) →
  **ENDGAME** 30s.
- **Both HUBs active** during AUTO / TRANSITION / ENDGAME.
- During each **Alliance Shift** only one alliance's HUB is active. The first
  shift's active alliance is whichever scored more **FUEL in AUTO** (random
  if tied), then alternates.

### Scoring

- **FUEL scored in an active HUB**: +1 point each.
- **Inactive HUB**: 0 points (motivates steal / deny / stage / defend / climb).
- **Tower** scoring is level-based with different values in AUTO vs TELEOP/ENDGAME.

### Stochasticity

- Scored FUEL is redistributed to the neutral zone through one of four exits.
  This drives much of the staging-vs-cycling tension.

### Logistics

- OUTPOST chute stores up to ~25 FUEL behind a door; robots can deliver via
  OUTPOST CORRAL for later feed.

### Penalty model

- Scoring from outside the alliance zone: major foul.
- Endgame contact with an opponent on their tower: major foul + can award the
  opponent L3 tower points.
- Pinning longer than 3s: minor foul.

What we **do not** model: kinematics, controller dynamics, real defense
geometry, fuel piece physics. These are intentionally abstracted into macro
timing distributions.

## Drive-coach MDP

### Observation

- `t`, `phase` ∈ {AUTO, TRANSITION, SHIFT1..4, ENDGAME}, `active_hubs`, score,
  penalties
- coarse FUEL bins (`neutral_fuel[NB]`), depot fuel, outpost chute & corral
- per-robot: region, fuel carried, busy_until, climb readiness, capability
  tier features (speed/intake/accuracy/climb time)

### Actions

For a single robot:

`COLLECT_NEUTRAL(bin)` · `COLLECT_DEPOT` · `SCORE_HUB` · `DELIVER_OUTPOST`
`DEFEND_OPPONENT_HUB_LANE` · `DEFEND_OPPONENT_COLLECTOR`
`PREP_CLIMB(level)` · `CLIMB(level)` · `IDLE`

For the MuZero wrapper we factor a **joint action over 3 robots** of the
current alliance (size `per_robot ** 3`), and use top-K policy expansion to
keep MCTS tractable.

### Reward

- Playoffs: terminal `score_us - score_them`, shaped by incremental points.
- Quals: match points (W/T/L) + RP attainment; margin as secondary signal.
- Penalty shaping: negative reward for fouls; explicit risk shaping for
  endgame tower-contact defense.

## Simulator implementation

- Internal state in `rebuilt_muzero/sim/state.py`.
- Macro transitions in `rebuilt_muzero/sim/env.py`. Each robot schedules a
  timed task (travel + operation, e.g. intake or shoot), with delays from
  defense pressure. Time advances in 1s steps (`decision_interval_s`).
- Two step entry points:
  - `step_fast(actions)` returns `(reward, terminated)` — used in MuZero hot
    loops, no observation construction.
  - `step(actions)` returns a structured `StepResult` for benchmarks and the
    debug renderer.

## MuZero stack

We implement a practical two-player MuZero on top of the macro sim:

- **Turn-based wrapper** (`muzero/game.py`): alternates RED/BLUE control over
  the underlying simultaneous macro sim. The opponent's robots continue any
  in-flight tasks while the other alliance acts.
- **Canonical observation** (`muzero/obs_encoder.py`): swaps alliances and
  mirrors bins/regions so the network always sees the "current player" POV.
- **Joint action encoding** (`muzero/joint_action.py`): bijective mapping
  between (a0, a1, a2) per-robot triples and a single int.
- **Networks** (`muzero/networks.py`): `h` representation, `g` dynamics +
  reward, `f` policy + value, all MLPs over a 96-d latent by default.
- **MCTS** (`muzero/mcts.py`): PUCT search with batched recurrent inference.
  The network's policy logits are restricted to the top-K actions
  (`MuZeroConfig.max_policy_actions`) so we never expand the full ~8000-action
  joint space.
- **Replay & training** (`muzero/replay.py`, `muzero/train.py`): standard
  unrolled targets — Huber for value/reward, sparse cross-entropy for policy.

## Curriculum

Suggested progression (not enforced; you can flip flags in `GameConfig`):

1. No defense, deterministic redistribution
2. Stochastic redistribution
3. Defense delays + pin constraints
4. Endgame tower protection (major foul)
5. Outpost logistics + chute/corral limits
