# MuZero (Multi-Agent Self-Play) — REBUILT Macro Strategy

This repo includes a **two-player MuZero** implementation on top of the macro-simulator:

- Each side (RED vs BLUE) controls **3 robots**.
- Actions are **macro intents** (collect / score / deliver / defend / climb), not joystick control.
- Training is **self-play** using MuZero-style **PUCT MCTS**.

## How it’s modeled

### Turn structure

We use a practical **turn-based wrapper** around the simultaneous macro-sim:

- The underlying simulator (`RebuiltMacroSim`) still advances in **1s** macro steps.
- For MuZero, we alternate control: RED chooses actions for its 3 robots on one step, then BLUE on the next.

This keeps the game in the standard “two-player alternating moves” form MuZero expects.

### Multi-robot actions

Each MuZero action is a **joint action** for the current alliance’s 3 robots:

- Per-robot action space size = `action_space_size(n_neutral_bins)` (default: 20)
- Joint action space size = `per_robot_actions^3` (default: `20^3 = 8000`)

MCTS does not expand all 8000 actions. Instead it takes the network’s policy logits and restricts to the **top-K**
(`MuZeroConfig.max_policy_actions`, default: 64).

## Quick start

Run a small training job on Apple Silicon using Metal (MPS):

`python3 scripts/train_muzero.py --device mps --iterations 50`

For faster iteration during development:

`python3 scripts/train_muzero.py --device mps --preset fast --iterations 50`

For stronger (slower) search:

`python3 scripts/train_muzero.py --device mps --preset full --iterations 50`

Useful knobs:

- `--num-sims 32` (MCTS simulations per move; higher = stronger but slower)
- `--games-per-iter 5`
- `--train-steps-per-iter 200`
- `--min-replay-games 20` (warmup threshold; set lower for smoke tests)
- `--eval-games 5` (optional “no-noise, greedy” evaluation)
  - `--plot-every 1` (write `.tmp/muzero/metrics.png` more/less often)

Checkpoints are written to:

- `.tmp/muzero/latest.pt`

Metrics are written to:

- `.tmp/muzero/metrics.csv`
- `.tmp/muzero/metrics.png` (loss + selfplay/eval score diff + win-rate/entropy)

## Where the code lives

- Turn-based game wrapper: `rebuilt_muzero/muzero/game.py`
- Observation encoder (canonical POV): `rebuilt_muzero/muzero/obs_encoder.py`
- Joint action encoding (3 robots): `rebuilt_muzero/muzero/joint_action.py`
- Network (h/g/f heads): `rebuilt_muzero/muzero/networks.py`
- PUCT MCTS: `rebuilt_muzero/muzero/mcts.py`
- Replay buffer + targets: `rebuilt_muzero/muzero/replay.py`
- Training step: `rebuilt_muzero/muzero/train.py`
- Training CLI: `scripts/train_muzero.py`
