# MuZero training

A two-player MuZero implementation on top of the REBUILT macro simulator.

- Each alliance (RED vs BLUE) controls 3 robots.
- Actions are macro intents (collect / score / deliver / defend / climb), not
  joystick control.
- Training is self-play with PUCT MCTS.

## Modelling

### Turn structure

The underlying simulator advances in 1 s macro steps with all 6 robots acting
simultaneously. For MuZero we wrap it as a standard two-player alternating
game: RED issues actions for its 3 robots on one step, then BLUE on the next.
The non-acting alliance's robots simply continue any in-flight tasks (IDLE
otherwise).

### Joint actions

Each MuZero action is a *joint action* over the current alliance's 3 robots:

- Per-robot action space size = `action_space_size(n_neutral_bins)` (default 20)
- Joint action space size = `per_robot ** 3` (default 8000)

MCTS never expands the full joint space. The network's policy logits are
restricted to the top-K joint actions (`MuZeroConfig.max_policy_actions`,
default 64), which is also what we store as the sparse policy target.

## Quick start

Smoke test on CPU:

```bash
python scripts/train_muzero.py --preset fast --iterations 10 --min-replay-games 4
```

Apple Silicon (MPS):

```bash
python scripts/train_muzero.py --device mps --iterations 50
```

Stronger search:

```bash
python scripts/train_muzero.py --device mps --preset full --iterations 50
```

Useful knobs:

| Flag | Effect |
| --- | --- |
| `--num-sims 32` | MCTS simulations per move |
| `--games-per-iter 5` | Self-play games per training iteration |
| `--train-steps-per-iter 200` | Optimizer steps per iteration |
| `--min-replay-games 20` | Warmup threshold before training |
| `--eval-games 5` | Greedy evaluation games per iteration |
| `--plot-every 1` | Frequency for writing `metrics.png` |
| `--set num_simulations=64` | Override any `MuZeroConfig` field |

Outputs land under `.tmp/muzero/`:

- `latest.pt` — most recent checkpoint
- `it_XXXX.pt` — periodic checkpoints (every 10 iterations)
- `metrics.csv` — per-iteration metrics
- `metrics.png` — loss, score diff, win rate, entropy, timing

## Code layout

| Concern | Module |
| --- | --- |
| Turn-based game wrapper | `rebuilt_muzero/muzero/game.py` |
| Canonical POV obs encoder | `rebuilt_muzero/muzero/obs_encoder.py` |
| Joint action encoding | `rebuilt_muzero/muzero/joint_action.py` |
| h/g/f networks | `rebuilt_muzero/muzero/networks.py` |
| PUCT MCTS (batched) | `rebuilt_muzero/muzero/mcts.py` |
| Replay buffer + targets | `rebuilt_muzero/muzero/replay.py` |
| Training step | `rebuilt_muzero/muzero/train.py` |
| CLI entry point | `scripts/train_muzero.py` |
