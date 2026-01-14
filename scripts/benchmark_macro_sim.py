from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rebuilt_muzero.sim import RebuiltMacroSim, default_config
from rebuilt_muzero.sim.actions import action_space_size


def total_fuel(env: RebuiltMacroSim) -> int:
    state = env.state
    assert state is not None
    return int(
        np.sum(state.neutral_fuel)
        + np.sum(state.depot_fuel)
        + np.sum(state.outpost_chute)
        + np.sum(state.outpost_corral)
        + np.sum(state.robot_carried)
        + np.sum(state.robot_task_reserved_fuel)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matches", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neutral-bins", type=int, default=8)
    parser.add_argument("--check-fuel", action="store_true")
    parser.add_argument("--with-obs", action="store_true", help="Benchmark env.step() (slower) instead of env.step_fast().")
    args = parser.parse_args()

    cfg = default_config(n_neutral_bins=args.neutral_bins)
    env = RebuiltMacroSim(cfg, seed=args.seed)

    rng = np.random.default_rng(args.seed)
    n_actions = action_space_size(n_neutral_bins=cfg.n_neutral_bins)

    steps = 0
    score_diff = 0.0

    t0 = time.perf_counter()
    for m in range(args.matches):
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        if args.check_fuel:
            start_fuel = total_fuel(env)

        while True:
            actions = rng.integers(0, n_actions, size=6, dtype=np.int32)
            if args.with_obs:
                terminated = env.step(actions).terminated
            else:
                _, terminated = env.step_fast(actions)
            steps += 1
            if args.check_fuel and total_fuel(env) != start_fuel:
                raise AssertionError("Fuel conservation violated (include reserved fuel in the total).")
            if terminated:
                state = env.state
                assert state is not None
                total = state.score + state.penalty_points
                score_diff += float(total[0] - total[1])
                break

    dt = time.perf_counter() - t0
    print(f"matches: {args.matches}  steps: {steps}  seconds: {dt:.3f}")
    print(f"matches/s: {args.matches / dt:.1f}  steps/s: {steps / dt:.1f}")
    print(f"mean final score diff (red-blue): {score_diff / args.matches:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
