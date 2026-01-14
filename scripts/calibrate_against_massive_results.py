from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rebuilt_muzero.sim import RebuiltMacroSim, RobotSpec, default_config  # noqa: E402
from rebuilt_muzero.sim.actions import action_space_size  # noqa: E402
from rebuilt_muzero.sim.state import Alliance, neutral_bin_region  # noqa: E402


def spec_from_row(row: dict) -> RobotSpec:
    return RobotSpec(
        fuel_capacity=int(row["capacity"]),
        intake_fuel_per_s=float(row["intake_rate"]),
        shoot_fuel_per_s=float(row["shooting_rate"]),
        shoot_accuracy=float(row["accuracy"]),
        max_speed=float(row["max_speed"]),
        acceleration=float(row["acceleration"]),
        align_time_s=float(row["align_time"]),
        dump_time_s=float(row["dump_time"]),
        shoot_on_move=bool(row["shoot_on_move"]),
        shoot_while_intake=bool(row["shoot_while_intake"]),
        cycle_variance_s=float(row["cycle_variance"]),
        defense_penalty=float(row["defense_penalty"]),
        max_climb_level=3,
        climb_time_s_by_level={1: 6, 2: 12, 3: 16},
    )


def baseline_red_spec() -> RobotSpec:
    # Anchor from dataset medians (turret=none-like).
    return RobotSpec(
        fuel_capacity=60,
        intake_fuel_per_s=8.0,
        shoot_fuel_per_s=20.0,
        shoot_accuracy=0.92,
        max_speed=15.5,
        acceleration=22.0,
        align_time_s=0.5,
        dump_time_s=0.3,
        shoot_on_move=False,
        shoot_while_intake=False,
        cycle_variance_s=1.0,
        defense_penalty=0.15,
        max_climb_level=3,
        climb_time_s_by_level={1: 6, 2: 12, 3: 16},
    )


def greedy_fuel_policy(env: RebuiltMacroSim, robot_id: int, *, idle_action_id: int) -> int:
    state = env.state
    assert state is not None
    cfg = env.config

    if state.t < int(state.robot_busy_until[robot_id]):
        return idle_action_id

    alliance = int(Alliance.RED) if robot_id < 3 else int(Alliance.BLUE)
    hubs = env.active_hubs()

    carried = int(state.robot_carried[robot_id])
    if carried > 0 and bool(hubs[alliance]):
        return cfg.n_neutral_bins + 1  # SCORE_HUB

    from_region = int(state.robot_region[robot_id])
    best_bin = None
    best_score = -1.0
    for bin_id in range(cfg.n_neutral_bins):
        fuel = int(state.neutral_fuel[bin_id])
        if fuel <= 0:
            continue
        travel = env._travel_time_s(from_region=from_region, to_region=neutral_bin_region(bin_id), robot_id=robot_id, opp_defenders=0)
        score = float(fuel) / (1.0 + float(travel))
        if score > best_score:
            best_score = score
            best_bin = bin_id

    if best_bin is not None:
        return int(best_bin)  # COLLECT_NEUTRAL(best_bin)

    if int(state.depot_fuel[alliance]) > 0 or int(state.outpost_chute[alliance]) > 0:
        return cfg.n_neutral_bins  # COLLECT_DEPOT

    return idle_action_id


def run_match(env: RebuiltMacroSim, *, seed: int, blue_robot_id: int = 3, red_robot_id: int = 0) -> tuple[int, int]:
    env.reset(seed=seed)
    n_actions = action_space_size(n_neutral_bins=env.config.n_neutral_bins)
    idle_action_id = n_actions - 1

    while True:
        actions = np.full(6, idle_action_id, dtype=np.int32)
        actions[red_robot_id] = greedy_fuel_policy(env, red_robot_id, idle_action_id=idle_action_id)
        actions[blue_robot_id] = greedy_fuel_policy(env, blue_robot_id, idle_action_id=idle_action_id)
        res = env.step(actions)
        if res.terminated:
            break

    state = env.state
    assert state is not None
    return int(state.score[int(Alliance.BLUE)]), int(state.score[int(Alliance.RED)])


def main() -> int:
    parser = argparse.ArgumentParser(description="Rough calibration harness against massive_results.json")
    parser.add_argument("--path", type=Path, default=Path("massive_results.json"))
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--episodes-per-sample", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hub-mode", type=str, default="rebuilt", choices=["rebuilt", "always_on"])
    parser.add_argument("--distance-scale", type=float, default=1.0)
    args = parser.parse_args()

    data = json.loads(args.path.read_text())
    if not isinstance(data, list):
        raise SystemExit("Expected list[dict].")

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(data), size=min(args.samples, len(data)), replace=False)

    cfg0 = default_config()
    cfg = replace(
        cfg0,
        hub_mode=args.hub_mode,
        region_distance_ft=cfg0.region_distance_ft * float(args.distance_scale),
    )

    red_spec = baseline_red_spec()

    preds: list[float] = []
    targets: list[float] = []
    red_pts: list[float] = []

    for i in idxs:
        row = data[int(i)]
        blue_spec = spec_from_row(row)

        # 1v1: only robot 0 (red) and robot 3 (blue) act; others idle.
        robot_specs = (red_spec, red_spec, red_spec, blue_spec, blue_spec, blue_spec)
        env = RebuiltMacroSim(cfg, robot_specs=robot_specs, seed=int(rng.integers(0, 2**31 - 1)))

        blue_scores = []
        red_scores = []
        for ep in range(args.episodes_per_sample):
            seed = int(rng.integers(0, 2**31 - 1))
            b, r = run_match(env, seed=seed)
            blue_scores.append(b)
            red_scores.append(r)

        pred = float(np.mean(blue_scores))
        preds.append(pred)
        targets.append(float(row["fuel"]))
        red_pts.append(float(np.mean(red_scores)))

    pred_arr = np.asarray(preds, dtype=np.float64)
    tgt_arr = np.asarray(targets, dtype=np.float64)
    err = pred_arr - tgt_arr
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err * err)))
    corr = float(np.corrcoef(pred_arr, tgt_arr)[0, 1]) if len(pred_arr) > 1 else float("nan")

    print(f"samples: {len(pred_arr)}  episodes/sample: {args.episodes_per_sample}")
    print(f"hub_mode: {args.hub_mode}  distance_scale: {args.distance_scale}")
    print(f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  corr(pred,target): {corr:.3f}")
    print(f"pred mean: {pred_arr.mean():.2f}  target mean: {tgt_arr.mean():.2f}  red mean: {float(np.mean(red_pts)):.2f}")

    worst = np.argsort(np.abs(err))[-10:][::-1]
    print("\nWorst 10 (abs error):")
    for j in worst:
        print(f"  target={tgt_arr[j]:>6.1f} pred={pred_arr[j]:>6.1f} err={err[j]:>7.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

