from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rebuilt_muzero.sim import RebuiltMacroSim, render_env_matplotlib  # noqa: E402
from rebuilt_muzero.sim.actions import action_space_size  # noqa: E402
from rebuilt_muzero.sim.state import Alliance, neutral_bin_region  # noqa: E402


def greedy_policy(env: RebuiltMacroSim, robot_id: int, *, idle_action_id: int) -> int:
    state = env.state
    assert state is not None
    cfg = env.config

    if int(state.t) < int(state.robot_busy_until[robot_id]):
        return idle_action_id

    alliance = int(Alliance.RED) if robot_id < 3 else int(Alliance.BLUE)
    hub_active = bool(env.active_hubs_mask() & (1 << int(alliance)))
    carried = int(state.robot_carried[robot_id])
    if carried > 0 and hub_active:
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Matplotlib debug visualization for RebuiltMacroSim.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--policy", choices=["random", "greedy", "idle"], default="greedy")
    parser.add_argument("--steps", type=int, default=9999, help="Max steps to run (default: full match).")
    parser.add_argument("--field-image", type=Path, default=Path("assets/field.png"))
    parser.add_argument("--layout-json", type=Path, default=Path("assets/saved_layout.json"))
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--save-frames", type=Path, default=None)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    env = RebuiltMacroSim(seed=args.seed)
    env.reset(seed=args.seed)

    rng = np.random.default_rng(args.seed)
    n_actions = action_space_size(n_neutral_bins=env.config.n_neutral_bins)
    idle_action_id = n_actions - 1

    fallback_field = root / ".tmp" / "CycletimeHeatmap" / "field.png"
    fallback_layout = root / ".tmp" / "CycletimeHeatmap" / "saved_layout.json"

    if args.field_image.exists():
        field_path = args.field_image
    elif fallback_field.exists():
        field_path = fallback_field
    else:
        field_path = None

    if args.layout_json.exists():
        layout_path = args.layout_json
    elif fallback_layout.exists():
        layout_path = fallback_layout
    else:
        layout_path = None

    import matplotlib.pyplot as plt

    if not args.no_show:
        plt.ion()

    outdir = args.save_frames
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)

    fig = None
    axes = None

    max_steps = min(args.steps, env.total_match_s() // env.config.decision_interval_s + 1)
    dt = 1.0 / max(1e-6, float(args.fps))

    for step in range(max_steps):
        fig, axes = render_env_matplotlib(
            env,
            field_image_path=field_path,
            layout_json_path=layout_path,
            fig=fig,
            axes=axes,
        )
        if outdir is not None:
            fig.savefig(outdir / f"frame_{step:04d}.png", dpi=150)

        if not args.no_show:
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(dt)

        if args.policy == "idle":
            actions = np.full(6, idle_action_id, dtype=np.int32)
        elif args.policy == "random":
            actions = rng.integers(0, n_actions, size=6, dtype=np.int32)
        else:
            actions = np.full(6, idle_action_id, dtype=np.int32)
            for rid in range(6):
                actions[rid] = greedy_policy(env, rid, idle_action_id=idle_action_id)

        _, terminated = env.step_fast(actions)
        if terminated:
            break

    if not args.no_show:
        plt.ioff()
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
