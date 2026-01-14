from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rebuilt_muzero.sim.actions import ActionKind
from rebuilt_muzero.sim.state import Alliance, Phase, n_regions, red_outpost_region, red_tower_region


def _compute_bin_mirror_map(coords: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Compute a mirror mapping for neutral bins under x -> -x (field-length mirror).
    """
    bin_xy = coords[2 : 2 + n_bins]
    mirrored = np.zeros(n_bins, dtype=np.int32)
    for i in range(n_bins):
        target = np.array([-float(bin_xy[i, 0]), float(bin_xy[i, 1])], dtype=np.float32)
        d2 = np.sum((bin_xy - target[None, :]) ** 2, axis=1)
        mirrored[i] = int(np.argmin(d2))
    return mirrored


def _compute_region_mirror_map(*, coords: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Region mirror map swapping RED/BLUE landmarks and mirroring neutral bins.
    """
    n_reg = n_regions(n_bins)
    out = np.arange(n_reg, dtype=np.int32)

    # Swap hubs.
    out[0] = 1
    out[1] = 0

    # Mirror neutral bins by x -> -x.
    bin_mirror = _compute_bin_mirror_map(coords, n_bins)
    for b in range(n_bins):
        out[2 + b] = 2 + int(bin_mirror[b])

    # Swap outposts and towers.
    out[2 + n_bins] = 2 + n_bins + 1
    out[2 + n_bins + 1] = 2 + n_bins
    out[2 + n_bins + 2] = 2 + n_bins + 3
    out[2 + n_bins + 3] = 2 + n_bins + 2
    return out


@dataclass(slots=True)
class ObsEncoder:
    n_bins: int
    n_regions: int
    obs_dim: int

    _bin_mirror: np.ndarray
    _region_mirror: np.ndarray
    _robot_order_by_player: np.ndarray  # shape (2,6)

    @staticmethod
    def build(*, coords: np.ndarray, n_bins: int) -> "ObsEncoder":
        n_reg = n_regions(n_bins)
        bin_mirror = _compute_bin_mirror_map(coords, n_bins)
        region_mirror = _compute_region_mirror_map(coords=coords, n_bins=n_bins)
        robot_order_by_player = np.asarray(
            [
                [0, 1, 2, 3, 4, 5],  # RED to play
                [3, 4, 5, 0, 1, 2],  # BLUE to play -> current alliance robots first
            ],
            dtype=np.int32,
        )

        # Match scalars: t_norm, remaining_norm, phase one-hot (7), active hubs (2), score/penalty (4)
        base = 2 + 7 + 2 + 4
        fuel = n_bins + 2 + 2 + 2  # neutral + depot + chute + corral
        # Per-robot: region one-hot (n_reg) + carried + busy + climbed + defending + 7 spec features
        per_robot = n_reg + 4 + 7
        obs_dim = base + fuel + 6 * per_robot
        return ObsEncoder(
            n_bins=n_bins,
            n_regions=n_reg,
            obs_dim=obs_dim,
            _bin_mirror=bin_mirror,
            _region_mirror=region_mirror,
            _robot_order_by_player=robot_order_by_player,
        )

    def encode(self, env: object, *, to_play: int) -> np.ndarray:
        """
        Encode `RebuiltMacroSim` state into a canonical float32 vector from the POV of `to_play`.

        Canonicalization:
        - swap alliance-indexed arrays so index0 = current player
        - reorder robots so first 3 = current player's robots
        - mirror region ids (including neutral bins) when BLUE is to play
        """
        state = getattr(env, "state", None)
        if state is None:
            raise RuntimeError("env.state is None; call reset() first.")

        cfg = env.config
        total = float(env.total_match_s())

        player = int(to_play)
        if player not in (0, 1):
            raise ValueError(f"to_play must be 0 (RED) or 1 (BLUE), got {to_play}")
        opp = 1 - player

        # Allocate once per call; keep it simple.
        out = np.zeros((self.obs_dim,), dtype=np.float32)
        idx = 0

        # Time
        t = float(state.t)
        out[idx] = t / total
        out[idx + 1] = (total - t) / total
        idx += 2

        # Phase one-hot
        phase = int(env.phase_at(int(state.t)))
        if not (0 <= phase <= int(Phase.ENDGAME)):
            phase = int(Phase.ENDGAME)
        out[idx + phase] = 1.0
        idx += 7

        # Active hubs (own, opp)
        mask = int(env.active_hubs_mask(int(state.t)))
        out[idx] = 1.0 if (mask & (1 << player)) else 0.0
        out[idx + 1] = 1.0 if (mask & (1 << opp)) else 0.0
        idx += 2

        # Score + penalty (own, opp), lightly scaled.
        score = state.score
        pen = state.penalty_points
        out[idx] = float(score[player]) / 500.0
        out[idx + 1] = float(score[opp]) / 500.0
        out[idx + 2] = float(pen[player]) / 200.0
        out[idx + 3] = float(pen[opp]) / 200.0
        idx += 4

        # Fuel (neutral bins possibly mirrored), then depot/chute/corral swapped to (own, opp)
        neutral = state.neutral_fuel
        if player == int(Alliance.BLUE):
            neutral = neutral[self._bin_mirror]
        out[idx : idx + self.n_bins] = neutral.astype(np.float32, copy=False) / 500.0
        idx += self.n_bins

        out[idx] = float(state.depot_fuel[player]) / 50.0
        out[idx + 1] = float(state.depot_fuel[opp]) / 50.0
        idx += 2

        out[idx] = float(state.outpost_chute[player]) / float(cfg.outpost_chute_capacity)
        out[idx + 1] = float(state.outpost_chute[opp]) / float(cfg.outpost_chute_capacity)
        idx += 2

        out[idx] = float(state.outpost_corral[player]) / 100.0
        out[idx + 1] = float(state.outpost_corral[opp]) / 100.0
        idx += 2

        # Robots: reorder + mirror regions if needed.
        robot_ids = self._robot_order_by_player[player]
        regions = state.robot_region[robot_ids].astype(np.int32, copy=False)
        if player == int(Alliance.BLUE):
            regions = self._region_mirror[regions]

        carried = state.robot_carried[robot_ids].astype(np.float32, copy=False)
        busy_rem = np.maximum(0, state.robot_busy_until[robot_ids].astype(np.int32, copy=False) - int(state.t)).astype(np.float32, copy=False)
        climbed = state.robot_climbed_level[robot_ids].astype(np.float32, copy=False)

        # Defending flag from current task id + busy.
        defending = np.zeros((6,), dtype=np.float32)
        task_ids = state.robot_task_action_id[robot_ids].astype(np.int32, copy=False)
        for i in range(6):
            if busy_rem[i] <= 0:
                continue
            a = int(task_ids[i])
            if a < 0:
                continue
            kind = int(env._action_kind[a])
            if kind == int(ActionKind.DEFEND_OPPONENT_HUB_LANE) or kind == int(ActionKind.DEFEND_OPPONENT_COLLECTOR):
                defending[i] = 1.0

        # Robot spec features (normalized). Mirror ordering with robot_ids.
        specs = env.robot_specs
        # NOTE: this assumes `robot_specs` is length 6.
        spec_cap = np.array([float(specs[r].fuel_capacity) for r in robot_ids], dtype=np.float32)
        spec_intake = np.array([float(specs[r].intake_fuel_per_s) for r in robot_ids], dtype=np.float32)
        spec_shoot = np.array([float(specs[r].shoot_fuel_per_s) for r in robot_ids], dtype=np.float32)
        spec_acc = np.array([float(specs[r].shoot_accuracy) for r in robot_ids], dtype=np.float32)
        spec_v = np.array([float(specs[r].max_speed) for r in robot_ids], dtype=np.float32)
        spec_a = np.array([float(specs[r].acceleration) for r in robot_ids], dtype=np.float32)
        spec_climb = np.array([float(specs[r].max_climb_level) for r in robot_ids], dtype=np.float32)

        # Normalize carried by capacity.
        carried_norm = np.where(spec_cap > 0, carried / spec_cap, 0.0).astype(np.float32, copy=False)

        # Append per-robot blocks.
        for i in range(6):
            r = int(regions[i])
            if 0 <= r < self.n_regions:
                out[idx + r] = 1.0
            idx += self.n_regions

            out[idx] = float(carried_norm[i])
            out[idx + 1] = float(min(1.0, busy_rem[i] / 30.0))
            out[idx + 2] = float(climbed[i] / 3.0)
            out[idx + 3] = float(defending[i])
            idx += 4

            out[idx] = float(min(1.0, spec_cap[i] / 80.0))
            out[idx + 1] = float(min(1.0, spec_intake[i] / 15.0))
            out[idx + 2] = float(min(1.0, spec_shoot[i] / 30.0))
            out[idx + 3] = float(np.clip(spec_acc[i], 0.0, 1.0))
            out[idx + 4] = float(min(1.0, spec_v[i] / 20.0))
            out[idx + 5] = float(min(1.0, spec_a[i] / 35.0))
            out[idx + 6] = float(min(1.0, spec_climb[i] / 3.0))
            idx += 7

        if idx != self.obs_dim:
            raise RuntimeError(f"internal encoder bug: wrote {idx} floats, expected {self.obs_dim}")
        return out

