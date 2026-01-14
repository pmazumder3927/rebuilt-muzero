from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rebuilt_muzero.sim.actions import ActionKind, action_space_size, decode_action
from rebuilt_muzero.sim.config import GameConfig, RobotSpec, default_config, default_robot_specs
from rebuilt_muzero.sim.state import Alliance, Phase, SimState, n_regions
from rebuilt_muzero.sim.state import (
    blue_outpost_region,
    blue_tower_region,
    blue_zone_region,
    is_in_alliance_zone,
    neutral_bin_region,
    red_outpost_region,
    red_tower_region,
    red_zone_region,
)


@dataclass(frozen=True, slots=True)
class StepResult:
    obs: dict[str, Any]
    reward: float
    terminated: bool
    info: dict[str, Any]


class RebuiltMacroSim:
    """
    REBUILT macro-simulator.

    - Time is advanced in `config.decision_interval_s` increments.
    - Each robot action schedules a timed macro (travel/intake/shoot/etc.) by setting `robot_busy_until`.
    - This environment is intentionally coarse: it is designed for strategy learning (MuZero/MCTS),
      not for control or kinematics.
    """

    def __init__(
        self,
        config: GameConfig | None = None,
        *,
        robot_specs: tuple[RobotSpec, ...] | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = config or default_config()
        self._rng = np.random.default_rng(seed)
        self.robot_specs = robot_specs or default_robot_specs()

        expected_regions = n_regions(self.config.n_neutral_bins)
        if self.config.region_distance_ft is None:
            raise ValueError("config.region_distance_ft must be set (use default_config() or provide one).")
        if self.config.region_distance_ft.shape != (expected_regions, expected_regions):
            raise ValueError(
                f"region_distance_ft must have shape {(expected_regions, expected_regions)}, "
                f"got {self.config.region_distance_ft.shape}"
            )

        for bin_id in self.config.hub_exit_bin_ids:
            if not (0 <= bin_id < self.config.n_neutral_bins):
                raise ValueError(f"hub_exit_bin_ids contains out-of-range bin id {bin_id}")
        for bin_id in self.config.missed_shot_bin_id_by_alliance:
            if not (0 <= bin_id < self.config.n_neutral_bins):
                raise ValueError(f"missed_shot_bin_id_by_alliance contains out-of-range bin id {bin_id}")
        if not np.isclose(sum(self.config.hub_exit_probs), 1.0):
            raise ValueError(f"hub_exit_probs must sum to 1.0, got {sum(self.config.hub_exit_probs):.6f}")

        self.n_actions = action_space_size(n_neutral_bins=self.config.n_neutral_bins)
        self._action_kind = np.empty(self.n_actions, dtype=np.int8)
        self._action_arg = np.empty(self.n_actions, dtype=np.int16)
        for action_id in range(self.n_actions):
            decoded = decode_action(action_id, n_neutral_bins=self.config.n_neutral_bins)
            self._action_kind[action_id] = int(decoded.kind)
            self._action_arg[action_id] = int(decoded.arg)

        self._match_end_t = int(self.config.total_match_s())
        self.state: SimState | None = None

    # ---- Clock / phase / hub schedule -------------------------------------------------

    def total_match_s(self) -> int:
        return self._match_end_t

    def phase_at(self, t: int) -> Phase:
        cfg = self.config
        if t < cfg.auto_s:
            return Phase.AUTO
        t -= cfg.auto_s
        if t < cfg.transition_s:
            return Phase.TRANSITION
        t -= cfg.transition_s
        shift_block_s = cfg.n_shifts * cfg.shift_s
        if t < shift_block_s:
            shift_idx = t // cfg.shift_s  # 0..3
            return Phase(Phase.SHIFT1 + int(shift_idx))
        return Phase.ENDGAME

    def time_remaining_s(self) -> int:
        if self.state is None:
            raise RuntimeError("Call reset() first.")
        return self.total_match_s() - self.state.t

    def active_hubs_mask(self, t: int | None = None) -> int:
        if self.state is None:
            raise RuntimeError("Call reset() first.")

        if t is None:
            t = int(self.state.t)

        if self.config.hub_mode == "always_on":
            return 0b11
        if self.config.hub_mode != "rebuilt":
            raise ValueError(f"Unknown hub_mode={self.config.hub_mode!r} (expected 'rebuilt' or 'always_on').")

        phase = self.phase_at(t)
        if phase != Phase.AUTO and self.state.first_shift_active_alliance < 0:
            self._resolve_first_shift_active_alliance()

        if phase in (Phase.AUTO, Phase.TRANSITION, Phase.ENDGAME):
            return 0b11

        shift_idx = int(phase) - int(Phase.SHIFT1)  # 0..3
        active_alliance = int(self.state.first_shift_active_alliance) ^ (shift_idx & 1)
        return 1 << int(active_alliance)

    def active_hubs(self) -> np.ndarray:
        mask = self.active_hubs_mask()
        return np.array([bool(mask & 0b01), bool(mask & 0b10)], dtype=np.bool_)

    def _resolve_first_shift_active_alliance(self) -> None:
        if self.state is None:
            raise RuntimeError("Call reset() first.")
        if self.state.first_shift_active_alliance >= 0:
            return

        red = int(self.state.auto_fuel_scored[Alliance.RED])
        blue = int(self.state.auto_fuel_scored[Alliance.BLUE])
        if red > blue:
            self.state.first_shift_active_alliance = int(Alliance.RED)
        elif blue > red:
            self.state.first_shift_active_alliance = int(Alliance.BLUE)
        else:
            self.state.first_shift_active_alliance = int(self._rng.integers(0, 2))

    # ---- Reset / observe --------------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        cfg = self.config
        neutral = np.full(cfg.n_neutral_bins, cfg.initial_neutral_fuel_per_bin, dtype=np.int32)
        depot = np.full(2, cfg.initial_depot_fuel, dtype=np.int32)
        outpost_chute = np.full(2, cfg.initial_outpost_chute_fuel, dtype=np.int32)
        outpost_corral = np.zeros(2, dtype=np.int32)

        # Robots 0..2 red, 3..5 blue; start in their alliance zones.
        robot_region = np.array([0, 0, 0, 1, 1, 1], dtype=np.int16)
        robot_carried = np.zeros(6, dtype=np.int16)
        robot_busy_until = np.zeros(6, dtype=np.int16)
        robot_task_action_id = np.full(6, -1, dtype=np.int16)
        robot_task_target_region = robot_region.copy()
        robot_task_reserved_fuel = np.zeros(6, dtype=np.int16)
        robot_climbed_level = np.zeros(6, dtype=np.int8)
        robot_pin_time = np.zeros(6, dtype=np.int8)

        self.state = SimState(
            t=0,
            first_shift_active_alliance=-1,
            score=np.zeros(2, dtype=np.int32),
            penalty_points=np.zeros(2, dtype=np.int32),
            auto_fuel_scored=np.zeros(2, dtype=np.int32),
            neutral_fuel=neutral,
            depot_fuel=depot,
            outpost_chute=outpost_chute,
            outpost_corral=outpost_corral,
            robot_region=robot_region,
            robot_carried=robot_carried,
            robot_busy_until=robot_busy_until,
            robot_task_action_id=robot_task_action_id,
            robot_task_target_region=robot_task_target_region,
            robot_task_reserved_fuel=robot_task_reserved_fuel,
            robot_climbed_level=robot_climbed_level,
            robot_pin_time=robot_pin_time,
        )
        return self.observe()

    def observe(self) -> dict[str, Any]:
        if self.state is None:
            raise RuntimeError("Call reset() first.")

        phase = self.phase_at(self.state.t)
        hubs = self.active_hubs()
        return {
            "t": int(self.state.t),
            "time_remaining_s": int(self.total_match_s() - self.state.t),
            "phase": int(phase),
            "active_hubs": hubs.astype(np.int8).copy(),
            "score": self.state.score.copy(),
            "penalty_points": self.state.penalty_points.copy(),
            "neutral_fuel": self.state.neutral_fuel.copy(),
            "depot_fuel": self.state.depot_fuel.copy(),
            "outpost_chute": self.state.outpost_chute.copy(),
            "outpost_corral": self.state.outpost_corral.copy(),
            "robot_region": self.state.robot_region.copy(),
            "robot_carried": self.state.robot_carried.copy(),
            "robot_busy_until": self.state.robot_busy_until.copy(),
            "robot_task_action_id": self.state.robot_task_action_id.copy(),
            "robot_task_target_region": self.state.robot_task_target_region.copy(),
            "robot_climbed_level": self.state.robot_climbed_level.copy(),
        }

    # ---- Step ------------------------------------------------------------------------

    def step_fast(self, actions: np.ndarray) -> tuple[float, bool]:
        """
        Fast step that avoids observation construction/copies.

        Returns `(reward, terminated)`.
        """
        if self.state is None:
            raise RuntimeError("Call reset() first.")
        if self.state.t >= self._match_end_t:
            return 0.0, True
        reward, terminated, _, _ = self._advance(actions, validate_actions=False)
        return reward, terminated

    def step(self, actions: np.ndarray) -> StepResult:
        """
        Advance the sim by one macro decision interval.

        `actions` is an int array of shape (6,) where each entry is an encoded per-robot action id.

        This is a 2-player zero-sum step; the returned `reward` is the incremental
        swing in (red_total_score - blue_total_score) over this macro interval.
        """
        if self.state is None:
            raise RuntimeError("Call reset() first.")
        if self.state.t >= self._match_end_t:
            return StepResult(obs=self.observe(), reward=0.0, terminated=True, info={"reason": "match_over"})

        reward, terminated, delta_red, delta_blue = self._advance(actions, validate_actions=True)
        obs = self.observe()
        info = {
            "phase": obs["phase"],
            "active_hubs": obs["active_hubs"],
            "delta_total": np.asarray([delta_red, delta_blue], dtype=np.int32),
        }
        return StepResult(obs=obs, reward=reward, terminated=terminated, info=info)

    def _advance(self, actions: np.ndarray, *, validate_actions: bool) -> tuple[float, bool, int, int]:
        if self.state is None:
            raise RuntimeError("Call reset() first.")

        cfg = self.config
        state = self.state

        if actions.shape != (6,):
            raise ValueError(f"actions must have shape (6,), got {actions.shape}")
        if validate_actions and (np.any(actions < 0) or np.any(actions >= self.n_actions)):
            raise ValueError(
                f"actions must be within [0, {self.n_actions}), got min={int(actions.min())} max={int(actions.max())}"
            )

        start_red = int(state.score[int(Alliance.RED)] + state.penalty_points[int(Alliance.RED)])
        start_blue = int(state.score[int(Alliance.BLUE)] + state.penalty_points[int(Alliance.BLUE)])

        # 1) Apply any tasks that finished at or before the current time.
        self._process_completions(now_t=int(state.t))

        # 2) Compute defense pressure (ongoing + newly requested defenders).
        ongoing_red = 0
        ongoing_blue = 0
        for robot_id in range(6):
            task_action_id = int(state.robot_task_action_id[robot_id])
            if task_action_id < 0:
                continue
            if int(state.robot_busy_until[robot_id]) <= int(state.t):
                continue
            kind = int(self._action_kind[task_action_id])
            if kind == ActionKind.DEFEND_OPPONENT_HUB_LANE or kind == ActionKind.DEFEND_OPPONENT_COLLECTOR:
                if self._robot_alliance(robot_id) == Alliance.RED:
                    ongoing_red += 1
                else:
                    ongoing_blue += 1

        new_red = 0
        new_blue = 0
        for robot_id in range(6):
            if int(state.t) < int(state.robot_busy_until[robot_id]):
                continue
            action_id = int(actions[robot_id])
            kind = int(self._action_kind[action_id])
            if kind == ActionKind.DEFEND_OPPONENT_HUB_LANE or kind == ActionKind.DEFEND_OPPONENT_COLLECTOR:
                if self._robot_alliance(robot_id) == Alliance.RED:
                    new_red += 1
                else:
                    new_blue += 1

        defense_red = ongoing_red + new_red
        defense_blue = ongoing_blue + new_blue

        # 3) Schedule actions for idle robots.
        for robot_id in range(6):
            if int(state.t) < int(state.robot_busy_until[robot_id]):
                continue
            action_id = int(actions[robot_id])
            kind = int(self._action_kind[action_id])
            arg = int(self._action_arg[action_id])
            alliance = self._robot_alliance(robot_id)
            opp_defenders = defense_blue if alliance == Alliance.RED else defense_red
            self._schedule_action(robot_id=robot_id, action_id=action_id, kind=kind, arg=arg, opp_defenders=opp_defenders)

        # 4) Advance time.
        state.t = int(state.t) + int(cfg.decision_interval_s)
        if int(state.t) > self._match_end_t:
            state.t = self._match_end_t

        # Human-player logistics (very coarse).
        self._fill_outpost_chutes()

        # Resolve shift order as soon as AUTO ends (uses AUTO fuel scored).
        if self.phase_at(int(state.t)) != Phase.AUTO and state.first_shift_active_alliance < 0:
            self._resolve_first_shift_active_alliance()

        # 5) Apply task completions that land exactly on this boundary.
        self._process_completions(now_t=int(state.t))

        end_red = int(state.score[int(Alliance.RED)] + state.penalty_points[int(Alliance.RED)])
        end_blue = int(state.score[int(Alliance.BLUE)] + state.penalty_points[int(Alliance.BLUE)])
        delta_red = end_red - start_red
        delta_blue = end_blue - start_blue
        reward = float(delta_red - delta_blue)
        terminated = int(state.t) >= self._match_end_t
        return reward, terminated, delta_red, delta_blue

    # ---- Transition helpers -----------------------------------------------------------

    @staticmethod
    def _robot_alliance(robot_id: int) -> int:
        return int(Alliance.RED) if robot_id < 3 else int(Alliance.BLUE)

    def _schedule_action(self, *, robot_id: int, action_id: int, kind: int, arg: int, opp_defenders: int) -> None:
        cfg = self.config
        state = self.state
        assert state is not None

        alliance = self._robot_alliance(robot_id)
        opponent = int(Alliance.BLUE) if alliance == int(Alliance.RED) else int(Alliance.RED)

        # Pin timer tracking (very coarse).
        if kind == ActionKind.DEFEND_OPPONENT_HUB_LANE or kind == ActionKind.DEFEND_OPPONENT_COLLECTOR:
            state.robot_pin_time[robot_id] = np.int8(state.robot_pin_time[robot_id] + cfg.decision_interval_s)
            if int(state.robot_pin_time[robot_id]) > cfg.pin_limit_s:
                state.penalty_points[opponent] += cfg.minor_foul_points
        else:
            state.robot_pin_time[robot_id] = np.int8(0)

        from_region = int(state.robot_region[robot_id])

        # Determine target region.
        target_region = from_region
        if kind == ActionKind.COLLECT_NEUTRAL:
            target_region = neutral_bin_region(arg)
        elif kind == ActionKind.COLLECT_DEPOT:
            target_region = red_zone_region() if alliance == Alliance.RED else blue_zone_region()
        elif kind == ActionKind.SCORE_HUB:
            target_region = red_zone_region() if alliance == Alliance.RED else blue_zone_region()
        elif kind == ActionKind.DELIVER_OUTPOST:
            target_region = red_outpost_region(cfg.n_neutral_bins) if alliance == Alliance.RED else blue_outpost_region(cfg.n_neutral_bins)
        elif kind == ActionKind.DEFEND_OPPONENT_HUB_LANE or kind == ActionKind.DEFEND_OPPONENT_COLLECTOR:
            target_region = blue_zone_region() if alliance == Alliance.RED else red_zone_region()
        elif kind == ActionKind.PREP_CLIMB or kind == ActionKind.CLIMB:
            target_region = red_tower_region(cfg.n_neutral_bins) if alliance == Alliance.RED else blue_tower_region(cfg.n_neutral_bins)

        # Reserve resources immediately to avoid oversubscription.
        reserved_fuel = 0
        if kind == ActionKind.COLLECT_NEUTRAL:
            capacity_left = int(self.robot_specs[robot_id].fuel_capacity) - int(state.robot_carried[robot_id])
            if capacity_left > 0:
                available = int(state.neutral_fuel[arg])
                reserved_fuel = min(capacity_left, available)
                if reserved_fuel > 0:
                    state.neutral_fuel[arg] -= reserved_fuel
        elif kind == ActionKind.COLLECT_DEPOT:
            capacity_left = int(self.robot_specs[robot_id].fuel_capacity) - int(state.robot_carried[robot_id])
            if capacity_left > 0:
                take_from_chute = min(capacity_left, int(state.outpost_chute[alliance]))
                state.outpost_chute[alliance] -= take_from_chute
                capacity_left -= take_from_chute

                take_from_depot = min(capacity_left, int(state.depot_fuel[alliance]))
                state.depot_fuel[alliance] -= take_from_depot
                reserved_fuel = take_from_chute + take_from_depot
        elif kind == ActionKind.SCORE_HUB:
            reserved_fuel = int(state.robot_carried[robot_id])
            state.robot_carried[robot_id] = 0
        elif kind == ActionKind.DELIVER_OUTPOST:
            reserved_fuel = int(state.robot_carried[robot_id])
            state.robot_carried[robot_id] = 0

        travel_s = self._travel_time_s(
            from_region=from_region,
            to_region=target_region,
            robot_id=robot_id,
            opp_defenders=opp_defenders,
        )
        op_s = self._operation_time_s(robot_id=robot_id, kind=kind, arg=arg, reserved_fuel=reserved_fuel)
        duration_s = int(travel_s + op_s)
        if duration_s < 1:
            duration_s = 1
        sigma = float(self.robot_specs[robot_id].cycle_variance_s)
        if sigma > 0:
            duration_s += int(np.rint(self._rng.normal(0.0, sigma)))
            if duration_s < 1:
                duration_s = 1

        state.robot_busy_until[robot_id] = np.int16(state.t + duration_s)
        state.robot_task_action_id[robot_id] = np.int16(action_id)
        state.robot_task_target_region[robot_id] = np.int16(target_region)
        state.robot_task_reserved_fuel[robot_id] = np.int16(reserved_fuel)

    def _travel_time_s(self, *, from_region: int, to_region: int, robot_id: int, opp_defenders: int) -> int:
        if from_region == to_region:
            return 0

        cfg = self.config
        dist = float(cfg.region_distance_ft[from_region, to_region])
        spec = self.robot_specs[robot_id]
        vmax = max(0.1, float(spec.max_speed))
        accel = max(0.1, float(spec.acceleration))

        # Symmetric accelerate/cruise/decelerate profile.
        d_min = (vmax * vmax) / accel
        if dist <= d_min:
            t = 2.0 * np.sqrt(dist / accel)
        else:
            t = 2.0 * vmax / accel + (dist - d_min) / vmax

        t += float(cfg.drive_overhead_s)

        if opp_defenders > 0:
            t *= 1.0 + float(spec.defense_penalty) * float(opp_defenders)

        return int(np.ceil(t))

    def _operation_time_s(self, *, robot_id: int, kind: int, arg: int, reserved_fuel: int) -> int:
        cfg = self.config
        spec = self.robot_specs[robot_id]

        if kind == ActionKind.COLLECT_NEUTRAL or kind == ActionKind.COLLECT_DEPOT:
            if reserved_fuel <= 0:
                return int(np.ceil(float(cfg.collect_overhead_s)))
            t = float(cfg.collect_overhead_s) + (float(reserved_fuel) / max(0.1, float(spec.intake_fuel_per_s)))
            return int(np.ceil(t))

        if kind == ActionKind.SCORE_HUB:
            if reserved_fuel <= 0:
                overhead = (0.0 if spec.shoot_on_move else float(spec.align_time_s)) + (
                    0.0 if spec.shoot_while_intake else float(spec.dump_time_s)
                )
                return int(np.ceil(overhead))
            overhead = (0.0 if spec.shoot_on_move else float(spec.align_time_s)) + (
                0.0 if spec.shoot_while_intake else float(spec.dump_time_s)
            )
            t = overhead + (float(reserved_fuel) / max(0.1, float(spec.shoot_fuel_per_s)))
            return int(np.ceil(t))

        if kind == ActionKind.DELIVER_OUTPOST:
            return int(np.ceil(float(cfg.deliver_overhead_s)))

        if kind == ActionKind.DEFEND_OPPONENT_HUB_LANE or kind == ActionKind.DEFEND_OPPONENT_COLLECTOR:
            return cfg.defend_duration_s

        if kind == ActionKind.IDLE:
            return 1

        if kind == ActionKind.PREP_CLIMB:
            return 1

        if kind == ActionKind.CLIMB:
            level = int(arg)
            level = min(level, int(spec.max_climb_level))
            return int(spec.climb_time_s_by_level.get(level, 20))

        return 1

    def _fill_outpost_chutes(self) -> None:
        cfg = self.config
        state = self.state
        assert state is not None

        for alliance in (Alliance.RED, Alliance.BLUE):
            chute_room = cfg.outpost_chute_capacity - int(state.outpost_chute[int(alliance)])
            if chute_room <= 0:
                continue
            move = min(int(state.outpost_corral[int(alliance)]), chute_room, cfg.outpost_fill_fuel_per_s * cfg.decision_interval_s)
            if move > 0:
                state.outpost_corral[int(alliance)] -= move
                state.outpost_chute[int(alliance)] += move

    def _process_completions(self, *, now_t: int) -> None:
        cfg = self.config
        state = self.state
        assert state is not None

        for robot_id in range(6):
            if state.robot_task_action_id[robot_id] < 0:
                continue
            if int(state.robot_busy_until[robot_id]) > now_t:
                continue

            action_id = int(state.robot_task_action_id[robot_id])
            kind = int(self._action_kind[action_id])
            arg = int(self._action_arg[action_id])
            alliance = self._robot_alliance(robot_id)
            opponent = int(Alliance.BLUE) if alliance == int(Alliance.RED) else int(Alliance.RED)

            # Move robot to its target region.
            state.robot_region[robot_id] = state.robot_task_target_region[robot_id]

            reserved_fuel = int(state.robot_task_reserved_fuel[robot_id])

            if kind == ActionKind.COLLECT_NEUTRAL or kind == ActionKind.COLLECT_DEPOT:
                state.robot_carried[robot_id] = np.int16(int(state.robot_carried[robot_id]) + reserved_fuel)

            elif kind == ActionKind.DELIVER_OUTPOST:
                state.outpost_corral[alliance] += reserved_fuel

            elif kind == ActionKind.SCORE_HUB:
                # Legality: must be in alliance zone to score. If not, major foul and no points.
                legal = is_in_alliance_zone(int(state.robot_region[robot_id]), alliance, cfg.n_neutral_bins)

                # Apply defense effect at completion time (coarse): current active defenders reduce accuracy.
                # We approximate with defenders that are currently in a defense task.
                opp_defenders = 0
                for r2 in range(6):
                    if self._robot_alliance(r2) != opponent:
                        continue
                    if state.robot_task_action_id[r2] < 0:
                        continue
                    if int(state.robot_busy_until[r2]) <= now_t:
                        continue
                    task_action_id2 = int(state.robot_task_action_id[r2])
                    kind2 = int(self._action_kind[task_action_id2])
                    if kind2 == ActionKind.DEFEND_OPPONENT_HUB_LANE or kind2 == ActionKind.DEFEND_OPPONENT_COLLECTOR:
                        opp_defenders += 1

                accuracy = float(self.robot_specs[robot_id].shoot_accuracy)
                if opp_defenders > 0:
                    accuracy = max(0.0, accuracy - float(self.robot_specs[robot_id].defense_penalty) * float(opp_defenders))

                successes = int(self._rng.binomial(reserved_fuel, accuracy)) if reserved_fuel > 0 else 0
                misses = reserved_fuel - successes

                # Fuel always gets redistributed physically; points only if legal + hub active.
                hub_active = bool(self.active_hubs_mask(now_t) & (1 << int(alliance)))
                if legal and hub_active:
                    state.score[alliance] += successes * cfg.fuel_point_value
                    if self.phase_at(now_t) == Phase.AUTO:
                        state.auto_fuel_scored[alliance] += successes
                elif not legal and reserved_fuel > 0:
                    state.penalty_points[opponent] += cfg.major_foul_points

                # HUB exits for successful fuel.
                if successes > 0:
                    counts = self._rng.multinomial(successes, np.asarray(cfg.hub_exit_probs, dtype=np.float64))
                    for exit_idx, bin_id in enumerate(cfg.hub_exit_bin_ids):
                        state.neutral_fuel[bin_id] += int(counts[exit_idx])

                # Misses drop near the alliance's hub.
                if misses > 0:
                    miss_bin = cfg.missed_shot_bin_id_by_alliance[alliance]
                    state.neutral_fuel[miss_bin] += misses

            elif kind == ActionKind.CLIMB:
                if state.robot_climbed_level[robot_id] > 0:
                    pass
                else:
                    level = int(arg)
                    level = min(level, int(self.robot_specs[robot_id].max_climb_level))
                    level = max(1, level)
                    state.robot_climbed_level[robot_id] = np.int8(level)

                    if self.phase_at(now_t) == Phase.AUTO:
                        points = cfg.tower_points_auto_by_level[level - 1]
                    else:
                        points = cfg.tower_points_teleop_by_level[level - 1]
                    state.score[alliance] += int(points)

            # Endgame tower protection foul (very coarse): defending while an opponent is on their tower.
            if self.phase_at(now_t) == Phase.ENDGAME and (
                kind == ActionKind.DEFEND_OPPONENT_HUB_LANE or kind == ActionKind.DEFEND_OPPONENT_COLLECTOR
            ):
                opp_tower = red_tower_region(cfg.n_neutral_bins) if opponent == Alliance.RED else blue_tower_region(cfg.n_neutral_bins)
                opp_on_tower = bool(np.any(state.robot_region[(0 if opponent == Alliance.RED else 3) : (3 if opponent == Alliance.RED else 6)] == opp_tower))
                if opp_on_tower and float(self._rng.random()) < cfg.tower_contact_foul_prob:
                    state.penalty_points[opponent] += cfg.major_foul_points
                    if cfg.tower_contact_awards_level3:
                        state.score[opponent] += int(cfg.tower_points_teleop_by_level[2])

            # Clear task.
            state.robot_task_action_id[robot_id] = np.int16(-1)
            state.robot_task_reserved_fuel[robot_id] = np.int16(0)
