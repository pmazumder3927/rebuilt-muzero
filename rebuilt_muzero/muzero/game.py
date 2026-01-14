from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rebuilt_muzero.muzero.joint_action import JointActionSpace
from rebuilt_muzero.muzero.obs_encoder import ObsEncoder
from rebuilt_muzero.sim import GameConfig, RebuiltMacroSim
from rebuilt_muzero.sim.actions import action_space_size
from rebuilt_muzero.sim.config import RobotSpec, default_config, default_robot_specs


@dataclass(slots=True)
class StepOutput:
    obs: np.ndarray  # canonical (current-player) obs vector
    reward: float  # from POV of player who just acted
    terminated: bool


class RebuiltTurnBasedGame:
    """
    Turn-based wrapper around `RebuiltMacroSim` for two-player MuZero.

    This wrapper alternates control between alliances (RED then BLUE...), where each "move"
    sets actions for the current player's 3 robots only. The opponent's action slots are filled
    with IDLE (their robots continue any in-progress tasks).

    Notes
    - This is a practical approximation for MuZero; it trades some simultaneity for a standard
      two-player turn structure.
    - Observations are canonicalized to the current player's POV (swap alliances, mirror bins/regions).
    """

    def __init__(
        self,
        config: GameConfig | None = None,
        *,
        robot_specs: tuple[RobotSpec, ...] | None = None,
        seed: int | None = None,
    ) -> None:
        self.sim = RebuiltMacroSim(config or default_config(), robot_specs=robot_specs or default_robot_specs(), seed=seed)
        self.to_play = 0  # 0=RED, 1=BLUE

        self.n_per_robot = action_space_size(n_neutral_bins=self.sim.config.n_neutral_bins)
        self.idle_action_id = self.n_per_robot - 1
        self.joint_action_space = JointActionSpace(n_per_robot=self.n_per_robot, n_robots=3)

        coords = self.sim.config.region_coords_ft
        if coords is None:
            raise ValueError("config.region_coords_ft is required.")
        self.encoder = ObsEncoder.build(coords=coords, n_bins=self.sim.config.n_neutral_bins)

    @property
    def action_space_size(self) -> int:
        return int(self.joint_action_space.size)

    @property
    def obs_dim(self) -> int:
        return int(self.encoder.obs_dim)

    def reset(self, *, seed: int | None = None) -> np.ndarray:
        self.sim.reset(seed=seed)
        self.to_play = 0
        return self.encoder.encode(self.sim, to_play=self.to_play)

    def legal_actions(self) -> np.ndarray:
        """
        Return encoded legal joint actions for the current player.

        For MuZero, we keep the legal set fixed across nodes (the model does not have explicit
        legality logic). Busy robots simply ignore actions in the underlying simulator.
        """
        return np.arange(self.action_space_size, dtype=np.int32)

    def step(self, joint_action: int) -> StepOutput:
        state = self.sim.state
        if state is None:
            raise RuntimeError("Call reset() first.")

        per_robot = self.joint_action_space.decode(int(joint_action))
        actions = np.full((6,), self.idle_action_id, dtype=np.int32)
        if self.to_play == 0:
            actions[0:3] = per_robot
        else:
            actions[3:6] = per_robot

        reward_red_pov, terminated = self.sim.step_fast(actions)
        reward = float(reward_red_pov) if self.to_play == 0 else float(-reward_red_pov)

        self.to_play = 1 - int(self.to_play)
        obs = self.encoder.encode(self.sim, to_play=self.to_play)
        return StepOutput(obs=obs, reward=reward, terminated=bool(terminated))
