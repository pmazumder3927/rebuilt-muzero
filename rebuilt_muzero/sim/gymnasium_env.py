from __future__ import annotations

from typing import Any

import numpy as np

from rebuilt_muzero.sim.config import GameConfig, RobotSpec
from rebuilt_muzero.sim.env import RebuiltMacroSim


def _require_gymnasium() -> Any:
    try:
        import gymnasium as gym  # type: ignore
        from gymnasium import spaces  # type: ignore

        return gym, spaces
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "gymnasium is not installed. Install it (e.g. `pip install gymnasium`) to use rebuilt_muzero.sim.gymnasium_env."
        ) from e


class RebuiltMacroGymEnv:  # pragma: no cover
    """
    Optional Gymnasium wrapper around `RebuiltMacroSim`.

    This file intentionally keeps Gymnasium as an *optional* dependency. Importing this module
    will fail with a clear error if `gymnasium` is not installed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: GameConfig | None = None,
        *,
        robot_specs: tuple[RobotSpec, ...] | None = None,
        seed: int | None = None,
    ) -> None:
        gym, spaces = _require_gymnasium()
        self._gym = gym
        self._spaces = spaces

        self.sim = RebuiltMacroSim(config, robot_specs=robot_specs, seed=seed)

        self.action_space = spaces.MultiDiscrete([self.sim.n_actions] * 6)
        self.observation_space = spaces.Dict(
            {
                "t": spaces.Discrete(self.sim.total_match_s() + 1),
                "time_remaining_s": spaces.Discrete(self.sim.total_match_s() + 1),
                "phase": spaces.Discrete(7),
                "active_hubs": spaces.MultiBinary(2),
                "score": spaces.Box(low=0, high=10**9, shape=(2,), dtype=np.int32),
                "penalty_points": spaces.Box(low=0, high=10**9, shape=(2,), dtype=np.int32),
                "neutral_fuel": spaces.Box(low=0, high=10**9, shape=(self.sim.config.n_neutral_bins,), dtype=np.int32),
                "depot_fuel": spaces.Box(low=0, high=10**9, shape=(2,), dtype=np.int32),
                "outpost_chute": spaces.Box(low=0, high=10**9, shape=(2,), dtype=np.int32),
                "outpost_corral": spaces.Box(low=0, high=10**9, shape=(2,), dtype=np.int32),
                "robot_region": spaces.Box(low=0, high=10**6, shape=(6,), dtype=np.int16),
                "robot_carried": spaces.Box(low=0, high=10**6, shape=(6,), dtype=np.int16),
                "robot_busy_until": spaces.Box(low=0, high=10**6, shape=(6,), dtype=np.int16),
                "robot_task_action_id": spaces.Box(low=-1, high=self.sim.n_actions, shape=(6,), dtype=np.int16),
                "robot_task_target_region": spaces.Box(low=0, high=10**6, shape=(6,), dtype=np.int16),
                "robot_climbed_level": spaces.Box(low=0, high=3, shape=(6,), dtype=np.int8),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        _ = options
        obs = self.sim.reset(seed=seed)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        res = self.sim.step(np.asarray(action, dtype=np.int32))
        truncated = False
        return res.obs, float(res.reward), bool(res.terminated), truncated, dict(res.info)

    def render(self) -> None:
        return None

