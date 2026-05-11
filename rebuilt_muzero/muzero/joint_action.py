from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class JointActionSpace:
    n_per_robot: int
    n_robots: int = 3

    @property
    def size(self) -> int:
        return int(self.n_per_robot) ** int(self.n_robots)

    def encode(self, actions: np.ndarray) -> int:
        if actions.shape != (self.n_robots,):
            raise ValueError(f"actions must have shape ({self.n_robots},), got {actions.shape}")
        base = int(self.n_per_robot)
        out = 0
        mul = 1
        for i in range(self.n_robots):
            a = int(actions[i])
            if not (0 <= a < base):
                raise ValueError(f"per-robot action out of range: {a} not in [0,{base})")
            out += a * mul
            mul *= base
        return int(out)

    def decode(self, joint_action: int) -> np.ndarray:
        base = int(self.n_per_robot)
        if not (0 <= int(joint_action) < self.size):
            raise ValueError(f"joint_action out of range: {joint_action} not in [0,{self.size})")
        x = int(joint_action)
        out = np.zeros(self.n_robots, dtype=np.int32)
        for i in range(self.n_robots):
            out[i] = x % base
            x //= base
        return out
