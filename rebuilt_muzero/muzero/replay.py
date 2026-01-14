from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class GameHistory:
    obs: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    root_values: list[float]
    policy_action_ids: list[np.ndarray]  # each (K,) int32 padded with -1
    policy_probs: list[np.ndarray]  # each (K,) float32

    @property
    def length(self) -> int:
        return len(self.actions)

    def compute_value_targets(self, *, discount: float, td_steps: int) -> np.ndarray:
        """
        Compute bootstrap value targets for each timestep, from the POV of the player to act at that timestep.

        Since players alternate each move, rewards and bootstrap values must alternate sign when viewed from
        a fixed timestep's POV:
          return_t = r_t - γ r_{t+1} + γ^2 r_{t+2} - ...
        """
        T = self.length
        out = np.zeros((T,), dtype=np.float32)
        gamma = float(discount)
        n = int(td_steps)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        roots = np.asarray(self.root_values, dtype=np.float32)
        for t in range(T):
            acc = 0.0
            g = 1.0
            for k in range(n):
                idx = t + k
                if idx >= T:
                    break
                sign = 1.0 if (k % 2 == 0) else -1.0
                acc += sign * g * float(rewards[idx])
                g *= gamma
            boot_idx = t + n
            if boot_idx < T:
                sign = 1.0 if (n % 2 == 0) else -1.0
                acc += sign * g * float(roots[boot_idx])
            out[t] = np.float32(acc)
        return out


class ReplayBuffer:
    def __init__(self, *, capacity_games: int, rng: np.random.Generator) -> None:
        self.capacity_games = int(capacity_games)
        self.rng = rng
        self._games: list[GameHistory] = []

    def __len__(self) -> int:
        return len(self._games)

    def add_game(self, game: GameHistory) -> None:
        self._games.append(game)
        if len(self._games) > self.capacity_games:
            # FIFO eviction
            self._games = self._games[-self.capacity_games :]

    def games(self) -> Iterable[GameHistory]:
        return list(self._games)

    def sample_batch(
        self,
        *,
        batch_size: int,
        unroll_steps: int,
        td_steps: int,
        discount: float,
        policy_k: int,
    ) -> dict[str, np.ndarray]:
        if len(self._games) == 0:
            raise RuntimeError("replay is empty")

        B = int(batch_size)
        K = int(unroll_steps)
        P = int(policy_k)

        # Sample games proportional to length.
        lengths = np.array([g.length for g in self._games], dtype=np.float32)
        probs = lengths / (float(np.sum(lengths)) + 1e-8)
        game_idxs = self.rng.choice(np.arange(len(self._games)), size=B, p=probs)

        obs0 = []
        actions = np.zeros((B, K), dtype=np.int32)
        rewards = np.zeros((B, K), dtype=np.float32)
        value_targets = np.zeros((B, K + 1), dtype=np.float32)
        policy_action_ids = np.full((B, K + 1, P), -1, dtype=np.int32)
        policy_probs = np.zeros((B, K + 1, P), dtype=np.float32)
        valid_steps = np.zeros((B, K + 1), dtype=np.float32)
        valid_rewards = np.zeros((B, K), dtype=np.float32)

        for bi, gi in enumerate(game_idxs):
            g = self._games[int(gi)]
            T = g.length
            if T <= 0:
                continue

            start = int(self.rng.integers(0, T))
            obs0.append(g.obs[start].astype(np.float32, copy=False))

            vt = g.compute_value_targets(discount=discount, td_steps=td_steps)

            for k in range(K + 1):
                si = start + k
                if si >= T:
                    break
                valid_steps[bi, k] = 1.0
                value_targets[bi, k] = vt[si]
                ids = g.policy_action_ids[si]
                pr = g.policy_probs[si]
                m = min(P, int(ids.shape[0]))
                policy_action_ids[bi, k, :m] = ids[:m]
                policy_probs[bi, k, :m] = pr[:m]

            for k in range(K):
                si = start + k
                if si >= T:
                    break
                valid_rewards[bi, k] = 1.0
                actions[bi, k] = int(g.actions[si])
                rewards[bi, k] = float(g.rewards[si])

        obs0 = np.stack(obs0, axis=0).astype(np.float32, copy=False) if obs0 else np.zeros((B, 0), dtype=np.float32)
        return {
            "obs0": obs0,
            "actions": actions,
            "rewards": rewards,
            "value_targets": value_targets,
            "policy_action_ids": policy_action_ids,
            "policy_probs": policy_probs,
            "valid_steps": valid_steps,
            "valid_rewards": valid_rewards,
        }

