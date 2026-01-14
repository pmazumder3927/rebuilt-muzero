from __future__ import annotations

import numpy as np
import torch

from rebuilt_muzero.muzero.config import MuZeroConfig
from rebuilt_muzero.muzero.game import RebuiltTurnBasedGame
from rebuilt_muzero.muzero.mcts import PolicyTarget, run_mcts
from rebuilt_muzero.muzero.networks import MuZeroNet
from rebuilt_muzero.muzero.replay import GameHistory


def play_selfplay_game(
    *,
    game: RebuiltTurnBasedGame,
    net: MuZeroNet,
    config: MuZeroConfig,
    seed: int,
    device: torch.device,
) -> GameHistory:
    rng = np.random.default_rng(seed)
    obs = game.reset(seed=seed)
    legal = np.arange(game.action_space_size, dtype=np.int32)

    history = GameHistory(obs=[], actions=[], rewards=[], root_values=[], policy_action_ids=[], policy_probs=[])

    step = 0
    while True:
        temp = float(config.temperature) if step < int(config.temperature_drop_step) else 0.0
        add_noise = bool(float(config.dirichlet_fraction) > 0.0 and temp > 0.0)

        mcts_out = run_mcts(
            net=net,
            config=config,
            obs=obs,
            legal_actions=legal,
            rng=rng,
            temperature=temp,
            add_exploration_noise=add_noise,
            device=device,
        )

        history.obs.append(obs.astype(np.float32, copy=False))
        history.actions.append(int(mcts_out.action))
        history.root_values.append(float(mcts_out.root_value))

        # Store sparse policy (padded to max_policy_actions).
        ids, probs = _pad_policy(mcts_out.policy, k=int(config.max_policy_actions))
        history.policy_action_ids.append(ids)
        history.policy_probs.append(probs)

        out = game.step(int(mcts_out.action))
        history.rewards.append(float(out.reward))
        obs = out.obs
        step += 1

        if out.terminated or step >= game.sim.total_match_s() + 5:
            break

    return history


def _pad_policy(policy: PolicyTarget, *, k: int) -> tuple[np.ndarray, np.ndarray]:
    ids = np.full((k,), -1, dtype=np.int32)
    probs = np.zeros((k,), dtype=np.float32)
    n = min(int(k), int(policy.action_ids.shape[0]))
    if n > 0:
        ids[:n] = policy.action_ids[:n].astype(np.int32, copy=False)
        probs[:n] = policy.probs[:n].astype(np.float32, copy=False)
    s = float(np.sum(probs))
    if s > 0:
        probs = (probs / s).astype(np.float32, copy=False)
    return ids, probs
