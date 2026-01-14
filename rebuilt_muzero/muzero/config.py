from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MuZeroConfig:
    # Game / search
    discount: float = 0.997
    num_simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    max_policy_actions: int = 64  # restrict search+targets to top-k actions

    # Network
    latent_dim: int = 128
    hidden_dim: int = 256
    action_embed_dim: int = 64

    # Training
    unroll_steps: int = 10
    batch_size: int = 64
    td_steps: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    train_steps_per_iteration: int = 200
    value_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0

    # Replay
    replay_capacity_games: int = 2000
    min_replay_games: int = 20

    # Self-play
    games_per_iteration: int = 10
    temperature: float = 1.0
    temperature_drop_step: int = 60  # after this many moves in a game, go greedy
