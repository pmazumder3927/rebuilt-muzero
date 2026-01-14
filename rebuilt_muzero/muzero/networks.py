from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True, slots=True)
class InferenceOutput:
    latent: torch.Tensor  # (B, latent_dim)
    policy_logits: torch.Tensor  # (B, action_space)
    value: torch.Tensor  # (B,)
    reward: torch.Tensor  # (B,)


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, n_layers: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MuZeroNet(nn.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_space: int,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        action_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_space = int(action_space)
        self.latent_dim = int(latent_dim)

        self.representation = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.latent_dim),
            nn.Tanh(),
        )

        self.action_embed = nn.Embedding(self.action_space, action_embed_dim)

        self.dynamics = _MLP(self.latent_dim + action_embed_dim, hidden_dim, self.latent_dim, n_layers=3)
        self.reward_head = _MLP(self.latent_dim + action_embed_dim, hidden_dim, 1, n_layers=2)

        self.policy_head = _MLP(self.latent_dim, hidden_dim, self.action_space, n_layers=2)
        self.value_head = _MLP(self.latent_dim, hidden_dim, 1, n_layers=2)

    def initial_inference(self, obs: torch.Tensor) -> InferenceOutput:
        if obs.ndim == 1:
            obs = obs[None, :]
        latent = self.representation(obs)
        policy_logits = self.policy_head(latent)
        value = self.value_head(latent).squeeze(-1)
        reward = torch.zeros_like(value)
        return InferenceOutput(latent=latent, policy_logits=policy_logits, value=value, reward=reward)

    def recurrent_inference(self, latent: torch.Tensor, action: torch.Tensor) -> InferenceOutput:
        if action.ndim == 0:
            action = action[None]
        if latent.ndim == 1:
            latent = latent[None, :]
        aemb = self.action_embed(action.to(torch.long))
        x = torch.cat([latent, aemb], dim=-1)
        next_latent = torch.tanh(self.dynamics(x))
        reward = self.reward_head(x).squeeze(-1)
        policy_logits = self.policy_head(next_latent)
        value = self.value_head(next_latent).squeeze(-1)
        return InferenceOutput(latent=next_latent, policy_logits=policy_logits, value=value, reward=reward)

    @staticmethod
    def policy_loss_from_sparse_targets(
        logits: torch.Tensor,
        target_action_ids: torch.Tensor,
        target_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-entropy with sparse targets.

        - logits: (B, A)
        - target_action_ids: (B, K) with -1 padding
        - target_probs: (B, K) probs aligned with ids (0 where padded)
        """
        logp = F.log_softmax(logits, dim=-1)
        ids = target_action_ids.to(torch.long)
        valid = ids >= 0
        safe_ids = torch.where(valid, ids, torch.zeros_like(ids))
        gathered = torch.gather(logp, dim=1, index=safe_ids)
        loss = -(gathered * target_probs) * valid.to(gathered.dtype)
        return loss.sum(dim=1).mean()

