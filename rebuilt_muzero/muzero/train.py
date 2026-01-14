from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from rebuilt_muzero.muzero.config import MuZeroConfig
from rebuilt_muzero.muzero.networks import MuZeroNet
from rebuilt_muzero.muzero.replay import ReplayBuffer


@dataclass(slots=True)
class TrainStats:
    loss: float
    value_loss: float
    reward_loss: float
    policy_loss: float


def train_step(
    *,
    net: MuZeroNet,
    optimizer: torch.optim.Optimizer,
    replay: ReplayBuffer,
    config: MuZeroConfig,
    device: torch.device,
) -> TrainStats:
    batch = replay.sample_batch(
        batch_size=int(config.batch_size),
        unroll_steps=int(config.unroll_steps),
        td_steps=int(config.td_steps),
        discount=float(config.discount),
        policy_k=int(config.max_policy_actions),
    )

    obs0 = torch.from_numpy(batch["obs0"]).to(device=device, dtype=torch.float32)
    actions = torch.from_numpy(batch["actions"]).to(device=device, dtype=torch.long)
    reward_targets = torch.from_numpy(batch["rewards"]).to(device=device, dtype=torch.float32)
    value_targets = torch.from_numpy(batch["value_targets"]).to(device=device, dtype=torch.float32)
    policy_action_ids = torch.from_numpy(batch["policy_action_ids"]).to(device=device, dtype=torch.long)
    policy_probs = torch.from_numpy(batch["policy_probs"]).to(device=device, dtype=torch.float32)
    valid_steps = torch.from_numpy(batch["valid_steps"]).to(device=device, dtype=torch.float32)
    valid_rewards = torch.from_numpy(batch["valid_rewards"]).to(device=device, dtype=torch.float32)

    B = obs0.shape[0]
    K = int(config.unroll_steps)

    net.train()
    optimizer.zero_grad(set_to_none=True)

    out0 = net.initial_inference(obs0)
    lat = out0.latent
    policy_logits = [out0.policy_logits]
    value_preds = [out0.value]
    reward_preds: list[torch.Tensor] = []

    for k in range(K):
        outk = net.recurrent_inference(lat, actions[:, k])
        lat = outk.latent
        reward_preds.append(outk.reward)
        policy_logits.append(outk.policy_logits)
        value_preds.append(outk.value)

    value_pred = torch.stack(value_preds, dim=1)  # (B, K+1)
    reward_pred = torch.stack(reward_preds, dim=1)  # (B, K)

    # Losses with masks.
    value_loss = _masked_huber(value_pred, value_targets[:, : K + 1], valid_steps[:, : K + 1])
    reward_loss = _masked_huber(reward_pred, reward_targets[:, :K], valid_rewards[:, :K])

    policy_loss_total = torch.zeros((), device=device)
    policy_weight_total = torch.zeros((), device=device)
    for k in range(K + 1):
        w = valid_steps[:, k]  # (B,)
        if torch.sum(w).item() <= 0:
            continue
        per = _sparse_policy_ce_per_sample(policy_logits[k], policy_action_ids[:, k, :], policy_probs[:, k, :])
        policy_loss_total = policy_loss_total + torch.sum(per * w)
        policy_weight_total = policy_weight_total + torch.sum(w)
    policy_loss = policy_loss_total / torch.clamp(policy_weight_total, min=1.0)

    loss = (
        float(config.value_loss_weight) * value_loss
        + float(config.reward_loss_weight) * reward_loss
        + float(config.policy_loss_weight) * policy_loss
    )

    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
    optimizer.step()

    return TrainStats(
        loss=float(loss.detach().cpu().item()),
        value_loss=float(value_loss.detach().cpu().item()),
        reward_loss=float(reward_loss.detach().cpu().item()),
        policy_loss=float(policy_loss.detach().cpu().item()),
    )


def _masked_huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # SmoothL1 (Huber) per element, average over valid entries.
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    loss = loss * mask
    denom = torch.clamp(torch.sum(mask), min=1.0)
    return torch.sum(loss) / denom


def _sparse_policy_ce_per_sample(
    logits: torch.Tensor,
    action_ids: torch.Tensor,
    target_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Per-sample sparse cross-entropy:
      L = - sum_i p_i * log softmax(logits)[a_i]

    Padding uses action_ids == -1.
    """
    logp = F.log_softmax(logits, dim=-1)
    ids = action_ids
    valid = ids >= 0
    safe_ids = torch.where(valid, ids, torch.zeros_like(ids))
    gathered = torch.gather(logp, dim=1, index=safe_ids)
    contrib = -(gathered * target_probs) * valid.to(gathered.dtype)
    return torch.sum(contrib, dim=1)


def preferred_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_optimizer(net: MuZeroNet, config: MuZeroConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(net.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))

