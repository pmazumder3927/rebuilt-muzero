from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rebuilt_muzero.muzero.config import MuZeroConfig
from rebuilt_muzero.muzero.networks import MuZeroNet


@dataclass(slots=True)
class PolicyTarget:
    action_ids: np.ndarray  # (K,) int32, -1 padded
    probs: np.ndarray  # (K,) float32, sums to 1 over valid ids


@dataclass(slots=True)
class MCTSOutput:
    action: int
    policy: PolicyTarget
    root_value: float


class _Node:
    __slots__ = ("latent", "reward", "action_ids", "priors", "visit_counts", "value_sums", "children")

    def __init__(
        self,
        *,
        latent: torch.Tensor,
        reward: float,
        action_ids: np.ndarray,
        priors: np.ndarray,
    ) -> None:
        self.latent = latent  # (latent_dim,)
        self.reward = float(reward)
        self.action_ids = action_ids.astype(np.int32, copy=False)
        self.priors = priors.astype(np.float32, copy=False)
        self.visit_counts = np.zeros_like(self.priors, dtype=np.int32)
        self.value_sums = np.zeros_like(self.priors, dtype=np.float32)
        self.children: list[_Node | None] = [None] * int(self.action_ids.shape[0])

    def expanded(self) -> bool:
        return self.action_ids.size > 0

    def q_values(self) -> np.ndarray:
        q = np.zeros_like(self.value_sums, dtype=np.float32)
        mask = self.visit_counts > 0
        q[mask] = self.value_sums[mask] / self.visit_counts[mask].astype(np.float32)
        return q


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    m = np.max(x) if x.size else 0.0
    e = np.exp(x - m)
    s = float(np.sum(e)) + 1e-8
    return (e / s).astype(np.float32, copy=False)


def run_mcts(
    *,
    net: MuZeroNet,
    config: MuZeroConfig,
    obs: np.ndarray,
    legal_actions: np.ndarray,
    rng: np.random.Generator,
    temperature: float,
    add_exploration_noise: bool,
    device: torch.device,
) -> MCTSOutput:
    """
    Run MuZero-style PUCT MCTS from a root observation.

    Values are from the POV of the player to act at `obs`. Rewards are also from that POV.
    The backup rule uses: v <- r + discount * (-v_next).
    """
    net.eval()
    obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        root_out = net.initial_inference(obs_t)

    root_latent = root_out.latent.squeeze(0)
    root_value = float(root_out.value.squeeze(0).item())

    root_action_ids, root_priors = _select_topk_priors(
        policy_logits=root_out.policy_logits.squeeze(0),
        legal_actions=legal_actions,
        k=int(config.max_policy_actions),
    )

    if add_exploration_noise and root_action_ids.size > 0:
        noise = rng.dirichlet([float(config.dirichlet_alpha)] * int(root_action_ids.size)).astype(np.float32, copy=False)
        frac = float(config.dirichlet_fraction)
        root_priors = (1.0 - frac) * root_priors + frac * noise
        root_priors = root_priors / (float(np.sum(root_priors)) + 1e-8)

    root = _Node(latent=root_latent, reward=0.0, action_ids=root_action_ids, priors=root_priors)

    for _ in range(int(config.num_simulations)):
        node = root
        search_path: list[tuple[_Node, int]] = []

        # Traverse
        while True:
            if node.action_ids.size == 0:
                break
            a_idx = _select_action_puct(node=node, c_puct=float(config.c_puct))
            search_path.append((node, a_idx))
            child = node.children[a_idx]
            if child is None:
                # Expand new leaf via model.
                act = int(node.action_ids[a_idx])
                with torch.no_grad():
                    out = net.recurrent_inference(node.latent, torch.tensor(act, device=device))
                child_action_ids, child_priors = _select_topk_priors(
                    policy_logits=out.policy_logits.squeeze(0),
                    legal_actions=legal_actions,  # fixed action-space (busy robots handled by env, not model)
                    k=int(config.max_policy_actions),
                )
                child = _Node(
                    latent=out.latent.squeeze(0),
                    reward=float(out.reward.squeeze(0).item()),
                    action_ids=child_action_ids,
                    priors=child_priors,
                )
                node.children[a_idx] = child
                leaf_value = float(out.value.squeeze(0).item())
                break
            node = child

        # Backup
        value = leaf_value if "leaf_value" in locals() else root_value
        for parent, a_idx in reversed(search_path):
            reward = 0.0
            child = parent.children[a_idx]
            if child is not None:
                reward = float(child.reward)
            value = reward + float(config.discount) * (-value)
            parent.visit_counts[a_idx] += 1
            parent.value_sums[a_idx] += float(value)

        if "leaf_value" in locals():
            del leaf_value

    # Build policy target from root visits.
    counts = root.visit_counts.astype(np.float32)
    if float(np.sum(counts)) <= 0:
        # Fallback to priors
        probs = root.priors.astype(np.float32, copy=False)
    else:
        probs = (counts / (float(np.sum(counts)) + 1e-8)).astype(np.float32, copy=False)

    # Pick action.
    action = _sample_action(root.action_ids, root.visit_counts, rng=rng, temperature=temperature)

    # Root value estimate from visit-weighted Q.
    q = root.q_values()
    if float(np.sum(root.visit_counts)) > 0:
        root_value_est = float(np.sum(q * root.visit_counts.astype(np.float32)) / float(np.sum(root.visit_counts)))
    else:
        root_value_est = root_value

    # Return sparse policy target.
    return MCTSOutput(
        action=int(action),
        policy=PolicyTarget(action_ids=root.action_ids.copy(), probs=probs.copy()),
        root_value=float(root_value_est),
    )


def _select_topk_priors(*, policy_logits: torch.Tensor, legal_actions: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (action_ids, priors) for the top-k legal actions by logit.
    """
    if legal_actions.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    la = torch.from_numpy(legal_actions.astype(np.int64, copy=False)).to(policy_logits.device)
    logits = policy_logits.index_select(0, la)
    if logits.numel() > k:
        topv, topi = torch.topk(logits, k=k)
        chosen = la.index_select(0, topi)
        priors = torch.softmax(topv, dim=0).cpu().numpy().astype(np.float32, copy=False)
        action_ids = chosen.cpu().numpy().astype(np.int32, copy=False)
        return action_ids, priors

    priors = torch.softmax(logits, dim=0).cpu().numpy().astype(np.float32, copy=False)
    action_ids = legal_actions.astype(np.int32, copy=False)
    return action_ids, priors


def _select_action_puct(*, node: _Node, c_puct: float) -> int:
    total = float(np.sum(node.visit_counts)) + 1e-8
    q = node.q_values()
    u = c_puct * node.priors * np.sqrt(total) / (1.0 + node.visit_counts.astype(np.float32))
    score = q + u
    return int(np.argmax(score))


def _sample_action(action_ids: np.ndarray, visit_counts: np.ndarray, *, rng: np.random.Generator, temperature: float) -> int:
    if action_ids.size == 0:
        return 0
    counts = visit_counts.astype(np.float32)
    if temperature <= 1e-6:
        return int(action_ids[int(np.argmax(counts))])
    # Temperature sampling on counts.
    p = counts ** (1.0 / float(temperature))
    s = float(np.sum(p))
    if s <= 0:
        return int(action_ids[int(rng.integers(0, action_ids.size))])
    p = p / s
    idx = int(rng.choice(np.arange(action_ids.size), p=p))
    return int(action_ids[idx])
