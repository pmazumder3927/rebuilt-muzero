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
    rng: np.random.Generator,
    temperature: float,
    add_exploration_noise: bool,
    device: torch.device,
) -> MCTSOutput:
    """
    Run MuZero-style PUCT MCTS from a root observation.

    The action space is fixed; per-step legality is handled by the simulator at
    execution time. Values and rewards are from the POV of the player to act
    at `obs`. The backup rule is: v <- r + discount * (-v_next).
    """
    net.eval()
    obs_t = torch.from_numpy(obs).to(device=device, dtype=torch.float32)

    with torch.inference_mode():
        root_out = net.initial_inference(obs_t)

    root_latent = root_out.latent.squeeze(0)
    root_value = float(root_out.value.squeeze(0).item())

    root_action_ids, root_priors = _topk_priors_full(policy_logits=root_out.policy_logits.squeeze(0), k=int(config.max_policy_actions))

    if add_exploration_noise and root_action_ids.size > 0 and float(config.dirichlet_fraction) > 0.0:
        noise = rng.dirichlet([float(config.dirichlet_alpha)] * int(root_action_ids.size)).astype(np.float32, copy=False)
        frac = float(config.dirichlet_fraction)
        root_priors = (1.0 - frac) * root_priors + frac * noise
        root_priors = root_priors / (float(np.sum(root_priors)) + 1e-8)

    root = _Node(latent=root_latent, reward=0.0, action_ids=root_action_ids, priors=root_priors)

    n_sims = int(config.num_simulations)
    batch = max(1, int(config.mcts_batch_size))
    batch = min(batch, n_sims)
    k = int(config.max_policy_actions)

    for batch_start in range(0, n_sims, batch):
        cur = min(batch, n_sims - batch_start)
        # Collect expansions for this batch.
        paths: list[list[tuple[_Node, int]]] = []
        exp_nodes: list[_Node] = []
        exp_edge_idxs: list[int] = []
        exp_action_ids: list[int] = []
        exp_map: dict[tuple[_Node, int], int] = {}

        for _ in range(cur):
            node = root
            path: list[tuple[_Node, int]] = []
            while True:
                if node.action_ids.size == 0:
                    break
                a_idx = _select_action_puct(node=node, c_puct=float(config.c_puct))
                # Virtual visit to diversify within the batch (kept as real).
                node.visit_counts[a_idx] += 1
                path.append((node, a_idx))
                child = node.children[a_idx]
                if child is None:
                    key = (node, a_idx)
                    if key not in exp_map:
                        exp_map[key] = len(exp_nodes)
                        exp_nodes.append(node)
                        exp_edge_idxs.append(a_idx)
                        exp_action_ids.append(int(node.action_ids[a_idx]))
                    break
                node = child
            paths.append(path)

        if exp_nodes:
            latents = torch.stack([n.latent for n in exp_nodes], dim=0)
            acts = torch.tensor(exp_action_ids, device=device, dtype=torch.long)
            with torch.inference_mode():
                out = net.recurrent_inference(latents, acts)

            # Top-k priors for all expanded leaves in one shot.
            topv, topi = torch.topk(out.policy_logits, k=min(k, out.policy_logits.shape[1]), dim=1)
            priors = torch.softmax(topv, dim=1)

            leaf_values = out.value.detach().cpu().numpy().astype(np.float32, copy=False)

            for i, (parent, edge_idx) in enumerate(zip(exp_nodes, exp_edge_idxs, strict=True)):
                if parent.children[edge_idx] is not None:
                    continue
                child = _Node(
                    latent=out.latent[i],
                    reward=float(out.reward[i].item()),
                    action_ids=topi[i].detach().cpu().numpy().astype(np.int32, copy=False),
                    priors=priors[i].detach().cpu().numpy().astype(np.float32, copy=False),
                )
                parent.children[edge_idx] = child

        # Backup for each simulation in this batch.
        for path in paths:
            if not path:
                continue
            last_node, last_edge = path[-1]
            leaf_idx = exp_map.get((last_node, last_edge))
            value = float(leaf_values[leaf_idx]) if leaf_idx is not None else root_value
            for parent, edge_idx in reversed(path):
                child = parent.children[edge_idx]
                if child is None:
                    break
                value = float(child.reward) + float(config.discount) * (-value)
                parent.value_sums[edge_idx] += float(value)

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

def _topk_priors_full(*, policy_logits: torch.Tensor, k: int) -> tuple[np.ndarray, np.ndarray]:
    if policy_logits.numel() <= 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    kk = min(int(k), int(policy_logits.shape[0]))
    if kk <= 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    topv, topi = torch.topk(policy_logits, k=kk)
    priors = torch.softmax(topv, dim=0).detach().cpu().numpy().astype(np.float32, copy=False)
    action_ids = topi.detach().cpu().numpy().astype(np.int32, copy=False)
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
