from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rebuilt_muzero.muzero.config import MuZeroConfig  # noqa: E402
from rebuilt_muzero.muzero.game import RebuiltTurnBasedGame  # noqa: E402
from rebuilt_muzero.muzero.replay import ReplayBuffer  # noqa: E402
from rebuilt_muzero.muzero.selfplay import play_selfplay_game  # noqa: E402
from rebuilt_muzero.muzero.train import TrainStats, make_optimizer, preferred_device, train_step  # noqa: E402
from rebuilt_muzero.muzero.networks import MuZeroNet  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="MuZero self-play training for REBUILT macro-sim (turn-based 3v3).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--out-dir", type=Path, default=Path(".tmp/muzero"))
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument("--games-per-iter", type=int, default=None)
    parser.add_argument("--train-steps-per-iter", type=int, default=None)
    parser.add_argument("--num-sims", type=int, default=None)
    parser.add_argument("--min-replay-games", type=int, default=None)
    args = parser.parse_args()

    cfg = MuZeroConfig()
    if args.games_per_iter is not None:
        cfg = replace(cfg, games_per_iteration=int(args.games_per_iter))
    if args.train_steps_per_iter is not None:
        cfg = replace(cfg, train_steps_per_iteration=int(args.train_steps_per_iter))
    if args.num_sims is not None:
        cfg = replace(cfg, num_simulations=int(args.num_sims))
    if args.min_replay_games is not None:
        cfg = replace(cfg, min_replay_games=int(args.min_replay_games))

    if args.device == "auto":
        device = preferred_device()
    else:
        device = torch.device(args.device)

    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    game = RebuiltTurnBasedGame(seed=args.seed)
    net = MuZeroNet(
        obs_dim=game.obs_dim,
        action_space=game.action_space_size,
        latent_dim=int(cfg.latent_dim),
        hidden_dim=int(cfg.hidden_dim),
        action_embed_dim=int(cfg.action_embed_dim),
    ).to(device=device)

    opt = make_optimizer(net, cfg)
    replay = ReplayBuffer(capacity_games=int(cfg.replay_capacity_games), rng=rng)

    print(f"device={device}  obs_dim={game.obs_dim}  action_space={game.action_space_size}  per_robot={game.n_per_robot}")

    total_games = 0
    t0 = time.time()
    for it in range(int(args.iterations)):
        # Self-play
        net.eval()
        for g in range(int(cfg.games_per_iteration)):
            seed = int(rng.integers(0, 2**31 - 1))
            hist = play_selfplay_game(game=game, net=net, config=cfg, seed=seed, device=device)
            replay.add_game(hist)
            total_games += 1

        # Train
        if len(replay) < int(cfg.min_replay_games):
            print(f"[it {it:04d}] replay games={len(replay)} (warming up)")
            _save_checkpoint(out_dir, it=it, net=net, opt=opt, cfg=cfg)
            continue

        stats = _train_many(net=net, opt=opt, replay=replay, cfg=cfg, device=device)
        dt = time.time() - t0
        print(
            f"[it {it:04d}] games={total_games} replay={len(replay)} "
            f"loss={stats.loss:.4f} (v={stats.value_loss:.4f} r={stats.reward_loss:.4f} p={stats.policy_loss:.4f}) "
            f"elapsed={dt:.1f}s"
        )

        _save_checkpoint(out_dir, it=it, net=net, opt=opt, cfg=cfg)

    return 0


def _train_many(*, net: MuZeroNet, opt: torch.optim.Optimizer, replay: ReplayBuffer, cfg: MuZeroConfig, device: torch.device) -> TrainStats:
    agg = TrainStats(loss=0.0, value_loss=0.0, reward_loss=0.0, policy_loss=0.0)
    n = int(cfg.train_steps_per_iteration)
    for _ in range(n):
        s = train_step(net=net, optimizer=opt, replay=replay, config=cfg, device=device)
        agg.loss += s.loss
        agg.value_loss += s.value_loss
        agg.reward_loss += s.reward_loss
        agg.policy_loss += s.policy_loss
    agg.loss /= max(1, n)
    agg.value_loss /= max(1, n)
    agg.reward_loss /= max(1, n)
    agg.policy_loss /= max(1, n)
    return agg


def _save_checkpoint(out_dir: Path, *, it: int, net: MuZeroNet, opt: torch.optim.Optimizer, cfg: MuZeroConfig) -> None:
    ckpt = {
        "it": int(it),
        "cfg": asdict(cfg),
        "net": net.state_dict(),
        "opt": opt.state_dict(),
    }
    path = out_dir / "latest.pt"
    torch.save(ckpt, path)
    if it % 10 == 0:
        torch.save(ckpt, out_dir / f"it_{it:04d}.pt")


if __name__ == "__main__":
    raise SystemExit(main())
