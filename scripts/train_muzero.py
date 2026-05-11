from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
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
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config (MuZeroConfig fields).")
    parser.add_argument("--preset", choices=["fast", "medium", "full"], default="medium", help="Convenience presets for speed/strength.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override MuZeroConfig field, e.g. --set num_simulations=32 (repeatable).",
    )
    parser.add_argument("--games-per-iter", type=int, default=None)
    parser.add_argument("--train-steps-per-iter", type=int, default=None)
    parser.add_argument("--num-sims", type=int, default=None)
    parser.add_argument("--min-replay-games", type=int, default=None)
    parser.add_argument("--eval-games", type=int, default=0, help="Optional evaluation games per iteration (no noise, temp=0).")
    parser.add_argument("--plot-every", type=int, default=1, help="Write `.tmp/muzero/metrics.png` every N iterations (0 disables).")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    args = parser.parse_args()

    cfg = _load_config(args.config) if args.config is not None else MuZeroConfig()
    cfg = _apply_preset(cfg, args.preset)
    cfg = _apply_overrides(cfg, args.overrides)
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
    metrics_path = out_dir / "metrics.csv"

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

    print(
        f"device={device}  obs_dim={game.obs_dim}  action_space={game.action_space_size}  per_robot={game.n_per_robot}",
        flush=True,
    )
    print(f"cfg={asdict(cfg)}", flush=True)

    total_games = 0
    t0 = time.time()
    for it in range(int(args.iterations)):
        it_start = time.time()

        # Self-play (training distribution)
        net.eval()
        sp_stats = _SelfPlayStats()
        t_sp = time.time()
        for _ in _progress_range(int(cfg.games_per_iteration), enabled=(not args.no_progress), desc=f"selfplay it={it:04d}"):
            seed = int(rng.integers(0, 2**31 - 1))
            hist = play_selfplay_game(game=game, net=net, config=cfg, seed=seed, device=device)
            replay.add_game(hist)
            total_games += 1
            _accumulate_selfplay_stats(sp_stats, game=game, hist=hist)
        selfplay_s = time.time() - t_sp

        # Train (updates)
        if len(replay) < int(cfg.min_replay_games):
            row = _make_metrics_row(
                it=it,
                total_games=total_games,
                replay_games=len(replay),
                selfplay=sp_stats,
                train=None,
                elapsed_s=time.time() - t0,
                iter_s=time.time() - it_start,
                selfplay_s=selfplay_s,
                train_s=0.0,
                eval_s=0.0,
            )
            _append_metrics(metrics_path, row)
            _maybe_plot_metrics(metrics_path, out_dir / "metrics.png", every=int(args.plot_every), it=it)
            print(f"[it {it:04d}] replay games={len(replay)} (warming up)  selfplay: {sp_stats.summary()}", flush=True)
            _save_checkpoint(out_dir, it=it, net=net, opt=opt, cfg=cfg)
            continue

        t_train = time.time()
        stats = _train_many(net=net, opt=opt, replay=replay, cfg=cfg, device=device, progress=(not args.no_progress))
        train_s = time.time() - t_train

        eval_stats = None
        eval_s = 0.0
        if int(args.eval_games) > 0:
            t_eval = time.time()
            eval_stats = _evaluate(net=net, base_game=game, cfg=cfg, rng=rng, device=device, n_games=int(args.eval_games), progress=(not args.no_progress), it=it)
            eval_s = time.time() - t_eval

        row = _make_metrics_row(
            it=it,
            total_games=total_games,
            replay_games=len(replay),
            selfplay=sp_stats,
            train=stats,
            elapsed_s=time.time() - t0,
            iter_s=time.time() - it_start,
            selfplay_s=selfplay_s,
            train_s=train_s,
            eval_s=eval_s,
            eval_stats=eval_stats,
        )
        _append_metrics(metrics_path, row)
        _maybe_plot_metrics(metrics_path, out_dir / "metrics.png", every=int(args.plot_every), it=it)

        msg = (
            f"[it {it:04d}] games={total_games} replay={len(replay)} "
            f"loss={stats.loss:.4f} (v={stats.value_loss:.4f} r={stats.reward_loss:.4f} p={stats.policy_loss:.4f}) "
            f"selfplay: {sp_stats.summary()}  "
            f"time(s): sp={selfplay_s:.1f} train={train_s:.1f} eval={eval_s:.1f} iter={row['iter_s']:.1f}  elapsed={row['elapsed_s']:.1f}"
        )
        if eval_stats is not None:
            msg += f"  eval: {eval_stats.summary()}"
        print(msg, flush=True)

        _save_checkpoint(out_dir, it=it, net=net, opt=opt, cfg=cfg)

    return 0


def _train_many(
    *, net: MuZeroNet, opt: torch.optim.Optimizer, replay: ReplayBuffer, cfg: MuZeroConfig, device: torch.device, progress: bool
) -> TrainStats:
    agg = TrainStats(loss=0.0, value_loss=0.0, reward_loss=0.0, policy_loss=0.0)
    n = int(cfg.train_steps_per_iteration)
    for _ in _progress_range(n, enabled=progress, desc="train"):
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


class _SelfPlayStats:
    def __init__(self) -> None:
        self.n_games = 0
        self.n_moves = 0
        self.red_total_sum = 0.0
        self.blue_total_sum = 0.0
        self.diff_sum = 0.0
        self.red_win = 0
        self.blue_win = 0
        self.tie = 0
        self.mean_entropy_sum = 0.0

    def summary(self) -> str:
        n = max(1, int(self.n_games))
        return (
            f"n={self.n_games} moves/g={self.n_moves/max(1,n):.1f} "
            f"diff={self.diff_sum/n:+.2f} win%_R={100.0*self.red_win/n:.1f} "
            f"Hπ={self.mean_entropy_sum/n:.2f}"
        )


def _accumulate_selfplay_stats(stats: _SelfPlayStats, *, game: RebuiltTurnBasedGame, hist) -> None:
    state = game.sim.state
    if state is None:
        return
    red_total = int(state.score[0] + state.penalty_points[0])
    blue_total = int(state.score[1] + state.penalty_points[1])
    diff = red_total - blue_total

    stats.n_games += 1
    stats.n_moves += int(hist.length)
    stats.red_total_sum += float(red_total)
    stats.blue_total_sum += float(blue_total)
    stats.diff_sum += float(diff)
    if diff > 0:
        stats.red_win += 1
    elif diff < 0:
        stats.blue_win += 1
    else:
        stats.tie += 1

    # Policy entropy (mean over moves in the game)
    ent = 0.0
    for p in hist.policy_probs:
        pp = np.asarray(p, dtype=np.float32)
        mask = pp > 0
        if not np.any(mask):
            continue
        ent += float(-np.sum(pp[mask] * np.log(pp[mask] + 1e-8)))
    if hist.length > 0:
        stats.mean_entropy_sum += ent / float(hist.length)


class _EvalStats(_SelfPlayStats):
    def summary(self) -> str:
        n = max(1, int(self.n_games))
        return f"n={self.n_games} diff={self.diff_sum/n:+.2f} win%_R={100.0*self.red_win/n:.1f}"


def _evaluate(
    *,
    net: MuZeroNet,
    base_game: RebuiltTurnBasedGame,
    cfg: MuZeroConfig,
    rng: np.random.Generator,
    device: torch.device,
    n_games: int,
    progress: bool,
    it: int,
) -> _EvalStats:
    # Use a fresh game instance to avoid any hidden state issues.
    g = RebuiltTurnBasedGame(config=base_game.sim.config, robot_specs=base_game.sim.robot_specs, seed=int(rng.integers(0, 2**31 - 1)))
    stats = _EvalStats()
    # Temporarily disable exploration.
    eval_cfg = replace(cfg, dirichlet_fraction=0.0, temperature=0.0)
    for _ in _progress_range(n_games, enabled=progress, desc=f"eval it={it:04d}"):
        seed = int(rng.integers(0, 2**31 - 1))
        hist = play_selfplay_game(game=g, net=net, config=eval_cfg, seed=seed, device=device)
        _accumulate_selfplay_stats(stats, game=g, hist=hist)
    return stats


def _append_metrics(path: Path, row: dict[str, float | int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    keys = list(row.keys())
    with path.open("a", encoding="utf-8") as f:
        if not exists:
            f.write(",".join(keys) + "\n")
        f.write(",".join(str(row[k]) for k in keys) + "\n")


def _maybe_plot_metrics(metrics_csv: Path, out_png: Path, *, every: int, it: int) -> None:
    if every <= 0:
        return
    if it % every != 0:
        return
    try:
        import matplotlib.pyplot as plt

        data = _read_metrics(metrics_csv)
        if not data:
            return

        iters = np.asarray(data["it"], dtype=np.int32)
        fig, axs = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

        axs[0].plot(iters, data.get("loss", []), label="loss")
        axs[0].plot(iters, data.get("value_loss", []), label="value")
        axs[0].plot(iters, data.get("reward_loss", []), label="reward")
        axs[0].plot(iters, data.get("policy_loss", []), label="policy")
        axs[0].set_ylabel("train loss")
        axs[0].grid(True, alpha=0.3)
        axs[0].legend(loc="upper right")

        axs[1].plot(iters, data.get("selfplay_diff", []), label="selfplay diff (R-B)")
        if "eval_diff" in data:
            axs[1].plot(iters, data.get("eval_diff", []), label="eval diff (R-B)")
        axs[1].set_ylabel("score diff")
        axs[1].grid(True, alpha=0.3)
        axs[1].legend(loc="upper right")

        axs[2].plot(iters, data.get("selfplay_win_rate_red", []), label="selfplay win% red")
        if "eval_win_rate_red" in data:
            axs[2].plot(iters, data.get("eval_win_rate_red", []), label="eval win% red")
        axs[2].plot(iters, data.get("selfplay_entropy", []), label="selfplay policy entropy")
        axs[2].set_ylabel("win% / entropy")
        axs[2].set_xlabel("iteration")
        axs[2].grid(True, alpha=0.3)
        axs[2].legend(loc="upper right")

        axs[3].plot(iters, data.get("selfplay_s", []), label="selfplay_s")
        axs[3].plot(iters, data.get("train_s", []), label="train_s")
        axs[3].plot(iters, data.get("eval_s", []), label="eval_s")
        axs[3].set_ylabel("seconds")
        axs[3].set_xlabel("iteration")
        axs[3].grid(True, alpha=0.3)
        axs[3].legend(loc="upper right")

        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
    except Exception:
        return


def _read_metrics(path: Path) -> dict[str, list[float]]:
    txt = path.read_text(encoding="utf-8").strip().splitlines()
    if len(txt) < 2:
        return {}
    header = txt[0].split(",")
    cols = {h: [] for h in header}
    for line in txt[1:]:
        parts = line.split(",")
        if len(parts) != len(header):
            continue
        for h, v in zip(header, parts, strict=True):
            try:
                cols[h].append(float(v))
            except Exception:
                pass
    return cols


def _make_metrics_row(
    *,
    it: int,
    total_games: int,
    replay_games: int,
    selfplay: _SelfPlayStats,
    train: TrainStats | None,
    elapsed_s: float,
    iter_s: float,
    selfplay_s: float,
    train_s: float,
    eval_s: float,
    eval_stats: _EvalStats | None = None,
) -> dict[str, float | int]:
    n = max(1, int(selfplay.n_games))
    row: dict[str, float | int] = {
        "it": int(it),
        "total_games": int(total_games),
        "replay_games": int(replay_games),
        "selfplay_games": int(selfplay.n_games),
        "selfplay_moves_per_game": float(selfplay.n_moves / n),
        "selfplay_diff": float(selfplay.diff_sum / n),
        "selfplay_win_rate_red": float(selfplay.red_win / n),
        "selfplay_entropy": float(selfplay.mean_entropy_sum / n),
        "elapsed_s": float(elapsed_s),
        "iter_s": float(iter_s),
        "selfplay_s": float(selfplay_s),
        "train_s": float(train_s),
        "eval_s": float(eval_s),
    }
    if train is not None:
        row.update(
            {
                "loss": float(train.loss),
                "value_loss": float(train.value_loss),
                "reward_loss": float(train.reward_loss),
                "policy_loss": float(train.policy_loss),
            }
        )
    if eval_stats is not None:
        en = max(1, int(eval_stats.n_games))
        row.update(
            {
                "eval_games": int(eval_stats.n_games),
                "eval_diff": float(eval_stats.diff_sum / en),
                "eval_win_rate_red": float(eval_stats.red_win / en),
            }
        )
    return row


def _load_config(path: Path) -> MuZeroConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config JSON must be an object with MuZeroConfig fields")
    return MuZeroConfig(**data)


def _apply_overrides(cfg: MuZeroConfig, overrides: list[str]) -> MuZeroConfig:
    if not overrides:
        return cfg
    d = asdict(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        if k not in d:
            raise ValueError(f"unknown config field {k!r}")
        cur = d[k]
        d[k] = _parse_value(v.strip(), cur)
    return MuZeroConfig(**d)


def _apply_preset(cfg: MuZeroConfig, preset: str) -> MuZeroConfig:
    preset = str(preset).lower().strip()
    if preset == "medium":
        return cfg
    if preset == "fast":
        return replace(
            cfg,
            num_simulations=16,
            mcts_batch_size=8,
            max_policy_actions=min(int(cfg.max_policy_actions), 48),
            latent_dim=min(int(cfg.latent_dim), 64),
            hidden_dim=min(int(cfg.hidden_dim), 128),
            unroll_steps=min(int(cfg.unroll_steps), 6),
            train_steps_per_iteration=min(int(cfg.train_steps_per_iteration), 40),
            games_per_iteration=min(int(cfg.games_per_iteration), 2),
        )
    if preset == "full":
        return replace(
            cfg,
            num_simulations=max(int(cfg.num_simulations), 96),
            mcts_batch_size=max(int(cfg.mcts_batch_size), 24),
            max_policy_actions=max(int(cfg.max_policy_actions), 64),
            latent_dim=max(int(cfg.latent_dim), 128),
            hidden_dim=max(int(cfg.hidden_dim), 256),
            unroll_steps=max(int(cfg.unroll_steps), 10),
            train_steps_per_iteration=max(int(cfg.train_steps_per_iteration), 200),
            games_per_iteration=max(int(cfg.games_per_iteration), 10),
        )
    raise ValueError(f"unknown preset {preset!r}")


def _parse_value(s: str, cur: object) -> object:
    if isinstance(cur, bool):
        if s.lower() in ("1", "true", "yes", "y", "on"):
            return True
        if s.lower() in ("0", "false", "no", "n", "off"):
            return False
        raise ValueError(f"invalid bool: {s!r}")
    if isinstance(cur, int) and not isinstance(cur, bool):
        return int(float(s))
    if isinstance(cur, float):
        return float(s)
    if isinstance(cur, str):
        return s
    return s


def _progress_range(n: int, *, enabled: bool, desc: str):
    if not enabled:
        return range(n)
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(range(n), desc=desc, leave=False)
    except Exception:
        return range(n)


if __name__ == "__main__":
    raise SystemExit(main())
