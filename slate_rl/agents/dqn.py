"""Item-wise DQN for slates (CleanRL-style, single file).

Each candidate gets a value q(s, i); the slate is the top-k by q and its value is the mean,
Q(s, A) = mean_{i in A} q(s, i). Targets use double DQN (online net picks the next slate,
target net scores it), n-step returns, and Polyak-averaged target weights. This decomposition
ignores how slate items compete for the user's single click; SlateQ (next agent) fixes that.
"""

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from slate_rl.agents.common import CandidateEncoder, CompactObs, encoder_kwargs
from slate_rl.agents.replay import NStepAccumulator, ReplayBuffer
from slate_rl.envs.batched import SlateObs


@dataclass
class DQNConfig:
    total_iters: int = 4000  # env steps per parallel env
    buffer_size: int = 500_000
    batch_size: int = 512
    lr: float = 3e-4
    gamma: float = 0.95
    n_step: int = 3
    tau: float = 0.005
    learning_starts: int = 20_000  # transitions collected before the first update
    updates_per_iter: int = 2
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.3  # share of training over which epsilon decays
    hidden: int = 128
    grad_clip: float = 10.0
    eval_interval: int = 250  # iterations between evaluations
    log_interval: int = 50  # iterations between training-metric logs


class QNetwork(nn.Module):
    def __init__(self, n_items: int, static_dim: int, ctx_dim: int, hidden: int = 128):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        self.encoder = CandidateEncoder(n_items, static_dim, ctx_dim, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, obs: CompactObs) -> torch.Tensor:
        """Per-candidate values [N, K]."""
        h, _ = self.encoder(obs)
        return self.head(h).squeeze(-1)


class DQNAgent:
    name = "dqn"

    def __init__(self, q: QNetwork, k: int, cfg: DQNConfig, device, seed: int = 0):
        self.q, self.k, self.cfg = q, k, cfg
        self.target = copy.deepcopy(q).requires_grad_(False)
        self.opt = torch.optim.Adam(q.parameters(), lr=cfg.lr)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.epsilon = 0.0

    @classmethod
    def for_env(cls, env, cfg: DQNConfig, seed: int = 0) -> "DQNAgent":
        q = QNetwork(**encoder_kwargs(env), hidden=cfg.hidden).to(env.device)
        q.encoder.item_static.copy_(env.item_static)
        return cls(q, env.k, cfg, env.device, seed)

    @torch.no_grad()
    def act(self, obs: SlateObs) -> torch.Tensor:
        """Top-k by q; with probability ``epsilon`` per row, a uniformly random slate instead."""
        q = self.q(CompactObs.from_obs(obs))
        greedy = q.topk(self.k, dim=1).indices
        if self.epsilon <= 0.0:
            return greedy
        n, num_cands = q.shape
        noise = torch.rand(n, num_cands, generator=self.gen, device=q.device)
        explore = torch.rand(n, generator=self.gen, device=q.device) < self.epsilon
        return torch.where(explore[:, None], noise.topk(self.k, dim=1).indices, greedy)

    def update(self, batch) -> dict[str, float]:
        obs, action, reward, discount, next_obs = batch
        with torch.no_grad():
            next_action = self.q(next_obs).topk(self.k, dim=1).indices  # double DQN
            next_value = self.target(next_obs).gather(1, next_action).mean(1)
            target = reward + discount * next_value
        value = self.q(obs).gather(1, action).mean(1)
        loss = F.smooth_l1_loss(value, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()
        with torch.no_grad():
            for p, tp in zip(self.q.parameters(), self.target.parameters(), strict=True):
                tp.lerp_(p, self.cfg.tau)
        return {
            "loss": loss.item(),
            "q": value.mean().item(),
            "target": target.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "agent": self.name,
                "net_config": self.q.config,
                "state_dict": self.q.state_dict(),
                "k": self.k,
                "cfg": self.cfg.__dict__,
                **(extra or {}),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device) -> "DQNAgent":
        ckpt = torch.load(path, map_location=device)
        q = QNetwork(**ckpt["net_config"]).to(device)
        q.load_state_dict(ckpt["state_dict"])
        return cls(q, ckpt["k"], DQNConfig(**ckpt["cfg"]), device)


def train_dqn(
    agent: DQNAgent,
    env,
    cfg: DQNConfig,
    evaluate: Callable[[DQNAgent], dict] | None = None,
    log: Callable[[dict, int], None] = lambda m, step: None,
    checkpoint: str | Path | None = None,
) -> dict:
    """Train on the auto-resetting ``env``; evaluate (greedy) every ``eval_interval`` iterations
    and keep the checkpoint with the best evaluation return."""
    obs = env.reset()
    compact = CompactObs.from_obs(obs)
    buffer = ReplayBuffer(cfg.buffer_size, compact, env.k, env.device)
    nstep = NStepAccumulator(cfg.n_step, cfg.gamma)
    decay_iters = max(1, int(cfg.eps_fraction * cfg.total_iters))
    best, best_metrics = -float("inf"), {}
    recent_returns: list[torch.Tensor] = []
    start = time.perf_counter()
    for it in range(1, cfg.total_iters + 1):
        frac = min(1.0, it / decay_iters)
        agent.epsilon = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_compact = CompactObs.from_obs(next_obs)
        if "final_obs" in info:  # true next state for truncated rows (before auto-reset)
            next_compact = CompactObs.where(
                truncated, CompactObs.from_obs(info["final_obs"]), next_compact
            )
        out = nstep.push(compact, action, reward, terminated, terminated | truncated, next_compact)
        if out is not None:
            buffer.add(*out)
        obs, compact = next_obs, CompactObs.from_obs(next_obs)
        if "episode" in info:
            recent_returns.append(info["episode"]["return"])

        metrics: dict[str, float] = {}
        if buffer.size >= cfg.learning_starts:
            agent.q.train()
            for _ in range(cfg.updates_per_iter):
                metrics = agent.update(buffer.sample(cfg.batch_size, agent.gen))

        if it % cfg.log_interval == 0:
            env_steps = it * env.num_envs
            row = {f"train/{k}": v for k, v in metrics.items()}
            row |= {
                "train/epsilon": agent.epsilon,
                "train/buffer": buffer.size,
                "train/sps": env_steps / (time.perf_counter() - start),
            }
            if recent_returns:
                row["train/episode_return"] = torch.cat(recent_returns).mean().item()
                recent_returns = []
            log(row, env_steps)

        if evaluate is not None and (it % cfg.eval_interval == 0 or it == cfg.total_iters):
            eps, agent.epsilon = agent.epsilon, 0.0
            agent.q.eval()
            result = evaluate(agent)
            agent.epsilon = eps
            log(
                {f"eval/{k}": v for k, v in result.items() if isinstance(v, float)},
                it * env.num_envs,
            )
            if result["return"] > best:
                best, best_metrics = result["return"], result
                if checkpoint is not None:
                    agent.save(
                        checkpoint,
                        {
                            "eval": {k: v for k, v in result.items() if isinstance(v, float)},
                            "iter": it,
                        },
                    )
    return best_metrics
