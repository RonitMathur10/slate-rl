"""PPO for slates (CleanRL-style, single file) with two interchangeable action heads.

``proto`` (Wolpertinger-style): the actor outputs k continuous proto-actions in the retrieval
embedding space, sampled from a diagonal Gaussian. Each is mapped to its nearest unused
candidate (cosine), so the policy never scores individual items and would scale to catalogues
far larger than K. With ``proto_anchor`` the means are the user's retrieval embedding plus a
learned residual whose last layer starts at zero, so the initial deterministic policy is exactly
retrieval ranking and PPO learns deviations from it; without the anchor, randomly initialised
means plus exploration noise map to near-random slates and learning stalls. The
log-probability is that of the proto-actions; the nearest-neighbour map is treated as part of
the environment. (Wolpertinger's critic re-scoring of the m nearest
neighbours needs a Q(s, a) critic, so it is not used with PPO's V(s) critic.)

``plackett_luce``: the actor scores every candidate and samples a ranked slate without
replacement (Gumbel top-k). Log-probabilities are exact:
    log pi(a_1..a_k) = sum_j [ s_{a_j} - logsumexp_{i not in a_<j} s_i ].
It is the control: same PPO, different action parameterisation.

Truncated episodes are bootstrapped by adding gamma * V(final_obs) to the last reward.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from slate_rl.agents.common import CandidateEncoder, CompactObs, encoder_kwargs
from slate_rl.envs.batched import SlateObs


@dataclass
class PPOConfig:
    head: str = "proto"  # "proto" or "plackett_luce"
    total_steps: int = 1_000_000  # environment steps (all parallel envs)
    rollout_len: int = 32
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.95
    gae_lambda: float = 0.95
    epochs: int = 4
    minibatches: int = 8
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    hidden: int = 128
    init_log_std: float = -2.5  # proto head: initial exploration scale in embedding space
    proto_anchor: bool = True  # proto head: means = user embedding + learned residual
    eval_interval: int = 10  # PPO iterations between evaluations


# ----- action heads -------------------------------------------------------------------------
def nearest_unused(proto: torch.Tensor, cand_emb: torch.Tensor) -> torch.Tensor:
    """Map proto-actions [N, k, d] to distinct candidate positions [N, k] by cosine similarity,
    slot by slot (earlier slots get first pick). ``cand_emb`` [N, K, d] is unit-normalised."""
    sim = torch.einsum("nkd,ncd->nkc", F.normalize(proto, dim=-1), cand_emb)
    n, k, num_cands = sim.shape
    rows = torch.arange(n, device=proto.device)
    used = torch.zeros(n, num_cands, dtype=torch.bool, device=proto.device)
    picks = []
    for j in range(k):
        pick = sim[:, j].masked_fill(used, float("-inf")).argmax(1)
        picks.append(pick)
        used[rows, pick] = True
    return torch.stack(picks, 1)


def plackett_luce_log_prob(scores: torch.Tensor, slate: torch.Tensor) -> torch.Tensor:
    """Log-probability [N] of the ordered slate [N, k] under Plackett-Luce with scores [N, K]."""
    n, k = slate.shape
    rows = torch.arange(n, device=scores.device)
    available = torch.ones_like(scores, dtype=torch.bool)
    logp = torch.zeros(n, device=scores.device)
    for j in range(k):
        masked = scores.masked_fill(~available, float("-inf"))
        logp = logp + scores.gather(1, slate[:, j : j + 1]).squeeze(1) - masked.logsumexp(1)
        available[rows, slate[:, j]] = False
    return logp


def gumbel_top_k(scores: torch.Tensor, k: int, gen: torch.Generator | None) -> torch.Tensor:
    """Sample an ordered slate without replacement from Plackett-Luce(scores)."""
    u = torch.rand(scores.shape, generator=gen, device=scores.device).clamp(1e-9, 1 - 1e-9)
    return (scores - torch.log(-torch.log(u))).topk(k, dim=1).indices


# ----- network ------------------------------------------------------------------------------
class ActorCritic(nn.Module):
    def __init__(
        self,
        n_items: int,
        static_dim: int,
        ctx_dim: int,
        k: int,
        emb_dim: int,
        head: str = "proto",
        hidden: int = 128,
        init_log_std: float = -1.0,
        proto_anchor: bool = False,
    ):
        super().__init__()
        self.config = {k_: v for k_, v in locals().items() if k_ not in ("self", "__class__")}
        if head not in ("proto", "plackett_luce"):
            raise ValueError(f"unknown head {head!r}")
        self.head, self.k, self.emb_dim, self.proto_anchor = head, k, emb_dim, proto_anchor
        self.encoder = CandidateEncoder(n_items, static_dim, ctx_dim, hidden)
        self.value_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        if head == "proto":
            self.proto_mu = nn.Sequential(
                nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, k * emb_dim)
            )
            self.log_std = nn.Parameter(torch.full((k, emb_dim), init_log_std))
            if proto_anchor:  # start exactly at the anchor: residual output is zero
                nn.init.zeros_(self.proto_mu[-1].weight)
                nn.init.zeros_(self.proto_mu[-1].bias)
        else:
            self.score_head = nn.Linear(hidden, 1)

    def cand_emb(self, obs: CompactObs) -> torch.Tensor:
        return self.encoder.item_static[obs.cand_items][..., : self.emb_dim]

    def forward(self, obs: CompactObs) -> tuple[torch.Tensor, torch.Tensor]:
        """Policy parameters (proto means [N, k, d] or candidate scores [N, K]) and V(s) [N]."""
        h, c = self.encoder(obs)
        value = self.value_head(c).squeeze(-1)
        if self.head == "proto":
            mu = self.proto_mu(c).view(-1, self.k, self.emb_dim)
            if self.proto_anchor:
                mu = mu + obs.user_emb[:, None, : self.emb_dim].float()
            return mu, value
        return self.score_head(h).squeeze(-1), value

    def dist_stats(self, params: torch.Tensor, raw_action: torch.Tensor):
        """(log_prob [N], entropy [N]) of a raw action under the policy parameters."""
        if self.head == "proto":
            dist = torch.distributions.Normal(params, self.log_std.exp())
            return dist.log_prob(raw_action).sum((1, 2)), dist.entropy().sum((1, 2))
        logp = plackett_luce_log_prob(params, raw_action)
        first = torch.softmax(params, dim=1)  # entropy of the first-position choice
        entropy = -(first * torch.log_softmax(params, dim=1)).sum(1)
        return logp, entropy


# ----- agent --------------------------------------------------------------------------------
class PPOAgent:
    name = "ppo"

    def __init__(self, net: ActorCritic, cfg: PPOConfig, device, seed: int = 0):
        self.net, self.cfg, self.k = net, cfg, net.k
        self.opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.deterministic = True  # evaluation default; training rollouts sample

    @classmethod
    def for_env(cls, env, cfg: PPOConfig, seed: int = 0) -> "PPOAgent":
        net = ActorCritic(
            **encoder_kwargs(env),
            k=env.k,
            emb_dim=env.emb_dim,
            head=cfg.head,
            hidden=cfg.hidden,
            init_log_std=cfg.init_log_std,
            proto_anchor=cfg.proto_anchor,
        ).to(env.device)
        net.encoder.item_static.copy_(env.item_static)
        return cls(net, cfg, env.device, seed)

    def to_slate(self, obs: CompactObs, raw_action: torch.Tensor) -> torch.Tensor:
        if self.net.head == "proto":
            return nearest_unused(raw_action, self.net.cand_emb(obs))
        return raw_action

    @torch.no_grad()
    def sample(self, obs: CompactObs):
        """(raw action, slate positions, log_prob, value) for a training rollout."""
        params, value = self.net(obs)
        if self.net.head == "proto":
            noise = torch.randn(params.shape, generator=self.gen, device=params.device)
            raw = params + noise * self.net.log_std.exp()
        else:
            raw = gumbel_top_k(params, self.k, self.gen)
        logp, _ = self.net.dist_stats(params, raw)
        return raw, self.to_slate(obs, raw), logp, value

    @torch.no_grad()
    def act(self, obs: SlateObs) -> torch.Tensor:
        compact = CompactObs.from_obs(obs)
        if not self.deterministic:
            return self.sample(compact)[1]
        params, _ = self.net(compact)
        if self.net.head == "proto":
            return self.to_slate(compact, params)
        return params.topk(self.k, dim=1).indices

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "agent": self.name,
                "net_config": self.net.config,
                "state_dict": self.net.state_dict(),
                "cfg": self.cfg.__dict__,
                **(extra or {}),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device) -> "PPOAgent":
        ckpt = torch.load(path, map_location=device)
        net = ActorCritic(**ckpt["net_config"]).to(device)
        net.load_state_dict(ckpt["state_dict"])
        return cls(net, PPOConfig(**ckpt["cfg"]), device)


# ----- training -----------------------------------------------------------------------------
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE over [T, N] tensors. ``dones[t]``: the episode ended after step t (its next value
    is not bootstrapped; truncation bootstraps are already folded into ``rewards``)."""
    steps = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(next_value)
    for t in reversed(range(steps)):
        nxt = next_value if t == steps - 1 else values[t + 1]
        keep = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * nxt * keep - values[t]
        last = delta + gamma * lam * keep * last
        adv[t] = last
    return adv, adv + values


def _flat(obs_list: list[CompactObs]) -> CompactObs:
    stacked = CompactObs.stack(obs_list)
    return CompactObs(*(getattr(stacked, f.name).flatten(0, 1) for f in fields(CompactObs)))


def train_ppo(
    agent: PPOAgent,
    env,
    cfg: PPOConfig,
    evaluate: Callable[[PPOAgent], dict] | None = None,
    log: Callable[[dict, int], None] = lambda m, step: None,
    checkpoint: str | Path | None = None,
) -> dict:
    net = agent.net
    n_envs, steps = env.num_envs, cfg.rollout_len
    batch = n_envs * steps
    n_iters = max(1, cfg.total_steps // batch)
    mb_size = batch // cfg.minibatches
    obs = env.reset()
    best, best_metrics = -float("inf"), {}
    start = time.perf_counter()
    for it in range(1, n_iters + 1):
        if cfg.anneal_lr:
            for group in agent.opt.param_groups:
                group["lr"] = cfg.lr * (1.0 - (it - 1) / n_iters)

        # ---- rollout ----
        net.eval()
        obs_buf, raw_buf, logp_buf, val_buf, rew_buf, done_buf = [], [], [], [], [], []
        episode_returns = []
        for _ in range(steps):
            compact = CompactObs.from_obs(obs)
            raw, slate, logp, value = agent.sample(compact)
            obs, reward, terminated, truncated, info = env.step(slate)
            if truncated.any():
                with torch.no_grad():
                    _, final_value = net(CompactObs.from_obs(info["final_obs"]))
                reward = reward + cfg.gamma * final_value * truncated.float()
            obs_buf.append(compact)
            raw_buf.append(raw)
            logp_buf.append(logp)
            val_buf.append(value)
            rew_buf.append(reward)
            done_buf.append(terminated | truncated)
            if "episode" in info:
                episode_returns.append(info["episode"]["return"])
        with torch.no_grad():
            _, next_value = net(CompactObs.from_obs(obs))
        values, rewards = torch.stack(val_buf), torch.stack(rew_buf)
        adv, returns = compute_gae(
            rewards, values, torch.stack(done_buf), next_value, cfg.gamma, cfg.gae_lambda
        )

        # ---- update ----
        net.train()
        b_obs = _flat(obs_buf)
        b_raw = torch.stack(raw_buf).flatten(0, 1)
        b_logp, b_adv = torch.stack(logp_buf).flatten(), adv.flatten()
        b_ret, b_val = returns.flatten(), values.flatten()
        stats: dict[str, list[float]] = {
            k: [] for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac")
        }
        for _ in range(cfg.epochs):
            perm = torch.randperm(batch, generator=agent.gen, device=b_adv.device)
            for i in range(0, batch, mb_size):
                idx = perm[i : i + mb_size]
                params, value = net(b_obs[idx])
                logp, entropy = net.dist_stats(params, b_raw[idx])
                log_ratio = logp - b_logp[idx]
                ratio = log_ratio.exp()
                a = b_adv[idx]
                a = (a - a.mean()) / (a.std() + 1e-8)
                policy_loss = torch.max(
                    -a * ratio, -a * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)
                ).mean()
                value_loss = 0.5 * ((value - b_ret[idx]) ** 2).mean()
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy.mean()
                agent.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                agent.opt.step()
                with torch.no_grad():
                    stats["policy_loss"].append(policy_loss.item())
                    stats["value_loss"].append(value_loss.item())
                    stats["entropy"].append(entropy.mean().item())
                    stats["approx_kl"].append(((ratio - 1) - log_ratio).mean().item())
                    stats["clipfrac"].append(((ratio - 1).abs() > cfg.clip).float().mean().item())

        env_steps = it * batch
        var_y = b_ret.var()
        row = {f"train/{k}": sum(v) / len(v) for k, v in stats.items()}
        row |= {
            "train/explained_variance": (1 - (b_ret - b_val).var() / var_y).item(),
            "train/lr": agent.opt.param_groups[0]["lr"],
            "train/sps": env_steps / (time.perf_counter() - start),
        }
        if net.head == "proto":
            row["train/policy_std"] = net.log_std.exp().mean().item()
        if episode_returns:
            row["train/episode_return"] = torch.cat(episode_returns).mean().item()
        log(row, env_steps)

        if evaluate is not None and (it % cfg.eval_interval == 0 or it == n_iters):
            net.eval()
            agent.deterministic = True
            result = evaluate(agent)
            log({f"eval/{k}": v for k, v in result.items() if isinstance(v, float)}, env_steps)
            if result["return"] > best:
                best, best_metrics = result["return"], result
                if checkpoint is not None:
                    floats = {k: v for k, v in result.items() if isinstance(v, float)}
                    agent.save(checkpoint, {"eval": floats, "iter": it})
    return best_metrics
