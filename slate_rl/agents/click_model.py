"""The agent's own one-step click model, fit on logged interactions from the environment.

A conditional logit over the shown slate plus a no-click option:
    P(position j | slate) ∝ exp(s(context, item_j) + b_j),  P(none) ∝ exp(v0(context))
with a learned position bias b_j. It powers the greedy-click and MMR baselines and the user
choice model inside SlateQ. It never touches the simulator's internals: it only sees the
same observations as any agent, plus the logged clicks.
"""

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from slate_rl.agents.common import CandidateEncoder, CompactObs


class ClickModel(nn.Module):
    def __init__(
        self, n_items: int, static_dim: int, ctx_dim: int, slate_size: int, hidden: int = 128
    ):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        self.encoder = CandidateEncoder(n_items, static_dim, ctx_dim, hidden)
        self.score_head = nn.Linear(hidden, 1)
        self.position_bias = nn.Parameter(torch.zeros(slate_size))
        self.none_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def scores(self, obs: CompactObs) -> tuple[torch.Tensor, torch.Tensor]:
        """Position-free item scores [N, K] and no-click logit [N]."""
        h, c = self.encoder(obs)
        return self.score_head(h).squeeze(-1), self.none_head(c).squeeze(-1)

    def slate_logits(self, obs: CompactObs, action: torch.Tensor) -> torch.Tensor:
        """Choice logits [N, k + 1] for the slate ``action`` (last column = no click)."""
        s, none = self.scores(obs)
        return torch.cat([s.gather(1, action) + self.position_bias, none[:, None]], dim=1)


@dataclass
class ClickLogs:
    obs: CompactObs
    action: torch.Tensor  # [M, k]
    position: torch.Tensor  # [M] clicked slate position, k = no click

    def __len__(self) -> int:
        return len(self.position)

    def split(self, frac: float, gen: torch.Generator) -> tuple["ClickLogs", "ClickLogs"]:
        perm = torch.randperm(len(self), generator=gen, device=self.position.device)
        cut = int(len(self) * (1 - frac))
        a, b = perm[:cut], perm[cut:]
        return (
            ClickLogs(self.obs[a], self.action[a], self.position[a]),
            ClickLogs(self.obs[b], self.action[b], self.position[b]),
        )


@torch.no_grad()
def collect_logs(env, n_iters: int, seed: int = 0) -> ClickLogs:
    """Log slates from an exploratory policy: k random items from the top-M candidates, with
    M drawn per row from {k, 20, K}, so relevant, mid and random slates are all represented."""
    gen = torch.Generator(device=env.device).manual_seed(seed)
    obs = env.reset(seed=seed)
    choices = torch.tensor([env.k, min(20, env.K), env.K], device=env.device)
    logged_obs, actions, positions = [], [], []
    for _ in range(n_iters):
        top_m = choices[torch.randint(3, (env.num_envs,), generator=gen, device=env.device)]
        noise = torch.rand(env.num_envs, env.K, generator=gen, device=env.device)
        noise = noise.masked_fill(torch.arange(env.K, device=env.device) >= top_m[:, None], -1.0)
        action = noise.topk(env.k, dim=1).indices
        compact = CompactObs.from_obs(obs)
        obs, _, _, _, info = env.step(action)
        logged_obs.append(compact)
        actions.append(action)
        positions.append(info["position"])
    cat = lambda xs: torch.cat(xs, 0)  # noqa: E731
    stacked = CompactObs(
        *(
            cat([getattr(o, f) for o in logged_obs])
            for f in ("user_emb", "session", "cand_items", "cand_dyn")
        )
    )
    return ClickLogs(stacked, cat(actions), cat(positions))


@torch.no_grad()
def evaluate_click_model(model: ClickModel, logs: ClickLogs, batch_size: int = 8192) -> dict:
    model.eval()
    nll = correct = p_click_sum = 0.0
    for i in range(0, len(logs), batch_size):
        sl = slice(i, i + batch_size)
        logits = model.slate_logits(logs.obs[sl], logs.action[sl])
        nll += F.cross_entropy(logits, logs.position[sl], reduction="sum").item()
        correct += (logits.argmax(1) == logs.position[sl]).sum().item()
        p_click_sum += (1 - torch.softmax(logits, 1)[:, -1]).sum().item()
    k = logs.action.shape[1]
    n = len(logs)
    ctr = (logs.position < k).float().mean().item()
    # Reference: predict the empirical position frequencies for every slate.
    freq = torch.bincount(logs.position, minlength=k + 1).float() / n
    base_nll = -(freq * torch.log(freq.clamp(min=1e-9))).sum().item()
    return {
        "nll": nll / n,
        "baseline_nll": base_nll,
        "accuracy": correct / n,
        "ctr": ctr,
        "predicted_ctr": p_click_sum / n,
    }


def fit_click_model(
    model: ClickModel,
    train: ClickLogs,
    val: ClickLogs,
    epochs: int = 20,
    batch_size: int = 1024,
    lr: float = 1e-3,
    patience: int = 3,
    gen: torch.Generator | None = None,
    log=lambda _: None,
) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best, best_state, best_metrics, bad = math.inf, None, {}, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(train), generator=gen, device=train.position.device)
        total = 0.0
        for i in range(0, len(train), batch_size):
            idx = perm[i : i + batch_size]
            logits = model.slate_logits(train.obs[idx], train.action[idx])
            loss = F.cross_entropy(logits, train.position[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        metrics = evaluate_click_model(model, val)
        log(
            {"epoch": epoch, "train/nll": total / len(train)}
            | {f"val/{k}": v for k, v in metrics.items()}
        )
        if metrics["nll"] < best:
            best, best_metrics, bad = metrics["nll"], metrics, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best_metrics


def save_click_model(model: ClickModel, path, metrics: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config, "state_dict": model.state_dict(), "metrics": metrics}, path)


def load_click_model(path, device) -> ClickModel:
    ckpt = torch.load(path, map_location=device)
    model = ClickModel(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


__all__ = [
    "ClickLogs",
    "ClickModel",
    "collect_logs",
    "evaluate_click_model",
    "fit_click_model",
    "load_click_model",
    "save_click_model",
]
