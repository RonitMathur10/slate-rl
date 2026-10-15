"""SlateQ (Ie et al., 2019): slate values decomposed through the user's choice model.

    Q(s, A) = R_slate(s, A) + sum_{i in A ∪ {none}} P(i | s, A) * Qbar(s, i)

- ``P`` is the agent's own conditional-logit click model (``ClickModel``), fit beforehand on
  logged data and kept fixed.
- ``Qbar(s, i)`` is the long-term value of the user consuming item i (``Qbar(s, none)`` for
  no click). It is learned by TD on the item that was actually clicked, so the user's
  single-choice behaviour is modelled instead of assumed away as in item-wise DQN.
- ``R_slate`` holds the slate-level reward terms (genre diversity of the slate, penalty for
  re-shown items). The agent computes them exactly from its observation, so they are not
  learned.

The max over slates is approximated by greedy construction: at each position add the
candidate that most increases Q(s, A), including position bias and pairwise diversity.
"""

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from slate_rl.agents.click_model import ClickModel
from slate_rl.agents.common import CandidateEncoder, CompactObs, encoder_kwargs
from slate_rl.agents.offpolicy import train_off_policy
from slate_rl.envs.batched import RewardConfig, SlateObs


@dataclass
class SlateQConfig:
    total_iters: int = 4000
    buffer_size: int = 500_000
    batch_size: int = 512
    lr: float = 3e-4
    gamma: float = 0.95
    n_step: int = 3
    tau: float = 0.005
    learning_starts: int = 20_000
    updates_per_iter: int = 2
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.3
    hidden: int = 128
    grad_clip: float = 10.0
    eval_interval: int = 250
    log_interval: int = 50


class SlateQNetwork(nn.Module):
    """Per-candidate consumption values Qbar(s, i) [N, K] and the no-click value [N]."""

    def __init__(self, n_items: int, static_dim: int, ctx_dim: int, hidden: int = 128):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        self.encoder = CandidateEncoder(n_items, static_dim, ctx_dim, hidden)
        self.item_head = nn.Linear(hidden, 1)
        self.none_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, obs: CompactObs) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = self.encoder(obs)
        return self.item_head(h).squeeze(-1), self.none_head(c).squeeze(-1)


class SlateValue:
    """Slate-level quantities computed from a compact observation (no learning involved)."""

    def __init__(self, genre_cols: slice, reward: RewardConfig, k: int):
        self.genre_cols, self.w, self.k = genre_cols, reward, k

    def genre_distance(self, obs: CompactObs, item_static: torch.Tensor) -> torch.Tensor:
        """Pairwise genre Jaccard distance between candidates [N, K, K] (same as the env's ILD)."""
        g = item_static[obs.cand_items][..., self.genre_cols]
        inter = torch.einsum("nig,njg->nij", g, g)
        size = g.sum(-1)
        union = size[:, :, None] + size[:, None, :] - inter
        return 1.0 - inter / union.clamp(min=1.0)

    def slate_reward(
        self, obs: CompactObs, action: torch.Tensor, item_static: torch.Tensor
    ) -> torch.Tensor:
        """w_div * ILD(A) - w_repeat * share of A already shown: the reward's slate-level part."""
        dist = self.genre_distance(obs, item_static)
        rows = torch.arange(len(action), device=action.device)[:, None, None]
        pair = dist[rows, action[:, :, None], action[:, None, :]]  # [N, k, k]
        off_diag = ~torch.eye(self.k, dtype=torch.bool, device=action.device)
        ild = (pair * off_diag).sum((1, 2)) / (self.k * (self.k - 1))
        repeat = obs.cand_dyn[..., -1].gather(1, action).mean(1)
        return self.w.diversity * ild - self.w.repeat * repeat


def slate_q_value(
    action: torch.Tensor,
    qbar: torch.Tensor,
    q_none: torch.Tensor,
    scores: torch.Tensor,
    none_logit: torch.Tensor,
    position_bias: torch.Tensor,
) -> torch.Tensor:
    """Choice-weighted value of a given slate (without R_slate): sum_i P(i|A) Qbar(s, i)."""
    logits = torch.cat([scores.gather(1, action) + position_bias, none_logit[:, None]], dim=1)
    values = torch.cat([qbar.gather(1, action), q_none[:, None]], dim=1)
    return (torch.softmax(logits, dim=1) * values).sum(1)


def greedy_slate(
    qbar: torch.Tensor,
    q_none: torch.Tensor,
    scores: torch.Tensor,
    none_logit: torch.Tensor,
    position_bias: torch.Tensor,
    dist: torch.Tensor,
    shown: torch.Tensor,
    reward: RewardConfig,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a slate position by position, each time adding the candidate that maximises the
    full Q(s, A) of the partial slate (choice-weighted value + diversity - repeat penalty).

    Choice weights are computed relative to the no-click option for numerical stability.
    Returns (slate positions [N, k], Q(s, A) of the final slate [N]).
    """
    n, num_cands = qbar.shape
    rows = torch.arange(n, device=qbar.device)
    rel = scores - none_logit[:, None]  # log(v_i / v_none) before position bias
    num = q_none.clone()  # sum over chosen of w_i * Qbar_i, plus 1 * Qbar_none
    den = torch.ones(n, device=qbar.device)
    pair_sum = torch.zeros(n, device=qbar.device)
    repeat_sum = torch.zeros(n, device=qbar.device)
    chosen = torch.zeros(n, num_cands, dtype=torch.bool, device=qbar.device)
    picks = []
    n_pairs = k * (k - 1)
    for j in range(k):
        w = torch.exp((rel + position_bias[j]).clamp(max=30.0))
        value = (num[:, None] + w * qbar) / (den[:, None] + w)
        new_pairs = pair_sum[:, None] + 2.0 * (dist * chosen[:, None, :].float()).sum(-1)
        obj = (
            value
            + reward.diversity * new_pairs / n_pairs
            - reward.repeat * (repeat_sum[:, None] + shown) / k
        )
        obj = obj.masked_fill(chosen, float("-inf"))
        pick = obj.argmax(1)
        picks.append(pick)
        w_p = w[rows, pick]
        num = num + w_p * qbar[rows, pick]
        den = den + w_p
        pair_sum = new_pairs[rows, pick]
        repeat_sum = repeat_sum + shown[rows, pick]
        chosen[rows, pick] = True
        best = obj[rows, pick]
    return torch.stack(picks, 1), best


class SlateQAgent:
    name = "slateq"
    stored_action_dim_offset = 1  # replay stores the slate plus the clicked position

    def __init__(
        self,
        net: SlateQNetwork,
        click_model: ClickModel,
        k: int,
        reward: RewardConfig,
        genre_cols: slice,
        cfg: SlateQConfig,
        device,
        seed: int = 0,
    ):
        self.net, self.k, self.cfg, self.reward = net, k, cfg, reward
        self.click_model = click_model.eval().requires_grad_(False)
        self.target = copy.deepcopy(net).requires_grad_(False)
        self.opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.slate = SlateValue(genre_cols, reward, k)
        self.genre_cols = genre_cols
        self.epsilon = 0.0

    @classmethod
    def for_env(
        cls, env, cfg: SlateQConfig, click_model: ClickModel, seed: int = 0
    ) -> "SlateQAgent":
        net = SlateQNetwork(**encoder_kwargs(env), hidden=cfg.hidden).to(env.device)
        net.encoder.item_static.copy_(env.item_static)
        genre_cols = slice(env.pop_feature_index + 1, env.item_static.shape[1])
        return cls(net, click_model, env.k, env.cfg.reward, genre_cols, cfg, env.device, seed)

    @property
    def item_static(self) -> torch.Tensor:
        return self.net.encoder.item_static

    def set_train(self, train: bool) -> None:
        self.net.train(train)

    def stored_action(self, action: torch.Tensor, info: dict) -> torch.Tensor:
        return torch.cat([action, info["position"][:, None]], dim=1)

    @torch.no_grad()
    def _greedy(self, net: SlateQNetwork, obs: CompactObs) -> tuple[torch.Tensor, torch.Tensor]:
        qbar, q_none = net(obs)
        scores, none_logit = self.click_model.scores(obs)
        return greedy_slate(
            qbar,
            q_none,
            scores,
            none_logit,
            self.click_model.position_bias,
            self.slate.genre_distance(obs, self.item_static),
            obs.cand_dyn[..., -1],
            self.reward,
            self.k,
        )

    @torch.no_grad()
    def act(self, obs: SlateObs) -> torch.Tensor:
        compact = CompactObs.from_obs(obs)
        greedy, _ = self._greedy(self.net, compact)
        if self.epsilon <= 0.0:
            return greedy
        n, num_cands = compact.cand_items.shape
        noise = torch.rand(n, num_cands, generator=self.gen, device=greedy.device)
        explore = torch.rand(n, generator=self.gen, device=greedy.device) < self.epsilon
        return torch.where(explore[:, None], noise.topk(self.k, dim=1).indices, greedy)

    def update(self, batch) -> dict[str, float]:
        obs, stored, reward, discount, next_obs = batch
        action, position = stored[:, : self.k], stored[:, self.k]
        clicked = position < self.k
        item_pos = action.gather(1, position.clamp(max=self.k - 1)[:, None]).squeeze(1)
        with torch.no_grad():
            # The first step's slate-level reward is not part of Qbar; it is added back in Q.
            item_reward = reward - self.slate.slate_reward(obs, action, self.item_static)
            next_slate, _ = self._greedy(self.net, next_obs)  # double Q: online net picks
            qbar_t, q_none_t = self.target(next_obs)  # target net evaluates
            scores, none_logit = self.click_model.scores(next_obs)
            next_value = slate_q_value(
                next_slate, qbar_t, q_none_t, scores, none_logit, self.click_model.position_bias
            ) + self.slate.slate_reward(next_obs, next_slate, self.item_static)
            target = item_reward + discount * next_value
        qbar, q_none = self.net(obs)
        pred = torch.where(clicked, qbar.gather(1, item_pos[:, None]).squeeze(1), q_none)
        loss = F.smooth_l1_loss(pred, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.net.parameters(), self.cfg.grad_clip)
        self.opt.step()
        with torch.no_grad():
            for p, tp in zip(self.net.parameters(), self.target.parameters(), strict=True):
                tp.lerp_(p, self.cfg.tau)
        return {
            "loss": loss.item(),
            "qbar_clicked": pred[clicked].mean().item() if clicked.any() else 0.0,
            "q_none": q_none.mean().item(),
            "target": target.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "agent": self.name,
                "net_config": self.net.config,
                "state_dict": self.net.state_dict(),
                "click_model_config": self.click_model.config,
                "click_model_state": self.click_model.state_dict(),
                "k": self.k,
                "reward": self.reward.__dict__,
                "genre_cols": (self.genre_cols.start, self.genre_cols.stop),
                "cfg": self.cfg.__dict__,
                **(extra or {}),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device) -> "SlateQAgent":
        ckpt = torch.load(path, map_location=device)
        net = SlateQNetwork(**ckpt["net_config"]).to(device)
        net.load_state_dict(ckpt["state_dict"])
        click_model = ClickModel(**ckpt["click_model_config"]).to(device)
        click_model.load_state_dict(ckpt["click_model_state"])
        return cls(
            net,
            click_model,
            ckpt["k"],
            RewardConfig(**ckpt["reward"]),
            slice(*ckpt["genre_cols"]),
            SlateQConfig(**ckpt["cfg"]),
            device,
        )


train_slateq = train_off_policy
