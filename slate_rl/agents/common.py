"""Pieces shared by all agents: compact observations, the candidate encoder, the policy API."""

from dataclasses import dataclass, fields
from typing import Protocol

import torch
from torch import nn

from slate_rl.envs.batched import N_DYNAMIC_FEATURES, N_SESSION_FEATURES, SlateObs


@dataclass
class CompactObs:
    """What agents store and consume: the static item features are looked up from item ids."""

    user_emb: torch.Tensor  # [N, d]
    session: torch.Tensor  # [N, S]
    cand_items: torch.Tensor  # [N, K] long
    cand_dyn: torch.Tensor  # [N, K, N_DYNAMIC_FEATURES]

    @classmethod
    def from_obs(cls, obs: SlateObs) -> "CompactObs":
        return cls(
            obs.user_emb, obs.session, obs.cand_items, obs.cand_feats[..., -N_DYNAMIC_FEATURES:]
        )

    def __getitem__(self, idx) -> "CompactObs":
        return CompactObs(*(getattr(self, f.name)[idx] for f in fields(self)))

    @staticmethod
    def stack(items: list["CompactObs"]) -> "CompactObs":
        return CompactObs(
            *(torch.stack([getattr(o, f.name) for o in items]) for f in fields(CompactObs))
        )

    @staticmethod
    def where(mask: torch.Tensor, a: "CompactObs", b: "CompactObs") -> "CompactObs":
        """Row-wise select: ``a`` where mask [N] is true, else ``b``."""
        out = []
        for f in fields(CompactObs):
            x, y = getattr(a, f.name), getattr(b, f.name)
            m = mask.view(-1, *([1] * (x.dim() - 1)))
            out.append(torch.where(m, x, y))
        return CompactObs(*out)


class Policy(Protocol):
    name: str

    def act(self, obs: SlateObs) -> torch.Tensor:
        """Slate as indices into ``obs.cand_items`` [N, k]."""
        ...


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class CandidateEncoder(nn.Module):
    """Per-candidate representations conditioned on the user/session context.

    h_i = MLP([c, x_i, c * x_i]) with c = MLP([user_emb, session]) and
    x_i = Linear([item_static[item_i], dynamic_i]).
    """

    def __init__(self, n_items: int, static_dim: int, ctx_dim: int, hidden: int = 128):
        super().__init__()
        self.register_buffer("item_static", torch.zeros(n_items + 1, static_dim))
        self.ctx = _mlp(ctx_dim, hidden, hidden)
        self.item = nn.Linear(static_dim + N_DYNAMIC_FEATURES, hidden)
        self.mix = _mlp(3 * hidden, hidden, hidden)
        self.hidden = hidden

    def forward(self, obs: CompactObs) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (candidate embeddings [N, K, h], context [N, h])."""
        c = self.ctx(torch.cat([obs.user_emb, obs.session], dim=-1).float())
        x = self.item(torch.cat([self.item_static[obs.cand_items], obs.cand_dyn.float()], dim=-1))
        c_ = c[:, None, :].expand_as(x)
        return self.mix(torch.cat([c_, x, c_ * x], dim=-1)), c


def encoder_kwargs(env) -> dict:
    """Constructor arguments that tie an encoder to an environment's feature layout."""
    return {
        "n_items": env.item_static.shape[0] - 1,
        "static_dim": env.item_static.shape[1],
        "ctx_dim": env.emb_dim + N_SESSION_FEATURES,
    }
