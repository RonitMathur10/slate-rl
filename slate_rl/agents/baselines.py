"""Non-learning baselines. All of them see exactly what an agent sees (``SlateObs``)."""

import torch
import torch.nn.functional as F

from slate_rl.agents.click_model import ClickModel
from slate_rl.agents.common import CompactObs
from slate_rl.agents.slates import mmr_indices
from slate_rl.envs.batched import SlateObs


class RandomPolicy:
    name = "random"

    def __init__(self, k: int, seed: int = 0, device="cpu"):
        self.k = k
        self.gen = torch.Generator(device=device).manual_seed(seed)

    def act(self, obs: SlateObs) -> torch.Tensor:
        noise = torch.rand(obs.cand_items.shape, generator=self.gen, device=obs.cand_items.device)
        return noise.topk(self.k, dim=1).indices


class PopularityPolicy:
    name = "popularity"

    def __init__(self, k: int, pop_feature_index: int):
        self.k, self.col = k, pop_feature_index

    def act(self, obs: SlateObs) -> torch.Tensor:
        return obs.cand_feats[..., self.col].topk(self.k, dim=1).indices


class RetrievalPolicy:
    """Top-k by two-tower score (candidates arrive sorted by it)."""

    name = "retrieval"

    def __init__(self, k: int):
        self.k = k

    def act(self, obs: SlateObs) -> torch.Tensor:
        n = obs.cand_items.shape[0]
        return torch.arange(self.k, device=obs.cand_items.device).expand(n, -1)


class GreedyClickPolicy:
    """Top-k by the learned one-step click model: maximises immediate click probability."""

    name = "greedy_click"

    def __init__(self, k: int, click_model: ClickModel):
        self.k, self.model = k, click_model.eval()

    @torch.no_grad()
    def act(self, obs: SlateObs) -> torch.Tensor:
        scores, _ = self.model.scores(CompactObs.from_obs(obs))
        return scores.topk(self.k, dim=1).indices


class MMRPolicy:
    """Greedy-click relevance re-ranked for diversity in retrieval-embedding space."""

    name = "mmr"

    def __init__(self, k: int, click_model: ClickModel, emb_dim: int, diversity: float = 1.0):
        self.k, self.model, self.emb_dim, self.diversity = k, click_model.eval(), emb_dim, diversity

    @torch.no_grad()
    def act(self, obs: SlateObs) -> torch.Tensor:
        scores, _ = self.model.scores(CompactObs.from_obs(obs))
        vecs = F.normalize(obs.cand_feats[..., : self.emb_dim], dim=-1)
        return mmr_indices(scores, vecs, self.k, self.diversity)


def make_baselines(env, click_model: ClickModel | None, seed: int = 0, mmr_diversity: float = 1.0):
    policies = [
        RandomPolicy(env.k, seed, env.device),
        PopularityPolicy(env.k, env.pop_feature_index),
        RetrievalPolicy(env.k),
    ]
    if click_model is not None:
        policies += [
            GreedyClickPolicy(env.k, click_model),
            MMRPolicy(env.k, click_model, env.emb_dim, mmr_diversity),
        ]
    return {p.name: p for p in policies}
