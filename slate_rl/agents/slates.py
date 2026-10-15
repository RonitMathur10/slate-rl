"""Slate construction from per-candidate scores. Shared by baselines and learned agents."""

import torch


def topk_slate(cands: torch.Tensor, scores: torch.Tensor, k: int) -> torch.Tensor:
    """The k highest-scoring candidates, best first. cands/scores [B, K] -> [B, k]."""
    return cands.gather(1, scores.topk(k, dim=1).indices)


def random_slate(cands: torch.Tensor, k: int, gen: torch.Generator | None = None) -> torch.Tensor:
    noise = torch.rand(cands.shape, generator=gen, device=cands.device)
    return cands.gather(1, noise.topk(k, dim=1).indices)


def mmr_indices(
    relevance: torch.Tensor, vecs: torch.Tensor, k: int, diversity: float
) -> torch.Tensor:
    """Maximal marginal relevance over candidate positions.

    Greedily picks argmax(relevance - diversity * max cosine to the slate so far).
    ``relevance`` [B, K]; ``vecs`` [B, K, d] unit-normalised. Returns positions [B, k].
    """
    b, n = relevance.shape
    rows = torch.arange(b, device=relevance.device)
    max_sim = torch.full((b, n), -1.0, device=relevance.device)
    available = torch.ones(b, n, dtype=torch.bool, device=relevance.device)
    picks = []
    for _ in range(k):
        score = (relevance - diversity * max_sim).masked_fill(~available, float("-inf"))
        pick = score.argmax(1)
        picks.append(pick)
        available[rows, pick] = False
        sim = torch.einsum("bkd,bd->bk", vecs, vecs[rows, pick])
        max_sim = torch.maximum(max_sim, sim)
    return torch.stack(picks, 1)


def mmr_slate(
    cands: torch.Tensor,
    relevance: torch.Tensor,
    item_vecs: torch.Tensor,
    k: int,
    diversity: float,
) -> torch.Tensor:
    """MMR returning item ids; ``item_vecs`` are unit-normalised embeddings indexed by item_idx."""
    return cands.gather(1, mmr_indices(relevance, item_vecs[cands], k, diversity))
