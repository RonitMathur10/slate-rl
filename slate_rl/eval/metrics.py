"""Slate- and session-level metrics that do not depend on any learned model."""

import torch


def intra_list_diversity(slates: torch.Tensor, genres: torch.Tensor) -> torch.Tensor:
    """Mean pairwise genre distance (1 - Jaccard) within each slate [B, k] -> [B]."""
    g = genres[slates]  # [B, k, G]
    inter = torch.einsum("big,bjg->bij", g, g)
    size = g.sum(-1)
    union = size[:, :, None] + size[:, None, :] - inter
    dist = 1.0 - inter / union.clamp(min=1.0)
    k = slates.shape[1]
    off_diag = ~torch.eye(k, dtype=torch.bool, device=slates.device)
    return dist[:, off_diag].mean(1)


def novelty(items: torch.Tensor, popularity: torch.Tensor) -> torch.Tensor:
    """Self-information -log2 p(item) under training popularity; elementwise over ``items``."""
    p = (popularity + 1.0) / (popularity + 1.0).sum()
    return -torch.log2(p[items])


def genre_entropy(genre_counts: torch.Tensor) -> torch.Tensor:
    """Shannon entropy (bits) of each row of genre counts [B, G] -> [B]; 0 for empty rows."""
    total = genre_counts.sum(-1, keepdim=True)
    p = genre_counts / total.clamp(min=1e-9)
    ent = -(p * torch.log2(p.clamp(min=1e-12))).sum(-1)
    return torch.where(total.squeeze(-1) > 0, ent, 0.0)
