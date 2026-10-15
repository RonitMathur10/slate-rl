"""Two-tower retrieval model.

The user tower has no user-ID embedding: it encodes a user from their recent history and
demographics only, so it works unchanged for held-out users and for simulated sessions.
History items and candidate items share the same input embedding (ID + genres + year).
"""

import torch
import torch.nn.functional as F
from torch import nn

from slate_rl.data.features import AGE_BUCKETS, N_OCCUPATIONS, ItemFeatures


def _mlp(in_dim: int, hidden: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim)
    )


class TwoTower(nn.Module):
    def __init__(
        self,
        n_items: int,
        n_genres: int,
        n_year_buckets: int,
        emb_dim: int = 64,
        hidden_dim: int = 256,
        demo_dim: int = 16,
        recent_k: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        self.recent_k = recent_k
        self.register_buffer("genres", torch.zeros(n_items + 1, n_genres))
        self.register_buffer("year_bucket", torch.zeros(n_items + 1, dtype=torch.long))

        self.item_id = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.genre_proj = nn.Linear(n_genres, emb_dim, bias=False)
        self.year_emb = nn.Embedding(n_year_buckets, emb_dim)
        self.item_mlp = _mlp(emb_dim, hidden_dim, emb_dim, dropout)

        self.gender_emb = nn.Embedding(2, demo_dim)
        self.age_emb = nn.Embedding(len(AGE_BUCKETS), demo_dim)
        self.occ_emb = nn.Embedding(N_OCCUPATIONS, demo_dim)
        self.user_mlp = _mlp(2 * emb_dim + 3 * demo_dim, hidden_dim, emb_dim, dropout)

    @classmethod
    def from_features(cls, item_feats: ItemFeatures, **kwargs) -> "TwoTower":
        n_items, n_genres = item_feats.genres.shape[0] - 1, item_feats.genres.shape[1]
        model = cls(n_items, n_genres, int(item_feats.year_bucket.max()) + 1, **kwargs)
        model.genres.copy_(item_feats.genres)
        model.year_bucket.copy_(item_feats.year_bucket)
        return model

    def item_inputs(self, items: torch.Tensor) -> torch.Tensor:
        return (
            self.item_id(items)
            + self.genre_proj(self.genres[items])
            + self.year_emb(self.year_bucket[items])
        )

    def item_embed(self, items: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_mlp(self.item_inputs(items)), dim=-1)

    def all_item_embeddings(self) -> torch.Tensor:
        """[n_items + 1, emb_dim]; row 0 (padding) is zeroed."""
        emb = self.item_embed(torch.arange(self.genres.shape[0], device=self.genres.device))
        emb[0] = 0.0
        return emb

    def user_embed(self, history: torch.Tensor, demo: torch.Tensor) -> torch.Tensor:
        """``history``: left-padded item ids [B, L]; ``demo``: (gender, age, occupation) [B, 3]."""
        x = self.item_inputs(history)
        mask = (history > 0).unsqueeze(-1).float()
        mean_all = (x * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        xr, mr = x[:, -self.recent_k :], mask[:, -self.recent_k :]
        mean_recent = (xr * mr).sum(1) / mr.sum(1).clamp(min=1.0)
        feats = torch.cat(
            [
                mean_all,
                mean_recent,
                self.gender_emb(demo[:, 0]),
                self.age_emb(demo[:, 1]),
                self.occ_emb(demo[:, 2]),
            ],
            dim=-1,
        )
        return F.normalize(self.user_mlp(feats), dim=-1)
