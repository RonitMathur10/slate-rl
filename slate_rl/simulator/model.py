"""Learned part of the user simulator: a SASRec-style history encoder with choice and rating heads.

Trained with a full-catalogue softmax, i.e. a conditional-logit choice model over all items.
Because a conditional logit has the IIA property, restricting the same utilities to a slate
(plus a no-click option) at simulation time stays consistent with how it was fit.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class UserChoiceModel(nn.Module):
    def __init__(
        self,
        n_items: int,
        max_hist: int = 50,
        emb_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.config = {k: v for k, v in locals().items() if k not in ("self", "__class__")}
        self.n_items, self.max_hist, self.emb_dim = n_items, max_hist, emb_dim
        self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_hist, emb_dim)
        self.dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            emb_dim,
            n_heads,
            4 * emb_dim,
            dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(emb_dim)
        self.item_bias = nn.Parameter(torch.zeros(n_items + 1))
        self.rating_head = nn.Sequential(
            nn.Linear(3 * emb_dim, emb_dim), nn.GELU(), nn.Linear(emb_dim, 1)
        )
        # Small init: item embeddings are scaled by sqrt(d) on input and tied to the output
        # logits, so N(0, 1) init makes both start far too large and training crawls.
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        with torch.no_grad():
            self.item_emb.weight[0].zero_()

    def encode(self, history: torch.Tensor) -> torch.Tensor:
        """Left-padded history [B, L<=max_hist] -> user state [B, emb_dim] (last position)."""
        length = history.shape[1]
        # Right-aligned positions: the most recent item always gets the same position embedding.
        pos = torch.arange(self.max_hist - length, self.max_hist, device=history.device)
        x = self.item_emb(history) * math.sqrt(self.emb_dim) + self.pos_emb(pos)
        pad = history == 0
        pad[:, -1] = False  # never fully mask a row (an empty history would give NaNs)
        h = self.encoder(self.dropout(x), src_key_padding_mask=pad)
        return self.norm(h[:, -1])

    def logits(self, user: torch.Tensor) -> torch.Tensor:
        """Choice logits over the whole catalogue [B, n_items + 1]; padding column is -inf."""
        out = user @ self.item_emb.weight.T + self.item_bias
        out[:, 0] = float("-inf")
        return out

    def predict_rating(self, user: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        """Predicted rating in [1, 5] for ``items`` [B, k] -> [B, k]."""
        v = self.item_emb(items)
        u = user[:, None, :].expand_as(v)
        return 1.0 + 4.0 * torch.sigmoid(self.rating_head(torch.cat([u, v, u * v], -1)).squeeze(-1))

    def item_vectors(self) -> torch.Tensor:
        """Unit-normalised item embeddings [n_items + 1, emb_dim], used for similarity."""
        return F.normalize(self.item_emb.weight, dim=-1)
