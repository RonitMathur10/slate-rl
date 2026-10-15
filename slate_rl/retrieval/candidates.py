"""Batched, on-device candidate generation (the env's per-step pre-filter).

FAISS (``Retriever``) is used for single requests (API, viewer). Inside the batched env a
matmul over a few thousand items on the same device is faster and avoids host round-trips.
"""

import torch

from slate_rl.retrieval.model import TwoTower


class CandidateGenerator:
    def __init__(self, model: TwoTower, demo: torch.Tensor):
        self.model = model.eval()
        self.demo = demo
        with torch.no_grad():
            self.item_emb = model.all_item_embeddings()

    @torch.no_grad()
    def __call__(
        self, users: torch.Tensor, history: torch.Tensor, seen: torch.Tensor, k: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k unseen items by two-tower score -> (item_idx [B, k], scores [B, k])."""
        scores = self.model.user_embed(history, self.demo[users]) @ self.item_emb.T
        scores[:, 0] = float("-inf")
        scores = scores.masked_fill(seen, float("-inf"))
        top = scores.topk(k, dim=1)
        return top.indices, top.values
