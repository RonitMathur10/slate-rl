"""Per-user chronological sequences laid out for fast, fixed-length history windows on device."""

import numpy as np
import polars as pl
import torch

from slate_rl.data.preprocess import ProcessedData


class UserSequences:
    """Each user's full item/rating sequence, left-padded with ``max_hist`` zeros.

    ``items[u, max_hist + t]`` is user u's t-th interaction, so the ``max_hist`` items before
    position t are the contiguous slice ``items[u, t : t + max_hist]`` (zeros = padding).
    Positions ``[0, n_train)`` are train, ``[n_train, n_train + n_val)`` val, the rest test.
    """

    def __init__(self, data: ProcessedData, max_hist: int):
        self.max_hist = max_hist
        self.n_items = data.meta["n_items"]
        inter = data.interactions
        stats = (
            inter.group_by("user_idx")
            .agg(
                pl.len().alias("n_total"),
                (pl.col("split") == "train").sum().alias("n_train"),
                (pl.col("split") == "val").sum().alias("n_val"),
            )
            .sort("user_idx")
        )
        n_users = data.meta["n_users"]
        assert stats["user_idx"].to_list() == list(range(n_users))

        width = max_hist + int(stats["n_total"].max())
        items = np.zeros((n_users, width), dtype=np.int64)
        ratings = np.zeros((n_users, width), dtype=np.float32)
        rows = inter["user_idx"].to_numpy()
        cols = max_hist + inter["seq_pos"].to_numpy()
        items[rows, cols] = inter["item_idx"].to_numpy()
        ratings[rows, cols] = inter["rating"].to_numpy()

        heldout = data.users.sort("user_idx")["group"].to_numpy() == "heldout"
        self.items = torch.from_numpy(items)
        self.ratings = torch.from_numpy(ratings)
        self.n_total = torch.from_numpy(stats["n_total"].to_numpy().astype(np.int64))
        self.n_train = torch.from_numpy(stats["n_train"].to_numpy().astype(np.int64))
        self.n_val = torch.from_numpy(stats["n_val"].to_numpy().astype(np.int64))
        self.heldout = torch.from_numpy(heldout)

    def to(self, device) -> "UserSequences":
        for name in ("items", "ratings", "n_total", "n_train", "n_val", "heldout"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    @property
    def device(self) -> torch.device:
        return self.items.device

    def model_users(self) -> torch.Tensor:
        return torch.nonzero(~self.heldout).squeeze(1)

    def heldout_users(self) -> torch.Tensor:
        return torch.nonzero(self.heldout).squeeze(1)

    def history(self, users: torch.Tensor, ends: torch.Tensor) -> torch.Tensor:
        """The ``max_hist`` items before position ``ends`` (exclusive), left-padded. [B, max_hist]"""
        cols = ends[:, None] + torch.arange(self.max_hist, device=self.device)
        return self.items[users[:, None], cols]

    def items_in_range(
        self, users: torch.Tensor, start: torch.Tensor, stop: torch.Tensor
    ) -> torch.Tensor:
        """Boolean [B, n_items + 1] marking items at positions ``[start, stop)`` per user."""
        rows = self.items[users]
        pos = torch.arange(rows.shape[1], device=self.device) - self.max_hist
        in_range = (pos >= start[:, None]) & (pos < stop[:, None])
        out = torch.zeros(len(users), self.n_items + 1, device=self.device)
        out.scatter_(1, torch.where(in_range, rows, 0), in_range.float())
        out[:, 0] = 0.0
        return out > 0

    def train_pairs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(user, position) for every model-user train interaction that has some history."""
        users = self.model_users()
        counts = (self.n_train[users] - 1).clamp(min=0)
        user_rep = torch.repeat_interleave(users, counts)
        offsets = torch.cumsum(counts, 0) - counts
        pos = torch.arange(len(user_rep), device=self.device) - torch.repeat_interleave(
            offsets, counts
        )
        return user_rep, pos + 1

    def item_at(self, users: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.items[users, self.max_hist + positions]
