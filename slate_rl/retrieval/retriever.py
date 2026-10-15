"""FAISS-backed candidate retrieval around a trained two-tower model."""

import json
import os
import sys
from pathlib import Path

# On macOS, the torch and faiss-cpu wheels each bundle their own libomp; the second one to
# initialise aborts the process unless duplicates are allowed. Must be set before import.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from slate_rl.retrieval.model import TwoTower  # noqa: E402

# The catalogue is a few thousand items, so a single thread is plenty and avoids the two
# OpenMP runtimes competing for threads.
faiss.omp_set_num_threads(1)


def pad_history(histories: list[list[int]], max_hist: int) -> torch.Tensor:
    """Left-pad (and truncate to the most recent ``max_hist``) item-id lists. [B, max_hist]"""
    out = torch.zeros(len(histories), max_hist, dtype=torch.long)
    for row, hist in enumerate(histories):
        recent = hist[-max_hist:]
        if recent:
            out[row, -len(recent) :] = torch.tensor(recent, dtype=torch.long)
    return out


class Retriever:
    """Exact inner-product search over item embeddings. Returned ids are ``item_idx`` (>= 1)."""

    def __init__(self, model: TwoTower, max_hist: int):
        self.model = model.eval()
        self.max_hist = max_hist
        with torch.no_grad():
            self.item_emb = model.all_item_embeddings().cpu().numpy().astype(np.float32)
        self.index = faiss.IndexFlatIP(self.item_emb.shape[1])
        self.index.add(self.item_emb[1:])

    @torch.no_grad()
    def user_embedding(self, histories: list[list[int]], demo: torch.Tensor) -> np.ndarray:
        device = self.model.genres.device
        hist = pad_history(histories, self.max_hist).to(device)
        emb = self.model.user_embed(hist, demo.to(device))
        return emb.cpu().numpy().astype(np.float32)

    def search(
        self, user_emb: np.ndarray, k: int, exclude: list[set[int]] | None = None
    ) -> np.ndarray:
        """Top-k item_idx per row [B, k], skipping any ids in ``exclude[row]``."""
        extra = max((len(e) for e in exclude), default=0) if exclude else 0
        _, ids = self.index.search(np.ascontiguousarray(user_emb), k + extra)
        ids = ids + 1  # index row r holds item_idx r + 1
        if not exclude:
            return ids[:, :k]
        return np.stack(
            [[i for i in row if i not in ex][:k] for row, ex in zip(ids, exclude, strict=True)]
        )

    def save(self, out_dir: str | Path, metrics: dict | None = None) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": self.model.config,
                "max_hist": self.max_hist,
            },
            out_dir / "model.pt",
        )
        np.save(out_dir / "item_emb.npy", self.item_emb)
        faiss.write_index(self.index, str(out_dir / "index.faiss"))
        if metrics is not None:
            (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    @classmethod
    def load(cls, out_dir: str | Path, device: str | torch.device = "cpu") -> "Retriever":
        ckpt = torch.load(Path(out_dir) / "model.pt", map_location=device)
        model = TwoTower(**ckpt["config"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, ckpt["max_hist"])
