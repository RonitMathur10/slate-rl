"""Training (in-batch sampled softmax with logQ correction) and Recall/NDCG evaluation."""

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from slate_rl.data.sequences import UserSequences
from slate_rl.retrieval.model import TwoTower

ScoreFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class TrainConfig:
    batch_size: int = 1024
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    temperature: float = 0.05
    logq_correction: bool = True
    eval_ks: list[int] = field(default_factory=lambda: [10, 50, 100])
    select_metric: str = "recall@100"
    patience: int = 3


def in_batch_softmax_loss(
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    targets: torch.Tensor,
    temperature: float,
    log_q: torch.Tensor | None = None,
) -> torch.Tensor:
    """Each row's positive is its own target; other targets in the batch are negatives.

    ``log_q`` subtracts each item's log sampling frequency so popular items are not
    over-penalised as negatives. Duplicate targets in the batch are masked out.
    """
    logits = user_emb @ item_emb.T / temperature
    if log_q is not None:
        logits = logits - log_q[targets][None, :]
    dup = targets[:, None] == targets[None, :]
    dup.fill_diagonal_(False)
    logits = logits.masked_fill(dup, float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(targets), device=logits.device))


def model_score_fn(model: TwoTower, demo: torch.Tensor) -> ScoreFn:
    model.eval()
    with torch.no_grad():
        item_emb = model.all_item_embeddings()

    def score(users: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        return model.user_embed(history, demo[users]) @ item_emb.T

    return score


def popularity_score_fn(popularity: torch.Tensor) -> ScoreFn:
    log_pop = torch.log1p(popularity)

    def score(users: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        return log_pop.expand(len(users), -1)

    return score


@torch.no_grad()
def evaluate(
    score_fn: ScoreFn,
    seqs: UserSequences,
    users: torch.Tensor,
    split: str,
    ks: list[int],
    batch_size: int = 1024,
) -> dict[str, float]:
    """Recall@K / NDCG@K of the ``split`` items given everything before them (seen items masked)."""
    assert split in ("val", "test")
    k_max = max(ks)
    discounts = 1.0 / torch.log2(torch.arange(k_max, device=seqs.device).float() + 2.0)
    ideal = torch.cumsum(discounts, 0)
    totals = {f"{m}@{k}": 0.0 for m in ("recall", "ndcg") for k in ks}
    for i in range(0, len(users), batch_size):
        b = users[i : i + batch_size]
        end = seqs.n_train[b] if split == "val" else seqs.n_train[b] + seqs.n_val[b]
        stop = end + seqs.n_val[b] if split == "val" else seqs.n_total[b]
        scores = score_fn(b, seqs.history(b, end)).clone()
        scores[:, 0] = float("-inf")
        scores[seqs.items_in_range(b, torch.zeros_like(end), end)] = float("-inf")
        targets = seqs.items_in_range(b, end, stop)
        hits = targets.gather(1, scores.topk(k_max, dim=1).indices).float()
        n_targets = targets.sum(1).float()
        for k in ks:
            totals[f"recall@{k}"] += (hits[:, :k].sum(1) / n_targets).sum().item()
            dcg = (hits[:, :k] * discounts[:k]).sum(1)
            idcg = ideal[(n_targets.long().clamp(max=k) - 1)]
            totals[f"ndcg@{k}"] += (dcg / idcg).sum().item()
    return {name: total / len(users) for name, total in totals.items()}


def fit(
    model: TwoTower,
    seqs: UserSequences,
    demo: torch.Tensor,
    cfg: TrainConfig,
    log: Callable[[dict], None] = lambda _: None,
) -> dict[str, float]:
    """Train with early stopping on validation ``select_metric``; restores the best weights."""
    users_all, pos_all = seqs.train_pairs()
    targets_all = seqs.item_at(users_all, pos_all)
    log_q = None
    if cfg.logq_correction:
        counts = torch.bincount(targets_all, minlength=seqs.n_items + 1).float()
        log_q = torch.log(counts.clamp(min=1.0) / counts.sum())

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    val_users = seqs.model_users()
    best, best_state, best_metrics, bad_epochs = -math.inf, None, {}, 0
    n = len(users_all)
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, device=seqs.device)
        total_loss, n_batches = 0.0, 0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            u, t, tgt = users_all[idx], pos_all[idx], targets_all[idx]
            user_emb = model.user_embed(seqs.history(u, t), demo[u])
            loss = in_batch_softmax_loss(
                user_emb, model.item_embed(tgt), tgt, cfg.temperature, log_q
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        metrics = evaluate(model_score_fn(model, demo), seqs, val_users, "val", cfg.eval_ks)
        log(
            {"epoch": epoch, "train/loss": total_loss / n_batches}
            | {f"val/{k}": v for k, v in metrics.items()}
        )
        score = metrics[cfg.select_metric]
        if score > best:
            best, best_metrics, bad_epochs = score, metrics, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return best_metrics
