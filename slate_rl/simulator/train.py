"""Fit the choice model (catalogue softmax + rating regression) and evaluate it."""

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from slate_rl.data.sequences import UserSequences
from slate_rl.simulator.model import UserChoiceModel


@dataclass
class SimTrainConfig:
    batch_size: int = 512
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 0.0
    rating_weight: float = 0.5
    patience: int = 3


def _masked_logits(logits: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    # Items in the visible history were already consumed; users do not re-pick them.
    return logits.scatter(1, history, float("-inf"))


def _loss(model: UserChoiceModel, history, targets, ratings, rating_weight: float):
    user = model.encode(history)
    choice = F.cross_entropy(_masked_logits(model.logits(user), history), targets)
    rating = F.mse_loss(model.predict_rating(user, targets[:, None])[:, 0], ratings)
    return choice + rating_weight * rating, choice, rating


@torch.no_grad()
def evaluate(
    model: UserChoiceModel,
    seqs: UserSequences,
    split: str,
    users: torch.Tensor | None = None,
    batch_size: int = 2048,
) -> dict[str, float]:
    """Next-item NLL, HR@10, NDCG@10 and rating RMSE at every ``split`` position."""
    model.eval()
    users_p, pos_p = seqs.split_pairs(split, users)
    nll = hr = ndcg = sq_err = 0.0
    for i in range(0, len(users_p), batch_size):
        u, p = users_p[i : i + batch_size], pos_p[i : i + batch_size]
        hist, tgt = seqs.history(u, p), seqs.item_at(u, p)
        rating = seqs.ratings[u, seqs.max_hist + p]
        user = model.encode(hist)
        logits = _masked_logits(model.logits(user), hist)
        logp = F.log_softmax(logits, dim=1)
        nll -= logp.gather(1, tgt[:, None]).sum().item()
        rank = (logits > logits.gather(1, tgt[:, None])).sum(1)
        hr += (rank < 10).sum().item()
        ndcg += torch.where(rank < 10, 1.0 / torch.log2(rank.float() + 2.0), 0.0).sum().item()
        sq_err += ((model.predict_rating(user, tgt[:, None])[:, 0] - rating) ** 2).sum().item()
    n = len(users_p)
    return {
        "nll": nll / n,
        "hr@10": hr / n,
        "ndcg@10": ndcg / n,
        "rating_rmse": math.sqrt(sq_err / n),
    }


@torch.no_grad()
def baseline_metrics(
    seqs: UserSequences, popularity: torch.Tensor, split: str, users: torch.Tensor | None = None
) -> dict[str, float]:
    """Reference numbers: a popularity (unigram) choice model and a global-mean rating."""
    users_p, pos_p = seqs.split_pairs(split, users)
    hist, tgt = seqs.history(users_p, pos_p), seqs.item_at(users_p, pos_p)
    logits = torch.log(popularity + 1.0).expand(len(users_p), -1).clone()
    logits[:, 0] = float("-inf")
    logp = F.log_softmax(_masked_logits(logits, hist), dim=1)
    train_u, train_p = seqs.split_pairs("train")
    mean_rating = seqs.ratings[train_u, seqs.max_hist + train_p].mean()
    ratings = seqs.ratings[users_p, seqs.max_hist + pos_p]
    return {
        "nll": -logp.gather(1, tgt[:, None]).mean().item(),
        "rating_rmse": torch.sqrt(((ratings - mean_rating) ** 2).mean()).item(),
    }


def fit(
    model: UserChoiceModel,
    seqs: UserSequences,
    cfg: SimTrainConfig,
    log: Callable[[dict], None] = lambda _: None,
) -> dict[str, float]:
    """Train with early stopping on validation NLL; restores the best weights."""
    users_all, pos_all = seqs.train_pairs()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best, best_state, best_metrics, bad_epochs = math.inf, None, {}, 0
    n = len(users_all)
    for epoch in range(cfg.epochs):
        model.train()
        perm = torch.randperm(n, device=seqs.device)
        sums = torch.zeros(3, device=seqs.device)
        n_batches = 0
        for i in range(0, n, cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            u, p = users_all[idx], pos_all[idx]
            loss, choice, rating = _loss(
                model,
                seqs.history(u, p),
                seqs.item_at(u, p),
                seqs.ratings[u, seqs.max_hist + p],
                cfg.rating_weight,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sums += torch.stack([loss.detach(), choice.detach(), rating.detach()])
            n_batches += 1

        metrics = evaluate(model, seqs, "val")
        train = (sums / n_batches).tolist()
        log(
            {
                "epoch": epoch,
                "train/loss": train[0],
                "train/nll": train[1],
                "train/rating_mse": train[2],
            }
            | {f"val/{k}": v for k, v in metrics.items()}
        )
        if metrics["nll"] < best:
            best, best_metrics, bad_epochs = metrics["nll"], metrics, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                break
    model.load_state_dict(best_state)
    return best_metrics


@torch.no_grad()
def consecutive_similarity(
    model: UserChoiceModel, seqs: UserSequences, window: int, max_pairs: int = 200_000
) -> float:
    """Mean cosine similarity between a real next item and the ``window`` items before it.

    This is the simulator's fatigue reference level: a typical real sequence leaves fatigue flat.
    """
    users, pos = seqs.train_pairs()
    if len(users) > max_pairs:
        keep = torch.randperm(len(users), device=seqs.device)[:max_pairs]
        users, pos = users[keep], pos[keep]
    vecs = model.item_vectors()
    recent = seqs.history(users, pos)[:, -window:]
    sim = torch.einsum("bd,bwd->bw", vecs[seqs.item_at(users, pos)], vecs[recent])
    mask = (recent > 0).float()
    return ((sim * mask).sum(1) / mask.sum(1).clamp(min=1.0)).mean().item()
