"""Roll reference policies through the simulator and summarise the resulting sessions.

The "oracle" policies read the simulator's own learned affinity; they exist only to sanity-check
the dynamics (an agent never gets this access). The key check: greedy-by-affinity should win on
early CTR but lose on session length to a diversified version of itself, otherwise fatigue
does not create the long-term trade-off the RL agents are meant to learn.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch

from slate_rl.agents import mmr_slate, random_slate, topk_slate
from slate_rl.eval import genre_entropy, intra_list_diversity, novelty
from slate_rl.retrieval import CandidateGenerator
from slate_rl.simulator.simulator import UserSimulator, initial_state

Policy = Callable[[UserSimulator, torch.Tensor, torch.Tensor, int], torch.Tensor]


def make_policies(
    popularity: torch.Tensor, mmr_diversity: float, novelty_weight: float, seed: int = 0
) -> dict:
    gen = torch.Generator(device=popularity.device).manual_seed(seed)
    return {
        "random": lambda sim, c, s, k: random_slate(c, k, gen),
        "popularity": lambda sim, c, s, k: topk_slate(c, popularity[c], k),
        "retrieval_greedy": lambda sim, c, s, k: c[:, :k],
        "oracle_greedy": lambda sim, c, s, k: topk_slate(c, sim.affinity(c), k),
        "oracle_mmr": lambda sim, c, s, k: mmr_slate(
            c, sim.affinity(c), sim.item_vecs, k, mmr_diversity
        ),
        # Trades immediate affinity for distance from what the user just consumed.
        "oracle_novel": lambda sim, c, s, k: topk_slate(
            c, sim.affinity(c) - novelty_weight * sim.recent_similarity(c), k
        ),
    }


@dataclass
class Rollout:
    alive: torch.Tensor  # [T, B] bool: user was still in the session at step t
    clicked: torch.Tensor  # [T, B] bool
    watch: torch.Tensor  # [T, B]
    fatigue: torch.Tensor  # [T, B]
    satisfaction: torch.Tensor  # [T, B]
    ild: torch.Tensor  # [T, B]
    items: torch.Tensor  # [T, B] clicked item_idx (0 = none)

    @property
    def length(self) -> torch.Tensor:
        return self.alive.sum(0)


@torch.no_grad()
def rollout(
    sim: UserSimulator,
    cand_gen: CandidateGenerator,
    policy: Policy,
    seqs,
    users: torch.Tensor,
    genres: torch.Tensor,
    n_candidates: int = 100,
    slate_size: int = 5,
) -> Rollout:
    """Run one session per user until every user has left (no auto-reset)."""
    sim.reset(*initial_state(seqs, users))
    alive = torch.ones(len(users), dtype=torch.bool, device=sim.device)
    rec = {k: [] for k in ("alive", "clicked", "watch", "fatigue", "satisfaction", "ild", "items")}
    for _ in range(sim.p.max_steps):
        cands, scores = cand_gen(users, sim.history, sim.seen, n_candidates)
        slates = policy(sim, cands, scores, slate_size)
        r = sim.step(slates)
        rec["alive"].append(alive.clone())
        rec["clicked"].append(r.clicked & alive)
        rec["watch"].append(torch.where(alive, r.watch, 0.0))
        rec["fatigue"].append(r.fatigue)
        rec["satisfaction"].append(r.satisfaction)
        rec["ild"].append(intra_list_diversity(slates, genres))
        rec["items"].append(torch.where(alive, r.item, 0))
        alive &= ~r.done
        if not alive.any():
            break
    return Rollout(**{k: torch.stack(v) for k, v in rec.items()})


def summarize(ro: Rollout, genres: torch.Tensor, popularity: torch.Tensor) -> dict[str, float]:
    alive, clicked = ro.alive.float(), ro.clicked.float()
    steps = alive.sum()
    early = alive[:5]
    late = alive[10:]
    consumed = ro.items[ro.items > 0]
    genre_counts = torch.einsum("tb,tbg->bg", clicked, genres[ro.items])
    has_clicks = clicked.sum(0) > 0
    return {
        "session_length": ro.length.float().mean().item(),
        "clicks_per_session": clicked.sum(0).mean().item(),
        "watch_per_session": ro.watch.sum(0).mean().item(),
        "ctr": (clicked.sum() / steps).item(),
        "ctr_first5": ((clicked[:5] * early).sum() / early.sum()).item(),
        "ctr_after10": ((clicked[10:] * late).sum() / late.sum().clamp(min=1)).item(),
        "watch_per_click": (ro.watch.sum() / clicked.sum().clamp(min=1)).item(),
        "ild": ((ro.ild * alive).sum() / steps).item(),
        "fatigue": ((ro.fatigue * alive).sum() / steps).item(),
        "genre_entropy": genre_entropy(genre_counts)[has_clicks].mean().item(),
        "novelty": novelty(consumed, popularity).mean().item() if len(consumed) else 0.0,
        "coverage": (len(torch.unique(consumed)) / (popularity.numel() - 1)),
    }


def per_step_curves(ro: Rollout) -> dict[str, torch.Tensor]:
    alive = ro.alive.float()
    n = alive.sum(1).clamp(min=1)
    return {
        "survival": alive.mean(1),
        "ctr": (ro.clicked.float() * alive).sum(1) / n,
        "fatigue": (ro.fatigue * alive).sum(1) / n,
        "satisfaction": (ro.satisfaction * alive).sum(1) / n,
    }
