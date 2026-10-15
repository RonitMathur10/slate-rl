"""Batched user simulator: learned preferences plus hand-specified session dynamics.

Per step, for each user and a slate of k items:

    affinity_i = log P(i | catalogue) + log N          (learned; "lift over uniform")
    logit_i    = choice_temp * affinity_i
                 - position_decay * rank_i
                 - fatigue_weight * fatigue * sim_i     (sim_i: cosine to recent items)
                 - repeat_penalty * [i already seen]
    logit_none = no_click_bias

The user picks one item or nothing from softmax(logits). A click appends the item to the
history (so preferences drift), yields a watch fraction from the predicted rating, and moves
fatigue towards how similar the item is to recent consumption. Satisfaction is an EMA of watch
fraction (0 on no-click), and the user leaves with probability
sigmoid(leave_bias - leave_sat * satisfaction + leave_fatigue * fatigue).

MovieLens has no impressions, so everything except the affinity and rating model is a
hand-set parameter (see ``SimParams``); ``sim_ref`` is calibrated from real data.
"""

import math
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

from slate_rl.simulator.model import UserChoiceModel


@dataclass
class SimParams:
    max_steps: int = 50
    # Choice
    choice_temp: float = 1.0
    no_click_bias: float = 5.0
    position_decay: float = 0.1
    repeat_penalty: float = 20.0
    # Fatigue / boredom
    fatigue_weight: float = 2.5
    fatigue_decay: float = 0.85
    fatigue_lr: float = 3.0
    fatigue_max: float = 5.0
    sim_window: int = 5
    sim_ref: float = 0.0  # overwritten with the value calibrated on real data
    # Watch-through
    watch_scale: float = 1.5
    watch_mid: float = 3.0
    watch_fatigue: float = 0.15
    watch_noise: float = 0.1
    # Satisfaction and leaving
    sat_alpha: float = 0.3
    sat_init: float = 0.6
    leave_bias: float = -1.5
    leave_sat: float = 4.0
    leave_fatigue: float = 0.75

    @classmethod
    def from_dict(cls, d: dict) -> "SimParams":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class SimResponse:
    clicked: torch.Tensor  # [B] bool
    position: torch.Tensor  # [B] long; slate position clicked, k = no click
    item: torch.Tensor  # [B] long; clicked item_idx, 0 = none
    watch: torch.Tensor  # [B] float in [0, 1]; 0 on no click
    rating: torch.Tensor  # [B] predicted rating of the clicked item (0 on no click)
    p_click: torch.Tensor  # [B] probability of clicking anything on this slate
    fatigue: torch.Tensor  # [B] after the update
    satisfaction: torch.Tensor  # [B] after the update
    left: torch.Tensor  # [B] bool; user chose to leave (terminal)
    done: torch.Tensor  # [B] bool; user left or hit max_steps


class UserSimulator:
    """Holds the state of a batch of simulated users. Rows can be reset independently."""

    def __init__(
        self,
        model: UserChoiceModel,
        params: SimParams,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.p = params
        self.n_items = model.n_items
        self.hist_len = model.max_hist
        self.log_n = math.log(model.n_items)
        with torch.no_grad():
            self.item_vecs = model.item_vectors().detach()
        gen_device = self.device if self.device.type in ("cpu", "cuda", "mps") else "cpu"
        self.gen = torch.Generator(device=gen_device).manual_seed(seed)
        self.history: torch.Tensor | None = None

    # ----- state -------------------------------------------------------------------------
    def reset(
        self, history: torch.Tensor, seen: torch.Tensor, idx: torch.Tensor | None = None
    ) -> None:
        """Start sessions. ``history`` [B, hist_len] left-padded; ``seen`` [B, n_items+1] bool.

        With ``idx``, only those rows are reset (the batch must already exist).
        """
        history, seen = history.to(self.device), seen.to(self.device)
        if idx is None:
            b = history.shape[0]
            self.history = history.clone()
            self.seen = seen.clone()
            self.fatigue = torch.zeros(b, device=self.device)
            self.satisfaction = torch.full((b,), self.p.sat_init, device=self.device)
            self.t = torch.zeros(b, dtype=torch.long, device=self.device)
            return
        self.history[idx] = history
        self.seen[idx] = seen
        self.fatigue[idx] = 0.0
        self.satisfaction[idx] = self.p.sat_init
        self.t[idx] = 0

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "history": self.history.clone(),
            "seen": self.seen.clone(),
            "fatigue": self.fatigue.clone(),
            "satisfaction": self.satisfaction.clone(),
            "t": self.t.clone(),
        }

    # ----- model pieces ------------------------------------------------------------------
    @torch.no_grad()
    def user_state(self) -> torch.Tensor:
        return self.model.encode(self.history)

    @torch.no_grad()
    def affinity(self, items: torch.Tensor, user: torch.Tensor | None = None) -> torch.Tensor:
        """Learned log-lift over uniform for ``items`` [B, k] -> [B, k] (no dynamics)."""
        user = self.user_state() if user is None else user
        logp = F.log_softmax(self.model.logits(user), dim=1)
        return logp.gather(1, items) + self.log_n

    @torch.no_grad()
    def recent_similarity(self, items: torch.Tensor) -> torch.Tensor:
        """Mean cosine between each of ``items`` [B, k] and the last ``sim_window`` consumed."""
        recent = self.history[:, -self.p.sim_window :]
        sim = torch.einsum("bkd,bwd->bkw", self.item_vecs[items], self.item_vecs[recent])
        mask = (recent > 0).float()[:, None, :]
        return (sim * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)

    @torch.no_grad()
    def click_logits(
        self, slates: torch.Tensor, user: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Choice logits [B, k + 1] (last column = no click) and recent similarity [B, k]."""
        p = self.p
        k = slates.shape[1]
        sim = self.recent_similarity(slates)
        rank = torch.arange(k, device=self.device, dtype=torch.float32)
        logits = (
            p.choice_temp * self.affinity(slates, user)
            - p.position_decay * rank
            - p.fatigue_weight * self.fatigue[:, None] * sim
            - p.repeat_penalty * self.seen.gather(1, slates).float()
        )
        none = torch.full((len(slates), 1), p.no_click_bias, device=self.device)
        return torch.cat([logits, none], dim=1), sim

    # ----- transition --------------------------------------------------------------------
    @torch.no_grad()
    def step(self, slates: torch.Tensor) -> SimResponse:
        """Show one slate [B, k] of item_idx to every user and advance their state."""
        p = self.p
        slates = slates.to(self.device)
        b, k = slates.shape
        user = self.user_state()
        logits, sim = self.click_logits(slates, user)
        probs = torch.softmax(logits, dim=1)
        position = torch.multinomial(probs, 1, generator=self.gen).squeeze(1)
        clicked = position < k
        pos_c = position.clamp(max=k - 1)
        item = torch.where(clicked, slates.gather(1, pos_c[:, None]).squeeze(1), 0)
        sim_c = sim.gather(1, pos_c[:, None]).squeeze(1)

        rating = self.model.predict_rating(user, item[:, None]).squeeze(1)
        noise = torch.randn(b, generator=self.gen, device=self.device) * p.watch_noise
        watch = torch.sigmoid(p.watch_scale * (rating - p.watch_mid))
        watch = watch - p.watch_fatigue * self.fatigue * sim_c.clamp(min=0.0) + noise
        watch = torch.where(clicked, watch.clamp(0.0, 1.0), 0.0)
        rating = torch.where(clicked, rating, 0.0)

        decayed = p.fatigue_decay * self.fatigue
        bumped = decayed + p.fatigue_lr * (sim_c - p.sim_ref)
        self.fatigue = torch.where(clicked, bumped, decayed).clamp(0.0, p.fatigue_max)
        self.satisfaction = (1 - p.sat_alpha) * self.satisfaction + p.sat_alpha * watch

        shifted = torch.cat([self.history[:, 1:], item[:, None]], dim=1)
        self.history = torch.where(clicked[:, None], shifted, self.history)
        rows = torch.nonzero(clicked).squeeze(1)
        self.seen[rows, item[rows]] = True
        self.t = self.t + 1

        p_leave = torch.sigmoid(
            p.leave_bias - p.leave_sat * self.satisfaction + p.leave_fatigue * self.fatigue
        )
        leave = torch.rand(b, generator=self.gen, device=self.device) < p_leave
        done = leave | (self.t >= p.max_steps)
        return SimResponse(
            clicked=clicked,
            position=position,
            item=item,
            watch=watch,
            rating=rating,
            p_click=1.0 - probs[:, -1],
            fatigue=self.fatigue.clone(),
            satisfaction=self.satisfaction.clone(),
            left=leave,
            done=done,
        )


def initial_state(seqs, users: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Session start for ``users``: their history up to the end of their train split.

    Returns (history [B, seqs.max_hist], seen [B, n_items + 1]).
    """
    cut = seqs.n_train[users]
    return seqs.history(users, cut), seqs.items_in_range(users, torch.zeros_like(cut), cut)
