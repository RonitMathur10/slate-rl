"""Torch-native vectorised slate environment: N simulated users stepped in parallel on device.

Each step the env proposes K candidates per user (two-tower pre-filter), the agent picks k of
them by index, the simulator responds, and the reward combines engagement with diversity,
novelty and a penalty for re-showing items. Done rows are auto-reset with a freshly sampled user.

The observation contains only what a real recommender could know (history, retrieval
embeddings, its own session logs). Simulator internals such as fatigue and satisfaction are
returned in ``info["sim"]`` for analysis and must not be fed to agents.
"""

from dataclasses import dataclass, field, fields

import torch
import torch.nn.functional as F

from slate_rl.eval.metrics import intra_list_diversity, novelty
from slate_rl.retrieval.candidates import CandidateGenerator
from slate_rl.simulator.simulator import UserSimulator, initial_state

N_SESSION_FEATURES = 6
RECENT_WINDOW = 5


@dataclass
class RewardConfig:
    click: float = 1.0
    watch: float = 1.0
    diversity: float = 0.1  # intra-list genre diversity of the shown slate
    novelty: float = 0.1  # normalised self-information of the clicked item
    repeat: float = 0.2  # penalty: share of the slate already shown this session


@dataclass
class EnvConfig:
    num_envs: int = 256
    n_candidates: int = 100
    slate_size: int = 5
    autoreset: bool = True
    reward: RewardConfig = field(default_factory=RewardConfig)

    @classmethod
    def from_dict(cls, d: dict) -> "EnvConfig":
        d = dict(d)
        reward = RewardConfig(**d.pop("reward", {}))
        names = {f.name for f in fields(cls)}
        return cls(reward=reward, **{k: v for k, v in d.items() if k in names})


@dataclass
class SlateObs:
    user_emb: torch.Tensor  # [N, d] retrieval user-tower embedding
    session: torch.Tensor  # [N, N_SESSION_FEATURES]
    history: torch.Tensor  # [N, L] recent consumed item_idx, left-padded
    cand_items: torch.Tensor  # [N, K] candidate item_idx
    cand_feats: torch.Tensor  # [N, K, F] per-candidate features

    def __getitem__(self, idx) -> "SlateObs":
        return SlateObs(*(getattr(self, f.name)[idx] for f in fields(self)))

    def to(self, device) -> "SlateObs":
        return SlateObs(*(getattr(self, f.name).to(device) for f in fields(self)))


class BatchedSlateEnv:
    """Observation features per candidate: retrieval embedding, retrieval score, normalised log
    popularity, cosine to the user's recent items, genre multi-hot. Session features: progress
    (t / max_steps), click-rate EMA, last click, last watch, no-click streak (scaled), and the
    similarity of the last clicked item to the items before it.
    """

    def __init__(
        self,
        sim: UserSimulator,
        cand_gen: CandidateGenerator,
        seqs,
        users: torch.Tensor,
        genres: torch.Tensor,
        popularity: torch.Tensor,
        cfg: EnvConfig,
        seed: int = 0,
    ):
        assert seqs.max_hist == sim.hist_len, "simulator and sequences must share history length"
        assert cfg.slate_size <= cfg.n_candidates
        self.sim, self.cand_gen, self.seqs, self.cfg = sim, cand_gen, seqs, cfg
        self.device = sim.device
        self.num_envs, self.K, self.k = cfg.num_envs, cfg.n_candidates, cfg.slate_size
        self.max_steps = sim.p.max_steps
        self.user_pool = users.to(self.device)
        self.genres = genres.to(self.device)
        log_pop = torch.log1p(popularity.to(self.device))
        self.log_pop = log_pop / log_pop.max()
        all_items = torch.arange(len(popularity), device=self.device)
        nov = novelty(all_items, popularity.to(self.device))
        self.novelty = nov / nov.max()
        self.item_emb = cand_gen.item_emb.to(self.device)
        self.emb_dim = self.item_emb.shape[1]
        self.cand_feat_dim = self.emb_dim + 3 + self.genres.shape[1]
        self.hist_len = sim.hist_len
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self._rows = torch.arange(self.num_envs, device=self.device)

    # ----- episode management -------------------------------------------------------------
    def _sample_users(self, n: int) -> torch.Tensor:
        idx = torch.randint(len(self.user_pool), (n,), generator=self.gen, device=self.device)
        return self.user_pool[idx]

    def _reset_rows(self, idx: torch.Tensor | None) -> None:
        n = self.num_envs if idx is None else len(idx)
        users = self._sample_users(n)
        self.sim.reset(*initial_state(self.seqs, users), idx=idx)
        if idx is None:
            z = lambda: torch.zeros(self.num_envs, device=self.device)  # noqa: E731
            self.users = users
            self.ema_click, self.last_click, self.last_watch = z(), z(), z()
            self.no_click_streak, self.last_sim = z(), z()
            self.ep_return, self.ep_clicks, self.ep_watch = z(), z(), z()
            self.ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            n_items = self.item_emb.shape[0]
            self.shown = torch.zeros(self.num_envs, n_items, dtype=torch.bool, device=self.device)
            return
        self.users[idx] = users
        for t in (
            self.ema_click,
            self.last_click,
            self.last_watch,
            self.no_click_streak,
            self.last_sim,
            self.ep_return,
            self.ep_clicks,
            self.ep_watch,
            self.ep_len,
        ):
            t[idx] = 0
        self.shown[idx] = False

    def reset(self, seed: int | None = None) -> SlateObs:
        if seed is not None:
            self.gen.manual_seed(seed)
            self.sim.gen.manual_seed(seed + 1)
        self._reset_rows(None)
        return self._observe()

    # ----- observation --------------------------------------------------------------------
    @torch.no_grad()
    def _observe(self) -> SlateObs:
        history = self.sim.history
        user_emb = self.cand_gen.user_embedding(self.users, history)
        cands, scores = self.cand_gen.topk(user_emb, self.sim.seen, self.K)
        recent = history[:, -RECENT_WINDOW:]
        mask = (recent > 0).float().unsqueeze(-1)
        recent_vec = F.normalize((self.item_emb[recent] * mask).sum(1), dim=-1)
        cand_emb = self.item_emb[cands]
        recent_sim = torch.einsum("bkd,bd->bk", cand_emb, recent_vec)
        self._cands, self._recent_sim = cands, recent_sim
        cand_feats = torch.cat(
            [
                cand_emb,
                scores.unsqueeze(-1),
                self.log_pop[cands].unsqueeze(-1),
                recent_sim.unsqueeze(-1),
                self.genres[cands],
            ],
            dim=-1,
        )
        session = torch.stack(
            [
                self.sim.t.float() / self.max_steps,
                self.ema_click,
                self.last_click,
                self.last_watch,
                (self.no_click_streak / 5.0).clamp(max=1.0),
                self.last_sim,
            ],
            dim=1,
        )
        return SlateObs(user_emb, session, history.clone(), cands, cand_feats)

    # ----- step ---------------------------------------------------------------------------
    def _dedupe(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace repeated indices with the lowest unused candidate. Returns (action, fixed)."""
        used = torch.zeros(len(action), self.K, dtype=torch.bool, device=self.device)
        rows = torch.arange(len(action), device=self.device)
        out, fixed = action.clone(), torch.zeros(len(action), dtype=torch.bool, device=self.device)
        for j in range(self.k):
            a = out[:, j]
            dup = used[rows, a]
            a = torch.where(dup, (~used).float().argmax(1), a)
            out[:, j] = a
            used[rows, a] = True
            fixed |= dup
        return out, fixed

    @torch.no_grad()
    def step(self, action: torch.Tensor):
        """``action``: [N, k] indices into the current candidates.

        Returns (obs, reward [N], terminated [N], truncated [N], info).
        """
        action = action.to(self.device).long()
        if action.shape != (self.num_envs, self.k):
            raise ValueError(f"action must be [{self.num_envs}, {self.k}], got {action.shape}")
        if (action < 0).any() or (action >= self.K).any():
            raise ValueError("action indices must be in [0, n_candidates)")
        action, fixed = self._dedupe(action)
        slates = self._cands.gather(1, action)
        recent_sim = self._recent_sim.gather(1, action)

        repeat = self.shown.gather(1, slates).float().mean(1)
        self.shown[self._rows[:, None], slates] = True
        diversity = intra_list_diversity(slates, self.genres)
        r = self.sim.step(slates)
        clicked = r.clicked.float()
        comps = {
            "click": clicked,
            "watch": r.watch,
            "diversity": diversity,
            "novelty": torch.where(r.clicked, self.novelty[r.item], 0.0),
            "repeat": repeat,
        }
        w = self.cfg.reward
        reward = (
            w.click * comps["click"]
            + w.watch * comps["watch"]
            + w.diversity * comps["diversity"]
            + w.novelty * comps["novelty"]
            - w.repeat * comps["repeat"]
        )

        pos = r.position.clamp(max=self.k - 1)
        clicked_sim = recent_sim.gather(1, pos[:, None]).squeeze(1)
        self.ema_click = 0.7 * self.ema_click + 0.3 * clicked
        self.last_click, self.last_watch = clicked, r.watch
        self.no_click_streak = torch.where(r.clicked, 0.0, self.no_click_streak + 1.0)
        self.last_sim = torch.where(r.clicked, clicked_sim, self.last_sim)
        self.ep_return += reward
        self.ep_clicks += clicked
        self.ep_watch += r.watch
        self.ep_len += 1

        terminated, truncated = r.left, r.done & ~r.left
        info = {
            "reward_components": comps,
            "slate": slates,
            "clicked": r.clicked,
            "position": r.position,
            "item": r.item,
            "watch": r.watch,
            "users": self.users.clone(),
            "action_fixed": fixed,
            "sim": {"fatigue": r.fatigue, "satisfaction": r.satisfaction, "p_click": r.p_click},
            "done": r.done,
        }
        done_idx = torch.nonzero(r.done).squeeze(1)
        if len(done_idx):
            info["episode"] = {
                "return": self.ep_return[done_idx].clone(),
                "length": self.ep_len[done_idx].clone(),
                "clicks": self.ep_clicks[done_idx].clone(),
                "watch": self.ep_watch[done_idx].clone(),
                "users": self.users[done_idx].clone(),
                "index": done_idx,
            }
        if self.cfg.autoreset and len(done_idx):
            if truncated.any():
                info["final_obs"] = self._observe()  # for bootstrapping truncated rows
            self._reset_rows(done_idx)
        return self._observe(), reward, terminated, truncated, info
