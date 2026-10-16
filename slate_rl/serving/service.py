"""Recommendation service: the trained policies behind a request/response interface.

External ids are MovieLens ``movie_id``s; internally everything uses ``item_idx``. A request
is treated as the start of (or a step in) a session: candidates exclude the given history, and
``shown_movie_ids`` marks impressions already made this session, as in the environment.
"""

from dataclasses import dataclass
from pathlib import Path

import torch

from slate_rl.agents.baselines import make_baselines
from slate_rl.agents.click_model import load_click_model
from slate_rl.agents.dqn import DQNAgent
from slate_rl.agents.ppo import PPOAgent
from slate_rl.agents.slateq import SlateQAgent
from slate_rl.data.features import AGE_BUCKETS
from slate_rl.envs import (
    N_SESSION_FEATURES,
    EnvAssets,
    EnvConfig,
    build_observation,
    load_assets,
    make_batched_env,
)
from slate_rl.eval.trajectories import record_trajectories

LOADERS = {"dqn": DQNAgent.load, "slateq": SlateQAgent.load, "ppo": PPOAgent.load}


@dataclass
class Demographics:
    gender: str = "M"  # "M" or "F"
    age: int = 25  # an ML-1M age code: 1, 18, 25, 35, 45, 50, 56
    occupation: int = 0  # 0-20

    def tensor(self, device) -> torch.Tensor:
        age = AGE_BUCKETS.index(self.age) if self.age in AGE_BUCKETS else 2
        occupation = min(max(self.occupation, 0), 20)
        return torch.tensor([[int(self.gender == "F"), age, occupation]], device=device)


class RecommenderService:
    def __init__(
        self,
        assets: EnvAssets,
        env_cfg: EnvConfig,
        policies: dict,
        movie_ids: list[int],
        device: torch.device | str = "cpu",
    ):
        self.assets, self.env_cfg, self.policies = assets, env_cfg, policies
        self.device = torch.device(device)
        self.item_to_movie = movie_ids  # item_idx -> movie_id (index 0 is padding)
        self.movie_to_item = {m: i for i, m in enumerate(movie_ids) if i > 0}
        # A template env provides the static item features and the env's feature layout.
        self._template = make_batched_env(
            assets, EnvConfig(**{**env_cfg.__dict__, "num_envs": 1, "autoreset": False})
        )

    @classmethod
    def from_config(cls, cfg, device: str = "cpu", agent_seed: int = 0) -> "RecommenderService":
        """Load data, retriever, simulator, click model and every trained agent found."""
        from slate_rl.data import load_processed

        device = torch.device(device)
        assets = load_assets(cfg, device)
        env_cfg = EnvConfig.from_dict(dict(cfg.env))
        movie_ids = [0] * len(assets.titles)
        items = load_processed(cfg.data.processed_dir).items
        for idx, movie in items.select("item_idx", "movie_id").iter_rows():
            movie_ids[idx] = movie
        service = cls(assets, env_cfg, {}, movie_ids, device)
        click_model = None
        if Path(cfg.click_model).exists():
            click_model = load_click_model(cfg.click_model, device)
        service.policies = make_baselines(service._template, click_model, seed=0)
        for path in sorted(Path("artifacts/agents").glob(f"*/seed{agent_seed}/model.pt")):
            name = path.parent.parent.name
            kind = torch.load(path, map_location="cpu")["agent"]
            agent = LOADERS[kind](path, device)
            agent.name = name
            service.policies[name] = agent
        return service

    # ----- id mapping --------------------------------------------------------------------
    def to_items(self, movie_ids: list[int]) -> list[int]:
        unknown = [m for m in movie_ids if m not in self.movie_to_item]
        if unknown:
            raise KeyError(f"unknown movie_id(s): {unknown[:10]}")
        return [self.movie_to_item[m] for m in movie_ids]

    def describe(self, item: int) -> dict:
        return {"movie_id": self.item_to_movie[item], "title": self.assets.titles[item]}

    # ----- endpoints ---------------------------------------------------------------------
    @torch.no_grad()
    def recommend(
        self,
        policy: str,
        history_movie_ids: list[int],
        shown_movie_ids: list[int] | None = None,
        demographics: Demographics | None = None,
        step: int = 0,
    ) -> list[dict]:
        if policy not in self.policies:
            raise KeyError(f"unknown policy {policy!r}; available: {sorted(self.policies)}")
        env = self._template
        n_items = env.item_static.shape[0]
        history_items = self.to_items(history_movie_ids)
        hist = torch.zeros(1, env.hist_len, dtype=torch.long, device=self.device)
        recent = history_items[-env.hist_len :]
        if recent:
            hist[0, -len(recent) :] = torch.tensor(recent, device=self.device)
        seen = torch.zeros(1, n_items, dtype=torch.bool, device=self.device)
        seen[0, history_items] = True
        if int((~seen[0, 1:]).sum()) < env.K:
            raise ValueError("history covers too much of the catalogue to propose candidates")
        shown = torch.zeros_like(seen)
        shown[0, self.to_items(shown_movie_ids or [])] = True
        session = torch.zeros(1, N_SESSION_FEATURES, device=self.device)
        session[0, 0] = min(step, env.max_steps) / env.max_steps
        demo = (demographics or Demographics()).tensor(self.device)
        user_emb = self.assets.cand_gen.embed(hist, demo)
        obs, _ = build_observation(
            self.assets.cand_gen, env.item_static, user_emb, hist, seen, shown, session, env.K
        )
        positions = self.policies[policy].act(obs)[0].tolist()
        scores = obs.cand_feats[0, :, env.item_static.shape[1]]  # retrieval score column
        return [
            {
                "rank": r + 1,
                **self.describe(int(obs.cand_items[0, p])),
                "retrieval_score": round(float(scores[p]), 4),
            }
            for r, p in enumerate(positions)
        ]

    @torch.no_grad()
    def simulate(self, policy: str, user_idx: int | None = None, seed: int = 0) -> dict:
        """Run one simulated session for a held-out user; returns the user's recent history
        and the step-by-step trajectory."""
        if policy not in self.policies:
            raise KeyError(f"unknown policy {policy!r}; available: {sorted(self.policies)}")
        heldout = self.assets.seqs.heldout_users()
        if user_idx is None:
            user_idx = int(heldout[seed % len(heldout)])
        if user_idx not in set(heldout.tolist()):
            raise KeyError(f"user {user_idx} is not a held-out user")
        env = make_batched_env(
            self.assets,
            EnvConfig(**{**self.env_cfg.__dict__, "num_envs": 1, "autoreset": False}),
            user_group="heldout",
            seed=seed,
        )
        users = torch.tensor([user_idx], device=self.device)
        traj = record_trajectories(
            env, self.policies[policy], users, self.assets.titles, policy, seed=seed
        )
        seqs = self.assets.seqs
        hist = seqs.history(users, seqs.n_train[users])[0, -10:].tolist()
        return {
            "user_idx": user_idx,
            "policy": policy,
            "history": [self.describe(i) for i in hist if i > 0],
            "total_reward": float(traj["reward"].sum()),
            "steps": [
                {
                    "step": row["step"],
                    "slate": [self.describe(i) for i in row["slate"]],
                    "clicked_position": row["clicked_position"],
                    "watch": round(row["watch"], 4),
                    "reward": round(row["reward"], 4),
                    "fatigue": round(row["fatigue"], 4),
                    "satisfaction": round(row["satisfaction"], 4),
                }
                for row in traj.iter_rows(named=True)
            ],
        }
