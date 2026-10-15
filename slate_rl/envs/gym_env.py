"""Single-user Gymnasium wrapper around ``BatchedSlateEnv`` (API compliance, viewer, serving).

Training uses the batched env directly; this wrapper exists so the environment can be checked
with ``gymnasium.utils.env_checker`` and stepped one user at a time.
"""

from dataclasses import fields

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from slate_rl.envs.batched import N_SESSION_FEATURES, BatchedSlateEnv, SlateObs


class SlateRecEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        env: BatchedSlateEnv,
        titles: list[str] | None = None,
        render_mode: str | None = None,
    ):
        if env.num_envs != 1 or env.cfg.autoreset:
            raise ValueError("SlateRecEnv needs a BatchedSlateEnv with num_envs=1, autoreset=False")
        self.env, self.titles, self.render_mode = env, titles, render_mode
        n_items = env.item_emb.shape[0] - 1
        # All float features are bounded: embeddings and cosines lie in [-1, 1]; popularity,
        # genres and the session features in [0, 1] (last-item similarity in [-1, 1]).
        self.observation_space = spaces.Dict(
            {
                "user_emb": spaces.Box(-1.0, 1.0, (env.emb_dim,), np.float32),
                "session": spaces.Box(-1.0, 1.0, (N_SESSION_FEATURES,), np.float32),
                "history": spaces.Box(0, n_items, (env.hist_len,), np.int64),
                "cand_items": spaces.Box(1, n_items, (env.K,), np.int64),
                "cand_feats": spaces.Box(-1.0, 1.0, (env.K, env.cand_feat_dim), np.float32),
            }
        )
        self.action_space = spaces.MultiDiscrete([env.K] * env.k)
        self._last: dict | None = None

    @staticmethod
    def _to_numpy(obs: SlateObs) -> dict[str, np.ndarray]:
        out = {}
        for f in fields(obs):
            arr = getattr(obs, f.name)[0].cpu().numpy()
            if arr.dtype.kind in "iu":
                out[f.name] = arr.astype(np.int64)
            else:  # clip float round-off just outside the unit bounds
                out[f.name] = np.clip(arr, -1.0, 1.0).astype(np.float32)
        return out

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        obs = self.env.reset(seed=seed)
        self._last = None
        return self._to_numpy(obs), {"user": int(self.env.users[0])}

    def step(self, action):
        act = torch.as_tensor(np.asarray(action), dtype=torch.long).reshape(1, -1)
        obs, reward, terminated, truncated, info = self.env.step(act)
        self._last = info
        out_info = {
            "slate": info["slate"][0].tolist(),
            "clicked": bool(info["clicked"][0]),
            "item": int(info["item"][0]),
            "watch": float(info["watch"][0]),
            "action_fixed": bool(info["action_fixed"][0]),
            "reward_components": {k: float(v[0]) for k, v in info["reward_components"].items()},
            "sim": {k: float(v[0]) for k, v in info["sim"].items()},
        }
        return (
            self._to_numpy(obs),
            float(reward[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            out_info,
        )

    def render(self):
        if self.render_mode != "ansi" or self._last is None:
            return None
        info = self._last
        slate = info["slate"][0].tolist()
        pos = int(info["position"][0])
        lines = [f"step {int(self.env.sim.t[0])}"]
        for i, item in enumerate(slate):
            name = self.titles[item] if self.titles else f"item {item}"
            mark = "*" if i == pos else " "
            lines.append(f" {mark} {i + 1}. {name}")
        lines.append(
            f"   watch={float(info['watch'][0]):.2f}" if pos < len(slate) else "   no click"
        )
        return "\n".join(lines)
