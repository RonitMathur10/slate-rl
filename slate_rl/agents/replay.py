"""Replay buffer of compact transitions and a vectorised n-step return accumulator."""

from collections import deque
from dataclasses import fields

import torch

from slate_rl.agents.common import CompactObs

_STORE_DTYPES = {
    "user_emb": torch.float16,
    "session": torch.float16,
    "cand_items": torch.int32,
    "cand_dyn": torch.float16,
}


class ReplayBuffer:
    """Circular buffer on device. ``discount`` already folds in gamma^n and termination."""

    def __init__(self, capacity: int, example: CompactObs, slate_size: int, device):
        self.capacity, self.device = capacity, torch.device(device)
        self.ptr, self.size = 0, 0

        def alloc(obs: CompactObs) -> dict[str, torch.Tensor]:
            return {
                f.name: torch.zeros(
                    (capacity, *getattr(obs, f.name).shape[1:]),
                    dtype=_STORE_DTYPES[f.name],
                    device=self.device,
                )
                for f in fields(CompactObs)
            }

        self.obs, self.next_obs = alloc(example), alloc(example)
        self.action = torch.zeros(capacity, slate_size, dtype=torch.int16, device=self.device)
        self.reward = torch.zeros(capacity, device=self.device)
        self.discount = torch.zeros(capacity, device=self.device)

    def add(self, obs, action, reward, discount, next_obs) -> None:
        n = len(reward)
        idx = (self.ptr + torch.arange(n, device=self.device)) % self.capacity
        for f in fields(CompactObs):
            self.obs[f.name][idx] = getattr(obs, f.name).to(_STORE_DTYPES[f.name])
            self.next_obs[f.name][idx] = getattr(next_obs, f.name).to(_STORE_DTYPES[f.name])
        self.action[idx] = action.to(torch.int16)
        self.reward[idx] = reward
        self.discount[idx] = discount
        self.ptr = (self.ptr + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int, gen: torch.Generator | None = None):
        idx = torch.randint(self.size, (batch_size,), generator=gen, device=self.device)

        def load(store):
            return CompactObs(
                store["user_emb"][idx].float(),
                store["session"][idx].float(),
                store["cand_items"][idx].long(),
                store["cand_dyn"][idx].float(),
            )

        return (
            load(self.obs),
            self.action[idx].long(),
            self.reward[idx],
            self.discount[idx],
            load(self.next_obs),
        )


class NStepAccumulator:
    """Turns per-step transitions from N parallel envs into n-step transitions.

    Each pushed step is emitted exactly once, n - 1 pushes later, with
    R = sum_j gamma^j r_j truncated at the first episode end within the window, the
    next observation taken where the window stopped, and discount = gamma^m (0 if terminated).
    ``next_obs`` must already hold the true final observation for truncated rows.
    """

    def __init__(self, n: int, gamma: float):
        self.n, self.gamma = n, gamma
        self.window: deque = deque(maxlen=n)

    def push(self, obs, action, reward, terminated, done, next_obs):
        self.window.append((obs, action, reward, terminated, done, next_obs))
        if len(self.window) < self.n:
            return None
        obs0, action0 = self.window[0][0], self.window[0][1]
        rewards = torch.stack([w[2] for w in self.window])  # [n, N]
        term = torch.stack([w[3] for w in self.window])
        done_ = torch.stack([w[4] for w in self.window])
        n, num = rewards.shape
        dev = rewards.device
        # alive[j] = no episode end strictly before step j
        ended_before = torch.cumsum(done_.float(), 0) - done_.float()
        alive = (ended_before == 0).float()
        gammas = self.gamma ** torch.arange(n, device=dev, dtype=torch.float32)
        ret = (gammas[:, None] * alive * rewards).sum(0)
        # Index of the step where the window stops: first done, else the last step.
        first_done = torch.where(done_.any(0), done_.float().argmax(0), n - 1)
        rows = torch.arange(num, device=dev)
        stacked = CompactObs.stack([w[5] for w in self.window])
        next_obs0 = stacked[first_done, rows]
        discount = self.gamma ** (first_done.float() + 1.0) * (~term[first_done, rows]).float()
        return obs0, action0, ret, discount, next_obs0

    def reset(self) -> None:
        self.window.clear()
