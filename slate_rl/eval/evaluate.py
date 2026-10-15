"""Run a policy for exactly one session per user and report session-level metrics."""

import torch

from slate_rl.envs.batched import BatchedSlateEnv


@torch.no_grad()
def evaluate_policy(
    env: BatchedSlateEnv, policy, users: torch.Tensor, seed: int = 0
) -> dict[str, float | torch.Tensor]:
    """``env`` must have ``autoreset=False`` and ``num_envs == len(users)``.

    Every policy is evaluated on the same users with the same seed, so results are paired.
    Returns scalar means plus ``per_user_return`` [len(users)] for confidence intervals.
    """
    if env.cfg.autoreset or env.num_envs != len(users):
        raise ValueError("evaluation env needs autoreset=False and one row per user")
    obs = env.reset(seed=seed, users=users)
    n = len(users)
    dev = env.device
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    ret, length, clicks, watch = (torch.zeros(n, device=dev) for _ in range(4))
    ild = repeat = nov = torch.zeros((), device=dev)
    steps = torch.zeros((), device=dev)
    consumed = []
    for _ in range(env.max_steps):
        obs, reward, term, trunc, info = env.step(policy.act(obs))
        a = alive.float()
        comps = info["reward_components"]
        ret += reward * a
        length += a
        clicks += comps["click"] * a
        watch += comps["watch"] * a
        ild = ild + (comps["diversity"] * a).sum()
        repeat = repeat + (comps["repeat"] * a).sum()
        nov = nov + (comps["novelty"] * a).sum()
        steps = steps + a.sum()
        consumed.append(info["item"][alive & info["clicked"]])
        alive &= ~(term | trunc)
        if not alive.any():
            break
    consumed = torch.cat(consumed)
    total_clicks = clicks.sum().clamp(min=1.0)
    return {
        "return": ret.mean().item(),
        "session_length": length.mean().item(),
        "clicks": clicks.mean().item(),
        "watch": watch.mean().item(),
        "ctr": (clicks.sum() / steps).item(),
        "ild": (ild / steps).item(),
        "repeat": (repeat / steps).item(),
        "novelty": (nov / total_clicks).item(),
        "coverage": len(torch.unique(consumed)) / (env.item_static.shape[0] - 1),
        "per_user_return": ret.cpu(),
    }
