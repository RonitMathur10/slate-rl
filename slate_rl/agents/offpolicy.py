"""Shared training loop for off-policy, value-based slate agents (DQN, SlateQ)."""

import time
from collections.abc import Callable
from pathlib import Path

import torch

from slate_rl.agents.common import CompactObs
from slate_rl.agents.replay import NStepAccumulator, ReplayBuffer


def train_off_policy(
    agent,
    env,
    cfg,
    evaluate: Callable[[object], dict] | None = None,
    log: Callable[[dict, int], None] = lambda m, step: None,
    checkpoint: str | Path | None = None,
) -> dict:
    """Train a value-based slate agent on the auto-resetting ``env``.

    Evaluates the greedy policy every ``eval_interval`` iterations and keeps the checkpoint with
    the best evaluation return. Agents provide ``act``, ``update``, ``save``, ``set_train``,
    ``stored_action(action, info)`` (what goes into replay) and ``stored_action_dim_offset``.
    """
    obs = env.reset()
    compact = CompactObs.from_obs(obs)
    buffer = ReplayBuffer(
        cfg.buffer_size, compact, env.k + agent.stored_action_dim_offset, env.device
    )
    nstep = NStepAccumulator(cfg.n_step, cfg.gamma)
    decay_iters = max(1, int(cfg.eps_fraction * cfg.total_iters))
    best, best_metrics = -float("inf"), {}
    recent_returns: list[torch.Tensor] = []
    start = time.perf_counter()
    for it in range(1, cfg.total_iters + 1):
        frac = min(1.0, it / decay_iters)
        agent.epsilon = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_compact = CompactObs.from_obs(next_obs)
        if "final_obs" in info:  # true next state for truncated rows (before auto-reset)
            next_compact = CompactObs.where(
                truncated, CompactObs.from_obs(info["final_obs"]), next_compact
            )
        stored = agent.stored_action(action, info)
        out = nstep.push(compact, stored, reward, terminated, terminated | truncated, next_compact)
        if out is not None:
            buffer.add(*out)
        obs, compact = next_obs, CompactObs.from_obs(next_obs)
        if "episode" in info:
            recent_returns.append(info["episode"]["return"])

        metrics: dict[str, float] = {}
        if buffer.size >= cfg.learning_starts:
            agent.set_train(True)
            for _ in range(cfg.updates_per_iter):
                metrics = agent.update(buffer.sample(cfg.batch_size, agent.gen))

        if it % cfg.log_interval == 0:
            env_steps = it * env.num_envs
            row = {f"train/{k}": v for k, v in metrics.items()}
            row |= {
                "train/epsilon": agent.epsilon,
                "train/buffer": buffer.size,
                "train/sps": env_steps / (time.perf_counter() - start),
            }
            if recent_returns:
                row["train/episode_return"] = torch.cat(recent_returns).mean().item()
                recent_returns = []
            log(row, env_steps)

        if evaluate is not None and (it % cfg.eval_interval == 0 or it == cfg.total_iters):
            eps, agent.epsilon = agent.epsilon, 0.0
            agent.set_train(False)
            result = evaluate(agent)
            agent.epsilon = eps
            log(
                {f"eval/{k}": v for k, v in result.items() if isinstance(v, float)},
                it * env.num_envs,
            )
            if result["return"] > best:
                best, best_metrics = result["return"], result
                if checkpoint is not None:
                    agent.save(
                        checkpoint,
                        {
                            "eval": {k: v for k, v in result.items() if isinstance(v, float)},
                            "iter": it,
                        },
                    )
    return best_metrics
