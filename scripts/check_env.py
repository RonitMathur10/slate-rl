"""Check the real-data environment: Gymnasium API compliance, a rendered episode, throughput.

Usage: uv run python scripts/check_env.py [device=cpu] ['benchmark_sizes=[256]']
"""

import time

import hydra
import numpy as np
import torch
from gymnasium.utils.env_checker import check_env
from omegaconf import DictConfig, OmegaConf

from slate_rl.envs import EnvConfig, load_assets, make_batched_env, make_gym_env
from slate_rl.utils import get_device, seed_everything


def random_action(n: int, n_candidates: int, k: int, device) -> torch.Tensor:
    return torch.rand(n, n_candidates, device=device).topk(k, dim=1).indices


@hydra.main(config_path="../configs", config_name="check_env", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    env_cfg = EnvConfig.from_dict(OmegaConf.to_container(cfg.env, resolve=True))
    assets = load_assets(cfg, device)

    gym_env = make_gym_env(assets, env_cfg, seed=cfg.seed, render_mode="ansi")
    check_env(gym_env, skip_render_check=True)
    print("gymnasium check_env: OK")
    print(f"observation space: {gym_env.observation_space}")
    print(f"action space: {gym_env.action_space}\n")

    obs, info = gym_env.reset(seed=cfg.seed)
    print(f"Example episode, held-out user {info['user']}, top-k-by-retrieval policy:")
    total, done = 0.0, False
    while not done:
        obs, reward, term, trunc, info = gym_env.step(np.arange(env_cfg.slate_size))
        total += reward
        print(gym_env.render() + f"   reward={reward:.2f}  fatigue={info['sim']['fatigue']:.2f}")
        done = term or trunc
    print(f"episode return {total:.2f}, ended by {'leaving' if term else 'time limit'}\n")

    print(f"Throughput on {device} (random policy, {cfg.benchmark_steps} steps):")
    for n in cfg.benchmark_sizes:
        env = make_batched_env(assets, EnvConfig(**{**env_cfg.__dict__, "num_envs": n}))
        env.reset(seed=cfg.seed)
        for _ in range(5):  # warm-up
            env.step(random_action(n, env.K, env.k, device))
        returns, lengths = [], []
        if device.type == "mps":
            torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(cfg.benchmark_steps):
            _, _, _, _, info = env.step(random_action(n, env.K, env.k, device))
            if "episode" in info:
                returns.append(info["episode"]["return"])
                lengths.append(info["episode"]["length"].float())
        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.perf_counter() - start
        sps = n * cfg.benchmark_steps / elapsed
        ret, length = torch.cat(returns).mean().item(), torch.cat(lengths).mean().item()
        print(
            f"  num_envs={n:>5}: {sps:>9,.0f} user-steps/s  "
            f"(random policy: return {ret:.2f}, length {length:.1f})"
        )


if __name__ == "__main__":
    main()
