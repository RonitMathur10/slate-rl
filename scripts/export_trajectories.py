"""Export full sessions of held-out users for the viewer: one agent vs the greedy-click baseline.

Usage: uv run python scripts/export_trajectories.py [n_users=100] [agent_seed=0]
"""

from pathlib import Path

import hydra
import polars as pl
import torch
from omegaconf import DictConfig, OmegaConf

from slate_rl.agents.baselines import make_baselines
from slate_rl.agents.click_model import load_click_model
from slate_rl.agents.dqn import DQNAgent
from slate_rl.agents.ppo import PPOAgent
from slate_rl.agents.slateq import SlateQAgent
from slate_rl.envs import EnvConfig, load_assets, make_batched_env
from slate_rl.eval.trajectories import record_trajectories
from slate_rl.utils import get_device, seed_everything

LOADERS = {"dqn": DQNAgent.load, "slateq": SlateQAgent.load, "ppo": PPOAgent.load}


@hydra.main(config_path="../configs", config_name="export_trajectories", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    env_cfg = EnvConfig.from_dict(OmegaConf.to_container(cfg.env, resolve=True))
    assets = load_assets(cfg, device)
    users = assets.seqs.heldout_users()[: cfg.n_users]
    env = make_batched_env(
        assets,
        EnvConfig(**{**env_cfg.__dict__, "num_envs": len(users), "autoreset": False}),
        "heldout",
        cfg.seed,
    )
    baselines = make_baselines(env, load_click_model(cfg.click_model, device), cfg.seed)
    policies = {name: baselines[name] for name in cfg.baselines}
    for name in cfg.agents:
        path = Path(cfg.checkpoint_pattern.format(agent=name, seed=cfg.agent_seed))
        kind = torch.load(path, map_location="cpu")["agent"]
        policies[name] = LOADERS[kind](path, device)

    frames = [
        record_trajectories(env, policy, users, assets.titles, name, seed=cfg.seed)
        for name, policy in policies.items()
    ]
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traj = pl.concat(frames)
    traj.write_parquet(out / "trajectories.parquet")

    # Each user's most recent real viewing before the session, for context in the viewer.
    hist = assets.seqs.history(users, assets.seqs.n_train[users])[:, -cfg.history_items :]
    user_rows = [
        {
            "user_idx": int(u),
            "history_titles": [assets.titles[i] for i in row.tolist() if i > 0],
        }
        for u, row in zip(users.cpu(), hist.cpu(), strict=True)
    ]
    pl.DataFrame(user_rows).write_parquet(out / "users.parquet")
    summary = (
        traj.group_by("policy", "user_idx")
        .agg(pl.col("reward").sum().alias("return"), pl.len().alias("length"))
        .group_by("policy")
        .agg(pl.col("return").mean(), pl.col("length").mean())
        .sort("policy")
    )
    print(f"wrote {out}/trajectories.parquet ({traj.height} steps) and users.parquet")
    print(summary)


if __name__ == "__main__":
    main()
