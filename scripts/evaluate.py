"""Evaluate baselines and trained agents on the held-out users (one session each, paired).

Usage: uv run python scripts/evaluate.py ['agents={dqn: artifacts/agents/dqn/seed0/model.pt}']
"""

import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from slate_rl.agents.baselines import make_baselines
from slate_rl.agents.click_model import load_click_model
from slate_rl.agents.dqn import DQNAgent
from slate_rl.envs import EnvConfig, load_assets, make_batched_env
from slate_rl.eval.evaluate import evaluate_policy
from slate_rl.eval.stats import paired_difference
from slate_rl.utils import get_device, seed_everything

LOADERS = {"dqn": DQNAgent.load}
COLUMNS = [
    "return",
    "session_length",
    "clicks",
    "watch",
    "ctr",
    "ild",
    "repeat",
    "novelty",
    "coverage",
]


def load_agent(path: str, device):
    kind = torch.load(path, map_location="cpu")["agent"]
    return LOADERS[kind](path, device)


@hydra.main(config_path="../configs", config_name="evaluate", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    env_cfg = EnvConfig.from_dict(OmegaConf.to_container(cfg.env, resolve=True))
    assets = load_assets(cfg, device)
    users = assets.seqs.heldout_users()
    env = make_batched_env(
        assets,
        EnvConfig(**{**env_cfg.__dict__, "num_envs": len(users), "autoreset": False}),
        user_group="heldout",
        seed=cfg.seed,
    )
    click_model = None
    if Path(cfg.click_model).exists():
        click_model = load_click_model(cfg.click_model, device)
    policies = make_baselines(env, click_model, cfg.seed, cfg.mmr_diversity)
    for name, path in (cfg.agents or {}).items():
        if Path(path).exists():
            agent = load_agent(path, device)
            agent.name = name
            policies[name] = agent
        else:
            print(f"skipping {name}: {path} not found")

    results = {}
    for name, policy in policies.items():
        results[name] = evaluate_policy(env, policy, users, seed=cfg.seed)

    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps(
            {n: {k: v for k, v in r.items() if k in COLUMNS} for n, r in results.items()}, indent=2
        )
    )
    header = f"{'policy':<14}" + "".join(f"{c[:10]:>11}" for c in COLUMNS)
    print(f"Held-out users: {len(users)}\n{header}")
    for name, r in results.items():
        print(f"{name:<14}" + "".join(f"{r[c]:11.3f}" for c in COLUMNS))

    # Paired comparisons: same users and seed, so per-user differences are resampled jointly.
    baselines = [b for b in cfg.compare_to if b in results]
    learned = [n for n in results if n in (cfg.agents or {})]
    if learned and baselines:
        print("\nReturn difference vs baselines (mean [95% bootstrap CI] over held-out users):")
        comparisons = {}
        for name in learned:
            for base in baselines:
                diff = paired_difference(
                    results[name]["per_user_return"], results[base]["per_user_return"]
                )
                comparisons[f"{name}-vs-{base}"] = diff
                print(f"  {name} - {base:<13} {diff[0]:+.3f} [{diff[1]:+.3f}, {diff[2]:+.3f}]")
        (out / "comparisons.json").write_text(json.dumps(comparisons, indent=2))
    torch.save({n: r["per_user_return"] for n, r in results.items()}, out / "per_user_return.pt")


if __name__ == "__main__":
    main()
