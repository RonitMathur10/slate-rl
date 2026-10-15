"""Train an RL agent in the batched environment, logging curves to W&B.

Usage: uv run python scripts/train_agent.py agent=dqn seed=0 [agent.total_iters=500]
"""

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from slate_rl.agents.dqn import DQNAgent, DQNConfig, train_dqn
from slate_rl.envs import EnvConfig, load_assets, make_batched_env
from slate_rl.eval.evaluate import evaluate_policy
from slate_rl.utils import get_device, init_wandb, seed_everything

AGENTS = {"dqn": (DQNAgent, DQNConfig, train_dqn)}


@hydra.main(config_path="../configs", config_name="train_agent", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    agent_cls, cfg_cls, train_fn = AGENTS[cfg.agent.name]
    agent_cfg = cfg_cls(
        **{
            k: v
            for k, v in OmegaConf.to_container(cfg.agent).items()
            if k in cfg_cls.__dataclass_fields__
        }
    )
    env_cfg = EnvConfig.from_dict(OmegaConf.to_container(cfg.env, resolve=True))
    assets = load_assets(cfg, device)

    train_env = make_batched_env(
        assets,
        EnvConfig(**{**env_cfg.__dict__, "num_envs": cfg.agent.num_envs}),
        user_group="model",
        seed=cfg.seed,
    )
    gen = torch.Generator(device=device).manual_seed(10_000)  # same val users for every run
    pool = assets.seqs.model_users()
    val_users = pool[torch.randperm(len(pool), generator=gen, device=device)[: cfg.val_users]]
    val_env = make_batched_env(
        assets,
        EnvConfig(**{**env_cfg.__dict__, "num_envs": len(val_users), "autoreset": False}),
        user_group="model",
        seed=10_000,
    )

    agent = agent_cls.for_env(train_env, agent_cfg, seed=cfg.seed)
    run = init_wandb(cfg, job_type="train", name=f"{cfg.agent.name}-seed{cfg.seed}")

    def log(metrics: dict, step: int) -> None:
        run.log(metrics, step=step)
        if any(k.startswith("eval/") for k in metrics):
            print(
                f"[{step:>9,}] "
                + "  ".join(f"{k[5:]}={v:.3f}" for k, v in metrics.items() if k.startswith("eval/"))
            )

    best = train_fn(
        agent,
        train_env,
        agent_cfg,
        evaluate=lambda a: evaluate_policy(val_env, a, val_users, seed=10_000),
        log=log,
        checkpoint=cfg.checkpoint,
    )
    run.summary.update({f"best_val/{k}": v for k, v in best.items() if isinstance(v, float)})
    run.finish()
    print(f"\nBest validation return {best['return']:.3f}; checkpoint at {cfg.checkpoint}")


if __name__ == "__main__":
    main()
