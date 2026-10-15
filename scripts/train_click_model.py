"""Log exploratory slates from the environment and fit the agent-side click model.

Usage: uv run python scripts/train_click_model.py [n_iters=50] [wandb.mode=disabled]
"""

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from slate_rl.agents.click_model import ClickModel, collect_logs, fit_click_model, save_click_model
from slate_rl.agents.common import encoder_kwargs
from slate_rl.envs import EnvConfig, load_assets, make_batched_env
from slate_rl.utils import get_device, init_wandb, seed_everything


@hydra.main(config_path="../configs", config_name="train_click_model", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    env_cfg = EnvConfig.from_dict(OmegaConf.to_container(cfg.env, resolve=True))
    env_cfg.num_envs = cfg.num_envs
    assets = load_assets(cfg, device)
    env = make_batched_env(assets, env_cfg, user_group="model", seed=cfg.seed + 100)

    logs = collect_logs(env, cfg.n_iters, seed=cfg.seed)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    train, val = logs.split(cfg.val_frac, gen)
    print(f"logged {len(logs):,} slates (train {len(train):,}, val {len(val):,})")

    model = ClickModel(**encoder_kwargs(env), slate_size=env.k, hidden=cfg.hidden).to(device)
    model.encoder.item_static.copy_(env.item_static)
    run = init_wandb(cfg, job_type="click-model")
    metrics = fit_click_model(
        model,
        train,
        val,
        cfg.epochs,
        cfg.batch_size,
        cfg.lr,
        gen=gen,
        log=lambda m: (run.log(m), print(m)),
    )
    save_click_model(model, cfg.artifact_path, metrics)
    run.summary.update({f"best/{k}": v for k, v in metrics.items()})
    run.finish()
    print(f"\nSaved {cfg.artifact_path}: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))


if __name__ == "__main__":
    main()
