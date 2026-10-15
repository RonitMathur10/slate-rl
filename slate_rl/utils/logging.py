from typing import Any

import wandb
from omegaconf import DictConfig, OmegaConf


def init_wandb(cfg: DictConfig, job_type: str, name: str | None = None) -> Any:
    """Start a W&B run from the ``wandb`` config group, recording the full Hydra config."""
    return wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        mode=cfg.wandb.mode,
        tags=list(cfg.wandb.tags),
        job_type=job_type,
        name=name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
