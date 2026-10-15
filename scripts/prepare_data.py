"""Download MovieLens-1M, index and split it, and write processed parquet files.

Usage: uv run python scripts/prepare_data.py [wandb.mode=disabled] [data.heldout_user_frac=0.2]
"""

import hydra
from omegaconf import DictConfig

from slate_rl.data import SplitConfig, download_ml1m, load_movies, load_ratings, load_users
from slate_rl.data.preprocess import preprocess, save_processed
from slate_rl.utils import init_wandb, seed_everything


@hydra.main(config_path="../configs", config_name="prepare_data", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    ml_dir = download_ml1m(cfg.data.raw_dir)
    split_cfg = SplitConfig(
        val_frac=cfg.data.val_frac,
        test_frac=cfg.data.test_frac,
        heldout_user_frac=cfg.data.heldout_user_frac,
        min_interactions=cfg.data.min_interactions,
        seed=cfg.seed,
    )
    data = preprocess(load_ratings(ml_dir), load_movies(ml_dir), load_users(ml_dir), split_cfg)
    save_processed(data, cfg.data.processed_dir)

    run = init_wandb(cfg, job_type="data")
    run.summary.update({k: v for k, v in data.meta.items() if k not in ("genres", "config")})
    run.finish()

    print(f"Wrote {cfg.data.processed_dir}")
    for key in ("n_users", "n_items", "n_interactions", "split_counts", "group_user_counts"):
        print(f"  {key}: {data.meta[key]}")


if __name__ == "__main__":
    main()
