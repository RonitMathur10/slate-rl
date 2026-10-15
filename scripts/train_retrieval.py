"""Train the two-tower retrieval model, report Recall/NDCG@K, and save the FAISS retriever.

Usage: uv run python scripts/train_retrieval.py [retrieval.train.epochs=5] [wandb.mode=disabled]
"""

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from slate_rl.data import load_processed
from slate_rl.data.features import ItemFeatures, UserFeatures
from slate_rl.data.sequences import UserSequences
from slate_rl.retrieval import (
    Retriever,
    TrainConfig,
    TwoTower,
    evaluate,
    fit,
    model_score_fn,
    popularity_score_fn,
)
from slate_rl.utils import get_device, init_wandb, seed_everything


@hydra.main(config_path="../configs", config_name="train_retrieval", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    rcfg = cfg.retrieval
    data = load_processed(cfg.data.processed_dir)
    item_feats = ItemFeatures.from_processed(data).to(device)
    demo = UserFeatures.from_processed(data).demo.to(device)
    seqs = UserSequences(data, rcfg.max_hist).to(device)
    model = TwoTower.from_features(item_feats, **rcfg.model).to(device)
    train_cfg = TrainConfig(**OmegaConf.to_container(rcfg.train, resolve=True))

    run = init_wandb(cfg, job_type="retrieval")
    print(f"device={device}  train pairs={len(seqs.train_pairs()[0]):,}")
    fit(model, seqs, demo, train_cfg, log=lambda m: (run.log(m), print(m)))

    ks = train_cfg.eval_ks
    model_users, heldout_users = seqs.model_users(), seqs.heldout_users()
    scorers = {
        "two_tower": model_score_fn(model, demo),
        "popularity": popularity_score_fn(item_feats.popularity),
    }
    results = {}
    for name, fn in scorers.items():
        results[name] = {
            "val": evaluate(fn, seqs, model_users, "val", ks),
            "test": evaluate(fn, seqs, model_users, "test", ks),
            "heldout_test": evaluate(fn, seqs, heldout_users, "test", ks),
        }
    run.summary.update(
        {
            f"{name}/{split}/{m}": v
            for name, r in results.items()
            for split, ms in r.items()
            for m, v in ms.items()
        }
    )

    retriever = Retriever(model, rcfg.max_hist)
    retriever.save(rcfg.artifact_dir, metrics=results)
    if cfg.log_artifact and cfg.wandb.mode != "disabled":
        artifact = wandb.Artifact(f"retrieval-{cfg.data.name}", type="model", metadata=results)
        artifact.add_dir(rcfg.artifact_dir)
        run.log_artifact(artifact)
    run.finish()

    print(f"\nSaved retriever to {rcfg.artifact_dir}")
    header = f"{'model':<12}{'split':<14}" + "".join(f"{f'R@{k}':>9}{f'N@{k}':>9}" for k in ks)
    print(header)
    for name, r in results.items():
        for split, ms in r.items():
            row = "".join(f"{ms[f'recall@{k}']:9.4f}{ms[f'ndcg@{k}']:9.4f}" for k in ks)
            print(f"{name:<12}{split:<14}{row}")


if __name__ == "__main__":
    main()
