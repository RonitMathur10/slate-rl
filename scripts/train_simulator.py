"""Train the simulator's choice/rating model and calibrate its fatigue reference level.

Usage: uv run python scripts/train_simulator.py [simulator.train.epochs=5] [wandb.mode=disabled]
"""

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from slate_rl.data import load_processed
from slate_rl.data.features import ItemFeatures
from slate_rl.data.sequences import UserSequences
from slate_rl.simulator import (
    SimTrainConfig,
    UserChoiceModel,
    baseline_metrics,
    consecutive_similarity,
    evaluate,
    fit,
    save_choice_model,
)
from slate_rl.utils import get_device, init_wandb, seed_everything


@hydra.main(config_path="../configs", config_name="train_simulator", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    scfg = cfg.simulator
    data = load_processed(cfg.data.processed_dir)
    seqs = UserSequences(data, scfg.model.max_hist).to(device)
    popularity = ItemFeatures.from_processed(data).popularity.to(device)
    model = UserChoiceModel(data.meta["n_items"], **scfg.model).to(device)
    train_cfg = SimTrainConfig(**OmegaConf.to_container(scfg.train, resolve=True))

    run = init_wandb(cfg, job_type="simulator")
    print(f"device={device}  train pairs={len(seqs.train_pairs()[0]):,}")
    fit(model, seqs, train_cfg, log=lambda m: (run.log(m), print(m)))

    heldout = seqs.heldout_users()
    results = {
        "choice_model": {
            "val": evaluate(model, seqs, "val"),
            "test": evaluate(model, seqs, "test"),
            "heldout_test": evaluate(model, seqs, "test", heldout),
        },
        "baseline": {
            "val": baseline_metrics(seqs, popularity, "val"),
            "test": baseline_metrics(seqs, popularity, "test"),
            "heldout_test": baseline_metrics(seqs, popularity, "test", heldout),
        },
    }
    sim_ref = consecutive_similarity(model, seqs, window=scfg.params.sim_window)
    results["sim_ref"] = sim_ref
    save_choice_model(model, scfg.artifact_dir, sim_ref, metrics=results)

    run.summary.update(
        {
            f"{name}/{split}/{m}": v
            for name in ("choice_model", "baseline")
            for split, ms in results[name].items()
            for m, v in ms.items()
        }
        | {"sim_ref": sim_ref}
    )
    if cfg.log_artifact and cfg.wandb.mode != "disabled":
        artifact = wandb.Artifact(f"simulator-{cfg.data.name}", type="model", metadata=results)
        artifact.add_dir(scfg.artifact_dir)
        run.log_artifact(artifact)
    run.finish()

    print(f"\nSaved choice model to {scfg.artifact_dir}  (sim_ref={sim_ref:.4f})")
    for name in ("choice_model", "baseline"):
        for split, ms in results[name].items():
            print(f"{name:<14}{split:<14}" + "  ".join(f"{k}={v:.4f}" for k, v in ms.items()))


if __name__ == "__main__":
    main()
