"""Build environments from the trained artifacts (processed data, retriever, simulator)."""

from dataclasses import dataclass

import torch
from omegaconf import DictConfig, OmegaConf

from slate_rl.data import load_processed
from slate_rl.data.features import ItemFeatures, UserFeatures
from slate_rl.data.sequences import UserSequences
from slate_rl.envs.batched import BatchedSlateEnv, EnvConfig
from slate_rl.envs.gym_env import SlateRecEnv
from slate_rl.retrieval import CandidateGenerator, Retriever
from slate_rl.simulator import load_choice_model
from slate_rl.simulator.simulator import SimParams, UserSimulator


@dataclass
class EnvAssets:
    """Everything loaded once and shared between env instances."""

    seqs: UserSequences
    cand_gen: CandidateGenerator
    item_feats: ItemFeatures
    choice_model: torch.nn.Module
    sim_params: SimParams
    titles: list[str]


def load_assets(cfg: DictConfig, device: torch.device) -> EnvAssets:
    data = load_processed(cfg.data.processed_dir)
    choice_model, sim_ref = load_choice_model(cfg.simulator.artifact_dir, device)
    params = OmegaConf.to_container(cfg.simulator.params, resolve=True)
    retriever = Retriever.load(cfg.retrieval.artifact_dir, device=device)
    demo = UserFeatures.from_processed(data).demo.to(device)
    titles = [""] * (data.meta["n_items"] + 1)
    for idx, title, year in data.items.select("item_idx", "title", "year").iter_rows():
        titles[idx] = f"{title} ({year})"
    return EnvAssets(
        seqs=UserSequences(data, choice_model.max_hist).to(device),
        cand_gen=CandidateGenerator(retriever.model, demo),
        item_feats=ItemFeatures.from_processed(data).to(device),
        choice_model=choice_model,
        sim_params=SimParams.from_dict({**params, "sim_ref": sim_ref}),
        titles=titles,
    )


def make_batched_env(
    assets: EnvAssets,
    env_cfg: EnvConfig,
    user_group: str = "model",
    seed: int = 0,
    sim_overrides: dict | None = None,
) -> BatchedSlateEnv:
    """``user_group``: ``model`` (training users) or ``heldout`` (evaluation users).

    ``sim_overrides`` perturbs simulator parameters (robustness evaluation).
    """
    params = SimParams.from_dict({**assets.sim_params.__dict__, **(sim_overrides or {})})
    device = assets.seqs.device
    sim = UserSimulator(assets.choice_model, params, device=device, seed=seed + 1)
    users = {"model": assets.seqs.model_users, "heldout": assets.seqs.heldout_users}[user_group]()
    return BatchedSlateEnv(
        sim,
        assets.cand_gen,
        assets.seqs,
        users,
        assets.item_feats.genres,
        assets.item_feats.popularity,
        env_cfg,
        seed=seed,
    )


def make_gym_env(
    assets: EnvAssets,
    env_cfg: EnvConfig,
    user_group: str = "heldout",
    seed: int = 0,
    render_mode: str | None = None,
) -> SlateRecEnv:
    single = EnvConfig(
        num_envs=1,
        n_candidates=env_cfg.n_candidates,
        slate_size=env_cfg.slate_size,
        autoreset=False,
        reward=env_cfg.reward,
    )
    batched = make_batched_env(assets, single, user_group, seed)
    return SlateRecEnv(batched, titles=assets.titles, render_mode=render_mode)
