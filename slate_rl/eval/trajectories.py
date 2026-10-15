"""Record full session trajectories (one session per user) for the viewer and analysis."""

import polars as pl
import torch

from slate_rl.envs.batched import BatchedSlateEnv


@torch.no_grad()
def record_trajectories(
    env: BatchedSlateEnv,
    policy,
    users: torch.Tensor,
    titles: list[str],
    policy_name: str,
    seed: int = 0,
) -> pl.DataFrame:
    """One row per (user, step) while the user is in session.

    Columns: policy, user_idx, step, slate (item_idx list), slate_titles, clicked_position
    (-1 = no click), clicked_item, clicked_title, watch, reward, reward components, and the
    simulator's hidden state after the step (fatigue, satisfaction, p_click) for analysis.
    """
    if env.cfg.autoreset or env.num_envs != len(users):
        raise ValueError("needs an env with autoreset=False and one row per user")
    obs = env.reset(seed=seed, users=users)
    alive = torch.ones(len(users), dtype=torch.bool, device=env.device)
    rows: dict[str, list] = {
        k: []
        for k in (
            "user_idx",
            "step",
            "slate",
            "clicked_position",
            "clicked_item",
            "watch",
            "reward",
            "click",
            "diversity",
            "novelty",
            "repeat",
            "fatigue",
            "satisfaction",
            "p_click",
            "terminated",
            "truncated",
        )
    }
    for step in range(env.max_steps):
        obs, reward, term, trunc, info = env.step(policy.act(obs))
        idx = torch.nonzero(alive).squeeze(1)
        comps, sim = info["reward_components"], info["sim"]
        pos = torch.where(info["clicked"], info["position"], -1)
        cols = {
            "user_idx": users[idx],
            "slate": info["slate"][idx],
            "clicked_position": pos[idx],
            "clicked_item": info["item"][idx],
            "watch": info["watch"][idx],
            "reward": reward[idx],
            "click": comps["click"][idx],
            "diversity": comps["diversity"][idx],
            "novelty": comps["novelty"][idx],
            "repeat": comps["repeat"][idx],
            "fatigue": sim["fatigue"][idx],
            "satisfaction": sim["satisfaction"][idx],
            "p_click": sim["p_click"][idx],
            "terminated": term[idx],
            "truncated": trunc[idx],
        }
        for key, value in cols.items():
            rows[key].extend(value.cpu().tolist())
        rows["step"].extend([step] * len(idx))
        alive &= ~(term | trunc)
        if not alive.any():
            break
    df = pl.DataFrame(rows)
    return df.with_columns(
        pl.lit(policy_name).alias("policy"),
        pl.col("slate")
        .map_elements(lambda s: [titles[i] for i in s], return_dtype=pl.List(pl.String))
        .alias("slate_titles"),
        pl.col("clicked_item")
        .map_elements(lambda i: titles[i] if i > 0 else None, return_dtype=pl.String)
        .alias("clicked_title"),
    ).select("policy", "user_idx", "step", pl.all().exclude("policy", "user_idx", "step"))
