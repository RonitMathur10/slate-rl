"""Validate the user simulator and write reports/simulator/report.md with figures.

Checks: (1) the learned choice/rating model beats simple baselines on held-out data,
(2) predicted ratings are calibrated, (3) simulated consumption resembles real consumption,
(4) the dynamics create a long-term trade-off (myopic greedy loses to a novelty-aware policy).

Usage: uv run python scripts/validate_simulator.py [wandb.mode=disabled]
"""

import json
from pathlib import Path

import hydra
import matplotlib
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from slate_rl.data import load_processed  # noqa: E402
from slate_rl.data.features import ItemFeatures, UserFeatures  # noqa: E402
from slate_rl.data.sequences import UserSequences  # noqa: E402
from slate_rl.eval import genre_entropy, novelty  # noqa: E402
from slate_rl.retrieval import CandidateGenerator, Retriever  # noqa: E402
from slate_rl.simulator import load_simulator  # noqa: E402
from slate_rl.simulator.validation import (  # noqa: E402
    make_policies,
    per_step_curves,
    rollout,
    summarize,
)
from slate_rl.utils import get_device, init_wandb, seed_everything  # noqa: E402

POLICY_ORDER = [
    "random",
    "popularity",
    "retrieval_greedy",
    "oracle_greedy",
    "oracle_mmr",
    "oracle_novel",
]
TABLE_COLS = [
    ("session_length", "Session len"),
    ("clicks_per_session", "Clicks"),
    ("watch_per_session", "Watch"),
    ("ctr", "CTR"),
    ("ctr_step0", "CTR@step0"),
    ("ctr_after10", "CTR after 10"),
    ("watch_per_click", "Watch/click"),
    ("ild", "ILD"),
    ("fatigue", "Fatigue"),
    ("genre_entropy", "Genre H"),
    ("novelty", "Novelty"),
    ("coverage", "Coverage"),
]


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return (a * torch.log2(a.clamp(min=1e-12) / b.clamp(min=1e-12))).sum()

    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).item()


def first_n_clicks(items: torch.Tensor, n: int) -> torch.Tensor:
    """Clicked items [T, B] (0 = none) -> each user's first n clicks [B, n], 0-padded."""
    rows = []
    for row in items.T:
        clicks = row[row > 0][:n]
        rows.append(torch.nn.functional.pad(clicks, (0, n - len(clicks))))
    return torch.stack(rows)


@torch.no_grad()
def rating_calibration(sim, seqs) -> tuple[list[int], list[float], list[int]]:
    """Mean predicted rating grouped by true rating, on held-out users' test positions."""
    users, pos = seqs.split_pairs("test", seqs.heldout_users())
    user = sim.model.encode(seqs.history(users, pos))
    pred = sim.model.predict_rating(user, seqs.item_at(users, pos)[:, None])[:, 0]
    true = seqs.ratings[users, seqs.max_hist + pos]
    levels = [1, 2, 3, 4, 5]
    means = [pred[true == r].mean().item() for r in levels]
    counts = [int((true == r).sum()) for r in levels]
    return levels, means, counts


@hydra.main(config_path="../configs", config_name="validate_simulator", version_base="1.3")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    data = load_processed(cfg.data.processed_dir)
    feats = ItemFeatures.from_processed(data).to(device)
    demo = UserFeatures.from_processed(data).demo.to(device)
    seqs = UserSequences(data, cfg.simulator.model.max_hist).to(device)
    retriever = Retriever.load(cfg.retrieval.artifact_dir, device=device)
    cand_gen = CandidateGenerator(retriever.model, demo)
    params = OmegaConf.to_container(cfg.simulator.params, resolve=True)
    sim = load_simulator(cfg.simulator.artifact_dir, params, device=device, seed=cfg.seed)
    users = seqs.heldout_users()[: cfg.n_users]
    model_metrics = json.loads((Path(cfg.simulator.artifact_dir) / "metrics.json").read_text())

    # ---- policy rollouts ---------------------------------------------------------------
    policies = make_policies(feats.popularity, cfg.mmr_diversity, cfg.novelty_weight, cfg.seed)
    summaries, curves, rollouts = {}, {}, {}
    for name in POLICY_ORDER:
        sim.gen.manual_seed(cfg.seed)
        ro = rollout(
            sim,
            cand_gen,
            policies[name],
            seqs,
            users,
            feats.genres,
            cfg.n_candidates,
            cfg.slate_size,
        )
        rollouts[name] = ro
        summaries[name] = summarize(ro, feats.genres, feats.popularity)
        summaries[name]["ctr_step0"] = ro.clicked[0].float().mean().item()
        curves[name] = {k: v.cpu() for k, v in per_step_curves(ro).items()}

    # ---- real vs simulated consumption (held-out users, first N consumptions) ----------
    # Each user's real items are truncated to as many as they consumed in simulation, so
    # count-sensitive statistics like genre entropy are compared like for like.
    h = cfg.real_horizon
    sim_items = first_n_clicks(rollouts["retrieval_greedy"].items, h)
    n_sim = (sim_items > 0).sum(1)
    start = seqs.n_train[users]
    offsets = torch.arange(h, device=device)
    real_items = seqs.items[users[:, None], seqs.max_hist + start[:, None] + offsets]
    real_items = torch.where(offsets < n_sim[:, None], real_items, 0)
    matched = (n_sim > 0) & ((real_items > 0).sum(1) == n_sim)

    def genre_counts(items):
        return torch.einsum("bn,bng->bg", (items > 0).float(), feats.genres[items])

    real_genres, sim_genres = genre_counts(real_items[matched]), genre_counts(sim_items[matched])
    realism = {
        "users_compared": int(matched.sum()),
        "genre_js_divergence": js_divergence(real_genres.sum(0), sim_genres.sum(0)),
        "real_genre_entropy": genre_entropy(real_genres).mean().item(),
        "sim_genre_entropy": genre_entropy(sim_genres).mean().item(),
        "real_novelty": novelty(real_items[real_items > 0], feats.popularity).mean().item(),
        "sim_novelty": novelty(sim_items[sim_items > 0], feats.popularity).mean().item(),
        "sim_items_per_user": n_sim.float().mean().item(),
    }

    levels, pred_means, counts = rating_calibration(sim, seqs)

    # ---- sanity checks -----------------------------------------------------------------
    g, n = summaries["oracle_greedy"], summaries["oracle_novel"]
    cm, base = (
        model_metrics["choice_model"]["heldout_test"],
        model_metrics["baseline"]["heldout_test"],
    )
    checks = {
        "choice model beats popularity NLL": cm["nll"] < base["nll"],
        "rating model beats global-mean RMSE": cm["rating_rmse"] < base["rating_rmse"],
        "greedy wins on immediate CTR (step 0)": g["ctr_step0"] > n["ctr_step0"],
        "novelty-aware policy has longer sessions": n["session_length"] > g["session_length"],
        "novelty-aware policy has more total watch": n["watch_per_session"]
        > g["watch_per_session"],
        "every informed policy clicks more than random": all(
            summaries[p]["clicks_per_session"] > summaries["random"]["clicks_per_session"]
            for p in POLICY_ORDER[1:]
        ),
    }

    # ---- figures -----------------------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for name in POLICY_ORDER:
        c = curves[name]
        steps = range(len(c["survival"]))
        for ax, key in zip(axes, ("survival", "ctr", "fatigue", "satisfaction"), strict=True):
            ax.plot(steps, c[key].numpy(), label=name)
            ax.set_title(key)
            ax.set_xlabel("step")
            ax.set_xlim(0, 30)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "session_curves.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    genre_names = data.meta["genres"]
    x = torch.arange(len(genre_names))
    rg = (real_genres.sum(0) / real_genres.sum()).cpu()
    sg = (sim_genres.sum(0) / sim_genres.sum()).cpu()
    axes[0].bar(x - 0.2, rg, 0.4, label="real (next consumptions)")
    axes[0].bar(x + 0.2, sg, 0.4, label="simulated (retrieval-greedy)")
    axes[0].set_xticks(x, genre_names, rotation=60, ha="right", fontsize=8)
    axes[0].set_title("Genre share of consumed items, held-out users")
    axes[0].legend(fontsize=8)
    axes[1].plot(levels, pred_means, "o-", label="mean predicted")
    axes[1].plot([1, 5], [1, 5], "--", color="gray", label="perfect calibration")
    axes[1].set_xlabel("true rating")
    axes[1].set_ylabel("predicted rating")
    axes[1].set_title("Rating head calibration (held-out users, test)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "realism.png", dpi=120)
    plt.close(fig)

    # ---- report ------------------------------------------------------------------------
    lines = ["# Simulator validation report", ""]
    lines += [
        f"Held-out users: {len(users)}  ·  candidates/step: {cfg.n_candidates}  ·  "
        f"slate size: {cfg.slate_size}  ·  calibrated `sim_ref`: {sim.p.sim_ref:.3f}",
        "",
        "## Sanity checks",
        "",
    ]
    lines += [f"- {'PASS' if ok else 'FAIL'}: {name}" for name, ok in checks.items()]
    lines += ["", "## Learned model (held-out users, test positions)", ""]
    lines += ["| Model | Next-item NLL | HR@10 | Rating RMSE |", "|---|---|---|---|"]
    lines += [f"| Choice model | {cm['nll']:.3f} | {cm['hr@10']:.3f} | {cm['rating_rmse']:.3f} |"]
    lines += [f"| Popularity / global mean | {base['nll']:.3f} | – | {base['rating_rmse']:.3f} |"]
    lines += ["", "## Policies rolled through the simulator", ""]
    lines += [
        "`oracle_*` policies read the simulator's own learned affinity and exist only to probe "
        "the dynamics. No agent gets this access.",
        "",
    ]
    lines += ["| Policy | " + " | ".join(label for _, label in TABLE_COLS) + " |"]
    lines += ["|---" * (len(TABLE_COLS) + 1) + "|"]
    for name in POLICY_ORDER:
        s = summaries[name]
        lines += [f"| {name} | " + " | ".join(f"{s[key]:.3f}" for key, _ in TABLE_COLS) + " |"]
    lines += ["", "![session curves](session_curves.png)", "", "## Realism", ""]
    lines += [
        f"Simulated sessions use the `retrieval_greedy` policy. Each user's real next "
        f"consumptions are truncated to as many items as they clicked in simulation "
        f"(mean {realism['sim_items_per_user']:.2f}; {realism['users_compared']} users compared).",
        "",
        "| Statistic | Real | Simulated |",
        "|---|---|---|",
        f"| Genre entropy per user (bits) | {realism['real_genre_entropy']:.3f} | "
        f"{realism['sim_genre_entropy']:.3f} |",
        f"| Novelty of consumed items (bits) | {realism['real_novelty']:.3f} | "
        f"{realism['sim_novelty']:.3f} |",
        "",
        f"Genre-distribution JS divergence (real vs simulated): "
        f"{realism['genre_js_divergence']:.4f} bits.",
        "",
        "Rating calibration (held-out test): "
        + ", ".join(
            f"true {r} → {p:.2f} (n={c})"
            for r, p, c in zip(levels, pred_means, counts, strict=True)
        ),
        "",
        "![realism](realism.png)",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))
    summary = {
        "checks": checks,
        "policies": summaries,
        "realism": realism,
        "calibration": dict(zip(map(str, levels), pred_means, strict=True)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    run = init_wandb(cfg, job_type="simulator-validation")
    run.log(
        {
            "session_curves": wandb.Image(str(out / "session_curves.png")),
            "realism": wandb.Image(str(out / "realism.png")),
            "policies": wandb.Table(
                columns=["policy"] + [k for k, _ in TABLE_COLS],
                data=[[p] + [summaries[p][k] for k, _ in TABLE_COLS] for p in POLICY_ORDER],
            ),
        }
    )
    run.summary.update(
        {f"check/{k}": v for k, v in checks.items()}
        | {f"realism/{k}": v for k, v in realism.items()}
    )
    run.finish()

    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
