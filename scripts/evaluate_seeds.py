"""Multi-seed evaluation and robustness to perturbed simulators, on held-out users.

For every scenario (nominal, perturbed dynamics, alternate simulator) each baseline is
evaluated once and each agent once per training seed. Reported intervals are 95% t-intervals
across seeds, so they reflect training variance. Writes reports/eval_seeds/{results.json,
report.md, *.png}.

Usage: uv run python scripts/evaluate_seeds.py ['seeds=[0,1]'] ['scenarios={nominal: {}}']
"""

import json
from pathlib import Path

import hydra
import matplotlib
import torch
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from slate_rl.agents.baselines import make_baselines  # noqa: E402
from slate_rl.agents.click_model import load_click_model  # noqa: E402
from slate_rl.agents.dqn import DQNAgent  # noqa: E402
from slate_rl.agents.ppo import PPOAgent  # noqa: E402
from slate_rl.agents.slateq import SlateQAgent  # noqa: E402
from slate_rl.envs import EnvConfig, load_assets, make_batched_env, with_simulator  # noqa: E402
from slate_rl.eval.evaluate import evaluate_policy  # noqa: E402
from slate_rl.eval.stats import seed_ci  # noqa: E402
from slate_rl.utils import get_device, init_wandb, seed_everything  # noqa: E402

LOADERS = {"dqn": DQNAgent.load, "slateq": SlateQAgent.load, "ppo": PPOAgent.load}
METRICS = ["return", "session_length", "clicks", "watch", "ctr", "ild", "repeat", "coverage"]
LABELS = {
    "dqn": "DQN",
    "slateq": "SlateQ",
    "ppo_pl": "PPO (Plackett-Luce)",
    "ppo": "PPO (proto-actions)",
    "random": "random",
    "popularity": "popularity",
    "retrieval": "retrieval",
    "greedy_click": "greedy-click",
    "mmr": "MMR",
}
# Reference palette (dataviz skill), categorical slots in fixed order; text in neutral ink.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK_2, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#a3a29d"


def load_agent(path: Path, device):
    kind = torch.load(path, map_location="cpu")["agent"]
    return LOADERS[kind](path, device)


def style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


@hydra.main(config_path="../configs", config_name="evaluate_seeds", version_base="1.3")
def compute_results(cfg: DictConfig) -> tuple[dict, int]:
    """Evaluate every policy in every scenario. Returns (results, number of held-out users)."""
    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    env_cfg = EnvConfig.from_dict(OmegaConf.to_container(cfg.env, resolve=True))
    eval_cfg = EnvConfig(**{**env_cfg.__dict__, "autoreset": False})
    assets = load_assets(cfg, device)
    users = assets.seqs.heldout_users()
    eval_cfg.num_envs = len(users)
    click_model = load_click_model(cfg.click_model, device)

    checkpoints = {}
    for agent in cfg.agents:
        for seed in cfg.seeds:
            path = Path(cfg.checkpoint_pattern.format(agent=agent, seed=seed))
            if path.exists():
                checkpoints[(agent, seed)] = path
            else:
                print(f"missing {path}; skipped")

    scenarios = [(name, assets, dict(o)) for name, o in cfg.scenarios.items()]
    for name, sim_dir in (cfg.alt_simulators or {}).items():
        if Path(sim_dir, "choice_model.pt").exists():
            scenarios.append((name, with_simulator(assets, sim_dir), {}))
        else:
            print(f"missing alternate simulator {sim_dir}; scenario {name} skipped")

    # results[scenario][policy] = list of metric dicts (one per seed; one entry for baselines)
    results: dict[str, dict[str, list[dict]]] = {}
    agents_cache = {key: load_agent(path, device) for key, path in checkpoints.items()}
    for scen, scen_assets, overrides in scenarios:
        env = make_batched_env(scen_assets, eval_cfg, "heldout", cfg.seed, overrides)
        res: dict[str, list[dict]] = {}
        for name, policy in make_baselines(env, click_model, cfg.seed, cfg.mmr_diversity).items():
            res[name] = [evaluate_policy(env, policy, users, seed=cfg.seed)]
        for (agent, _train_seed), policy in agents_cache.items():
            res.setdefault(agent, []).append(evaluate_policy(env, policy, users, seed=cfg.seed))
        results[scen] = {
            p: [{k: v for k, v in r.items() if k in METRICS} for r in runs]
            for p, runs in res.items()
        }
        print(
            f"{scen}: "
            + "  ".join(
                f"{p}={sum(r['return'] for r in runs) / len(runs):.2f}" for p, runs in res.items()
            )
        )

    return results, len(users)


def write_report(cfg: DictConfig, results: dict, n_users: int) -> None:
    out = Path(cfg.out_dir)
    agents = [a for a in cfg.agents if a in results["nominal"]]
    baselines = [p for p in results["nominal"] if p not in agents]
    ref = cfg.reference_baseline

    def ci(scen: str, policy: str, metric: str = "return"):
        return seed_ci([r[metric] for r in results[scen][policy]])

    def gain(scen: str, agent: str):
        base = results[scen][ref][0]["return"]
        return seed_ci([r["return"] - base for r in results[scen][agent]])

    # ---- figure 1: nominal return with across-seed CIs --------------------------------
    order = sorted(baselines + agents, key=lambda p: ci("nominal", p)[0])
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(order) + 1.2))
    for i, p in enumerate(order):
        mean, lo, hi = ci("nominal", p)
        is_agent = p in agents
        ax.barh(i, mean, height=0.6, color=SERIES[0] if is_agent else NEUTRAL, linewidth=0)
        if is_agent and len(results["nominal"][p]) > 1:
            ax.errorbar(mean, i, xerr=[[mean - lo], [hi - mean]], color=INK, capsize=3, lw=1.2)
        ax.text(hi + 0.12, i, f"{mean:.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(order)), [LABELS.get(p, p) for p in order], color=INK)
    ax.set_xlabel("Return per session (held-out users)", color=INK_2)
    n_seeds = max(len(results["nominal"][a]) for a in agents) if agents else 1
    ax.set_title(
        f"RL agents (blue): mean and 95% CI over {n_seeds} seeds; baselines (gray)",
        fontsize=10,
        color=INK,
        loc="left",
    )
    style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out / "nominal_returns.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: robustness, gain over the reference baseline per scenario ----------
    scen_names = list(results)
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(scen_names) + 1.4))
    offsets = torch.linspace(0.24, -0.24, len(agents)).tolist() if len(agents) > 1 else [0.0]
    for j, agent in enumerate(agents):
        color = SERIES[j % len(SERIES)]
        for i, scen in enumerate(scen_names):
            mean, lo, hi = gain(scen, agent)
            y = len(scen_names) - 1 - i + offsets[j]
            ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round")
            ax.plot(
                mean,
                y,
                "o",
                ms=6,
                color=color,
                mec="white",
                mew=1.5,
                label=LABELS[agent] if i == 0 else None,
            )
    ax.axvline(0, color=INK_2, lw=1)
    ax.set_yticks(range(len(scen_names)), list(reversed(scen_names)), color=INK)
    ax.set_xlabel(f"Return gain over {LABELS[ref]} (mean and 95% CI over seeds)", color=INK_2)
    ax.set_title(
        "Robustness: agents trained on the nominal simulator, evaluated unchanged",
        fontsize=10,
        color=INK,
        loc="left",
    )
    style(ax)
    ax.legend(fontsize=8, frameon=False, loc="lower right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=150)
    plt.close(fig)

    # ---- report ------------------------------------------------------------------------
    lines = ["# Multi-seed evaluation and robustness", ""]
    lines += [
        f"Held-out users: {n_users}. Agents: {n_seeds} training seeds each; intervals are "
        "95% t-intervals across seeds. Baselines are deterministic (random uses a fixed seed) "
        "and are evaluated once. Every policy sees the same users and evaluation seed.",
        "",
        "## Nominal simulator",
        "",
        "| Policy | Return | Session length | Watch | CTR | ILD | Repeat |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in sorted(baselines + agents, key=lambda p: -ci("nominal", p)[0]):
        cells = []
        for m in ("return", "session_length", "watch", "ctr", "ild", "repeat"):
            mean, lo, hi = ci("nominal", p, m)
            cells.append(
                f"{mean:.2f} [{lo:.2f}, {hi:.2f}]"
                if p in agents and m == "return"
                else f"{mean:.3f}"
                if m in ("ctr", "ild", "repeat")
                else f"{mean:.2f}"
            )
        lines.append(f"| {LABELS.get(p, p)} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "![nominal returns](nominal_returns.png)",
        "",
        f"## Robustness (return gain over {LABELS[ref]})",
        "",
    ]
    lines += [
        "| Scenario | " + " | ".join(LABELS[a] for a in agents) + " |",
        "|---" * (len(agents) + 1) + "|",
    ]
    for scen in scen_names:
        cells = [
            f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]" for m, lo, hi in (gain(scen, a) for a in agents)
        ]
        lines.append(f"| {scen} | " + " | ".join(cells) + " |")
    ranks = {}
    for scen in scen_names:
        ranked = sorted(agents + baselines, key=lambda p: -ci(scen, p)[0])
        ranks[scen] = ranked
    lines += ["", "Policy ranking by mean return per scenario:", ""]
    lines += [
        f"- **{scen}**: " + " > ".join(LABELS.get(p, p) for p in r) for scen, r in ranks.items()
    ]
    lines += ["", "![robustness](robustness.png)", ""]
    (out / "report.md").write_text("\n".join(lines))

    if cfg.plot_only:
        print((out / "report.md").read_text())
        return
    run = init_wandb(cfg, job_type="evaluate-seeds")
    run.summary.update(
        {f"{scen}/{p}/return": ci(scen, p)[0] for scen in scen_names for p in results[scen]}
    )
    run.finish()
    print((out / "report.md").read_text())


@hydra.main(config_path="../configs", config_name="evaluate_seeds", version_base="1.3")
def main(cfg: DictConfig) -> None:
    out = Path(cfg.out_dir)
    if cfg.plot_only:  # rebuild report and figures from a previous run, no evaluation
        saved = json.loads((out / "results.json").read_text())
    else:
        results, n_users = compute_results(cfg)
        out.mkdir(parents=True, exist_ok=True)
        saved = {"meta": {"n_users": n_users}, "results": results}
        (out / "results.json").write_text(json.dumps(saved, indent=2))
    if "results" not in saved:  # files written before the meta block existed
        saved = {"meta": {"n_users": 604}, "results": saved}
    write_report(cfg, saved["results"], saved["meta"]["n_users"])


if __name__ == "__main__":
    main()
