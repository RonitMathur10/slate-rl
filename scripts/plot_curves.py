"""Plot validation learning curves of agent runs from W&B (mean ± min/max across seeds).

Usage: uv run python scripts/plot_curves.py --agents dqn --out reports/eval/learning_curves.png
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import wandb  # noqa: E402

METRICS = [
    ("eval/return", "Validation return"),
    ("eval/session_length", "Session length"),
    ("eval/ctr", "CTR"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="ronit-mathur-ucberkeley")
    parser.add_argument("--project", default="slate-rl")
    parser.add_argument("--agents", nargs="+", default=["dqn"])
    parser.add_argument("--out", default="reports/eval/learning_curves.png")
    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs(
        f"{args.entity}/{args.project}",
        filters={"jobType": "train", "state": "finished"},
        order="-created_at",
    )
    # Keep only the newest finished run per (agent, seed): reruns supersede older versions.
    latest: dict[tuple[str, str], object] = {}
    for run in runs:
        agent, _, seed = run.name.rpartition("-seed")
        if agent in args.agents and (agent, seed) not in latest:
            latest[(agent, seed)] = run
    curves: dict[str, list] = defaultdict(list)
    keys = ["_step"] + [m for m, _ in METRICS]
    for (agent, _), run in sorted(latest.items()):
        rows = run.history(keys=keys, samples=1000, pandas=False)
        curves[agent].append(sorted(rows, key=lambda r: r["_step"]))

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5 * len(METRICS), 3.6))
    for agent, seeds in curves.items():
        steps = np.array([r["_step"] for r in seeds[0]])
        for ax, (key, title) in zip(axes, METRICS, strict=True):
            n = min(len(s) for s in seeds)
            vals = np.array([[r[key] for r in s[:n]] for s in seeds])
            mean = vals.mean(0)
            ax.plot(steps[:n], mean, label=f"{agent} ({len(seeds)} seed{'s' * (len(seeds) > 1)})")
            if len(seeds) > 1:
                ax.fill_between(steps[:n], vals.min(0), vals.max(0), alpha=0.2)
            ax.set_title(title)
            ax.set_xlabel("environment steps")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"wrote {args.out}: " + ", ".join(f"{a}: {len(s)} runs" for a, s in curves.items()))


if __name__ == "__main__":
    main()
