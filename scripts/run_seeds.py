"""Train agents over several seeds, a few processes at a time; finished runs are skipped.

A run counts as finished when ``artifacts/agents/<agent>/seed<seed>/DONE`` exists, so the
script can be re-run after an interruption and only redoes what is missing.

Usage:
  uv run python scripts/run_seeds.py --agents dqn slateq ppo ppo_pl --seeds 0 1 2 3 4 --parallel 4
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", nargs="+", default=["dqn", "slateq", "ppo", "ppo_pl"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--log-dir", default="logs/seeds")
    parser.add_argument("overrides", nargs="*", help="extra Hydra overrides for every run")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for seed in args.seeds:  # seed-major order: every agent gets seed 1 before any gets seed 4
        for agent in args.agents:
            run_dir = Path(f"artifacts/agents/{agent}/seed{seed}")
            if (run_dir / "DONE").exists():
                print(f"skip {agent} seed {seed} (done)")
                continue
            jobs.append((agent, seed, run_dir))

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    running: list[tuple[subprocess.Popen, str, int, Path, float]] = []
    failed = []
    start = time.time()
    while jobs or running:
        while jobs and len(running) < args.parallel:
            agent, seed, run_dir = jobs.pop(0)
            log = open(log_dir / f"{agent}-seed{seed}.log", "w")  # noqa: SIM115
            cmd = [
                sys.executable,
                "scripts/train_agent.py",
                f"agent={agent}",
                f"seed={seed}",
                *args.overrides,
            ]
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((proc, agent, seed, run_dir, time.time()))
            print(f"start {agent} seed {seed} (pid {proc.pid})", flush=True)
        time.sleep(5)
        for item in list(running):
            proc, agent, seed, run_dir, t0 = item
            if proc.poll() is None:
                continue
            running.remove(item)
            minutes = (time.time() - t0) / 60
            if proc.returncode == 0:
                (run_dir / "DONE").write_text(f"{minutes:.1f} min\n")
                print(f"done  {agent} seed {seed} in {minutes:.1f} min", flush=True)
            else:
                failed.append((agent, seed))
                print(
                    f"FAIL  {agent} seed {seed} (exit {proc.returncode}); see {log_dir}", flush=True
                )
    print(f"all jobs finished in {(time.time() - start) / 60:.1f} min; failed: {failed or 'none'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
