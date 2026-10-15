#!/usr/bin/env bash
# All milestone-8 compute, for a GPU machine. Resumable: finished runs are skipped.
#   PARALLEL=4 bash scripts/remote/milestone8.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH" PYTHONUNBUFFERED=1
mkdir -p logs

# Second simulator (different seed) for the robustness check, trained alongside the agents.
SIM2=artifacts/simulator/ml-1m-seed2
if [ ! -f "$SIM2/choice_model.pt" ]; then
  uv run python scripts/train_simulator.py seed=2 simulator.artifact_dir=$SIM2 \
    log_artifact=false > logs/simulator-seed2.log 2>&1 &
  SIM_PID=$!
fi

uv run python scripts/run_seeds.py --agents dqn slateq ppo ppo_pl --seeds 0 1 2 3 4 \
  --parallel "${PARALLEL:-4}"
if [ -n "${SIM_PID:-}" ]; then wait "$SIM_PID"; fi

uv run python scripts/evaluate_seeds.py 2>&1 | tee logs/evaluate_seeds.log
uv run python scripts/export_trajectories.py 2>&1 | tee logs/export_trajectories.log
echo "milestone 8 compute finished"
