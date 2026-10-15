#!/usr/bin/env bash
# One-time setup on a fresh Linux GPU machine (e.g. a RunPod PyTorch pod).
set -euo pipefail
cd "$(dirname "$0")/../.."
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync
uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
