#!/bin/bash
# launchd entry point (com.nonagentic.mlx-api). Env comes from ~/.config/mlx-api/env, rendered by
# host-config-sync from the external-services ConfigMap; MLX_API_KEY must be set there.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f "$HOME/.config/mlx-api/env" ]; then
  set -a; . "$HOME/.config/mlx-api/env"; set +a
fi
export PORT="${PORT:-9131}" HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
exec uv run --frozen python -m mlx_api.main
