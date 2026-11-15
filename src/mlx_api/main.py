"""Entrypoint: ``MLX_API_KEY=... python -m mlx_api.main``."""

from __future__ import annotations

import logging

import uvicorn

from mlx_api.app import create_app
from mlx_api.config import Settings

settings = Settings.from_env()
app = create_app(settings)


def apply_mlx_limits(cfg: Settings) -> None:
    if cfg.cache_limit_mb is None:
        return
    import mlx.core as mx

    mx.set_cache_limit(cfg.cache_limit_mb * 1024 * 1024)


def run() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    apply_mlx_limits(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    run()
