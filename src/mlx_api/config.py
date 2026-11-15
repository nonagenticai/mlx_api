"""Service configuration, read from the environment once at startup."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

CAPABILITIES = ("embed", "rerank", "ocr", "transcribe", "vision")


class ConfigError(RuntimeError):
    """Raised when the service must refuse to start."""


@dataclass(frozen=True)
class Settings:
    api_key: str
    host: str = "0.0.0.0"  # noqa: S104  # nosec B104 - tailnet service, bearer auth
    port: int = 9131
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    ocr_model: str = "mlx-community/PaddleOCR-VL-1.5-bf16"
    transcribe_model: str = "mlx-community/whisper-large-v3-turbo"
    vision_model: str = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    # Padded token slots (batch_size x longest sequence) per forward pass.
    embed_token_budget: int = 16384
    rerank_token_budget: int = 8192
    embed_max_length: int = 8192
    rerank_max_length: int = 1024
    dtype: str = "float32"
    # Capabilities this process serves; the rest answer 503. Lets one host carry a subset.
    capabilities: tuple[str, ...] = CAPABILITIES
    # MLX buffer-cache cap in MB (None = MLX default, i.e. unbounded up to the memory limit).
    # The cache, not the weights, dominates footprint under load; cap it on small hosts.
    cache_limit_mb: int | None = None

    def model_for(self, capability: str) -> str:
        return {
            "embed": self.embed_model,
            "rerank": self.rerank_model,
            "ocr": self.ocr_model,
            "transcribe": self.transcribe_model,
            "vision": self.vision_model,
        }[capability]

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        key = env.get("MLX_API_KEY", "").strip()
        if not key:
            raise ConfigError("MLX_API_KEY is required; refusing to start without it")
        dtype = env.get("MLX_API_DTYPE", cls.dtype)
        if dtype not in ("float32", "float16", "bfloat16"):
            raise ConfigError(f"MLX_API_DTYPE must be float32|float16|bfloat16, got {dtype!r}")

        raw_caps = env.get("MLX_API_CAPABILITIES", "").strip()
        if raw_caps:
            caps = tuple(dict.fromkeys(c.strip() for c in raw_caps.split(",") if c.strip()))
            unknown = [c for c in caps if c not in CAPABILITIES]
            if unknown or not caps:
                raise ConfigError(
                    f"MLX_API_CAPABILITIES must be a comma list of {','.join(CAPABILITIES)}, "
                    f"got {raw_caps!r}"
                )
        else:
            caps = CAPABILITIES

        def _int(name: str, default: int) -> int:
            raw = env.get(name)
            if raw is None or raw == "":
                return default
            try:
                value = int(raw)
            except ValueError as exc:
                raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
            if value <= 0:
                raise ConfigError(f"{name} must be > 0, got {value}")
            return value

        raw_cache = env.get("MLX_API_CACHE_LIMIT_MB", "").strip()
        cache_limit_mb: int | None = None
        if raw_cache:
            try:
                cache_limit_mb = int(raw_cache)
            except ValueError as exc:
                raise ConfigError(
                    f"MLX_API_CACHE_LIMIT_MB must be an integer, got {raw_cache!r}"
                ) from exc
            if cache_limit_mb < 0:
                raise ConfigError(f"MLX_API_CACHE_LIMIT_MB must be >= 0, got {cache_limit_mb}")

        return cls(
            api_key=key,
            host=env.get("HOST", cls.host),
            port=_int("PORT", cls.port),
            embed_model=env.get("MLX_API_EMBED_MODEL", cls.embed_model),
            rerank_model=env.get("MLX_API_RERANK_MODEL", cls.rerank_model),
            ocr_model=env.get("MLX_API_OCR_MODEL", cls.ocr_model),
            transcribe_model=env.get("MLX_API_TRANSCRIBE_MODEL", cls.transcribe_model),
            vision_model=env.get("MLX_API_VISION_MODEL", cls.vision_model),
            embed_token_budget=_int("MLX_API_EMBED_TOKEN_BUDGET", cls.embed_token_budget),
            rerank_token_budget=_int("MLX_API_RERANK_TOKEN_BUDGET", cls.rerank_token_budget),
            embed_max_length=_int("MLX_API_EMBED_MAX_LENGTH", cls.embed_max_length),
            rerank_max_length=_int("MLX_API_RERANK_MAX_LENGTH", cls.rerank_max_length),
            dtype=dtype,
            capabilities=caps,
            cache_limit_mb=cache_limit_mb,
        )


def model_alias(model_id: str) -> str:
    return model_id.rsplit("/", 1)[-1]


def matches_model(requested: str, configured: str) -> bool:
    """A request may name the configured id (``BAAI/bge-m3``) or its short alias (``bge-m3``)."""
    return requested in (configured, model_alias(configured))
