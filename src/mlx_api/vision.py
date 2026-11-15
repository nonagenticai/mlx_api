"""Vision engine — the `vision` contract (OpenAI chat completions with image parts).

Captions / describes images for strata-ingest. A Qwen2-VL-class model through mlx-vlm, the same
library the OCR engine uses. Text-only chat is NOT this capability (that is `chat`); a request
without any image is still answered, since the contract does not forbid it.

Blocking by design: callers run ``VisionEngine.complete`` on the single GPU thread
(``mlx_api.gpu``). The model is loaded lazily on the first call, i.e. on that same thread.
"""

from __future__ import annotations

import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PIL import Image

MAX_SIDE_PX = 1536


@dataclass(frozen=True)
class Turn:
    role: str  # system | user | assistant
    text: str


@dataclass(frozen=True)
class VisionResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop" | "length"


# (turns, images, max_tokens, temperature) -> result. Injected in tests; default runs mlx-vlm.
GenerateFn = Callable[[list[Turn], list[Image.Image], int, float], VisionResult]


class VisionEngine:
    def __init__(self, model_id: str, *, generate_fn: GenerateFn | None = None) -> None:
        self.model_id = model_id
        self._generate_fn = generate_fn
        self._model: Any = None
        self._processor: Any = None
        self._load_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def complete(
        self, turns: list[Turn], images: list[Image.Image], max_tokens: int, temperature: float
    ) -> VisionResult:
        prepared = [_fit(img) for img in images]
        if self._generate_fn is not None:
            return self._generate_fn(turns, prepared, max_tokens, temperature)
        return self._mlx_generate(turns, prepared, max_tokens, temperature)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is None:
                from mlx_vlm import load

                self._model, self._processor = load(self.model_id)

    def _mlx_generate(
        self, turns: list[Turn], images: list[Image.Image], max_tokens: int, temperature: float
    ) -> VisionResult:
        self._ensure_loaded()
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        config = getattr(self._model, "config", None)
        messages = [{"role": t.role, "content": t.text} for t in turns]
        formatted = apply_chat_template(self._processor, config, messages, num_images=len(images))
        prompt = formatted if isinstance(formatted, str) else str(formatted)
        # mlx-vlm's `image` is typed as path(s); hand it temp PNGs (the proven OCR path).
        paths: list[str] = []
        try:
            for img in images:
                fd, path = tempfile.mkstemp(suffix=".png", prefix="mlx-vision-")
                paths.append(path)
                with os.fdopen(fd, "wb") as fh:
                    img.save(fh, format="PNG")
            result = generate(
                self._model,
                self._processor,
                prompt,
                image=paths or None,
                verbose=False,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        finally:
            for path in paths:
                os.unlink(path)
        text = getattr(result, "text", result)
        text = text if isinstance(text, str) else str(text)
        completion = int(getattr(result, "generation_tokens", 0) or 0)
        reason = getattr(result, "finish_reason", None)
        if reason not in ("stop", "length"):
            reason = "length" if completion >= max_tokens else "stop"
        return VisionResult(
            text=text.strip(),
            prompt_tokens=int(getattr(result, "prompt_tokens", 0) or 0),
            completion_tokens=completion,
            finish_reason=reason,
        )


def _fit(img: Image.Image) -> Image.Image:
    out = img.convert("RGB")
    if max(out.size) > MAX_SIDE_PX:
        out.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX), Image.Resampling.LANCZOS)
    return out
