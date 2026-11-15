"""VisionEngine unit tests (injected generate) + one real-weights test (``-m models``)."""

from __future__ import annotations

import pytest
from PIL import Image

from mlx_api.vision import MAX_SIDE_PX, Turn, VisionEngine, VisionResult

MODEL_ID = "mlx-community/Qwen2-VL-2B-Instruct-4bit"


def test_engine_passes_turns_and_prepared_images() -> None:
    seen: list[tuple[list[Turn], list[Image.Image], int, float]] = []

    def gen(turns: list[Turn], images: list[Image.Image], mt: int, temp: float) -> VisionResult:
        seen.append((turns, images, mt, temp))
        return VisionResult("a red square", 12, 3, "stop")

    engine = VisionEngine(MODEL_ID, generate_fn=gen)
    big = Image.new("RGBA", (MAX_SIDE_PX * 2, MAX_SIDE_PX), (255, 0, 0, 128))
    small = Image.new("L", (10, 20))
    out = engine.complete([Turn("user", "What is it?")], [big, small], 64, 0.2)

    assert out == VisionResult("a red square", 12, 3, "stop")
    assert not engine.is_loaded  # injected generate never loads weights
    turns, images, mt, temp = seen[0]
    assert turns == [Turn("user", "What is it?")]
    assert (mt, temp) == (64, 0.2)
    # longest side capped, aspect kept, always RGB; small images untouched in size
    assert images[0].size == (MAX_SIDE_PX, MAX_SIDE_PX // 2)
    assert images[1].size == (10, 20)
    assert {img.mode for img in images} == {"RGB"}


@pytest.mark.models
def test_real_model_describes_a_colour(request: pytest.FixtureRequest) -> None:
    if "models" not in (request.config.getoption("markexpr") or ""):
        pytest.skip("loads real weights; run with -m models")
    from PIL import ImageDraw

    engine = VisionEngine(MODEL_ID)
    # Not a flat single-colour image: Qwen2-VL-2B answers "Blue" for every solid fill
    # (measured red/green/blue/yellow on mini2), while it reads a simple scene correctly.
    img = Image.new("RGB", (448, 224), "white")
    draw = ImageDraw.Draw(img)
    draw.ellipse((20, 20, 200, 200), fill=(220, 20, 20))
    draw.rectangle((260, 40, 420, 190), fill=(20, 20, 220))
    out = engine.complete(
        [Turn("user", "Describe the shapes and their colours in one sentence.")], [img], 48, 0.0
    )
    text = out.text.lower()
    assert "red" in text and "blue" in text and "circle" in text, out
    assert out.prompt_tokens > 0 and out.completion_tokens > 0
    assert out.finish_reason in ("stop", "length")
