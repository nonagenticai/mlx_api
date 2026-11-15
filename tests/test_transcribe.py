"""Transcription engine tests. Fast tests stub the whisper call; `-m models` runs the real model."""

from __future__ import annotations

import io
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from mlx_api import transcribe as tr
from mlx_api.errors import InvalidRequest

PRODUCTION_MODEL = "mlx-community/whisper-large-v3-turbo"


def _wav_bytes(seconds: float, rate: int = 22_050) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    pcm = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class _StubWhisper:
    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, samples: np.ndarray, model_id: str, language: str | None, prompt: str | None
    ) -> dict[str, Any]:
        self.calls.append(
            {"n": samples.size, "model": model_id, "language": language, "prompt": prompt}
        )
        return self.result


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> _StubWhisper:
    s = _StubWhisper({"text": " hello ", "language": "en", "segments": []})
    monkeypatch.setattr(tr, "_whisper_transcribe", s)
    return s


def test_empty_bytes_is_invalid_request(stub: _StubWhisper) -> None:
    with pytest.raises(InvalidRequest) as exc:
        tr.Transcriber("m").transcribe(b"", "a.wav", None, None)
    assert exc.value.status == 400
    assert stub.calls == []


@pytest.mark.parametrize("payload", [b"not audio at all" * 64, b"RIFF\x00\x00\x00\x00WAVEjunk"])
def test_undecodable_audio_is_invalid_request(stub: _StubWhisper, payload: bytes) -> None:
    with pytest.raises(InvalidRequest, match="could not decode|no decodable"):
        tr.Transcriber("m").transcribe(payload, "x.mp3", None, None)
    assert stub.calls == []


def test_hostile_filename_is_never_used(stub: _StubWhisper) -> None:
    out = tr.Transcriber("m").transcribe(
        _wav_bytes(1.0), "; rm -rf / $(touch pwned).wav", None, None
    )
    assert out.duration == pytest.approx(1.0, abs=0.01)
    assert not Path("pwned").exists()


def test_decodes_to_16k_mono_with_real_duration(stub: _StubWhisper) -> None:
    out = tr.Transcriber("model-x").transcribe(_wav_bytes(2.5), "a.wav", None, "ctx")
    assert out.duration == pytest.approx(2.5, abs=0.01)
    assert stub.calls == [{"n": 40_000, "model": "model-x", "language": None, "prompt": "ctx"}]
    assert out.text == "hello"


def test_segments_sorted_clamped_and_renumbered(stub: _StubWhisper) -> None:
    stub.result = {
        "text": "b a c",
        "language": "en",
        "segments": [
            {"id": 7, "start": 1.0, "end": 9.0, "text": " b"},
            {"id": 3, "start": -0.4, "end": 0.8, "text": " a"},
            {"id": 9, "start": 1.9, "end": 1.5, "text": " c"},
        ],
    }
    out = tr.Transcriber("m").transcribe(_wav_bytes(2.0), "a.wav", None, None)
    assert [s.id for s in out.segments] == [0, 1, 2]
    assert [s.text for s in out.segments] == [" a", " b", " c"]
    for s in out.segments:
        assert 0.0 <= s.start <= s.end <= out.duration
    assert out.segments[0].start == 0.0
    assert out.segments[1].end == out.duration
    assert out.segments[2].start == out.segments[2].end == 1.9


@pytest.mark.parametrize(
    ("detected", "requested", "expected_name", "expected_code"),
    [
        ("en", None, "english", None),
        ("de", None, "german", None),
        ("fr", "fr", "french", "fr"),
        ("nl", "Dutch", "dutch", "nl"),
        (None, "es", "spanish", "es"),
    ],
)
def test_language_is_full_lowercase_name(
    stub: _StubWhisper,
    detected: str | None,
    requested: str | None,
    expected_name: str,
    expected_code: str | None,
) -> None:
    stub.result = {"text": "x", "language": detected, "segments": []}
    out = tr.Transcriber("m").transcribe(_wav_bytes(0.5), "a.wav", requested, None)
    assert out.language == expected_name
    assert stub.calls[0]["language"] == expected_code


def test_unknown_language_is_invalid_request(stub: _StubWhisper) -> None:
    with pytest.raises(InvalidRequest, match="unsupported language"):
        tr.Transcriber("m").transcribe(_wav_bytes(0.5), "a.wav", "klingon", None)
    assert stub.calls == []


@pytest.mark.models
def test_real_model_transcribes_say_output(tmp_path: Path, request: pytest.FixtureRequest) -> None:
    if "models" not in (request.config.getoption("markexpr") or ""):
        pytest.skip("loads real weights; run with `-m models`")
    say = shutil.which("say")
    if say is None:
        pytest.skip("macOS `say` not available")
    ffprobe = shutil.which("ffprobe")
    assert ffprobe is not None, "ffprobe required"
    aiff = tmp_path / "fox.aiff"
    subprocess.run(  # noqa: S603
        [say, "-o", str(aiff), "The quick brown fox jumps over the lazy dog"], check=True
    )
    probed = float(
        subprocess.run(  # noqa: S603
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(aiff),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )

    engine = tr.Transcriber(PRODUCTION_MODEL)
    engine.transcribe(aiff.read_bytes(), aiff.name, None, None)  # warm: load weights
    t0 = time.perf_counter()
    out = engine.transcribe(aiff.read_bytes(), aiff.name, None, None)
    elapsed = time.perf_counter() - t0

    print(
        f"\ntranscript={out.text!r} language={out.language} duration={out.duration}s "
        f"elapsed={elapsed:.3f}s RTF={elapsed / out.duration:.3f}"
    )
    text = out.text.lower()
    assert "fox" in text
    assert "dog" in text
    assert out.language == "english"
    assert abs(out.duration - probed) <= 0.5
    assert out.segments
    for s in out.segments:
        assert 0.0 <= s.start <= s.end <= out.duration
