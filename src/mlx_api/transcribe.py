"""Transcription engine (contract section "transcribe", OpenAI shape) backed by mlx-whisper.

Audio of any container ffmpeg understands (wav/mp3/m4a/mp4/webm/...) is decoded by an ffmpeg
subprocess reading the raw bytes from stdin — no shell, and the client-supplied filename is never
put on a command line or the filesystem. Everything here is synchronous: callers run it on the
single GPU thread.
"""

from __future__ import annotations

import functools
import shutil
import subprocess  # noqa: S404 - fixed argv, no shell, input via stdin
from dataclasses import dataclass
from typing import Any

import numpy as np

from mlx_api.errors import InvalidRequest, Unavailable

SAMPLE_RATE = 16_000
DECODE_TIMEOUT_S = 300.0


@functools.cache
def _languages() -> dict[str, str]:
    """ISO-639-1 code -> lower-case English name (what OpenAI verbose_json reports)."""
    try:
        from mlx_whisper.tokenizer import LANGUAGES
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        raise Unavailable(
            "transcription backend not installed (install extra 'transcribe')"
        ) from exc
    return dict(LANGUAGES)


@dataclass(frozen=True)
class Segment:
    id: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Transcript:
    text: str
    language: str
    duration: float
    segments: list[Segment]


def decode_audio(audio: bytes) -> np.ndarray:
    """Decode any ffmpeg-readable container from bytes to 16 kHz mono float32 samples."""
    if not audio:
        raise InvalidRequest("audio file is empty", code="empty_file")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise Unavailable("ffmpeg is not installed on this host")
    argv = [
        ffmpeg,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-vn",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "pipe:1",
    ]
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            argv, input=audio, capture_output=True, timeout=DECODE_TIMEOUT_S, check=False
        )
    except subprocess.TimeoutExpired as exc:
        raise InvalidRequest("audio decoding timed out", code="undecodable_audio") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        reason = detail[-1][:200] if detail else f"ffmpeg exit {proc.returncode}"
        raise InvalidRequest(f"could not decode audio: {reason}", code="undecodable_audio")
    usable = len(proc.stdout) - len(proc.stdout) % 4
    samples = np.frombuffer(proc.stdout[:usable], dtype=np.float32)
    if samples.size == 0:
        raise InvalidRequest("audio contains no decodable samples", code="undecodable_audio")
    return samples


def _normalize_language(language: str | None) -> str | None:
    """Accept an ISO-639-1 code (OpenAI's contract) or an English name; return the code."""
    if language is None or not language.strip():
        return None
    value = language.strip().lower()
    langs = _languages()
    if value in langs:
        return value
    by_name = {name: code for code, name in langs.items()}
    if value in by_name:
        return by_name[value]
    raise InvalidRequest(f"unsupported language: {language!r}", code="invalid_language")


def _language_name(code: str | None) -> str:
    if not code:
        return "unknown"
    return _languages().get(code, code)


def _whisper_transcribe(
    samples: np.ndarray, model_id: str, language: str | None, prompt: str | None
) -> dict[str, Any]:
    """The only call into mlx-whisper (tests stub this). Model weights are cached process-wide."""
    try:
        import mlx_whisper
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        raise Unavailable(
            "transcription backend not installed (install extra 'transcribe')"
        ) from exc
    return mlx_whisper.transcribe(
        samples,
        path_or_hf_repo=model_id,
        language=language,
        initial_prompt=prompt or None,
        verbose=None,
    )


def _build_segments(raw: list[dict[str, Any]], duration: float) -> list[Segment]:
    clamped: list[tuple[float, float, str]] = []
    for seg in raw:
        start = min(max(float(seg.get("start", 0.0)), 0.0), duration)
        end = min(max(float(seg.get("end", start)), start), duration)
        clamped.append((start, end, str(seg.get("text", ""))))
    clamped.sort(key=lambda s: (s[0], s[1]))
    return [Segment(id=i, start=s, end=e, text=t) for i, (s, e, t) in enumerate(clamped)]


class Transcriber:
    def __init__(self, model_id: str) -> None:
        # Lazy: weights load on the first transcribe() (mlx-whisper caches them per process).
        self.model_id = model_id

    def transcribe(
        self, audio: bytes, filename: str, language: str | None, prompt: str | None
    ) -> Transcript:
        # `filename` is informational only; ffmpeg probes the container from the bytes.
        del filename
        lang_code = _normalize_language(language)
        samples = decode_audio(audio)
        duration = round(samples.size / SAMPLE_RATE, 3)
        result = _whisper_transcribe(samples, self.model_id, lang_code, prompt)
        segments = _build_segments(list(result.get("segments") or []), duration)
        return Transcript(
            text=str(result.get("text", "")).strip(),
            language=_language_name(result.get("language") or lang_code),
            duration=duration,
            segments=segments,
        )
