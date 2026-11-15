"""Shared fixtures.

Route tests never load weights: ``client`` overrides the ``get_engines`` dependency with an
``Engines`` built from fake specs. Tests marked ``models`` load real weights and only run when
selected explicitly (``pytest -m models``).
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from mlx_api.app import Engines, EngineSpec, create_app, get_engines
from mlx_api.config import Settings
from mlx_api.embed import EmbedResult
from mlx_api.rerank import RerankResult

API_KEY = "test-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("-m"):
        return
    skip = pytest.mark.skip(reason="loads model weights; run with -m models")
    for item in items:
        if "models" in item.keywords:
            item.add_marker(skip)


@dataclass
class FakeEmbedder:
    dim: int = 1024
    threads: list[str] = field(default_factory=list)
    calls: list[list[str]] = field(default_factory=list)

    def embed(self, texts: list[str]) -> EmbedResult:
        self.threads.append(threading.current_thread().name)
        self.calls.append(list(texts))
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little")
            v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
        return EmbedResult(vectors=np.stack(rows), prompt_tokens=sum(len(t.split()) for t in texts))


@dataclass
class FakeReranker:
    scores: dict[str, float] = field(default_factory=dict)
    threads: list[str] = field(default_factory=list)

    def score(self, query: str, documents: list[str]) -> RerankResult:
        self.threads.append(threading.current_thread().name)
        return RerankResult(
            scores=[self.scores.get(d, 0.5) for d in documents],
            tokens=len(query.split()) + sum(len(d.split()) for d in documents),
        )


@dataclass
class FakeOcrPage:
    index: int
    markdown: str
    width: int
    height: int
    dpi: int


@dataclass
class FakeOcr:
    calls: list[tuple[bytes, str, list[int] | None]] = field(default_factory=list)

    def ocr(self, data: bytes, mime: str, pages: list[int] | None) -> list[FakeOcrPage]:
        self.calls.append((data, mime, pages))
        wanted = pages if pages is not None else [0]
        # returned out of order on purpose: the route must sort by index
        return [FakeOcrPage(i, f"# page {i}", 100, 200, 72) for i in reversed(wanted)]


@dataclass
class FakeSegment:
    id: int
    start: float
    end: float
    text: str


@dataclass
class FakeTranscript:
    text: str
    language: str
    duration: float
    segments: list[FakeSegment]


@dataclass
class FakeTranscriber:
    calls: list[tuple[bytes, str, str | None, str | None]] = field(default_factory=list)

    def transcribe(
        self, audio: bytes, filename: str, language: str | None, prompt: str | None
    ) -> FakeTranscript:
        self.calls.append((audio, filename, language, prompt))
        return FakeTranscript(
            text="hello world",
            language="english",
            duration=3.5,
            segments=[FakeSegment(0, 0.0, 2.0, "hello"), FakeSegment(1, 2.0, 3.5, "world")],
        )


@dataclass
class FakeVisionResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


@dataclass
class FakeVision:
    calls: list[tuple[list[Any], list[Any], int, float]] = field(default_factory=list)

    def complete(
        self, turns: list[Any], images: list[Any], max_tokens: int, temperature: float
    ) -> FakeVisionResult:
        self.calls.append((turns, images, max_tokens, temperature))
        return FakeVisionResult(f"{len(images)} image(s)", 30, 4, "stop")


@dataclass
class Fakes:
    embed: FakeEmbedder = field(default_factory=FakeEmbedder)
    rerank: FakeReranker = field(default_factory=FakeReranker)
    ocr: FakeOcr = field(default_factory=FakeOcr)
    transcribe: FakeTranscriber = field(default_factory=FakeTranscriber)
    vision: FakeVision = field(default_factory=FakeVision)
    builds: dict[str, int] = field(default_factory=dict)
    missing: set[str] = field(default_factory=set)

    def spec(self, cap: str) -> EngineSpec:
        def probe() -> None:
            if cap in self.missing:
                raise ModuleNotFoundError(f"No module named 'fake_{cap}_backend'")

        def build() -> Any:
            self.builds[cap] = self.builds.get(cap, 0) + 1
            return getattr(self, cap)

        return EngineSpec(build=build, probe=probe)

    def engines(self) -> Engines:
        return Engines(
            {cap: self.spec(cap) for cap in ("embed", "rerank", "ocr", "transcribe", "vision")}
        )


@pytest.fixture
def settings() -> Settings:
    return Settings.from_env({"MLX_API_KEY": API_KEY})


@pytest.fixture
def fakes() -> Fakes:
    return Fakes()


@pytest.fixture
def make_client(settings: Settings, fakes: Fakes) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def make(**kwargs: Any) -> TestClient:
        # create_app would build real (lazy, unloaded) specs; the override replaces them.
        app = create_app(settings)
        engines = fakes.engines()
        app.dependency_overrides[get_engines] = lambda: engines
        c = TestClient(app, **kwargs)
        clients.append(c)
        return c

    yield make
    for c in clients:
        c.close()


@pytest.fixture
def client(make_client: Callable[..., TestClient]) -> TestClient:
    return make_client()


def assert_contract_error(resp: Any, status: int, type_: str) -> dict[str, Any]:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error"}
    err = body["error"]
    assert set(err) == {"message", "type", "code"}
    assert isinstance(err["message"], str) and err["message"]
    assert err["type"] == type_
    return err
