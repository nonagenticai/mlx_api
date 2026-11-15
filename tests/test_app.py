"""Cross-cutting route behaviour: auth, errors, health, models, ocr, transcribe."""

from __future__ import annotations

import base64
import threading
from collections.abc import Callable

import httpx
import pytest
from conftest import API_KEY, AUTH, Fakes, assert_contract_error
from fastapi.testclient import TestClient

from mlx_api import app as app_module
from mlx_api.app import create_app, sniff_mime
from mlx_api.config import ConfigError, Settings
from mlx_api.errors import InvalidRequest

_REAL_ASYNC_CLIENT = httpx.AsyncClient
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
PDF = b"%PDF-1.7\n" + b"x" * 32


def data_url(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


# ----------------------------------------------------------------- config


def test_refuses_to_start_without_key() -> None:
    with pytest.raises(ConfigError):
        Settings.from_env({})
    with pytest.raises(ConfigError):
        Settings.from_env({"MLX_API_KEY": "  "})


def test_config_defaults_and_overrides() -> None:
    s = Settings.from_env({"MLX_API_KEY": "k"})
    assert (s.host, s.port) == ("0.0.0.0", 9131)  # noqa: S104
    assert s.embed_model == "BAAI/bge-m3"
    assert s.rerank_model == "BAAI/bge-reranker-v2-m3"
    assert s.ocr_model == "mlx-community/PaddleOCR-VL-1.5-bf16"
    assert s.transcribe_model == "mlx-community/whisper-large-v3-turbo"
    s = Settings.from_env({"MLX_API_KEY": "k", "PORT": "9999", "MLX_API_EMBED_MODEL": "x/y"})
    assert (s.port, s.embed_model) == (9999, "x/y")
    with pytest.raises(ConfigError):
        Settings.from_env({"MLX_API_KEY": "k", "PORT": "nope"})


def test_capabilities_subset_config() -> None:
    assert Settings.from_env({"MLX_API_KEY": "k"}).capabilities == (
        "embed",
        "rerank",
        "ocr",
        "transcribe",
        "vision",
    )
    s = Settings.from_env({"MLX_API_KEY": "k", "MLX_API_CAPABILITIES": " rerank, embed ,embed"})
    assert s.capabilities == ("rerank", "embed")
    for bad in ("embed,video", ",", "chat"):
        with pytest.raises(ConfigError):
            Settings.from_env({"MLX_API_KEY": "k", "MLX_API_CAPABILITIES": bad})


def test_cache_limit_config() -> None:
    assert Settings.from_env({"MLX_API_KEY": "k"}).cache_limit_mb is None
    s = Settings.from_env({"MLX_API_KEY": "k", "MLX_API_CACHE_LIMIT_MB": "512"})
    assert s.cache_limit_mb == 512
    assert (
        Settings.from_env({"MLX_API_KEY": "k", "MLX_API_CACHE_LIMIT_MB": "0"}).cache_limit_mb == 0
    )
    for bad in ("-1", "lots"):
        with pytest.raises(ConfigError):
            Settings.from_env({"MLX_API_KEY": "k", "MLX_API_CACHE_LIMIT_MB": bad})


def test_disabled_capability_is_503_and_never_loads(fakes: Fakes) -> None:
    settings = Settings.from_env({"MLX_API_KEY": API_KEY, "MLX_API_CAPABILITIES": "ocr,transcribe"})
    app = create_app(settings)
    engines = fakes.engines()
    engines.enabled = frozenset(settings.capabilities)
    app.dependency_overrides[app_module.get_engines] = lambda: engines
    with TestClient(app) as c:
        resp = c.post("/v1/embeddings", headers=AUTH, json={"model": "bge-m3", "input": "hi"})
        assert_contract_error(resp, 503, "retryable")
        assert fakes.builds.get("embed", 0) == 0
        caps = c.get("/health").json()["capabilities"]
        assert caps["embed"]["enabled"] is False and caps["embed"]["available"] is False
        assert caps["ocr"]["enabled"] is True and caps["ocr"]["available"] is True
        listed = {m["capability"] for m in c.get("/v1/models", headers=AUTH).json()["data"]}
        assert listed == {"ocr", "transcribe"}


def test_create_app_refuses_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLX_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        create_app()


# ----------------------------------------------------------------- auth + errors


def test_health_is_open_and_reports_capabilities(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    caps = body["capabilities"]
    assert set(caps) == {"embed", "rerank", "ocr", "transcribe", "vision"}
    assert caps["embed"] == {
        "model": "BAAI/bge-m3",
        "available": True,
        "enabled": True,
        "loaded": False,
        "error": None,
    }


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/v1/models"),
        ("post", "/v1/embeddings"),
        ("post", "/v2/rerank"),
        ("post", "/v1/ocr"),
        ("post", "/v1/audio/transcriptions"),
        ("get", "/nope"),
    ],
)
@pytest.mark.parametrize("header", [None, "Bearer wrong", "Basic test-key", "Bearer "])
def test_auth_required_everywhere_but_health(
    client: TestClient, method: str, path: str, header: str | None
) -> None:
    headers = {"Authorization": header} if header is not None else {}
    resp = getattr(client, method)(path, headers=headers)
    assert_contract_error(resp, 401, "auth")


def test_unknown_path_is_contract_404(client: TestClient) -> None:
    assert_contract_error(client.get("/v1/nope", headers=AUTH), 404, "not_found")


def test_wrong_method_is_contract_405(client: TestClient) -> None:
    assert_contract_error(client.get("/v1/embeddings", headers=AUTH), 405, "invalid_request")


def test_malformed_json_is_contract_400(client: TestClient) -> None:
    resp = client.post(
        "/v1/embeddings",
        content=b"{not json",
        headers={**AUTH, "Content-Type": "application/json"},
    )
    assert_contract_error(resp, 400, "invalid_request")


def test_unhandled_exception_is_contract_500(
    make_client: Callable[..., TestClient], fakes: Fakes
) -> None:
    def boom(texts: list[str]) -> None:
        raise RuntimeError("kaput")

    fakes.embed.embed = boom  # type: ignore[method-assign]
    client = make_client(raise_server_exceptions=False)
    resp = client.post("/v1/embeddings", json={"model": "bge-m3", "input": "x"}, headers=AUTH)
    assert_contract_error(resp, 500, "retryable")


def test_models_lists_every_capability(client: TestClient) -> None:
    resp = client.get("/v1/models", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert [m["id"] for m in body["data"]] == [
        "BAAI/bge-m3",
        "BAAI/bge-reranker-v2-m3",
        "mlx-community/PaddleOCR-VL-1.5-bf16",
        "mlx-community/whisper-large-v3-turbo",
        "mlx-community/Qwen2-VL-2B-Instruct-4bit",
    ]
    for m in body["data"]:
        assert m["object"] == "model"
        assert isinstance(m["owned_by"], str) and m["owned_by"]


# ----------------------------------------------------------------- lazy engines


def test_engine_built_once_lazily_on_gpu_thread(client: TestClient, fakes: Fakes) -> None:
    assert fakes.builds == {}
    for _ in range(3):
        r = client.post("/v1/embeddings", json={"model": "bge-m3", "input": "a"}, headers=AUTH)
        assert r.status_code == 200
    assert fakes.builds == {"embed": 1}
    assert set(fakes.embed.threads) == {t for t in fakes.embed.threads if t.startswith("mlx-gpu")}
    assert fakes.embed.threads and threading.current_thread().name not in fakes.embed.threads
    health = client.get("/health").json()["capabilities"]
    assert health["embed"]["loaded"] is True
    assert health["rerank"]["loaded"] is False


def test_missing_backend_is_unavailable_not_fatal(
    make_client: Callable[..., TestClient], fakes: Fakes
) -> None:
    fakes.missing = {"transcribe"}
    client = make_client()
    caps = client.get("/health").json()["capabilities"]
    assert caps["transcribe"]["available"] is False
    assert "fake_transcribe_backend" in caps["transcribe"]["error"]
    assert caps["embed"]["available"] is True
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", b"RIFF....", "audio/wav")},
        data={"model": "whisper-large-v3-turbo"},
        headers=AUTH,
    )
    assert_contract_error(resp, 503, "retryable")
    ok = client.post("/v1/embeddings", json={"model": "bge-m3", "input": "a"}, headers=AUTH)
    assert ok.status_code == 200


def test_real_specs_report_missing_module(settings: Settings) -> None:
    """The default specs import lazily; an absent module is reported, not raised at startup."""
    app = create_app(settings)
    engines = app.state.engines
    engines.specs["ocr"] = app_module._module_spec("mlx_api._no_such_ocr", "X", "m")
    with TestClient(app) as c:
        caps = c.get("/health").json()["capabilities"]
        assert caps["ocr"]["available"] is False
        assert caps["embed"]["loaded"] is False  # health never loads weights
        resp = c.post(
            "/v1/ocr",
            json={
                "model": "PaddleOCR-VL-1.5-bf16",
                "document": {"type": "image_url", "image_url": data_url("image/png", PNG)},
            },
            headers=AUTH,
        )
        assert_contract_error(resp, 503, "retryable")


# ----------------------------------------------------------------- ocr


def test_ocr_data_url_image(client: TestClient, fakes: Fakes) -> None:
    resp = client.post(
        "/v1/ocr",
        json={
            "model": "mlx-community/PaddleOCR-VL-1.5-bf16",
            "document": {"type": "image_url", "image_url": data_url("image/png", PNG)},
        },
        headers=AUTH,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["model"] == "mlx-community/PaddleOCR-VL-1.5-bf16"
    assert body["pages"] == [
        {
            "index": 0,
            "markdown": "# page 0",
            "images": [],
            "dimensions": {"dpi": 72, "height": 200, "width": 100},
        }
    ]
    assert body["usage_info"] == {"pages_processed": 1, "doc_size_bytes": len(PNG)}
    assert fakes.ocr.calls == [(PNG, "image/png", None)]


def test_ocr_pdf_pages_sorted_unique_ascending(client: TestClient, fakes: Fakes) -> None:
    resp = client.post(
        "/v1/ocr",
        json={
            "model": "PaddleOCR-VL-1.5-bf16",
            "document": {"type": "document_url", "document_url": data_url("x/y", PDF)},
            "pages": [2, 0, 2],
        },
        headers=AUTH,
    )
    assert resp.status_code == 200, resp.text
    assert [p["index"] for p in resp.json()["pages"]] == [0, 2]
    assert fakes.ocr.calls[0][1:] == ("application/pdf", [0, 2])


@pytest.mark.parametrize(
    ("document", "status"),
    [
        ({"type": "document_url", "document_url": data_url("text/plain", b"hello")}, 400),
        ({"type": "document_url", "document_url": "data:application/pdf;base64"}, 400),
        ({"type": "document_url", "document_url": "ftp://x/y.pdf"}, 400),
        ({"type": "document_url", "document_url": data_url("application/pdf", b"")}, 400),
        ({"type": "video_url", "video_url": "data:,x"}, 400),
        ({"type": "image_url"}, 400),
    ],
)
def test_ocr_bad_documents(
    client: TestClient, fakes: Fakes, document: dict[str, str], status: int
) -> None:
    resp = client.post(
        "/v1/ocr", json={"model": "PaddleOCR-VL-1.5-bf16", "document": document}, headers=AUTH
    )
    assert_contract_error(resp, status, "invalid_request")
    assert fakes.builds.get("ocr") is None  # rejected before any model load


def test_ocr_negative_page_is_400(client: TestClient) -> None:
    resp = client.post(
        "/v1/ocr",
        json={
            "model": "PaddleOCR-VL-1.5-bf16",
            "document": {"type": "document_url", "document_url": data_url("a/b", PDF)},
            "pages": [-1],
        },
        headers=AUTH,
    )
    assert_contract_error(resp, 400, "invalid_request")


def test_ocr_engine_invalid_request_passes_through(client: TestClient, fakes: Fakes) -> None:
    def out_of_range(data: bytes, mime: str, pages: list[int] | None) -> list[object]:
        raise InvalidRequest("page 9 out of range", code="page_out_of_range")

    fakes.ocr.ocr = out_of_range  # type: ignore[method-assign]
    resp = client.post(
        "/v1/ocr",
        json={
            "model": "PaddleOCR-VL-1.5-bf16",
            "document": {"type": "document_url", "document_url": data_url("a/b", PDF)},
            "pages": [9],
        },
        headers=AUTH,
    )
    err = assert_contract_error(resp, 400, "invalid_request")
    assert err["code"] == "page_out_of_range"


def test_ocr_data_url_too_large_is_413(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app_module, "MAX_DOCUMENT_BYTES", 16)
    resp = client.post(
        "/v1/ocr",
        json={
            "model": "PaddleOCR-VL-1.5-bf16",
            "document": {"type": "document_url", "document_url": data_url("a/b", PDF)},
        },
        headers=AUTH,
    )
    assert_contract_error(resp, 413, "invalid_request")


def _mock_fetch(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
):
    real = _REAL_ASYNC_CLIENT

    def factory(**kwargs: object) -> httpx.AsyncClient:
        return real(transport=httpx.MockTransport(handler), **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(app_module.httpx, "AsyncClient", factory)

    async def public(url: str) -> None:  # example.test does not resolve; guard is tested apart
        return None

    monkeypatch.setattr(app_module, "_require_public_host", public)


def test_ocr_fetches_http_url(
    client: TestClient, fakes: Fakes, monkeypatch: pytest.MonkeyPatch
) -> None:
    _mock_fetch(monkeypatch, lambda req: httpx.Response(200, content=PDF))
    resp = client.post(
        "/v1/ocr",
        json={
            "model": "PaddleOCR-VL-1.5-bf16",
            "document": {"type": "document_url", "document_url": "https://example.test/a.pdf"},
        },
        headers=AUTH,
    )
    assert resp.status_code == 200, resp.text
    assert fakes.ocr.calls[0][:2] == (PDF, "application/pdf")
    assert resp.json()["usage_info"]["doc_size_bytes"] == len(PDF)


def test_ocr_fetch_errors(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    body = {
        "model": "PaddleOCR-VL-1.5-bf16",
        "document": {"type": "document_url", "document_url": "https://example.test/a.pdf"},
    }
    _mock_fetch(monkeypatch, lambda req: httpx.Response(404))
    assert_contract_error(client.post("/v1/ocr", json=body, headers=AUTH), 400, "invalid_request")

    def timeout(req: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=req)

    _mock_fetch(monkeypatch, timeout)
    assert_contract_error(client.post("/v1/ocr", json=body, headers=AUTH), 400, "invalid_request")

    monkeypatch.setattr(app_module, "MAX_DOCUMENT_BYTES", 16)
    _mock_fetch(monkeypatch, lambda req: httpx.Response(200, content=PDF))
    assert_contract_error(client.post("/v1/ocr", json=body, headers=AUTH), 413, "invalid_request")


@pytest.mark.parametrize(
    ("data", "mime"),
    [
        (PDF, "application/pdf"),
        (b"\n\n%PDF-1.4 junk", "application/pdf"),
        (PNG, "image/png"),
        (b"\xff\xd8\xff\xe0rest", "image/jpeg"),
        (b"GIF89a....", "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
        (b"II*\x00rest", "image/tiff"),
        (b"BMrest", "image/bmp"),
    ],
)
def test_sniff_mime(data: bytes, mime: str) -> None:
    assert sniff_mime(data) == mime


def test_sniff_mime_rejects_other() -> None:
    with pytest.raises(InvalidRequest):
        sniff_mime(b"PK\x03\x04 a zip")


# ----------------------------------------------------------------- transcribe


def _transcribe(client: TestClient, audio: bytes = b"RIFFaudio", **data: str) -> httpx.Response:
    form = {"model": "whisper-large-v3-turbo", **data}
    return client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.wav", audio, "audio/wav")},
        data=form,
        headers=AUTH,
    )


def test_transcribe_json_default(client: TestClient, fakes: Fakes) -> None:
    resp = _transcribe(client, language="en", prompt="names")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"text": "hello world", "usage": {"type": "duration", "seconds": 3.5}}
    assert fakes.transcribe.calls == [(b"RIFFaudio", "clip.wav", "en", "names")]


def test_transcribe_text(client: TestClient) -> None:
    resp = _transcribe(client, response_format="text")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text == "hello world"


def test_transcribe_verbose_json(client: TestClient) -> None:
    resp = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("clip.mp3", b"ID3audio", "audio/mpeg")},
        data={
            "model": "mlx-community/whisper-large-v3-turbo",
            "response_format": "verbose_json",
            "timestamp_granularities[]": "segment",
        },
        headers=AUTH,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["task"] == "transcribe"
    assert body["language"] == "english"
    assert body["duration"] == 3.5
    assert body["text"] == "hello world"
    assert body["usage"] == {"type": "duration", "seconds": 3.5}
    segs = body["segments"]
    assert [s["id"] for s in segs] == [0, 1]
    assert all(0 <= s["start"] <= s["end"] <= body["duration"] for s in segs)


def test_transcribe_errors(client: TestClient) -> None:
    assert_contract_error(_transcribe(client, audio=b""), 400, "invalid_request")
    assert_contract_error(_transcribe(client, response_format="srt"), 400, "invalid_request")
    assert_contract_error(
        _transcribe(client, **{"timestamp_granularities[]": "word"}), 400, "invalid_request"
    )
    assert_contract_error(_transcribe(client, model="whisper-tiny"), 404, "not_found")
    no_file = client.post(
        "/v1/audio/transcriptions", data={"model": "whisper-large-v3-turbo"}, headers=AUTH
    )
    assert_contract_error(no_file, 400, "invalid_request")
    no_model = client.post(
        "/v1/audio/transcriptions", files={"file": ("a.wav", b"x", "audio/wav")}, headers=AUTH
    )
    assert_contract_error(no_model, 400, "invalid_request")
    as_json = client.post("/v1/audio/transcriptions", json={"model": "x"}, headers=AUTH)
    assert_contract_error(as_json, 400, "invalid_request")


# ----------------------------------------------------------------- vision


VISION_MODEL = "mlx-community/Qwen2-VL-2B-Instruct-4bit"


def _png(w: int = 4, h: int = 3) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (w, h), (200, 10, 10)).save(buf, format="PNG")
    return buf.getvalue()


def _vision_body(*parts: dict[str, object], **extra: object) -> dict[str, object]:
    return {
        "model": VISION_MODEL,
        "messages": [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": list(parts)},
        ],
        **extra,
    }


def test_vision_data_url_completion(client: TestClient, fakes: Fakes) -> None:
    body = _vision_body(
        {"type": "text", "text": "Describe."},
        {"type": "image_url", "image_url": {"url": data_url("image/png", _png())}},
        max_tokens=32,
    )
    resp = client.post("/v1/chat/completions", json=body, headers=AUTH)
    assert resp.status_code == 200, resp.text
    out = resp.json()
    assert out["object"] == "chat.completion"
    assert out["id"].startswith("chatcmpl-")
    assert isinstance(out["created"], int)
    assert out["model"] == VISION_MODEL
    assert out["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "1 image(s)"},
            "finish_reason": "stop",
        }
    ]
    assert out["usage"] == {"prompt_tokens": 30, "completion_tokens": 4, "total_tokens": 34}
    turns, images, max_tokens, temperature = fakes.vision.calls[0]
    assert [(t.role, t.text) for t in turns] == [("system", "Be brief."), ("user", "Describe.")]
    assert [img.size for img in images] == [(4, 3)]
    assert (max_tokens, temperature) == (32, 0.0)


def test_vision_max_completion_tokens_wins_and_default(client: TestClient, fakes: Fakes) -> None:
    img = {"type": "image_url", "image_url": {"url": data_url("image/png", _png())}}
    r = client.post(
        "/v1/chat/completions",
        json=_vision_body(img, max_tokens=10, max_completion_tokens=20),
        headers=AUTH,
    )
    assert r.status_code == 200, r.text
    r = client.post("/v1/chat/completions", json=_vision_body(img), headers=AUTH)
    assert r.status_code == 200, r.text
    assert [c[2] for c in fakes.vision.calls] == [20, app_module.DEFAULT_VISION_MAX_TOKENS]


def test_vision_short_alias_accepted(client: TestClient) -> None:
    body = _vision_body({"type": "text", "text": "hi"})
    body["model"] = "Qwen2-VL-2B-Instruct-4bit"
    assert client.post("/v1/chat/completions", json=body, headers=AUTH).status_code == 200


@pytest.mark.parametrize(
    ("body", "status", "type_"),
    [
        ({"model": VISION_MODEL, "max_tokens": 16}, 400, "invalid_request"),
        ({"model": VISION_MODEL, "messages": []}, 400, "invalid_request"),
        (_vision_body({"type": "text", "text": "x"}, stream=True), 400, "invalid_request"),
        (
            _vision_body({"type": "text", "text": "x"}, response_format={"type": "json_object"}),
            400,
            "invalid_request",
        ),
        (_vision_body({"type": "text", "text": "x"}, max_tokens=4097), 400, "invalid_request"),
        (
            _vision_body({"type": "image_url", "image_url": {"url": data_url("x/y", PDF)}}),
            400,
            "invalid_request",
        ),
        (
            _vision_body({"type": "image_url", "image_url": {"url": data_url("image/png", PNG)}}),
            400,
            "invalid_request",
        ),
        (
            _vision_body({"type": "image_url", "image_url": {"url": "ftp://x/y.png"}}),
            400,
            "invalid_request",
        ),
        (
            {**_vision_body({"type": "text", "text": "x"}), "model": "gpt-4o"},
            404,
            "not_found",
        ),
    ],
)
def test_vision_bad_requests(
    client: TestClient, fakes: Fakes, body: dict[str, object], status: int, type_: str
) -> None:
    resp = client.post("/v1/chat/completions", json=body, headers=AUTH)
    assert_contract_error(resp, status, type_)
    assert fakes.builds.get("vision") is None  # rejected before any model load


def test_vision_image_only_in_user_messages(client: TestClient, fakes: Fakes) -> None:
    img = {"type": "image_url", "image_url": {"url": data_url("image/png", _png())}}
    body = {"model": VISION_MODEL, "messages": [{"role": "assistant", "content": [img]}]}
    assert_contract_error(
        client.post("/v1/chat/completions", json=body, headers=AUTH), 400, "invalid_request"
    )
    assert fakes.vision.calls == []


def test_vision_too_many_images_is_413(client: TestClient, fakes: Fakes) -> None:
    img = {"type": "image_url", "image_url": {"url": data_url("image/png", _png())}}
    n = app_module.MAX_VISION_IMAGES + 1
    resp = client.post("/v1/chat/completions", json=_vision_body(*[img] * n), headers=AUTH)
    assert_contract_error(resp, 413, "invalid_request")
    assert fakes.vision.calls == []


def test_vision_requires_auth(client: TestClient) -> None:
    resp = client.post("/v1/chat/completions", json=_vision_body({"type": "text", "text": "x"}))
    assert_contract_error(resp, 401, "auth")


def test_vision_disabled_is_503(fakes: Fakes) -> None:
    settings = Settings.from_env({"MLX_API_KEY": API_KEY, "MLX_API_CAPABILITIES": "embed,rerank"})
    app = create_app(settings)
    engines = fakes.engines()
    engines.enabled = frozenset(settings.capabilities)
    app.dependency_overrides[app_module.get_engines] = lambda: engines
    with TestClient(app) as c:
        resp = c.post(
            "/v1/chat/completions",
            json=_vision_body({"type": "text", "text": "x"}),
            headers=AUTH,
        )
        assert_contract_error(resp, 503, "retryable")
        models = c.get("/v1/models", headers=AUTH).json()["data"]
        assert {m["capability"] for m in models} == {"embed", "rerank"}
    assert fakes.builds.get("vision") is None
