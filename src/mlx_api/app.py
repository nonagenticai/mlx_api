"""FastAPI app: provider-contract routes over lazily loaded MLX engines.

Contract: the provider wire contract documented in README. Every engine call goes through
``gpu.run_on_gpu``.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import importlib
import io
import ipaddress
import logging
import math
import time
import urllib.parse
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any

import httpx
import numpy as np
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from PIL import Image, UnidentifiedImageError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from mlx_api import __version__
from mlx_api.auth import BearerAuthMiddleware
from mlx_api.config import CAPABILITIES, Settings, matches_model, model_alias
from mlx_api.errors import (
    ContractError,
    InvalidRequest,
    ModelNotFound,
    PayloadTooLarge,
    Unavailable,
)
from mlx_api.gpu import run_on_gpu
from mlx_api.schemas import (
    ChatRequest,
    DocumentURL,
    EmbeddingsRequest,
    OcrRequest,
    RerankRequest,
    TextPart,
    error_response,
)

log = logging.getLogger("mlx_api")

MAX_EMBED_INPUTS = 2048
MAX_RERANK_DOCUMENTS = 1000
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
MAX_AUDIO_BYTES = 200 * 1024 * 1024
MAX_VISION_IMAGES = 8
DEFAULT_VISION_MAX_TOKENS = 512
MAX_VISION_MAX_TOKENS = 4096
FETCH_TIMEOUT_S = 10.0


# --------------------------------------------------------------------------- engines


@dataclass
class EngineSpec:
    """How to make one capability's engine.

    ``probe`` checks the capability can exist at all (its module imports); ``build`` constructs
    the engine and runs on the GPU thread. An ``ImportError`` from either marks the capability
    unavailable for the life of the process.
    """

    build: Callable[[], Any]
    probe: Callable[[], object] = lambda: None


def _module_spec(module: str, cls: str, model_id: str, **kwargs: Any) -> EngineSpec:
    def probe() -> object:
        return importlib.import_module(module)

    def build() -> Any:
        return getattr(importlib.import_module(module), cls)(model_id, **kwargs)

    return EngineSpec(build=build, probe=probe)


def default_specs(settings: Settings) -> dict[str, EngineSpec]:
    return {
        "embed": _module_spec(
            "mlx_api.embed",
            "Embedder",
            settings.embed_model,
            max_length=settings.embed_max_length,
            token_budget=settings.embed_token_budget,
            dtype=settings.dtype,
        ),
        "rerank": _module_spec(
            "mlx_api.rerank",
            "Reranker",
            settings.rerank_model,
            max_length=settings.rerank_max_length,
            token_budget=settings.rerank_token_budget,
            dtype=settings.dtype,
        ),
        "ocr": _module_spec("mlx_api.ocr", "OcrEngine", settings.ocr_model),
        "transcribe": _module_spec("mlx_api.transcribe", "Transcriber", settings.transcribe_model),
        "vision": _module_spec("mlx_api.vision", "VisionEngine", settings.vision_model),
    }


@dataclass
class Engines:
    specs: dict[str, EngineSpec]
    enabled: frozenset[str] = frozenset(CAPABILITIES)
    _engines: dict[str, Any] = field(default_factory=dict)
    _import_errors: dict[str, str] = field(default_factory=dict)
    _load_errors: dict[str, str] = field(default_factory=dict)
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    def _unavailable(self, cap: str, exc: BaseException) -> Unavailable:
        self._import_errors[cap] = f"{type(exc).__name__}: {exc}"
        return Unavailable(f"{cap} is unavailable on this host: {self._import_errors[cap]}")

    def _disabled(self, cap: str) -> Unavailable:
        return Unavailable(f"{cap} is not enabled on this host (MLX_API_CAPABILITIES)")

    async def probe(self, cap: str) -> bool:
        if cap not in self.enabled:
            return False
        if cap in self._engines:
            return True
        if cap in self._import_errors:
            return False
        try:
            await asyncio.to_thread(self.specs[cap].probe)
        except ImportError as exc:
            self._unavailable(cap, exc)
            return False
        return True

    async def get(self, cap: str) -> Any:
        if cap not in self.enabled:
            raise self._disabled(cap)
        if cap in self._engines:
            return self._engines[cap]
        if cap in self._import_errors:
            raise Unavailable(f"{cap} is unavailable on this host: {self._import_errors[cap]}")
        lock = self._locks.setdefault(cap, asyncio.Lock())
        async with lock:
            if cap in self._engines:
                return self._engines[cap]
            try:
                await asyncio.to_thread(self.specs[cap].probe)
                engine = await run_on_gpu(self.specs[cap].build)
            except ImportError as exc:
                raise self._unavailable(cap, exc) from exc
            except ContractError as exc:
                self._load_errors[cap] = exc.message
                raise
            except Exception as exc:
                self._load_errors[cap] = f"{type(exc).__name__}: {exc}"
                log.exception("loading %s failed", cap)
                raise Unavailable(f"{cap} failed to load: {self._load_errors[cap]}") from exc
            self._load_errors.pop(cap, None)
            self._engines[cap] = engine
            return engine

    async def status(self, cap: str) -> dict[str, Any]:
        available = await self.probe(cap)
        return {
            "available": available,
            "enabled": cap in self.enabled,
            "loaded": cap in self._engines,
            "error": (
                self._import_errors.get(cap)
                or self._load_errors.get(cap)
                or (None if cap in self.enabled else "not enabled on this host")
            ),
        }


def get_engines(request: Request) -> Engines:
    return request.app.state.engines


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


EnginesDep = Annotated[Engines, Depends(get_engines)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


def require_model(requested: str, settings: Settings, cap: str) -> None:
    configured = settings.model_for(cap)
    if not matches_model(requested, configured):
        raise ModelNotFound(
            f"model {requested!r} not found; {cap} serves {configured!r}", code="model_not_found"
        )


# --------------------------------------------------------------------------- documents (ocr)


_MAGIC: tuple[tuple[bytes, int, str], ...] = (
    (b"%PDF-", 0, "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", 0, "image/png"),
    (b"\xff\xd8\xff", 0, "image/jpeg"),
    (b"GIF87a", 0, "image/gif"),
    (b"GIF89a", 0, "image/gif"),
    (b"WEBP", 8, "image/webp"),
    (b"II*\x00", 0, "image/tiff"),
    (b"MM\x00*", 0, "image/tiff"),
    (b"BM", 0, "image/bmp"),
)


def sniff_mime(data: bytes) -> str:
    head = data[:1024]
    for magic, offset, mime in _MAGIC:
        if magic == b"WEBP" and not head.startswith(b"RIFF"):
            continue
        if head[offset : offset + len(magic)] == magic:
            return mime
    # PDFs may carry leading junk before the header; the spec allows it within the first 1 KiB.
    if b"%PDF-" in head:
        return "application/pdf"
    raise InvalidRequest("document is neither a PDF nor a supported image", code="unsupported_mime")


def decode_data_url(url: str) -> bytes:
    header, sep, payload = url[len("data:") :].partition(",")
    if not sep:
        raise InvalidRequest("malformed data: URL", code="invalid_document")
    if len(payload) > MAX_DOCUMENT_BYTES * 4 // 3 + 4:
        raise PayloadTooLarge("document exceeds 50 MB", code="document_too_large")
    if header.endswith(";base64"):
        try:
            data = base64.b64decode(payload, validate=False)
        except (binascii.Error, ValueError) as exc:
            raise InvalidRequest("data: URL is not valid base64", code="invalid_document") from exc
    else:
        data = urllib.parse.unquote_to_bytes(payload)
    if len(data) > MAX_DOCUMENT_BYTES:
        raise PayloadTooLarge("document exceeds 50 MB", code="document_too_large")
    return data


async def _require_public_host(url: str) -> None:
    """SSRF guard: this runs on a tailnet host, so only globally routable targets may be fetched.

    Private, loopback, link-local and CGNAT (tailnet 100.64/10) addresses are all non-global.
    Internal documents must be sent as data: URLs, exactly as for a commercial provider.
    """
    host = urllib.parse.urlsplit(url).hostname
    if not host:
        raise InvalidRequest("document URL has no host", code="invalid_document")
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None)
    except OSError as exc:
        raise InvalidRequest(
            "document host does not resolve", code="document_fetch_failed"
        ) from exc
    for info in infos:
        if not ipaddress.ip_address(info[4][0]).is_global:
            raise InvalidRequest(
                "document URL must resolve to a public address; send a data: URL instead",
                code="document_url_not_public",
            )


async def fetch_url(url: str) -> bytes:
    await _require_public_host(url)
    try:
        async with (
            # No redirects: a public URL could otherwise bounce to an internal one.
            httpx.AsyncClient(timeout=FETCH_TIMEOUT_S, follow_redirects=False) as client,
            client.stream("GET", url) as resp,
        ):
            if resp.status_code >= 300:
                raise InvalidRequest(
                    f"fetching document failed: HTTP {resp.status_code}",
                    code="document_fetch_failed",
                )
            declared = resp.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > MAX_DOCUMENT_BYTES:
                raise PayloadTooLarge("document exceeds 50 MB", code="document_too_large")
            chunks: list[bytes] = []
            size = 0
            async for chunk in resp.aiter_bytes():
                size += len(chunk)
                if size > MAX_DOCUMENT_BYTES:
                    raise PayloadTooLarge("document exceeds 50 MB", code="document_too_large")
                chunks.append(chunk)
            return b"".join(chunks)
    except httpx.HTTPError as exc:
        raise InvalidRequest(
            f"fetching document failed: {type(exc).__name__}", code="document_fetch_failed"
        ) from exc


async def load_document(url: str) -> bytes:
    if url.startswith("data:"):
        data = decode_data_url(url)
    elif url.startswith(("http://", "https://")):
        data = await fetch_url(url)
    else:
        raise InvalidRequest("document URL must be data:, http:// or https://", code="invalid_url")
    if not data:
        raise InvalidRequest("document is empty", code="invalid_document")
    return data


async def load_image(url: str) -> Image.Image:
    data = await load_document(url)
    mime = sniff_mime(data)
    if not mime.startswith("image/"):
        raise InvalidRequest(f"image_url must be an image, got {mime}", code="unsupported_mime")
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            return img.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        raise InvalidRequest(f"corrupt or unreadable image: {exc}", code="corrupt_image") from exc


# --------------------------------------------------------------------------- app


def create_app(
    settings: Settings | None = None, specs: dict[str, EngineSpec] | None = None
) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(title="mlx_api", version=__version__, docs_url=None, redoc_url=None)
    app.state.settings = settings
    app.state.engines = Engines(
        specs if specs is not None else default_specs(settings),
        enabled=frozenset(settings.capabilities),
    )
    app.add_middleware(BearerAuthMiddleware, api_key=settings.api_key)

    @app.exception_handler(ContractError)
    async def _contract_error(_: Request, exc: ContractError) -> JSONResponse:
        return error_response(exc.status, exc.message, exc.type, exc.code)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        parts = []
        for err in exc.errors()[:5]:
            loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
            parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
        return error_response(400, "; ".join(parts) or "invalid request", code="validation_error")

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        resp = error_response(exc.status_code, str(exc.detail))
        for key, value in (exc.headers or {}).items():
            resp.headers[key] = value
        return resp

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", exc_info=exc)
        return error_response(500, f"internal error: {type(exc).__name__}", "retryable")

    @app.get("/health")
    async def health(engines: EnginesDep, cfg: SettingsDep) -> dict[str, Any]:
        caps = {}
        for cap in CAPABILITIES:
            caps[cap] = {"model": cfg.model_for(cap), **(await engines.status(cap))}
        return {"status": "ok", "version": __version__, "capabilities": caps}

    @app.get("/v1/models")
    async def models(cfg: SettingsDep) -> dict[str, Any]:
        data = []
        for cap in cfg.capabilities:
            model_id = cfg.model_for(cap)
            owner = model_id.split("/", 1)[0] if "/" in model_id else "mlx_api"
            data.append({"id": model_id, "object": "model", "owned_by": owner, "capability": cap})
        return {"object": "list", "data": data}

    @app.post("/v1/embeddings")
    async def embeddings(
        body: EmbeddingsRequest, engines: EnginesDep, cfg: SettingsDep
    ) -> dict[str, Any]:
        require_model(body.model, cfg, "embed")
        texts = [body.input] if isinstance(body.input, str) else body.input
        if not texts:
            raise InvalidRequest("input must not be empty", code="empty_input")
        if len(texts) > MAX_EMBED_INPUTS:
            raise PayloadTooLarge(f"at most {MAX_EMBED_INPUTS} inputs per request")
        if any(t == "" for t in texts):
            raise InvalidRequest("input strings must not be empty", code="empty_input")
        engine = await engines.get("embed")
        if body.dimensions is not None and body.dimensions != engine.dim:
            raise InvalidRequest(
                f"{model_alias(cfg.embed_model)} produces {engine.dim} dimensions, "
                f"not {body.dimensions}",
                code="unsupported_dimensions",
            )
        result = await run_on_gpu(lambda: engine.embed(texts))
        vectors = np.asarray(result.vectors, dtype=np.float32)
        data = []
        for i, vec in enumerate(vectors):
            if body.encoding_format == "base64":
                emb: Any = base64.b64encode(vec.astype("<f4").tobytes()).decode("ascii")
            else:
                emb = vec.tolist()
            data.append({"object": "embedding", "index": i, "embedding": emb})
        tokens = int(result.prompt_tokens)
        return {
            "object": "list",
            "model": body.model,
            "data": data,
            "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
        }

    @app.post("/v2/rerank")
    async def rerank(body: RerankRequest, engines: EnginesDep, cfg: SettingsDep) -> dict[str, Any]:
        require_model(body.model, cfg, "rerank")
        docs = body.documents
        if not docs:
            raise InvalidRequest("documents must not be empty", code="empty_documents")
        if len(docs) > MAX_RERANK_DOCUMENTS:
            raise PayloadTooLarge(f"at most {MAX_RERANK_DOCUMENTS} documents per request")
        engine = await engines.get("rerank")
        result = await run_on_gpu(lambda: engine.score(body.query, docs))
        scores = [min(1.0, max(0.0, float(s))) for s in result.scores]
        order = sorted(range(len(docs)), key=lambda i: (-scores[i], i))
        top_n = min(body.top_n or len(docs), len(docs))
        return {
            "id": str(uuid.uuid4()),
            "results": [{"index": i, "relevance_score": scores[i]} for i in order[:top_n]],
            "meta": {
                "billed_units": {"search_units": math.ceil(len(docs) / 100)},
                "tokens": {"input_tokens": int(result.tokens)},
            },
        }

    @app.post("/v1/ocr")
    async def ocr(body: OcrRequest, engines: EnginesDep, cfg: SettingsDep) -> dict[str, Any]:
        require_model(body.model, cfg, "ocr")
        doc = body.document
        url = doc.document_url if isinstance(doc, DocumentURL) else doc.image_url
        pages = None
        if body.pages is not None:
            if any(p < 0 for p in body.pages):
                raise InvalidRequest(
                    "pages must be 0-based non-negative indices", code="invalid_pages"
                )
            pages = sorted(set(body.pages))
        data = await load_document(url)
        mime = sniff_mime(data)
        engine = await engines.get("ocr")
        result = await run_on_gpu(lambda: engine.ocr(data, mime, pages))
        out_pages = [
            {
                "index": p.index,
                "markdown": p.markdown,
                "images": [],
                "dimensions": {"dpi": p.dpi, "height": p.height, "width": p.width},
            }
            for p in sorted(result, key=lambda p: p.index)
        ]
        return {
            "model": body.model,
            "pages": out_pages,
            "usage_info": {"pages_processed": len(out_pages), "doc_size_bytes": len(data)},
        }

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(request: Request, engines: EnginesDep, cfg: SettingsDep) -> Response:
        if not request.headers.get("content-type", "").startswith("multipart/form-data"):
            raise InvalidRequest("expected multipart/form-data", code="invalid_content_type")
        try:
            form = await request.form()
        except Exception as exc:
            raise InvalidRequest(f"malformed multipart body: {exc}") from exc

        def field_str(name: str) -> str | None:
            value = form.get(name)
            if value is None or isinstance(value, UploadFile):
                return None
            return value or None

        model = field_str("model")
        if model is None:
            raise InvalidRequest("model is required", code="missing_model")
        require_model(model, cfg, "transcribe")
        upload = form.get("file")
        if not isinstance(upload, UploadFile):
            raise InvalidRequest("file is required", code="missing_file")
        response_format = field_str("response_format") or "json"
        if response_format not in ("json", "text", "verbose_json"):
            raise InvalidRequest(
                f"response_format {response_format!r} is not supported", code="invalid_format"
            )
        granularities = [
            g
            for k in ("timestamp_granularities[]", "timestamp_granularities")
            for g in form.getlist(k)
        ]
        if any(g != "segment" for g in granularities):
            raise InvalidRequest("only segment timestamp granularity is supported")
        audio = await upload.read(MAX_AUDIO_BYTES + 1)
        if len(audio) > MAX_AUDIO_BYTES:
            raise PayloadTooLarge("audio exceeds 200 MB", code="file_too_large")
        if not audio:
            raise InvalidRequest("file is empty", code="empty_file")
        language, prompt = field_str("language"), field_str("prompt")
        filename = upload.filename or "audio"
        engine = await engines.get("transcribe")
        t = await run_on_gpu(lambda: engine.transcribe(audio, filename, language, prompt))
        usage = {"type": "duration", "seconds": float(t.duration)}
        if response_format == "text":
            return PlainTextResponse(t.text)
        if response_format == "json":
            return JSONResponse({"text": t.text, "usage": usage})
        return JSONResponse(
            {
                "task": "transcribe",
                "language": t.language,
                "duration": float(t.duration),
                "text": t.text,
                "segments": [
                    {"id": s.id, "start": float(s.start), "end": float(s.end), "text": s.text}
                    for s in t.segments
                ],
                "usage": usage,
            }
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(
        body: ChatRequest, engines: EnginesDep, cfg: SettingsDep
    ) -> dict[str, Any]:
        from mlx_api.vision import Turn

        # This process serves the `vision` capability only (image captioning/description);
        # text `chat` is not implemented here, so the model must be the vision model.
        require_model(body.model, cfg, "vision")
        if body.stream:
            raise InvalidRequest(
                "stream is not supported by this provider", code="stream_unsupported"
            )
        if body.response_format is not None and body.response_format.get("type") != "text":
            raise InvalidRequest(
                "only response_format {'type': 'text'} is supported",
                code="response_format_unsupported",
            )
        max_tokens = body.max_completion_tokens or body.max_tokens or DEFAULT_VISION_MAX_TOKENS
        if max_tokens > MAX_VISION_MAX_TOKENS:
            raise InvalidRequest(
                f"max_tokens must be <= {MAX_VISION_MAX_TOKENS}", code="max_tokens_too_large"
            )
        turns: list[Turn] = []
        image_urls: list[str] = []
        for msg in body.messages:
            if isinstance(msg.content, str):
                turns.append(Turn(msg.role, msg.content))
                continue
            texts = []
            for part in msg.content:
                if isinstance(part, TextPart):
                    texts.append(part.text)
                else:
                    if msg.role != "user":
                        raise InvalidRequest("images are only allowed in user messages")
                    image_urls.append(part.image_url.url)
            turns.append(Turn(msg.role, "\n".join(texts)))
        if len(image_urls) > MAX_VISION_IMAGES:
            raise PayloadTooLarge(f"at most {MAX_VISION_IMAGES} images per request")
        images = [await load_image(u) for u in image_urls]
        engine = await engines.get("vision")
        temperature = body.temperature
        result = await run_on_gpu(lambda: engine.complete(turns, images, max_tokens, temperature))
        prompt_tokens, completion_tokens = int(result.prompt_tokens), int(result.completion_tokens)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result.text},
                    "finish_reason": result.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app
