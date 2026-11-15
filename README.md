# mlx_api

Local Apple-silicon implementation of the provider wire contract: the same routes and schemas as
commercial embedding, reranking, OCR, transcription and vision providers, so an application can
switch between local models and a hosted API with no code change.

Requires macOS on Apple silicon, Python 3.14+ and `ffmpeg` on `PATH`.

| Capability | Route | Default model |
|---|---|---|
| embed | `POST /v1/embeddings` (OpenAI) | `BAAI/bge-m3` (dense, CLS, L2, 1024-d, 8192 tokens) |
| rerank | `POST /v2/rerank` (Cohere v2) | `BAAI/bge-reranker-v2-m3` (sigmoid scores) |
| ocr | `POST /v1/ocr` (Mistral) | `mlx-community/PaddleOCR-VL-1.5-bf16` |
| transcribe | `POST /v1/audio/transcriptions` (OpenAI) | `mlx-community/whisper-large-v3-turbo` |
| vision | `POST /v1/chat/completions` (OpenAI, `image_url` parts) | `mlx-community/Qwen2-VL-2B-Instruct-4bit` |

Also `GET /health` (no auth) and `GET /v1/models`. Requests may name a model by its full id or
its short alias (`bge-m3`); anything else is `404 not_found`. Every non-2xx uses the contract
error body.

`vision` is image captioning/description only: non-streaming, `response_format` text only,
at most 8 images (data: URLs or public https URLs, same SSRF guard as ocr). Text `chat` is not
served here; `/v1/chat/completions` answers only for the vision model.

## Run

```bash
uv sync
MLX_API_KEY=dev uv run python -m mlx_api.main      # 0.0.0.0:9131
curl -s localhost:9131/health
curl -s localhost:9131/v1/embeddings -H 'Authorization: Bearer dev' \
  -H 'Content-Type: application/json' -d '{"model":"bge-m3","input":["hello"]}'
```

Engines load lazily on first use and are cached. A capability whose module cannot be imported
(e.g. a broken mlx-vlm install) shows `available: false` in `/health` and
its route answers `503`; the others keep working. All MLX work runs on one GPU thread
(`mlx_api.gpu.run_on_gpu`).

Weights are read from the local Hugging Face cache; nothing is downloaded at runtime
(`HF_HUB_OFFLINE=1` is fine). A model id may also be a local directory.

## Configuration (env)

| Variable | Default |
|---|---|
| `MLX_API_KEY` | required, refuses to start without it |
| `HOST` / `PORT` | `0.0.0.0` / `9131` |
| `MLX_API_EMBED_MODEL`, `MLX_API_RERANK_MODEL`, `MLX_API_OCR_MODEL`, `MLX_API_TRANSCRIBE_MODEL`, `MLX_API_VISION_MODEL` | see table above |
| `MLX_API_EMBED_MAX_LENGTH` / `MLX_API_RERANK_MAX_LENGTH` | `8192` / `1024` tokens (truncation) |
| `MLX_API_EMBED_TOKEN_BUDGET` / `MLX_API_RERANK_TOKEN_BUDGET` | `16384` / `8192` padded token slots per forward pass |
| `MLX_API_CAPABILITIES` | all five; comma subset (e.g. `embed,rerank`) — the rest answer 503 and never load |
| `MLX_API_CACHE_LIMIT_MB` | unset (MLX default); caps MLX's buffer cache, which dominates footprint under load |
| `MLX_API_DTYPE` | `float32` (`float16`/`bfloat16` ~15% faster, not parity-tested) |

## Implementation notes

bge-m3 and bge-reranker-v2-m3 are both XLM-RoBERTa-large; `models.py` is a small mlx.nn port
that loads the HF safetensors directly (torch is only present because mlx-whisper requires it). Inputs are tokenized with the
model's own `AutoTokenizer`, sorted by real length and packed into batches under a token budget,
so short texts are not padded to long ones and long texts are never cut to a bucket size.

## Tests

```bash
uv run pytest -q              # fast: routes run against fake engines
uv run pytest -q -m models    # real weights; parity vs PyTorch (needs the `parity` group)
```

Parity acceptance: bge-m3 cosine ≥ 0.999 per text vs `transformers.AutoModel` CLS-pool on CPU;
reranker |score diff| ≤ 0.01 vs `AutoModelForSequenceClassification` sigmoid.

## OCR document URLs

`data:` URLs always work. `http(s)` URLs are fetched only if the host resolves to a public address
(no private, loopback, link-local or CGNAT targets) and redirects are not followed. Private
documents go as `data:` URLs, exactly as they must for a commercial provider.

## Conformance

All five capabilities were verified against the provider wire contract on Apple-silicon hardware.
The conformance harness is not part of this repository; the route table above and the request
models in `src/mlx_api/schemas.py` are the contract this service implements.

## Gates

`uv run ruff check . && uv run ruff format --check . && uv run ty check src && uv run pytest -q`;
real-model tests (parity vs PyTorch, OCR, whisper): `uv run --group parity pytest -q -m models`.
CI runs the fast suite on a macOS runner; the real-model tests run on device.

## License

Apache 2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

## Licensing

This project is **not** licence-gated. There is no licence server, activation step or runtime
check: the service runs entirely on your machine and never contacts a licensing service. The
Apache-2.0 licence is the only condition of use.
