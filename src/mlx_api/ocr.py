"""OCR engine — the scanned-document path of the `ocr` contract (Mistral OCR shape).

Renders PDF pages with pypdfium2 (no text layer is used: born-digital text is docling's job
upstream) and runs PaddleOCR-VL's ``"OCR:"`` prompt per page through mlx-vlm.

Blocking by design: callers run ``OcrEngine.ocr`` on the single GPU thread (``mlx_api.gpu``).
The model is loaded lazily on the first ``ocr()`` call, i.e. on that same thread.
"""

from __future__ import annotations

import io
import os
import re
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from PIL import Image, UnidentifiedImageError

from mlx_api.errors import InvalidRequest

PDF_MIME = "application/pdf"
IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/jpg", "image/webp", "image/tiff"})

RENDER_DPI = 200
MAX_SIDE_PX = 2000
DEFAULT_IMAGE_DPI = 72
MAX_TOKENS = 2048
OCR_PROMPT = "OCR:"

# PaddleOCR-VL / ERNIE special tokens: <|LOC_12|>, <|end_of_sentence|>, <|IMAGE_START|>, </s>, ...
_SPECIAL_TOKEN_RE = re.compile(r"<\|[^<>|]{1,64}\|>|</?s>|<unk>|<pad>")

# (prompt, page image) -> raw model text. Injected in tests; the default runs mlx-vlm.
GenerateFn = Callable[[str, Image.Image], str]


@dataclass(frozen=True)
class OcrPage:
    index: int
    markdown: str
    width: int
    height: int
    dpi: int


@dataclass(frozen=True)
class _RenderedPage:
    index: int
    image: Image.Image  # what the model sees (longest side <= MAX_SIDE_PX)
    width: int
    height: int
    dpi: int


class OcrEngine:
    def __init__(self, model_id: str, *, generate_fn: GenerateFn | None = None) -> None:
        self.model_id = model_id
        self._generate_fn = generate_fn
        self._model: Any = None
        self._processor: Any = None
        self._load_lock = threading.Lock()

    # ----- public -------------------------------------------------------------------------------

    def ocr(self, data: bytes, mime: str, pages: list[int] | None) -> list[OcrPage]:
        rendered = render_pages(data, mime, pages)
        out: list[OcrPage] = []
        for page in rendered:
            raw = self._generate(OCR_PROMPT, page.image)
            out.append(
                OcrPage(
                    index=page.index,
                    markdown=clean_output(raw, self._special_tokens()),
                    width=page.width,
                    height=page.height,
                    dpi=page.dpi,
                )
            )
        return out

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ----- model --------------------------------------------------------------------------------

    def _generate(self, prompt: str, image: Image.Image) -> str:
        if self._generate_fn is not None:
            return self._generate_fn(prompt, image)
        return self._mlx_generate(prompt, image)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is None:
                from mlx_vlm import load  # optional extra `ocr`

                self._model, self._processor = load(self.model_id)

    def _mlx_generate(self, prompt: str, image: Image.Image) -> str:
        self._ensure_loaded()
        from mlx_vlm import generate
        from mlx_vlm.prompt_utils import apply_chat_template

        config = getattr(self._model, "config", None)
        formatted = apply_chat_template(self._processor, config, prompt, num_images=1)
        prompt_str = formatted if isinstance(formatted, str) else str(formatted)
        # mlx-vlm's `image` is typed as path(s); hand it a temp PNG (the proven path).
        fd, path = tempfile.mkstemp(suffix=".png", prefix="mlx-ocr-")
        try:
            with os.fdopen(fd, "wb") as fh:
                image.save(fh, format="PNG")
            result = generate(
                self._model,
                self._processor,
                prompt_str,
                image=[path],
                verbose=False,
                max_tokens=MAX_TOKENS,
                temperature=0.0,
            )
        finally:
            os.unlink(path)
        text = getattr(result, "text", result)
        return text if isinstance(text, str) else str(text)

    def _special_tokens(self) -> list[str]:
        tok = getattr(self._processor, "tokenizer", self._processor)
        tokens = getattr(tok, "all_special_tokens", None) if tok is not None else None
        return [t for t in tokens or [] if isinstance(t, str) and t]


# ----- rendering ---------------------------------------------------------------------------------


def _normalize_mime(mime: str) -> str:
    return mime.split(";", 1)[0].strip().lower()


def select_pages(pages: list[int] | None, page_count: int) -> list[int]:
    """Validate requested 0-based indices; default all. Returns unique indices ascending."""
    if pages is None:
        return list(range(page_count))
    if not pages:
        raise InvalidRequest("pages must be non-empty when given", code="invalid_pages")
    for p in pages:
        if isinstance(p, bool) or not isinstance(p, int) or p < 0 or p >= page_count:
            raise InvalidRequest(
                f"page index {p!r} out of range (document has {page_count} page(s))",
                code="page_out_of_range",
            )
    return sorted(set(pages))


def render_pages(data: bytes, mime: str, pages: list[int] | None) -> list[_RenderedPage]:
    if not data:
        raise InvalidRequest("document is empty", code="empty_document")
    kind = _normalize_mime(mime)
    if kind == PDF_MIME:
        return _render_pdf(data, pages)
    if kind in IMAGE_MIMES:
        return _render_image(data, pages)
    raise InvalidRequest(f"unsupported document type {mime!r}", code="unsupported_mime")


def _render_pdf(data: bytes, pages: list[int] | None) -> list[_RenderedPage]:
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as exc:
        raise InvalidRequest(f"corrupt or unreadable PDF: {exc}", code="corrupt_document") from exc
    try:
        selected = select_pages(pages, len(pdf))
        out: list[_RenderedPage] = []
        for idx in selected:
            page = pdf[idx]
            try:
                w_pt, h_pt = page.get_size()
                longest_pt = max(w_pt, h_pt, 1.0)
                scale = min(RENDER_DPI / 72.0, MAX_SIDE_PX / longest_pt)
                image = page.render(scale=scale).to_pil().convert("RGB")
            except pdfium.PdfiumError as exc:
                raise InvalidRequest(
                    f"cannot render page {idx}: {exc}", code="corrupt_document"
                ) from exc
            finally:
                page.close()
            out.append(
                _RenderedPage(
                    index=idx,
                    image=image,
                    width=image.width,
                    height=image.height,
                    dpi=round(scale * 72),
                )
            )
        return out
    finally:
        pdf.close()


def _render_image(data: bytes, pages: list[int] | None) -> list[_RenderedPage]:
    select_pages(pages, 1)
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()  # first frame only: an image document is one page
            dpi = _image_dpi(img.info.get("dpi"))
            width, height = img.size
            model_img = img.convert("RGB")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
        raise InvalidRequest(
            f"corrupt or unreadable image: {exc}", code="corrupt_document"
        ) from exc
    if max(width, height) > MAX_SIDE_PX:
        model_img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX), Image.Resampling.LANCZOS)
    return [_RenderedPage(index=0, image=model_img, width=width, height=height, dpi=dpi)]


def _image_dpi(raw: object) -> int:
    """PIL reports dpi as an (x, y) tuple of floats/rationals; absent or bogus → 72."""
    if isinstance(raw, tuple) and raw:
        try:
            value = round(float(raw[0]))
        except TypeError, ValueError:
            return DEFAULT_IMAGE_DPI
        if 1 <= value <= 10_000:
            return value
    return DEFAULT_IMAGE_DPI


# ----- output ------------------------------------------------------------------------------------


def clean_output(raw: str, special_tokens: list[str] | None = None) -> str:
    text = raw
    for tok in special_tokens or []:
        text = text.replace(tok, "")
    text = _SPECIAL_TOKEN_RE.sub("", text)
    lines = [line.rstrip() for line in text.strip().splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
