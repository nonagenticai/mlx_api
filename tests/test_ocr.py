import io
import os
import time

import pytest
from PIL import Image, ImageDraw, ImageFont

from mlx_api.errors import InvalidRequest
from mlx_api.ocr import MAX_SIDE_PX, MAX_TOKENS, OCR_PROMPT, OcrEngine, OcrPage, clean_output

MODEL_ID = os.environ.get("MLX_API_OCR_MODEL", "mlx-community/PaddleOCR-VL-1.5-bf16")


def _pdf(pages: list[tuple[int, int]], resolution: float = 72.0) -> bytes:
    """Image-only PDF (no text layer). Each page gets a distinct grey level so tests can tell
    which page the engine rendered."""
    imgs = [Image.new("RGB", size, (10 * i, 10 * i, 10 * i)) for i, size in enumerate(pages)]
    buf = io.BytesIO()
    imgs[0].save(buf, format="PDF", resolution=resolution, save_all=True, append_images=imgs[1:])
    return buf.getvalue()


def _image(fmt: str, size: tuple[int, int] = (320, 200), **save_kw: object) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, format=fmt, **save_kw)
    return buf.getvalue()


class _Recorder:
    def __init__(self, reply: str = "text") -> None:
        self.calls: list[tuple[str, tuple[int, int], int]] = []
        self.reply = reply

    def __call__(self, prompt: str, image: Image.Image) -> str:
        self.calls.append((prompt, image.size, image.getpixel((0, 0))[0]))  # ty: ignore[not-subscriptable]
        return self.reply


def _engine(reply: str = "text") -> tuple[OcrEngine, _Recorder]:
    rec = _Recorder(reply)
    return OcrEngine("stub", generate_fn=rec), rec


# ----- page selection -----------------------------------------------------------------------------


def test_pdf_all_pages_by_default_ascending():
    engine, rec = _engine()
    out = engine.ocr(_pdf([(200, 100)] * 3), "application/pdf", None)
    assert [p.index for p in out] == [0, 1, 2]
    assert all(c[0] == OCR_PROMPT for c in rec.calls)
    assert [c[2] for c in rec.calls] == [0, 10, 20]  # page order == document order


def test_pdf_requested_pages_sorted_and_deduped():
    engine, rec = _engine()
    out = engine.ocr(_pdf([(200, 100)] * 4), "application/pdf", [3, 1, 3])
    assert [p.index for p in out] == [1, 3]
    assert [c[2] for c in rec.calls] == [10, 30]  # the right pages were rendered


@pytest.mark.parametrize("pages", [[3], [-1], [0, 7], []])
def test_pdf_bad_pages_rejected_before_any_generate(pages):
    engine, rec = _engine()
    with pytest.raises(InvalidRequest):
        engine.ocr(_pdf([(200, 100)] * 3), "application/pdf", pages)
    assert rec.calls == []


# ----- rendering dimensions -----------------------------------------------------------------------


def test_pdf_rendered_at_200_dpi():
    engine, rec = _engine()
    # 360 x 216 pt page = 5 x 3 in -> 1000 x 600 px at 200 dpi
    [page] = engine.ocr(_pdf([(360, 216)]), "application/pdf", None)
    assert (page.width, page.height, page.dpi) == (1000, 600, 200)
    assert rec.calls[0][1] == (1000, 600)  # the model sees exactly the reported size


def test_pdf_longest_side_capped():
    engine, rec = _engine()
    # US letter 612 x 792 pt -> 1700 x 2200 at 200 dpi, capped to longest side 2000
    [page] = engine.ocr(_pdf([(850, 1100)], resolution=100), "application/pdf", None)
    assert page.height == MAX_SIDE_PX
    assert page.width == pytest.approx(1545, abs=2)
    assert page.dpi == 182  # effective dpi actually rendered
    assert rec.calls[0][1] == (page.width, page.height)


def test_image_one_page_default_dpi():
    engine, rec = _engine()
    out = engine.ocr(_image("PNG"), "image/png", None)
    assert out == [OcrPage(index=0, markdown="text", width=320, height=200, dpi=72)]
    assert len(rec.calls) == 1


@pytest.mark.parametrize(
    ("fmt", "mime"),
    [("JPEG", "image/jpeg"), ("WEBP", "image/webp"), ("TIFF", "image/tiff"), ("PNG", "image/png")],
)
def test_image_formats_and_stated_dpi(fmt, mime):
    engine, _ = _engine()
    kw = {} if fmt == "WEBP" else {"dpi": (300, 300)}  # webp carries no dpi
    [page] = engine.ocr(_image(fmt, **kw), mime, [0])
    assert (page.index, page.width, page.height) == (0, 320, 200)
    assert page.dpi == (72 if fmt == "WEBP" else 300)


def test_large_image_reports_original_size_but_model_sees_capped():
    engine, rec = _engine()
    [page] = engine.ocr(_image("PNG", size=(4000, 1000)), "image/png", None)
    assert (page.width, page.height) == (4000, 1000)
    assert rec.calls[0][1] == (2000, 500)


def test_image_page_out_of_range():
    engine, _ = _engine()
    with pytest.raises(InvalidRequest):
        engine.ocr(_image("PNG"), "image/png", [1])


# ----- error cases --------------------------------------------------------------------------------


@pytest.mark.parametrize("mime", ["text/plain", "application/msword", "image/gif", ""])
def test_unsupported_mime(mime):
    engine, _ = _engine()
    with pytest.raises(InvalidRequest):
        engine.ocr(_image("PNG"), mime, None)


def test_mime_parameters_and_case_tolerated():
    engine, _ = _engine()
    assert len(engine.ocr(_pdf([(100, 100)]), "Application/PDF; charset=binary", None)) == 1


@pytest.mark.parametrize(
    ("data", "mime"),
    [
        (b"%PDF-1.7 garbage", "application/pdf"),
        (b"not an image at all", "image/png"),
        (b"", "application/pdf"),
        (_image("PNG")[:40], "image/png"),  # truncated
    ],
)
def test_corrupt_documents(data, mime):
    engine, rec = _engine()
    with pytest.raises(InvalidRequest):
        engine.ocr(data, mime, None)
    assert rec.calls == []


def test_model_not_loaded_without_ocr_call():
    assert OcrEngine(MODEL_ID).is_loaded is False


# ----- output cleaning ----------------------------------------------------------------------------


def test_special_tokens_stripped():
    raw = "<|LOC_10|>Invoice 4711<|LOC_20|>\n\n\n\ntotal</s><|end_of_sentence|>  "
    assert clean_output(raw) == "Invoice 4711\n\ntotal"
    assert clean_output("a[EOS]b", ["[EOS]"]) == "ab"


def test_engine_output_is_cleaned():
    engine, _ = _engine(reply="Hello<|end_of_sentence|>")
    [page] = engine.ocr(_image("PNG"), "image/png", None)
    assert page.markdown == "Hello"


def test_max_tokens_bounded():
    assert 0 < MAX_TOKENS <= 4096


# ----- real model ---------------------------------------------------------------------------------


def _scanned_invoice_pdf() -> bytes:
    img = Image.new("RGB", (1700, 2200), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 96)
    except OSError:
        font = ImageFont.load_default(size=96)
    draw.text((150, 300), "Invoice 4711 total 99 EUR", fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PDF", resolution=200)
    return buf.getvalue()


@pytest.mark.models
def test_real_model_reads_scanned_pdf(request):
    if "models" not in (request.config.getoption("markexpr") or ""):
        pytest.skip("loads real weights; run with -m models")
    import pypdfium2 as pdfium

    data = _scanned_invoice_pdf()
    pdf = pdfium.PdfDocument(data)
    assert pdf[0].get_textpage().get_text_range().strip() == ""  # genuinely scanned
    pdf.close()

    engine = OcrEngine(MODEL_ID)
    engine.ocr(data, "application/pdf", [0])  # warm-up: load model + first compile
    t0 = time.perf_counter()
    [page] = engine.ocr(data, "application/pdf", [0])
    elapsed = time.perf_counter() - t0
    print(f"\nOCR seconds/page: {elapsed:.2f}\nOCR markdown: {page.markdown[:300]!r}")
    assert page.index == 0 and page.dpi > 0
    assert "4711" in page.markdown
    assert "99" in page.markdown
