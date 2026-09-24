"""
modules/ocr.py -- previously 0% covered. Tesseract itself is not guaranteed
to be on every dev machine (it's an OS binary, not a pip package -- see the
module's own docstring), so the "real OCR" tests are skipped rather than
failed when it's absent; the ImportError-wrapping path is tested either way
since it doesn't need the binary.
"""
import io
import shutil

import pytest
from PIL import Image, ImageDraw

from modules.ocr import ocr_extract

TESSERACT_AVAILABLE = shutil.which("tesseract") is not None


def _text_image_bytes(text: str) -> bytes:
    # Large canvas + PIL's built-in bitmap font scaled up: the default font at
    # normal size is too thin/small for tesseract to read reliably, which made
    # this flaky. A big, blocky rendering is legible even without a real
    # TrueType font installed on the host.
    img = Image.new("RGB", (max(400, 40 * len(text)), 160), "white")
    draw = ImageDraw.Draw(img)
    try:
        from PIL import ImageFont
        font = ImageFont.load_default(size=64)
    except TypeError:
        font = ImageFont.load_default()
    draw.text((15, 40), text, fill="black", font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_missing_pytesseract_raises_runtime_error_with_install_hint(monkeypatch):
    """The dependency isn't installed -> a clear RuntimeError, not an ImportError
    leaking out of a module the caller may not know exists."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pytesseract":
            raise ImportError("no module named pytesseract")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="pytesseract not installed"):
        ocr_extract(_text_image_bytes("hello"))


@pytest.mark.skipif(not TESSERACT_AVAILABLE, reason="tesseract binary not installed on this host")
class TestRealOCR:
    def test_extracts_text_from_a_rendered_image(self):
        result = ocr_extract(_text_image_bytes("HELLO WORLD"))
        assert "HELLO" in result.upper()

    def test_result_is_stripped(self):
        """The function's own contract (`return text.strip()`) -- a blank/near-blank
        image shouldn't come back padded with tesseract's leading/trailing whitespace."""
        blank = Image.new("RGB", (50, 50), "white")
        buf = io.BytesIO()
        blank.save(buf, format="PNG")
        result = ocr_extract(buf.getvalue())
        assert result == result.strip()

    def test_numeric_content_is_recognizable(self):
        """A physics-problem-shaped snippet: OCR is used here as a fallback text
        extractor ahead of the LLM, so it just needs to get the characters roughly
        right, not perfectly -- this checks the digits survive, which is what the
        rest of the pipeline actually depends on."""
        result = ocr_extract(_text_image_bytes("v = 9.8 m/s"))
        assert "9" in result and "8" in result
