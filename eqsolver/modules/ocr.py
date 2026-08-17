"""
Fallback text extraction for when the user's LM Studio model isn't
multimodal. Uses pytesseract (requires the Tesseract binary installed
on the host OS -- see README).
"""
from PIL import Image
import io


def ocr_extract(image_bytes: bytes) -> str:
    try:
        import pytesseract
    except ImportError as e:
        raise RuntimeError(
            "pytesseract not installed. Run `pip install pytesseract` and "
            "install the Tesseract binary (see README), or load a vision "
            "model in LM Studio instead."
        ) from e

    img = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(img)
    return text.strip()
