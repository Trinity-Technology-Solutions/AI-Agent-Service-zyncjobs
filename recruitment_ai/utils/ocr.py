"""OCR utility — extracts text from PDF, DOCX, and image files.
Architecture doc: Resume Parser + Job Parser → OCR support.
Uses pdfplumber (primary), PyPDF2 (fallback), pytesseract (image OCR), python-docx (DOCX).
"""
import logging
import io
import base64

logger = logging.getLogger(__name__)


def extract_text(file_content: str, file_type: str) -> str:
    """Extract plain text from base64-encoded file content.

    Args:
        file_content: base64-encoded file bytes OR plain text
        file_type: 'pdf' | 'docx' | 'doc' | 'image' | 'png' | 'jpg' | 'jpeg' | 'txt'

    Returns:
        Extracted plain text string.
    """
    file_type = (file_type or "txt").lower().strip(".")

    if file_type == "txt":
        return file_content

    try:
        raw_bytes = base64.b64decode(file_content)
    except Exception:
        return file_content

    if file_type == "pdf":
        return _extract_pdf(raw_bytes)
    elif file_type in ("docx", "doc"):
        return _extract_docx(raw_bytes)
    elif file_type in ("png", "jpg", "jpeg", "image"):
        return _extract_image(raw_bytes)

    try:
        return raw_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _extract_pdf(data: bytes) -> str:
    # Primary: pdfplumber
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages).strip()
            if text:
                return text
    except Exception as e:
        logger.warning("pdfplumber failed: %s", e)

    # Fallback: PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        text = "\n".join(p.extract_text() or "" for p in reader.pages).strip()
        if text:
            return text
    except Exception as e:
        logger.warning("PyPDF2 failed: %s", e)

    # Last resort: OCR for scanned PDFs
    return _pdf_ocr(data)


def _pdf_ocr(data: bytes) -> str:
    try:
        from pdf2image import convert_from_bytes
        images = convert_from_bytes(data)
        return "\n".join(_ocr_image(img) for img in images).strip()
    except Exception as e:
        logger.warning("PDF OCR failed: %s", e)
    return ""


def _extract_docx(data: bytes) -> str:
    try:
        from docx import Document
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip()).strip()
    except Exception as e:
        logger.warning("DOCX extraction failed: %s", e)
    return ""


def _extract_image(data: bytes) -> str:
    try:
        from PIL import Image
        img = Image.open(io.BytesIO(data))
        return _ocr_image(img)
    except Exception as e:
        logger.warning("Image extraction failed: %s", e)
    return ""


def _ocr_image(img) -> str:
    try:
        import pytesseract
        return pytesseract.image_to_string(img).strip()
    except Exception as e:
        logger.warning("pytesseract OCR failed: %s", e)
    return ""
