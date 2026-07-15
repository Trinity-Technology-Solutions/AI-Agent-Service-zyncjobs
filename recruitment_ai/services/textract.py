"""Amazon Textract service — OCR for scanned PDFs and images.
Architecture doc: AI Service → Textract → Extracted Text → Brain.
Falls back to pdf-parse and pytesseract when Textract is unavailable.

Usage:
    from recruitment_ai.services.textract import textract_service

    # Extract text from S3 file
    text = await textract_service.extract_from_s3("s3://bucket/key.pdf")

    # Extract text from raw bytes
    text = await textract_service.extract_bytes(pdf_bytes, filename="resume.pdf")
"""
import logging
from typing import Optional
from recruitment_ai.config.settings import settings

logger = logging.getLogger(__name__)


class TextractService:
    """AWS Textract wrapper with fallback to local PDF/image extraction."""

    def __init__(self):
        self._client = None

    @property
    def enabled(self) -> bool:
        return settings.TEXTRACT_ENABLED

    async def _get_client(self):
        if self._client is None:
            try:
                import aioboto3
                session = aioboto3.Session()
                self._client = await session.client("textract", region_name=settings.S3_REGION).__aenter__()
                logger.info("Textract client initialized")
            except ImportError:
                logger.warning("aioboto3 not installed — Textract disabled")
                return None
            except Exception as e:
                logger.warning("Textract init failed: %s", e)
                return None
        return self._client

    async def extract_from_s3(self, s3_key: str) -> Optional[str]:
        """Extract text from a document stored in S3 using Textract."""
        if not self.enabled:
            return None
        client = await self._get_client()
        if not client:
            return None
        try:
            resp = await client.detect_document_text(
                Document={"S3Object": {"Bucket": settings.S3_BUCKET, "Name": s3_key}}
            )
            blocks = resp.get("Blocks", [])
            lines = [b["Text"] for b in blocks if b.get("BlockType") == "LINE"]
            text = "\n".join(lines)
            logger.info("Textract extracted %d lines from S3 object %s", len(lines), s3_key)
            return text
        except Exception as e:
            logger.warning("Textract S3 extraction failed: %s", e)
            return None

    async def extract_bytes(self, data: bytes, filename: str = "document.pdf") -> Optional[str]:
        """Extract text from raw bytes. Tries Textract first, falls back to local."""
        if self.enabled and len(data) < 10 * 1024 * 1024:
            text = await self._extract_textract_bytes(data)
            if text:
                return text

        return await self._extract_local(data, filename)

    async def _extract_textract_bytes(self, data: bytes) -> Optional[str]:
        client = await self._get_client()
        if not client:
            return None
        try:
            resp = await client.detect_document_text(Document={"Bytes": data})
            blocks = resp.get("Blocks", [])
            lines = [b["Text"] for b in blocks if b.get("BlockType") == "LINE"]
            return "\n".join(lines) if lines else None
        except Exception as e:
            logger.warning("Textract bytes extraction failed: %s", e)
            return None

    async def _extract_local(self, data: bytes, filename: str) -> Optional[str]:
        """Local fallback — pdf-parse for PDFs, pytesseract for images."""
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

        if ext == "pdf":
            try:
                import pdfplumber
                import io
                text_parts = []
                with pdfplumber.open(io.BytesIO(data)) as pdf:
                    for page in pdf.pages:
                        t = page.extract_text()
                        if t:
                            text_parts.append(t)
                result = "\n".join(text_parts)
                if result.strip():
                    logger.info("pdfplumber extracted %d chars", len(result))
                    return result
            except ImportError:
                try:
                    import PyPDF2
                    import io
                    reader = PyPDF2.PdfReader(io.BytesIO(data))
                    result = "\n".join(p.extract_text() or "" for p in reader.pages)
                    if result.strip():
                        return result
                except ImportError:
                    logger.warning("No PDF library available (pdfplumber or PyPDF2)")

        if ext in ("png", "jpg", "jpeg", "tiff", "bmp"):
            try:
                import pytesseract
                from PIL import Image
                import io
                image = Image.open(io.BytesIO(data))
                text = pytesseract.image_to_string(image)
                if text.strip():
                    logger.info("Tesseract extracted %d chars", len(text))
                    return text
            except ImportError:
                logger.warning("pytesseract not available for image OCR")

        return None


textract_service = TextractService()
