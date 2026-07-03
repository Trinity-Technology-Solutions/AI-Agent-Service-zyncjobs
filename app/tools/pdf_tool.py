import os
import re
import tempfile
from .base_tool import BaseTool


class PDFExtractor:
    """
    Swappable PDF extraction backend.
    Tomorrow you can subclass this and swap PyMuPDF → OCR → Azure Document Intelligence
    without touching PDFTool or ResumeAgent.
    """

    def extract_blocks(self, path: str) -> list[dict]:
        """
        Returns a list of blocks, each:
        { "page": int, "x0": float, "y0": float, "x1": float, "y1": float, "text": str }
        """
        raise NotImplementedError

    def reconstruct_reading_order(self, blocks: list[dict], page_width: float) -> str:
        """
        Given blocks with bounding boxes, reconstruct correct reading order.
        For two-column layouts: sort left column top→bottom, then right column top→bottom.
        For single-column: sort all blocks top→bottom.
        """
        if not blocks:
            return ""

        # Group blocks by page
        pages: dict[int, list[dict]] = {}
        for b in blocks:
            pages.setdefault(b["page"], []).append(b)

        lines = []
        for page_num in sorted(pages):
            page_blocks = pages[page_num]

            # Detect column split: if blocks cluster into two x-ranges, it's two-column
            mid = page_width / 2
            left = [b for b in page_blocks if b["x0"] < mid - 20]
            right = [b for b in page_blocks if b["x0"] >= mid - 20]

            # Two-column: left column first (top→bottom), then right column (top→bottom)
            if left and right and len(right) > 2:
                ordered = sorted(left, key=lambda b: b["y0"]) + sorted(right, key=lambda b: b["y0"])
            else:
                ordered = sorted(page_blocks, key=lambda b: (b["y0"], b["x0"]))

            for b in ordered:
                text = b["text"].strip()
                if text:
                    lines.append(text)

        return "\n".join(lines)

    def normalize_text(self, text: str) -> str:
        # Collapse 3+ spaces to newline (common in two-column PDFs after extraction)
        text = re.sub(r" {3,}", "\n", text)
        # Collapse 3+ newlines to two
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


class FitzExtractor(PDFExtractor):
    """PyMuPDF-based extractor using block bounding boxes."""

    def extract_blocks(self, path: str) -> tuple[list[dict], float]:
        import fitz
        all_blocks = []
        page_width = 595.0  # A4 default fallback

        with fitz.open(path) as doc:
            for page_num, page in enumerate(doc):
                page_width = page.rect.width
                raw_blocks = page.get_text("blocks")
                # raw_blocks: (x0, y0, x1, y1, text, block_no, block_type)
                for b in raw_blocks:
                    if b[6] != 0:  # skip image blocks (type != 0)
                        continue
                    text = b[4].strip()
                    if not text:
                        continue
                    all_blocks.append({
                        "page": page_num,
                        "x0": b[0], "y0": b[1],
                        "x1": b[2], "y1": b[3],
                        "text": text,
                    })

        return all_blocks, page_width

    def extract(self, path: str) -> str:
        blocks, page_width = self.extract_blocks(path)
        raw = self.reconstruct_reading_order(blocks, page_width)
        return self.normalize_text(raw)


class PDFTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="pdf_tool",
            description="Extracts text from PDF files preserving reading order (PyMuPDF blocks, fallback: pdfminer)",
        )

    def run(self, file_path: str) -> str:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF not found: {file_path}")
        return self._extract(file_path)

    def run_from_bytes(self, data: bytes) -> str:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        try:
            return self._extract(tmp_path)
        finally:
            os.unlink(tmp_path)

    def _extract(self, path: str) -> str:
        try:
            return FitzExtractor().extract(path)
        except ImportError:
            pass

        try:
            from pdfminer.high_level import extract_text
            return extract_text(path)
        except ImportError:
            pass

        raise ImportError(
            "No PDF library found. Install one:\n"
            "  pip install PyMuPDF\n"
            "  or\n"
            "  pip install pdfminer.six"
        )
