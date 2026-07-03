import re
from .base_tool import BaseTool


class TextCleanerTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="text_cleaner",
            description="Cleans and normalizes extracted resume text",
        )

    def run(self, text: str) -> str:
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            line = self._clean_line(line)
            if line:
                cleaned.append(line)
        text = "\n".join(cleaned)
        text = self._normalize(text)
        return text.strip()

    def _clean_line(self, line: str) -> str:
        line = re.sub(r"[•●⬤▪▸➢✦✧★☆◆◇✪⚡☑✅✗✘❌⚠✔✖➤➜☛☞▹▻›»‹«]", "", line)
        line = re.sub(r"[\u2600-\u27BF\u2700-\u27BF\U0001F300-\U0001FAFF]", "", line)
        line = re.sub(r"[★☆✦✧✪✫✬✭✮✯⚡]", "", line)
        line = re.sub(r"[📧📞☎📱✉📍🌐🔗]", "", line)
        line = re.sub(r"[│┃|]\s*", "", line)
        line = re.sub(r"[‐‑–—]", "-", line)
        page_pattern = re.compile(r"\bPage\s+\d+\b|\bPage\s*\d+\s*of\s*\d+\b|\bPg\.?\s*\d+\b", re.IGNORECASE)
        line = page_pattern.sub("", line)
        line = re.sub(r"[-]+\s*Page\s+\d+\s*[-]+", "", line)
        line = line.strip()
        return line

    def _normalize(self, text: str) -> str:
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"^\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[:\s]+", "", text, flags=re.MULTILINE)
        text = re.sub(r"\s+$", "", text, flags=re.MULTILINE)
        return text.strip()
