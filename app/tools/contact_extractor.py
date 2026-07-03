import re
from .base_tool import BaseTool


class ContactExtractorTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="contact_extractor",
            description="Extracts contact information (email, phone, LinkedIn, GitHub, portfolio) using regex",
        )

    def run(self, text: str) -> dict:
        return {
            "email": self._extract_email(text),
            "phone": self._extract_phone(text),
            "linkedin": self._extract_linkedin(text),
            "github": self._extract_github(text),
            "portfolio": self._extract_urls(text),
        }

    def _extract_email(self, text: str) -> str:
        emails = re.findall(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text)
        seen = set()
        unique = []
        for e in emails:
            e_lower = e.lower()
            if e_lower not in seen:
                seen.add(e_lower)
                unique.append(e)
        return unique[0] if unique else ""

    def _extract_phone(self, text: str) -> str:
        patterns = [
            r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4}",
            r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        ]
        for pattern in patterns:
            phones = re.findall(pattern, text)
            if phones:
                return phones[0].strip()
        return ""

    def _extract_linkedin(self, text: str) -> str:
        patterns = [
            r"linkedin\.com/in/[a-zA-Z0-9_-]+[/]?",
            r"linkedin\.com/pub/[a-zA-Z0-9_-]+",
        ]
        for pattern in patterns:
            urls = re.findall(pattern, text, re.IGNORECASE)
            if urls:
                url = urls[0]
                if not url.startswith("http"):
                    url = "https://" + url
                return url
        return ""

    def _extract_github(self, text: str) -> str:
        patterns = [
            r"github\.com/[a-zA-Z0-9_-]+[/]?",
        ]
        for pattern in patterns:
            urls = re.findall(pattern, text, re.IGNORECASE)
            if urls:
                url = urls[0]
                if not url.startswith("http"):
                    url = "https://" + url
                return url
        return ""

    def _extract_urls(self, text: str) -> str:
        urls = re.findall(
            r"(?:https?://)?(?:www\.)?[a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:/[a-zA-Z0-9&%_./~\-+=#?]*)?",
            text,
        )
        exclude = {"linkedin.com", "github.com"}
        for url in urls:
            domain_match = re.search(r"(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z]{2,})", url)
            if domain_match:
                domain = domain_match.group(1).lower()
                if not any(ex in domain for ex in exclude):
                    if not url.startswith("http"):
                        url = "https://" + url
                    return url
        return ""
