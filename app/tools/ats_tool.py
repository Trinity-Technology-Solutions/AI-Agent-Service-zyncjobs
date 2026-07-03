import re
from .base_tool import BaseTool


_STOP_WORDS = {
    "the", "a", "an", "and", "or", "in", "of", "to", "for",
    "with", "on", "at", "by", "is", "are", "be", "will", "we",
    "our", "your", "this", "that", "has", "have", "been", "was",
    "were", "but", "not", "all", "can", "may", "also", "very",
    "just", "its", "than", "then", "each", "any", "per", "etc",
}


class ATSTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="ats_tool",
            description="Compares resume against job description to calculate ATS score and find matching/missing keywords",
        )

    @property
    def result_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 0, "maximum": 100},
                "matching_keywords": {"type": "array", "items": {"type": "string"}},
                "missing_keywords": {"type": "array", "items": {"type": "string"}},
            },
        }

    def run(self, resume_text: str, job_description: str) -> dict:
        resume_lower = resume_text.lower()
        jd_lower = job_description.lower()

        resume_keywords = self._extract_keywords(resume_lower)
        jd_keywords = self._extract_keywords(jd_lower)

        matching = sorted(jd_keywords & resume_keywords)
        missing = sorted(jd_keywords - resume_keywords)
        score = int((len(matching) / max(len(jd_keywords), 1)) * 100) if jd_keywords else 0

        return {
            "score": min(score, 100),
            "matching_keywords": matching[:25],
            "missing_keywords": missing[:25],
        }

    def _extract_keywords(self, text: str) -> set:
        words = re.findall(r'\b[a-z]{3,}\b', text)
        return set(w for w in words if w not in _STOP_WORDS)
