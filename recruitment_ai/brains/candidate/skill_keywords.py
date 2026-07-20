"""Shared skill keyword constants used across all candidate brains.
Single source of truth — prevents drift between resume parser, ATS, and job matching.
"""
import re

SKILL_KEYWORDS = [
    "python", "java", "javascript", "typescript", "react", "node", "nodejs",
    "sql", "aws", "docker", "kubernetes", "git", "linux", "agile", "scrum",
    "rest", "api", "html", "css", "go", "rust", "c\\+\\+", "c#", "\\.net",
    "spring", "django", "flask", "fastapi", "postgresql", "mongodb", "redis",
    "swift", "kotlin", "ruby", "php", "vue", "angular", "svelte",
    "tensorflow", "pytorch", "machine learning", "deep learning",
    "graphql", "grpc", "kafka", "rabbitmq", "terraform", "ansible",
    "jenkins", "github actions", "ci/cd",
]


def extract_matched_skills(text: str) -> set[str]:
    """Extract which SKILL_KEYWORDS appear in text, case-insensitive."""
    lower = text.lower()
    matched = set()
    for kw in SKILL_KEYWORDS:
        if re.search(rf"\b{kw}\b", lower):
            matched.add(kw.replace("\\", ""))
    return matched
