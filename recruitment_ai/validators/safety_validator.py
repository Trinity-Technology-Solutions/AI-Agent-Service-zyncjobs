"""Safety validator — PII detection, profanity filter, prompt injection guard."""
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
AADHAAR_PATTERN = re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b")
PAN_PATTERN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)",
    r"forget\s+(all\s+)?(previous|above|prior)",
    r"you\s+are\s+(now|not)",
    r"system\s+prompt",
    r"disregard",
    r"bypass",
    r"role\s*:\s*(system|assistant)",
]

SENSITIVE_KEYWORDS = [
    r"\bpassword\b",
    r"\bsecret\b",
    r"\bapi[_-]?key\b",
    r"\bauth[_-]?token\b",
    r"\baccess[_-]?token\b",
    r"\bsecret[_-]?key\b",
]

PII_LABELS = {
    "email": EMAIL_PATTERN,
    "phone": PHONE_PATTERN,
    "aadhaar": AADHAAR_PATTERN,
    "pan": PAN_PATTERN,
    "ip": IP_PATTERN,
}


class SafetyReport:
    def __init__(self):
        self.flagged: bool = False
        self.issues: list[dict] = []
        self.warnings: list[str] = []

    def add_issue(self, category: str, detail: str, severity: str = "warning") -> None:
        self.flagged = True
        self.issues.append({"category": category, "detail": detail, "severity": severity})

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> dict:
        return {"flagged": self.flagged, "issues": self.issues, "warnings": self.warnings}


def check_pii(text: str) -> list[dict]:
    found = []
    for label, pattern in PII_LABELS.items():
        matches = pattern.findall(text)
        for m in matches:
            found.append({"type": label, "value": m, "severity": "high" if label in ("aadhaar", "pan") else "medium"})
    return found


def check_prompt_injection(text: str) -> list[str]:
    flagged = []
    for p in PROMPT_INJECTION_PATTERNS:
        if re.search(p, text, re.IGNORECASE):
            flagged.append(f"Prompt injection pattern detected: {p}")
    return flagged


def check_sensitive_keywords(text: str) -> list[str]:
    flagged = []
    for p in SENSITIVE_KEYWORDS:
        if re.search(p, text, re.IGNORECASE):
            flagged.append(f"Sensitive keyword detected: {p}")
    return flagged


def check_profanity(text: str) -> list[str]:
    flagged = []
    profanity_patterns = getattr(check_profanity, "_patterns", None)
    if profanity_patterns is None:
        check_profanity._patterns = []
    for p in check_profanity._patterns:
        if re.search(p, text, re.IGNORECASE):
            flagged.append(f"Profanity detected: {p}")
    return flagged


def validate_safety(text: str) -> SafetyReport:
    report = SafetyReport()
    pii = check_pii(text)
    for item in pii:
        report.add_issue("pii", f"{item['type']} detected", item["severity"])
    for w in check_prompt_injection(text):
        report.add_issue("prompt_injection", w, "high")
    for w in check_sensitive_keywords(text):
        report.add_issue("sensitive_keyword", w, "medium")
    for w in check_profanity(text):
        report.add_issue("profanity", w, "low")
    return report


def redact_pii(text: str) -> str:
    for label, pattern in PII_LABELS.items():
        text = pattern.sub(f"[REDACTED_{label.upper()}]", text)
    return text
