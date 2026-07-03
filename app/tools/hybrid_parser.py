import re
from .base_tool import BaseTool
from .text_cleaner import TextCleanerTool
from .section_detector import SectionDetectorTool
from .contact_extractor import ContactExtractorTool
from .section_parser import SectionParserTool
from .validator import ValidationTool
ENTERPRISE_SCHEMA = {
    "personal": {"name": "", "email": "", "phone": "", "linkedin": "", "github": ""},
    "summary": "",
    "experience": [],
    "education": [],
    "skills": {"programming": [], "frameworks": [], "databases": [], "cloud": [], "tools": []},
    "projects": [],
    "certifications": [],
    "languages": [],
    "publications": [],
    "honors": [],
}

HEADER_BLACKLIST = re.compile(
    r"^(resume|curriculum vitae|cv|profile|summary|carrer summary|"
    r"professional summary|career objective|objective|about me|"
    r"email|phone|linkedin|github|address|contact)\b",
    re.IGNORECASE,
)
ROLE_KEYWORDS = re.compile(
    r"\b(developer|engineer|designer|manager|analyst|intern|architect|"
    r"consultant|director|president|vice|lead|head|officer|coordinator|"
    r"specialist|associate|executive|founder|trainee|fresher|graduate|"
    r"student|software|full.?stack|front.?end|back.?end|data|devops|"
    r"cloud|mobile|web|senior|junior|mid|entry)\b",
    re.IGNORECASE,
)
LOCATION_WORDS = re.compile(
    r"\b(chennai|bangalore|bengaluru|mumbai|hyderabad|pune|delhi|kolkata|"
    r"ahmedabad|coimbatore|kochi|jaipur|india|remote|online|"
    r"noida|gurgaon|trivandrum|madurai|mysore|pondicherry|"
    r"nagpur|indore|bhopal|surat|lucknow|visakhapatnam)\b",
    re.IGNORECASE,
)


class HybridParserTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="hybrid_parser",
            description="Orchestrates the full hybrid resume parsing pipeline",
        )
        self.cleaner = TextCleanerTool()
        self.detector = SectionDetectorTool()
        self.contact_extractor = ContactExtractorTool()
        self.section_parser = SectionParserTool()
        self.validator = ValidationTool()

    def run(self, text: str) -> dict:
        cleaned = self.cleaner.run(text)
        sections = self.detector.run(cleaned)
        all_text = cleaned
        contact = self.contact_extractor.run(all_text)
        name = self._extract_name(sections.get("header", ""), cleaned, contact)

        parsed = dict(ENTERPRISE_SCHEMA)
        parsed["personal"] = {
            "name": name,
            "email": contact.get("email", ""),
            "phone": contact.get("phone", ""),
            "linkedin": contact.get("linkedin", ""),
            "github": contact.get("github", ""),
        }

        ai_sections = [
            "summary", "experience", "education", "projects",
            "skills", "languages", "certifications", "publications", "honors",
        ]
        for section_name in ai_sections:
            section_text = sections.get(section_name, "")
            if section_text.strip():
                try:
                    result = self.section_parser.run(section_name, section_text)
                    if result is not None:
                        parsed[section_name] = result
                except Exception as e:
                    print(f"[HybridParser] Error in section '{section_name}': {e}")

        if isinstance(parsed.get("summary"), dict):
            parsed["summary"] = parsed["summary"].get("summary", "")

        validated = self.validator.run(parsed)
        return validated

    def _extract_name(self, header_text: str, full_text: str, contact: dict) -> str:
        header_lines = [
            l.strip()
            for l in header_text.strip().split("\n")
            if l.strip()
        ]
        full_lines = [
            l.strip()
            for l in full_text.strip().split("\n")
            if l.strip()
        ]
        candidates = header_lines if header_lines else full_lines[:15]

        for line in candidates:
            if HEADER_BLACKLIST.match(line):
                continue
            if "@" in line or re.search(r"[+\d]{7,}", line):
                continue
            if "http" in line.lower() or "linkedin" in line.lower() or "github" in line.lower():
                continue
            if LOCATION_WORDS.search(line) and len(line.split()) <= 3:
                continue
            if ROLE_KEYWORDS.search(line) and len(line.split()) >= 3:
                continue
            words = line.split()
            if 2 <= len(words) <= 5:
                if all(w[0].isupper() and w.isalpha() for w in words if w[0].isalpha()):
                    return line
        for line in candidates:
            if HEADER_BLACKLIST.match(line):
                continue
            if "@" in line or re.search(r"[+\d]{7,}", line):
                continue
            words = line.split()
            if 2 <= len(words) <= 5:
                if all(w[0].isupper() or w[0].isdigit() for w in words if w[0].isalpha()):
                    return line
        email = contact.get("email", "")
        if email:
            local = email.split("@")[0]
            parts = [p for p in re.split(r"[._0-9]", local) if p]
            if parts:
                return " ".join(p.capitalize() for p in parts)
        return candidates[0] if candidates else ""
