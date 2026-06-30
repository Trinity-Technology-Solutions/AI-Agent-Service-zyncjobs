import re
from .base_tool import BaseTool


_SECTION_PATTERNS = {
    "contact": r"\b(contact|phone)\b[:\s]+(.*?)(?=\n\s*\n|\Z)",
    "summary": r"\b(summary|profile|objective|about me)\b[:\s]*(.*?)(?=\n\s*\n|\Z)",
    "experience": r"\b(experience|work history|employment|work experience)\b[:\s]*(.*?)(?=\n\s*(education|skills|projects|certifications)\b|\Z)",
    "education": r"\b(education|academic|qualification|degree)\b[:\s]*(.*?)(?=\n\s*(skills|projects|certifications|experience)\b|\Z)",
    "skills": r"\b(skills|technologies|competencies|expertise)\b[:\s]*(.*?)(?=\n\s*(projects|certifications|education|experience)\b|\Z)",
    "projects": r"\b(projects|portfolio)\b[:\s]*(.*?)(?=\n\s*(skills|education|certifications|experience)\b|\Z)",
    "certifications": r"\b(certifications|certificates|licenses)\b[:\s]*(.*?)(?=\n\s*\n|\Z)",
}


class ResumeParserTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="resume_parser",
            description="Parses resume text into structured sections (contact, skills, experience, education)",
        )

    def run(self, resume_text: str) -> dict:
        sections = {}
        for section, pattern in _SECTION_PATTERNS.items():
            match = re.search(pattern, resume_text, re.IGNORECASE | re.DOTALL)
            sections[section] = match.group(2).strip() if match else ""

        lines = [l for l in resume_text.split("\n") if l.strip()]
        if not sections.get("contact"):
            contact_lines = []
            for line in lines[:4]:
                if re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b|linkedin\.com|github\.com', line, re.IGNORECASE):
                    contact_lines.append(line.strip())
                elif not contact_lines and line == lines[0]:
                    contact_lines.append(line.strip())
            sections["contact"] = " | ".join(contact_lines) if contact_lines else (lines[0] if lines else "")

        return sections
