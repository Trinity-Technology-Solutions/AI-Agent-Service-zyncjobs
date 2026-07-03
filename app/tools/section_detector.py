import re
from .base_tool import BaseTool


HEADING_PATTERNS = [
    (r"^(summary|professional summary|career summary|profile summary|executive summary)\s*$", "summary"),
    (r"^(professional objective|career objective|objective|career goal)\s*$", "summary"),
    (r"^(about me|about|profile|professional profile)\s*$", "summary"),
    (r"^(experience|work experience|professional experience|employment|employment history|work history)\s*$", "experience"),
    (r"^(career history|professional background|relevant experience|work background)\s*$", "experience"),
    (r"^(internship|internships|internship experience|industrial training|industrial experience)\s*$", "experience"),
    (r"^(education|academic qualification|qualification|educational qualification|academic background)\s*$", "education"),
    (r"^(educational background|academic history|education background|qualifications|degrees)\s*$", "education"),
    (r"^(skills|technical skills|core competencies|competencies|expertise|key skills|technologies)\s*$", "skills"),
    (r"^(technical expertise|proficiencies|skill set|tools and technologies|computer skills)\s*$", "skills"),
    (r"^(technical summary|technical proficiency|professional skills|it skills)\s*$", "skills"),
    (r"^(projects|project experience|academic projects|personal projects|key projects|major projects)\s*$", "projects"),
    (r"^(project work|project details|technical projects|software projects|project portfolio)\s*$", "projects"),
    (r"^(certifications|certificates|licenses|professional certifications|certification|credentials)\s*$", "certifications"),
    (r"^(languages|language proficiency|language skills)\s*$", "languages"),
    (r"^(publications|research|research papers|papers|research work)\s*$", "publications"),
    (r"^(honors|honours|awards|achievements|awards and recognition|accomplishments)\s*$", "honors"),
    (r"^(personal details|personal information|personal info|contact|contact information)\s*$", "contact"),
    (r"^(declaration|declarations)\s*$", "declaration"),
    (r"^(references|professional references)\s*$", "references"),
    (r"^(training|trainings|workshops|workshop|courses|coursework)\s*$", "training"),
    (r"^(volunteer|volunteering|volunteer experience|volunteer work|social work)\s*$", "volunteer"),
    (r"^(interests|hobbies|activities|extracurricular|co-curricular)\s*$", "interests"),
    (r"^(leadership|leadership experience|positions of responsibility|positions)\s*$", "leadership"),
]

# Lines shorter than this that are ALL CAPS are checked as potential headings
MAX_HEADING_LINE_LENGTH = 50


class SectionDetectorTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="section_detector",
            description="Detects and maps resume section headings to standard names",
        )

    def run(self, text: str) -> dict:
        lines = text.split("\n")
        sections = {}
        current_section = "header"
        current_lines = []

        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                if current_lines:
                    sections.setdefault(current_section, []).extend(current_lines)
                    current_lines = []
                continue
            matched_section = self._detect_section(line_stripped)
            if matched_section:
                if current_lines:
                    sections.setdefault(current_section, []).extend(current_lines)
                    current_lines = []
                current_section = matched_section
            else:
                if current_section == "header" and current_lines:
                    sections.setdefault("header", []).extend(current_lines)
                    current_lines = []
                current_lines.append(line_stripped)

        if current_lines:
            sections.setdefault(current_section, []).extend(current_lines)

        result = {}
        for section_name, section_lines in sections.items():
            result[section_name] = "\n".join(section_lines)

        return result

    def _detect_section(self, line: str) -> str | None:
        stripped = line.strip()
        if len(stripped) > MAX_HEADING_LINE_LENGTH:
            return None
        lower = stripped.lower().strip()
        for pattern, section_name in HEADING_PATTERNS:
            if re.match(pattern, lower):
                return section_name
        is_all_caps = stripped == stripped.upper() and len(stripped) > 2
        is_title_case = stripped.istitle() and len(stripped) > 2
        if is_all_caps or is_title_case:
            for pattern, section_name in HEADING_PATTERNS:
                pat_clean = pattern.replace(r"\s*$", "").replace(r"^\s*", "").replace(r"\s*\(.*?\)\s*", "")
                base_text = re.sub(r"[^a-zA-Z\s]", "", lower).strip()
                pat_lower = re.sub(r"[^a-zA-Z\s]", "", pat_clean).strip()
                if base_text == pat_lower or base_text.startswith(pat_lower) or base_text.endswith(pat_lower):
                    return section_name
        return None
