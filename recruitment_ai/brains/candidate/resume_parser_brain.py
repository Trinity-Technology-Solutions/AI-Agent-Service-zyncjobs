"""Resume Parser Brain - parses resumes into structured data."""
import re
import json
from recruitment_ai.shared.brain import Brain, BrainState
from recruitment_ai.shared.ollama_service import ollama_service

RESUME_PARSER_SYSTEM = """You are a precise resume parser. Extract structured information from resumes.
Return ONLY a single valid JSON object. No markdown, no code blocks, no explanation, no extra text."""

RESUME_PARSER_PROMPT = """Parse the resume text below into a JSON object.

FIELD RULES:
- name: Full name only (2-4 words, Title Case). NOT a job title, NOT a section heading.
- email: email address string
- phone: phone number string. NEVER a year like "2023" or "2023-2024".
- location: city name only
- title: job designation e.g. "Software Engineer"
- summary: professional summary paragraph
- skills: array of strings — ALL programming languages, frameworks, libraries mentioned
- softSkills: array of strings — communication, leadership, teamwork, problem-solving etc.
- tools: array of strings — Git, Docker, Figma, Jira, Postman etc.
- workExperiences: array of objects. Each job/internship has:
  - jobTitle: role name string
  - company: company name string
  - date: date range string e.g. "Jan 2023 - Present"
  - descriptions: array of bullet point strings
- educations: array of objects. ONLY real academic degrees:
  - school: real institution name (university/college/school). NEVER a sentence or job description.
  - degree: real degree/course name. NEVER a bullet point or sentence.
  - date: graduation year or range
  - grade: GPA/percentage if mentioned, else empty string
- projects: array of {name, description}
- certifications: array of {name, provider, date}
- competitions: array of strings

EXAMPLE OUTPUT:
{"name":"John Smith","email":"john@example.com","phone":"+91 9876543210","location":"Chennai","title":"Software Engineer","summary":"Experienced developer with 3 years in web development.","skills":["Python","React","SQL"],"softSkills":["Communication","Teamwork"],"tools":["Git","Docker"],"workExperiences":[{"jobTitle":"Software Engineer","company":"ABC Corp","date":"Jan 2022 - Present","descriptions":["Built REST APIs","Improved performance by 30%"]}],"educations":[{"degree":"B.Tech Computer Science","school":"Anna University","date":"2022","grade":"8.5"}],"projects":[{"name":"Portfolio Website","description":"Built with React and Node.js"}],"certifications":[{"name":"AWS Cloud Practitioner","provider":"Amazon","date":"2023"}],"competitions":[]}

Resume text:
"""


class ResumeParserBrain(Brain):
    """Parses resumes into structured data using Ollama."""

    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainState:
        from recruitment_ai.utils.ocr import extract_text
        raw_content = state.file_content or state.query or ""
        content = extract_text(raw_content, state.file_type or "txt") if state.file_type else raw_content
        if not content.strip():
            state.error = "No resume content provided"
            return state

        try:
            raw = await ollama_service.generate(
                brain_name="resume_parser",
                prompt=f"{RESUME_PARSER_PROMPT}{content[:5000]}",
                system=RESUME_PARSER_SYSTEM,
                temperature=0.1,
                max_tokens=3000,
            )
            parsed = self._parse_json(raw)
            state.result = self._validate(parsed, content)
            state.metadata["parser"] = "llm"
        except Exception as e:
            state.result = self._fallback_parse(content)
            state.metadata["parser"] = "fallback"
            state.metadata["fallback_reason"] = str(e)

        return state

    def _parse_json(self, text: str) -> dict:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Strip markdown code fences if present
        text = re.sub(r"```(?:json)?", "", text).strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                # Try to fix truncated JSON by finding last complete field
                partial = match.group()
                try:
                    return json.loads(partial[:partial.rfind("}") + 1])
                except Exception:
                    pass
        return {}

    def _is_sentence(self, s: str) -> bool:
        """Returns True if the string looks like a sentence rather than a name/title."""
        if not s:
            return False
        return (
            len(s) > 100
            or bool(re.search(
                r"\b(developed|built|created|analyzed|implemented|designed|completed|"
                r"worked|gained|contributed|internship|july|august|september|october|"
                r"november|december|present|currently|involving|enabling|reducing|improving|"
                r"responsible|managed|led|handled|assisted|supported|performed)\b",
                s, re.IGNORECASE
            ))
            or (s and s[0].islower())
            or bool(re.search(r"\d{4}\s*[-–]\s*(\d{4}|present)", s, re.IGNORECASE))
        )

    def _validate(self, p: dict, raw_text: str) -> dict:
        """Validate and clean AI output."""
        # Filter bad education entries
        educations = [
            e for e in (p.get("educations") or [])
            if (e.get("school") or e.get("degree"))
            and not self._is_sentence(e.get("school", ""))
            and not self._is_sentence(e.get("degree", ""))
        ]

        # Filter bad work experience entries
        work = [
            e for e in (p.get("workExperiences") or [])
            if e.get("jobTitle") or e.get("company")
        ]

        # Validate phone — reject year ranges
        phone = p.get("phone", "")
        if phone and re.match(r"^(19|20)\d{2}", str(phone).strip()):
            phone = ""

        return {
            "name": p.get("name", ""),
            "email": p.get("email", ""),
            "phone": phone,
            "location": p.get("location", ""),
            "title": p.get("title", ""),
            "summary": p.get("summary", ""),
            "skills": [s for s in (p.get("skills") or []) if isinstance(s, str) and s.strip()],
            "softSkills": [s for s in (p.get("softSkills") or []) if isinstance(s, str) and s.strip()],
            "tools": [s for s in (p.get("tools") or []) if isinstance(s, str) and s.strip()],
            "workExperiences": work,
            "educations": educations,
            "projects": p.get("projects") or [],
            "certifications": p.get("certifications") or [],
            "competitions": p.get("competitions") or [],
        }

    def _fallback_parse(self, text: str) -> dict:
        email = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", text)
        phone = re.search(r"(?:\+91[\s-]?)?[6-9]\d{9}", text)
        return {
            "name": "", "email": email.group() if email else "",
            "phone": phone.group() if phone else "",
            "location": "", "title": "", "summary": "",
            "skills": [], "softSkills": [], "tools": [],
            "workExperiences": [], "educations": [],
            "projects": [], "certifications": [], "competitions": [],
        }


resume_parser_brain = ResumeParserBrain()
