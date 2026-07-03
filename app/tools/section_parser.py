import json
import re
from .base_tool import BaseTool
from .skill_dict import filter_skills, categorize_skills
from app.llm.ollama import OllamaLLM


EXPERIENCE_PROMPT = """You are a resume experience parser. Extract all work experiences from the text below.

For each experience, return: company (string), role (string), start_date (string), end_date (string), location (string), responsibilities (list of strings).

Rules:
- company is the employer/organization name
- role is the job title/designation
- start_date and end_date are date ranges like "Jan 2020", "2020", "2020-01", or "Present"
- location is the work location city if mentioned
- responsibilities are SPECIFIC bullet points describing actual work done, each as a separate string
- NEVER merge multiple experiences into one entry
- NEVER include project descriptions or personal details as work experience
- If the text mentions a project separately from work experience, do NOT include it here
- Return ONLY valid JSON array. No markdown, no backticks, no explanation."""


EDUCATION_PROMPT = """You are a resume education parser. Extract all education entries from the text below.

For each education entry, return: degree (string), institution (string), year (string), cgpa (string).

Rules:
- Each degree/school combination is a SEPARATE entry. Never merge them.
- degree is the degree name like "B.Tech", "M.Sc", "Bachelor of Science", "12th", "HSC", "SSLC"
- institution is the school/college/university name
- year is the graduation or passing year (single year, not a range). If a range like "2018-2022", use the end year "2022"
- cgpa is the GPA/CGPA/percentage if mentioned, otherwise empty string
- Return ONLY valid JSON array. No markdown, no backticks, no explanation."""


PROJECTS_PROMPT = """You are a resume projects parser. Extract all projects from the text below.

For each project, return: title (string), description (string), technologies (list of strings), responsibilities (list of strings).

Rules:
- title is the project name
- description is a brief summary of what the project does
- technologies are tools/languages/frameworks used in the project
- responsibilities are YOUR specific contributions to the project, each as a separate string
- Only extract projects, NOT work experience or skills
- Return ONLY valid JSON array. No markdown, no backticks, no explanation."""


SKILLS_PROMPT = """You are a resume skills parser. Extract all technical and professional skills from the text below.

Return ONLY a JSON array of strings. Each string is one skill.

Rules:
- Extract individual skills like "Python", "React", "Docker", "Machine Learning", "Project Management"
- Do NOT extract full sentences, bullet points, or descriptions
- Do NOT extract soft skills phrases like "team player", "hard working", "good communication"
- If you see a group like "Python, Java, C++", split it into separate skills
- Remove duplicates
- Return ONLY valid JSON array. No markdown, no backticks, no explanation."""


SUMMARY_PROMPT = """You are a resume summary parser. Extract the professional summary or objective from the text.

Return ONLY: {"summary": "the summary text here"}

Rules:
- Keep the original wording as much as possible
- Truncate if over 500 characters
- If no summary exists, return {"summary": ""}
- Return ONLY valid JSON. No markdown, no backticks, no explanation."""


LANGUAGES_PROMPT = """You are a resume languages parser. Extract all languages mentioned.

Return ONLY a JSON array of strings. Each string is a language name.
Do NOT include proficiency levels. Return only the language names.
Return ONLY valid JSON. No markdown, no backticks, no explanation."""


CERTIFICATIONS_PROMPT = """You are a resume certifications parser. Extract all certifications from the text.

For each certification, return: name (string), issuer (string), year (string).

Rules:
- name is the certification title
- issuer is the organization that issued it (e.g., "Coursera", "AWS", "Google")
- year is the year obtained
- Return ONLY valid JSON array. No markdown, no backticks, no explanation."""


PUBLICATIONS_PROMPT = """You are a resume publications parser. Extract all publications from the text.

For each publication, return: title (string), venue (string), year (string).
Return ONLY valid JSON array. No markdown, no backticks, no explanation."""


HONORS_PROMPT = """You are a resume honors/awards parser. Extract all awards and honors from the text.

For each award, return: name (string), issuer (string), year (string).
Return ONLY valid JSON. No markdown, no backticks, no explanation."""


SECTION_PROMPTS: dict[str, str] = {
    "summary": SUMMARY_PROMPT,
    "experience": EXPERIENCE_PROMPT,
    "education": EDUCATION_PROMPT,
    "projects": PROJECTS_PROMPT,
    "skills": SKILLS_PROMPT,
    "languages": LANGUAGES_PROMPT,
    "certifications": CERTIFICATIONS_PROMPT,
    "publications": PUBLICATIONS_PROMPT,
    "honors": HONORS_PROMPT,
}


class SectionParserTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="section_parser",
            description="Dedicated per-section resume parser using Ollama",
        )
        self.llm = OllamaLLM()

    def run(self, section_name: str, section_text: str) -> dict | list:
        if not section_text.strip():
            return self._default(section_name)
        system_prompt = SECTION_PROMPTS.get(section_name, SUMMARY_PROMPT)
        user_prompt = f"Parse this {section_name} section from a resume:\n\n{section_text}"
        try:
            response = self.llm.generate(prompt=user_prompt, system=system_prompt)
            parsed = self._extract_json(response, section_name)
            return self._post_process(section_name, parsed)
        except Exception as e:
            print(f"[SectionParser] Error parsing '{section_name}': {e}")
            return self._default(section_name)

    def _post_process(self, section_name: str, parsed) -> dict | list:
        if section_name == "skills":
            if isinstance(parsed, list):
                filtered = filter_skills(parsed)
                return categorize_skills(filtered)
            if isinstance(parsed, dict):
                all_skills = []
                for v in parsed.values():
                    if isinstance(v, list):
                        all_skills.extend(v)
                filtered = filter_skills(all_skills)
                return categorize_skills(filtered)
            return self._default(section_name)
        if section_name == "experience" and isinstance(parsed, list):
            for exp in parsed:
                if isinstance(exp, dict):
                    resp = exp.get("responsibilities", [])
                    if isinstance(resp, str):
                        exp["responsibilities"] = [r.strip() for r in resp.split("\n") if r.strip()]
        if section_name == "projects" and isinstance(parsed, list):
            for proj in parsed:
                for field in ["technologies", "responsibilities"]:
                    val = proj.get(field, [])
                    if isinstance(val, str):
                        proj[field] = [v.strip() for v in val.split("\n") if v.strip()]
        return parsed

    def _extract_json(self, text: str, section_name: str):
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        array_start = text.find("[")
        array_end = text.rfind("]")
        if start != -1 and end != -1 and (array_start == -1 or start < array_start):
            json_str = text[start : end + 1]
        elif array_start != -1 and array_end != -1:
            json_str = text[array_start : array_end + 1]
        else:
            return self._default(section_name)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return self._default(section_name)

    def _default(self, section_name: str):
        if section_name == "skills":
            return {"programming": [], "frameworks": [], "databases": [], "cloud": [], "tools": []}
        if section_name in ("summary",):
            return {}
        return []
