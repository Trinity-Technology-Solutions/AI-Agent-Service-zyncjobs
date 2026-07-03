import re
from .base_tool import BaseTool


EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")
PHONE_CLEAN = re.compile(r"[^\d+]")
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
NONSENSE_RE = re.compile(
    r"\b(ensuring|merging|including|creating|adaptability|developing|"
    r"implementing|managing|coordinating|improving|handling|working|"
    r"responsible|involved)\b",
    re.IGNORECASE,
)
# These are NOT real skills
STOP_SKILLS = {
    "ensuring", "adaptability", "merging", "including", "creating",
    "developing", "implementing", "managing", "coordinating", "improving",
    "handling", "working", "responsible", "involved", "leading", "testing",
    "monitoring", "maintaining", "supporting", "collaborating", "communicating",
    "analyzing", "designing", "building", "writing", "editing", "reviewing",
    "planning", "organizing", "participating", "contributing", "assisting",
    "performing", "conducting", "preparing", "processing", "producing",
    "providing", "utilizing", "understanding", "learning", "training",
    "mentoring", "coaching", "reporting", "documenting", "researching",
    "branch", "branches", "merge", "pull request", "commit", "repository",
}


class ValidationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="validator",
            description="Validates and normalizes parsed resume data",
        )

    def run(self, parsed: dict) -> dict:
        result = dict(parsed)
        personal = result.get("personal", {})
        if isinstance(personal, dict):
            if personal.get("email"):
                personal["email"] = self._validate_email(personal["email"])
            if personal.get("phone"):
                personal["phone"] = self._validate_phone(personal["phone"])
        if "experience" in result:
            result["experience"] = self._validate_experience(result["experience"])
        if "skills" in result:
            result["skills"] = self._validate_skills(result["skills"])
        if "education" in result:
            result["education"] = self._validate_education(result["education"])
        if "projects" in result:
            result["projects"] = self._validate_projects(result["projects"])
        return result

    def _validate_email(self, email: str) -> str:
        m = EMAIL_RE.match(email.strip())
        return m.group(0).lower() if m else ""

    def _validate_phone(self, phone: str) -> str:
        cleaned = PHONE_CLEAN.sub("", phone)
        if len(cleaned) >= 7 and cleaned.count("+") <= 1 and not YEAR_RE.match(cleaned):
            return cleaned
        return ""

    def _validate_experience(self, experiences: list) -> list:
        validated = []
        for exp in experiences:
            if not isinstance(exp, dict):
                continue
            company = (exp.get("company") or "").strip()
            role = (exp.get("role") or "").strip()
            if not company and not role:
                continue
            if not company:
                exp["company"] = "Unknown"
            start = (exp.get("start_date") or "").strip()
            end = (exp.get("end_date") or "").strip()
            if start and end and end.lower() != "present":
                sy = self._extract_year(start)
                ey = self._extract_year(end)
                if sy and ey and ey < sy:
                    exp["start_date"], exp["end_date"] = end, start
            resp = exp.get("responsibilities", [])
            if isinstance(resp, str):
                resp = [r.strip() for r in resp.split("\n") if r.strip()]
            elif not isinstance(resp, list):
                resp = []
            exp["responsibilities"] = [r for r in resp if r and r.strip()]
            validated.append(exp)
        return validated

    def _validate_education(self, education: list) -> list:
        validated = []
        for edu in education:
            if not isinstance(edu, dict):
                continue
            degree = (edu.get("degree") or "").strip()
            institution = (edu.get("institution") or "").strip()
            if not degree and not institution:
                continue
            year = (edu.get("year") or "").strip()
            if year and YEAR_RE.match(year):
                edu["year"] = YEAR_RE.match(year).group(1)
            validated.append(edu)
        return validated

    def _validate_skills(self, skills) -> dict:
        default = {"programming": [], "frameworks": [], "databases": [], "cloud": [], "tools": []}
        if not isinstance(skills, dict):
            return default
        for key in default:
            vals = skills.get(key, [])
            if isinstance(vals, list):
                skills[key] = [s for s in self._dedup(vals) if s.lower() not in STOP_SKILLS]
            else:
                skills[key] = []
        return skills

    def _validate_projects(self, projects: list) -> list:
        validated = []
        for proj in projects:
            if not isinstance(proj, dict):
                continue
            title = (proj.get("title") or "").strip()
            if not title:
                continue
            for field in ["technologies", "responsibilities"]:
                val = proj.get(field, [])
                if isinstance(val, str):
                    val = [v.strip() for v in val.split("\n") if v.strip()]
                elif not isinstance(val, list):
                    val = []
                proj[field] = [v for v in val if v and v.strip()]
            validated.append(proj)
        return validated

    def _dedup(self, items: list) -> list:
        seen = set()
        out = []
        for item in items:
            if isinstance(item, str):
                key = item.strip().lower()
                if key and key not in seen:
                    seen.add(key)
                    out.append(item.strip())
            else:
                out.append(item)
        return out

    def _extract_year(self, date_str: str) -> int | None:
        m = YEAR_RE.search(str(date_str))
        return int(m.group(1)) if m else None
