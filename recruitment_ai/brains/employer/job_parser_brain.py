"""Job Parser Brain — full enterprise pipeline.
Pipeline: BrainState.request → LLM → JSON Validator → BrainResult
"""
import re
import json
import time
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service
from recruitment_ai.validators.json_validator import validate_json_strict
from recruitment_ai.prompts import get_prompt, get_system_prompt

SKILL_KEYWORDS = [
    # ── Tech (existing) ─────────────────────────────────────────────
    "python", "java", "javascript", "react", "node", "sql", "aws", "docker",
    "kubernetes", "git", "linux", "agile", "scrum", "rest", "api", "html",
    "css", "typescript", "go", "rust", "c++", "c#", ".net", "spring",
    "django", "flask", "fastapi", "postgresql", "mongodb", "redis",
    # ── HSE / Oil & Gas ─────────────────────────────────────────────
    "nebosh", "osha", "hse", "fire safety", "risk assessment", "hazard analysis",
    "first aid", "confined space", "permit to work", "iso 45001", "iso 14001",
    "process safety", "drilling", "offshore", "onshore", "pipeline", "petrochemical",
    "oil and gas", "safety management", "incident investigation",
    # ── Construction & Engineering ──────────────────────────────────
    "autocad", "revit", "civil engineering", "structural engineering",
    "mechanical engineering", "electrical engineering", "hvac", "plumbing",
    "primavera", "quantity surveying", "estimation", "site supervision",
    "quality control", "qa/qc", "blueprint", "construction management",
    # ── Finance & Accounting ────────────────────────────────────────
    "accounting", "tally", "quickbooks", "taxation", "audit", "financial analysis",
    "budgeting", "payroll", "reconciliation", "invoicing", "financial reporting",
    "accounts payable", "accounts receivable", "credit analysis", "underwriting",
    "kyc", "aml", "insurance", "claims processing", "banking",
    # ── HR & Admin ──────────────────────────────────────────────────
    "recruitment", "talent acquisition", "employee relations", "hr policies",
    "labor law", "visa processing", "documentation", "administration",
    "scheduling", "office management", "attendance", "compensation",
    # ── Sales & Marketing ───────────────────────────────────────────
    "sales", "marketing", "digital marketing", "seo", "customer relationship",
    "lead generation", "negotiation", "telemarketing", "e-commerce", "retail",
    "merchandising", "customer service", "crm", "telecalling", "b2b sales",
    "key account management", "cold calling",
    # ── Healthcare ──────────────────────────────────────────────────
    "nursing", "patient care", "pharmacology", "medical records", "clinic",
    "physiotherapy", "home care", "emergency", "infection control", "hospital",
    "cpr", "radiology", "laboratory", "wound care", "clinical research",
    # ── Education & Training ────────────────────────────────────────
    "teaching", "curriculum", "lesson planning", "ielts", "training",
    "classroom management", "student assessment", "instructional design",
    # ── Logistics & Supply Chain ────────────────────────────────────
    "logistics", "warehouse", "supply chain", "inventory", "procurement",
    "shipping", "freight", "customs", "forklift", "driving license",
    "vendor management", "dispatch", "route planning",
    # ── Hospitality ─────────────────────────────────────────────────
    "housekeeping", "food and beverage", "catering", "front office", "chef",
    "hotel management", "guest relations", "event planning", "reservation",
    # ── Legal & Compliance ──────────────────────────────────────────
    "contract law", "legal research", "compliance", "litigation",
    "legal documentation", "due diligence", "intellectual property",
    # ── Media & Communications ──────────────────────────────────────
    "content writing", "copywriting", "video editing", "photography",
    "journalism", "public relations", "advertising", "scriptwriting",
    # ── Common global soft skills / office ──────────────────────────
    "communication", "time management", "multitasking", "data entry",
    "microsoft excel", "ms office", "report writing", "presentation skills",
]

_TITLE_VERBS = re.compile(
    r"\b(design|develop|test|maintain|build|lead|manage|create|implement|write|"
    r"support|monitor|ensure|analyze|collaborate|deliver|prepare|responsible|drive|own)\w*\b",
    re.IGNORECASE,
)
_META_WORDS = re.compile(
    r"^(experience|exp|salary|location|skills?|department|employment|job type|work type|"
    r"notice|joining|ctc|lpa|about|apply|hiring|urgent|immediate|duties|responsibilities|qualifications)\b",
    re.IGNORECASE,
)
_SENTENCE_LIKE = re.compile(r",|\b(and|to|for|with|the|that|this|which|of)\b", re.IGNORECASE)
_EXP_RANGE = re.compile(r"(\d+)\s*(?:-|to|–|—)\s*(\d+)\s*years?", re.IGNORECASE)
_EXP_PLUS = re.compile(r"(\d+)\+\s*years?", re.IGNORECASE)
_ARRAY_FIELDS = ("mustHaveSkills", "goodToHaveSkills", "responsibilities", "requirements", "benefits", "jobType")


class JobParserBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainResult:
        start = time.perf_counter()
        from recruitment_ai.utils.ocr import extract_text

        prefs = state.context_data.user_preferences or {}
        context = state.context or {}

        # 1) Clean job text sent via context (preferred)
        content = ""
        job_text = context.get("job_text") or prefs.get("job_text") or ""
        if isinstance(job_text, str) and job_text.strip():
            content = job_text.strip()
        else:
            # 2) File upload
            raw_content = state.request.file_content or state.file_content or ""
            file_type = state.request.file_type or state.file_type or "txt"
            if raw_content:
                content = extract_text(raw_content, file_type) if file_type != "txt" else raw_content
            else:
                # 3) Legacy: query may carry the raw JD, possibly wrapped in a prompt
                content = state.request.query or state.query or ""
            # Strip legacy prompt wrapper if present
            if "JOB POST:" in content:
                content = content.split("JOB POST:", 1)[1].strip()
            content = content.strip()

        if not content:
            return BrainResult(success=False, response={"error": "No job description provided"})

        system = get_system_prompt("job_parser")
        prompt = get_prompt("job_parser_prompt", job_text=content)

        try:
            result = await llm_service.generate(
                brain_name="job_parser",
                prompt=prompt,
                system=system,
                temperature=0.1,
                max_tokens=1024,
            )
            parsed = validate_json_strict(result, "object") or {}
            parsed = self._normalize(parsed, content)
            return BrainResult(
                response=parsed,
                metadata={"parser": "llm"},
                execution_time=time.perf_counter() - start,
            )
        except Exception as e:
            return BrainResult(
                response=self._fallback_parse(content),
                metadata={"parser": "fallback", "fallback_reason": str(e)},
                execution_time=time.perf_counter() - start,
            )

    def _normalize(self, parsed: dict, content: str) -> dict:
        """Post-parse strict validation: correct keys, types, formats, and the job title."""
        # Normalize title keys → jobTitle
        if "jobTitle" not in parsed or not parsed["jobTitle"]:
            parsed["jobTitle"] = parsed.get("title") or parsed.get("job_title") or ""
        for k in ("title", "job_title"):
            parsed.pop(k, None)
        parsed["jobTitle"] = self._sanitize_title(parsed.get("jobTitle") or "", content)

        # Enforce array types
        for k in _ARRAY_FIELDS:
            if not isinstance(parsed.get(k), list):
                parsed[k] = []

        # Enforce experienceRange format: "X-Y years" / "X+ years"
        exp = str(parsed.get("experienceRange") or "")
        m = _EXP_RANGE.search(exp)
        if m:
            parsed["experienceRange"] = f"{m.group(1)}-{m.group(2)} years"
        else:
            m2 = _EXP_PLUS.search(exp)
            parsed["experienceRange"] = f"{m2.group(1)}+ years" if m2 else ""

        # Enforce experienceLevel enum
        if parsed.get("experienceLevel") not in ("Entry", "Mid", "Senior", "Lead"):
            parsed["experienceLevel"] = "Mid"

        # Enforce workSetting enum
        if parsed.get("workSetting") not in ("Remote", "Hybrid", "On-site"):
            parsed["workSetting"] = "On-site"

        return parsed

    def _sanitize_title(self, title: str, content: str) -> str:
        """Reject sentence/responsibility-like titles; fall back to the JD's first meaningful line."""
        t = re.sub(r"[*#]+", "", title or "").strip()
        words = t.split()
        bad = (
            len(words) > 6 or len(t) > 60
            or bool(re.search(r"@|^\d|\.$", t))
            or bool(_META_WORDS.match(t))
            or (len(words) >= 3 and bool(_TITLE_VERBS.search(t)) and bool(_SENTENCE_LIKE.search(t)))
        )
        if t and not bad:
            return t
        lines = content.split("\n")
        for line in lines[:8]:
            line = re.sub(r"^[-*#\d+.)\s]+", "", line.strip())
            line = re.sub(r"[-|].+$", "", line.strip()).strip()
            if not line or len(line) > 80 or len(line) < 3:
                continue
            lw = line.split()
            if len(lw) > 6 or "," in line or re.search(r"@|^\d", line) or _META_WORDS.match(line):
                continue
            if len(lw) >= 3 and _TITLE_VERBS.search(line) and _SENTENCE_LIKE.search(line):
                continue
            return line
        return ""

    def _fallback_parse(self, content: str) -> dict:
        lines = content.split("\n")
        skills = [kw for kw in SKILL_KEYWORDS if re.search(rf"\b{kw}\b", content, re.IGNORECASE)]
        currency = "USD"
        if re.search(r"[₹]|INR|lakh|LPA", content, re.IGNORECASE):
            currency = "INR"
        elif re.search(r"AED|dirham|Dhs?\.?\s", content, re.IGNORECASE):
            currency = "AED"
        elif re.search(r"OMR|Omani", content, re.IGNORECASE):
            currency = "OMR"
        elif re.search(r"QAR|Qatari", content, re.IGNORECASE):
            currency = "QAR"
        elif re.search(r"SAR|Saudi", content, re.IGNORECASE):
            currency = "SAR"
        elif re.search(r"KWD|Kuwaiti", content, re.IGNORECASE):
            currency = "KWD"
        elif re.search(r"EUR|€", content):
            currency = "EUR"
        elif re.search(r"GBP|£", content):
            currency = "GBP"
        return {
            "jobTitle": self._sanitize_title("", content),
            "company": "", "location": "", "jobType": ["Full-time"],
            "workSetting": "On-site", "experienceLevel": "Mid",
            "experienceRange": "", "salaryMin": None, "salaryMax": None,
            "currency": currency, "mustHaveSkills": skills[:10], "goodToHaveSkills": [],
            "responsibilities": [], "requirements": [], "benefits": [],
            "description": content[:2000],
        }


job_parser_brain = JobParserBrain()
