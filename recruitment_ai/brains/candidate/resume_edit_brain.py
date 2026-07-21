"""Resume Edit Brain — per-section resume AI actions.
Uses state.context_data.resume for pre-loaded resume data.
"""
import re
import json
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service

RESUME_EDIT_SYSTEM = """You are a professional resume writer. Return ONLY the requested content.
No explanations, no markdown, no code fences, no bullet symbols, no labels, no prefixes, no placeholders like [X] or [Y]."""
SYSTEM = RESUME_EDIT_SYSTEM

SECTION_PROMPTS = {
    "summary": {
        "generate": """Write a professional summary (2-3 sentences) for a {role}.
Candidate context: {context}
Use ONLY actual skills and experience from the context. Never invent or add technologies not mentioned. If the context has no specific data, write a generic high-quality summary for the given role.
Return ONLY the summary text — no labels, no prefixes, no placeholders, no quotes.""",
        "rewrite": """Rewrite this professional summary to be more impactful. Preserve all original technologies and skills.
{content}
Return ONLY the rewritten summary text.""",
        "professional": """Rewrite this professional summary in a formal, professional tone. Preserve all original content.
{content}
Return ONLY the rewritten summary.""",
        "shorten": """Shorten this professional summary to 2-3 concise sentences. Preserve key skills and role.
{content}
Return ONLY the shortened version.""",
        "friendly": """Rewrite this professional summary in a friendly, conversational tone.
{content}
Return ONLY the rewritten summary.""",
    },
    "experience": {
        "improve": """Improve this resume bullet point. Make it quantifiable, specific, and impactful. Use past tense action verbs. Include metrics where possible.
CRITICAL: Preserve ALL original technologies and tools exactly as written. Do NOT add any new programming languages, frameworks, or tools.

Original: {content}
Return ONLY the improved version — one sentence, no labels.""",
        "quantify": """Add a specific metric or number to this resume bullet point. Make it measurable and impactful while keeping the original meaning.
CRITICAL: Preserve ALL original technologies and tools exactly as written. Do NOT add or change any technologies.

Original: {content}
Return ONLY the quantified version — one sentence, no labels.""",
        "generate": """Generate 2-3 resume bullet points for {content}.
Each bullet must be quantifiable, start with a past-tense action verb.
Use ONLY technologies mentioned in the content. If the input has no specific technologies, keep the bullets general.
Return one bullet per line, no numbering.""",
    },
    "projects": {
        "improve": """Improve this project bullet point to be more specific and impactful:
{content}
CRITICAL: Preserve ALL original technologies exactly as written. Do NOT add any new ones.
Return ONLY the improved version.""",
        "generate": """Generate 2-3 bullet points describing project work for: {content}
Use ONLY technologies mentioned in the content. If none are mentioned, keep bullets general.
Return one per line, no numbering.""",
    },
    "skills": {
        "generate": """List relevant technical and soft skills for a {role} candidate.
Current profile: {content}
Use ONLY skills mentioned in the context. If the context has no specific skills, suggest common skills for that role (max 10).
Return as a comma-separated list — no labels, no numbering.""",
        "find_missing": """Given these current skills: {content}
Target role: {role}
Suggest 5-8 complementary skills relevant to the target role that build on the given skills.
Return as a comma-separated list.""",
    },
    "education": {
        "generate": """Suggest 2-3 relevant education entries for a candidate targeting: {content}
Each entry: Degree Name, Institution Name
Return one per line, no numbering.""",
    },
    "languages": {
        "generate": """List 5 common languages found on professional resumes.
Return as a comma-separated list.""",
    },
    "certifications": {
        "generate": """Suggest 3 certifications relevant for a {role} candidate. Target: {content}
Each line: Certification Name, Issuing Organization
Return one per line, no numbering.""",
    },
}

FALLBACKS = {
    ("summary", "generate"): lambda ctx: f"Experienced professional with expertise in {ctx.get('role', 'software development')}.",
    ("summary", "rewrite"): lambda ctx: ctx.get("content", ""),
    ("summary", "professional"): lambda ctx: ctx.get("content", ""),
    ("summary", "shorten"): lambda ctx: ctx.get("content", ""),
    ("summary", "friendly"): lambda ctx: ctx.get("content", ""),
    ("experience", "improve"): lambda ctx: ctx.get("content", ""),
    ("experience", "generate"): lambda ctx: "Implemented key features\nImproved system performance\nCollaborated with cross-functional teams",
    ("projects", "improve"): lambda ctx: ctx.get("content", ""),
    ("projects", "generate"): lambda ctx: "Built core functionality\nOptimized performance\nIntegrated APIs",
    ("skills", "generate"): lambda ctx: "Python, JavaScript, React, Node.js, SQL, Git, Docker, AWS, Agile, Communication",
    ("skills", "find_missing"): lambda ctx: "TypeScript, Kubernetes, CI/CD, GraphQL, Microservices",
    ("education", "generate"): lambda ctx: "B.E Computer Science, Anna University\nMCA, University of Madras",
    ("languages", "generate"): lambda _: "English, Tamil, Hindi, French, Spanish",
    ("certifications", "generate"): lambda ctx: "AWS Certified Solutions Architect, Amazon Web Services",
}


def _extract_role_from_query(query: str) -> str | None:
    match = re.search(r"Target Role:\s*(.+?)(?:\n|$)", query, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _infer_action_from_query(query: str) -> str:
    q = query.lower()
    if "shorten" in q: return "shorten"
    if "quantify" in q: return "quantify"
    if "rewrite" in q: return "rewrite"
    if "improve" in q or "optimize" in q: return "improve"
    if "professional" in q: return "professional"
    if "friendly" in q or "conversational" in q: return "friendly"
    if "generate" in q or "create" in q or "write" in q: return "generate"
    if "fix grammar" in q or "grammar" in q: return "improve"
    if "find missing" in q or "missing skills" in q: return "find_missing"
    return "improve"


def _infer_section_from_query(query: str) -> str:
    q = query.lower()
    for s in ["summary", "experience", "education", "skills", "projects", "languages", "certifications"]:
        if s in q:
            return s
    return "summary"


class ResumeEditBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainResult:
        query = state.request.query or state.query or ""
        section = (state.context.get("section") or _infer_section_from_query(query)).lower()
        action = (state.context.get("action") or _infer_action_from_query(query)).lower()
        content = state.context.get("content") or query
        role = (
            state.context.get("role")
            or state.context_data.user_preferences.get("targetRole")
            or _extract_role_from_query(query)
            or section
        )

        if section == "resume" and action in ("score_advice", "analyze"):
            return await self._score_advice(content)

        prompt = self._build_prompt(section, action, content, role)
        if not prompt:
            return BrainResult(success=False, response={"reply": "I can help improve your resume. Please try a specific section."})

        try:
            result = await llm_service.generate(
                brain_name="resume_edit", prompt=prompt, system=SYSTEM,
                temperature=0.3, max_tokens=512,
            )
            result = self._clean(result)
            if not result.strip():
                raise ValueError("Empty result from LLM")
            return BrainResult(response={"reply": result, "section": section, "action": action})
        except Exception:
            fb = FALLBACKS.get((section, action))
            return BrainResult(response={"reply": fb({"content": content, "role": role}) if fb else "", "section": section, "action": action})

    def _build_prompt(self, section: str, action: str, content: str, role: str) -> str:
        if section not in SECTION_PROMPTS:
            return ""
        actions = SECTION_PROMPTS[section]
        if action not in actions:
            return ""
        return actions[action].format(role=role, content=content, context=content)

    async def _score_advice(self, content: str) -> BrainResult:
        prompt = f"""Analyze this resume and give 3-4 specific, actionable improvement tips.
Focus on: missing sections, weak bullet points, skill gaps, formatting issues, and ATS optimization.
Be concrete — mention specific skills, sections, or metrics whenever possible.

Resume context: {content}

Return one tip per line as plain text. No labels, no numbering."""
        try:
            result = await llm_service.generate(
                brain_name="resume_edit", prompt=prompt, system=SYSTEM,
                temperature=0.3, max_tokens=512,
            )
            result = self._clean(result)
            return BrainResult(response={"reply": result, "section": "resume", "action": "score_advice"})
        except Exception:
            return BrainResult(response={"reply": "Add more quantifiable achievements\nUse stronger action verbs\nEnsure consistent formatting", "section": "resume", "action": "score_advice"})

    def _clean(self, text: str) -> str:
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
        text = re.sub(r"^[-•*#]\s*", "", text, flags=re.MULTILINE)
        return text.strip()


resume_edit_brain = ResumeEditBrain()
