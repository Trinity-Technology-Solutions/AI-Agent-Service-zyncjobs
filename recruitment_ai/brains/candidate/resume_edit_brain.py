"""Resume Edit Brain — per-section resume AI actions.
Uses state.context_data.resume for pre-loaded resume data.
"""
import re
import json
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service

RESUME_EDIT_SYSTEM = """You are a professional resume writer. Return ONLY the requested content.
No explanations, no markdown, no code fences, no bullet symbols unless part of the output format."""
SYSTEM = RESUME_EDIT_SYSTEM

SECTION_PROMPTS = {
    "summary": {
        "generate": """Write a professional summary (2-3 sentences) for a {role}.
Candidate context: {context}
Return ONLY the summary text — no labels, no prefixes, no placeholders.""",
        "rewrite": """Rewrite this professional summary to be more impactful:\n{content}\nReturn ONLY the rewritten summary text.""",
        "professional": """Rewrite this professional summary in a formal, professional tone:\n{content}\nReturn ONLY the rewritten summary.""",
        "shorten": """Shorten this professional summary to 2-3 concise sentences:\n{content}\nReturn ONLY the shortened version.""",
        "friendly": """Rewrite this professional summary in a friendly, conversational tone:\n{content}\nReturn ONLY the rewritten summary.""",
    },
    "experience": {
        "improve": """Improve this resume bullet point. Make it quantifiable, specific, and impactful.\nUse past tense action verbs. Include metrics where possible.\n\nOriginal: {content}\nReturn ONLY the improved version.""",
        "quantify": """Add a specific metric or number to this resume bullet point.\n\nOriginal: {content}\nReturn ONLY the improved version.""",
        "generate": """Generate 3-4 resume bullet points for {content}.\nEach bullet must be quantifiable, start with a past-tense action verb.\nReturn one bullet per line.""",
    },
    "projects": {
        "improve": """Improve this project bullet point:\n{content}\nReturn ONLY the improved version.""",
        "generate": """Generate 3-4 bullet points describing project work for: {content}\nReturn one per line.""",
    },
    "skills": {
        "generate": """List relevant technical and soft skills for a {role} candidate.\nContext: {context}\nReturn as a comma-separated list.""",
        "find_missing": """Given these current skills: {content}\nSuggest 5-8 complementary skills.\nReturn as a comma-separated list.""",
    },
    "education": {
        "generate": """Suggest 2-3 relevant education entries for a candidate targeting: {content}\nEach entry: Degree Name, Institution Name\nReturn one per line.""",
    },
    "languages": {
        "generate": """List 5 common languages found on professional resumes.\nReturn as a comma-separated list.""",
    },
    "certifications": {
        "generate": """Suggest 3 certifications relevant for a {role} candidate. Target: {content}\nEach line: Certification Name, Issuing Organization""",
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


def _infer_action_from_query(query: str) -> str:
    q = query.lower()
    if "shorten" in q: return "shorten"
    if "improve" in q or "optimize" in q or "quantify" in q: return "improve"
    if "professional" in q: return "professional"
    if "friendly" in q or "conversational" in q: return "friendly"
    if "generate" in q or "create" in q or "write" in q: return "generate"
    if "fix grammar" in q or "grammar" in q: return "improve"
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
        role = state.context.get("role", section)

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
        prompt = f"""Analyze this resume score context and give 3-4 specific improvement tips:\n\n{content}\n\nReturn one tip per line as plain text."""
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
