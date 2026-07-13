"""Resume Edit Brain — handles per-section resume AI actions.
Matches the frontend's { section, action, content } pattern and produces
output in the exact format each UI component expects."""

import re
import json
from recruitment_ai.shared.brain import Brain, BrainState
from recruitment_ai.shared.ollama_service import ollama_service

RESUME_EDIT_SYSTEM = """You are a professional resume writer. Return ONLY the requested content.
No explanations, no markdown, no code fences, no bullet symbols unless part of the output format."""

SYSTEM = RESUME_EDIT_SYSTEM

SECTION_PROMPTS = {
    "summary": {
        "generate": """Write a professional summary (2-3 sentences) for a {role}.
Candidate context: {context}
Return ONLY the summary text — no labels, no prefixes, no placeholders.""",
        "rewrite": """Rewrite this professional summary to be more impactful:
{content}
Return ONLY the rewritten summary text.""",
        "professional": """Rewrite this professional summary in a formal, professional tone:
{content}
Return ONLY the rewritten summary.""",
        "shorten": """Shorten this professional summary to 2-3 concise sentences:
{content}
Return ONLY the shortened version.""",
        "friendly": """Rewrite this professional summary in a friendly, conversational tone:
{content}
Return ONLY the rewritten summary.""",
    },
    "experience": {
        "improve": """Improve this resume bullet point. Make it quantifiable, specific, and impactful.
Use past tense action verbs. Include metrics where possible.

Original: {content}
Return ONLY the improved version — one sentence, no labels.""",
        "quantify": """Add a specific metric or number to this resume bullet point to make it more impactful.
Keep the original meaning. Add percentage, dollar amount, time saved, or user count.

Original: {content}
Return ONLY the improved version — one sentence, no labels.""",
        "generate": """Generate 3-4 resume bullet points for {content}.
Each bullet must be quantifiable, start with a past-tense action verb.
Return one bullet per line. No numbering, no bullet symbols.""",
    },
    "projects": {
        "improve": """Improve this project bullet point. Make it specific and impactful.
Use past tense action verbs. Include technical details.

Original: {content}
Return ONLY the improved version — one sentence.""",
        "generate": """Generate 3-4 bullet points describing project work for: {content}.
Each bullet must be specific, technical, and start with a past-tense action verb.
Return one per line. No numbering, no bullet symbols.""",
    },
    "skills": {
        "generate": """List relevant technical and soft skills for a {role} candidate.
Context: {context}
Return as a comma-separated list. Each skill should be 1-3 words.
Example: Python, JavaScript, AWS, React, Team Leadership, Agile Methodology""",
        "find_missing": """Given these current skills: {content}
Suggest 5-8 complementary skills that would strengthen this profile.
Return as a comma-separated list.""",
    },
    "education": {
        "generate": """Suggest 2-3 relevant education entries for a candidate targeting: {content}
Each entry should be formatted as: Degree Name, Institution Name
Return one per line. Example:
B.E Computer Science, Anna University
BCA, University of Madras""",
    },
    "languages": {
        "generate": """List 5 common languages found on professional resumes.
Return as a comma-separated list.
Example: English, Tamil, Hindi, Spanish, French""",
    },
    "certifications": {
        "generate": """Suggest 3 certifications relevant for a {role} candidate. Target: {content}
Each line: Certification Name, Issuing Organization
Example: AWS Solutions Architect, Amazon
    Certified Kubernetes Administrator, CNCF
    Project Management Professional, PMI""",
    },
}

FALLBACKS = {
    ("summary", "generate"): lambda ctx: f"Experienced professional with expertise in {ctx.get('role', 'software development')}. Proven track record of delivering high-quality results.",
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
    query_lower = query.lower()
    if "shorten" in query_lower:
        return "shorten"
    if "improve" in query_lower or "optimize" in query_lower or "quantify" in query_lower:
        return "improve"
    if "professional" in query_lower:
        return "professional"
    if "friendly" in query_lower or "conversational" in query_lower:
        return "friendly"
    if "generate" in query_lower or "create" in query_lower or "write" in query_lower:
        return "generate"
    if "fix grammar" in query_lower or "grammar" in query_lower:
        return "improve"
    return "improve"


def _infer_section_from_query(query: str) -> str:
    query_lower = query.lower()
    sections = ["summary", "experience", "education", "skills", "projects", "languages", "certifications"]
    for s in sections:
        if s in query_lower:
            return s
    return "summary"


class ResumeEditBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainState:
        context = state.context or {}
        section = (context.get("section") or _infer_section_from_query(state.query or "")).lower()
        action = (context.get("action") or _infer_action_from_query(state.query or "")).lower()
        content = context.get("content") or state.query or ""
        role = context.get("role", section)

        if section == "resume" and action in ("score_advice", "analyze"):
            return await self._score_advice(state, content)

        prompt = self._build_prompt(section, action, content, role)
        if not prompt:
            state.error = f"Unknown section/action: {section}/{action}"
            state.result = {"reply": "I can help improve your resume. Please try a specific section."}
            return state

        try:
            result = await ollama_service.generate(
                brain_name="resume_edit",
                prompt=prompt,
                system=SYSTEM,
                temperature=0.3,
                max_tokens=512,
            )
            result = self._clean(result)
            if not result.strip():
                raise ValueError("Empty result from LLM")
            state.result = {"reply": result, "section": section, "action": action}
        except Exception:
            fb = FALLBACKS.get((section, action))
            state.result = {"reply": fb({"content": content, "role": role}) if fb else "", "section": section, "action": action}

        return state

    def _build_prompt(self, section: str, action: str, content: str, role: str) -> str:
        if section not in SECTION_PROMPTS:
            return ""
        actions = SECTION_PROMPTS[section]
        if action not in actions:
            return ""
        template = actions[action]
        ctx = {"role": role, "content": content, "context": content}
        return template.format(**ctx)

    async def _score_advice(self, state: BrainState, content: str) -> BrainState:
        prompt = f"""Analyze this resume score context and give 3-4 specific improvement tips:

{content}

Return one tip per line as plain text. No numbering, no bullet symbols."""
        try:
            result = await ollama_service.generate(
                brain_name="resume_edit",
                prompt=prompt,
                system=SYSTEM,
                temperature=0.3,
                max_tokens=512,
            )
            result = self._clean(result)
            state.result = {"reply": result, "section": "resume", "action": "score_advice"}
        except Exception:
            state.result = {
                "reply": "Add more quantifiable achievements\nUse stronger action verbs\nEnsure consistent formatting",
                "section": "resume",
                "action": "score_advice",
            }
        return state

    def _clean(self, text: str) -> str:
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
        text = re.sub(r"^[-•*#]\s*", "", text, flags=re.MULTILINE)
        text = text.strip()
        return text


resume_edit_brain = ResumeEditBrain()