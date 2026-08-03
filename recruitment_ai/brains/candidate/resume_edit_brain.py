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
        "generate": """You are a professional resume writer. Write exactly 3 different professional summary options.

RULES:
- Each option must be 2-3 sentences
- Use the EXACT job role from "Target Role:" in the profile below — do NOT substitute or invent a different role
- Reference the candidate's actual skills, experience, and achievements from the profile
- Start each with a strong action-oriented opening
- No placeholders like [X] or [Y], no labels, no markdown, no numbering
- Separate each option with "---" on its own line
- Return ONLY the 3 options, nothing else

Candidate Profile:
{context}""",
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
        "find_missing": """You are a resume skill advisor. List ONLY the skill names that are missing for the target role.

Current skills: {content}
Target role: {role}

RULES:
- Return ONLY short skill names (1-4 words max each), like: Selenium, JIRA, API Testing, SQL
- Do NOT return sentences, descriptions, or explanations
- Do NOT include skills already in the current skills list
- Return exactly 6-8 skills specific to the target role
- Comma-separated, no numbering, no bullets, no extra text
- Example output: Selenium, JIRA, TestNG, API Testing, SQL, Agile, Cypress, Postman

Return ONLY the comma-separated skill names:""",
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
    ("summary", "generate"): lambda ctx: f"Experienced {ctx.get('role', 'professional')} with a proven track record of delivering high-quality results. Skilled across the full {ctx.get('role', 'professional')} lifecycle with strong collaboration across cross-functional teams.",
    ("summary", "rewrite"): lambda ctx: ctx.get("content", ""),
    ("summary", "professional"): lambda ctx: ctx.get("content", ""),
    ("summary", "shorten"): lambda ctx: ctx.get("content", ""),
    ("summary", "friendly"): lambda ctx: ctx.get("content", ""),
    ("experience", "improve"): lambda ctx: ctx.get("content", ""),
    ("experience", "quantify"): lambda ctx: ctx.get("content", ""),
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
            or _extract_role_from_query(content)
            or state.context.get("experienceId")  # SkillGapLearning passes role here
            or section
        )

        if section == "resume" and action in ("score_advice", "analyze"):
            return await self._score_advice(content)

        if section == "resume" and action == "optimize":
            return await self._jd_optimize(content)

        # Normalize action: "improve" maps to "rewrite" for summary (no "improve" in summary prompts)
        if section == "summary" and action == "improve":
            action = "rewrite"

        prompt = self._build_prompt(section, action, content, role)
        if not prompt:
            # Graceful fallback — use rewrite/improve fallback instead of error
            fb = FALLBACKS.get((section, "rewrite")) or FALLBACKS.get((section, "improve"))
            reply = fb({"content": content, "role": role}) if fb else content or "I can help improve your resume. Please specify a section."
            return BrainResult(response={"reply": reply, "section": section, "action": action})

        try:
            result = await llm_service.generate(
                brain_name="resume_edit", prompt=prompt, system=SYSTEM,
                temperature=0.3, max_tokens=80 if action == "find_missing" else 512,
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

    async def _jd_optimize(self, content: str) -> BrainResult:
        prompt = f"""You are a professional resume optimizer. Given the job description and resume below, return a JSON object with these exact keys:
- "optimized_summary": a 2-3 sentence professional summary tailored to the JD
- "keywords": array of up to 10 important keywords/skills from the JD not already in the resume
- "optimized_bullets": array of up to 5 improved experience bullet points incorporating JD keywords
- "improvements": array of 3 specific actionable tips

RULES:
- Use ONLY skills and technologies already in the resume — do NOT invent new ones
- Keywords must come directly from the JD
- Return ONLY valid JSON, no markdown, no explanation

Content:
{content}"""
        try:
            result = await llm_service.generate(
                brain_name="resume_edit", prompt=prompt, system=SYSTEM,
                temperature=0.2, max_tokens=600,
            )
            result = self._clean(result)
            # Extract JSON from result
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return BrainResult(response={"reply": json.dumps(parsed), "section": "resume", "action": "optimize"})
            raise ValueError("No JSON in response")
        except Exception:
            # Fallback: extract keywords from JD text
            parts = content.split('---')
            jd_text = parts[0] if parts else content
            words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9+#.]{2,}\b', jd_text)
            stop = {'the','and','for','with','that','this','are','you','will','have','from','experience','work','team','role','position','candidate','required','preferred','ability','strong','good','excellent'}
            freq: dict = {}
            for w in words:
                c = w.lower()
                if c not in stop: freq[c] = freq.get(c, 0) + 1
            keywords = [w.capitalize() for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:10]]
            return BrainResult(response={"reply": json.dumps({
                "optimized_summary": "",
                "keywords": keywords,
                "optimized_bullets": [],
                "improvements": [
                    f"Add these JD keywords to your skills: {', '.join(keywords[:4])}",
                    "Quantify achievements with numbers and percentages",
                    "Start every bullet with a strong past-tense action verb",
                ],
            }), "section": "resume", "action": "optimize"})

    async def _score_advice(self, content: str) -> BrainResult:
        prompt = f"""You are a professional resume reviewer. Analyze this resume snapshot and give 4 specific, actionable improvement tips.

Focus ONLY on what is actually weak or missing based on the data below.
Be specific — mention exact sections, missing fields, or weak patterns you see.
Do NOT give generic advice. Do NOT repeat the scores back.
Return exactly 4 tips, one per line, plain text, no bullets, no numbering.

Resume Snapshot:
{content}"""
        try:
            result = await llm_service.generate(
                brain_name="resume_edit", prompt=prompt, system=SYSTEM,
                temperature=0.3, max_tokens=400,
            )
            result = self._clean(result)
            return BrainResult(response={"reply": result, "section": "resume", "action": "score_advice"})
        except Exception:
            return BrainResult(response={"reply": "Add quantified achievements with numbers and percentages\nStart every bullet with a strong past-tense action verb\nAdd LinkedIn profile and portfolio URL to boost credibility\nExpand experience bullets with specific tools and measurable outcomes", "section": "resume", "action": "score_advice"})

    def _clean(self, text: str) -> str:
        text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL)
        text = re.sub(r"^[-•*#]\s*", "", text, flags=re.MULTILINE)
        return text.strip()


resume_edit_brain = ResumeEditBrain()
