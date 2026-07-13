"""Cover Letter Brain — generates personalized cover letters from resume data."""
import re
from recruitment_ai.shared.brain import Brain, BrainState
from recruitment_ai.shared.ollama_service import ollama_service

SYSTEM = """You are a professional cover letter writer.
Write a compelling, personalized cover letter. Return ONLY the cover letter text.
No labels, no JSON, no markdown, no explanation. Start directly with 'Dear Hiring Manager,'"""

PROMPT = """Write a {tone} cover letter for {candidate_name} applying for {job_title} at {company}.

Candidate Profile:
- Summary: {summary}
- Experience: {experience}
- Skills: {skills}

Requirements:
- 3-4 paragraphs
- Opening: express interest in the specific role at {company}
- Middle: highlight 2-3 relevant achievements from their experience
- Closing: call to action
- Sign off with candidate name
- Tone: {tone}

Return ONLY the cover letter text starting with 'Dear Hiring Manager,'"""


class CoverLetterBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainState:
        context = state.context or {}
        candidate_name = context.get("candidate_name", "the candidate")
        job_title = context.get("job_title", "the position")
        company = context.get("company", "your company")
        tone = context.get("tone", "professional")
        summary = context.get("summary", "")
        experience = context.get("experience", "")
        skills = context.get("skills", "")

        prompt = PROMPT.format(
            candidate_name=candidate_name,
            job_title=job_title,
            company=company,
            tone=tone,
            summary=summary[:300] if summary else "Experienced professional",
            experience=experience[:300] if experience else "Relevant work experience",
            skills=skills[:200] if skills else "Various technical skills",
        )

        try:
            result = await ollama_service.generate(
                brain_name="cover_letter",
                prompt=prompt,
                system=SYSTEM,
                temperature=0.4,
                max_tokens=800,
            )
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
            if not result.strip():
                raise ValueError("Empty result")
            state.result = {"cover_letter": result, "company": company, "job_title": job_title}
        except Exception:
            state.result = {"cover_letter": self._fallback(candidate_name, job_title, company, summary, experience, skills, tone)}

        return state

    def _fallback(self, name, job_title, company, summary, experience, skills, tone):
        greeting = "I am thrilled to apply" if tone == "enthusiastic" else "I am applying" if tone == "concise" else "I am writing to express my interest in applying"
        return f"""Dear Hiring Manager,

{greeting} for the {job_title} position at {company}.

{summary or f"As an experienced professional, I bring strong expertise in {skills}."}

{f"My experience includes {experience}, where I consistently delivered high-quality results." if experience else ""}

I am particularly drawn to {company} because of its commitment to innovation. My skills in {skills} align well with the requirements of this role, and I am confident I can make a meaningful contribution to your team.

I would welcome the opportunity to discuss how my background can contribute to {company}'s continued success.

Thank you for your consideration.

Sincerely,
{name or "Your Name"}"""


cover_letter_brain = CoverLetterBrain()
