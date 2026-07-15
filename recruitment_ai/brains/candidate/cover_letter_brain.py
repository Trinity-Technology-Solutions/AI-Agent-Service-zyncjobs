"""Cover Letter Brain — full enterprise pipeline.
Pipeline: BrainState.context_data → LLM → BrainResult
"""
import re
import time
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service

COVER_LETTER_SYSTEM = """You are a professional cover letter writer.
Write a compelling, personalized cover letter. Return ONLY the cover letter text.
No labels, no JSON, no markdown, no explanation. Start directly with 'Dear Hiring Manager,'"""

COVER_LETTER_PROMPT = """Write a {tone} cover letter for {candidate_name} applying for {job_title} at {company}.

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

    async def run(self, state: BrainState) -> BrainResult:
        start = time.perf_counter()
        ctx = state.context_data
        candidate_name = state.user.name or state.context.get("candidate_name", "the candidate")
        job_title = ctx.job.title or state.context.get("job_title", "the position")
        company = ctx.job.company_name or state.context.get("company", "your company")
        tone = state.context.get("tone", "professional")
        resume = ctx.resume
        summary = str(resume.parsed)[:300] if resume.parsed else "Experienced professional"
        experience_text = "\n".join(e.get("description", "") for e in resume.experience) if resume.experience else "Relevant work experience"
        skills_text = ", ".join(resume.skills[:10]) if resume.skills else "Various technical skills"

        prompt = COVER_LETTER_PROMPT.format(
            candidate_name=candidate_name,
            job_title=job_title,
            company=company,
            tone=tone,
            summary=summary[:300],
            experience=experience_text[:300],
            skills=skills_text[:200],
        )

        try:
            result = await llm_service.generate(
                brain_name="cover_letter",
                prompt=prompt,
                system=COVER_LETTER_SYSTEM,
                temperature=0.4,
                max_tokens=800,
            )
            result = re.sub(r"<think>.*?</think>", "", result, flags=re.DOTALL).strip()
            if not result.strip():
                raise ValueError("Empty result")
            return BrainResult(
                response={"cover_letter": result, "company": company, "job_title": job_title},
                execution_time=time.perf_counter() - start,
            )
        except Exception:
            return BrainResult(
                response={"cover_letter": self._fallback(candidate_name, job_title, company, summary, experience_text, skills_text, tone)},
            )

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
