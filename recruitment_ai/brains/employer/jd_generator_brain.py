"""JD Generator Brain — full enterprise pipeline.
Pipeline: BrainState.context_data → LLM → BrainResult
"""
import time
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service
from recruitment_ai.prompts import get_prompt

JD_GENERATOR_SYSTEM = """You are an expert HR professional and technical recruiter.
Write professional, inclusive, and compelling job descriptions. Return only the job description text."""


class JDGeneratorBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainResult:
        start = time.perf_counter()
        ctx = state.context_data
        context = state.context or {}
        query = state.request.query or state.query or ""

        if not context and not query and not ctx.job.title:
            return BrainResult(success=False, response={"error": "No job details provided"})

        params = self._extract_params(context, query, ctx)
        prompt = get_prompt("jd_generator_template", **params)

        try:
            result = await llm_service.generate(
                brain_name="jd_generator",
                prompt=prompt,
                system=JD_GENERATOR_SYSTEM,
                temperature=0.3,
                max_tokens=2048,
            )
            return BrainResult(
                response={"job_description": result.strip(), "title": params["title"], "model_used": "llm"},
                execution_time=time.perf_counter() - start,
            )
        except Exception as e:
            return BrainResult(
                response={"job_description": self._template_fallback(params), "fallback": True},
                metadata={"fallback_reason": str(e)},
            )

    def _extract_params(self, context: dict, query: str, ctx) -> dict:
        defaults = {
            "title": ctx.job.title or "Software Engineer",
            "company": ctx.company.name or "ZyncJobs",
            "location": ctx.job.description or "Remote",
            "experience_level": "mid",
            "skills": ", ".join(ctx.job.skills) if ctx.job.skills else "Python, JavaScript, SQL",
        }
        defaults.update(context)
        if query and not context:
            defaults["title"] = query
        return defaults

    def _template_fallback(self, params: dict) -> str:
        return f"""
# {params['title']} at {params['company']}

## About the Company
{params['company']} is a leading technology company.

## About the Role
We are looking for a {params.get('experience_level', 'mid')} {params['title']} to join our team in {params.get('location', 'Remote')}.

## Key Responsibilities
Develop features, Write tests, Code review

## Required Qualifications
3+ years experience, CS degree or equivalent

## Skills Required
{params.get('skills', 'Python, JavaScript, SQL')}

## Benefits
Health insurance, Remote work, Learning budget

## How to Apply
Please submit your resume and cover letter.
""".strip()


jd_generator_brain = JDGeneratorBrain()
