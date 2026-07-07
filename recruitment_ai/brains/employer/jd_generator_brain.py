"""JD Generator Brain - generates job descriptions from requirements."""
from typing import Optional
from recruitment_ai.shared.brain import Brain, BrainState
from recruitment_ai.shared.ollama_service import ollama_service

JD_GENERATOR_SYSTEM = """You are an expert HR professional and technical recruiter.
Write professional, inclusive, and compelling job descriptions. Return only the job description text."""


JD_GENERATOR_PROMPT = """Generate a professional job description.

Input:
- Title: {title}
- Company: {company}
- Location: {location}
- Job Type: {job_type}
- Experience Level: {experience_level}
- Skills Required: {skills_required}
- Skills Preferred: {skills_preferred}
- Responsibilities: {responsibilities}
- Requirements: {requirements}
- Benefits: {benefits}
- Salary Range: {salary_min} - {salary_max} {currency}

Output a well-structured job description with these sections:
1. About the Company
2. About the Role
3. Key Responsibilities
4. Required Qualifications
5. Preferred Qualifications
6. Benefits & Perks
7. Salary Range
8. How to Apply

Write in professional, inclusive tone. Use bullet points. Be specific and actionable.
Return only the job description text."""


class JDGeneratorBrain(Brain):
    """Generates job descriptions from structured input."""

    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainState:
        context = state.context or {}
        query = state.query or ""

        if not context and not query:
            state.error = "No job details provided"
            return state

        params = self._extract_params(context, query)
        prompt = JD_GENERATOR_PROMPT.format(**params)

        try:
            result = await ollama_service.generate(
                brain_name="jd_generator",
                prompt=prompt,
                system=JD_GENERATOR_SYSTEM,
                temperature=0.3,
                max_tokens=2048,
            )
            state.result = {
                "job_description": result.strip(),
                "title": params["title"],
                "model_used": "llama3.1:8b",
            }
        except Exception as e:
            state.result = {"job_description": self._template_fallback(params), "fallback": True}
            state.metadata["fallback_reason"] = str(e)

        return state

    def _extract_params(self, context: dict, query: str) -> dict:
        defaults = {
            "title": "Software Engineer",
            "company": "ZyncJobs",
            "location": "Remote",
            "job_type": "full-time",
            "experience_level": "mid",
            "skills_required": "Python, JavaScript, SQL",
            "skills_preferred": "AWS, Docker, React",
            "responsibilities": "Develop features, Write tests, Code review",
            "requirements": "3+ years experience, CS degree or equivalent",
            "benefits": "Health insurance, Remote work, Learning budget",
            "salary_min": "80000",
            "salary_max": "120000",
            "currency": "USD",
        }
        defaults.update(context)
        return defaults

    def _template_fallback(self, params: dict) -> str:
        return f"""
# {params['title']} at {params['company']}

## About the Company
{params['company']} is a leading technology company.

## About the Role
We are looking for a {params['experience_level']} {params['title']} to join our team in {params['location']}.

## Key Responsibilities
{params['responsibilities']}

## Required Qualifications
{params['requirements']}

## Skills Required
{params['skills_required']}

## Preferred Skills
{params['skills_preferred']}

## Benefits
{params['benefits']}

## Salary Range
{params['salary_min']} - {params['salary_max']} {params['currency']}

## How to Apply
Please submit your resume and cover letter.
""".strip()


jd_generator_brain = JDGeneratorBrain()