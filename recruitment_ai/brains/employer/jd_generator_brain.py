"""JD Generator Brain — full enterprise pipeline.
Pipeline: BrainState.context_data → LLM → BrainResult
"""
import re
import time
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service
from recruitment_ai.prompts import get_prompt, get_system_prompt

JD_GENERATOR_SYSTEM = get_system_prompt("jd_generator")


class JDGeneratorBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainResult:
        start = time.perf_counter()
        ctx = state.context_data
        # context comes from both state.context (backward compat) and user_preferences
        context = {**state.context_data.user_preferences, **(state.context or {})}
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
                temperature=0.75,
                max_tokens=3000,
            )
            jd = self._clean_jd(result)
            return BrainResult(
                response={"job_description": jd, "title": params["title"], "model_used": "llm"},
                execution_time=time.perf_counter() - start,
            )
        except Exception as e:
            return BrainResult(
                response={"job_description": self._template_fallback(params), "fallback": True},
                metadata={"fallback_reason": str(e)},
            )

    @staticmethod
    def _clean_jd(text: str) -> str:
        """Strip markdown headings/bold, normalise bullets, collapse blank runs."""
        cleaned = re.sub(r"^#{1,6}\s*(.*)$", r"\1", text or "", flags=re.MULTILINE)
        cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)  # bold
        cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)       # italic
        cleaned = re.sub(r"^[\*\u2022]\s+", "- ", cleaned, flags=re.MULTILINE)  # normalise bullets
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    def _extract_params(self, context: dict, query: str, ctx) -> dict:
        # Resolve title: typed ctx > flat context > query
        title = (
            ctx.job.title
            or context.get("title")
            or context.get("jobTitle")
            or (query.replace("generate job description for ", "").strip() if query else "")
            or "Software Engineer"
        )
        # Resolve skills: typed ctx > flat context
        skills_list = ctx.job.skills or context.get("skills") or []
        skills_str = ", ".join(skills_list) if isinstance(skills_list, list) else str(skills_list)

        defaults = {
            "title": title,
            "company": ctx.company.name or context.get("company") or "Our Company",
            "location": context.get("location") or "Remote",
            "experience_level": context.get("experience_level", "mid"),
            "experience_range": context.get("experienceRange") or context.get("experience_range") or "",
            "education": context.get("educationLevel") or context.get("education") or "",
            "skills": skills_str,
            "job_type": (", ".join(context["jobType"]) if isinstance(context.get("jobType"), list) else context.get("jobType")) or context.get("job_type") or "Full-time",
            "salary": context.get("salary") or "",
            "benefits": ", ".join(context["benefits"]) if isinstance(context.get("benefits"), list) else context.get("benefits") or "",
            "responsibilities": context.get("responsibilities") or [],
            "requirements": context.get("requirements") or [],
            "variation": context.get("variation", 1),
        }
        return defaults

    def _template_fallback(self, params: dict) -> str:
        title = params.get("title", "the role")
        company = params.get("company", "Our Company")
        location = params.get("location", "Remote")
        job_type = params.get("job_type", "Full-time")
        experience = params.get("experience_range", "")
        education = params.get("education", "")
        skills = params.get("skills", "")
        salary = params.get("salary", "")
        benefits = params.get("benefits", "")
        responsibilities = params.get("responsibilities") or []
        requirements = params.get("requirements") or []

        resp_items = [r for r in responsibilities if r]
        if not resp_items:
            resp_items = [
                f"Own and drive end-to-end {title} activities with a focus on quality and impact",
                "Collaborate with cross-functional teams to define, plan, and deliver key initiatives",
                "Analyse performance metrics and translate insights into actionable improvements",
                "Ensure timely, high-quality delivery aligned with business objectives",
                "Contribute to process improvements and operational best practices",
                "Communicate progress, risks, and outcomes clearly to stakeholders",
            ]

        req_items = [q for q in requirements if q]
        if not req_items:
            req_items = [
                f"{experience} of relevant experience in a {title} or similar role" if experience else f"Proven experience in a {title} or closely related role",
                education if education else "Bachelor's degree in a relevant field or equivalent practical experience",
                f"Strong proficiency in: {skills}" if skills else "Proficiency in role-relevant tools and technologies",
                "Excellent communication, collaboration, and stakeholder management skills",
                "Strong analytical thinking and problem-solving ability",
                "Ability to manage multiple priorities and deliver under deadlines",
            ]

        resp_text = "\n".join(f"- {r}" for r in resp_items)
        req_text = "\n".join(f"- {r}" for r in req_items)
        skills_text = skills if skills else "Role-relevant tools and technologies"
        benefits_text = benefits if benefits else "- Comprehensive health, dental, and vision insurance\n- Flexible working arrangements\n- Annual learning and development budget\n- Performance-based bonuses\n- Paid time off and wellness days\n- Collaborative, inclusive work culture"
        salary_text = salary if salary else "Competitive compensation package commensurate with experience"

        return f"""Job Summary
We are seeking a talented and driven {title} to join {company} in {location}. In this {job_type} role, you will take ownership of key responsibilities, collaborate with high-performing teams, and deliver measurable impact for the business. This is an excellent opportunity for a motivated professional looking to grow their career in a dynamic and supportive environment.

About the Company
{company} is a forward-thinking organisation committed to excellence and innovation, hiring through ZyncJobs to connect with top talent.

What You Will Do
{resp_text}

What We Are Looking For
{req_text}

Technical Skills & Expertise
{skills_text}

What Makes You Stand Out
- Strong attention to detail with a commitment to delivering high-quality work
- Ability to work effectively both independently and as part of a collaborative team
- Clear and confident communication with stakeholders at all levels
- Ownership mindset with a results-driven, proactive approach
- Adaptability and eagerness to learn in a fast-paced environment

Experience & Education
{experience if experience else f"Relevant professional experience in a {title} role"}
{education if education else "Relevant degree or equivalent practical experience"}

Compensation & Benefits
Salary: {salary_text}
{benefits_text}

How to Apply
Interested candidates are invited to apply directly through this ZyncJobs job posting. Click the Apply button to submit your application. We look forward to hearing from you.""".strip()


jd_generator_brain = JDGeneratorBrain()
