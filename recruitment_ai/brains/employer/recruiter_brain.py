"""Recruiter Brain â€” full enterprise pipeline with AI Recruiter Assistant mode.

Pipeline: BrainState.context_data (recruiter_context bundle) â†’ LLM â†’ JSON Validator â†’ BrainResult

Two modes:
  1. ASSISTANT mode (chat): real employer data (company, jobs, candidates, ATS scores, stats)
     loaded by the recruiter_context loader is injected into every answer. Queries are
     routed by intent: ranking, comparison, shortlist, JD review, salary, interview
     questions, analytics, emails, general chat.
  2. LEGACY mode: structured endpoints (/ai/recruiter/*) and fallback chat using
     frontend-sent jobs_context / user_profile when no real data is available.
"""
import re
import json
import time
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service
from recruitment_ai.validators.json_validator import validate_json_strict
from recruitment_ai.prompts import get_prompt, get_system_prompt

ASSISTANT_SYSTEM = f"""You are the ZyncJobs AI Recruiter Assistant â€” a data-aware recruitment assistant for employers, NOT a generic chatbot.

You are talking to a REAL employer whose REAL hiring data is provided below in "Employer Context" (company profile, active jobs, candidates with match scores, ATS scores, missing skills, application stats).

STRICT RULES:
1. GROUND every answer in the Employer Context. Never give generic recruitment advice that ignores the employer's actual jobs and candidates.
2. Reference actual candidate names, their match %, ATS scores, missing skills, and actual job titles from the context.
3. NEVER invent candidates, scores, jobs, or stats that are not in the context.
4. If the data is insufficient for the question, say exactly what data is missing â€” then give best-effort advice based on what IS available.
5. Format answers with concise markdown: bold key names, bullet points, short paragraphs. Use âœ“ (recommend) and âœ— (reject) symbols for shortlist decisions.
6. NEVER mention competitor platforms (LinkedIn, Indeed, Glassdoor, Naukri, Monster).
7. Never recommend candidate-facing features (Resume Builder, Skill Gap Analysis, Career Roadmap) â€” do the analysis yourself inline.
8. Be professional, concise, and actionable."""


class RecruiterBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainResult:
        start = time.perf_counter()
        query = (state.request.query or state.query or "").strip() or "Help me with my hiring"
        intent = (state.intent or "").upper()

        # â”€â”€ Structured endpoints (/ai/recruiter/*) â€” legacy JSON behavior â”€â”€
        if intent in ("RECRUITER_SEARCH", "RECRUITER_SHORTLIST", "RANKING"):
            if intent == "RECRUITER_SEARCH":
                return await self._search(query, state.context, start)
            if intent == "RECRUITER_SHORTLIST":
                return await self._shortlist(state.context, start)
            return await self._score_candidate(state, start)

        # â”€â”€ ASSISTANT mode: real employer data from the context loader â”€â”€
        rc = getattr(state.context_data, "recruiter_context", None) or {}
        if isinstance(rc, dict) and (rc.get("jobs") or rc.get("stats") or rc.get("company")):
            return await self._assistant_chat(state, rc, query, start)

        # â”€â”€ Legacy fallback (no real data available) â”€â”€
        ctx = state.context_data
        has_structured_context = bool(
            ctx.job.title or ctx.job.description or ctx.company.name
            or state.context.get("candidates") or state.context.get("candidate") or state.context.get("job")
        )
        if not has_structured_context:
            return await self._chat_recruiter(state, query, start)

        if "shortlist" in query.lower() or "evaluate" in query.lower():
            return await self._shortlist(state.context, start)
        if "score" in query.lower() or "rank" in query.lower():
            return await self._score_candidate(state, start)
        return await self._search(query, state.context, start)

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # ASSISTANT MODE
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def _build_assistant_context(self, rc: dict) -> str:
        """Build a compact, complete employer context block from the loaded bundle."""
        parts = []
        company = rc.get("company") or {}
        name = company.get("name") or "Your company"
        parts.append(f"Company: {name}")
        if company.get("industry"):
            parts.append(f"Industry: {company['industry']}")
        if company.get("location"):
            parts.append(f"Location: {company['location']}")

        stats = rc.get("stats") or {}
        if stats:
            parts.append(
                f"Pipeline stats: {stats.get('applications', 0)} applications, "
                f"{stats.get('pending', 0)} pending, {stats.get('shortlisted', 0)} shortlisted, "
                f"{stats.get('interviewed', 0)} interviewed, {stats.get('rejected', 0)} rejected, "
                f"{stats.get('hired', 0)} hired"
            )

        jobs = rc.get("jobs") or []
        if jobs:
            job_lines = []
            for j in jobs[:5]:
                title = j.get("title") or "Untitled role"
                line = f"- {title}"
                if j.get("location"):
                    line += f" | Location: {j['location']}"
                if j.get("experienceRange"):
                    line += f" | Experience: {j['experienceRange']}"
                if j.get("salaryMin") or j.get("salaryMax"):
                    line += f" | Salary: {j.get('salaryMin')}-{j.get('salaryMax')} {j.get('currency', '')}"
                if j.get("jobType"):
                    line += f" | Type: {j['jobType']}"
                skills = j.get("skills") or []
                if skills:
                    line += f" | Required skills: {', '.join(skills[:8])}"
                cand_lines = []
                for c in (j.get("topCandidates") or [])[:4]:
                    cname = c.get("name") or "Unnamed candidate"
                    ctext = f"    - {cname}"
                    if c.get("overallScore") is not None:
                        ctext += f" | Match: {c['overallScore']}%"
                    if c.get("aiScore") is not None:
                        ctext += f" | ATS: {c['aiScore']}"
                    missing = c.get("missingSkills") or []
                    if missing:
                        ctext += f" | Missing skills: {', '.join(missing[:4])}"
                    cand_skills = c.get("skills") or []
                    if cand_skills:
                        ctext += f" | Skills: {', '.join(cand_skills[:6])}"
                    ctext += f" | Status: {c.get('applicationStatus', 'applied')}"
                    cand_lines.append(ctext)
                if cand_lines:
                    line += "\n" + "\n".join(cand_lines)
                else:
                    line += "\n    (no applications yet)"
                job_lines.append(line)
            parts.append("Employer Jobs & Candidates:\n" + "\n".join(job_lines))
        else:
            parts.append("Active jobs: none yet")

        return "\n".join(parts)

    def _assistant_history(self, state: BrainState) -> str:
        prefs = state.context_data.user_preferences or {}
        history_text = ""
        for turn in (prefs.get("history", []) or [])[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_text += f"{role}: {content}\n"
        return history_text

    async def _generate(self, system: str, prompt: str, temperature: float = 0.4, max_tokens: int = 900) -> str:
        try:
            reply = await llm_service.generate(
                brain_name="recruiter", prompt=prompt, system=system,
                temperature=temperature, max_tokens=max_tokens,
            )
            return re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
        except Exception:
            return ""

    def _assistant_intent(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ("shortlist", "short-list", "who should i shortlist", "should i shortlist")):
            return "shortlist"
        if any(k in q for k in ("compare", "comparison", "versus", " vs ", "between john", "between david")):
            return "compare"
        if any(k in q for k in ("rank", "best candidate", "top candidate", "best fit", "who is best", "which candidate")):
            return "rank"
        if any(k in q for k in ("job description", " jd", "optimize", "optimise", "improve my job", "posting", "write a job", "attract more")):
            return "jd"
        if any(k in q for k in ("salary", "offer", "compensation", "package", "how much to pay", "what should i pay")):
            return "salary"
        if any(k in q for k in ("interview question", "questions for", "prepare interview", "interview questions")):
            return "interview"
        if any(k in q for k in ("analytics", "drop", "dropping", "trend", "why are applicant", "applications are", "improve my applications", "insight")):
            return "analytics"
        if any(k in q for k in ("email", "template", "rejection", "follow-up", "follow up", "reach out", "message to", "write a message")):
            return "email"
        return "general"

    _INTENT_PROMPTS = {
        "rank": """The employer asks: "{query}"

Using ONLY the candidates in the Employer Context, rank them for the relevant job. Structure:
- **Top candidates** â€” name, match %, ATS score, and 1-2 specific reasons from their data (skills matched, missing skills)
- Clear verdict on who is the best fit and why (reference their match score, missing skills, experience)
- A note on which candidates are weak fits and what is missing""",
        "compare": """The employer asks: "{query}"

Compare the candidates mentioned (or the top candidates in the Employer Context) directly:
- For each candidate: match %, ATS score, key strengths from their actual skills, and gaps (missing skills)
- Side-by-side verdict â€” who is the better fit for the job and why
- If the specific names aren't in the context, compare the top 2 candidates you DO have and say so""",
        "shortlist": """The employer asks: "{query}"

Using ONLY the candidates in the Employer Context, give a shortlist recommendation:
- âœ“ **Shortlist**: candidate names with match %, key strengths, why they fit the role
- âœ— **Reject**: candidate names with the missing skills or red flags
- One-line next step (e.g. schedule interviews with the shortlisted)""",
        "jd": """The employer asks: "{query}"

Review the employer's ACTUAL job descriptions from the Employer Context. For the most relevant job:
- What the JD is missing (salary range, benefits, remote policy, certifications, growth path)
- How to improve the requirements list using the actual required skills in the context
- Rewrite or improve a section inline if helpful â€” keep it grounded in the real job""",
        "salary": """The employer asks: "{query}"

Give a salary recommendation grounded in the Employer Context:
- Reference the actual job (title, location, experience range, salary band already posted) from the context
- Recommend a range based on location and experience level
- If no salary data exists in the context, say so and give a market-based estimate for that role/location
- Always state total-comp considerations (base, bonus, benefits) briefly""",
        "interview": """The employer asks: "{query}"

Generate interview questions for the employer's ACTUAL role(s) from the Employer Context:
- Use the real job title and required skills listed in the context
- Structure: technical/domain questions (using the actual required skills), scenario questions, behavioral questions
- Keep it to 8-12 sharp questions""",
        "analytics": """The employer asks: "{query}"

Analyze the employer's REAL pipeline stats and jobs from the Employer Context:
- Interpret the actual numbers (applications, shortlisted, interviewed, rejected) â€” spot drop-offs between stages
- Diagnose likely causes using the real jobs (experience bar, salary band, skills, location)
- Give 3-4 concrete fixes""",
        "email": """The employer asks: "{query}"

Write a professional recruiter email based on the Employer Context:
- Use the real company name and role title(s) from the context
- Provide a ready-to-send template with subject line
- Keep it warm, professional, and concise""",
        "general": """The employer asks: "{query}"

Answer using the Employer Context. Ground your answer in the real company, jobs, candidates, and stats provided. If the question is about something outside the data, say what data would help and give practical advice based on what is available.""",
    }

    async def _assistant_chat(self, state: BrainState, rc: dict, query: str, start: float) -> BrainResult:
        intent = self._assistant_intent(query)
        system = ASSISTANT_SYSTEM + "\n\nEmployer Context (REAL DATA â€” ground every answer in this):\n" + self._build_assistant_context(rc)
        clean_query = re.sub(r'^recruiter:\s*', '', query, flags=re.IGNORECASE).strip()
        prompt = self._INTENT_PROMPTS.get(intent, self._INTENT_PROMPTS["general"]).format(query=clean_query)
        history_text = self._assistant_history(state)
        if history_text:
            prompt = f"Previous conversation:\n{history_text}\n\n{prompt}"

        reply = await self._generate(system, prompt, temperature=0.4, max_tokens=900)
        if not reply:
            reply = self._assistant_fallback(rc, clean_query)
        return BrainResult(
            response={"reply": reply, "intent": "RECRUITER", "assistant_intent": intent},
            execution_time=time.perf_counter() - start,
        )

    def _assistant_fallback(self, rc: dict, query: str) -> str:
        """Deterministic fallback built entirely from the real context â€” usable even if the LLM is down."""
        jobs = rc.get("jobs") or []
        stats = rc.get("stats") or {}
        lines = ["Here's what your current hiring data shows:"]
        if jobs:
            for j in jobs[:3]:
                title = j.get("title") or "Untitled role"
                tops = j.get("topCandidates") or []
                lines.append(f"\n**{title}** ({j.get('location', '')}) â€” {j.get('totalApplicants', 0)} applicant(s)")
                for c in tops[:3]:
                    missing = ", ".join(c.get("missingSkills") or [])[:60] or "none"
                    lines.append(
                        f"  â€¢ {c.get('name')} â€” match {c.get('overallScore')}% | "
                        f"ATS {c.get('aiScore') if c.get('aiScore') is not None else 'n/a'} | missing: {missing}"
                    )
                if not tops:
                    lines.append("  â€¢ No applications yet for this role.")
        if stats.get("applications"):
            lines.append(
                f"\nPipeline: {stats['applications']} applications, {stats.get('shortlisted', 0)} shortlisted, "
                f"{stats.get('interviewed', 0)} interviewed, {stats.get('rejected', 0)} rejected."
            )
        if not jobs:
            lines.append("\nNo active jobs with applications found yet â€” once candidates apply, the AI Recruiter Assistant will rank and analyze them here.")
        lines.append("\nAsk me to rank candidates, compare applicants, review a JD, or suggest salaries for these roles.")
        return "\n".join(lines)

    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # LEGACY MODE (no real context â€” frontend-sent jobs_context / user_profile)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

    def _build_employer_context(self, prefs: dict) -> str:
        """Build employer context block from frontend-sent jobs_context and user_profile."""
        parts = []
        profile = prefs.get("user_profile") or {}
        if isinstance(profile, dict):
            name = profile.get("name") or profile.get("fullName") or ""
            company = profile.get("company") or profile.get("companyName") or ""
            if name:
                parts.append(f"Recruiter Name: {name}")
            if company:
                parts.append(f"Company: {company}")
        jobs = prefs.get("jobs_context") or []
        if isinstance(jobs, list) and jobs:
            job_lines = []
            for j in jobs[:5]:
                if not isinstance(j, dict):
                    continue
                title = j.get("jobTitle") or j.get("title") or ""
                skills = j.get("skills")
                skills_text = ", ".join(skills[:5]) if isinstance(skills, list) else (str(skills) if skills else "")
                loc = j.get("location") or ""
                exp = j.get("experienceRange") or j.get("experienceLevel") or ""
                line = f"- {title} (location: {loc}, experience: {exp}".rstrip(", ")
                if skills_text:
                    line += f", skills: {skills_text}"
                job_lines.append(line + ")")
            if job_lines:
                parts.append("Employer Active Jobs:\n" + "\n".join(job_lines))
        return "\n".join(parts)

    async def _chat_recruiter(self, state: BrainState, query: str, start: float) -> BrainResult:
        prefs = state.context_data.user_preferences or {}
        clean_query = re.sub(r'^recruiter:\s*', '', query, flags=re.IGNORECASE).strip()
        history_text = ""
        for turn in (prefs.get("history", []) or [])[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_text += f"{role}: {content}\n"

        # Frontend systemPrompt override wins (it already embeds job context)
        system = prefs.get("systemPrompt") or get_system_prompt("recruiter_chat")
        employer_context = self._build_employer_context(prefs)
        if "{{ employer_context }}" in system:
            system = system.replace(
                "{{ employer_context }}",
                f"Employer Context:\n{employer_context}" if employer_context
                else "Employer Context: (No active jobs available yet.)",
            )

        prompt = clean_query
        if history_text:
            prompt = f"Previous conversation:\n{history_text}\n\nRecruiter: {clean_query}"
        try:
            reply = await llm_service.generate(
                brain_name="recruiter", prompt=prompt, system=system,
                temperature=0.4, max_tokens=600,
            )
            reply = re.sub(r"<think>.*?</think>", "", reply, flags=re.DOTALL).strip()
            return BrainResult(
                response={"reply": reply, "intent": "RECRUITER"},
                execution_time=time.perf_counter() - start,
            )
        except Exception:
            return BrainResult(
                response={"reply": "I'm having trouble right now. Please try again.", "intent": "RECRUITER"},
            )

    async def _search(self, query: str, context: dict, start: float) -> BrainResult:
        prompt = get_prompt("recruiter_prompt",
            query=query or "Find candidates",
            filters=json.dumps(context.get("filters", {})),
            skills=", ".join(context.get("skills", [])),
            experience_level=context.get("experience_level", "mid"),
            location=context.get("location", ""),
        )
        system = get_system_prompt("recruiter")
        try:
            result = await llm_service.generate(
                brain_name="recruiter", prompt=prompt, system=system,
                temperature=0.3, max_tokens=1024,
            )
            parsed = validate_json_strict(result, "object") or {}
            return BrainResult(response=parsed, execution_time=time.perf_counter() - start)
        except Exception as e:
            return BrainResult(
                response=self._fallback_search(context),
                metadata={"fallback": True, "error": str(e)},
            )

    async def _shortlist(self, context: dict, start: float) -> BrainResult:
        job = context.get("job_requirements", "")
        candidates = context.get("candidates", [])
        prompt = SHORTLIST_PROMPT.format(job_requirements=job[:2000], candidates=json.dumps(candidates)[:3000])
        system = get_system_prompt("recruiter")
        try:
            result = await llm_service.generate(
                brain_name="recruiter", prompt=prompt, system=system,
                temperature=0.1, max_tokens=1024,
            )
            parsed = validate_json_strict(result, "object") or {}
            return BrainResult(response=parsed, execution_time=time.perf_counter() - start)
        except Exception as e:
            return BrainResult(
                response={"shortlisted": [], "top_candidate_id": "", "summary": "Shortlisting evaluation unavailable"},
                metadata={"error": str(e)},
            )

    async def _score_candidate(self, state: BrainState, start: float) -> BrainResult:
        prefs = state.context_data.user_preferences
        context = state.context
        job_data = context.get("job") or prefs.get("job") or {}
        job_desc = job_data.get("description") or job_data.get("jobDescription") or state.context_data.job.description or ""
        job_skills = job_data.get("skills") or state.context_data.job.skills or []
        if isinstance(job_skills, list) and job_skills:
            job_desc = f"{job_desc}\n\nRequired Skills: {', '.join(str(s) for s in job_skills if s)}".strip()
        candidate = context.get("candidate")
        if candidate is None:
            candidates = context.get("candidates") or []
            candidate = candidates[0] if candidates else prefs.get("candidate") or {}
        if isinstance(candidate, list):
            candidate = candidate[0] if candidate else {}
        candidate = dict(candidate or {})
        # Never discard candidate data: build a readable resume from whatever fields exist
        if not candidate.get("resume"):
            parts = []
            if candidate.get("skills"):
                parts.append("Skills: " + ", ".join(str(s) for s in candidate["skills"] if s))
            if candidate.get("yearsExp") or candidate.get("experience"):
                parts.append("Experience: " + str(candidate.get("yearsExp") or candidate.get("experience")))
            if candidate.get("education"):
                parts.append("Education: " + str(candidate["education"]))
            if candidate.get("location"):
                parts.append("Location: " + str(candidate["location"]))
            if candidate.get("profileSummary"):
                parts.append("Summary: " + str(candidate["profileSummary"]))
            if parts:
                candidate["resume"] = "\n".join(parts)
        prompt = f"""Score this candidate for the job.

Job: {job_desc[:2000]}
Candidate: {json.dumps(candidate)[:2000]}

Return JSON with:
{{"overallScore": 0-100, "skillsScore": 0-100, "experienceScore": 0-100,
  "shouldReject": false, "reasons": [], "feedback": "",
  "recommendation": "strong|consider|reject",
  "matchingSkills": [], "missingSkills": [],
  "summary": "", "breakdown": {{"skill_weight": 0, "experience_weight": 0, "education_weight": 0, "location_weight": 0}}
}}"""
        system = get_system_prompt("recruiter")
        try:
            result = await llm_service.generate(
                brain_name="recruiter", prompt=prompt, system=system,
                temperature=0.1, max_tokens=1024,
            )
            parsed = validate_json_strict(result, "object") or {}
            if "overallScore" in parsed or "score" in parsed:
                return BrainResult(response=parsed, execution_time=time.perf_counter() - start)
            return BrainResult(
                response={"overallScore": 50, "skillsScore": 50, "reasons": ["AI evaluation unavailable"]},
                execution_time=time.perf_counter() - start,
            )
        except Exception as e:
            return BrainResult(
                response={"overallScore": 50, "skillsScore": 50, "reasons": [str(e)]},
                metadata={"fallback": True, "error": str(e)},
            )

    def _fallback_search(self, context: dict) -> dict:
        skills = context.get("skills", ["Python", "JavaScript"])
        return {
            "search_strategy": f"Search candidates with skills: {', '.join(skills)}",
            "recommended_filters": {
                "skills": skills, "experience": context.get("experience_level", "mid"),
                "location": context.get("location", "remote"),
            },
            "screening_questions": [
                f"Can you describe your experience with {skills[0] if skills else 'your primary skill'}?",
                "What project are you most proud of?",
            ],
            "evaluation_criteria": {"skill_weight": 40, "experience_weight": 30, "education_weight": 15, "location_weight": 10, "other_weight": 5},
            "interview_suggestions": {"rounds": 3, "topics": skills[:3], "estimated_duration_minutes": 60},
            "advice": "Focus on practical skills assessment rather than years of experience.",
        }


SHORTLIST_PROMPT = """Evaluate candidates for shortlisting.

Job Requirements: {job_requirements}
Candidates: {candidates}

Return JSON with:
{{"shortlisted": [{{"candidate_id": "", "name": "", "match_score": 0, "strengths": [], "gaps": [], "recommendation": "strong|consider|reject"}}],
  "top_candidate_id": "",
  "summary": "Brief evaluation summary"
}}"""


recruiter_brain = RecruiterBrain()
