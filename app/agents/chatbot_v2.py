"""
ZyncJobs Chatbot V2 — Agent + Tool + RAG Architecture

Flow:
  User → Safety → Intent → Agent Router
                              ├── Candidate Agent (job search, resume, career, interview, salary)
                              ├── Recruiter Agent (JD gen, candidate search)
                              └── Platform Agent (FAQ, help, account)
                                        ↓
                              Tool / RAG / AI Service
                                        ↓
                              Response Validator → Reply
"""
import re
import random
from typing import Optional
from app.middleware.safety import validate, is_off_topic, OFF_TOPIC_MESSAGE
from app.middleware.intent_classifier import classify
from app.memory.conversation import ConversationMemory
from app.services.ai_service import ai_service
from app.tools.search_jobs import SearchJobsTool
from app.tools.salary_tool import SalaryTool
from app.tools.faq_tool import FAQTool
from app.tools.ats_tool import ATSTool
from app.agents.career_agent import CareerAgent
from app.agents.interview_agent import InterviewAgent
from app.agents.resume_agent import ResumeAgent
from app.agents.recruiter_agent import RecruiterAgent
from app.utils.logger import logger

# ── Singletons ────────────────────────────────────────────────────────────────
memory = ConversationMemory()
search_jobs_tool = SearchJobsTool()
salary_tool = SalaryTool()
faq_tool = FAQTool()
ats_tool = ATSTool()
career_agent = CareerAgent()
interview_agent = InterviewAgent()
resume_agent = ResumeAgent()
recruiter_agent = RecruiterAgent()

# ── Local Templates (no AI needed) ──────────────────────────────────────────
GREETING_REPLIES = [
    "Hi! I'm the **ZyncJobs AI Assistant**.\n\nI can help you with:\n- Find jobs by skill and location\n- Resume improvement and ATS score\n- Interview preparation\n- Career roadmap and skill gap analysis\n- Salary insights\n- Recruiter tools\n\nWhat can I help you with today?",
    "Hello! Welcome to ZyncJobs AI.\n\nAsk me anything about:\n- Jobs and openings\n- Resume improvement\n- Interview prep\n- Career guidance\n\nHow can I assist you?",
    "Hey there! I'm your ZyncJobs AI Assistant.\n\nTry asking:\n- Find React jobs in Chennai\n- Improve my resume\n- Prepare me for a Python interview\n\nWhat would you like help with?",
]

SMALL_TALK_REPLIES = {
    "thanks": "You're welcome! Is there anything else I can help you with?",
    "thank you": "Happy to help! Let me know if you need anything else.",
    "ok": "Got it! Feel free to ask anything about jobs, resumes, or career guidance.",
    "okay": "Sure! Let me know if you need help with jobs, resumes, or interviews.",
    "cool": "Great! Anything else I can help you with?",
    "great": "Glad to hear that! Anything else?",
    "awesome": "Let me know if you need anything else!",
    "who are you": "I'm the ZyncJobs AI Assistant — built to help you find jobs, improve your resume, prepare for interviews, and grow your career. What can I do for you?",
    "what are you": "I'm the ZyncJobs AI Assistant — built to help you find jobs, improve your resume, prepare for interviews, and grow your career.",
    "what can you do": "I can help you with:\n- Job search\n- Resume improvement and ATS score\n- Interview preparation\n- Career roadmap\n- Salary insights\n- Recruiter tools\n\nJust ask!",
    "default": "I'm here to help! You can ask me about jobs, resumes, interviews, career guidance, or salary insights.",
}


def _local_greeting() -> str:
    return random.choice(GREETING_REPLIES)


def _local_small_talk(message: str) -> str:
    lower = message.lower().strip()
    for key, reply in SMALL_TALK_REPLIES.items():
        if key in lower:
            return reply
    return SMALL_TALK_REPLIES["default"]


DOMAIN_CLASSIFIER_PROMPT = (
    "You are a classifier. Determine whether the user's message is related to: "
    "careers, jobs, recruitment, resumes, interviews, salary, companies, education, or professional development. "
    "Reply ONLY with YES or NO."
)

UNKNOWN_AI_SYSTEM = (
    "You are ZyncJobs AI Assistant. Answer the user's career or recruitment related question helpfully and concisely. "
    "You may discuss job portals, companies, technologies, education, and professional development."
)

# ── Fallbacks ─────────────────────────────────────────────────────────────────
FALLBACK_UNKNOWN = (
    "I specialize in careers, jobs, resumes, hiring, interviews, and professional development. "
    "Your question seems unrelated to those topics. "
    "If you have a career-related question, I'd be happy to help."
)
FALLBACK_ERROR = "I'm having trouble processing your request right now. Please try again in a few moments."

# ── System prompts (AI-only intents) ─────────────────────────────────────────
SYSTEM_PROMPTS = {
    "JOB_APPLICATION": (
        "You are ZyncJobs Application Guide. Help users apply for jobs effectively, "
        "write cover letters, and track applications on ZyncJobs."
    ),
    "RESUME_BUILDER": (
        "You are ZyncJobs Resume Builder. Guide the user step-by-step to build a professional resume."
    ),
    "PLATFORM_HELP": (
        "You are the ZyncJobs Support Assistant. "
        "ZyncJobs is an AI-powered recruitment platform for job seekers and employers. "
        "Help users navigate its features: job search, resume tools, ATS score, interview prep, career roadmap, and recruiter tools. "
        "Use the provided context when available, otherwise answer from your knowledge about ZyncJobs."
    ),
    "ACCOUNT_HELP": (
        "You are ZyncJobs Account Support. "
        "Help users with account setup, login, profile, password reset, and settings on ZyncJobs. "
        "Use the provided context when available, otherwise answer helpfully."
    ),
    "COMPANY_FAQ": (
        "You are the ZyncJobs Platform Expert. "
        "ZyncJobs is an AI-powered recruitment platform that connects job seekers with employers using AI-driven job matching, "
        "resume analysis, interview preparation, career roadmaps, and salary insights. "
        "Answer questions about ZyncJobs features, how it works, and its benefits. "
        "Use the provided context when available, otherwise answer from your knowledge about ZyncJobs."
    ),
    "JOB_DETAILS": (
        "You are ZyncJobs Job Information Assistant. Provide details about job roles and requirements."
    ),
    "JOB_RECOMMENDATION": (
        "You are ZyncJobs Job Recommender. Suggest suitable jobs based on the user's skills and experience."
    ),
}

JOB_SEARCH_SYSTEM = (
    "You are the ZyncJobs Job Search Assistant. "
    "Your ONLY job is to present job listings from the ZyncJobs database. "
    "STRICT RULES:\n"
    "1. NEVER mention LinkedIn, Indeed, Glassdoor, Naukri, Monster, or any other job portal.\n"
    "2. NEVER invent or suggest jobs that are not in the provided data.\n"
    "3. NEVER answer from your own general knowledge about job markets.\n"
    "4. If jobs are found, present them clearly with title, company, location, type, and skills.\n"
    "5. If no jobs are found, suggest the user refine their search on ZyncJobs or try similar role names — do NOT recommend other platforms.\n"
    "6. Always end with an offer to filter by location, experience, or job type."
)

DEFAULT_SYSTEM = (
    "You are ZyncJobs AI Assistant. Help with careers, resumes, jobs, interviews, and recruitment only. "
    "Never answer questions unrelated to recruitment and careers."
)


class ChatbotV2:

    async def handle(self, message: str, session_id: str, user_id: Optional[str] = None, user_role: str = "candidate") -> dict:

        # ── Step 1: Input Validation ──────────────────────────────────────────
        validation = validate(message)
        if not validation["valid"]:
            return self._resp(validation["message"], "BLOCKED", session_id)

        # ── Step 2: Off-topic Guardrail ───────────────────────────────────────
        if is_off_topic(message):
            return self._resp(OFF_TOPIC_MESSAGE, "UNKNOWN", session_id)

        # ── Step 3: Intent Classification ────────────────────────────────────
        clf = classify(message)
        intent = clf["intent"]
        entities = clf["entities"]
        logger.info(f"ChatbotV2 | session={session_id} | intent={intent} | entities={entities}")

        if intent == "UNKNOWN":
            # AI domain classifier — single tiny YES/NO prompt, no keyword list
            try:
                clf_result = ai_service.generate(
                    prompt=message,
                    system=DOMAIN_CLASSIFIER_PROMPT,
                    feature_name="chatbot_v2_domain_clf"
                )
                is_recruitment = clf_result.content.strip().upper().startswith("YES")
            except Exception:
                is_recruitment = False

            if is_recruitment:
                history = memory.get_history(session_id, limit=6)
                prompt = self._build_prompt(message, history, {})
                try:
                    result = ai_service.generate(
                        prompt=prompt,
                        system=UNKNOWN_AI_SYSTEM,
                        feature_name="chatbot_v2_unknown"
                    )
                    reply = result.content.strip()
                    if reply:
                        memory.add(session_id, {"role": "user", "content": message})
                        memory.add(session_id, {"role": "assistant", "content": reply[:500]})
                        return self._resp(reply, "UNKNOWN", session_id)
                except Exception as e:
                    logger.error(f"ChatbotV2 UNKNOWN AI error | {e}")

            return self._resp(FALLBACK_UNKNOWN, "UNKNOWN", session_id)

        # ── Step 4: Rule-based replies — no AI, no memory needed ──────────────
        if intent == "GREETING":
            return self._resp(_local_greeting(), "GREETING", session_id)

        if intent == "SMALL_TALK":
            return self._resp(_local_small_talk(message), "SMALL_TALK", session_id)

        # ── Step 5: Conversation Memory ───────────────────────────────────────
        history = memory.get_history(session_id, limit=10)

        # ── Step 6: Agent Router ──────────────────────────────────────────────
        try:
            reply = await self._route(intent, message, entities, history, session_id, user_id, user_role)
        except Exception as e:
            logger.error(f"ChatbotV2 route error | session={session_id} | {e}")
            reply = FALLBACK_ERROR

        # ── Step 7: Response Validation ───────────────────────────────────────
        if not reply or not reply.strip():
            reply = FALLBACK_ERROR

        # ── Step 8: Store in Memory ───────────────────────────────────────────
        memory.add(session_id, {"role": "user", "content": message})
        memory.add(session_id, {"role": "assistant", "content": reply[:500]})

        return self._resp(reply, intent, session_id, entities)

    # ── Router ────────────────────────────────────────────────────────────────
    async def _route(self, intent: str, message: str, entities: dict, history: list, session_id: str, user_id: Optional[str], user_role: str) -> str:

        # ── CANDIDATE AGENT ───────────────────────────────────────────────────

        if intent == "JOB_SEARCH":
            return await self._handle_job_search(message, entities, history)

        if intent == "SALARY_QUERY":
            return await self._handle_salary(message, entities, history)

        if intent == "ATS_SCORE":
            return await self._handle_ats(message, entities, history)

        if intent in ("RESUME_IMPROVE", "RESUME_PARSE"):
            resume_text = message if len(message) > 200 else ""
            if not resume_text:
                return (
                    "To improve your resume, please paste your resume text directly in the chat "
                    "and I'll analyze it for you.\n\n"
                    "Or ask me specific questions like:\n"
                    "- How do I make my resume ATS-friendly?\n"
                    "- What skills should I add to my resume?\n"
                    "- How do I write a strong resume summary?"
                )
            result = await resume_agent.execute(query=message, user_id=user_id, resume_text=resume_text)
            return result.get("improved_resume") or result.get("reply", FALLBACK_ERROR)

        if intent in ("CAREER_ROADMAP", "SKILL_GAP"):
            result = await career_agent.execute(query=message, user_id=user_id)
            return result.get("advice", FALLBACK_ERROR)

        if intent == "INTERVIEW":
            skills = entities.get("skills", [])
            result = await interview_agent.execute(
                query=message,
                user_id=user_id,
                job_title=message,
                skills=skills,
            )
            return result.get("questions", FALLBACK_ERROR)

        # ── RECRUITER AGENT ───────────────────────────────────────────────────

        if intent == "JD_GENERATION":
            result = await recruiter_agent.execute(
                query=message,
                user_id=user_id,
                title=entities.get("skills", [message])[0] if entities.get("skills") else message,
            )
            return result.get("job_description", FALLBACK_ERROR)

        if intent == "RECRUITER":
            # Tool first: search jobs from DB, AI summarizes
            job_data = search_jobs_tool.run(query=message, limit=5)
            context = self._format_jobs(job_data) if job_data["found"] else ""
            prompt = self._build_prompt(message, history, entities, tool_data=context)
            result = ai_service.generate(
                prompt=prompt,
                system="You are ZyncJobs Recruiter Assistant. Use the provided job/candidate data to help the recruiter. Never invent data.",
                feature_name="chatbot_v2_recruiter"
            )
            return result.content.strip()

        # ── PLATFORM AGENT ────────────────────────────────────────────────────

        if intent in ("COMPANY_FAQ", "PLATFORM_HELP", "ACCOUNT_HELP"):
            return await self._handle_faq(message, intent, history)

        # ── AI-ONLY INTENTS ───────────────────────────────────────────────────

        system = SYSTEM_PROMPTS.get(intent, DEFAULT_SYSTEM)
        prompt = self._build_prompt(message, history, entities)
        result = ai_service.generate(prompt=prompt, system=system, feature_name=f"chatbot_v2_{intent.lower()}")
        return result.content.strip()

    # ── Tool Handlers ─────────────────────────────────────────────────────────

    async def _handle_job_search(self, message: str, entities: dict, history: list) -> str:
        """Tool first → DB → AI formats. Never falls back to general LLM knowledge."""
        skills = entities.get("skills", [])
        location = entities.get("location", "")
        job_type = entities.get("job_type", "")
        # Use longest skill phrase as query (catches "software developer"), else full message
        query = max(skills, key=len) if skills else message
        # Strip location from query if it leaked in
        if location and location.lower() in query.lower():
            query = re.sub(re.escape(location), "", query, flags=re.IGNORECASE).strip()

        job_data = search_jobs_tool.run(query=query, location=location, job_type=job_type, limit=8)

        if not job_data["found"]:
            # AI suggests similar roles on ZyncJobs — never recommends other portals
            no_result_prompt = (
                f"The user searched for: {message}\n"
                f"No matching jobs were found in the ZyncJobs database right now.\n\n"
                f"Respond by:\n"
                f"1. Telling the user no results were found on ZyncJobs for their query.\n"
                f"2. Suggesting 2-3 similar job titles they could search for on ZyncJobs.\n"
                f"3. Offering to refine by location, experience level, or job type.\n"
                f"Do NOT mention any other job portals."
            )
            result = ai_service.generate(
                prompt=no_result_prompt,
                system=JOB_SEARCH_SYSTEM,
                feature_name="chatbot_v2_job_search"
            )
            return result.content.strip()

        jobs_text = self._format_jobs(job_data)
        prompt = (
            f"The user asked: {message}\n\n"
            f"Here are the REAL jobs from the ZyncJobs database — present ONLY these:\n"
            f"{jobs_text}\n\n"
            f"Format each job clearly with its title, company, location, type, and skills. "
            f"Do NOT add any jobs not listed above. Do NOT mention other job portals."
        )
        result = ai_service.generate(
            prompt=prompt,
            system=JOB_SEARCH_SYSTEM,
            feature_name="chatbot_v2_job_search"
        )
        return result.content.strip()

    async def _handle_salary(self, message: str, entities: dict, history: list) -> str:
        """Tool first → DB salary data → AI explains"""
        import re as _re
        skills = entities.get("skills", [])
        # Try to extract a clean job title from the message
        title_match = _re.search(
            r'(?:salary|pay|compensation|ctc|package|earn|make)\s+(?:for|of|a|an)?\s*([a-zA-Z][\w\s]+?)(?:\?|$|\bin\b|\bfor\b|\bat\b)',
            message, _re.IGNORECASE
        )
        if title_match:
            title = title_match.group(1).strip()
        elif skills:
            title = skills[0]
        else:
            title = message.strip()

        data = salary_tool.run(title=title)

        if not data.get("found"):
            prompt = self._build_prompt(message, history, entities)
            result = ai_service.generate(
                prompt=prompt,
                system="You are ZyncJobs Salary Expert. Provide general salary guidance since no specific data was found.",
                feature_name="chatbot_v2_salary"
            )
            return result.content.strip()

        salary_context = (
            f"Real salary data from ZyncJobs for '{data.get('title', title)}':\n"
            f"- Average Salary: {data.get('avgSalary', 'N/A'):,}\n"
            f"- Range: {data.get('avgMin', 'N/A'):,} - {data.get('avgMax', 'N/A'):,}\n"
            f"- Market Min: {data.get('marketMin', 'N/A'):,} | Market Max: {data.get('marketMax', 'N/A'):,}\n"
            f"- Total Jobs with salary data: {data.get('totalJobs', 0)}\n"
        )
        if data.get("byLevel"):
            salary_context += "By Experience Level:\n"
            for level, info in data["byLevel"].items():
                salary_context += f"  {level}: avg {info['avg']:,} ({info['count']} jobs)\n"

        prompt = (
            f"User asked: {message}\n\n"
            f"{salary_context}\n"
            f"Explain this salary data clearly and give negotiation tips."
        )
        result = ai_service.generate(
            prompt=prompt,
            system="You are ZyncJobs Salary Insights Expert. Use only the provided real data.",
            feature_name="chatbot_v2_salary"
        )
        return result.content.strip()

    async def _handle_ats(self, message: str, entities: dict, history: list) -> str:
        """RAG → AI explains ATS using ZyncJobs knowledge base"""
        from app.knowledge.knowledge_base import knowledge_base
        context = knowledge_base.build_context("ATS score resume improvement keywords", max_chars=2000)
        prompt = self._build_prompt(message, history, entities, tool_data=context)
        result = ai_service.generate(
            prompt=prompt,
            system="You are ZyncJobs ATS Expert. Explain ATS scores, keyword matching, and how to improve resume visibility on ZyncJobs.",
            feature_name="chatbot_v2_ats"
        )
        return result.content.strip()

    async def _handle_faq(self, message: str, intent: str, history: list) -> str:
        """RAG first → AI answers using context. Always answers — never dead-ends."""
        from app.knowledge.knowledge_base import knowledge_base
        context = knowledge_base.build_context(message, max_chars=3000)

        if context:
            prompt = (
                f"Answer the user's question using the provided ZyncJobs documentation.\n\n"
                f"Documentation:\n{context}\n\n"
                f"User question: {message}\n\n"
                f"Answer clearly and professionally based on the documentation above."
            )
        else:
            prompt = self._build_prompt(message, history, {})

        result = ai_service.generate(
            prompt=prompt,
            system=SYSTEM_PROMPTS.get(intent, DEFAULT_SYSTEM),
            feature_name=f"chatbot_v2_{intent.lower()}"
        )
        return result.content.strip()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_jobs(self, job_data: dict) -> str:
        if not job_data.get("found") or not job_data.get("jobs"):
            return ""
        lines = [f"Found {job_data['count']} jobs:\n"]
        for i, j in enumerate(job_data["jobs"], 1):
            line = f"{i}. {j['title']} at {j['company']} — {j['location']}"
            if j.get("salary"):
                line += f" | {j['salary']}"
            if j.get("type"):
                line += f" | {j['type']}"
            if j.get("skills"):
                line += f"\n   Skills: {', '.join(j['skills'])}"
            lines.append(line)
        return "\n".join(lines)

    def _build_prompt(self, message: str, history: list, entities: dict, tool_data: str = "") -> str:
        parts = []
        if history:
            parts.append("Conversation history:")
            for msg in history[-6:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                parts.append(f"{role}: {msg['content'][:300]}")
            parts.append("")
        if entities:
            parts.append(f"Extracted context: {', '.join(f'{k}: {v}' for k, v in entities.items())}")
            parts.append("")
        if tool_data:
            parts.append(f"Data from ZyncJobs:\n{tool_data}")
            parts.append("")
        parts.append(f"User: {message}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _resp(self, message: str, intent: str, session_id: str, entities: dict = None) -> dict:
        return {
            "reply": message,
            "intent": intent,
            "session_id": session_id,
            "entities": entities or {},
            "is_fallback": intent in ("UNKNOWN", "BLOCKED"),
        }


chatbot_v2 = ChatbotV2()
