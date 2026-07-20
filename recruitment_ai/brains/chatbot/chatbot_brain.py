"""Chatbot Brain — full enterprise RAG chatbot.
Pipeline: BrainState → Memory → User Context → RAG → Prompt Builder → LLM → Validators → BrainResult
"""
import re
import time
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service
from recruitment_ai.prompts import get_prompt, get_system_prompt
from recruitment_ai.validators.response_validator import validate_response
from recruitment_ai.validators.citation_validator import deduplicate_citations

GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|hii|helo|hai|good\s*(morning|afternoon|evening|night)|"
    r"what'?s up|howdy|greetings|sup|yo|namaste|vanakkam|how are you|"
    r"who are you|what are you|what can you do|help me|help)[\s!?.]*$",
    re.IGNORECASE
)

GREETING_REPLY = """Hi! I'm the **ZyncJobs AI Assistant**.

I can help you with:
- **Job Search** — Find jobs matching your skills
- **Resume** — Build, parse, or get ATS score
- **Career Advice** — Roadmaps, skill gaps, growth tips
- **Interview Prep** — Mock questions and tips
- **For Employers** — Post jobs, find candidates, generate JDs
- **ZyncJobs Platform** — Pricing, features, how it works

What would you like help with today?"""


class ChatbotBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainResult:
        start = time.perf_counter()
        query = state.request.query or state.query or ""
        query = query.strip()

        if not query:
            return BrainResult(response={"reply": "How can I help you with ZyncJobs?", "sources": []})

        if GREETING_PATTERNS.match(query):
            return BrainResult(response={"reply": GREETING_REPLY, "sources": []})

        rag_chunks = state.retrieved_documents.chunks or []

        if not rag_chunks:
            history_list = state.context_data.user_preferences.get("history", [])
            system_override = state.context_data.user_preferences.get("systemPrompt")
            reply = await self._general_answer(query, history_list, system_override)
            return BrainResult(
                response={"reply": reply, "sources": [], "intent": "ANSWERED"},
                execution_time=time.perf_counter() - start,
            )

        user_profile = self._build_user_profile(state)
        conversation_history = self._build_conversation_history(state)
        context = self._build_context(rag_chunks)
        reply = await self._generate_answer(query, context, conversation_history, user_profile)
        citations = self._build_citations(rag_chunks)

        result = BrainResult(
            response={"reply": reply, "sources": citations, "intent": "ANSWERED"},
            citations=citations,
            execution_time=time.perf_counter() - start,
        )
        state.response = result.response

        return result

    def _build_user_profile(self, state: BrainState) -> str:
        parts = []
        if state.user.name:
            parts.append(f"Name: {state.user.name}")
        if state.user.role:
            parts.append(f"Role: {state.user.role}")
        resume = state.context_data.resume
        if resume.skills:
            parts.append(f"Skills: {', '.join(resume.skills[:10])}")
        if resume.parsed:
            parts.append("Resume: Available")
        return "\n".join(parts)

    def _build_conversation_history(self, state: BrainState) -> str:
        history = state.conversation.history or state.memory or []
        if not history:
            return ""
        recent = [f"{m['role']}: {m['content'][:200]}" for m in history[-4:]]
        return "\n".join(recent)

    def _build_context(self, chunks: list) -> str:
        parts = []
        for c in chunks:
            text = c.get("text", "") or c.get("content", "")
            if not text or not isinstance(text, str):
                continue
            title = c.get("title", "ZyncJobs")
            parts.append(f"[{title}]\n{text}\n")
        ctx = "\n\n".join(parts)
        return ctx[:1500] + "..." if len(ctx) > 1500 else ctx

    def _build_citations(self, chunks: list) -> list[dict]:
        citations = []
        for c in chunks:
            text = c.get("text", "") or c.get("content", "")
            title = c.get("title", "ZyncJobs")
            url = c.get("url", "")
            if not text:
                continue
            citations.append({
                "title": title,
                "url": url or "",
                "snippet": text[:120],
            })
        return deduplicate_citations(citations)

    async def _generate_answer(self, query: str, context: str, conversation: str, user_profile: str) -> str:
        prompt = get_prompt("chatbot_prompt",
            query=query,
            context=context,
            conversation_history=conversation,
            user_profile=user_profile,
        )
        system = get_system_prompt("chatbot")
        try:
            return await llm_service.generate(
                brain_name="chatbot",
                prompt=prompt,
                system=system,
                temperature=0.2,
                max_tokens=400,
            )
        except Exception:
            return "I'm having trouble processing your request right now. Please try again."

    async def _general_answer(self, query: str, history: list, system_override: str | None = None) -> str:
        system = system_override or get_system_prompt("chatbot")
        history_text = ""
        for turn in history[-6:]:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            history_text += f"{role}: {content}\n"
        prompt = query
        if history_text:
            prompt = f"Previous conversation:\n{history_text}\n\nUser: {query}"
        try:
            return await llm_service.generate(
                brain_name="chatbot",
                prompt=prompt,
                system=system,
                temperature=0.3,
                max_tokens=400,
            )
        except Exception:
            return "I'm having trouble connecting right now. Please try again in a moment."


chatbot_brain = ChatbotBrain()
