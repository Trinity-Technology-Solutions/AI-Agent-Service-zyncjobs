"""Chatbot Brain - RAG-based chatbot using vector store."""
import re
from recruitment_ai.shared.brain import Brain, BrainState
from recruitment_ai.shared.ollama_service import ollama_service
from recruitment_ai.vector.store import vector_store
from recruitment_ai.prompts import get_prompt, get_system_prompt

# Greetings and small talk — answer directly, no RAG needed
GREETING_PATTERNS = re.compile(
    r"^(hi|hello|hey|hii|helo|hai|good\s*(morning|afternoon|evening|night)|"
    r"what'?s up|howdy|greetings|sup|yo|namaste|vanakkam|how are you|"
    r"who are you|what are you|what can you do|help me|help)[\s!?.]*$",
    re.IGNORECASE
)

GREETING_REPLY = """Hi! 👋 I'm the **ZyncJobs AI Assistant**.

I can help you with:
- 🔍 **Job Search** — Find jobs matching your skills
- 📄 **Resume** — Build, parse, or get ATS score
- 🎯 **Career Advice** — Roadmaps, skill gaps, growth tips
- 🎤 **Interview Prep** — Mock questions and tips
- 💼 **For Employers** — Post jobs, find candidates, generate JDs
- ℹ️ **ZyncJobs Platform** — Pricing, features, how it works

What would you like help with today?"""


class ChatbotBrain(Brain):
    """RAG-based chatbot for ZyncJobs platform questions."""

    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainState:
        query = (state.query or "").strip()

        if not query:
            state.result = {"reply": "How can I help you with ZyncJobs?", "sources": []}
            return state

        # Handle greetings directly — no RAG
        if GREETING_PATTERNS.match(query):
            state.result = {"reply": GREETING_REPLY, "sources": [], "intent": "CHAT"}
            return state

        # Search vector store for ZyncJobs-specific knowledge
        context_docs = await vector_store.search(query, top_k=5)

        # If no relevant docs found — use LLM general knowledge about ZyncJobs
        if not context_docs:
            reply = await self._general_answer(query)
            state.result = {"reply": reply, "sources": [], "intent": "CHAT"}
            return state

        context = self._build_context(context_docs)
        reply = await self._generate_answer(query, context)
        sources = self._extract_sources(context_docs)

        state.result = {"reply": reply, "sources": sources, "intent": "ANSWERED"}
        return state

    def _build_context(self, docs: list) -> str:
        parts = []
        for d in docs:
            meta = d.metadata
            parts.append(f"[{meta.get('title', 'ZyncJobs')}]\n{d.text}\n")
        context = "\n\n".join(parts)
        return context[:1500] + "..." if len(context) > 1500 else context

    def _extract_sources(self, docs: list) -> list[dict]:
        return [
            {
                "title": d.metadata.get("title", "ZyncJobs"),
                "url": d.metadata.get("url", ""),
                "category": d.metadata.get("category", ""),
            }
            for d in docs
        ]

    async def _generate_answer(self, query: str, context: str) -> str:
        prompt = get_prompt("chatbot_prompt", query=query, context=context)
        system = get_system_prompt("chatbot")
        try:
            return await ollama_service.generate(
                brain_name="chatbot",
                prompt=prompt,
                system=system,
                temperature=0.2,
                max_tokens=400,
            )
        except Exception:
            return "I'm having trouble processing your request right now. Please try again."

    async def _general_answer(self, query: str) -> str:
        """Answer general questions about ZyncJobs using LLM knowledge."""
        system = """You are the ZyncJobs AI Assistant — a helpful recruitment platform assistant.
ZyncJobs is an AI-powered job portal connecting candidates and employers in India and globally.
Answer helpfully and concisely. If you don't know something specific about ZyncJobs, say so and suggest contacting support."""
        try:
            return await ollama_service.generate(
                brain_name="chatbot",
                prompt=query,
                system=system,
                temperature=0.3,
                max_tokens=400,
            )
        except Exception:
            return "I'm having trouble connecting right now. Please try again in a moment."


chatbot_brain = ChatbotBrain()
