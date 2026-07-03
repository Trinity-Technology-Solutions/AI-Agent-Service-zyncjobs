"""
ZyncJobs Chatbot V2 — RAG-only pipeline
Flow:
  User → Safety → RAG Search → Found? → AI formats from context : Domain Redirect
  No intent classification needed. Only GREETING gets special handling.
"""
import random
from typing import Optional
from app.middleware.safety import validate, is_off_topic, OFF_TOPIC_MESSAGE
from app.knowledge.ingest import ingester
from app.knowledge.reranker import rerank
from app.services.ai_service import ai_service
from app.memory.conversation import ConversationMemory
from app.utils.logger import logger

# ChromaDB returns cosine DISTANCE (0 = identical). Lower = better match.
MAX_DISTANCE = 0.5
FALLBACK_MAX_DISTANCE = 0.7

memory = ConversationMemory()

GREETING_REPLIES = [
    "Hi! I'm the **ZyncJobs AI Assistant**.\n\nI can answer questions about:\n- Jobs on ZyncJobs\n- Resume building and ATS scores\n- Interview preparation\n- Career guidance\n- ZyncJobs platform features\n\nWhat can I help you with?",
    "Hello! Welcome to ZyncJobs AI.\n\nAsk me anything about ZyncJobs, our platform, jobs, resumes, or career tools.\n\nHow can I assist you?",
    "Hey there! I'm your ZyncJobs AI Assistant.\n\nTry asking:\n- How do I create a resume?\n- What jobs are available?\n- How does ATS scoring work?\n\nWhat would you like help with?",
]

GREETING_KEYWORDS = [
    "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
    "howdy", "what's up", "sup", "greetings", "namaste",
]

DOMAIN_REDIRECT = (
    "I'm the ZyncJobs AI Assistant. I can only answer questions about ZyncJobs "
    "— our platform, jobs, resumes, employers, and recruitment features. "
    "Please ask something related to ZyncJobs!"
)

SYSTEM_PROMPT = (
    "You are the official ZyncJobs AI Assistant. "
    "Answer ONLY using the provided context from the ZyncJobs website. "
    "If the answer is not present in the context, reply: "
    "\"I couldn't find that information in the current ZyncJobs knowledge base.\" "
    "Do not use your own knowledge. Do not guess. Do not invent. "
    "When you use context, cite the source URL and page title from the metadata."
)


class ChatbotV2:

    async def handle(
        self, message: str, session_id: str,
        user_id: Optional[str] = None, user_role: str = "candidate",
    ) -> dict:

        # ── Step 1: Input Validation ───────────────────────────────────────────
        validation = validate(message)
        if not validation["valid"]:
            return self._resp(validation["message"], "BLOCKED", session_id)

        # ── Step 2: Off-topic Guardrail (quick regex before RAG) ───────────────
        if is_off_topic(message):
            return self._resp(DOMAIN_REDIRECT, "OFF_TOPIC", session_id)

        # ── Step 3: Greeting (local, no AI) ────────────────────────────────────
        lower = message.lower().strip()
        if any(kw in lower for kw in GREETING_KEYWORDS) and len(lower.split()) <= 4:
            return self._resp(random.choice(GREETING_REPLIES), "GREETING", session_id)

        # ── Step 4: RAG Search ─────────────────────────────────────────────────
        docs = ingester.search(query=message, top_k=5)

        # ── Step 5: Rerank ─────────────────────────────────────────────────────
        docs = rerank(message, docs, top_k=3)

        # ── Step 6: Distance Threshold Check ────────────────────────────────────
        # ChromaDB returns cosine DISTANCE (0=identical, higher=less similar).
        # Max distance 0.5 means at least ~85% similarity.
        if not docs or docs[0].score > MAX_DISTANCE:
            # Try broader search
            docs = ingester.search(query=message, top_k=10)
            docs = rerank(message, docs, top_k=3)
            if not docs or docs[0].score > FALLBACK_MAX_DISTANCE:
                reply = self._build_no_context_response(message)
                memory.add(session_id, {"role": "user", "content": message})
                memory.add(session_id, {"role": "assistant", "content": reply[:500]})
                return self._resp(reply, "NO_MATCH", session_id)

        # ── Step 7: Build Context from matched docs ────────────────────────────
        context_parts = []
        sources = []
        for doc in docs:
            meta = doc.metadata or {}
            title = meta.get("title", "ZyncJobs")
            url = meta.get("url", "")
            category = meta.get("category", "")
            context_parts.append(f"[{title}]\n{doc.text}\n")
            sources.append({"title": title, "url": url, "category": category})

        # Trim context to ~1500 chars for speed
        context = "\n\n".join(context_parts)
        if len(context) > 1500:
            context = context[:1500] + "..."

        prompt = (
            f"User question: {message}\n\n"
            f"Context from ZyncJobs website:\n{context}\n\n"
            f"Answer the user using ONLY the above context. Be concise (2-3 sentences). "
            f"Cite the source page title when you use information from it."
        )

        try:
            result = ai_service.generate(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                feature_name="chatbot_v2"
            )
            reply = result.content.strip()
        except Exception as e:
            logger.error(f"ChatbotV2 AI error | {e}")
            reply = "I'm having trouble processing your request right now. Please try again."

        # ── Step 9: Store in Memory ────────────────────────────────────────────
        memory.add(session_id, {"role": "user", "content": message})
        memory.add(session_id, {"role": "assistant", "content": reply[:500]})

        return {
            "reply": reply,
            "intent": "ANSWERED",
            "session_id": session_id,
            "sources": sources,
            "is_fallback": False,
        }

    def _build_no_context_response(self, message: str) -> str:
        return (
            "I couldn't find that information in the current ZyncJobs knowledge base. "
            "I can only answer based on what's available on the ZyncJobs website. "
            "Try asking about our platform features, jobs, resumes, or employer tools."
        )

    def _resp(self, message: str, intent: str, session_id: str) -> dict:
        return {
            "reply": message,
            "intent": intent,
            "session_id": session_id,
            "sources": [],
            "is_fallback": intent in ("BLOCKED", "OFF_TOPIC", "NO_MATCH"),
        }


chatbot_v2 = ChatbotV2()
