"""Memory manager — unified orchestration for workflow nodes.
load_memory(state)  → populates state.memory + state.context["history"]
store_memory(state) → persists the current conversation turn

Workflow nodes call memory_manager instead of talking to Redis/DB directly.
"""
import logging
from typing import Optional
from recruitment_ai.brains.base import BrainState
from recruitment_ai.memory.conversation_memory import conversation_memory, CHAT_INTENTS
from recruitment_ai.memory.user_memory import user_memory

logger = logging.getLogger(__name__)


class MemoryManager:
    """Orchestrates conversation + user memory. Called by load_memory_node and store_memory_node."""

    async def load_memory(self, state: BrainState) -> BrainState:
        """Load conversation history + user preferences into BrainState."""
        history = await conversation_memory.load(state.session_id)
        if history:
            state.memory = history
            state.context["history"] = history
            state.metadata["history_turns"] = len(history)

        prefs = await user_memory.get_preferences(state.user_id)
        if prefs:
            state.context["user_preferences"] = prefs

        return state

    async def store_memory(self, state: BrainState) -> None:
        """Store the current conversation turn if it's a conversational intent."""
        if not state.session_id or not state.query or not state.result:
            return

        intent = state.intent or ""
        if intent not in CHAT_INTENTS:
            return

        await conversation_memory.store(
            session_id=state.session_id,
            user_id=state.user_id,
            query=state.query,
            result=state.result,
            intent=intent,
        )

    async def summarize_conversation(self, session_id: str) -> Optional[str]:
        """Summarize a long conversation for context injection."""
        return await conversation_memory.summarize(session_id)


memory_manager = MemoryManager()
