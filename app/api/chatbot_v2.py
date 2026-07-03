from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional
from app.agents.chatbot_v2 import chatbot_v2
from app.memory.conversation import ConversationMemory
from app.utils.logger import logger
import uuid

router = APIRouter()
_memory = ConversationMemory()


class ChatbotV2Request(BaseModel):
    message: str
    session_id: Optional[str] = Field(default=None)
    user_id: Optional[str] = None
    user_role: Optional[str] = "candidate"


class SourceInfo(BaseModel):
    title: str = ""
    url: str = ""
    category: str = ""

class ChatbotV2Response(BaseModel):
    reply: str
    intent: str
    session_id: str
    sources: list[SourceInfo] = []
    is_fallback: bool = False


@router.post("", response_model=ChatbotV2Response)
async def chatbot_v2_endpoint(request: ChatbotV2Request):
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"ChatbotV2 request | session={session_id} | user={request.user_id}")

    result = await chatbot_v2.handle(
        message=request.message,
        session_id=session_id,
        user_id=request.user_id,
        user_role=request.user_role or "candidate",
    )
    return ChatbotV2Response(**result)


@router.delete("/{session_id}/history")
async def clear_history(session_id: str):
    _memory.clear(session_id)
    return {"success": True, "session_id": session_id}
