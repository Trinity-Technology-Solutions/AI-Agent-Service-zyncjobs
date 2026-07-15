"""Enterprise BrainState — the single typed state object flowing through every LangGraph node and Brain.

Architecture doc:
  state.user.id              — authenticated user
  state.session.id           — current session
  state.conversation         — turn history + summary (new)
  state.intent               — classified intent string
  state.context_data         — typed context sub-objects (new)
  state.retrieved_documents  — RAG results (new)
  state.provider_info        — LLM provider + model (new)
  state.request              — original query + files (new)
  state.response             — final output (new)

Backward-compat fields preserved:
  state.query, state.user_id, state.session_id, state.result,
  state.memory, state.context, state.provider, state.model
  These continue to work until all nodes/brains migrate to new patterns.
"""
from pydantic import BaseModel


# ═══════════════════════════════════════════════════════════════════════════
# Sub-objects
# ═══════════════════════════════════════════════════════════════════════════

class UserContext(BaseModel):
    id: str | None = None
    email: str | None = None
    role: str = "candidate"
    name: str | None = None


class SessionInfo(BaseModel):
    id: str | None = None
    conversation_id: str | None = None


class ConversationMemory(BaseModel):
    """Loaded conversation turn history + optional summary."""
    history: list[dict] = []
    summary: str | None = None
    turn_count: int = 0


class ResumeContext(BaseModel):
    parsed: dict | None = None
    skills: list[str] = []
    experience: list[dict] = []
    education: list[dict] = []


class JobContext(BaseModel):
    title: str | None = None
    description: str | None = None
    company_name: str | None = None
    skills: list[str] = []


class CompanyContext(BaseModel):
    name: str | None = None
    industry: str | None = None


class ContextData(BaseModel):
    """Structured context. Extra keys allowed for backward compat e.g. context_data["history"]."""
    resume: ResumeContext = ResumeContext()
    job: JobContext = JobContext()
    company: CompanyContext = CompanyContext()
    user_preferences: dict = {}

    model_config = {"extra": "allow"}


class RetrievedDocuments(BaseModel):
    chunks: list[dict] = []
    sources: list[str] = []


class ProviderInfo(BaseModel):
    name: str = ""
    model: str = ""


class ExecutionInfo(BaseModel):
    """Execution tracking — separates workflow metadata from LLM provider info.
    Populated by MasterBrain after each brain.run().
    """
    brain: str = ""
    workflow_id: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    duration_ms: float = 0.0
    retry_count: int = 0
    cache_hit: bool = False


class RequestInfo(BaseModel):
    query: str | None = None
    intent: str | None = None
    file_content: str | None = None
    file_type: str | None = None


# ═══════════════════════════════════════════════════════════════════════════
# BrainState
# ═══════════════════════════════════════════════════════════════════════════

class BrainState(BaseModel):
    """Enterprise state object flowing through every LangGraph node and Brain.

    ── New enterprise fields ───────────────────────────────────────────
    Use these in new code:
        state.user.id
        state.session.id
        state.conversation.history
        state.context_data.resume
        state.retrieved_documents.chunks
        state.provider_info.name
        state.request.query
        state.response

    ── Backward-compat fields ──────────────────────────────────────────
    These remain available until all nodes migrate:
        state.query         (str)
        state.user_id       (str)
        state.session_id    (str)
        state.memory        (list[dict])
        state.context       (dict)
        state.result        (dict | None)
        state.provider      (str)
        state.model         (str)
    """

    # ── New enterprise fields ─────────────────────────────────────────
    user: UserContext = UserContext()
    session: SessionInfo = SessionInfo()
    conversation: ConversationMemory = ConversationMemory()
    intent: str | None = None
    context_data: ContextData = ContextData()
    retrieved_documents: RetrievedDocuments = RetrievedDocuments()
    provider_info: ProviderInfo = ProviderInfo()
    execution: ExecutionInfo = ExecutionInfo()
    metadata: dict = {}
    request: RequestInfo = RequestInfo()
    response: dict | None = None
    error: str | None = None

    # ── Backward-compat fields ────────────────────────────────────────
    query: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    conversation_id: str | None = None
    user_role: str = "candidate"
    file_content: str | None = None
    file_type: str | None = None
    context: dict = {}
    retrieved_context: list[dict] = []
    memory: list[dict] = []
    result: dict | None = None
    provider: str = ""
    model: str = ""

    model_config = {"extra": "allow"}
