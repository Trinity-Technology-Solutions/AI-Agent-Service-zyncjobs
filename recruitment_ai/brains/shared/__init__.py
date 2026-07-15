"""Enterprise shared types — BrainState, BrainResult, BrainMetadata.
Every brain, node, and service uses these types.
"""
from recruitment_ai.brains.shared.brain_state import (
    BrainState,
    UserContext,
    SessionInfo,
    ConversationMemory,
    ContextData,
    RetrievedDocuments,
    ProviderInfo,
    ExecutionInfo,
    RequestInfo,
    ResumeContext,
    JobContext,
    CompanyContext,
)
from recruitment_ai.brains.shared.brain_result import BrainResult
from recruitment_ai.brains.shared.brain_metadata import ExecutionMetadata

__all__ = [
    "BrainState",
    "BrainResult",
    "ExecutionMetadata",
    "ExecutionInfo",
    "UserContext",
    "SessionInfo",
    "ConversationMemory",
    "ContextData",
    "RetrievedDocuments",
    "ProviderInfo",
    "RequestInfo",
    "ResumeContext",
    "JobContext",
    "CompanyContext",
]
