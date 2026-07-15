"""Base brain framework — re-exports from brains/shared for backward compatibility."""
from recruitment_ai.brains.base.brain import Brain
from recruitment_ai.brains.shared.brain_state import BrainState
from recruitment_ai.brains.shared.brain_result import BrainResult
from recruitment_ai.brains.base.brain_registry import BrainRegistry

# Backward-compat aliases for renamed types
from recruitment_ai.brains.shared.brain_state import (
    UserContext as UserInfo,
    ResumeContext as ResumeInfo,
    JobContext as JobInfo,
    CompanyContext as CompanyInfo,
)

__all__ = [
    "Brain", "BrainState", "BrainResult", "BrainRegistry",
    "UserInfo", "ResumeInfo", "JobInfo", "CompanyInfo",
]
