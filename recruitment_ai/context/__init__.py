"""Context Manager — loads user, resume, job, company, and assessment data into BrainState.
Every brain automatically receives the same context without loading data itself.
"""
from recruitment_ai.context.context_manager import context_manager
from recruitment_ai.context.user_context import user_context
from recruitment_ai.context.resume_context import resume_context
from recruitment_ai.context.job_context import job_context
from recruitment_ai.context.company_context import company_context
from recruitment_ai.context.assessment_context import assessment_context

__all__ = [
    "context_manager",
    "user_context",
    "resume_context",
    "job_context",
    "company_context",
    "assessment_context",
]
