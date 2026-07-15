"""ATS agent — resume scoring against job descriptions.
Wraps ATSBrain and JobMatchingBrain from brains.candidate.
"""
from recruitment_ai.brains.candidate.ats_brain import ats_brain
from recruitment_ai.brains.candidate.job_matching_brain import job_matching_brain

ats_agent = {
    "scanner": ats_brain,
    "matcher": job_matching_brain,
}
