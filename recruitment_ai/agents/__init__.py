"""Agent layer — matches roadmap structure.
Each agent wraps a brain from the brains/ package.
Architecture doc: agents/supervisor → agents/resume_agent → brains/candidate/resume_parser_brain
"""

from recruitment_ai.agents.supervisor import supervisor_agent
from recruitment_ai.agents.resume_agent import resume_agent
from recruitment_ai.agents.recruiter_agent import recruiter_agent
from recruitment_ai.agents.chatbot_agent import chatbot_agent
from recruitment_ai.agents.ats_agent import ats_agent
from recruitment_ai.agents.jd_generator_agent import jd_generator_agent
from recruitment_ai.agents.parser_agent import parser_agent
from recruitment_ai.agents.skill_gap_agent import skill_gap_agent
from recruitment_ai.agents.roadmap_agent import roadmap_agent

__all__ = [
    "supervisor_agent",
    "resume_agent",
    "recruiter_agent",
    "chatbot_agent",
    "ats_agent",
    "jd_generator_agent",
    "parser_agent",
    "skill_gap_agent",
    "roadmap_agent",
]
