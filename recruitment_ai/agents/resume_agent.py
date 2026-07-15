"""Resume agent — parse, build, and edit resumes.
Wraps ResumeParserBrain, ResumeBuilderBrain, ResumeEditBrain from brains.candidate.
"""
from recruitment_ai.brains.candidate.resume_parser_brain import resume_parser_brain
from recruitment_ai.brains.candidate.resume_edit_brain import resume_edit_brain

resume_agent = {
    "parser": resume_parser_brain,
    "editor": resume_edit_brain,
}
