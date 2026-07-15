"""Skill Gap agent — analyze missing skills for target roles.
Wraps SkillGapBrain from brains.candidate.
"""
from recruitment_ai.brains.candidate.skill_gap_brain import skill_gap_brain

skill_gap_agent = skill_gap_brain
