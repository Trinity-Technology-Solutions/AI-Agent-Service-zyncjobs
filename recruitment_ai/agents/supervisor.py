"""Supervisor agent — orchestrates intent classification → brain routing → execution.
Wraps MasterBrain from brains.master.
"""
from recruitment_ai.brains.master.master_brain import master_brain as _master_brain

supervisor_agent = _master_brain
