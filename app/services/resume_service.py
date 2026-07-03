from typing import Optional
from app.agents.resume_agent import ResumeAgent
from app.tools.resume_parser import ResumeParserTool
from app.tools.ats_tool import ATSTool
from app.tools.ai_grammar_tool import AIGrammarTool
from app.resume.builder.json_builder import JsonBuilder
from app.resume.scorer.ats_score import ATSScorer
from app.resume.intelligence.intelligence_service import ResumeIntelligenceService
from app.utils.logger import logger

resume_agent = ResumeAgent()
parser = ResumeParserTool()
ats_tool = ATSTool()
ats_scorer = ATSScorer()
intelligence_service = ResumeIntelligenceService()
grammar_tool = AIGrammarTool()
json_builder = JsonBuilder()


async def handle(query: str, user_id: Optional[str] = None, **kwargs) -> dict:
    logger.info("ResumeService.handle")
    resume_text = kwargs.get("resume_text", "")
    job_description = kwargs.get("job_description", "")
    result = await resume_agent.execute(
        query=query, user_id=user_id,
        resume_text=resume_text,
        job_description=job_description,
    )
    return {
        "improved_resume": result.get("improved_resume", ""),
        "ats_score": result.get("ats_score"),
        "summary": result.get("summary", ""),
        "skills_suggested": result.get("skills_suggested", []),
        "grammar_issues": result.get("grammar_issues", []),
    }


async def improve_resume(resume_text: str, job_description: str = ""):
    return await handle(
        query="Improve my resume",
        resume_text=resume_text,
        job_description=job_description,
    )


def parse_resume(resume_text: str):
    logger.info("ResumeService.parse_resume")
    sections = parser.run(resume_text)
    return {
        "contact": sections.get("contact", ""),
        "summary": sections.get("summary", ""),
        "experience": sections.get("experience", ""),
        "education": sections.get("education", ""),
        "skills": sections.get("skills", ""),
    }


async def ats_score_v2(resume_text: str, job_description: str):
    """Hybrid ATS Score — Rule 70% + AI 30%."""
    logger.info("ResumeService.ats_score_v2")
    resume = _text_to_resume_dict(resume_text)
    result = ats_scorer.score(resume, job_description)
    return result


def _text_to_resume_dict(text: str) -> dict:
    """Convert raw text to minimal resume dict for the scorer."""
    from app.resume.parsers.contact_parser import ContactParser
    from app.resume.parsers.skill_parser import SkillParser
    from app.resume.detector.section_detector import SectionDetector
    detector = SectionDetector()
    sections = detector.detect(text)
    contact_parser = ContactParser()
    contact = contact_parser.parse(sections.get("contact", ""))
    skill_parser = SkillParser()
    skills = skill_parser.parse(sections.get("skills", ""))
    return {
        "profile": contact,
        "summary": sections.get("summary", ""),
        "experience": _parse_section_to_entries(sections.get("experience", "")),
        "education": _parse_section_to_entries(sections.get("education", "")),
        "skills": skills,
        "projects": _parse_section_to_entries(sections.get("projects", "")),
        "certifications": _parse_section_to_entries(sections.get("certifications", "")),
        "achievements": _parse_section_to_entries(sections.get("achievements", "")),
        "languages": _parse_section_to_entries(sections.get("languages", "")),
    }


def _parse_section_to_entries(text: str) -> list[dict]:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    return [{"text": l} for l in lines[:20]]


def analyze_intelligence(resume_json: dict) -> dict:
    """Resume Intelligence — Hybrid completeness + AI analysis."""
    logger.info("ResumeService.analyze_intelligence")
    return intelligence_service.analyze(resume_json)


def grammar_check(text: str, mode: str = "check") -> dict:
    """Grammar Checker — Pure AI via Ollama."""
    logger.info("ResumeService.grammar_check")
    return grammar_tool.run(text, mode=mode)


def hybrid_parse_resume(resume_text: str, layout_blocks: Optional[list[dict]] = None):
    logger.info("ResumeService.hybrid_parse_resume")
    layout = None
    if layout_blocks:
        logger.info(f"ResumeService.hybrid_parse_resume — using {len(layout_blocks)} layout blocks")
        max_x = max((b.get("x1", 0) for b in layout_blocks), default=595.0)
        page_width = max(max_x, 595.0)
        layout = json_builder.layout.analyze_blocks(layout_blocks, page_width)
    result = json_builder.build(resume_text, layout=layout)
    # DEBUG: Log what we're returning to the backend
    logger.info("=== SERVICE OUTPUT ===")
    logger.info(f"  profile.name: {result.get('profile', {}).get('name', 'MISSING')}")
    logger.info(f"  experience: {len(result.get('experience', []))} entries")
    logger.info(f"  education: {len(result.get('education', []))} entries")
    logger.info(f"  skills keys: {list(result.get('skills', {}).keys())}")
    logger.info("======================")
    return result