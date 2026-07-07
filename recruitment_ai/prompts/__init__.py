"""Jinja2 prompt templates for all brains."""
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

PROMPTS_DIR = Path(__file__).parent
env = Environment(loader=FileSystemLoader(str(PROMPTS_DIR)))


TEMPLATES = {
    "chatbot_system": """You are the official ZyncJobs AI Assistant — a helpful, friendly recruitment platform assistant.
Use the provided context to answer questions about ZyncJobs features, pricing, and how things work.
If the context doesn't fully answer the question, use your general knowledge to help.
Be concise, friendly, and helpful. Use bullet points when listing multiple items.""",

    "chatbot_prompt": """User question: {{ query }}

Relevant ZyncJobs information:
{{ context }}

Answer the user's question helpfully using the above context. Be concise and friendly.""",

    "job_parser_system": """Extract structured information from job descriptions into valid JSON.
Only return valid JSON. No extra text.""",

    "jd_generator_template": """Generate a professional job description for:
Title: {{ title }}
Company: {{ company }}
Location: {{ location }}
Experience Level: {{ experience_level }}
Skills: {{ skills }}

Include: About, Responsibilities, Requirements, Benefits""",

    "ats_system": """Analyze resumes against job descriptions for ATS compatibility.
Return valid JSON with ats_score, keyword_match, suggestions.""",

    "career_system": """Provide actionable career advice and roadmaps.
Return valid JSON with career_path, skill_gaps, recommendations.""",

    "interview_system": """Generate relevant interview questions for given roles.
Return valid JSON with questions, topics, tips.""",
}


def get_prompt(name: str, **kwargs) -> str:
    template_str = TEMPLATES.get(name)
    if not template_str:
        return ""
    from jinja2 import Template
    return Template(template_str).render(**kwargs)


def get_system_prompt(name: str) -> str:
    return TEMPLATES.get(f"{name}_system", "")
