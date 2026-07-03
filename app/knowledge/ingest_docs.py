"""Script to ingest all .md docs into ChromaDB with proper metadata."""
from pathlib import Path
from app.knowledge.ingest import ChromaIngester
from app.utils.logger import logger

DOCS_DIR = Path(__file__).parent / "docs"

# Map filename -> (category, title)
DOC_META = {
    "about_zyncjobs.md": ("about", "About ZyncJobs"),
    "ats_resume.md": ("ats", "ATS Score and Resume Analysis"),
    "ats_score.md": ("ats", "ATS Score on ZyncJobs"),
    "candidate_features.md": ("candidate", "Candidate Features on ZyncJobs"),
    "career_resources.md": ("career", "Career Resources on ZyncJobs"),
    "career_roadmap.md": ("career", "Career Roadmap on ZyncJobs"),
    "career_tools.md": ("career", "Career Tools on ZyncJobs"),
    "employer_features.md": ("employer", "Employer Features on ZyncJobs"),
    "faq.md": ("faq", "ZyncJobs Frequently Asked Questions"),
    "how_to_apply.md": ("candidate", "How to Apply for Jobs on ZyncJobs"),
    "interview_prep.md": ("candidate", "Interview Preparation on ZyncJobs"),
    "interview_preparation.md": ("candidate", "Interview Preparation on ZyncJobs"),
    "recruiter_guide.md": ("employer", "Recruiter Guide on ZyncJobs"),
    "resume_builder.md": ("candidate", "Resume Builder on ZyncJobs"),
    "skill_assessment.md": ("candidate", "Skill Assessment on ZyncJobs"),
}

ingester = ChromaIngester()
total = 0

for fname, (category, title) in DOC_META.items():
    fpath = DOCS_DIR / fname
    if not fpath.exists():
        logger.warn(f"File not found: {fpath}")
        continue
    text = fpath.read_text(encoding="utf-8")
    url = f"/docs/{fname.replace('.md', '')}"
    n = ingester.ingest_page(text=text, url=url, title=title, category=category)
    total += n

logger.info(f"Done. Ingested {total} chunks from {len(DOC_META)} doc files.")
logger.info(f"Total chunks in collection: {ingester.count}")
