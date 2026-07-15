"""Knowledge ingester — scrapes ZyncJobs documentation and ingests into vector store.
Architecture doc: Qdrant + BGE-M3 for RAG.
"""
import json
import logging
import re
from typing import Optional
from recruitment_ai.vector.store import vector_store
from recruitment_ai.llm.router import LLMRouter
llm_router = LLMRouter()

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 50

KNOWLEDGE_SOURCES = [
    {
        "url": "https://zyncjobs.com/about",
        "title": "About ZyncJobs",
        "category": "about",
        "content": """ZyncJobs is an AI-powered recruitment platform connecting candidates and employers.
It uses artificial intelligence to match job seekers with relevant positions, parse resumes,
generate job descriptions, and provide career guidance. The platform serves both Indian and global markets.""",
    },
    {
        "url": "https://zyncjobs.com/features",
        "title": "Platform Features",
        "category": "features",
        "content": """ZyncJobs offers AI resume parsing, ATS scoring, job matching, career advice,
skill assessment, interview preparation, resume building, cover letter generation,
JD generation, recruiter search, and chatbot assistance.""",
    },
    {
        "url": "https://zyncjobs.com/pricing",
        "title": "Pricing",
        "category": "pricing",
        "content": """ZyncJobs offers both free and premium plans.
Free plan includes basic job search and resume parsing.
Premium plans include ATS scoring, career advice, and recruiter features.
Contact sales for enterprise pricing with custom requirements.""",
    },
    {
        "url": "https://zyncjobs.com/how-it-works",
        "title": "How ZyncJobs Works",
        "category": "how-it-works",
        "content": """Candidates create profiles, upload resumes, and get AI-matched to jobs.
Employers post jobs, search for candidates, and use AI tools for recruitment.
The AI brain analyzes resumes against job descriptions for optimal matching.""",
    },
    {
        "url": "https://zyncjobs.com/for-candidates",
        "title": "For Candidates",
        "category": "candidates",
        "content": """Build and optimize your resume with AI assistance. Get ATS scores and suggestions.
Take skill assessments to showcase your abilities. Receive personalized career advice
and interview preparation. Get matched to relevant job openings automatically.""",
    },
    {
        "url": "https://zyncjobs.com/for-employers",
        "title": "For Employers",
        "category": "employers",
        "content": """Post jobs and generate professional job descriptions with AI. Search and shortlist
candidates using AI-powered matching. Parse candidate resumes automatically.
Get AI-generated candidate evaluations and ranking.""",
    },
    {
        "url": "https://zyncjobs.com/support",
        "title": "Support",
        "category": "support",
        "content": """ZyncJobs support is available via email and chat. Premium users get priority support.
Common issues include resume upload problems, account setup, and billing inquiries.
Contact support@zyncjobs.com for assistance.""",
    },
]


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            last_period = text.rfind(".", start, end)
            if last_period > start + chunk_size // 2:
                end = last_period + 1
        chunks.append(text[start:end].strip())
        start = end - overlap if end < len(text) else len(text)
    return [c for c in chunks if c]


async def ingest_knowledge(source: Optional[str] = None) -> int:
    count = 0
    sources = [s for s in KNOWLEDGE_SOURCES if not source or s["category"] == source]
    for entry in sources:
        chunks = chunk_text(entry["content"])
        for i, chunk in enumerate(chunks):
            doc_id = f"{entry['category']}_{i}"
            await vector_store.upsert(
                doc_id=doc_id,
                text=chunk,
                metadata={
                    "url": entry["url"],
                    "title": entry["title"],
                    "category": entry["category"],
                    "chunk_index": i,
                },
            )
            count += 1
    logger.info("Ingested %d chunks from %d sources", count, len(sources))
    return count


async def ingest_file(file_path: str, category: str = "custom") -> int:
    count = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        title = file_path.split("/")[-1].replace(".txt", "").replace("_", " ").title()
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            doc_id = f"{category}_{hash(chunk) % 1000000}_{i}"
            await vector_store.upsert(
                doc_id=doc_id,
                text=chunk,
                metadata={
                    "url": file_path,
                    "title": title,
                    "category": category,
                    "chunk_index": i,
                },
            )
            count += 1
        logger.info("Ingested %d chunks from file %s", count, file_path)
    except Exception as e:
        logger.error("File ingestion failed for %s: %s", file_path, e)
    return count


async def reindex_all() -> int:
    await vector_store.connect()
    return await ingest_knowledge()
