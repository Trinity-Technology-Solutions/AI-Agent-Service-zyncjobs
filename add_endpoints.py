import os

# 1. Add interview-questions to chat.py
chat_path = "recruitment_ai/api/routers/chat.py"
with open(chat_path, "r", encoding="utf-8") as f:
    content = f.read()

new_endpoint = """

@router.post("/interview-questions", response_model=dict)
async def interview_questions(job_title: str, skills: str = "", interview_type: str = "technical", user: dict = Depends(get_current_user)):
    \"\"\"Generate interview questions for a given role.\"\"\"
    result = await graph.ainvoke(_build_state({
        "job_title": job_title, "skills": skills.split(",") if skills else [],
        "interview_type": interview_type,
        "query": f"Prepare interview questions for {job_title}",
    }, user, "INTERVIEW_PREP"))
    r = result.get("result") or {}
    return {
        "success": result.get("error") is None,
        "questions": r.get("questions") or [],
        "tips": r.get("tips") or [],
        "error": result.get("error"),
    }"""

content = content.replace('@router.post("/cover-letter"', new_endpoint + '\n\n@router.post("/cover-letter"')
with open(chat_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Added /interview-questions to chat.py")

# 2. Add candidates/rank to recruiter.py
rec_path = "recruitment_ai/api/routers/recruiter.py"
with open(rec_path, "r", encoding="utf-8") as f:
    content = f.read()

new_rank = """

@router.post("/candidates/rank", response_model=dict)
async def rank_candidates(job_description: str, candidates: list = [], user: dict = Depends(get_current_user)):
    \"\"\"Rank candidates by fit score for a given job.\"\"\"
    result = await graph.ainvoke(_build_state({
        "job_description": job_description, "candidates": candidates,
        "query": "Rank candidates for job",
    }, user, "RECRUITER"))
    r = result.get("result") or {}
    return {
        "success": result.get("error") is None,
        "ranked_candidates": r.get("candidates") or r.get("ranked") or [],
        "error": result.get("error"),
    }"""

content = content.replace('@router.post("/shortlist"', new_rank + '\n\n@router.post("/shortlist"')
with open(rec_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Added /candidates/rank to recruiter.py")
