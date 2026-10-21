"""Offline regression check for the resume builder -> Skill Gap brain flow.

Runs the real intent classifier, router, Skill Gap brain, prompt construction,
and response cleanup while replacing only the external LLM call with a fixture.
Run from the AI service repo with: python scripts/check_skill_gap_flow.py
"""
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from urllib.request import Request, urlopen
from unittest.mock import AsyncMock, patch

os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recruitment_ai.brains.master.master_brain import master_brain
from recruitment_ai.brains.candidate.skill_gap_brain import llm_service
from recruitment_ai.brains.shared.brain_state import BrainState


async def live_check() -> None:
    base_url = os.getenv("SKILL_GAP_TEST_URL", "http://127.0.0.1:8001").rstrip("/")
    role = os.getenv("SKILL_GAP_TEST_ROLE", "Backend Developer")
    skills = [skill.strip() for skill in os.getenv("SKILL_GAP_TEST_SKILLS", "Python, Java, REST").split(",") if skill.strip()]
    experience_evidence = os.getenv(
        "SKILL_GAP_TEST_EXPERIENCE",
        "Built REST APIs using Java and Spring Boot.",
    )
    project_evidence = os.getenv(
        "SKILL_GAP_TEST_PROJECTS",
        "Created a booking service with PostgreSQL.",
    )
    body = {
        "query": (
            "Find matching jobs for this candidate. Analyze skill gaps for the exact target role using the candidate's resume evidence. "
            f"Target role: {role}. Diagnostic run: {uuid.uuid4()}. Return only a JSON analysis."
        ),
        "user_role": "candidate",
        "context": {
            "section": "skills",
            "action": "analyze_gap",
            "target_role": role,
            "current_role": "Backend Developer Intern",
            "current_skills": skills,
            "experience_evidence": experience_evidence,
            "project_evidence": project_evidence,
            "education_evidence": "B.Tech Computer Science",
            "content": (
                f"Target role: {role}\nCurrent skills: {', '.join(skills)}\n"
                f"Experience: {experience_evidence}\nProject: {project_evidence}"
            ),
        },
    }

    def post_request() -> dict:
        request = Request(
            f"{base_url}/ai/execute",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=150) as response:
            return json.loads(response.read().decode("utf-8"))

    response = await asyncio.to_thread(post_request)
    assert response.get("success") is True, response
    assert response.get("intent") == "SKILL_GAP", response.get("intent")
    result = response.get("result") or {}
    assert result.get("analysis_available") is True, result
    assert isinstance(result.get("missing_skills"), list), result
    if os.getenv("SKILL_GAP_TEST_REQUIRE_GAPS", "true").lower() in {"true", "1", "yes"}:
        assert result["missing_skills"], (
        "The thin Backend Developer fixture should produce at least one gap; "
        f"the live model returned none. Summary: {result.get('summary')}"
        )
    print(f"PASS: live AI service responded via {base_url}/ai/execute")
    print("PASS: live request routed to SKILL_GAP and returned the expected response contract")
    print("AI-suggested skills:", ", ".join(
        item if isinstance(item, str) else str(item.get("skill", ""))
        for item in result["missing_skills"]
    ) or "none")
    if result.get("summary"):
        print("AI summary:", result["summary"])


async def main() -> None:
    if "--live" in sys.argv:
        await live_check()
        return

    llm_reply = json.dumps({
        "missing_skills": [
            {"skill": "Docker", "priority": "important", "reason": "Commonly used for backend deployment."},
            {"skill": "Python", "priority": "nice_to_have", "reason": "Already listed in this resume."},
            {"skill": "PostgreSQL", "priority": "important", "reason": "Already used in the candidate's project."},
        ],
        "existing_relevant_skills": ["Python", "Java", "REST"],
        "gap_score": 20,
        "learning_resources": [],
        "quick_wins": [],
        "summary": "One relevant gap was identified.",
    })
    state = BrainState(
        # Candidate resume text may include job-matching language; explicit
        # Resume Builder metadata must still route this to the skill-gap brain.
        query=(
            "Find matching jobs for this candidate. Analyze skill gaps for the "
            "exact target role using resume evidence. Target role: Backend Developer."
        ),
        user_role="candidate",
        context={"section": "skills", "action": "analyze_gap"},
        context_data={
            "user_preferences": {
                "section": "skills",
                "action": "analyze_gap",
                "target_role": "Backend Developer",
                "current_role": "Backend Developer Intern",
                "current_skills": ["Python", "Java", "REST"],
                "experience_evidence": "Built REST APIs using Java and Spring Boot.",
                "project_evidence": "Created a booking service with PostgreSQL.",
                "education_evidence": "B.Tech Computer Science",
            },
            "resume": {"skills": ["Python", "Java", "REST"]},
        },
    )

    with patch.object(llm_service, "generate", new_callable=AsyncMock, return_value=llm_reply) as generate:
        result = await master_brain.execute(state)

    assert result.intent == "SKILL_GAP", f"Expected SKILL_GAP intent; got {result.intent!r}"
    assert result.metadata.get("brain") == "SkillGapBrain", result.metadata
    assert result.error is None, result.error
    assert generate.await_count == 1, "Skill Gap brain did not call the LLM"
    prompt = generate.await_args.kwargs["prompt"]
    for expected in ("Backend Developer", "Python, Java, REST", "Built REST APIs", "booking service"):
        assert expected in prompt, f"AI prompt is missing resume context: {expected}"

    payload = result.result
    assert payload and payload.get("analysis_available") is True, payload
    skills = [item["skill"] for item in payload["missing_skills"]]
    assert skills == ["Docker"], f"Existing skills/evidence were not filtered: {skills}"

    # Keep compatibility with the legacy frontend prompt, which requested an array.
    with patch.object(llm_service, "generate", new_callable=AsyncMock, return_value='["Docker", "Python"]'):
        legacy_result = await master_brain.execute(state)
    assert legacy_result.error is None, legacy_result.error
    assert legacy_result.result.get("analysis_available") is True, legacy_result.result
    legacy_skills = [item if isinstance(item, str) else item["skill"] for item in legacy_result.result["missing_skills"]]
    assert legacy_skills == ["Docker"], f"Legacy array response was not normalized/filtered: {legacy_skills}"

    with patch.object(llm_service, "generate", new_callable=AsyncMock, side_effect=["not valid JSON", llm_reply]) as retry_generate:
        retried_result = await master_brain.execute(state)
    assert retry_generate.await_count == 2, "Invalid model output did not trigger one repair attempt"
    assert retried_result.result.get("analysis_available") is True, retried_result.result
    print("PASS: request routed to SkillGapBrain")
    assert result.metadata.get("classifier") == "structured_context", result.metadata
    print("PASS: explicit Skill Gap action wins over job-match wording in resume context")
    print("PASS: target role and resume evidence reached the AI prompt")
    print("PASS: skills already listed or demonstrated in resume evidence were removed")
    print("PASS: frontend-compatible missing_skills response returned")
    print("PASS: legacy AI array response is normalized instead of shown as unavailable")
    print("PASS: malformed model output is retried once with a strict JSON schema")


if __name__ == "__main__":
    asyncio.run(main())
