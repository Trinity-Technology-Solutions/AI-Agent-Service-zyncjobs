import httpx
import os
from .base_tool import BaseTool

NODE_API = os.getenv("NODE_BACKEND_URL", "http://localhost:5000/api")
TIMEOUT = 8


class SearchJobsTool(BaseTool):
    def __init__(self):
        super().__init__(name="search_jobs", description="Search real jobs from ZyncJobs database")

    @property
    def result_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "found": {"type": "boolean"},
                "count": {"type": "integer"},
                "jobs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "company": {"type": "string"},
                            "location": {"type": "string"},
                            "type": {"type": "string"},
                            "experience": {"type": "string"},
                            "salary": {"type": "string"},
                            "skills": {"type": "array", "items": {"type": "string"}},
                            "posted": {"type": "string"},
                        },
                    },
                },
                "message": {"type": "string"},
            },
        }

    def run(self, query: str = "", location: str = "", job_type: str = "", limit: int = 8) -> dict:
        try:
            params = {"limit": limit}
            if query.strip():
                params["q"] = query.strip()
            if location:
                params["location"] = location
            if job_type:
                params["jobType"] = job_type

            with httpx.Client(timeout=TIMEOUT) as client:
                res = client.get(f"{NODE_API}/jobs/search", params=params)
                res.raise_for_status()
                jobs = res.json()

            if not jobs or not isinstance(jobs, list):
                return {"found": False, "count": 0, "jobs": [], "message": "No jobs found matching your criteria."}

            formatted = []
            for j in jobs[:limit]:
                # salary can be nested {min, max, currency} or flat salaryMin/salaryMax
                salary_obj = j.get("salary") or {}
                sal_min = salary_obj.get("min") or j.get("salaryMin")
                sal_max = salary_obj.get("max") or j.get("salaryMax")
                currency = salary_obj.get("currency") or j.get("currency", "")
                salary = ""
                if sal_min and sal_max:
                    salary = f"{currency} {int(sal_min):,} - {int(sal_max):,}"

                # skills may be a PG array string like {React,Node} or a real list
                raw_skills = j.get("skills", []) or []
                if isinstance(raw_skills, str):
                    raw_skills = raw_skills.strip("{}").split(",") if raw_skills else []
                skills = [s.strip().strip('"') for s in raw_skills if s.strip()][:5]

                formatted.append({
                    "title": j.get("jobTitle", ""),
                    "company": j.get("company", ""),
                    "location": j.get("location", ""),
                    "type": j.get("jobType", ""),
                    "experience": j.get("experienceLevel", ""),
                    "salary": salary,
                    "skills": skills,
                    "posted": (j.get("createdAt") or "")[:10],
                })

            return {"found": bool(formatted), "count": len(formatted), "jobs": formatted}

        except httpx.TimeoutException:
            return {"found": False, "count": 0, "jobs": [], "error": "timeout"}
        except Exception as e:
            return {"found": False, "count": 0, "jobs": [], "error": str(e)}
