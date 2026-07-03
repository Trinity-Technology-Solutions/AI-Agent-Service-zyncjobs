import httpx
import os
from .base_tool import BaseTool

NODE_API = os.getenv("NODE_BACKEND_URL", "http://localhost:5000/api")
TIMEOUT = 8


class SalaryTool(BaseTool):
    def __init__(self):
        super().__init__(name="salary_tool", description="Get real salary insights from ZyncJobs database")

    def run(self, title: str) -> dict:
        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                res = client.get(f"{NODE_API}/salary-insights/by-title", params={"title": title})
                res.raise_for_status()
                return res.json()
        except httpx.TimeoutException:
            return {"found": False, "error": "timeout"}
        except Exception as e:
            return {"found": False, "error": str(e)}
