import asyncio
from typing import Optional, Any
from app.agents.base_agent import BaseAgent
from app.orchestrator.intent_classifier import IntentClassifier
from app.config.settings import settings
from app.utils.logger import logger

# Agents that can hand off to another agent after their result
_HANDOFF_MAP: dict[str, str] = {
    "resume": "job_match",   # after resume analysis, check job match if JD provided
    "job_match": "career",   # after job match, suggest career improvements
}

# Agents that can run in parallel (both run, results merged)
_PARALLEL_GROUPS: list[list[str]] = [
    ["resume", "ats_tool"],          # resume + ATS analysis simultaneously
    ["career", "job_match"],         # career + job match simultaneously
]


class MasterAgent:
    def __init__(self):
        self._agents: dict[str, BaseAgent] = {}
        self.classifier = IntentClassifier()

    def register(self, name: str, agent: BaseAgent):
        self._agents[name] = agent

    def get_agent(self, name: str) -> Optional[BaseAgent]:
        return self._agents.get(name)

    def _find_parallel_group(self, intent: str) -> Optional[list[str]]:
        for group in _PARALLEL_GROUPS:
            if intent in group and all(a in self._agents for a in group):
                return group
        return None

    async def _resolve_intent_agent(self, query: str, kwargs: dict) -> tuple[str, BaseAgent]:
        intent = self.classifier.classify(query, kwargs)
        agent = self._agents.get(intent)
        if not agent:
            agent = next(iter(self._agents.values()), None)
            intent = "general"
        if not agent:
            raise ValueError("No agents registered")
        return intent, agent

    async def _run_single_chain(self, query: str, intent: str, agent: BaseAgent, user_id: Optional[str], kwargs: dict) -> dict:
        combined_result: dict = {"intent": intent, "agent": agent.name}
        current_intent = intent
        iterations = 0
        max_iter = settings.AGENT_MAX_ITERATIONS

        while iterations < max_iter:
            iterations += 1
            result = await agent.run(query, user_id=user_id, **kwargs)
            combined_result.update(result)
            logger.info(f"MasterAgent iteration={iterations} | agent={agent.name} | intent={current_intent}")

            next_intent = _HANDOFF_MAP.get(current_intent)
            if not next_intent:
                break
            next_agent = self._agents.get(next_intent)
            if not next_agent:
                break
            has_jd = bool(kwargs.get("job_description") or result.get("job_description"))
            if current_intent == "resume" and not has_jd:
                break

            logger.info(f"MasterAgent handoff | {current_intent} → {next_intent}")
            combined_result["handoff"] = next_intent
            agent = next_agent
            current_intent = next_intent

        combined_result["iterations"] = iterations
        return combined_result

    async def _run_parallel(self, query: str, group: list[str], user_id: Optional[str], kwargs: dict) -> list[dict]:
        async def _run_one(name: str) -> dict:
            a = self._agents[name]
            result = await a.run(query, user_id=user_id, **kwargs)
            return {"agent": name, **result}

        tasks = [_run_one(name) for name in group]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged: list[dict] = []
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"MasterAgent parallel agent failed", error=str(r))
                merged.append({"error": str(r)})
            else:
                merged.append(r)
        return merged

    async def process(
        self, query: str, user_id: Optional[str] = None, **kwargs
    ) -> dict:
        intent, agent = await self._resolve_intent_agent(query, kwargs)
        logger.info(f"MasterAgent routing | intent={intent} | query_length={len(query)}")

        parallel_group = self._find_parallel_group(intent)
        if parallel_group:
            logger.info(f"MasterAgent parallel | group={parallel_group}")
            results = await self._run_parallel(query, parallel_group, user_id, kwargs)
            combined = {"intent": intent, "parallel": True, "agents": parallel_group}
            for r in results:
                combined.update(r)
            combined["iterations"] = 1
            return combined

        return await self._run_single_chain(query, intent, agent, user_id, kwargs)


master_agent = MasterAgent()
