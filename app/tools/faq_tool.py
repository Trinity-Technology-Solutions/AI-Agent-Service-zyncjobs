from .base_tool import BaseTool
from app.knowledge.knowledge_base import knowledge_base


class FAQTool(BaseTool):
    def __init__(self):
        super().__init__(name="faq_tool", description="Search ZyncJobs knowledge base using RAG")

    @property
    def result_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "found": {"type": "boolean"},
                "context": {"type": "string"},
                "sources": {"type": "array", "items": {"type": "string"}},
            },
        }

    def run(self, query: str, top_k: int = 3) -> dict:
        docs = knowledge_base.query(query, top_k=top_k)
        if not docs:
            return {"found": False, "context": ""}

        context = "\n\n".join(f"[{d.source}] {d.text}" for d in docs)
        return {"found": True, "context": context, "sources": [d.source for d in docs]}
