import re
from .base_tool import BaseTool


_TYPOS = {
    "responsiblities": "responsibilities",
    "managment": "management",
    "experiance": "experience",
    "acheivement": "achievement",
    "recieved": "received",
    "tehnical": "technical",
    "developped": "developed",
    "programing": "programming",
    "collaborate": "collaborate",
    "impletement": "implement",
    "maintainance": "maintenance",
    "comunicate": "communicate",
    "analize": "analyze",
    "orginization": "organization",
    "departm ent": "department",
}


class GrammarTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="grammar_tool",
            description="Checks text for common grammar and spelling issues",
        )

    @property
    def result_schema(self) -> dict:
        return {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "message": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                },
            },
        }

    def run(self, text: str) -> list[dict]:
        issues = []

        if re.search(r'  ', text):
            issues.append({"type": "spacing", "message": "Double spaces found", "severity": "low"})

        for typo, correction in _TYPOS.items():
            for match in re.finditer(rf'\b{typo}\b', text, re.IGNORECASE):
                issues.append({
                    "type": "typo",
                    "message": f"'{match.group()}' should be '{correction}'",
                    "severity": "medium",
                    "position": match.start(),
                })

        for match in re.finditer(r'\b(was|were|been|being)\s+\w+ed\b', text, re.IGNORECASE):
            issues.append({
                "type": "passive_voice",
                "message": f"Passive voice: '{match.group()}' — consider active voice",
                "severity": "low",
                "position": match.start(),
            })

        return issues
