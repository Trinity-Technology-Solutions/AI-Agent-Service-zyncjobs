"""Intent classifier for routing requests to appropriate brains."""
import re
from typing import Optional
from recruitment_ai.shared.brain import BrainState


class IntentClassifier:
    """Classifies user intent to route to the correct brain."""

    INTENT_PATTERNS = {
        "RECRUITER": [
            r"recruit",
            r"candidate.*search",
            r"find.*candidate",
            r"hire.*candidate",
            r"scout",
        ],
        "RECRUITER_SHORTLIST": [
            r"shortlist",
            r"evaluate.*candidate",
            r"top.*candidate",
        ],
        "JOB_PARSER": [
            r"parse.*job",
            r"extract.*job",
            r"job.*description.*parse",
            r"jd.*parser",
        ],
        "JD_GENERATOR": [
            r"generate.*job.*description",
            r"generate.*jd",
            r"create.*job.*description",
            r"write.*job.*description",
            r"job.*description.*generate",
        ],
        "RESUME_PARSER": [
            r"parse.*resume",
            r"extract.*resume",
            r"resume.*parser",
            r"cv.*parser",
        ],
        "ATS_SCORE": [
            r"ats.*score",
            r"ats.*check",
            r"resume.*score",
            r"check.*ats",
        ],
        "JOB_MATCH": [
            r"match.*job",
            r"job.*match",
            r"find me.*job",
            r"recommend.*job",
            r"job.*recommend",
        ],
        "CAREER_ADVICE": [
            r"career.*advice",
            r"career.*roadmap",
            r"skill.*gap",
            r"learning.*path",
            r"career.*path",
        ],
        "SKILL_ASSESSMENT": [
            r"skill.*assessment",
            r"take.*test",
            r"assessment.*question",
        ],
        "INTERVIEW_PREP": [
            r"interview.*prep",
            r"mock.*interview",
            r"interview.*question",
            r"prepare.*interview",
        ],
        "RESUME_BUILDER": [
            r"build.*resume",
            r"create.*resume",
            r"resume.*builder",
            r"make.*resume",
        ],
        "CHAT": [
            r"hi|hello|hey",
            r"help",
            r"what.*can.*you",
            r"how.*does.*work",
            r"zyncjobs",
            r"platform",
            r"feature",
        ],
    }

    def __init__(self):
        self._compiled = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.INTENT_PATTERNS.items()
        }

    async def classify(self, state: BrainState) -> BrainState:
        query = state.query or ""
        if not query.strip():
            state.intent = "CHAT"
            return state

        for intent, patterns in self._compiled.items():
            for pattern in patterns:
                if pattern.search(query):
                    state.intent = intent
                    return state

        state.intent = "CHAT"
        return state

    def classify_sync(self, query: str) -> str:
        """Synchronous classification for simple use cases."""
        if not query.strip():
            return "CHAT"
        for intent, patterns in self._compiled.items():
            for pattern in patterns:
                if pattern.search(query):
                    return intent
        return "CHAT"


intent_classifier = IntentClassifier()