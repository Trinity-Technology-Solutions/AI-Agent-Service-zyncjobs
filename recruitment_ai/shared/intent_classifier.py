"""Intent classifier for routing requests to appropriate brains."""
import re
from typing import Optional
from recruitment_ai.shared.brain import BrainState


class IntentClassifier:
    """Classifies user intent to route to the correct brain."""

    INTENT_PATTERNS = {
        "RECRUITER": [
            r"recruiter:",
            r"candidate.*search",
            r"find.*candidate",
            r"hire.*candidate",
            r"scout.*candidate",
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
        "RESUME_EDIT": [
            r"improve.*resume.*section",
            r"rewrite.*resume.*section",
            r"generate.*resume.*section",
            r"shorten.*resume.*summary",
            r"optimize.*resume.*section",
            r"quantify.*resume.*section",
            r"generate content for resume",
            r"resume.*score",
            r"improve this.*section",
            r"generate.*for resume",
            r"generate.*skills.*resume",
            r"find.*missing.*skills",
            r"missing.*skills.*resume",
            r"fix grammar.*resume",
            r"write.*professional summary",
            r"concise.*professional.*summary",
            r"score this resume",
            r"analyze this resume",
            r"this resume.*section",
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
        "COVER_LETTER": [
            r"cover letter:",
            r"cover letter.*generate",
            r"generate.*cover letter",
            r"write.*cover letter",
            r"cover letter.*for",
        ],
        "CAREER_ADVICE": [
            r"career.*advice",
            r"career.*path",
            r"career.*coach",
            r"career.*plan",
            r"plan.*career",
            r"learning.*path",
            r"improve.*resume",
            r"salary.*tip",
            r"interview.*prep",
            r"interview.*tip",
        ],
        "SKILL_ASSESSMENT": [
            r"skill.*assessment",
            r"take.*test",
            r"assessment.*question",
            r"generate.*assessment",
            r"assessment.*for",
            r"quiz.*for",
            r"mcq.*for",
        ],
        "SKILL_GAP": [
            r"skill.*gap",
            r"gap.*analysis",
            r"what.*skills.*missing",
            r"missing.*skills",
            r"skills.*need.*learn",
            r"what.*learn.*become",
        ],
        "CAREER_ROADMAP": [
            r"career.*roadmap",
            r"roadmap.*from",
            r"roadmap.*to",
            r"career.*path.*plan",
            r"how.*become.*\w+",
            r"plan.*become",
        ],
        "INTERVIEW_PREP": [
            r"mock.*interview",
            r"prepare.*interview",
        ],
        "RESUME_PARSER": [
            r"parse.*resume",
            r"extract.*resume",
            r"resume.*parser",
            r"cv.*parser",
            r"analyze.*resume",
            r"read.*resume",
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