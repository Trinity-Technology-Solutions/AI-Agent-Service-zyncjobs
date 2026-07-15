"""Intent classifier — rules/regex primary, LLM fallback for low confidence."""
import re
from recruitment_ai.brains.base import BrainState

INTENT_PATTERNS: dict[str, list[str]] = {
    "RECRUITER": [
        r"recruiter:",
        r"candidate.*search",
        r"find.*candidate",
        r"hire.*candidate",
        r"scout.*candidate",
        r"find.*developer",
        r"find.*engineer",
        r"search.*developer",
        r"search.*engineer",
        r"hire.*developer",
    ],
    "RECRUITER_SEARCH": [
        r"search.*candidate",
        r"find.*candidate",
        r"candidate.*search",
        r"search.*developer",
        r"search.*engineer",
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
        r"edit.*resume",
        r"improve.*resume.*section",
        r"rewrite.*resume.*section",
        r"generate.*resume.*section",
        r"shorten.*resume.*summary",
        r"optimize.*resume.*section",
        r"quantify.*resume.*section",
        r"generate content for resume",
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
        r"assess.*skill",
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
        r"^(hi|hello|hey|help)[!?.]*$",
        r"what.*can.*you",
        r"how.*does.*work",
        r"zyncjobs",
        r"platform",
        r"feature",
    ],
}

_LLM_CLASSIFY_PROMPT = """Classify this user query into exactly one intent.

Query: {query}

Intents: CHAT, RESUME_BUILDER, RESUME_PARSER, RESUME_EDIT, ATS_SCORE,
CAREER_ADVICE, CAREER_ROADMAP, SKILL_GAP, SKILL_ASSESSMENT,
JOB_MATCH, JOB_PARSER, JD_GENERATOR, INTERVIEW_PREP,
COVER_LETTER, RECRUITER, RECRUITER_SHORTLIST

Reply with ONLY the intent name. Nothing else."""


class IntentClassifier:

    def __init__(self):
        self._compiled = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in INTENT_PATTERNS.items()
        }

    def _rule_classify(self, query: str) -> str | None:
        """Returns intent if a rule matches, else None."""
        for intent, patterns in self._compiled.items():
            for pattern in patterns:
                if pattern.search(query):
                    return intent
        return None

    async def _llm_classify(self, query: str) -> str:
        """LLM fallback for ambiguous queries."""
        try:
            from recruitment_ai.llm import llm_service
            result = await llm_service.generate(
                brain_name="chatbot",
                prompt=_LLM_CLASSIFY_PROMPT.format(query=query),
                temperature=0.0,
                max_tokens=10,
            )
            intent = result.strip().upper().split()[0]
            if intent in self._compiled:
                return intent
        except Exception:
            pass
        return "CHAT"

    async def classify(self, state: BrainState) -> BrainState:
        query = (state.query or "").strip()
        if not query:
            state.intent = "CHAT"
            return state

        intent = self._rule_classify(query)
        if intent:
            state.intent = intent
            state.metadata["classifier"] = "rule"
        else:
            state.intent = await self._llm_classify(query)
            state.metadata["classifier"] = "llm"

        return state

    def classify_sync(self, query: str) -> str:
        return self._rule_classify(query) or "CHAT"


intent_classifier = IntentClassifier()
