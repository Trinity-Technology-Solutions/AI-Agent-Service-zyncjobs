import re

BLOCKED = {
    "hacking": [
        r"\b(hack|exploit|bypass|crack|brute.?force|phish|ddos)\b",
        r"how to (hack|break into|steal password)",
        r"(gmail|facebook|instagram|bank).*(hack|password)",
    ],
    "illegal": [
        r"\b(drugs|cocaine|heroin)\b.*(buy|sell|get)",
        r"\b(weapon|bomb|explosive)\b.*(make|build|buy)",
        r"\b(counterfeit|fake (id|passport))\b",
    ],
    "adult": [r"\b(porn|pornography|xxx|nude|erotic)\b"],
    "violence": [
        r"how to (kill|murder|hurt|harm)",
        r"\b(kill|murder|assassinate|shoot|stab|bomb)\b.{0,30}\b(modi|gandhi|president|minister|politician|person|people|someone|anyone)\b",
        r"\b(want to|going to|plan to|how to)\b.{0,20}\b(kill|murder|hurt|harm|attack)\b",
    ],
    "medical": [r"should i take (medicine|drug|pill)", r"what (medicine|pill) (should|can) i take"],
    "politics": [r"\b(prime minister|president|election|political party|parliament)\b"],
    "religion": [r"\b(god|allah|jesus|bible|quran)\b.*(better|superior|wrong|evil|hate)"],
    "prompt_injection": [
        r"ignore (all |previous |above )?(instructions|prompts|rules)",
        r"you are now",
        r"forget (everything|all|your instructions)",
        r"disregard (your|all|previous)",
        r"override (your|all|previous)",
    ],
    "sql_injection": [
        r"(union|select|insert|drop|truncate)\s+(all\s+)?(select|from|into|table)",
        r"1\s*=\s*1",
    ],
    "xss": [r"<script[^>]*>", r"javascript\s*:", r"on(load|click|error)\s*="],
}

OFF_TOPIC = [
    r"\b(ipl|cricket|football|sports|match score|tournament)\b",
    r"\b(movie|film|actor|bollywood|netflix|series)\b",
    r"\b(recipe|cook|restaurant|travel|vacation)\b",
    r"\b(joke|funny|meme|comedy)\b",
    r"\b(weather|temperature|forecast)\b",
    r"\b(stock market|crypto|bitcoin|nft)\b",
]

BLOCKED_MESSAGES = {
    "hacking": "I'm unable to assist with hacking or illegal activities. If you have questions about careers, resumes, jobs, or interviews, I'd be happy to help.",
    "illegal": "I can't assist with requests involving illegal or harmful activities. If you need help with careers or recruitment, I'd be happy to assist.",
    "adult": "I'm the ZyncJobs AI Assistant. I specialize in careers, resumes, jobs, interviews, and recruitment.",
    "violence": "I can't assist with requests involving harming or killing people. If your question is about careers, jobs, resumes, or interviews, I'd be happy to help.",
    "medical": "I'm not a medical professional. Please consult a doctor. I can help you with career and job-related questions.",
    "politics": "I'm the ZyncJobs AI Assistant. I specialize in careers, resumes, jobs, and recruitment. I can't answer political questions.",
    "religion": "I'm the ZyncJobs AI Assistant. I specialize in careers, resumes, jobs, interviews, and recruitment.",
    "prompt_injection": "I can only assist with career, recruitment, and ZyncJobs-related topics.",
    "sql_injection": "I can only assist with career, recruitment, and ZyncJobs-related topics.",
    "xss": "I can only assist with career, recruitment, and ZyncJobs-related topics.",
}

OFF_TOPIC_MESSAGE = (
    "I'm the ZyncJobs AI Assistant. I specialize in careers, resumes, jobs, interviews, and recruitment. "
    "I can't answer general questions outside of recruitment topics. "
    "Feel free to ask about jobs, resume tips, interview prep, or career guidance!"
)


def validate(message: str) -> dict:
    """Returns {valid: bool, message: str | None, category: str | None}"""
    if not message or not message.strip():
        return {"valid": False, "message": "Please type your question. For example: \"Find Python jobs in Chennai\" or \"Improve my resume\".", "category": "empty"}

    stripped = message.strip()

    if len(stripped) > 2000:
        return {"valid": False, "message": "Your message is too long. Please keep it under 2000 characters.", "category": "too_long"}

    if re.match(r'^[\W_]+$', stripped) and not re.search(r'[a-zA-Z0-9]', stripped):
        return {"valid": False, "message": "Please enter a valid question. I can help with jobs, resumes, interviews, and career guidance.", "category": "spam"}

    lower = stripped.lower()

    for category, patterns in BLOCKED.items():
        for pattern in patterns:
            if re.search(pattern, lower, re.IGNORECASE):
                return {"valid": False, "message": BLOCKED_MESSAGES.get(category, "I can only assist with career and ZyncJobs-related topics."), "category": category}

    return {"valid": True, "message": None, "category": None}


def is_off_topic(message: str) -> bool:
    lower = message.lower()
    return any(re.search(p, lower, re.IGNORECASE) for p in OFF_TOPIC)
