import re

# Intent keyword rules — ordered by specificity
INTENT_RULES = {
    "GREETING":          [r"\b(hi|hello|hey|good morning|good afternoon|good evening|howdy|what's up|sup|greetings|namaste)\b"],
    "SMALL_TALK":        [r"\b(how are you|how r u|what are you|who are you|what can you do|tell me about yourself|are you (a |an )?(bot|ai|robot|human)|what('s| is) your name)\b",
                          r"\b(thanks|thank you|thx|ok|okay|got it|cool|great|awesome|nice|good|perfect|sounds good)\b"],
    "JOB_SEARCH":        [r"\b(find|search|show|get|list|looking for).{0,20}(job|jobs|opening|vacancy|vacancies|position|role)\b",
                          r"\b(job|jobs|opening|vacancy).{0,20}(in|at|for|near)\b",
                          r"\b(hiring|available jobs|job listing)\b",
                          r"\b(jobs|openings|vacancies|positions|roles)\b",
                          r"\b(developer|engineer|designer|analyst|manager|lead|architect|consultant|specialist).{0,10}(job|jobs|role|roles|opening|position)\b",
                          r"\b(job|jobs|role|roles|opening|position).{0,10}(developer|engineer|designer|analyst|manager|lead|architect)\b",
                          r"\b(software developer|software engineer|web developer|frontend developer|backend developer|full.?stack developer|mobile developer|data engineer|data analyst|data scientist|devops engineer|java developer|python developer|react developer|node developer|android developer|ios developer)\b"],
    "JOB_DETAILS":       [r"\b(details|description|requirements|responsibilities|about).{0,20}(job|role|position)\b",
                          r"\b(what does.{0,20}(job|role) (involve|require|include))\b"],
    "JOB_APPLICATION":   [r"\b(apply|applied|application|how to apply|submit (resume|application))\b"],
    "RESUME_PARSE":      [r"\b(parse|extract|read|analyze).{0,20}resume\b",
                          r"\b(upload|scan).{0,20}(resume|cv)\b"],
    "RESUME_IMPROVE":    [r"\b(improve|enhance|update|fix|rewrite|optimize|better).{0,20}(resume|cv)\b",
                          r"\b(resume|cv).{0,20}(improve|enhance|update|fix|rewrite|optimize|better)\b",
                          r"\b(make my resume|resume tips|resume help)\b"],
    "ATS_SCORE":         [r"\b(ats|applicant tracking).{0,20}(score|check|analyze|test)\b",
                          r"\b(ats score|check ats|ats analysis)\b"],
    "RESUME_BUILDER":    [r"\b(build|create|make|generate|write).{0,20}(resume|cv)\b",
                          r"\b(resume builder|create resume|new resume)\b"],
    "JOB_RECOMMENDATION":[r"\b(recommend|suggest|suitable|best).{0,20}(job|jobs|role|roles)\b",
                          r"\b(job recommendation|jobs for me|what jobs)\b"],
    "CAREER_ROADMAP":    [r"\b(career (roadmap|path|plan|goal|growth|progression))\b",
                          r"\b(how to become|roadmap (to|for)|career in)\b"],
    "SKILL_GAP":         [r"\b(skill gap|missing skills|skills (i need|to learn|required))\b",
                          r"\b(what skills|which skills|upskill|reskill)\b",
                          r"\b(improve|develop|build|grow|enhance).{0,20}(skill|skills|knowledge|expertise)\b",
                          r"\b(how (to|do i|can i|should i)).{0,20}(learn|improve|develop|grow).{0,20}(skill|skills|myself|career)\b"],
    "INTERVIEW":         [r"\b(interview (question|prep|practice|tips|mock|preparation))\b",
                          r"\b(prepare for interview|mock interview|behavioral question|technical question)\b"],
    "JD_GENERATION":     [r"\b(generate|create|write|draft).{0,20}(job description|jd)\b",
                          r"\b(job description (for|template)|write jd)\b"],
    "RECRUITER":         [r"\b(recruiter|hiring manager|post (a )?job|find candidates|talent search)\b",
                          r"\b(compare candidates|candidate ranking|shortlist)\b"],
    "COMPANY_FAQ":       [r"\b(zyncjobs|about (the )?platform|how does zyncjobs|company (info|details|about))\b",
                          r"\b(what is zyncjobs|zync jobs|zync platform)\b"],
    "PLATFORM_HELP":     [r"\b(how (do i|to) (use|navigate|access|find)|platform (help|guide|tutorial))\b",
                          r"\b(where (can i|do i)|how (does|do) (the|this) (platform|site|portal))\b"],
    "ACCOUNT_HELP":      [r"\b(account|login|sign (in|up)|register|password|profile (setup|update|edit))\b",
                          r"\b(forgot password|reset password|create account|my account)\b"],
    "SALARY_QUERY":      [r"\b(salary|pay|compensation|ctc|package|how much (does|do|can)).{0,20}(earn|make|get|expect)\b",
                          r"\b(salary (range|insight|data|for)|average salary|market rate)\b"],
}


def classify(message: str) -> dict:
    """
    Returns {intent: str, confidence: float, entities: dict}
    """
    lower = message.strip().lower()
    scores = {}

    for intent, patterns in INTENT_RULES.items():
        count = sum(1 for p in patterns if re.search(p, lower, re.IGNORECASE))
        if count:
            scores[intent] = count

    if not scores:
        return {"intent": "UNKNOWN", "confidence": 0.0, "entities": _extract_entities(lower)}

    best = max(scores, key=lambda k: scores[k])
    confidence = min(scores[best] / len(INTENT_RULES[best]), 1.0)

    return {"intent": best, "confidence": round(confidence, 2), "entities": _extract_entities(lower)}


def _extract_entities(text: str) -> dict:
    entities = {}

    # Extract location — with or without preposition
    location_match = re.search(r'\b(?:in|at|near|from)\s+([a-z]+(?:\s+[a-z]+)?)', text, re.IGNORECASE)
    if not location_match:
        # bare city at end of message: "software developer chennai"
        location_match = re.search(
            r'\b(chennai|bangalore|bengaluru|mumbai|delhi|hyderabad|pune|kolkata|noida|gurgaon|gurugram|'
            r'ahmedabad|jaipur|coimbatore|kochi|indore|bhopal|nagpur|surat|vadodara|remote|india)\b',
            text, re.IGNORECASE
        )
        if location_match:
            entities["location"] = location_match.group(1).title()
    else:
        entities["location"] = location_match.group(1).title()

    # Extract skills/technologies (expanded list)
    skills = re.findall(
        r'\b(python|java(?:script)?|react|node(?:\.js)?|angular|vue|django|flask|sql|aws|docker|kubernetes|'
        r'typescript|golang|rust|php|ruby|swift|kotlin|c\+\+|machine learning|data science|devops|'
        r'fullstack|full.?stack|frontend|front.?end|backend|back.?end|'
        r'software developer|software engineer|web developer|mobile developer|'
        r'data engineer|data analyst|data scientist|ml engineer|ai engineer|'
        r'product manager|project manager|ui.?ux|designer|qa|tester|devops engineer|'
        r'java developer|python developer|react developer|node developer|android|ios)\b',
        text, re.IGNORECASE
    )
    if skills:
        entities["skills"] = list(set(s.lower() for s in skills))

    # Extract experience
    exp_match = re.search(r'(\d+)\s*(year|yr)s?\s*(of\s+)?(experience|exp)', text, re.IGNORECASE)
    if exp_match:
        entities["experience"] = exp_match.group(1) + " years"

    # Extract job type
    job_type = re.search(r'\b(full.?time|part.?time|remote|hybrid|contract|freelance|internship)\b', text, re.IGNORECASE)
    if job_type:
        entities["job_type"] = job_type.group(1).lower()

    return entities
