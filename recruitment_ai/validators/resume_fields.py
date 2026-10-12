"""Field-specific, source-grounded resume validation shared by LLM and fallback paths."""
import re
from datetime import datetime
from urllib.parse import urlparse

ROLES = re.compile(r"\b(developer|engineer|analyst|manager|architect|consultant|director|lead|officer|executive|specialist|associate|coordinator|administrator|admin|recruiter|designer|tester|technician|operator|accountant|auditor|teacher|professor|researcher|scientist|chemist|nurse|doctor|physician|surgeon|dentist|lawyer|advocate|attorney|pharmacist|therapist|intern|trainee|apprentice|assistant|supervisor|president|head|chef|cook|driver|mechanic|electrician|plumber|welder|representative|salesperson|clerk|cashier|banker|writer|editor|journalist|ceo|cto|cfo|qa|sre)\b", re.I)
PROSE = re.compile(r"\b(with a|track record|responsible for|worked on|looking for|seeking|passionate|experienced in|experience in|proficient in|skilled in|i am|my|delivering|publishing|work experience|professional summary|career objective|contact details|education|curriculum vitae)\b", re.I)

def clean_field(value, kind):
    if not isinstance(value, str):
        return ""
    raw = value.strip()
    if not raw or re.fullmatch(r"unknown|n/?a|none|null|undefined|not (found|available|provided)|resume|cv", raw, re.I):
        return ""
    value = " ".join(raw.split())
    if kind == "url":
        parsed = urlparse(raw)
        return raw if parsed.scheme in {"http", "https"} and parsed.netloc else ""
    if kind == "summary":
        return value[:4000]
    if "\n" in raw or "\r" in raw or len(value) > 120 or PROSE.search(value):
        return ""
    if kind == "name":
        return value if value[0].isalpha() and all(c.isalpha() or c in " .'-" for c in value) and len(value.split()) <= 8 and not ROLES.search(value) and not re.search(r"\b(email|phone|skills|profile|summary|objective|address)\b", value, re.I) else ""
    if kind == "email":
        return value.lower() if re.fullmatch(r"[^\s@]+@(?:[a-z0-9-]+\.)+[a-z]{2,63}", value, re.I) else ""
    if kind == "phone":
        digits = re.sub(r"\D", "", value)
        return value if re.fullmatch(r"\+?[\d ().-]+", value) and 7 <= len(digits) <= 15 and len(set(digits)) > 1 and not re.fullmatch(r"\d{4}[-/]\d{2}[-/]\d{2}|\d{4}\s*[-/]\s*\d{4}", value) else ""
    if kind == "degree":
        return value if re.search(r"\b(bachelor|master|doctorate|ph\.?d|diploma|b\.?tech|m\.?tech|b\.?e|m\.?e|b\.?sc|m\.?sc|b\.?com|m\.?com|mba|bba|mca|bca|mbbs|bds|hsc|sslc|ssc|secondary|10th|12th)\b", value, re.I) and not ROLES.search(value) else ""
    if kind == "skill" and ROLES.search(value):
        return ""
    if kind == "jobTitle":
        return value if len(value) <= 100 and len(value.split()) <= 12 and value[0].isalpha() and not re.search(r"[@\d;,:|!?]", value) and ROLES.search(value) else ""
    if kind == "dob":
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                if datetime.strptime(value, fmt) <= datetime.now():
                    return value
            except ValueError:
                pass
        return ""
    if kind == "gender":
        return value if value.lower() in {"male", "female", "non-binary", "other"} else ""
    if kind == "date" and re.fullmatch(r"\d{4}\s*[-–]\s*\d{4}", value):
        return value
    if kind == "date":
        return value if re.fullmatch(r"(present|current|now|\d{4}|\d{4}[-/]\d{1,2}([-/]\d{1,2})?|\d{1,2}[-/]\d{4}|[a-z]{3,9}[ ,]+\d{4})(\s*[-–]\s*(present|current|now|\d{4}|[a-z]{3,9}[ ,]+\d{4}))?", value, re.I) else ""
    if re.search(r"@|https?://|\b(phone|email|mobile)\b", value, re.I):
        return ""
    if kind in {"location", "country"}:
        return value if len(value) <= 100 and not ROLES.search(value) and not re.search(r"\d{5}", value) else ""
    if kind == "company":
        return value
    return value if len(value) <= 100 and len(value.split()) <= 10 else ""

def evidence(value, source, kind):
    if not value or not source:
        return False
    if kind == "date":
        return re.sub(r"\s", "", value.lower()) in re.sub(r"\s", "", source.lower())
    if kind == "phone":
        return re.sub(r"\D", "", value) in re.sub(r"\D", "", source)
    text = " ".join(source.lower().split())
    pattern = re.escape(value.lower()).replace(r"\ ", r"\s+")
    return bool(re.search(r"(?<!\w)" + pattern + r"(?!\w)", text))

def validate_resume_fields(parsed, content):
    data = dict(parsed) if isinstance(parsed, dict) else {}
    contact = data.get("personalInfo") or data.get("personal_info") or data.get("contact") or {}
    contact = contact if isinstance(contact, dict) else {}
    def pick(kind, values):
        for value in values:
            clean = clean_field(value, kind)
            if evidence(clean, content, kind):
                return clean
        return ""
    data["name"] = pick("name", [data.get("name"), data.get("fullName"), contact.get("name"), contact.get("fullName")])
    data["email"] = pick("email", [data.get("email"), data.get("emailAddress"), contact.get("email")] + re.findall(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", content))
    data["phone"] = pick("phone", [data.get("phone"), data.get("phoneNumber"), data.get("mobile"), contact.get("phone")] + re.findall(r"\+?\d[\d ().-]{6,}\d", content))
    for key in ("location", "country", "dob", "gender", "currentCompany"):
        data[key] = pick("company" if key == "currentCompany" else key, [data.get(key)])
    kinds = {"jobTitle": "jobTitle", "title": "jobTitle", "role": "jobTitle", "company": "company", "school": "company", "institution": "company", "provider": "company", "degree": "degree", "name": "label", "location": "location", "startDate": "date", "endDate": "date", "date": "date", "issuedDate": "date", "expiryDate": "date", "gpa": "label", "score": "label", "credentialId": "label", "url": "url", "projectUrl": "url", "certificationUrl": "url"}
    aliases = {"name": ("projectName", "certificationName", "certificateName"), "company": ("companyName", "organization", "employer"), "jobTitle": ("designation", "role"), "provider": ("issuer", "issuingOrganization"), "startDate": ("start_date",), "endDate": ("end_date",), "issuedDate": ("issueDate", "issue_date", "year"), "expiryDate": ("expirationDate", "expiry_date"), "credentialId": ("certificateId", "completionId"), "url": ("link", "credentialUrl")}
    for key in ("workExperiences", "internships", "educations", "projects", "certifications", "awards", "competitions"):
        result = []
        for raw in data.get(key, []) if isinstance(data.get(key), list) else []:
            item = {"name": raw} if isinstance(raw, str) else dict(raw) if isinstance(raw, dict) else {}
            for field, names in aliases.items():
                if not item.get(field):
                    item[field] = next((item[name] for name in names if item.get(name)), "")
            if key in {"projects", "certifications", "awards", "competitions"} and not item.get("name"):
                item["name"] = item.get("title", "")
            cleaned = {field: pick(kind, [item[field]]) for field, kind in kinds.items() if field in item}
            descriptions = []
            for field in (("descriptions",) if isinstance(item.get("descriptions"), list) and item["descriptions"] else ("description",)) + ("details", "responsibilities", "highlights"):
                values = item.get(field)
                for value in values if isinstance(values, list) else [values]:
                    clean = clean_field(value, "summary")
                    if clean and clean not in descriptions:
                        descriptions.append(clean)
            cleaned["descriptions"] = descriptions
            cleaned["description"] = "\n".join(descriptions)
            if isinstance(item.get("technologies"), list):
                cleaned["technologies"] = list(filter(None, (pick("label", [value]) for value in item["technologies"])))
            if item.get("noExpiry") is True and re.search(r"no expiry|does not expire|lifetime", content, re.I):
                cleaned["noExpiry"] = True
            if any(isinstance(value, str) and value for field, value in cleaned.items() if field not in {"description", "descriptions"}):
                result.append(cleaned)
        data[key] = result
    data["title"] = pick("jobTitle", [data.get(key) for key in ("title", "jobTitle", "currentRole", "current_role", "profession")] + [item.get("jobTitle") or item.get("title") or item.get("role") for item in data["workExperiences"] + data["internships"]])
    data["jobTitle"] = data["title"]
    for key in ("skills", "tools", "softSkills", "languages"):
        data[key] = list(dict.fromkeys(filter(None, [pick("skill" if key == "skills" else "label", [value]) for value in data.get(key, [])]))) if isinstance(data.get(key), list) else []
    for key in ("location", "country", "currentCompany"):
        if data.get(key) and any(value and value.lower() == data[key].lower() for value in (data["name"], data["title"])):
            data[key] = ""
    data["summary"] = clean_field(data.get("summary"), "summary")
    experience = data.get("totalExperience")
    data["totalExperience"] = experience if isinstance(experience, (int, float)) and not isinstance(experience, bool) and 0 <= experience <= 80 else None
    value = data.get("experience")
    data["experience"] = value.strip() if isinstance(value, str) and re.fullmatch(r"\d+(?:\.\d+)?\s*(years?|yrs?|months?|mos?)", value.strip(), re.I) else ""
    data.pop("total_experience", None)
    for key in ("fullName", "emailAddress", "phoneNumber", "mobile", "currentRole", "current_role", "profession", "personalInfo", "personal_info", "contact"):
        data.pop(key, None)
    return data
