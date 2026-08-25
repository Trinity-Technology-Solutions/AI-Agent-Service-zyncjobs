"""Resume Parser Brain — full enterprise pipeline.
Pipeline: BrainState.request → LLM → JSON Validator → BrainResult
"""
import re
import time
import json
from recruitment_ai.brains.base import Brain, BrainState, BrainResult
from recruitment_ai.llm import llm_service
from recruitment_ai.validators.json_validator import validate_json_strict, ensure_json_fields
from recruitment_ai.prompts import get_prompt, get_system_prompt
from recruitment_ai.brains.candidate.skill_keywords import SKILL_KEYWORDS, extract_matched_skills


class ResumeParserBrain(Brain):
    def __init__(self):
        super().__init__()

    async def run(self, state: BrainState) -> BrainResult:
        start = time.perf_counter()
        from recruitment_ai.utils.ocr import extract_text
        raw_content = state.request.file_content or state.file_content or state.request.query or state.query or ""
        file_type = state.request.file_type or state.file_type or "txt"
        content = extract_text(raw_content, file_type) if file_type != "txt" else raw_content

        if not content.strip():
            return BrainResult(success=False, response={"error": "No resume content provided"})

        if not self._is_resume(content):
            return BrainResult(success=False, response={"error": "The uploaded file does not appear to be a resume. Please upload a valid resume."})

        # Pre-label sections for better LLM understanding (like backend does)
        labeled_content = self._section_label_text(content)
        
        prompt = get_prompt("resume_parser_prompt", resume_text=labeled_content[:15000])
        system = get_system_prompt("resume_parser")

        try:
            raw = await llm_service.generate(
                brain_name="resume_parser",
                prompt=prompt,
                system=system,
                temperature=0.1,
                max_tokens=4000,
            )
            parsed = validate_json_strict(raw, "object") or {}
            data, confidence = self._validate_and_score(parsed, content)
            
            return BrainResult(
                response=data,
                metadata={"parser": "llm", "extracted": True, "confidence": confidence},
                execution_time=time.perf_counter() - start,
            )
        except Exception as e:
            fallback_data = self._fallback_parse(content)
            return BrainResult(
                response=fallback_data,
                metadata={"parser": "fallback", "fallback_reason": str(e), "confidence": 0.3},
                execution_time=time.perf_counter() - start,
            )

    def _section_label_text(self, text: str) -> str:
        """Label resume sections like the backend does for better LLM parsing."""
        # Section headings — anchored (full-line)
        SECTION_HEADINGS = [
            (re.compile(r'^(summary|professional summary|profile|career summary|objective|career objective|about me|personal profile)$', re.I), 'SUMMARY'),
            (re.compile(r'^(work experience|professional experience|experience|employment history|work history|employment|career history|work)$', re.I), 'EXPERIENCE'),
            (re.compile(r'^(internships?|internship training|industrial training|training|teaching experience)$', re.I), 'INTERNSHIPS'),
            (re.compile(r'^(education|academic qualifications?|academic background|academic details|qualifications?|educational qualification)$', re.I), 'EDUCATION'),
            (re.compile(r'^(technical skills|skills|skills summary|core skills|key skills|technologies|tech stack|skills[\s&/]+technologies)$', re.I), 'SKILLS'),
            (re.compile(r'^(projects?|academic projects?|personal projects?|major projects?|project work)$', re.I), 'PROJECTS'),
            (re.compile(r'^(certifications?|licenses?|licences?|certificates|courses?|professional development)$', re.I), 'CERTIFICATIONS'),
            (re.compile(r'^(languages|language proficiency|language skills)$', re.I), 'LANGUAGES'),
            (re.compile(r'^(awards?|honors?|honours?|achievements?|accomplishments?|recognitions?)$', re.I), 'AWARDS'),
            (re.compile(r'^(contact|contact information|personal details|personal information)$', re.I), 'CONTACT'),
            (re.compile(r'^(extra.?curricular|co.?curricular|volunteer(ing)?|interests|hobbies|activities)$', re.I), 'EXTRACURRICULAR'),
            (re.compile(r'^(publications?|research( work)?|papers?|patents?)$', re.I), 'PUBLICATIONS'),
            (re.compile(r'^(declaration|references|additional information)$', re.I), 'OTHER'),
        ]
        
        lines = text.split('\n')
        sections = []
        cur = {'heading': 'HEADER', 'body': []}
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # Check if it's a heading
            heading = None
            for pattern, name in SECTION_HEADINGS:
                # Remove trailing punctuation like :, •, -, –, *, #
                clean_stripped = re.sub(r'[:•\-–*#]+$', '', stripped).strip()
                if pattern.match(clean_stripped):
                    heading = name
                    break
            if heading:
                if cur['body']:
                    sections.append(cur)
                cur = {'heading': heading, 'body': []}
            else:
                cur['body'].append(line)
        
        if cur['body']:
            sections.append(cur)
        
        # Rebuild with markers
        parts = []
        for s in sections:
            if s['heading'] == 'HEADER':
                parts.append('\n'.join(s['body']))
            else:
                parts.append(f'[{s["heading"]}]\n' + '\n'.join(s['body']))
        
        return '\n\n'.join(parts)

    def _is_resume(self, content: str) -> bool:
        """Return True only if content has enough resume signals to be worth parsing."""
        text = content.lower()
        # Must have at least one contact signal (email or phone)
        has_contact = bool(
            re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", content) or
            re.search(r"(?:\+\d{1,3}[\s\-]?)?\d[\d\s\-\.\(\)]{8,}\d", content)
        )
        # Must have at least 2 of these resume section keywords
        section_keywords = [
            "experience", "education", "skill", "summary", "objective",
            "profile", "certification", "project", "internship",
            "employment", "qualification", "achievement",
        ]
        section_hits = sum(1 for kw in section_keywords if kw in text)
        return has_contact and section_hits >= 2

    def _strict_sanitize(self, value: str, field_type: str) -> str:
        """Strict sanitization - each field only allows its intended data type."""
        if not value or not isinstance(value, str):
            return ""
        v = value.strip()
        
        if field_type == "name":
            # Only letters, spaces, dots, hyphens - no digits, @, +, job title keywords
            if re.search(r'[\d@+]', v): return ""
            if re.search(r'\b(developer|engineer|manager|analyst|intern|architect|consultant|director|lead|senior|junior|hr|ceo|cto|founder|student|fresher|software|full.?stack|front.?end|back.?end|data|devops|cloud|mobile|web|recruiter|designer|tester|qa|admin|executive|specialist|associate|coordinator|officer|president|vice|head|principal|staff|trainee)\b', v, re.I): return ""
            if not re.match(r'^[A-Za-z][A-Za-z.\'\-\s]{1,50}$', v): return ""
            if len(v.split()) > 4: return ""  # Max 4 words
            return re.sub(r'\b\w+\b', lambda m: m.group().capitalize(), v)
        
        elif field_type == "email":
            if not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', v): return ""
            return v.lower()
        
        elif field_type == "phone":
            if re.search(r'[a-zA-Z]', v): return ""
            digits = re.sub(r'\D', '', v)
            if len(digits) < 10 or len(digits) > 15: return ""
            if re.match(r'^(19|20)\d{2}', digits): return ""  # Reject years
            return v
        
        elif field_type == "jobTitle":
            if re.search(r'[@+]', v): return ""
            if re.match(r'^\d+$', v): return ""
            if len(v.split()) > 6: return ""
            return v
        
        elif field_type == "location":
            if re.search(r'[@+]', v): return ""
            if re.search(r'\b(developer|engineer|manager|analyst|intern|hr|ceo|cto)\b', v, re.I): return ""
            if len(v) > 50: return ""
            return v
        
        elif field_type == "company":
            if re.search(r'[@+]', v): return ""
            if len(v) > 100: return ""
            return v
        
        elif field_type == "summary":
            return v[:2000] if len(v) > 2000 else v
        
        elif field_type == "degree":
            # Degree labels like B.Tech, MBA, HSC, etc.
            if len(v) > 50: return ""
            return v
        
        return v

    def _sanitize_list(self, items: list, field_type: str) -> list:
        """Sanitize a list of strings or objects."""
        if not isinstance(items, list):
            return []
        result = []
        for item in items:
            if isinstance(item, dict):
                sanitized = {}
                for key, val in item.items():
                    if key in ("jobTitle", "title", "role"):
                        sanitized[key] = self._strict_sanitize(str(val or ""), "jobTitle")
                    elif key in ("company", "school", "provider", "institution"):
                        sanitized[key] = self._strict_sanitize(str(val or ""), "company")
                    elif key == "degree":
                        sanitized[key] = self._strict_sanitize(str(val or ""), "degree")
                    elif key == "name":  # project name, cert name
                        sanitized[key] = self._strict_sanitize(str(val or ""), "jobTitle")
                    elif key == "descriptions" and isinstance(val, list):
                        sanitized[key] = [self._strict_sanitize(str(d or ""), "summary") for d in val]
                    else:
                        sanitized[key] = str(val or "")
                # Only keep if has meaningful content
                if any(v for v in sanitized.values() if v):
                    result.append(sanitized)
            else:
                sanitized = self._strict_sanitize(str(item or ""), field_type)
                if sanitized:
                    result.append(sanitized)
        return result

    def _validate_and_score(self, parsed: dict, content: str) -> tuple:
        """Validate parsed data and compute confidence scores per field."""
        confidence_scores = {}
        
        # Name validation
        if not parsed.get("name") or len(parsed["name"].split()) < 2:
            parsed["name"] = self._extract_name(content) or "Unknown"
            confidence_scores["name"] = 0.3
        else:
            confidence_scores["name"] = 0.9
        
        # Email validation (regex is more reliable)
        if not parsed.get("email"):
            match = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", content)
            parsed["email"] = match.group() if match else ""
            confidence_scores["email"] = 0.9 if match else 0.0
        else:
            confidence_scores["email"] = 0.95
        
        # Phone validation
        if not parsed.get("phone"):
            match = re.search(r"(?:\+\d{1,3}[\s\-]?)?\d[\d\s\-\.\(\)]{8,}\d", content)
            parsed["phone"] = match.group().strip() if match else ""
            confidence_scores["phone"] = 0.8 if match else 0.0
        else:
            # Validate phone format
            phone = parsed["phone"]
            digits = re.sub(r"\D", "", phone)
            if len(digits) >= 10 and not re.match(r"^(19|20)\d{2}", digits):
                confidence_scores["phone"] = 0.9
            else:
                confidence_scores["phone"] = 0.4
        
        # Location validation
        if not parsed.get("location"):
            parsed["location"] = self._extract_location(content)
            confidence_scores["location"] = 0.6 if parsed["location"] else 0.0
        else:
            confidence_scores["location"] = 0.8
        
        # Country inference
        if not parsed.get("country"):
            if parsed.get("location") or "+91" in content or "india" in content.lower():
                parsed["country"] = "India"
            confidence_scores["country"] = 0.7 if parsed.get("country") else 0.0
        else:
            confidence_scores["country"] = 0.8
        
        # Apply strict sanitization to all fields BEFORE validation
        parsed["name"] = self._strict_sanitize(parsed.get("name", ""), "name")
        parsed["email"] = self._strict_sanitize(parsed.get("email", ""), "email")
        parsed["phone"] = self._strict_sanitize(parsed.get("phone", ""), "phone")
        parsed["jobTitle"] = parsed.get("title", "")
        parsed["title"] = self._strict_sanitize(parsed.get("title", ""), "jobTitle")
        parsed["location"] = self._strict_sanitize(parsed.get("location", ""), "location")
        parsed["summary"] = self._strict_sanitize(parsed.get("summary", ""), "summary")
        
        # Skills validation - cross-reference with known keywords
        skills = self._sanitize_list(parsed.get("skills", []), "jobTitle")
        # Add regex-extracted skills as backup
        regex_skills = extract_matched_skills(content)
        for skill in regex_skills:
            if skill not in skills:
                skills.append(skill)
        parsed["skills"] = skills
        confidence_scores["skills"] = 0.7 if skills else 0.3
        
        # Soft skills - validate against known list
        soft_skills = self._sanitize_list(parsed.get("softSkills", []), "jobTitle")
        known_soft = {"communication", "leadership", "teamwork", "problem solving", "adaptability", "time management", "critical thinking", "creativity", "collaboration", "interpersonal", "negotiation", "mentoring"}
        soft_skills = [s for s in soft_skills if s.lower() in known_soft]
        parsed["softSkills"] = soft_skills
        confidence_scores["softSkills"] = 0.6 if soft_skills else 0.3
        
        # Tools validation
        tools = self._sanitize_list(parsed.get("tools", []), "jobTitle")
        parsed["tools"] = tools
        confidence_scores["tools"] = 0.6 if tools else 0.3
        
        # Work experiences validation
        work_exps = self._sanitize_list(parsed.get("workExperiences", []), "jobTitle")
        parsed["workExperiences"] = work_exps
        confidence_scores["workExperiences"] = 0.8 if work_exps else 0.2
        
        # Internships validation - also relocate any intern-like entries from workExperiences
        internships = self._sanitize_list(parsed.get("internships", []), "jobTitle")
        # Check workExperiences for intern-like titles and move them
        kept_work = []
        for exp in work_exps:
            title = exp.get("jobTitle", "").lower()
            if re.search(r'\b(intern|trainee|apprentice|industrial training|co-?op|teaching assistant)\b', title):
                internships.append(exp)
            else:
                kept_work.append(exp)
        parsed["workExperiences"] = kept_work
        parsed["internships"] = internships
        confidence_scores["internships"] = 0.7 if internships else 0.3
        
        # Educations validation
        educations = self._sanitize_list(parsed.get("educations", []), "jobTitle")
        # Ensure degree/school aren't swapped
        for edu in educations:
            degree = edu.get("degree", "")
            school = edu.get("school", "")
            if re.search(r'\b(b\.?tech|m\.?tech|b\.?e|m\.?e|mba|bachelor|master|phd|diploma|hsc|sslc|10th|12th)\b', school, re.I):
                degree, school = school, degree
                edu["degree"] = degree
                edu["school"] = school
        parsed["educations"] = educations
        confidence_scores["educations"] = 0.8 if educations else 0.3
        
        # Projects validation
        projects = self._sanitize_list(parsed.get("projects", []), "jobTitle")
        parsed["projects"] = projects
        confidence_scores["projects"] = 0.7 if projects else 0.3
        
        # Certifications
        certs = self._sanitize_list(parsed.get("certifications", []), "jobTitle")
        parsed["certifications"] = certs
        confidence_scores["certifications"] = 0.7 if certs else 0.3
        
        # Languages
        langs = self._sanitize_list(parsed.get("languages", []), "jobTitle")
        parsed["languages"] = langs
        confidence_scores["languages"] = 0.6 if langs else 0.3
        
        # Awards
        awards = self._sanitize_list(parsed.get("awards", []), "jobTitle")
        parsed["awards"] = awards
        confidence_scores["awards"] = 0.6 if awards else 0.3
        
        # Competitions
        comps = self._sanitize_list(parsed.get("competitions", []), "jobTitle")
        parsed["competitions"] = comps
        confidence_scores["competitions"] = 0.6 if comps else 0.3
        
        # Summary
        if not parsed.get("summary"):
            parsed["summary"] = content[:500]
            confidence_scores["summary"] = 0.3
        else:
            confidence_scores["summary"] = 0.7
        
        # Title
        if not parsed.get("title"):
            # Try to infer from first work experience
            if parsed["workExperiences"]:
                parsed["title"] = parsed["workExperiences"][0].get("jobTitle", "")
            elif parsed["internships"]:
                parsed["title"] = parsed["internships"][0].get("jobTitle", "")
            confidence_scores["title"] = 0.5 if parsed.get("title") else 0.0
        else:
            confidence_scores["title"] = 0.8
        
        # Overall confidence
        overall_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0.0
        
        return parsed, overall_confidence

    def _extract_name(self, content: str) -> str:
        lines = [l.strip() for l in content.split("\n") if l.strip()]
        for line in lines[:5]:
            normalized = line
            if line == line.upper() and len(line) > 1:
                normalized = line.title()
            if re.match(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z.]*){1,3}$", normalized):
                return normalized
        return ""

    def _extract_location(self, content: str) -> str:
        indian_cities = [
            'Chennai', 'Bangalore', 'Bengaluru', 'Mumbai', 'Hyderabad', 'Pune', 'Delhi', 'New Delhi', 'Noida', 'Gurgaon', 'Gurugram',
            'Kolkata', 'Ahmedabad', 'Coimbatore', 'Kochi', 'Jaipur', 'Indore', 'Bhopal', 'Nagpur', 'Surat', 'Lucknow', 'Visakhapatnam',
            'Vizag', 'Mysore', 'Mysuru', 'Madurai', 'Trichy', 'Tiruchirappalli', 'Vellore', 'Pondicherry', 'Puducherry', 'Thiruvananthapuram',
            'Trivandrum', 'Kozhikode', 'Salem', 'Erode', 'Tirupur', 'Chandigarh', 'Kanpur', 'Agra', 'Varanasi', 'Patna', 'Ranchi',
            'Bhubaneswar', 'Guwahati', 'Dehradun', 'Raipur', 'Vijayawada', 'Guntur', 'Nellore', 'Kakinada', 'Warangal', 'Aurangabad',
            'Nashik', 'Amritsar', 'Jalandhar', 'Ludhiana', 'Goa', 'Panaji', 'Mangalore', 'Mangaluru', 'Belgaum', 'Hubli', 'Udaipur',
            'Jodhpur', 'Rajkot', 'Vadodara', 'Jamshedpur', 'Siliguri', 'Durgapur', 'Meerut', 'Ghaziabad', 'Faridabad',
        ]
        for city in indian_cities:
            if re.search(rf'\b{re.escape(city)}\b', content, re.I):
                if city == 'Bengaluru': return 'Bangalore'
                if city == 'Mysuru': return 'Mysore'
                if city == 'Tiruchirappalli': return 'Trichy'
                if city == 'Vizag': return 'Visakhapatnam'
                if city == 'Gurugram': return 'Gurgaon'
                if city == 'New Delhi': return 'Delhi'
                if city == 'Trivandrum': return 'Thiruvananthapuram'
                return city
        return ''

    def _fallback_parse(self, content: str) -> dict:
        name = self._extract_name(content)
        email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", content)
        phone_match = None
        for m in re.finditer(r"(?:\+\d{1,3}[\s\-]?)?\d[\d\s\-\.\(\)]{8,}\d", content):
            candidate = m.group().strip()
            digits = re.sub(r"\D", "", candidate)
            if re.match(r"^(19|20)\d{2}.{0,3}(19|20)\d{2}$", candidate.strip()):
                continue
            if len(digits) < 7:
                continue
            phone_match = candidate
            break
        skills = list(extract_matched_skills(content))
        edu_lines = [
            l.strip() for l in content.split("\n")
            if re.search(r"\b(b\.?tech|b\.?e|m\.?tech|mba|bachelor|master|phd|diploma|university|college|institute|school|icam|loyola|iit|nit|hsc|sslc|10th|12th|higher secondary|secondary)\b", l, re.IGNORECASE)
        ]
        educations = [self._classify_education(l) for l in edu_lines[:3]]
        location = self._extract_location(content)
        return {
            "name": name or "Unknown",
            "email": email.group() if email else "",
            "phone": phone_match or "",
            "location": location,
            "country": "India" if location else "",
            "title": (content.split("\n")[0] if content else ""),
            "skills": skills, "softSkills": [], "tools": [],
            "workExperiences": [], "internships": [], "educations": educations,
            "projects": [], "certifications": [], "competitions": [],
            "summary": content[:200],
            "languages": [], "awards": []
        }

    def _classify_education(self, line: str) -> dict:
        is_school = re.search(r"\b(hsc|sslc|10th|12th|higher secondary|secondary school)\b", line, re.IGNORECASE)
        year_match = re.search(r"(\d{4})", line)
        year = year_match.group(1) if year_match else ""
        if is_school:
            return {"type": "school", "school": line, "class": "", "year": year, "percentage": ""}
        return {"type": "college", "college": line, "degree": "", "year": year, "gpa": ""}


resume_parser_brain = ResumeParserBrain()
