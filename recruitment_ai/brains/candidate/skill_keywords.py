"""Shared skill keyword constants used across all candidate brains.
Single source of truth — prevents drift between resume parser, ATS, and job matching.
"""
import re

SKILL_KEYWORDS = [
    "python", "java", "javascript", "js", "typescript", "react", "node", "nodejs",
    "sql", "aws", "docker", "kubernetes", "git", "linux", "agile", "scrum",
    "rest", "api", "html", "css", "go", "rust", "c++", "c#", ".net",
    "spring", "django", "flask", "fastapi", "postgresql", "mongodb", "redis",
    "swift", "kotlin", "ruby", "php", "vue", "angular", "svelte",
    "tensorflow", "pytorch", "machine learning", "deep learning",
    "graphql", "grpc", "kafka", "rabbitmq", "terraform", "ansible",
    "jenkins", "github actions", "ci/cd",
    # Soft skills
    "communication", "leadership", "teamwork", "problem solving",
    "adaptability", "adaptable", "time management", "critical thinking",
    "creativity", "collaboration", "interpersonal", "negotiation", "mentoring",
]

SKILL_NORMALIZATION = {
    "js": "javascript",
    "c++": "c++",
    "c#": "c#",
    ".net": ".net",
}

# Expanded skill variations and aliases for better matching
SKILL_VARIATIONS = {
    # Javascript variations
    "javascript": ["javascript", "js"],
    "js": ["js", "javascript"],
    # Java variations
    "java": ["java"],
    # Python variations
    "python": ["python", "python3"],
    # React variations
    "react": ["react", "reactjs", "react.js", "react js"],
    # Node/JS variations
    "nodejs": ["nodejs", "node", "node.js", "node js"],
    # SQL variations
    "sql": ["sql", "mysql", "postgresql", "sqlite", "tsql", "t-sql"],
    # CSS variations
    "css": ["css", "css3", "stylesheet", "styles"],
    # HTML variations
    "html": ["html", "html5", "html5", "markup"],
    # Angular variations
    "angular": ["angular", "angularjs", "angular js", "angular.js"],
    # Vue variations
    "vue": ["vue", "vuejs", "vue.js", "vue js"],
    # Docker variations
    "docker": ["docker", "containerization"],
    # Kubernetes variations
    "kubernetes": ["kubernetes", "k8s", "k-8s"],
    # Git variations
    "git": ["git", "version control", "github"],
    # Soft skill variations
    "communication": ["communication", "communicating", "verbal communication", "written communication"],
    "leadership": ["leadership", "leading", "team lead", "leading teams"],
    "teamwork": ["teamwork", "team work", "team player", "collaborative", "collaboration"],
    "problem solving": ["problem solving", "problem-solving", "analytical thinking", "critical thinking"],
    "adaptability": ["adaptability", "adaptable", "flexible", "adapt"],
    "time management": ["time management", "time-management", "organizational skills", "organised"],
    "critical thinking": ["critical thinking", "critical-thinking", "analytical", "analysis"],
    "creativity": ["creativity", "creative", "innovative", "innovation"],
    "collaboration": ["collaboration", "collaborative", "cross-functional", "cross functional"],
    "interpersonal": ["interpersonal", "inter-personal", "people skills"],
    "negotiation": ["negotiation", "negotiating"],
    "mentoring": ["mentoring", "mentor", "coaching", "coached"],
}


def extract_matched_skills(text: str) -> set[str]:
    """Extract which SKILL_KEYWORDS appear in text, case-insensitive.
    Handles skill variations (e.g., JS → javascript) and normalizes output."""
    if not text:
        return set()
    
    lower = text.lower()
    matched = set()
    
    # First, collect ALL possible patterns from variations
    all_patterns = {}
    for normalized, variations in SKILL_VARIATIONS.items():
        for var in variations:
            all_patterns[var] = normalized
            # Add with word boundaries
            all_patterns[rf"\b{var}\b"] = normalized
            # Also add without boundaries for within-word matches
            if var not in all_patterns:
                all_patterns[var] = normalized
    
    # Add the base keywords as well
    for kw in SKILL_KEYWORDS:
        normalized = SKILL_NORMALIZATION.get(kw, kw)
        all_patterns[normalized.lower()] = normalized
        all_patterns[rf"\b{normalized.lower()}\b"] = normalized
    
    # Then check all patterns
    for pattern, normalized in all_patterns.items():
        if re.search(pattern, lower):
            matched.add(normalized)
    
    return matched
