"""Re-ingest ZyncJobs knowledge base after cleanup."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recruitment_ai.vector.ingest import ingester

KNOWLEDGE_PAGES = [
    {
        "markdown": "# ZyncJobs Platform Overview\n\nZyncJobs is an AI-powered recruitment platform that connects candidates with employers.\n\n## Key Features\n- **AI-Powered Job Matching**: Smart algorithms match candidates to relevant positions\n- **Resume Analysis**: Automated resume parsing with ATS score calculation\n- **Career Development**: Career roadmaps, skill assessments, interview preparation\n- **Recruiter Tools**: Candidate search, shortlisting, and evaluation\n- **Real-time Chat**: Instant communication between candidates and employers\n\n## For Candidates\n- Create a professional resume with AI assistance\n- Get personalized job recommendations\n- Track applications and interview status\n- Receive career advice and skill gap analysis\n\n## For Employers\n- Post jobs and manage listings\n- Search and filter candidates by skills, experience, location\n- Use AI recruiter assistant for candidate evaluation\n- Access analytics and insights",
        "relative_path": "/platform-overview",
        "title": "ZyncJobs Platform Overview",
        "category": "platform"
    },
    {
        "markdown": "# Getting Started with ZyncJobs\n\n## Creating an Account\n1. Visit zyncjobs.com and click 'Sign Up'\n2. Choose your account type: Candidate or Employer\n3. Fill in your email, name, and create a password\n4. Verify your email address\n\n## For Candidates\nComplete your profile with skills, experience, and education. Upload your resume for automatic parsing and ATS scoring. Browse jobs and apply with one click.\n\n## For Employers\nSet up your company profile with detailed information about your organization. Post job openings and start receiving applications. Use AI-powered tools to find the best candidates.",
        "relative_path": "/getting-started",
        "title": "Getting Started",
        "category": "getting-started"
    },
    {
        "markdown": "# AI-Powered Job Matching\n\nZyncJobs uses advanced AI algorithms to match candidates with relevant job opportunities.\n\n## How It Works\n1. **Profile Analysis**: AI analyzes candidate skills, experience, and preferences\n2. **Job Matching**: Matches candidates against job requirements using skill-based scoring\n3. **Recommendations**: Provides personalized job recommendations\n4. **Match Score**: Shows percentage match between candidate and job\n\n## Match Score Factors\n- Skill match (40% weight)\n- Experience relevance (30% weight)\n- Education alignment (15% weight)\n- Location preference (10% weight)\n- Other factors (5% weight)",
        "relative_path": "/ai-job-matching",
        "title": "AI Job Matching",
        "category": "features"
    },
    {
        "markdown": "# Resume Builder and Analysis\n\nZyncJobs offers comprehensive resume tools powered by AI.\n\n## Resume Builder\nCreate professional resumes with AI assistance:\n- Choose from multiple templates\n- AI-powered content suggestions\n- Real-time preview and formatting\n- Export as PDF\n\n## Resume Analysis\nGet detailed analysis of your resume:\n- **ATS Score**: Check how your resume performs against Applicant Tracking Systems\n- **Keyword Analysis**: Identify missing keywords for your target role\n- **Format Check**: Ensure proper section headings and structure\n- **Suggestions**: Receive actionable improvement recommendations\n\n## Resume Parsing\nUpload any resume format (PDF, DOCX, TXT) for automatic extraction of:\n- Personal information\n- Work experience\n- Education history\n- Skills and certifications\n- Contact details",
        "relative_path": "/resume-tools",
        "title": "Resume Builder & Analysis",
        "category": "features"
    },
    {
        "markdown": "# ATS Score and Optimization\n\nATS (Applicant Tracking System) score measures how well your resume performs against automated screening systems.\n\n## What is ATS?\nATS is software used by employers to filter and rank resumes before human review. Most large companies use ATS to manage high volumes of applications.\n\n## Understanding Your ATS Score\n- **0-40**: Poor - Resume needs significant improvement\n- **40-60**: Below Average - Several areas need attention\n- **60-75**: Good - Resume is competitive\n- **75-90**: Very Good - Strong resume with minor improvements needed\n- **90-100**: Excellent - Highly optimized resume\n\n## How to Improve ATS Score\n1. Use standard section headings (Experience, Education, Skills)\n2. Include keywords from the job description\n3. Use simple formatting without tables or columns\n4. Quantify achievements with numbers and percentages\n5. Save as PDF or DOCX format",
        "relative_path": "/ats-score",
        "title": "ATS Score and Optimization",
        "category": "features"
    },
    {
        "markdown": "# Career Development Features\n\nZyncJobs provides comprehensive career development tools.\n\n## Career Roadmaps\nAI-generated career paths that help you plan your professional journey:\n- Current role to target role progression\n- Required skills and certifications\n- Estimated timeline for each step\n- Milestone tracking\n\n## Skill Assessments\nTest your knowledge across various skills:\n- Multiple-choice questions for technical and soft skills\n- Difficulty levels from beginner to advanced\n- Instant scoring and feedback\n- Identify areas for improvement\n\n## Skill Gap Analysis\nIdentify gaps between your current skills and target role requirements. Get recommendations for courses, certifications, and learning resources.\n\n## Interview Preparation\nPractice with AI-generated interview questions:\n- Technical questions for your role\n- Behavioral questions with expected answers\n- System design scenarios\n- Tips and best practices",
        "relative_path": "/career-development",
        "title": "Career Development Features",
        "category": "features"
    },
    {
        "markdown": "# Recruiter Tools and Features\n\nZyncJobs provides powerful tools for recruiters and employers.\n\n## Job Posting\n- Create detailed job descriptions with AI assistance\n- Set requirements, responsibilities, and benefits\n- Manage multiple job listings\n- Bulk job import\n\n## Candidate Search\nSearch across all candidates with powerful filters:\n- Skills and technologies\n- Experience level and years\n- Location and remote preference\n- Salary expectations\n- Education and certifications\n\n## Candidate Evaluation\nAI-powered candidate evaluation:\n- Automated shortlisting based on job requirements\n- Match scores for each candidate\n- Comparison views\n- Interview scheduling\n\n## Team Collaboration\n- Share candidate profiles with team members\n- Collaborative evaluation and notes\n- Pipeline management\n- Activity tracking",
        "relative_path": "/recruiter-tools",
        "title": "Recruiter Tools",
        "category": "employer"
    },
    {
        "markdown": "# Job Descriptions on ZyncJobs\n\nCreating effective job descriptions is crucial for attracting the right talent.\n\n## Key Elements of a Job Description\n1. **Job Title**: Clear and specific title\n2. **Company Information**: About the company and culture\n3. **Location**: Office location or remote options\n4. **Job Type**: Full-time, part-time, contract, internship\n5. **Experience Level**: Entry, mid, senior, lead, executive\n6. **Salary Range**: Transparent compensation information\n7. **Responsibilities**: Key duties and expectations\n8. **Requirements**: Required qualifications and skills\n9. **Benefits**: Perks and benefits offered\n\n## AI-Generated Job Descriptions\nUse ZyncJobs AI to generate professional job descriptions:\n- Input basic requirements\n- AI generates complete JD with proper sections\n- Professional and inclusive language\n- ATS-optimized formatting",
        "relative_path": "/job-descriptions",
        "title": "Job Descriptions",
        "category": "employer"
    },
    {
        "markdown": "# Pricing and Plans\n\nZyncJobs offers flexible pricing for both candidates and employers.\n\n## For Candidates (Free)\n- Create and manage profile\n- Upload and parse resumes\n- Browse and apply to jobs\n- Get AI-powered job recommendations\n- Access career development tools\n- Basic ATS score check\n\n## For Employers\n### Starter Plan\n- Up to 5 active job postings\n- Basic candidate search\n- Team collaboration\n\n### Professional Plan\n- Up to 25 active job postings\n- Advanced candidate search and filtering\n- AI recruiter assistant\n- Analytics and insights\n- Priority support\n\n### Enterprise Plan\n- Unlimited job postings\n- Full AI feature access\n- Custom integrations\n- Dedicated account manager\n- API access\n- Custom reporting",
        "relative_path": "/pricing",
        "title": "Pricing and Plans",
        "category": "general"
    },
    {
        "markdown": "# Privacy and Security\n\nZyncJobs is committed to protecting user data and privacy.\n\n## Data Protection\n- All data encrypted in transit (TLS 1.3)\n- Data stored securely with encryption at rest\n- Regular security audits and penetration testing\n- GDPR compliant\n- SOC 2 certified\n\n## User Privacy\n- Control what information is visible on your profile\n- Manage notification preferences\n- Choose whether to receive job alerts\n- Request data download or account deletion\n- Cookie consent management\n\n## Candidate Data\n- Resume data stored securely\n- Control which employers can view your profile\n- Set visibility preferences\n- Data retention policies\n\n## Employer Data\n- Company information verified\n- Job postings reviewed for quality\n- Secure payment processing\n- Confidential candidate data handling",
        "relative_path": "/privacy-security",
        "title": "Privacy and Security",
        "category": "general"
    },
]

from types import SimpleNamespace

pages = [SimpleNamespace(**p) for p in KNOWLEDGE_PAGES]
print(f"Ingesting {len(pages)} knowledge pages...")
total = ingester.ingest_pages(pages)
print(f"Total chunks ingested: {total}")
print(f"Knowledge base count: {ingester.count}")
