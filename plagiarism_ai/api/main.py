from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from plagiarism_ai.schemas.plagiarism import PlagiarismRequest, PlagiarismReport
from plagiarism_ai.services.plagiarism_service import PlagiarismService

app = FastAPI(title="Plagiarism AI Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = PlagiarismService()

@app.post("/plagiarism/check", response_model=PlagiarismReport)
async def check_plagiarism(request: PlagiarismRequest):
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required for plagiarism checking.")
    
    try:
        report = service.check_article(request.text)
        return report
    except Exception as e:
        print(f"Error checking plagiarism: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during plagiarism detection.")

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "plagiarism_ai"}

# Run with: uvicorn plagiarism_ai.api.main:app --port 8001
