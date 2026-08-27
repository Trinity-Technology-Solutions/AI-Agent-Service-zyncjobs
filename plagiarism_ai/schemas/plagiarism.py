from typing import List, Optional
from pydantic import BaseModel

class TextChunk(BaseModel):
    id: str
    text: str
    start_index: int
    end_index: int
    embedding: Optional[List[float]] = None

class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    content: Optional[str] = None

class QwenAnalysisResult(BaseModel):
    isPlagiarized: bool
    confidenceScore: float
    reasoning: str
    matchedUrl: Optional[str] = None
    suggestedAction: Optional[str] = None

class PlagiarismMatch(BaseModel):
    textChunk: str
    matchedUrl: str
    similarityScore: float
    analysis: QwenAnalysisResult

class PlagiarismReport(BaseModel):
    overallPlagiarismScore: float
    isClean: bool
    matches: List[PlagiarismMatch]
    sourcesFound: List[str]
    durationMs: int

class PlagiarismRequest(BaseModel):
    text: str
