import os
import time
import json
import requests
from typing import List, Dict, Any, Set
from plagiarism_ai.schemas.plagiarism import (
    PlagiarismReport, PlagiarismMatch, QwenAnalysisResult
)
from plagiarism_ai.utils.text_chunker import TextChunker
from plagiarism_ai.utils.similarity import Similarity
from plagiarism_ai.services.crawler_service import CrawlerService
from plagiarism_ai.services.search_service import SearchService

QWEN_PLAGIARISM_PROMPT = """
You are an expert AI trained to detect plagiarism in Tamil text. 
You will be provided with an "Original/External text" (source from the web) and an "Internal/Suspicious text" (text to check).
Your task is to analyze the semantic similarity between the two texts and determine if the Internal text is plagiarized from the External text.

Consider the following:
- Direct copy-pasting (exact matches).
- Paraphrasing (rewriting the same concept with different words).
- Translation-based plagiarism (if any text seems translated but maintains exact structure).
- Minor differences (like changing a few words) still count as plagiarism if the core structure and meaning are identical.

Return the result ONLY as a JSON object with the following structure:
{
  "isPlagiarized": boolean,
  "confidenceScore": number,
  "reasoning": string,
  "suggestedAction": string
}

Do not include any other text, markdown blocks like ```json, or explanations outside the JSON object. Just the raw JSON.
"""

class PlagiarismService:
    def __init__(self):
        self.lm_base_url = os.getenv("LM_STUDIO_BASE_URL", "http://127.0.0.1:1234/v1")
        self.embed_model = os.getenv("LM_STUDIO_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
        self.chat_model = os.getenv("LM_STUDIO_CHAT_MODEL", "local-model")
        self.api_key = os.getenv("LM_STUDIO_API_KEY", "lm-studio")
        self.search_service = SearchService()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            response = requests.post(
                f"{self.lm_base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.embed_model, "input": texts},
                timeout=120
            )
            response.raise_for_status()
            data = response.json().get("data", [])
            # Sort by index just in case the API returns them out of order
            data.sort(key=lambda x: x["index"])
            return [item["embedding"] for item in data]
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            raise e

    def _get_overlap_ratio(self, text1: str, text2: str) -> float:
        # Fast lexical pre-filter to skip embedding completely unrelated text
        words1 = set(text1.split())
        words2 = set(text2.split())
        if not words1 or not words2:
            return 0.0
        return len(words1.intersection(words2)) / min(len(words1), len(words2))

    def analyze_plagiarism(self, internal_text: str, external_text: str) -> QwenAnalysisResult:
        prompt = f'Original/External text:\n"{external_text}"\n\nInternal/Suspicious text:\n"{internal_text}"\n\nAnalyze the similarity and return the JSON report.'
        
        try:
            response = requests.post(
                f"{self.lm_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.chat_model,
                    "messages": [
                        {"role": "system", "content": QWEN_PLAGIARISM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1
                },
                timeout=120
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            
            # Manually clean markdown if the model wrapped the JSON in markdown blocks
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            data = json.loads(content)
            return QwenAnalysisResult(
                isPlagiarized=data.get("isPlagiarized", False),
                confidenceScore=float(data.get("confidenceScore", 0)),
                reasoning=data.get("reasoning", "Parsed from LLM"),
                suggestedAction=data.get("suggestedAction", "")
            )
        except Exception as e:
            err_msg = str(e)
            if 'response' in locals() and hasattr(response, 'text'):
                err_msg += f" - Details: {response.text}"
            print(f"Error analyzing plagiarism: {err_msg}")
            return QwenAnalysisResult(
                isPlagiarized=False,
                confidenceScore=0,
                reasoning=f"Error communicating with LM Studio: {err_msg}"
            )

    def _find_matches(self, internal_chunks: List[Any], crawled_news: Dict[str, str], sources_found: Set[str]) -> List[PlagiarismMatch]:
        candidate_external_chunks = []
        # Sort crawled_news URLs based on TARGET_SOURCES priority order
        target_order = {url: idx for idx, url in enumerate(CrawlerService.TARGET_SOURCES)}
        sorted_urls = sorted(crawled_news.keys(), key=lambda u: target_order.get(u, 999))

        for url in sorted_urls:
            text = crawled_news[url]
            ext_chunks = TextChunker.chunk_text(text)
            for ext_chunk in ext_chunks:
                for internal_chunk in internal_chunks:
                    if self._get_overlap_ratio(internal_chunk.text, ext_chunk.text) > 0.25:
                        candidate_external_chunks.append({
                            "url": url,
                            "text": ext_chunk.text,
                            "embedding": None
                        })
                        break # Move to next external chunk if it matches any internal chunk

        # Generate embeddings for the vastly reduced candidate list
        batch_size = 15
        for i in range(0, len(candidate_external_chunks), batch_size):
            batch = candidate_external_chunks[i:i + batch_size]
            batch_texts = [b["text"] for b in batch]
            try:
                embeddings = self.get_embeddings(batch_texts)
                for j, emb in enumerate(embeddings):
                    batch[j]["embedding"] = emb
            except Exception:
                continue

        # Filter out any candidates that failed to embed
        valid_external_chunks = [c for c in candidate_external_chunks if c["embedding"] is not None]

        matches: List[PlagiarismMatch] = []
        for internal_chunk in internal_chunks:
            highest_similarity = 0.0
            most_similar_text = ""
            matched_url = ""

            for ext_data in valid_external_chunks:
                sim = Similarity.cosine_similarity(internal_chunk.embedding, ext_data["embedding"])
                
                # If exact substring match, assign top similarity
                internal_clean = internal_chunk.text.strip()
                ext_clean = ext_data["text"].strip()
                if internal_clean in ext_clean or ext_clean in internal_clean:
                    sim = max(sim, 0.9999)

                # Only overwrite if strictly higher similarity + threshold to respect target source priority
                if sim > (highest_similarity + 0.0005):
                    highest_similarity = sim
                    most_similar_text = ext_data["text"]
                    matched_url = ext_data["url"]

            if highest_similarity > 0.85:
                sources_found.add(matched_url)
                analysis = self.analyze_plagiarism(internal_chunk.text, most_similar_text)
                
                # Fallback: If LLM timed out or returned False, trust high vector similarity!
                if not analysis.isPlagiarized:
                    analysis = QwenAnalysisResult(
                        isPlagiarized=True,
                        confidenceScore=highest_similarity,
                        reasoning=f"High semantic vector similarity match ({highest_similarity * 100:.1f}%) detected with external source.",
                        suggestedAction="Review content for originality."
                    )

                analysis.matchedUrl = matched_url
                matches.append(PlagiarismMatch(
                    textChunk=internal_chunk.text,
                    matchedUrl=matched_url,
                    similarityScore=highest_similarity,
                    analysis=analysis
                ))
                break # Stop checking this chunk after first match to save time
        return matches

    def check_article(self, article_text: str) -> PlagiarismReport:
        start_time = int(time.time() * 1000)
        internal_chunks = TextChunker.chunk_text(article_text)

        if not internal_chunks:
            return self._create_clean_report(0)

        # 1. Fetch Latest News from predefined Tamil sources (Local Crawl)
        crawled_news = CrawlerService.get_latest_news()

        # 2. Embed the internal chunks first
        try:
            internal_embeddings = self.get_embeddings([chunk.text for chunk in internal_chunks])
            for i, chunk in enumerate(internal_chunks):
                chunk.embedding = internal_embeddings[i]
        except Exception:
            return self._create_clean_report(int(time.time() * 1000) - start_time)

        sources_found = set()
        matches = []
        
        # 3. Check local crawl first
        if crawled_news:
            matches = self._find_matches(internal_chunks, crawled_news, sources_found)

        # 4. OPTION A: Fallback to Google Search if no matches found locally
        if len(matches) == 0:
            # Pick the longest chunk to search for better deep-link accuracy, but truncate to 32 words to respect Google's limit
            longest_chunk = max(internal_chunks, key=lambda c: len(c.text))
            query_text = " ".join(longest_chunk.text.split()[:20]) # Take first 20 words
            # Remove quotes from the query text itself to prevent malformed searches
            query_text = query_text.replace('"', '').replace("'", '')
            
            search_results = self.search_service.search(f'"{query_text}"')
            
            if search_results:
                fallback_crawled_news = {}
                for res in search_results:
                    scraped_text = CrawlerService._scrape_single_url(res.link)
                    if scraped_text:
                        fallback_crawled_news[res.link] = scraped_text
                
                if fallback_crawled_news:
                    fallback_matches = self._find_matches(internal_chunks, fallback_crawled_news, sources_found)
                    matches.extend(fallback_matches)

        duration_ms = int(time.time() * 1000) - start_time
        is_clean = len(matches) == 0
        overall_score = len(matches) / len(internal_chunks) if internal_chunks else 0.0

        return PlagiarismReport(
            overallPlagiarismScore=overall_score,
            isClean=is_clean,
            matches=matches,
            sourcesFound=list(sources_found),
            durationMs=duration_ms
        )

    def _create_clean_report(self, duration_ms: int) -> PlagiarismReport:
        return PlagiarismReport(
            overallPlagiarismScore=0.0,
            isClean=True,
            matches=[],
            sourcesFound=[],
            durationMs=duration_ms
        )
