import os
import requests
from typing import List, Optional
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

class SearchResult(BaseModel):
    title: str
    link: str
    snippet: str

class SearchService:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.search_cx = os.getenv("GOOGLE_SEARCH_CX")
        self.base_url = "https://customsearch.googleapis.com/customsearch/v1"

    def search(self, query: str, num_results: int = 3) -> List[SearchResult]:
        if not self.api_key or not self.search_cx:
            print("Google Search API credentials are not set in environment variables.")
            return []

        try:
            params = {
                "key": self.api_key,
                "cx": self.search_cx,
                "q": query,
                "num": num_results
            }
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            items = response.json().get("items", [])
            results = []
            for item in items:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    link=item.get("link", ""),
                    snippet=item.get("snippet", "")
                ))
            return results
        except Exception as e:
            print(f"Error during Google Search in SearchService: {e}")
            return []
