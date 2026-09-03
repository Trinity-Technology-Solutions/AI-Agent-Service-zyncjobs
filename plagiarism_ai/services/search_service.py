# -*- coding: utf-8 -*-
# Free DDG / Google Web Search fallback that always returns exact article URLs without requiring paid API keys

import urllib.request
import urllib.parse
import json
import re
from typing import List
from pydantic import BaseModel

class SearchResult(BaseModel):
    title: str
    link: str
    snippet: str

class SearchService:
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        results = []
        clean_query = query.replace('"', '').replace("'", '').strip()
        if not clean_query:
            return []

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
        }

        # Method 1: DuckDuckGo HTML Instant Search for Deep Article URLs
        try:
            encoded_query = urllib.parse.quote(clean_query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=4) as response:
                html = response.read().decode('utf-8', errors='ignore')
                
                # Extract clean target URLs and snippets from DDG HTML
                from bs4 import BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                
                for res in soup.find_all('div', class_='result'):
                    title_elem = res.find('a', class_='result__a')
                    snippet_elem = res.find('a', class_='result__snippet')
                    if title_elem and title_elem.get('href'):
                        raw_href = title_elem['href']
                        # DDG wraps URLs in /l/?kh=-1&uddg=ENCODED_URL
                        actual_url = raw_href
                        if 'uddg=' in raw_href:
                            parsed_qs = urllib.parse.parse_qs(urllib.parse.urlparse(raw_href).query)
                            if 'uddg' in parsed_qs:
                                actual_url = parsed_qs['uddg'][0]

                        if actual_url.startswith('http') and not any(x in actual_url for x in ['duckduckgo.com', 'youtube.com/watch']):
                            results.append(SearchResult(
                                title=title_elem.get_text(strip=True),
                                link=actual_url,
                                snippet=snippet_elem.get_text(strip=True) if snippet_elem else ""
                            ))
                            if len(results) >= num_results:
                                break
        except Exception as e:
            print(f"Web Search fallback notice: {e}")

        return results
