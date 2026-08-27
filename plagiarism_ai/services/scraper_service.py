import requests
from bs4 import BeautifulSoup
from typing import Optional

class ScraperService:
    @staticmethod
    def scrape_url(url: str) -> Optional[str]:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove irrelevant elements
            for element in soup(["script", "style", "noscript", "nav", "header", "footer", "aside"]):
                element.extract()
            
            # Find elements by class is more complex in BS4 in one pass, so we do it explicitly
            for element in soup.find_all(class_=["ads", "sidebar"]):
                element.extract()
            for element in soup.find_all(attrs={"role": "navigation"}):
                element.extract()

            text_parts = []
            for element in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "article"]):
                text = element.get_text(strip=True)
                if text:
                    text_parts.append(text)

            # Also check content/post-content classes
            for element in soup.find_all(class_=["content", "post-content"]):
                text = element.get_text(strip=True)
                if text:
                    text_parts.append(text)

            joined_text = ' '.join(text_parts)
            import re
            return re.sub(r'[\r\n\s]+', ' ', joined_text).strip()

        except Exception as e:
            print(f"Error scraping URL {url}: {e}")
            return None
