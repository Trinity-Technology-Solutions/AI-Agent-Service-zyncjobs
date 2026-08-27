import requests
from bs4 import BeautifulSoup
from typing import Dict
import concurrent.futures

class CrawlerService:
    # A hardcoded list of major Tamil news websites to crawl for plagiarism detection
    TARGET_SOURCES = [
        "https://www.dailythanthi.com/",
        "https://www.dinakaran.com/",
        "https://www.dinamalar.com/",
        "https://www.dinamani.com/",
        "https://www.vikatan.com/",
        "https://www.puthiyathalaimurai.com/",
        "https://www.polimernews.com/",
        "https://tamil.news18.com/",
        "https://tamil.oneindia.com/",
        "https://tamil.samayam.com/",
        "https://tamil.thehindu.com/",
        "https://www.hindutamil.in/",
        "https://www.bbc.com/tamil/",
        "https://ibctamil.com/",
        "https://tamil.webdunia.com/",
        "https://www.thanthitv.com/",
        "https://sathiyam.tv/",
        "https://www.maalaimalar.com/",
        "https://tamil.indianexpress.com/",
        "https://tamil.abplive.com/",
        "https://www.virakesari.lk/",
        "https://tamil.adaderana.lk/",
        "https://www.tamilmirror.lk/",
        "https://www.tamilmurasu.com.sg/"
    ]

    @staticmethod
    def _scrape_single_url(url: str) -> Dict[str, str]:
        results: Dict[str, str] = {}
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # 1. Extract specific article URLs and headlines from link tags
            from urllib.parse import urljoin
            for a in soup.find_all('a', href=True):
                headline_text = a.get_text(strip=True)
                if len(headline_text) > 15:
                    link = urljoin(url, a['href'])
                    if link.startswith('http') and not link.endswith('#') and len(link) > len(url) + 5:
                        results[link] = headline_text

            # 2. Extract full page text under main homepage URL
            for element in soup(["script", "style", "noscript", "nav", "header", "footer"]):
                element.extract()
            for element in soup.find_all(class_=["ads", "sidebar", "menu", "footer"]):
                element.extract()

            text = soup.get_text(separator=' ', strip=True)
            if text:
                results[url] = text

            return results
        except Exception as e:
            print(f"Crawler failed to scrape {url}: {e}")
            return {}

    @staticmethod
    def get_latest_news() -> Dict[str, str]:
        """
        Fetches the front pages of target news websites in parallel.
        Returns a dictionary mapping article URLs to text content.
        """
        all_news: Dict[str, str] = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(CrawlerService.TARGET_SOURCES)) as executor:
            future_to_url = {executor.submit(CrawlerService._scrape_single_url, url): url for url in CrawlerService.TARGET_SOURCES}
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    site_news = future.result()
                    if site_news:
                        all_news.update(site_news)
                except Exception as exc:
                    print(f'{url} generated an exception: {exc}')
                    
        return all_news
