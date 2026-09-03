import requests
from bs4 import BeautifulSoup
from typing import Dict
from urllib.parse import urljoin
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

            # 1. Discover actual article links from the homepage
            article_links = []
            for a in soup.find_all('a', href=True):
                href = a['href']
                full_link = urljoin(url, href)
                if (full_link.startswith('http') and 
                    len(full_link) > len(url) + 10 and 
                    not full_link.endswith(('.jpg', '.png', '.jpeg', '.pdf', '.svg', '#')) and
                    not any(x in full_link for x in ['/tag/', '/category/', '/author/', '/contact', '/about', '/privacy', '/terms'])):
                    if full_link not in article_links:
                        article_links.append(full_link)

            # Limit to top 5 most recent article links per source to keep speed fast
            article_links = article_links[:5]

            # Scrape each discovered specific article link
            for art_link in article_links:
                try:
                    art_resp = requests.get(art_link, headers=headers, timeout=3)
                    if art_resp.status_code == 200:
                        art_soup = BeautifulSoup(art_resp.text, 'html.parser')
                        for el in art_soup(["script", "style", "noscript", "nav", "header", "footer", "aside"]):
                            el.extract()
                        body_text = art_soup.get_text(separator=' ', strip=True)
                        if len(body_text) > 200:
                            results[art_link] = body_text
                except Exception:
                    continue

            # Fallback to main page text if deep links could not be fetched
            if not results:
                for el in soup(["script", "style", "noscript", "nav", "header", "footer", "aside"]):
                    el.extract()
                text = soup.get_text(separator=' ', strip=True)
                if text and len(text) > 300:
                    results[url] = text

            return results
        except Exception as e:
            print(f"Crawler failed to scrape {url}: {e}")
            return {}

    @staticmethod
    def get_latest_news() -> Dict[str, str]:
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
