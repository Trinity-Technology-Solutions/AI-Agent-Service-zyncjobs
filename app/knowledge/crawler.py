import json
import asyncio
import time
from pathlib import Path
from typing import Optional
import httpx
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from app.utils.logger import logger


DOCS_DIR = Path(__file__).parent / "crawled"
SITEMAP_URLS = [
    "https://www.zyncjobs.com/sitemap.xml",
]
BASE_URL = "https://www.zyncjobs.com"


class CrawledPage:
    def __init__(self, url: str, title: str, category: str, markdown: str):
        self.url = url
        self.title = title
        self.category = category
        self.markdown = markdown

    @property
    def relative_path(self) -> str:
        return self.url.replace(BASE_URL, "").rstrip("/") or "/"

    @property
    def filename(self) -> str:
        clean = self.relative_path.strip("/").replace("/", "_").replace("-", "_") or "home"
        return f"{clean}.md"

    def to_metadata(self) -> dict:
        return {
            "url": self.relative_path,
            "title": self.title,
            "category": self.category,
        }


def _infer_category(url: str) -> str:
    path = url.replace(BASE_URL, "").lower()
    if "/candidate/" in path or "/job-seeker/" in path:
        return "Candidate"
    if "/employer/" in path or "/recruiter/" in path or "/company/" in path:
        return "Employer"
    if "/blog/" in path or "/resources/" in path:
        return "Blog"
    if "/faq" in path or "/help" in path:
        return "Support"
    if "/about" in path or "/contact" in path:
        return "Company"
    if "/auth/" in path or "/login" in path or "/register" in path:
        return "Auth"
    return "General"


def _path_to_title(path: str) -> str:
    parts = path.strip("/").replace("-", " ").replace("_", " ").split("/")
    return " ".join(p.title() for p in parts if p) or "Home"


def _html_to_markdown(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "nav", "footer", "noscript"]):
        tag.decompose()
    lines = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "a", "code", "pre", "td", "th", "span"]):
        text = el.get_text(strip=True)
        if not text:
            continue
        if el.name.startswith("h"):
            level = int(el.name[1])
            lines.append("#" * level + " " + text)
        elif el.name == "li":
            lines.append("- " + text)
        elif el.name in ("code", "pre"):
            lines.append("`" + text + "`")
        else:
            lines.append(text)
    return "\n\n".join(lines)


def _discover_urls() -> list[str]:
    urls = set()
    client = httpx.Client(timeout=15)
    for sitemap_url in SITEMAP_URLS:
        try:
            res = client.get(sitemap_url, timeout=15)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "xml")
            for loc in soup.find_all("loc"):
                url = loc.get_text(strip=True)
                if url.startswith(BASE_URL):
                    urls.add(url)
            logger.info(f"Crawler | Discovered {len(urls)} URLs from {sitemap_url}")
        except Exception as e:
            logger.warn(f"Crawler | Failed sitemap {sitemap_url}: {e}")
    return list(urls)


async def crawl_async(sitemap_only: bool = False, delay: float = 0.5, max_pages: Optional[int] = None) -> list[CrawledPage]:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    urls = _discover_urls()
    if not urls:
        logger.warn("Crawler | No URLs discovered from sitemaps")
        return []

    if max_pages and len(urls) > max_pages:
        urls = urls[:max_pages]

    pages: list[CrawledPage] = []
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ZyncJobsCrawler/1.0",
            viewport={"width": 1280, "height": 800},
        )

        for i, url in enumerate(urls):
            try:
                page = await context.new_page()
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await asyncio.sleep(1)
                html = await page.content()
                await page.close()

                soup = BeautifulSoup(html, "html.parser")
                markdown = _html_to_markdown(soup)
                title_tag = soup.find("title")
                title = title_tag.get_text(strip=True) if title_tag else _path_to_title(url)
                category = _infer_category(url)

                cp = CrawledPage(url=url, title=title, category=category, markdown=markdown)
                pages.append(cp)
                if not sitemap_only:
                    (DOCS_DIR / cp.filename).write_text(markdown, encoding="utf-8")
                logger.info(f"Crawler | [{i+1}/{len(urls)}] {cp.relative_path} — {len(markdown)} chars")
                await asyncio.sleep(delay)
            except Exception as e:
                logger.warn(f"Crawler | Failed [{i+1}/{len(urls)}] {url}: {e}")

        await browser.close()

    logger.info(f"Crawler | Done — {len(pages)} pages crawled")
    return pages


def crawl(sitemap_only: bool = False, delay: float = 0.5, max_pages: Optional[int] = None) -> list[CrawledPage]:
    return asyncio.run(crawl_async(sitemap_only=sitemap_only, delay=delay, max_pages=max_pages))
