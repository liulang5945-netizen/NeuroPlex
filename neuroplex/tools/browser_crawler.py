"""
Browser crawler using Playwright for human-like browsing.
"""
import os, re, time, json, logging, threading, concurrent.futures
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("BrowserCrawler")

@dataclass
class BrowsedPage:
    url: str = ""
    title: str = ""
    content: str = ""
    links: list = field(default_factory=list)
    ts: float = 0


class BrowserCrawler:
    """Human-like browser crawler using Playwright: open tabs, click links, navigate."""

    def __init__(self, data_dir=None, headless=True):
        self.data_dir = data_dir or os.path.join("taiji_data", "browser_crawl")
        os.makedirs(self.data_dir, exist_ok=True)
        self.headless = headless
        self._visited = set()
        self._pages = []
        self._lock = threading.Lock()
        self._browser = None
        self._playwright = None
        self.max_tabs = 4
        self.max_pages = 50
        self.timeout = 15000

    def close(self):
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None

    def _ensure_browser(self):
        if self._browser is not None: return
        from playwright.sync_api import sync_playwright
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless, args=["--no-sandbox"])
        logger.info("Browser started (Chromium)")

    def browse(self, seed_url, depth=2, max_pages=None):
        """Browse from seed URL: open pages, extract visible links, follow them."""
        if max_pages is None: max_pages = self.max_pages
        self._ensure_browser()
        self._visited = set()
        self._pages = []
        to_visit = [(seed_url, 0)]
        collected = 0
        while to_visit and collected < max_pages:
            batch = to_visit[:self.max_tabs]
            to_visit = to_visit[self.max_tabs:]
            for url, cur_depth in batch:
                if url in self._visited: continue
                self._visited.add(url)
                result = self._browse_single(url, cur_depth)
                if result:
                    self._pages.append(result)
                    collected += 1
                    logger.info(f"  [{collected}/{max_pages}] {result.title[:60]} ({result.url[:80]})")
                    if cur_depth < depth:
                        for link in result.links:
                            if link not in self._visited:
                                to_visit.append((link, cur_depth + 1))
            if collected >= max_pages: break
        self._save_results()
        logger.info(f"Browse done: {collected} pages, seed={seed_url}")
        return self._pages

    def browse_with_pagination(self, seed_url, max_pages=20):
        """Follow pagination like a human clicking Next Page."""
        self._ensure_browser()
        self._pages = []
        context = self._browser.new_context(viewport={"width": 1280, "height": 800})
        page = context.new_page()
        collected = 0
        current_url = seed_url
        try:
            while collected < max_pages:
                page.goto(current_url, wait_until="networkidle", timeout=self.timeout)
                time.sleep(1.0)
                title = page.title()
                content = self._extract_main(page)
                self._pages.append(BrowsedPage(url=current_url, title=title, content=content, ts=time.time()))
                collected += 1
                logger.info(f"  [{collected}/{max_pages}] {title[:60]} ({current_url[:80]})")
                next_url = self._eval(page, "var s=document.querySelector('a[rel=next], a:has-text(next)'); return s?s.href:null")
                if not next_url or next_url == current_url:
                    break
                current_url = next_url
        finally:
            page.close()
            context.close()
        self._save_results()
        return self._pages

    def _eval(self, pg, js):
        try:
            return pg.evaluate("() => { " + js + " }")
        except:
            return None

    def _browse_single(self, url, depth):
        context = self._browser.new_context(viewport={"width": 1280, "height": 800}, user_agent="Mozilla/5.0")
        pg = context.new_page()
        try:
            pg.goto(url, wait_until="networkidle", timeout=self.timeout)
            time.sleep(1.0)
            title = pg.title()
            content = self._extract_main(pg)
            links = self._extract_visible_links(pg, url)
            return BrowsedPage(url=url, title=title, content=content[:20000], links=links[:30], ts=time.time())
        except Exception as e:
            logger.debug(f"  Page fail {url[:60]}: {e}")
            return None
        finally:
            pg.close()
            context.close()

    def _extract_main(self, pg):
        try:
            js = r'(m||document.body).innerText||""'
            js = r'const m=document.querySelector("main,article,[role=main],.content,#content");' + js
            return pg.evaluate("()=>{" + js + "}") or ""
        except:
            return ""

    def _extract_visible_links(self, pg, base_url):
        try:
            import urllib.parse
            js = r'Array.from(document.querySelectorAll("a[href]"))'
            js += r'.filter(a=>{var r=a.getBoundingClientRect();return r.width>0&&r.height>0})'
            js += r'.map(a=>a.href).filter(h=>h.startsWith("http"))'
            links = pg.evaluate("()=>(" + js + ")") or []
            base_domain = urllib.parse.urlparse(base_url).netloc
            res = []
            for l in links[:50]:
                try:
                    if urllib.parse.urlparse(l).netloc == base_domain and not any(x in l for x in [".jpg","png","pdf","zip","css","js"]):
                        res.append(l)
                except: pass
            return list(set(res))
        except:
            return []

    def _save_results(self):
        if not self._pages: return
        try:
            data = [{"url": p.url, "title": p.title, "content": p.content[:5000], "links": p.links[:20], "ts": p.ts} for p in self._pages]
            with open(os.path.join(self.data_dir, "browse_results.json"), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except: pass

    def stats(self): return {"pages": len(self._pages), "dir": self.data_dir}


_crawler = None
def get_browser_crawler():
    global _crawler
    if _crawler is None: _crawler = BrowserCrawler()
    return _crawler

def browser_crawl(url):
    c = get_browser_crawler()
    try:
        pages = c.browse(url, depth=1, max_pages=5)
        if not pages: return f"Browse {url} returned no pages."
        lines = [f"Browser crawl {url}, {len(pages)} pages:"]
        for i, p in enumerate(pages, 1):
            lines.append(f"{i}. {p.title}\n   URL: {p.url}\n   {p.content[:300]}...\n")
        return "\n".join(lines)
    finally: c.close()