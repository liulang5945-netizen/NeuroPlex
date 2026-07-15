"""
态极自建搜索引擎 (Mini Search Engine)
======================================

从零实现的搜索引擎，不依赖任何外部 API（Serper/Tavily/DDG）。

架构：
  Crawler  → 下载网页，提取正文和链接
  Indexer  → 分词，建倒排索引（SQLite FTS5 + BM25 排序）
  Searcher → 接收查询，分词，查索引，返回排序结果

使用方式：
    from taiji.tools.mini_search import MiniSearchEngine
    engine = MiniSearchEngine()
    engine.crawl("https://docs.python.org/3/", depth=2)
    results = engine.search("async programming")
"""
import os
import re
import json
import time
import hashlib
import logging
import threading
import concurrent.futures
import random
import urllib.parse
from pathlib import Path
from collections import defaultdict
import threading as _thr
_tokenizer_lock = _thr.Lock()
_cached_tokenizer = None

def _get_tokenizer():
    global _cached_tokenizer
    if _cached_tokenizer is not None:
        return _cached_tokenizer if _cached_tokenizer else None
    with _tokenizer_lock:
        if _cached_tokenizer is not None:
            return _cached_tokenizer if _cached_tokenizer else None
        try:
            from taiji.tokenizer import ModelSelfTokenizer
            _cached_tokenizer = ModelSelfTokenizer()
            return _cached_tokenizer
        except Exception:
            _cached_tokenizer = False
            return None

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("MiniSearch")


@dataclass
class CrawledPage:
    """爬取的网页"""
    url: str
    title: str = ""
    text: str = ""
    links: List[str] = field(default_factory=list)
    crawled_at: float = 0


@dataclass
class SearchHit:
    """搜索结果"""
    url: str
    title: str
    snippet: str
    score: float = 0.0


class MiniSearchEngine:
    """
    自建迷你搜索引擎

    不依赖任何外部搜索 API。自己爬、自己建索引、自己搜。
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.join("taiji_data", "mini_search")
        os.makedirs(self.data_dir, exist_ok=True)
        self.db_path = os.path.join(self.data_dir, "search_index.db")
        self._lock = threading.Lock()
        self._rate_limits = {}
        self._default_delay = 1.0
        self._max_workers = 5
        self._init_db()

    # ═══════════════════════════════════════════
    # 数据库初始化
    # ═══════════════════════════════════════════

    def _init_db(self):
        """内存索引 + JSON 持久化（sandbox 不支持 SQLite 磁盘写入）"""
        self._memory_pages = []
        import atexit
        self._load_from_disk()
        atexit.register(self._save_to_disk)

    def _json_path(self):
        return os.path.join(self.data_dir, "index_snapshot.json")

    def _save_to_disk(self):
        try:
            with open(self._json_path(), "w", encoding="utf-8") as fh:
                json.dump(self._memory_pages, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_from_disk(self):
        try:
            jp = self._json_path()
            if os.path.exists(jp):
                with open(jp, "r", encoding="utf-8") as fh:
                    self._memory_pages = json.load(fh)
        except Exception:
            self._memory_pages = []

    # ═══════════════════════════════════════════
    # 1. 爬虫 —— 下载网页 + 提取正文 + 提取链接
    # ═══════════════════════════════════════════

    def crawl(self, seed_url: str, depth: int = 2, max_pages: int = 50,
              respect_robots: bool = True, sitemap_first: bool = True) -> int:
        """Concurrent crawler with rate limiting, robots.txt, sitemap, smart extraction"""
        from taiji.tools.web import fetch as web_fetch
        import queue

        visited_lock = threading.Lock()
        visited = set()
        total_crawled = [0]
        to_visit = []

        if sitemap_first:
            sitemap_urls = self._discover_sitemap_urls(seed_url)
            if sitemap_urls:
                logger.info(f"Sitemap: {len(sitemap_urls)} URLs")
                to_visit = [(u, 0) for u in sitemap_urls[:max_pages * 2]]
                with visited_lock:
                    visited.add(self._normalize_url(seed_url))

        if not to_visit:
            to_visit = [(seed_url, 0)]

        url_queue = queue.Queue()
        for item in to_visit:
            url_queue.put(item)

        def _worker():
            while total_crawled[0] < max_pages:
                try:
                    url, cur_depth = url_queue.get(timeout=3.0)
                except queue.Empty:
                    break
                normal = self._normalize_url(url)
                with visited_lock:
                    if normal in visited:
                        url_queue.task_done()
                        continue
                    visited.add(normal)

                result = self._crawl_single_page(url, web_fetch, seed_url, cur_depth, depth, respect_robots)
                if result:
                    with visited_lock:
                        if total_crawled[0] < max_pages:
                            self._save_page(result["url"], result["title"], result["text"], result["links"])
                            total_crawled[0] += 1
                            logger.info(f"  [{total_crawled[0]}/{max_pages}] {result['title'][:60]} ({result['url'][:80]})")
                    if cur_depth < depth:
                        for link in result.get("links", []):
                            n = self._normalize_url(link)
                            with visited_lock:
                                if n not in visited:
                                    url_queue.put((link, cur_depth + 1))
                url_queue.task_done()

        workers = min(self._max_workers, max_pages)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_worker) for _ in range(workers)]
            for f in futures:
                try:
                    f.result(timeout=300)
                except Exception:
                    pass

        self._rebuild_fts()
        logger.info(f"Crawl done: {total_crawled[0]} pages, seed={seed_url}")
        return total_crawled[0]

    def _crawl_single_page(self, url, web_fetch, seed_url, depth, max_depth, respect_robots):
        domain = urllib.parse.urlparse(url).netloc
        self._throttle(domain)
        if respect_robots and not self._robots_allowed(url):
            return None
        try:
            html = web_fetch(url, as_markdown=False, max_length=200000)
            if not html or html.startswith(("F","H")):
                return None
            if len(html) < 100:
                return None
            title = self._extract_title(html)
            text = self._extract_main_content(html)
            links = []
            if depth < max_depth:
                links = self._extract_links(html, seed_url)
            if len(text) > 50:
                return {"url": url, "title": title, "text": text, "links": links}
        except Exception as e:
            logger.debug(f"  Crawl fail {url[:60]}: {e}")
        return None

    def _throttle(self, domain):
        now = time.time()
        last, delay = self._rate_limits.get(domain, (0, self._default_delay))
        elapsed = now - last
        if elapsed < delay:
            time.sleep(delay - elapsed + random.uniform(0, delay * 0.3))
        self._rate_limits[domain] = (time.time(), delay)

    def _robots_allowed(self, url):
        if not hasattr(self, "_robots_cache"):
            self._robots_cache = {}
        domain = urllib.parse.urlparse(url).netloc
        if domain not in self._robots_cache:
            try:
                import urllib.robotparser
                rp = urllib.robotparser.RobotFileParser()
                rp.set_url("https://" + domain + "/robots.txt")
                rp.read()
                self._robots_cache[domain] = rp
            except Exception:
                self._robots_cache[domain] = None
        rp = self._robots_cache.get(domain)
        return rp.can_fetch("TaijiCrawler/1.0", url) if rp else True

    def _discover_sitemap_urls(self, seed_url):
        from taiji.tools.web import fetch as web_fetch
        parsed = urllib.parse.urlparse(seed_url)
        base = parsed.scheme + "://" + parsed.netloc
        for path in ["/sitemap.xml", "/sitemap_index.xml"]:
            try:
                xml = web_fetch(base + path, as_markdown=False, max_length=500000)
                if xml and not xml.startswith(("F","H")) and len(xml) > 50:
                    urls = re.findall(r"<loc>(https?://[^<]+)</loc>", xml, re.IGNORECASE)
                    if urls:
                        return [self._normalize_url(u) for u in urls if self._normalize_url(u)]
            except Exception:
                pass
        return []

    def _normalize_url(self, url):
        try:
            p = urllib.parse.urlparse(url)
            return urllib.parse.urlunparse((
                p.scheme.lower(), p.netloc.lower(),
                p.path.rstrip("/") or "/", p.params, p.query, ""
            ))
        except Exception:
            return url


    def _extract_title(self, html: str) -> str:
        m = re.search(r'<title[^>]*>(.*?)</title>', html, re.DOTALL | re.IGNORECASE)
        return re.sub(r'<[^>]+>', '', m.group(1)).strip()[:200] if m else ""

    def _extract_main_content(self, html: str) -> str:
        """Smart content extraction: remove boilerplate, score by text density"""
        for tag in ["script", "style", "nav", "footer", "header", "aside",
                     "noscript", "iframe", "svg", "form"]:
            html = re.sub(r"<" + tag + r"[^>]*>.*?</" + tag + r">", " ", html, flags=re.DOTALL | re.IGNORECASE)
        bp = (
            r'(?:class|id)\s*=\s*"[^"]*(?:nav|menu|sidebar|footer|header'
            r'|comment|widget|advertisement|banner|breadcrumb|pagination'
            r'|social|share|related|recommend|popular|trending)[^"]*"'
        )
        html = re.sub(r"<[^>]*" + bp + r"[^>]*>.*?</[^>]+>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        blocks = re.split(r"(?:<article[^>]*>|<section[^>]*>|<main[^>]*>|<div[^>]*>)|(?:</article>|</section>|</main>|</div>)", html)
        scored = []
        for blk in blocks:
            text = self._strip_tags(blk)
            if not text.strip():
                continue
            char_count = len(text)
            tag_count = len(re.findall(r"<[^>]+>", blk))
            links = len(re.findall(r"<a\s", blk, re.IGNORECASE))
            link_density = links / max(tag_count, 1)
            density = char_count / max(tag_count, 1)
            if link_density > 0.5:
                density *= 0.3
            scored.append((density, text))
        scored.sort(key=lambda x: x[0], reverse=True)
        main_parts = [text for _, text in scored[:5] if len(text) > 30]
        return "\n\n".join(main_parts) if main_parts else self._strip_tags(html)

    def _strip_tags(self, html: str) -> str:
        """Strip HTML tags, keep line breaks"""
        text = re.sub(r"<(?:br|p|div|h[1-6]|li|tr)[^>]*/?>", "\n", html, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&nbsp;", " ")
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _extract_text(self, html: str) -> str:
        """Backward-compatible wrapper"""
        return self._extract_main_content(html)

    def _extract_links(self, html: str, base_url: str) -> List[str]:
        """提取页面中的同域链接"""
        base_domain = urllib.parse.urlparse(base_url).netloc
        links = re.findall(r'href="(https?://[^"]+)"', html, re.IGNORECASE)
        result = []
        for link in links:
            try:
                domain = urllib.parse.urlparse(link).netloc
                if domain == base_domain and not any(
                    x in link for x in ['.jpg', '.png', '.pdf', '.zip', '.css', '.js']
                ):
                    result.append(link)
            except Exception:
                pass
        return list(set(result))  # 去重

    def _save_page(self, url: str, title: str, text: str, links: List[str]):
        """存储页面到内存列表"""
        with self._lock:
            entry = {
                "url": url, "title": title[:200], "text": text[:10000],
                "links": links, "crawled_at": time.time(),
            }
            self._memory_pages = [p for p in self._memory_pages if p["url"] != url]
            self._memory_pages.append(entry)

    def _rebuild_fts(self):
        """持久化到磁盘"""
        self._save_to_disk()

    # ═══════════════════════════════════════════
    # 2+3+4. 搜索 —— 分词 → 倒排查 → BM25 排序
    # ═══════════════════════════════════════════

    def search(self, query: str, top_k: int = 10) -> List[SearchHit]:
        """BM25 内存搜索"""
        terms = self._tokenize(query)
        if not terms or not self._memory_pages:
            return []

        with self._lock:
            N = len(self._memory_pages)
            avgdl = sum(len(p["text"]) for p in self._memory_pages) / max(N, 1)
            k1, b = 1.5, 0.75
            scored = []

            for p in self._memory_pages:
                doc_text = p["text"]
                doc_len = len(doc_text)
                score = 0.0
                words = doc_text.lower().split()
                for term in terms:
                    tf = words.count(term)
                    df = sum(1 for pp in self._memory_pages if term in pp["text"].lower())
                    idf = max(0, __import__("math").log((N - df + 0.5) / (df + 0.5) + 1))
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * doc_len / max(avgdl, 1))
                    score += idf * numerator / max(denominator, 0.1)
                if score > 0:
                    snip = self._make_snippet(doc_text, terms)
                    scored.append((score, p["url"], p["title"], snip))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [SearchHit(url=u, title=t, snippet=s[:200], score=round(sc, 2))
                    for sc, u, t, s in scored[:top_k]]

    def _tokenize(self, text):
        """分词：优先态极原生 tokenizer（与模型同词汇表），回退到 regex"""
        tok = _get_tokenizer()
        if tok:
            try:
                ids = tok.encode(text)
                return [str(tid) for tid in ids]
            except Exception:
                pass
        text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
        return [t for t in text.split() if len(t) >= 2]

    def _make_snippet(self, text, terms, window=80):
        low = text.lower()
        for t in terms:
            idx = low.find(t)
            if idx >= 0:
                start = max(0, idx - window // 2)
                end = min(len(text), idx + window // 2)
                return ("..." if start > 0 else "") + text[start:end] + ("..." if end < len(text) else "")
        return text[:window * 2]

    def search_to_text(self, query: str, top_k: int = 10) -> str:
        """搜索并返回格式化文本（供 Agent 工具调用）"""
        hits = self.search(query, top_k)
        if not hits:
            return f"在本地索引中未找到与 '{query}' 相关的结果。"
        lines = [f"搜索 '{query}' 找到 {len(hits)} 条本地结果:\n"]
        for i, h in enumerate(hits, 1):
            lines.append(f"{i}. {h.title}")
            lines.append(f"   {h.snippet}")
            lines.append(f"   URL: {h.url}  (相关度: {h.score})")
            lines.append("")
        return "\n".join(lines)

    # ═══════════════════════════════════════════
    # 管理接口
    # ═══════════════════════════════════════════

    def stats(self) -> dict:
        """索引统计"""
        return {"indexed_pages": len(self._memory_pages), "data_dir": self.data_dir}

    def clear_index(self):
        """清空索引"""
        with self._lock:
            self._memory_pages = []
            self._save_to_disk()


# ═══════════════════════════════════════════
# 全局单例 + 工具接口
# ═══════════════════════════════════════════

_engine: Optional[MiniSearchEngine] = None


def get_mini_search() -> MiniSearchEngine:
    global _engine
    if _engine is None:
        _engine = MiniSearchEngine()
    return _engine


def mini_search(query: str) -> str:
    """Agent 工具统一接口"""
    return get_mini_search().search_to_text(query)
