"""
发现层 (Discovery Layer)
========================

不依赖任何外部搜索 API。自己爬取搜索引擎结果页，自己发现 sitemap。

三种发现方式：
1. WebSearchProvider — 爬取 DDG/Bing/Baidu 结果页，提取 URL
2. SitemapProvider   — robots.txt → sitemap.xml → 批量 URL
3. WikipediaProvider — 维基百科 API（免费、无密钥、结构化）

输出统一 SearchResult(title, url, snippet, source)
"""

import re
import time
import logging
import urllib.parse
import urllib.request
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Optional, Callable

logger = logging.getLogger("Taiji.Search.Discovery")


@dataclass
class SearchResult:
    """统一的搜索结果"""
    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = ""          # 来源搜索引擎
    score: float = 0.0        # 原始排序分数

    def to_dict(self) -> dict:
        return {
            "title": self.title, "url": self.url,
            "snippet": self.snippet, "source": self.source,
        }


# ═══════════════════════════════════════════════
# HTTP 工具
# ═══════════════════════════════════════════════

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]


def _random_ua() -> str:
    import random
    return random.choice(_USER_AGENTS)


def http_get(url: str, timeout: int = 10, verify_ssl: bool = True) -> str:
    """纯 stdlib HTTP GET，随机 UA，可选 SSL 容错"""
    import ssl
    headers = {
        "User-Agent": _random_ua(),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    req = urllib.request.Request(url, headers=headers)
    if not verify_ssl:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def _strip_tags(html: str) -> str:
    """快速去标签"""
    return re.sub(r"<[^>]+>", "", html).strip()


# ═══════════════════════════════════════════════
# 1. 网页搜索引擎（正则爬取结果页）
# ═══════════════════════════════════════════════

def _search_duckduckgo(query: str, max_results: int = 8) -> List[SearchResult]:
    """DuckDuckGo HTML 版搜索结果页爬取"""
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        html = http_get(url, timeout=12)
        results = []
        # DDG 结果块：<a class="result__a" href="...">标题</a> ... <a class="result__snippet">摘要</a>
        blocks = re.findall(
            r'<a[^>]*class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>',
            html, re.DOTALL,
        )
        for href, title, snippet in blocks[:max_results]:
            real_url = href
            if "uddg=" in href:
                m = re.search(r"uddg=([^&]+)", href)
                if m:
                    real_url = urllib.parse.unquote(m.group(1))
            results.append(SearchResult(
                title=_strip_tags(title),
                url=real_url,
                snippet=_strip_tags(snippet),
                source="DuckDuckGo",
            ))
        return results
    except Exception as e:
        logger.debug(f"DDG 搜索失败: {e}")
        return []


def _search_bing(query: str, max_results: int = 8) -> List[SearchResult]:
    """Bing 搜索结果页爬取（2024+ 新 HTML 结构兼容）"""
    try:
        url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
        html = http_get(url, timeout=12)
        results = []
        # Bing 新结构：结果在 <li class="b_algo"> 或 #b_results > li
        blocks = re.findall(r'<li[^>]*class="[^"]*b_algo[^"]*"[^>]*>(.*?)</li>', html, re.DOTALL)
        if not blocks:
            # 新版 Bing 可能用不同结构，直接找 h2 > a
            pattern = r'<h2[^>]*>\s*<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?(?:<p[^>]*>(.*?)</p>)?'
            for href, title, snippet in re.findall(pattern, html, re.DOTALL)[:max_results]:
                results.append(SearchResult(
                    title=_strip_tags(title), url=href,
                    snippet=_strip_tags(snippet) if snippet else "",
                    source="Bing",
                ))
            return results
        for block in blocks[:max_results]:
            title_m = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>', block, re.DOTALL)
            desc_m = re.search(r"<p[^>]*>(.*?)</p>", block, re.DOTALL)
            if title_m:
                results.append(SearchResult(
                    title=_strip_tags(title_m.group(2)),
                    url=title_m.group(1),
                    snippet=_strip_tags(desc_m.group(1)) if desc_m else "",
                    source="Bing",
                ))
        return results
    except Exception as e:
        logger.debug(f"Bing 搜索失败: {e}")
        return []


def _search_baidu(query: str, max_results: int = 8) -> List[SearchResult]:
    """百度搜索结果页爬取"""
    try:
        url = f"https://www.baidu.com/s?wd={urllib.parse.quote(query)}"
        html = http_get(url, timeout=12)
        results = []
        blocks = re.findall(
            r'<div[^>]*class="[^"]*result[^"]*c-container[^"]*"[^>]*>(.*?)(?=<div[^>]*class="[^"]*result[^"]*c-container|</div>\s*$)',
            html, re.DOTALL,
        )
        for block in blocks[:max_results]:
            title_m = re.search(r'<h3[^>]*>\s*<a[^>]*>(.*?)</a>', block, re.DOTALL)
            link_m = re.search(r'<h3[^>]*>\s*<a[^>]*href="([^"]*)"', block, re.DOTALL)
            desc_m = re.search(
                r'<span[^>]*class="[^"]*content-right[^"]*"[^>]*>(.*?)</span>'
                r'|<div[^>]*class="[^"]*c-abstract[^"]*"[^>]*>(.*?)</div>',
                block, re.DOTALL,
            )
            if title_m:
                snippet = ""
                if desc_m:
                    snippet = _strip_tags(desc_m.group(1) or desc_m.group(2) or "")
                results.append(SearchResult(
                    title=_strip_tags(title_m.group(1)),
                    url=link_m.group(1) if link_m else "",
                    snippet=snippet,
                    source="Baidu",
                ))
        return results
    except Exception as e:
        logger.debug(f"Baidu 搜索失败: {e}")
        return []


def _search_wikipedia(query: str, max_results: int = 5) -> List[SearchResult]:
    """维基百科 API（免费、无密钥、结构化 JSON）"""
    try:
        url = (f"https://zh.wikipedia.org/w/api.php?"
               f"action=query&list=search&srsearch={urllib.parse.quote(query)}"
               f"&format=json&srlimit={max_results}")
        html = http_get(url, timeout=8)
        import json
        data = json.loads(html)
        results = []
        for item in data.get("query", {}).get("search", []):
            title = item.get("title", "")
            snippet = _strip_tags(item.get("snippet", ""))
            page_url = f"https://zh.wikipedia.org/wiki/{urllib.parse.quote(title)}"
            results.append(SearchResult(
                title=title, url=page_url, snippet=snippet, source="Wikipedia",
            ))
        return results
    except Exception as e:
        logger.debug(f"Wikipedia 搜索失败: {e}")
        return []


def _search_google(query: str, max_results: int = 8) -> List[SearchResult]:
    """Google 搜索结果页爬取（纯 stdlib）"""
    try:
        url = f"https://www.google.com/search?q={urllib.parse.quote(query)}&num={max_results}"
        html = http_get(url, timeout=12)
        results = []
        # Google 结果块：<div class="ZINbbc"><div class="kCrYT"><a href="/url?q=...">...
        blocks = re.findall(
            r'<a[^>]*href="/url\?q=([^&"]+)[^>]*>(.*?)</a>.*?'
            r'<div[^>]*class="[^"]*BNeawe[^"]*"[^>]*>(.*?)</div>',
            html, re.DOTALL,
        )
        for href, title, snippet in blocks[:max_results]:
            real_url = urllib.parse.unquote(href)
            results.append(SearchResult(
                title=_strip_tags(title),
                url=real_url,
                snippet=_strip_tags(snippet),
                source="Google",
            ))
        # 回退：找所有 /url?q= 链接
        if not results:
            links = re.findall(r'href="/url\?q=([^&"]+)', html)
            for href in links[:max_results]:
                results.append(SearchResult(
                    title="", url=urllib.parse.unquote(href),
                    snippet="", source="Google",
                ))
        return results
    except Exception as e:
        logger.debug(f"Google 搜索失败: {e}")
        return []


def _search_yandex(query: str, max_results: int = 8) -> List[SearchResult]:
    """Yandex 搜索结果页爬取"""
    try:
        url = f"https://yandex.com/search/?text={urllib.parse.quote(query)}"
        html = http_get(url, timeout=12)
        results = []
        # Yandex: <a class="OrganicTitle-Link" href="...">标题</a>
        blocks = re.findall(
            r'<a[^>]*class="[^"]*OrganicTitle[^"]*"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'<span[^>]*class="[^"]*OrganicText[^"]*"[^>]*>(.*?)</span>',
            html, re.DOTALL,
        )
        for href, title, snippet in blocks[:max_results]:
            results.append(SearchResult(
                title=_strip_tags(title), url=href,
                snippet=_strip_tags(snippet), source="Yandex",
            ))
        return results
    except Exception as e:
        logger.debug(f"Yandex 搜索失败: {e}")
        return []


def _search_mojeek(query: str, max_results: int = 8) -> List[SearchResult]:
    """Mojeek（独立搜索引擎，无 Google 依赖）"""
    try:
        url = f"https://www.mojeek.com/search?q={urllib.parse.quote(query)}"
        html = http_get(url, timeout=12)
        results = []
        # Mojeek: <a class="ob" href="...">标题</a> ... <p class="s">摘要</p>
        blocks = re.findall(
            r'<a[^>]*class="[^"]*ob[^"]*"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'<p[^>]*class="[^"]*s[^"]*"[^>]*>(.*?)</p>',
            html, re.DOTALL,
        )
        for href, title, snippet in blocks[:max_results]:
            results.append(SearchResult(
                title=_strip_tags(title), url=href,
                snippet=_strip_tags(snippet), source="Mojeek",
            ))
        return results
    except Exception as e:
        logger.debug(f"Mojeek 搜索失败: {e}")
        return []


def _search_searx(query: str, max_results: int = 8) -> List[SearchResult]:
    """Searx 公共实例（元搜索引擎，聚合多个后端）"""
    instances = [
        "https://searx.be/search",
        "https://search.sapti.me/search",
        "https://searxng.nicecarrots.net/search",
    ]
    for base in instances:
        try:
            url = f"{base}?q={urllib.parse.quote(query)}&format=json"
            html = http_get(url, timeout=10)
            import json
            data = json.loads(html)
            results = []
            for item in data.get("results", [])[:max_results]:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    source=f"Searx({item.get('engine', '?')})",
                ))
            if results:
                return results
        except Exception:
            continue
    return []


def _search_arxiv(query: str, max_results: int = 5) -> List[SearchResult]:
    """arXiv API（学术论文，免费无密钥，SSL 容错）"""
    try:
        url = (f"http://export.arxiv.org/api/query?search_query=all:{urllib.parse.quote(query)}"
               f"&start=0&max_results={max_results}")
        xml = http_get(url, timeout=10, verify_ssl=False)
        results = []
        entries = re.findall(r'<entry>(.*?)</entry>', xml, re.DOTALL)
        for entry in entries[:max_results]:
            title_m = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            link_m = re.search(r'<id>(.*?)</id>', entry, re.DOTALL)
            summary_m = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            if title_m:
                results.append(SearchResult(
                    title=_strip_tags(title_m.group(1)).strip(),
                    url=link_m.group(1).strip() if link_m else "",
                    snippet=_strip_tags(summary_m.group(1))[:300] if summary_m else "",
                    source="arXiv",
                ))
        return results
    except Exception as e:
        logger.debug(f"arXiv 搜索失败: {e}")
        return []


def _search_hackernews(query: str, max_results: int = 5) -> List[SearchResult]:
    """Hacker News Algolia API（免费、无密钥、结构化 JSON）"""
    try:
        url = (f"https://hn.algolia.com/api/v1/search?query={urllib.parse.quote(query)}"
               f"&tags=story&hitsPerPage={max_results}")
        html = http_get(url, timeout=8)
        import json
        data = json.loads(html)
        results = []
        for hit in data.get("hits", [])[:max_results]:
            results.append(SearchResult(
                title=hit.get("title", ""),
                url=hit.get("url", "") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
                snippet=hit.get("story_text", "")[:200] or f"Points: {hit.get('points', 0)}, Comments: {hit.get('num_comments', 0)}",
                source="HackerNews",
            ))
        return results
    except Exception as e:
        logger.debug(f"HackerNews 搜索失败: {e}")
        return []


def _search_reddit(query: str, max_results: int = 5) -> List[SearchResult]:
    """Reddit 搜索（JSON API，无密钥）"""
    try:
        url = (f"https://www.reddit.com/search.json?q={urllib.parse.quote(query)}"
               f"&limit={max_results}&sort=relevance")
        headers = {"User-Agent": _random_ua()}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            import json
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
        results = []
        children = data.get("data", {}).get("children", [])
        for child in children[:max_results]:
            d = child.get("data", {})
            results.append(SearchResult(
                title=d.get("title", ""),
                url=f"https://www.reddit.com{d.get('permalink', '')}",
                snippet=d.get("selftext", "")[:200] or f"r/{d.get('subreddit', '?')} | Score: {d.get('score', 0)}",
                source="Reddit",
            ))
        return results
    except Exception as e:
        logger.debug(f"Reddit 搜索失败: {e}")
        return []


def _search_stackoverflow(query: str, max_results: int = 5) -> List[SearchResult]:
    """Stack Overflow API（免费、无密钥、结构化 JSON）"""
    try:
        url = (f"https://api.stackexchange.com/2.3/search/advanced?"
               f"order=desc&sort=relevance&q={urllib.parse.quote(query)}"
               f"&site=stackoverflow&pagesize={max_results}")
        html = http_get(url, timeout=10)
        import json
        data = json.loads(html)
        results = []
        for item in data.get("items", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("link", ""),
                snippet=_strip_tags(item.get("body", ""))[:200],
                source="StackOverflow",
            ))
        return results
    except Exception as e:
        logger.debug(f"StackOverflow 搜索失败: {e}")
        return []


def _search_github(query: str, max_results: int = 5) -> List[SearchResult]:
    """GitHub 搜索（无密钥时有限速但有结果）"""
    try:
        url = (f"https://api.github.com/search/repositories?q={urllib.parse.quote(query)}"
               f"&per_page={max_results}&sort=stars")
        html = http_get(url, timeout=10)
        import json
        data = json.loads(html)
        results = []
        for item in data.get("items", [])[:max_results]:
            results.append(SearchResult(
                title=item.get("full_name", ""),
                url=item.get("html_url", ""),
                snippet=item.get("description", "")[:200],
                source="GitHub",
            ))
        return results
    except Exception as e:
        logger.debug(f"GitHub 搜索失败: {e}")
        return []


# ═══════════════════════════════════════════════
# WebSearchProvider — 多引擎并行搜索
# ═══════════════════════════════════════════════

class WebSearchProvider:
    """多引擎并行搜索，取最快返回的非空结果"""

    # 所有可用引擎
    ENGINES: dict = {
        # 通用搜索
        "duckduckgo": _search_duckduckgo,
        "google": _search_google,
        "bing": _search_bing,
        "baidu": _search_baidu,
        "yandex": _search_yandex,
        "mojeek": _search_mojeek,
        "searx": _search_searx,
        # 结构化数据源
        "wikipedia": _search_wikipedia,
        "arxiv": _search_arxiv,
        "hackernews": _search_hackernews,
        "reddit": _search_reddit,
        "stackoverflow": _search_stackoverflow,
        "github": _search_github,
    }

    def __init__(self, engines: Optional[List[str]] = None):
        # 默认引擎：5 个 JSON API（快、稳定）+ 2 个 HTML 爬取（Baidu/Bing）
        # Google/Yandex/Mojeek/Searx/DDG 作为可选，需手动指定
        # 因为它们的 HTML 结构频繁变化或被反爬虫挡
        self.engine_names = engines or [
            # JSON API 层（最可靠，<1s）
            "wikipedia", "hackernews", "stackoverflow", "github", "arxiv",
            # HTML 爬取层（可用但不稳定）
            "baidu", "bing",
        ]

    def search(self, query: str, max_results: int = 8) -> List[SearchResult]:
        """
        并行搜索所有引擎，合并去重，按引擎分组排序。

        策略：
        - 所有引擎同时发，6s 内收集所有返回结果
        - 不再「谁先返回用谁」，而是收集所有引擎结果后合并
        - 这样质量更高：10 个引擎的结果加在一起 > 1 个引擎的结果
        - 最后轮询合并去重
        """
        if not query or not query.strip():
            return []

        engines_to_use = {
            name: func for name, func in self.ENGINES.items()
            if name in self.engine_names
        }
        if not engines_to_use:
            return []

        all_results: List[SearchResult] = []
        engine_results: dict = {}

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(engines_to_use)) as pool:
            future_map = {
                pool.submit(func, query, max_results): name
                for name, func in engines_to_use.items()
            }
            try:
                for future in concurrent.futures.as_completed(future_map, timeout=8.0):
                    name = future_map[future]
                    try:
                        results = future.result(timeout=2.0)
                        if results:
                            engine_results[name] = results
                            logger.debug(f"  {name}: {len(results)} 条")
                    except Exception:
                        pass
            except concurrent.futures.TimeoutError:
                logger.debug("  部分引擎超时，用已返回结果")
            for f in future_map:
                f.cancel()

        # 合并：轮询取每个引擎的第 i 个结果，保证多样性
        max_per_engine = max((len(r) for r in engine_results.values()), default=0)
        seen_urls = set()
        for i in range(max_per_engine):
            for name in self.engine_names:
                if name not in engine_results:
                    continue
                results = engine_results[name]
                if i < len(results):
                    r = results[i]
                    if r.url and r.url not in seen_urls:
                        seen_urls.add(r.url)
                        all_results.append(r)
                        if len(all_results) >= max_results:
                            return all_results
        return all_results

    def search_single(self, query: str, engine: str, max_results: int = 8) -> List[SearchResult]:
        """单引擎搜索"""
        func = self.ENGINES.get(engine)
        if not func:
            return []
        return func(query, max_results)


# ═══════════════════════════════════════════════
# SitemapProvider — 通过 robots.txt / sitemap 发现 URL
# ═══════════════════════════════════════════════

class SitemapProvider:
    """通过 robots.txt 和 sitemap.xml 发现站点 URL"""

    def discover(self, seed_url: str, max_urls: int = 200) -> List[str]:
        """
        从种子 URL 出发：
        1. 尝试 /robots.txt 找 Sitemap
        2. 尝试 /sitemap.xml
        3. 如果都没有，返回空（交给爬虫自行发现）
        """
        parsed = urllib.parse.urlparse(seed_url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        sitemap_urls = []

        # 1. robots.txt
        try:
            robots = http_get(f"{base}/robots.txt", timeout=8)
            for line in robots.splitlines():
                line = line.strip()
                if line.lower().startswith("sitemap:"):
                    sm = line.split(":", 1)[1].strip()
                    if sm:
                        sitemap_urls.append(sm)
        except Exception:
            pass

        # 2. 默认 sitemap.xml
        if not sitemap_urls:
            sitemap_urls.append(f"{base}/sitemap.xml")

        # 3. 解析 sitemap
        all_urls: List[str] = []
        for sm_url in sitemap_urls[:3]:
            urls = self._parse_sitemap(sm_url)
            all_urls.extend(urls)
            if len(all_urls) >= max_urls:
                break

        return all_urls[:max_urls]

    def _parse_sitemap(self, sitemap_url: str) -> List[str]:
        """解析 sitemap.xml，支持 sitemap index"""
        try:
            xml = http_get(sitemap_url, timeout=10)
            urls = re.findall(r"<loc>(.*?)</loc>", xml, re.DOTALL)
            cleaned = [u.strip() for u in urls if u.strip().startswith("http")]

            # 如果是 sitemap index（嵌套 sitemap），递归解析
            if cleaned and ".xml" in cleaned[0] and "<urlset" not in xml[:500]:
                nested = []
                for sub_url in cleaned[:5]:
                    nested.extend(self._parse_sitemap(sub_url))
                return nested
            return cleaned
        except Exception as e:
            logger.debug(f"  sitemap 解析失败 {sitemap_url}: {e}")
            return []


# ═══════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════

_default_provider: Optional[WebSearchProvider] = None


def get_search_provider() -> WebSearchProvider:
    global _default_provider
    if _default_provider is None:
        _default_provider = WebSearchProvider()
    return _default_provider


def search(query: str, max_results: int = 8) -> List[SearchResult]:
    """快捷搜索接口"""
    return get_search_provider().search(query, max_results)
