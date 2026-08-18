"""
编排层 (Pipeline Layer)
=======================

把 Discovery → Fetcher → Extractor → Index 串成闭环。

核心管道：
1. search_and_index(query)  — 搜索 → 抓取 → 提取 → 入索引 → 返回摘要
2. search_deep(query)       — 先查本地索引 → 未命中则联网搜索 → 入索引
3. browse_and_index(url)    — 浏览器抓取单页 → 提取 → 入索引
4. crawl_site(seed_url)     — sitemap → 批量抓取 → 入索引

这是整个搜索引擎的"临门一脚"：搜索结果不再是一次性的，
而是沉淀到本地索引里，越用越聪明。
"""

import time
import logging
import concurrent.futures
from typing import List, Optional, Dict
from dataclasses import dataclass

from .discovery import (
    WebSearchProvider, SitemapProvider, SearchResult,
    search as discovery_search,
)
from .fetcher import DualFetcher, FetchedPage, get_fetcher
from .extractor import ReadabilityExtractor, PageContent, get_extractor
from .index import InvertedIndex, IndexedPage, SearchHit, get_index

logger = logging.getLogger("Taiji.Search.Pipeline")


# ═══════════════════════════════════════════════
# SearchPipeline — 统一编排
# ═══════════════════════════════════════════════

class SearchPipeline:
    """
    搜索引擎总编排。

    线程安全设计：Discovery/Fetcher/Extractor 无状态可并行，
    Index 有自己的锁。
    """

    def __init__(self):
        self.discovery = WebSearchProvider()
        self.sitemap = SitemapProvider()
        self.fetcher = DualFetcher(use_browser=True)
        self.extractor = ReadabilityExtractor()
        self.index = InvertedIndex()

    # ─── 核心：搜索 + 抓取 + 入索引 ───

    def search_and_index(self, query: str, max_results: int = 8,
                         fetch_pages: int = 3) -> str:
        """
        完整管道：搜索 → 抓取顶部页面 → 提取正文 → 入索引 → 返回结构化摘要。

        这是闭环的关键：搜索结果不再是"看完就丢"，
        而是沉淀到本地索引里，下次搜类似的东西可以直接查本地。
        """
        t0 = time.time()
        results = self.discovery.search(query, max_results=max_results)
        if not results:
            return f"搜索 '{query}' 未找到结果。"

        # 并行抓取顶部页面
        fetched_contents = self._fetch_and_extract_batch(
            [r.url for r in results[:fetch_pages]]
        )

        # 入索引
        indexed = 0
        for content in fetched_contents:
            if content and content.word_count > 20:
                page = IndexedPage(
                    url=content.url, title=content.title,
                    text=content.text, links=content.links,
                    crawled_at=time.time(), word_count=content.word_count,
                    source="discovery",
                )
                self.index.add_page(page)
                indexed += 1

        # 生成摘要
        elapsed = time.time() - t0
        lines = [f"## 搜索: {query}\n"]
        lines.append(f"找到 {len(results)} 条结果，抓取 {len(fetched_contents)} 页，入索引 {indexed} 页 ({elapsed:.1f}s)\n")

        for i, r in enumerate(results):
            lines.append(f"### {i+1}. {r.title}")
            lines.append(f"**来源**: {r.url}  ({r.source})")
            lines.append(f"> {r.snippet[:300]}")
            # 如果这页被抓取了，附上正文摘要
            for content in fetched_contents:
                if content and content.url == r.url:
                    lines.append(f"\n**页面摘要**:\n{content.text[:500]}...")
                    break
            lines.append("")

        return "\n".join(lines)

    # ─── 深度搜索：先查本地 ───

    def search_deep(self, query: str, max_results: int = 8,
                    fetch_pages: int = 3) -> str:
        """
        深度搜索：先查本地索引 → 命中则直接返回 → 未命中则联网搜索 + 入索引。

        这样高频查询会越来越快（本地命中），低频查询会自动补充索引。
        """
        # 1. 先查本地
        local_hits = self.index.search(query, top_k=5)
        if local_hits:
            logger.info(f"本地索引命中: {len(local_hits)} 条")
            lines = [f"## 搜索: {query} (本地索引)\n"]
            lines.append(f"本地索引中找到 {len(local_hits)} 条结果:\n")
            for i, h in enumerate(local_hits, 1):
                lines.append(f"### {i}. {h.title}")
                lines.append(f"**来源**: {h.url}  (相关度: {h.score})")
                lines.append(f"> {h.snippet[:300]}\n")

            # 本地结果不足时，补充联网搜索
            if len(local_hits) < 3:
                lines.append("---\n补充联网搜索:\n")
                lines.append(self.search_and_index(query, max_results, fetch_pages))
            return "\n".join(lines)

        # 2. 本地未命中 → 联网搜索 + 入索引
        logger.info(f"本地未命中，联网搜索: {query}")
        return self.search_and_index(query, max_results, fetch_pages)

    # ─── 浏览器抓取 + 入索引 ───

    def browse_and_index(self, url: str, follow_links: int = 0) -> str:
        """
        浏览器抓取单页 → 提取 → 入索引。

        follow_links > 0 时，会跟随页面内的链接继续抓取。
        模拟人类浏览：打开页面 → 看内容 → 点感兴趣的链接。
        """
        t0 = time.time()
        page = self.fetcher.fetch(url, force_browser=True)
        if page.status != "ok":
            return f"浏览器抓取失败: {url} ({page.error})"

        content = self.extractor.extract(page.html, url)
        if content.word_count < 20:
            return f"页面内容太少: {url}"

        # 入索引
        indexed_page = IndexedPage(
            url=url, title=content.title, text=content.text,
            links=content.links, crawled_at=time.time(),
            word_count=content.word_count, source="browse",
        )
        self.index.add_page(indexed_page)

        # 跟随链接
        followed = 0
        if follow_links > 0 and content.links:
            for link_url in content.links[:follow_links]:
                if self.index.has_url(link_url):
                    continue
                sub_content = self._fetch_and_extract_single(link_url)
                if sub_content and sub_content.word_count > 20:
                    self.index.add_page(IndexedPage(
                        url=link_url, title=sub_content.title,
                        text=sub_content.text, links=sub_content.links,
                        crawled_at=time.time(), word_count=sub_content.word_count,
                        source="browse_follow",
                    ))
                    followed += 1

        elapsed = time.time() - t0
        lines = [f"## 浏览: {url}\n"]
        lines.append(f"标题: {content.title}")
        lines.append(f"字数: {content.word_count}，链接: {len(content.links)}，跟随: {followed} ({elapsed:.1f}s)\n")
        lines.append(f"**正文摘要**:\n{content.text[:800]}...")
        return "\n".join(lines)

    # ─── 批量爬取建索引 ───

    def crawl_site(self, seed_url: str, max_pages: int = 50) -> str:
        """
        从种子 URL 出发批量爬取：
        sitemap 发现 → 并行 HTTP 抓取 → 提取 → 入索引。

        适合给一个文档站点建立本地索引。
        """
        t0 = time.time()
        # 1. sitemap 发现 URL
        urls = self.sitemap.discover(seed_url, max_urls=max_pages * 2)
        if not urls:
            urls = [seed_url]
        logger.info(f"Sitemap 发现 {len(urls)} 个 URL")

        # 2. 并行抓取 + 提取
        contents = self._fetch_and_extract_batch(urls[:max_pages])

        # 3. 入索引
        indexed = 0
        for content in contents:
            if content and content.word_count > 20:
                self.index.add_page(IndexedPage(
                    url=content.url, title=content.title,
                    text=content.text, links=content.links,
                    crawled_at=time.time(), word_count=content.word_count,
                    source="crawl",
                ))
                indexed += 1

        elapsed = time.time() - t0
        return (f"## 爬取: {seed_url}\n"
                f"发现 URL: {len(urls)}，成功抓取: {len([c for c in contents if c])}，"
                f"入索引: {indexed} ({elapsed:.1f}s)\n"
                f"索引统计: {self.index.stats()}")

    # ─── 纯本地搜索 ───

    def search_local(self, query: str, top_k: int = 10) -> str:
        """只查本地索引"""
        hits = self.index.search(query, top_k)
        if not hits:
            return f"本地索引中未找到 '{query}'。索引: {self.index.stats()}"
        lines = [f"## 本地搜索: {query}\n"]
        lines.append(f"找到 {len(hits)} 条结果:\n")
        for i, h in enumerate(hits, 1):
            lines.append(f"### {i}. {h.title}")
            lines.append(f"**来源**: {h.url}  (相关度: {h.score})")
            lines.append(f"> {h.snippet[:300]}\n")
        return "\n".join(lines)

    # ─── 内部工具 ───

    def _fetch_and_extract_single(self, url: str) -> Optional[PageContent]:
        """抓取 + 提取单个 URL"""
        page = self.fetcher.fetch(url)
        if page.status != "ok":
            return None
        content = self.extractor.extract(page.html, url)
        content.url = url
        return content

    def _fetch_and_extract_batch(self, urls: List[str], max_workers: int = 4) -> List[Optional[PageContent]]:
        """并行抓取 + 提取"""
        results: List[Optional[PageContent]] = [None] * len(urls)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(self._fetch_and_extract_single, url): i
                for i, url in enumerate(urls)
            }
            for future in concurrent.futures.as_completed(future_to_idx, timeout=30):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result(timeout=5)
                except Exception:
                    results[idx] = None
            for f in future_to_idx:
                f.cancel()
        return results

    def close(self):
        self.fetcher.close()


# ═══════════════════════════════════════════════
# 统一入口 + 工具接口
# ═══════════════════════════════════════════════

_pipeline: Optional[SearchPipeline] = None


def get_pipeline() -> SearchPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = SearchPipeline()
    return _pipeline


# ─── 工具接口（供 tool_registry 注册）───

def tool_search(query: str) -> str:
    """快速搜索（不入索引）"""
    results = discovery_search(query, max_results=8)
    if not results:
        return f"搜索 '{query}' 未找到结果。"
    lines = [f"## 搜索: {query}\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.title}")
        lines.append(f"   {r.snippet[:200]}")
        lines.append(f"   URL: {r.url}  ({r.source})\n")
    return "\n".join(lines)


def tool_search_deep(query: str) -> str:
    """深度搜索（先查本地，再联网 + 入索引）"""
    return get_pipeline().search_deep(query)


def tool_search_local(query: str) -> str:
    """纯本地索引搜索"""
    return get_pipeline().search_local(query)


def tool_crawl_site(url: str) -> str:
    """批量爬取站点建索引"""
    return get_pipeline().crawl_site(url)


def tool_browse_web(url: str) -> str:
    """浏览器浏览单页 + 入索引"""
    return get_pipeline().browse_and_index(url, follow_links=2)
