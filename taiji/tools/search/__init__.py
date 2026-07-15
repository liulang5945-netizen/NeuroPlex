"""
态极自建搜索引擎 (Taiji Search)
================================

完全自建的搜索引擎，不依赖任何外部搜索 API。

五层架构：
  Discovery  → 发现 URL（DDG/Bing/Baidu/Wikipedia 正则 + sitemap）
  Fetcher    → 抓取网页（HTTP + Browser 双通道）
  Extractor  → 提取正文（Readability 启发式 + Markdown 转换）
  Index      → 倒排索引 + BM25 排序（词级分词，JSON 持久化）
  Pipeline   → 编排闭环（搜索 → 抓取 → 入索引 → 摘要）

使用方式：
    from taiji.tools.search import tool_search, tool_search_deep, tool_search_local

    # 快速搜索
    result = tool_search("Python async")

    # 深度搜索（先查本地索引，再联网 + 入索引）
    result = tool_search_deep("transformer 原理")

    # 纯本地搜索
    result = tool_search_local("async programming")

    # 批量爬取建索引
    result = tool_crawl_site("https://docs.python.org/3/")

    # 浏览器浏览 + 入索引
    result = tool_browse_web("https://example.com")
"""

from .discovery import WebSearchProvider, SitemapProvider, SearchResult, search
from .fetcher import DualFetcher, HttpFetcher, BrowserFetcher, FetchedPage
from .extractor import ReadabilityExtractor, PageContent
from .index import InvertedIndex, IndexedPage, SearchHit, Tokenizer
from .pipeline import (
    SearchPipeline, get_pipeline,
    tool_search, tool_search_deep, tool_search_local,
    tool_crawl_site, tool_browse_web,
)
from .smart_crawler import SmartCrawler, LinkScorer, ContentQuality, tool_smart_crawl

__all__ = [
    # Discovery
    "WebSearchProvider", "SitemapProvider", "SearchResult", "search",
    # Fetcher
    "DualFetcher", "HttpFetcher", "BrowserFetcher", "FetchedPage",
    # Extractor
    "ReadabilityExtractor", "PageContent",
    # Index
    "InvertedIndex", "IndexedPage", "SearchHit", "Tokenizer",
    # Pipeline
    "SearchPipeline", "get_pipeline",
    "tool_search", "tool_search_deep", "tool_search_local",
    "tool_crawl_site", "tool_browse_web",
    # SmartCrawler
    "SmartCrawler", "LinkScorer", "ContentQuality", "tool_smart_crawl",
]
