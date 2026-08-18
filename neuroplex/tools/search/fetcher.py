"""
抓取层 (Fetcher Layer)
=======================

双通道网页抓取：
1. HttpFetcher  — urllib 快速抓取静态 HTML
2. BrowserFetcher — Playwright 抓取 JS 渲染页面

策略：先 HTTP，失败或内容太少则回退 Browser。
"""

import os
import re
import time
import logging
import concurrent.futures
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from urllib.parse import urlparse

from .discovery import http_get, _random_ua

logger = logging.getLogger("Taiji.Search.Fetcher")


@dataclass
class FetchedPage:
    """抓取的原始页面"""
    url: str = ""
    html: str = ""
    status: str = "ok"           # ok / http_error / empty / browser_fallback
    fetcher: str = "http"        # http / browser
    fetch_time: float = 0.0
    error: str = ""


# ═══════════════════════════════════════════════
# HttpFetcher
# ═══════════════════════════════════════════════

class HttpFetcher:
    """快速 HTTP 抓取，纯 stdlib"""

    SKIP_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico",
        ".pdf", ".zip", ".tar", ".gz", ".rar",
        ".mp3", ".mp4", ".avi", ".mov", ".wav",
        ".css", ".js", ".woff", ".woff2", ".ttf",
    }

    def fetch(self, url: str, timeout: int = 12) -> FetchedPage:
        """抓取单个 URL"""
        if self._should_skip(url):
            return FetchedPage(url=url, status="empty", error="skipped extension")

        t0 = time.time()
        try:
            html = http_get(url, timeout=timeout)
            if not html or len(html.strip()) < 200:
                return FetchedPage(url=url, status="empty", fetch_time=time.time() - t0)
            return FetchedPage(
                url=url, html=html, status="ok",
                fetcher="http", fetch_time=time.time() - t0,
            )
        except Exception as e:
            return FetchedPage(
                url=url, status="http_error",
                fetcher="http", fetch_time=time.time() - t0, error=str(e),
            )

    def fetch_batch(self, urls: List[str], max_workers: int = 4, timeout: int = 12) -> List[FetchedPage]:
        """并行抓取多个 URL"""
        results: List[FetchedPage] = [None] * len(urls)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {pool.submit(self.fetch, url, timeout): i for i, url in enumerate(urls)}
            for future in concurrent.futures.as_completed(future_to_idx, timeout=timeout + 5):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result(timeout=2.0)
                except Exception as e:
                    results[idx] = FetchedPage(url=urls[idx], status="http_error", error=str(e))
            for f in future_to_idx:
                f.cancel()
        return results

    def _should_skip(self, url: str) -> bool:
        """跳过非 HTML 资源"""
        path = urlparse(url).path.lower()
        return any(path.endswith(ext) for ext in self.SKIP_EXTENSIONS)


# ═══════════════════════════════════════════════
# BrowserFetcher
# ═══════════════════════════════════════════════

class BrowserFetcher:
    """Playwright 浏览器抓取，处理 JS 渲染页面"""

    def __init__(self, headless: bool = True, timeout: int = 15000):
        self.headless = headless
        self.timeout = timeout
        self._playwright = None
        self._browser = None

    def _ensure_browser(self):
        if self._browser is not None:
            return
        from playwright.sync_api import sync_playwright
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.headless, args=["--no-sandbox", "--disable-gpu"]
        )
        logger.info("BrowserFetcher: Chromium 已启动")

    def fetch(self, url: str) -> FetchedPage:
        """用浏览器抓取单个 URL"""
        t0 = time.time()
        try:
            self._ensure_browser()
            context = self._browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent=_random_ua(),
            )
            page = context.new_page()
            page.goto(url, wait_until="networkidle", timeout=self.timeout)
            time.sleep(0.8)  # 等 JS 渲染完
            html = page.content()
            page.close()
            context.close()
            if not html or len(html.strip()) < 200:
                return FetchedPage(url=url, status="empty", fetcher="browser", fetch_time=time.time() - t0)
            return FetchedPage(
                url=url, html=html, status="ok",
                fetcher="browser", fetch_time=time.time() - t0,
            )
        except Exception as e:
            return FetchedPage(
                url=url, status="http_error",
                fetcher="browser", fetch_time=time.time() - t0, error=str(e),
            )

    def fetch_with_pagination(self, seed_url: str, max_pages: int = 10) -> List[FetchedPage]:
        """模拟人类翻页：打开页面 → 提取"下一页"链接 → 跳转"""
        pages: List[FetchedPage] = []
        try:
            self._ensure_browser()
            context = self._browser.new_context(
                viewport={"width": 1280, "height": 800}, user_agent=_random_ua(),
            )
            page = context.new_page()
            current_url = seed_url
            for _ in range(max_pages):
                t0 = time.time()
                page.goto(current_url, wait_until="networkidle", timeout=self.timeout)
                time.sleep(0.8)
                pages.append(FetchedPage(
                    url=current_url, html=page.content(), status="ok",
                    fetcher="browser", fetch_time=time.time() - t0,
                ))
                # 找"下一页"
                next_url = self._find_next_page(page)
                if not next_url or next_url == current_url:
                    break
                current_url = next_url
            page.close()
            context.close()
        except Exception as e:
            logger.debug(f"  browser pagination 失败: {e}")
        return pages

    def _find_next_page(self, page) -> Optional[str]:
        """查找"下一页"链接"""
        try:
            js = (
                'var s = document.querySelector("a[rel=next], a.next, a:has-text(next), '
                'a:has-text(下一页), a:has-text(后页)"); '
                "return s ? s.href : null;"
            )
            return page.evaluate("() => { " + js + " }")
        except Exception:
            return None

    def close(self):
        if self._browser:
            self._browser.close()
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None


# ═══════════════════════════════════════════════
# DualFetcher — 自动选择 HTTP / Browser
# ═══════════════════════════════════════════════

class DualFetcher:
    """
    双通道抓取器：先 HTTP，失败或内容太少则回退浏览器。

    判断"内容太少"的标准：
    - HTML 长度 < 2000 字符
    - 或提取后的正文 < 500 字符（需要 Extractor 配合）
    """

    def __init__(self, use_browser: bool = True):
        self.http = HttpFetcher()
        self.browser = BrowserFetcher() if use_browser else None

    def fetch(self, url: str, force_browser: bool = False) -> FetchedPage:
        """抓取单个 URL，自动选择通道"""
        if force_browser and self.browser:
            return self.browser.fetch(url)

        page = self.http.fetch(url)
        if page.status == "ok" and len(page.html) >= 2000:
            return page

        # HTTP 失败或内容太少，回退浏览器
        if self.browser:
            logger.debug(f"  HTTP 内容不足，回退浏览器: {url[:80]}")
            bp = self.browser.fetch(url)
            if bp.status == "ok":
                bp.status = "browser_fallback"
                return bp
        return page

    def fetch_batch(self, urls: List[str], max_workers: int = 4) -> List[FetchedPage]:
        """并行抓取多个 URL（仅 HTTP，浏览器串行）"""
        return self.http.fetch_batch(urls, max_workers=max_workers)

    def close(self):
        if self.browser:
            self.browser.close()


# ═══════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════

_default_fetcher: Optional[DualFetcher] = None


def get_fetcher() -> DualFetcher:
    global _default_fetcher
    if _default_fetcher is None:
        _default_fetcher = DualFetcher()
    return _default_fetcher
