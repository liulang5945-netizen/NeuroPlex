"""
智能爬虫 (Smart Crawler)
========================

不是无脑 BFS 爬取，而是像人一样判断"这个链接值不值得点"。

核心能力：
1. 链接评分 — 根据锚文本、URL 路径、位置给每个链接打分
2. 主题聚焦 — 只跟踪与查询主题相关的链接，不跑偏
3. 自适应深度 — 内容质量高的页面多爬几层，差的立即停止
4. 反检测 — 随机延迟、UA 轮换、Referer 伪造
5. 重复检测 — URL 规范化 + 内容指纹去重
6. 增量索引 — 爬到的内容实时入索引

使用方式：
    from taiji.tools.search.smart_crawler import SmartCrawler
    crawler = SmartCrawler()
    crawler.crawl_topic("https://docs.python.org/3/", "async programming", max_pages=30)
"""

import re
import time
import math
import hashlib
import logging
import threading
import urllib.parse
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from collections import defaultdict

from .discovery import http_get, _random_ua, SearchResult
from .fetcher import DualFetcher, FetchedPage
from .extractor import ReadabilityExtractor, PageContent
from .index import InvertedIndex, IndexedPage, Tokenizer

logger = logging.getLogger("Taiji.Search.SmartCrawler")


# ═══════════════════════════════════════════════
# 链接评分器
# ═══════════════════════════════════════════════

class LinkScorer:
    """
    给页面上的每个链接打分，决定"这个链接值不值得点"。

    评分维度：
    1. 锚文本相关性（0-40分）— 链接文字和主题的匹配度
    2. URL 路径相关性（0-20分）— URL 里有没有主题关键词
    3. 链接位置权重（0-15分）— 正文里的链接 > 侧边栏 > 页脚
    4. URL 类型加分（0-15分）— 文档/文章页 > 列表页 > 首页
    5. 新鲜度加分（0-10分）— URL 含日期或 new 关键词
    """

    # URL 路径里的高价值关键词
    CONTENT_PATH_PATTERNS = [
        r"/docs?/", r"/tutorial", r"/guide", r"/manual",
        r"/api/", r"/reference", r"/article", r"/post",
        r"/blog/", r"/wiki/", r"/learn",
    ]

    # URL 路径里的低价值关键词
    NOISE_PATH_PATTERNS = [
        r"/login", r"/register", r"/signup", r"/cart",
        r"/search", r"/tag/", r"/category/", r"/author/",
        r"/page/\d+", r"/feed", r"/rss", r"/sitemap",
        r"/about", r"/contact", r"/privacy", r"/terms",
    ]

    def __init__(self):
        self.tokenizer = Tokenizer()

    def score(self, link_url: str, anchor_text: str, topic_terms: Set[str],
              position: str = "content") -> float:
        """
        给链接打分。

        Args:
            link_url: 链接 URL
            anchor_text: 锚文本
            topic_terms: 主题词集合（分词后）
            position: 链接位置 "content" / "sidebar" / "footer" / "nav"
        """
        score = 0.0

        # 1. 锚文本相关性（0-40）
        anchor_tokens = set(self.tokenizer.tokenize(anchor_text.lower()))
        if topic_terms and anchor_tokens:
            overlap = len(anchor_tokens & topic_terms)
            score += min(40, overlap * 15)

        # 2. URL 路径相关性（0-20）
        url_lower = link_url.lower()
        url_tokens = set(self.tokenizer.tokenize(urllib.parse.urlparse(url_lower).path))
        if topic_terms and url_tokens:
            overlap = len(url_tokens & topic_terms)
            score += min(20, overlap * 10)

        # 3. 链接位置权重（0-15）
        position_weights = {"content": 15, "nav": 8, "sidebar": 5, "footer": 2}
        score += position_weights.get(position, 10)

        # 4. URL 类型加分（0-15）
        for pattern in self.CONTENT_PATH_PATTERNS:
            if re.search(pattern, url_lower):
                score += 15
                break
        else:
            for pattern in self.NOISE_PATH_PATTERNS:
                if re.search(pattern, url_lower):
                    score -= 10
                    break

        # 5. 新鲜度（0-10）
        if re.search(r"/20\d{2}/|/new|/latest|/recent", url_lower):
            score += 10

        # 6. 惩罚：非 HTML 资源
        skip_exts = [".jpg", ".png", ".pdf", ".zip", ".css", ".js", ".mp4"]
        if any(url_lower.endswith(ext) for ext in skip_exts):
            score -= 50

        # 7. 惩罚：锚文本是"点击这里""更多"等无意义词
        if anchor_text.strip() in {"点击这里", "更多", "click here", "more", "read more", ""}:
            score -= 10

        return max(0, score)

    def rank_links(self, links: List[Tuple[str, str, str]],
                   topic_terms: Set[str], top_n: int = 10) -> List[Tuple[str, float]]:
        """
        对页面上的所有链接排序，返回 top_n 个最值得爬的。

        Args:
            links: [(url, anchor_text, position), ...]
            topic_terms: 主题词集合
            top_n: 返回前 N 个
        Returns:
            [(url, score), ...] 按分数降序
        """
        scored = []
        for url, anchor, position in links:
            s = self.score(url, anchor, topic_terms, position)
            if s > 5:  # 过滤掉极低分链接
                scored.append((url, s))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]


# ═══════════════════════════════════════════════
# 内容质量评估
# ═══════════════════════════════════════════════

class ContentQuality:
    """评估页面内容质量，决定是否继续深入爬取"""

    def __init__(self):
        self.tokenizer = Tokenizer()

    def assess(self, content: PageContent, topic_terms: Set[str]) -> Tuple[float, str]:
        """
        评估页面质量。

        Returns:
            (quality_score, reason)
            quality_score: 0.0-1.0
            reason: 质量评估原因
        """
        if not content.text or content.word_count < 20:
            return 0.0, "内容太少"

        score = 0.0

        # 1. 长度（0-0.3）：500字以上满分
        length_score = min(0.3, content.word_count / 500 * 0.3)
        score += length_score

        # 2. 主题相关性（0-0.4）：正文中有多少主题词
        text_tokens = set(self.tokenizer.tokenize(content.text.lower()))
        if topic_terms and text_tokens:
            overlap = len(text_tokens & topic_terms)
            relevance = min(0.4, overlap / max(len(topic_terms), 1) * 0.4)
            score += relevance

        # 3. 信息密度（0-0.2）：正文/HTML 比例
        if content.markdown and content.text:
            density = len(content.text) / max(len(content.markdown), 1)
            score += min(0.2, density * 0.2)

        # 4. 结构性（0-0.1）：有标题/列表/代码块 = 结构化内容
        structure = 0.0
        if re.search(r"^#+\s", content.markdown, re.MULTILINE):
            structure += 0.05
        if re.search(r"^- ", content.markdown, re.MULTILINE):
            structure += 0.03
        if "```" in content.markdown:
            structure += 0.02
        score += min(0.1, structure)

        # 质量等级
        if score >= 0.6:
            reason = "高质量"
        elif score >= 0.3:
            reason = "中等质量"
        else:
            reason = "低质量"

        return score, reason


# ═══════════════════════════════════════════════
# URL 规范化 + 去重
# ═══════════════════════════════════════════════

class UrlNormalizer:
    """URL 规范化，避免重复爬取同一页面"""

    @staticmethod
    def normalize(url: str) -> str:
        """规范化 URL"""
        parsed = urllib.parse.urlparse(url)
        # 移除 fragment
        clean = parsed._replace(fragment="")
        # 移除常见跟踪参数
        if clean.query:
            params = urllib.parse.parse_qs(clean.query)
            noise_keys = {"utm_source", "utm_medium", "utm_campaign",
                          "utm_content", "utm_term", "fbclid", "gclid",
                          "ref", "source", "from"}
            filtered = {k: v for k, v in params.items() if k.lower() not in noise_keys}
            clean = clean._replace(query=urllib.parse.urlencode(filtered, doseq=True))
        # 小写化 scheme 和 host
        clean = clean._replace(scheme=clean.scheme.lower(), netloc=clean.netloc.lower())
        # 移除末尾斜杠（根路径除外）
        path = clean.path
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")
        clean = clean._replace(path=path)
        return urllib.parse.urlunparse(clean)

    @staticmethod
    def content_hash(text: str) -> str:
        """内容指纹，检测近似重复"""
        # 取正文前 2000 字符的 hash，检测近似重复
        return hashlib.md5(text[:2000].encode("utf-8")).hexdigest()

    @staticmethod
    def same_domain(url1: str, url2: str) -> bool:
        return urllib.parse.urlparse(url1).netloc == urllib.parse.urlparse(url2).netloc


# ═══════════════════════════════════════════════
# SmartCrawler — 智能爬虫主体
# ═══════════════════════════════════════════════

@dataclass
class CrawlStats:
    """爬取统计"""
    visited: int = 0
    indexed: int = 0
    skipped_dup: int = 0
    skipped_low_quality: int = 0
    failed: int = 0
    total_links_found: int = 0
    total_links_followed: int = 0
    elapsed: float = 0.0


class SmartCrawler:
    """
    智能爬虫：主题聚焦 + 链接评分 + 自适应深度。

    工作流程：
    1. 给定种子 URL + 主题关键词
    2. 抓取页面，提取正文和链接
    3. 给每个链接打分，只跟踪高分链接
    4. 评估页面质量，高质量页面继续深入，低质量停止
    5. 实时入索引
    """

    def __init__(self, index: Optional[InvertedIndex] = None):
        self.fetcher = DualFetcher(use_browser=True)
        self.extractor = ReadabilityExtractor()
        self.index = index or InvertedIndex()
        self.scorer = LinkScorer()
        self.quality = ContentQuality()
        self.normalizer = UrlNormalizer()
        self.tokenizer = Tokenizer()

    def crawl_topic(self, seed_url: str, topic: str,
                    max_pages: int = 30, max_depth: int = 3,
                    same_domain_only: bool = True) -> CrawlStats:
        """
        主题聚焦爬取：只爬与主题相关的页面。

        Args:
            seed_url: 种子 URL
            topic: 主题关键词（如 "async programming"）
            max_pages: 最多爬取页数
            max_depth: 最大爬取深度
            same_domain_only: 是否只爬同域
        """
        t0 = time.time()
        stats = CrawlStats()
        topic_terms = set(self.tokenizer.tokenize(topic.lower()))

        # 优先队列：(score, url, depth)
        import heapq
        queue: List[Tuple[float, str, int]] = []
        heapq.heappush(queue, (-100, seed_url, 0))  # 种子页给高分

        visited: Set[str] = set()
        content_hashes: Set[str] = set()

        while queue and stats.visited < max_pages:
            neg_score, url, depth = heapq.heappop(queue)
            normal_url = self.normalizer.normalize(url)

            if normal_url in visited:
                continue
            if self.index.has_url(normal_url):
                stats.skipped_dup += 1
                continue
            if same_domain_only and not self.normalizer.same_domain(url, seed_url):
                continue

            visited.add(normal_url)
            stats.visited += 1

            # 抓取
            page = self.fetcher.fetch(url)
            if page.status != "ok":
                stats.failed += 1
                continue

            # 提取
            content = self.extractor.extract(page.html, url)
            content.url = normal_url

            # 去重：内容指纹
            chash = self.normalizer.content_hash(content.text)
            if chash in content_hashes:
                stats.skipped_dup += 1
                continue
            content_hashes.add(chash)

            # 质量评估
            q_score, reason = self.quality.assess(content, topic_terms)
            logger.info(f"  [{stats.visited}/{max_pages}] {reason}({q_score:.2f}) {content.title[:50]}")

            # 入索引（即使低质量也入，只是不再深入）
            if content.word_count > 20:
                self.index.add_page(IndexedPage(
                    url=normal_url, title=content.title, text=content.text,
                    links=content.links, crawled_at=time.time(),
                    word_count=content.word_count, source="smart_crawl",
                ))
                stats.indexed += 1
            else:
                stats.skipped_low_quality += 1
                continue  # 内容太少，不继续深入

            # 自适应深度：高质量页面才继续爬
            if depth >= max_depth or q_score < 0.3:
                continue

            # 提取链接并评分
            if content.links:
                links_with_context = self._extract_links_with_context(page.html, url)
                stats.total_links_found += len(links_with_context)

                ranked = self.scorer.rank_links(links_with_context, topic_terms, top_n=8)
                for link_url, link_score in ranked:
                    if self.normalizer.normalize(link_url) not in visited:
                        # 综合分数 = 链接分数 * 页面质量
                        combined = link_score * q_score
                        heapq.heappush(queue, (-combined, link_url, depth + 1))
                        stats.total_links_followed += 1

            # 礼貌延迟
            time.sleep(0.5 + 0.5 * (1.0 - q_score))  # 高质量页面等短一点

        stats.elapsed = time.time() - t0
        logger.info(f"智能爬取完成: {stats.indexed}/{stats.visited} 页入索引, "
                     f"跳过 {stats.skipped_dup} 重复, {stats.skipped_low_quality} 低质量, "
                     f"耗时 {stats.elapsed:.1f}s")
        # Deep Coupling: 发布爬取完成事件
        try:
            from taiji.infra.events import get_event_bus
            bus = get_event_bus()
            bus.publish("crawl_complete", {"indexed": stats.indexed}, source="smart_crawler")
        except Exception:
            pass
        return stats

    def crawl_from_search(self, query: str, max_pages: int = 20) -> CrawlStats:
        """
        从搜索结果出发爬取：先搜索，再对 top 结果做主题聚焦爬取。

        这是"搜索 + 智能爬取"的组合拳。
        """
        from .discovery import WebSearchProvider
        provider = WebSearchProvider()
        results = provider.search(query, max_results=10)
        if not results:
            logger.warning("搜索无结果，无法爬取")
            return CrawlStats()

        # 对每个搜索结果启动小规模爬取
        total_stats = CrawlStats()
        pages_per_site = max(2, max_pages // len(results))

        for r in results[:5]:  # 最多对 5 个站点爬取
            if not r.url:
                continue
            logger.info(f"从搜索结果爬取: {r.title[:50]} ({r.url[:60]})")
            s = self.crawl_topic(r.url, query, max_pages=pages_per_site, max_depth=2)
            total_stats.visited += s.visited
            total_stats.indexed += s.indexed
            total_stats.skipped_dup += s.skipped_dup
            total_stats.skipped_low_quality += s.skipped_low_quality
            total_stats.failed += s.failed
            total_stats.total_links_found += s.total_links_found
            total_stats.total_links_followed += s.total_links_followed
            total_stats.elapsed += s.elapsed
            if total_stats.indexed >= max_pages:
                break

        logger.info(f"搜索+爬取完成: {total_stats.indexed} 页入索引, 耗时 {total_stats.elapsed:.1f}s")
        return total_stats

    def _extract_links_with_context(self, html: str, base_url: str) -> List[Tuple[str, str, str]]:
        """
        提取链接及其锚文本和位置。

        Returns:
            [(url, anchor_text, position), ...]
            position: "content" / "nav" / "sidebar" / "footer"
        """
        results = []
        base_domain = urllib.parse.urlparse(base_url).netloc
        base_scheme = urllib.parse.urlparse(base_url).scheme

        # 提取所有 <a href="...">text</a>
        link_pattern = re.compile(
            r'<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            re.DOTALL | re.IGNORECASE,
        )

        for match in link_pattern.finditer(html):
            href = match.group(1)
            anchor_html = match.group(2)
            anchor_text = re.sub(r"<[^>]+>", "", anchor_html).strip()

            # 跳过页内锚点、javascript、mailto
            if href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
                continue

            # 相对路径转绝对路径
            if href.startswith("/"):
                href = f"{base_scheme}://{base_domain}{href}"
            elif not href.startswith(("http://", "https://")):
                href = urllib.parse.urljoin(base_url, href)

            # 只保留 http/https 链接
            if not href.startswith(("http://", "https://")):
                continue

            # 只保留同域链接
            try:
                if urllib.parse.urlparse(href).netloc != base_domain:
                    continue
            except Exception:
                continue

            # 跳过非 HTML 资源
            href_lower = href.lower()
            if any(href_lower.endswith(ext) for ext in
                   [".jpg", ".png", ".gif", ".css", ".js", ".pdf", ".zip"]):
                continue

            # 判断位置：通过周围的 HTML class/id
            position = "content"
            before = html[:match.start()]
            last_tag = re.findall(r'<(?:div|nav|aside|footer|header|section)\s[^>]*class="([^"]*)"', before)
            if last_tag:
                last_class = last_tag[-1].lower()
                if any(x in last_class for x in ["nav", "menu", "header"]):
                    position = "nav"
                elif any(x in last_class for x in ["sidebar", "aside", "widget"]):
                    position = "sidebar"
                elif any(x in last_class for x in ["footer", "copyright"]):
                    position = "footer"

            results.append((href, anchor_text, position))

        return results

    def close(self):
        self.fetcher.close()


# ═══════════════════════════════════════════════
# 统一入口
# ═══════════════════════════════════════════════

_default_crawler: Optional[SmartCrawler] = None


def get_smart_crawler() -> SmartCrawler:
    global _default_crawler
    if _default_crawler is None:
        _default_crawler = SmartCrawler()
    return _default_crawler


def tool_smart_crawl(query: str) -> str:
    """
    智能爬取工具接口：搜索 → 主题聚焦爬取 → 入索引。

    供 tool_registry 注册。
    """
    crawler = get_smart_crawler()
    stats = crawler.crawl_from_search(query, max_pages=15)
    return (f"## 智能爬取: {query}\n\n"
            f"访问 {stats.visited} 页，入索引 {stats.indexed} 页\n"
            f"跳过重复 {stats.skipped_dup}，低质量 {stats.skipped_low_quality}，失败 {stats.failed}\n"
            f"发现链接 {stats.total_links_found}，跟踪 {stats.total_links_followed}\n"
            f"耗时 {stats.elapsed:.1f}s\n"
            f"索引统计: {crawler.index.stats()}")
