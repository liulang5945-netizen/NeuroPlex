"""
态极 SearXNG 集成
==================

SearXNG 是一个开源的元搜索引擎，聚合 70+ 搜索引擎。

集成方式：
1. 本地部署：Docker 一键部署，完全私有
2. 公共实例：使用社区维护的公共实例
3. 自定义：连接任何 SearXNG 实例

SearXNG 优势：
- 聚合 Google、Bing、DuckDuckGo、Yahoo、Brave、Qwant 等 70+ 引擎
- 无追踪、无 profiling
- 支持 JSON API
- 支持分类搜索（新闻、图片、视频、学术等）

使用方式：
    from taiji.tools.searxng import searxng_search
    results = searxng_search("Python programming")
"""
import json
import logging
import urllib.request
import urllib.parse
from typing import List, Dict, Optional

logger = logging.getLogger("Taiji.SearXNG")

# 公共 SearXNG 实例列表（会自动尝试）
PUBLIC_INSTANCES = [
    "https://search.sapti.me",
    "https://searx.be",
    "https://search.bus-hit.me",
    "https://searx.info",
    "https://search.ononoki.org",
    "https://searxng.site",
]

# 本地实例地址（如果用户自己部署）
LOCAL_INSTANCE = "http://localhost:8080"


class SearXNGClient:
    """SearXNG API 客户端"""

    def __init__(self, instance_url: str = None):
        self.instance_url = instance_url
        self._working_instance = None

    def _find_working_instance(self) -> str:
        """找到一个可用的实例"""
        if self._working_instance:
            return self._working_instance

        # 先检查本地
        candidates = [LOCAL_INSTANCE] + PUBLIC_INSTANCES
        if self.instance_url:
            candidates.insert(0, self.instance_url)

        for url in candidates:
            try:
                test_url = f"{url}/search?q=test&format=json"
                req = urllib.request.Request(test_url, headers={
                    "User-Agent": "TaijiBot/1.0",
                    "Accept": "application/json",
                })
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        self._working_instance = url
                        logger.info(f"SearXNG 实例可用: {url}")
                        return url
            except Exception:
                continue

        logger.warning("没有可用的 SearXNG 实例")
        return None

    def search(self, query: str, categories: str = "general",
               language: str = "auto", max_results: int = 10) -> List[Dict]:
        """
        搜索。

        Args:
            query: 搜索关键词
            categories: 搜索类别 (general/images/videos/news/music/files/science/it)
            language: 语言 (auto/zh/en/...)
            max_results: 最大结果数

        Returns:
            搜索结果列表
        """
        instance = self._find_working_instance()
        if not instance:
            return []

        try:
            params = urllib.parse.urlencode({
                "q": query,
                "format": "json",
                "categories": categories,
                "language": language,
                "pageno": 1,
            })
            url = f"{instance}/search?{params}"

            req = urllib.request.Request(url, headers={
                "User-Agent": "TaijiBot/1.0",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = []
            for item in data.get("results", [])[:max_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                    "engine": item.get("engine", ""),
                    "score": item.get("score", 0),
                })

            return results
        except Exception as e:
            logger.warning(f"SearXNG 搜索失败: {e}")
            return []

    def search_news(self, query: str, max_results: int = 5) -> List[Dict]:
        """新闻搜索"""
        return self.search(query, categories="news", max_results=max_results)

    def search_science(self, query: str, max_results: int = 5) -> List[Dict]:
        """学术搜索"""
        return self.search(query, categories="science", max_results=max_results)

    def search_it(self, query: str, max_results: int = 5) -> List[Dict]:
        """IT/技术搜索"""
        return self.search(query, categories="it", max_results=max_results)


# 全局客户端
_client: Optional[SearXNGClient] = None


def get_searxng_client() -> SearXNGClient:
    """获取全局 SearXNG 客户端"""
    global _client
    if _client is None:
        _client = SearXNGClient()
    return _client


def _auto_category(query: str) -> str:
    """根据查询内容自动选择搜索分类"""
    q = query.lower()

    # 新闻类
    if any(kw in q for kw in ["新闻", "最新", "今日", "近日", "发生", "事件", "报道", "news", "latest"]):
        return "news"

    # 学术类
    if any(kw in q for kw in ["论文", "研究", "学术", "期刊", "paper", "research", "study", "journal"]):
        return "science"

    # IT/技术类
    if any(kw in q for kw in ["代码", "编程", "api", "sdk", "github", "框架", "库", "code", "programming", "framework"]):
        return "it"

    # 图片类
    if any(kw in q for kw in ["图片", "照片", "图像", "image", "photo", "picture"]):
        return "images"

    # 视频类
    if any(kw in q for kw in ["视频", "教程", "video", "tutorial", "youtube"]):
        return "videos"

    # 音乐类
    if any(kw in q for kw in ["音乐", "歌曲", "music", "song"]):
        return "music"

    return "general"


def searxng_search(query: str, categories: str = "auto", max_results: int = 5) -> str:
    """
    SearXNG 智能搜索（供工具调用）。

    态极根据查询内容自动选择最合适的搜索分类：
    - 新闻 → news
    - 学术 → science
    - 代码/技术 → it
    - 图片 → images
    - 视频 → videos
    - 音乐 → music
    - 其他 → general

    聚合 70+ 搜索引擎的结果。
    """
    # 自动选择分类
    if categories == "auto":
        categories = _auto_category(query)

    client = get_searxng_client()
    results = client.search(query, categories=categories, max_results=max_results)

    if not results:
        return f"SearXNG 搜索 '{query}' 未找到结果。可能是没有可用的 SearXNG 实例。"

    lines = [f"SearXNG 搜索 '{query}' 找到 {len(results)} 条结果:\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        if r['content']:
            lines.append(f"   {r['content'][:200]}")
        lines.append(f"   URL: {r['url']}")
        if r['engine']:
            lines.append(f"   来源: {r['engine']}")
        lines.append("")

    return "\n".join(lines)
