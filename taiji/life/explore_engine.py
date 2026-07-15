"""
态极自主探索引擎 (Explore Engine)
==================================

态极的"好奇心"驱动的自主学习能力。

不同于被动回答问题，态极会：
1. 感知自己的知识盲区（哪些领域了解不足）
2. 主动搜索相关信息（联网检索）
3. 阅读并理解网页内容
4. 将知识存入记忆和知识库
5. 在睡觉时整合到模型中

这就是 OpenClaw 的核心：自主爬取 → 理解 → 学习。

生命节律：
  好奇心 > 70 → 自主探索
  发现新领域 → 深入搜索
  知识积累 → 睡眠整合
"""
import os
import json
import time
import logging
import hashlib
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("ExploreEngine")


@dataclass
class ExploreResult:
    """一次探索的结果"""
    topic: str
    sources_found: int = 0
    pages_read: int = 0
    knowledge_stored: int = 0
    duration_seconds: float = 0
    discoveries: List[str] = field(default_factory=list)
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class ExploreConfig:
    """探索配置"""
    auto_explore_enabled: bool = True
    curiosity_threshold: float = 70.0      # 好奇心超过此值时触发探索
    max_pages_per_explore: int = 5         # 每次探索最多读几个网页
    max_search_results: int = 3            # 每次搜索最多几个结果
    explore_cooldown_minutes: int = 30     # 探索冷却时间
    knowledge_dir: str = ""                # 知识存储目录


class ExploreEngine:
    """
    态极自主探索引擎

    好奇心驱动，自动搜索、阅读、学习。
    """

    def __init__(self, config: Optional[ExploreConfig] = None, data_dir: str = None):
        self.config = config or ExploreConfig()
        if data_dir is None:
            try:
                from taiji.config import get_taiji_data_path
                data_dir = get_taiji_data_path("explore_data")
            except ImportError:
                data_dir = "taiji/explore_data"
        self.data_dir = data_dir
        self._explore_history: List[ExploreResult] = []
        self._known_topics: set = set()  # 已探索的主题
        self._content_hashes: set = set()  # 去重用
        self._last_explore_time: Optional[datetime] = None
        self._is_exploring = False

        self._data_dir_ready = False
        self._load_history()

        logger.info("ExploreEngine initialized")

    # ─── 公开接口 ───────────────────────────────────

    def explore(self, topic: str = "", reason: str = "auto") -> ExploreResult:
        """
        让态极自主探索一个主题。

        Args:
            topic: 指定主题（空则自动选择）
            reason: 探索原因

        Returns:
            ExploreResult 探索报告
        """
        if self._is_exploring:
            logger.warning("Already exploring, skipping")
            return ExploreResult(topic="")

        self._is_exploring = True
        start_time = time.time()

        # 自动选择主题
        if not topic:
            topic = self._choose_topic()

        logger.info(f"🔍 态极开始探索: {topic} (reason: {reason})")

        result = ExploreResult(topic=topic)

        try:
            # 1. 搜索
            search_results = self._search(topic)
            result.sources_found = len(search_results)

            # 2. 阅读网页
            for url in search_results[:self.config.max_pages_per_explore]:
                content = self._read_page(url)
                if content:
                    result.pages_read += 1

                    # 3. 存储知识
                    stored = self._store_knowledge(topic, url, content)
                    if stored:
                        result.knowledge_stored += 1

            # 4. 记录发现
            result.discoveries = self._extract_discoveries(topic, search_results)

            # 5. 通知进化引擎
            self._record_to_evolution(topic, result)

        except Exception as e:
            logger.error(f"Explore failed: {e}")

        result.duration_seconds = round(time.time() - start_time, 1)
        self._last_explore_time = datetime.now()
        self._is_exploring = False

        # 保存
        self._explore_history.append(result)
        self._known_topics.add(topic)
        self._save_history()

        logger.info(
            f"🔍 探索完成: {topic} | "
            f"搜索 {result.sources_found} | 阅读 {result.pages_read} | "
            f"存储 {result.knowledge_stored} | 耗时 {result.duration_seconds}s"
        )

        return result

    def should_explore(self) -> bool:
        """判断是否应该探索（好奇心驱动）"""
        try:
            from taiji.life.life_scheduler import get_life_scheduler
            life = get_life_scheduler()
            curiosity = life.needs.curiosity

            if curiosity < self.config.curiosity_threshold:
                return False

            # 冷却时间
            if self._last_explore_time:
                elapsed = (datetime.now() - self._last_explore_time).total_seconds() / 60
                if elapsed < self.config.explore_cooldown_minutes:
                    return False

            return True
        except Exception:
            return False

    def get_status(self) -> dict:
        """获取探索引擎状态"""
        return {
            "is_exploring": self._is_exploring,
            "total_explores": len(self._explore_history),
            "known_topics": len(self._known_topics),
            "last_explore": self._last_explore_time.isoformat() if self._last_explore_time else None,
            "auto_explore_enabled": self.config.auto_explore_enabled,
        }

    # ─── 内部实现 ───────────────────────────────────

    def _choose_topic(self) -> str:
        """自动选择探索主题（基于知识短板和好奇心）"""
        # 从进化引擎获取知识域
        try:
            from taiji.life.evolution_engine import get_evolution_engine
            evo = get_evolution_engine()
            weak_domains = [
                d for d, score in evo.metrics.knowledge_domains.items()
                if score < 3.0
            ]
            if weak_domains:
                import random
                return random.choice(weak_domains)
        except Exception as e:
            logger.debug("explore_engine: non-critical %s", e, exc_info=True)

        # 从好奇心主题池选择
        curiosity_topics = [
            "人工智能最新进展",
            "量子计算原理",
            "宇宙的起源",
            "人类大脑如何工作",
            "编程语言发展趋势",
            "机器学习算法",
            "自然语言处理技术",
            "计算机视觉应用",
            "机器人技术发展",
            "生物信息学",
        ]
        import random
        topic = random.choice(curiosity_topics)

        # 避免重复探索
        attempts = 0
        while topic in self._known_topics and attempts < 10:
            topic = random.choice(curiosity_topics)
            attempts += 1

        return topic

    def _search(self, query: str) -> List[str]:
        """搜索互联网，返回 URL 列表"""
        try:
            from taiji.agent_ext.tool_registry import registry
            search_tool = registry.get("search")
            if search_tool and search_tool.func:
                result = search_tool.func(query)
                # 从搜索结果中提取 URL
                if isinstance(result, str):
                    urls = []
                    for line in result.split("\n"):
                        line = line.strip()
                        if line.startswith("http"):
                            urls.append(line.split(" ")[0])
                        elif "http" in line:
                            import re
                            found = re.findall(r'https?://[^\s\)]+', line)
                            urls.extend(found)
                    return urls[:self.config.max_search_results]
        except Exception as e:
            logger.debug(f"Search failed: {e}")
        return []

    def _read_page(self, url: str) -> Optional[str]:
        """阅读网页内容"""
        try:
            from taiji.agent_ext.tool_registry import registry
            read_tool = registry.get("read_webpage")
            if read_tool and read_tool.func:
                content = read_tool.func(url)
                if content and len(content) > 100:
                    return content[:5000]  # 限制长度
        except Exception as e:
            logger.debug(f"Read page failed: {e}")
        return None

    def _ensure_data_dir(self):
        """延迟创建数据目录（只在首次写入时创建）"""
        if not self._data_dir_ready:
            os.makedirs(self.data_dir, exist_ok=True)
            self._data_dir_ready = True

    def _store_knowledge(self, topic: str, url: str, content: str) -> bool:
        """将知识存入知识库"""
        # 去重
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if content_hash in self._content_hashes:
            return False

        self._ensure_data_dir()
        try:
            knowledge_dir = os.path.join(self.data_dir, "discoveries")
            os.makedirs(knowledge_dir, exist_ok=True)

            # 保存为知识文件
            safe_topic = topic.replace("/", "_").replace("\\", "_")[:50]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_topic}_{timestamp}.json"
            filepath = os.path.join(knowledge_dir, filename)

            knowledge = {
                "topic": topic,
                "url": url,
                "content": content[:3000],
                "discovered_at": datetime.now().isoformat(),
                "content_hash": content_hash,
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(knowledge, f, indent=2, ensure_ascii=False)

            self._content_hashes.add(content_hash)

            # 同时存入 RAG 知识库
            try:
                from taiji.tools.rag_kb import get_rag_kb
                kb = get_rag_kb()
                if kb:
                    kb.add_document(content[:2000], metadata={"topic": topic, "source": url})
            except Exception as e:
                logger.debug("explore_engine: non-critical %s", e, exc_info=True)

            return True
        except Exception as e:
            logger.warning(f"Store knowledge failed: {e}")
            return False

    def _extract_discoveries(self, topic: str, urls: List[str]) -> List[str]:
        """从探索中提取发现"""
        discoveries = []
        if urls:
            discoveries.append(f"在 {topic} 领域发现了 {len(urls)} 个信息源")
        return discoveries

    def _record_to_evolution(self, topic: str, result: ExploreResult):
        """记录到进化引擎"""
        try:
            from taiji.life.evolution_engine import get_evolution_engine
            evo = get_evolution_engine()
            # 探索成功 → 增加知识域分数
            evo.metrics.knowledge_domains[topic] = \
                evo.metrics.knowledge_domains.get(topic, 0) + 1.0
            evo._save_metrics()
        except Exception as e:
            logger.debug("explore_engine: non-critical %s", e, exc_info=True)

    # ─── 持久化 ─────────────────────────────────────

    def _save_history(self):
        """保存探索历史"""
        path = os.path.join(self.data_dir, "explore_history.json")
        try:
            data = []
            for r in self._explore_history[-50:]:
                data.append({
                    "topic": r.topic,
                    "sources_found": r.sources_found,
                    "pages_read": r.pages_read,
                    "knowledge_stored": r.knowledge_stored,
                    "duration_seconds": r.duration_seconds,
                    "discoveries": r.discoveries,
                    "timestamp": r.timestamp,
                })
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save explore history: {e}")

    def _load_history(self):
        """加载探索历史"""
        path = os.path.join(self.data_dir, "explore_history.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                self._explore_history.append(ExploreResult(**item))
                self._known_topics.add(item.get("topic", ""))
        except Exception as e:
            logger.warning(f"Failed to load explore history: {e}")


# ─── 全局实例 ─────────────────────────────────────

_global_explore: Optional[ExploreEngine] = None


def get_explore_engine() -> ExploreEngine:
    """获取全局探索引擎实例"""
    global _global_explore
    if _global_explore is None:
        _global_explore = ExploreEngine()
    return _global_explore
