"""
态极喂养引擎 (Feed Engine)
============================

态极的"吃饭"能力 — 与睡眠引擎对应的生命节律。

睡觉是"消化整理"，吃饭是"摄入新知"。
态极通过喂养引擎主动从外部源获取知识、收集交互数据、
评估数据质量，为睡眠时的训练准备"营养"。

生命节律：
  ☀️ 醒来 → 🍚 吃饭（收集新知识）→ 🏃 活动（服务用户）
  → 🍚 吃饭（收集交互数据）→ 💤 睡觉（消化训练）→ ☀️ 醒来

核心能力：
1. 知识进食：从外部源（文件、网页、API）获取新知识
2. 数据消化：将原始数据转换为训练样本
3. 营养评估：评估数据质量，过滤垃圾
4. 进食计划：根据能力短板有针对性地"点菜"
5. 进食记录：追踪吃了什么、消化了多少
"""
import os
import json
import time
import logging
import hashlib
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger("FeedEngine")


@dataclass
class FeedItem:
    """一次进食记录"""
    source: str           # 来源类型: "file", "web", "api", "conversation", "knowledge"
    source_path: str      # 来源路径/URL
    content_hash: str     # 内容哈希（去重用）
    quality_score: float  # 营养评分 0~1
    category: str         # 分类: "code", "knowledge", "conversation", "creative"
    sample_count: int     # 生成的训练样本数
    timestamp: str = ""
    status: str = "pending"  # pending / digested / rejected

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


@dataclass
class FeedReport:
    """一次进食的报告"""
    timestamp: str
    duration_seconds: float
    items_fed: int = 0
    items_rejected: int = 0
    samples_generated: int = 0
    categories: Dict[str, int] = field(default_factory=dict)
    avg_quality: float = 0.0
    recommendations: List[str] = field(default_factory=list)


@dataclass
class FeedConfig:
    """喂养配置"""
    auto_feed_enabled: bool = True
    feed_interval_hours: float = 2.0     # 每 2 小时喂一次
    min_quality_score: float = 0.3       # 最低营养评分
    max_items_per_feed: int = 100        # 每次最多吃多少
    max_content_length: int = 10000      # 单条内容最大长度
    dedup_enabled: bool = True           # 去重
    category_weights: Dict[str, float] = field(default_factory=lambda: {
        "code": 1.0,
        "knowledge": 0.8,
        "conversation": 0.6,
        "creative": 0.5,
    })


class FeedEngine:
    """
    态极的喂养引擎

    与 SleepEngine 配合，形成完整的"吃饭→睡觉"生命节律。
    喂养引擎负责"吃"（收集数据），睡眠引擎负责"消化"（训练模型）。
    """

    def __init__(self, config: Optional[FeedConfig] = None,
                 data_dir: str = None):
        self.config = config or FeedConfig()
        if data_dir is None:
            try:
                from taiji.config import get_taiji_data_path
                data_dir = get_taiji_data_path("feed_data")
            except ImportError:
                data_dir = "taiji/feed_data"
        self.data_dir = data_dir
        self._feed_history: List[FeedReport] = []
        self._feed_items: List[FeedItem] = []
        self._content_hashes: set = set()  # 去重用
        self._last_feed_time: Optional[datetime] = None
        self._on_feed_complete: Optional[Callable] = None

        self._data_dir_ready = False
        self._load_history()

        logger.info(f"FeedEngine initialized: auto={self.config.auto_feed_enabled}")

    # ─── 公开接口 ───────────────────────────────────

    def feed(self, reason: str = "manual") -> FeedReport:
        """
        让态极"吃饭" — 从各来源收集新知识。

        Args:
            reason: 喂养原因（"manual", "auto", "scheduled"）

        Returns:
            FeedReport 喂养报告
        """
        start_time = time.time()
        logger.info(f"🍚 Taiji is eating... (reason: {reason})")

        report = FeedReport(
            timestamp=datetime.now().isoformat(),
            duration_seconds=0,
        )

        # 1. 吃用户交互数据
        self._feed_from_conversations(report)

        # 2. 吃知识库数据
        self._feed_from_knowledge_store(report)

        # 3. 吃收集器数据
        self._feed_from_data_collector(report)

        # 4. 评估营养、过滤垃圾
        self._digest_and_filter(report)

        # 5. 生成进食计划建议
        self._generate_recommendations(report)

        # 计算进食时长
        report.duration_seconds = round(time.time() - start_time, 1)
        self._last_feed_time = datetime.now()

        # 保存报告
        self._feed_history.append(report)
        self._save_history()
        self._save_pending_items()

        logger.info(
            f"🍽️ Taiji finished eating! "
            f"Items: {report.items_fed}, Samples: {report.samples_generated}, "
            f"Avg quality: {report.avg_quality:.2f}, Duration: {report.duration_seconds}s"
        )

        return report

    def feed_file(self, file_path: str, category: str = "knowledge") -> Optional[FeedItem]:
        """
        喂态极吃一个文件。

        Args:
            file_path: 文件路径
            category: 分类

        Returns:
            FeedItem 或 None（如果质量不达标）
        """
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read(self.config.max_content_length)

            return self._process_content(
                content=content,
                source="file",
                source_path=file_path,
                category=category,
            )
        except Exception as e:
            logger.warning(f"Failed to feed file {file_path}: {e}")
            return None

    def feed_text(self, text: str, source: str = "manual",
                  category: str = "knowledge") -> Optional[FeedItem]:
        """
        直接喂态极一段文字。

        Args:
            text: 文本内容
            source: 来源描述
            category: 分类

        Returns:
            FeedItem 或 None
        """
        return self._process_content(
            content=text[:self.config.max_content_length],
            source="text",
            source_path=source,
            category=category,
        )

    def feed_directory(self, dir_path: str, extensions: List[str] = None,
                       category: str = "code") -> int:
        """
        喂态极吃一个目录下的所有文件。

        Returns:
            成功进食的文件数
        """
        if not os.path.isdir(dir_path):
            logger.warning(f"Directory not found: {dir_path}")
            return 0

        extensions = extensions or [".py", ".js", ".ts", ".md", ".txt", ".json"]
        count = 0

        for root, dirs, files in os.walk(dir_path):
            # 跳过隐藏目录和常见无用目录
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in
                       {'__pycache__', 'node_modules', '.git', 'venv', 'dist', 'build'}]

            for fname in files:
                if any(fname.endswith(ext) for ext in extensions):
                    fpath = os.path.join(root, fname)
                    item = self.feed_file(fpath, category=category)
                    if item and item.status != "rejected":
                        count += 1
                    if count >= self.config.max_items_per_feed:
                        return count

        return count

    def get_pending_samples(self) -> List[dict]:
        """
        获取待消化的训练样本（供睡眠引擎调用）。

        Returns:
            训练样本列表
        """
        samples_path = os.path.join(self.data_dir, "pending_samples.jsonl")
        if not os.path.exists(samples_path):
            return []

        samples = []
        try:
            with open(samples_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        except Exception as e:
            logger.warning(f"Failed to load pending samples: {e}")

        return samples

    def clear_pending_samples(self):
        """清除已消化的样本（睡眠训练后调用）"""
        samples_path = os.path.join(self.data_dir, "pending_samples.jsonl")
        if os.path.exists(samples_path):
            os.remove(samples_path)
            logger.info("Pending samples cleared")

    # ─── 内部实现 ───────────────────────────────────

    def _process_content(self, content: str, source: str,
                         source_path: str, category: str) -> Optional[FeedItem]:
        """处理一段内容：评估质量、去重、生成样本"""
        # 去重
        content_hash = hashlib.md5(content.encode()).hexdigest()
        if self.config.dedup_enabled and content_hash in self._content_hashes:
            logger.debug(f"Duplicate content skipped: {source_path}")
            return None

        # 评估营养质量
        quality = self._assess_quality(content, category)

        item = FeedItem(
            source=source,
            source_path=source_path,
            content_hash=content_hash,
            quality_score=quality,
            category=category,
            sample_count=0,
        )

        # 质量不达标
        if quality < self.config.min_quality_score:
            item.status = "rejected"
            logger.debug(f"Low quality rejected: {source_path} (score={quality:.2f})")
            return item

        # 生成训练样本
        samples = self._generate_samples(content, category, source_path)
        item.sample_count = len(samples)
        item.status = "digested"

        # 保存样本
        self._append_samples(samples)
        self._content_hashes.add(content_hash)
        self._feed_items.append(item)

        return item

    def _assess_quality(self, content: str, category: str) -> float:
        """
        评估内容的"营养价值"。

        评分维度：
        - 长度合理性（太短=没营养，太长=可能是垃圾）
        - 信息密度（非空行比例）
        - 结构化程度（代码有缩进、文档有标题等）
        - 重复度（重复行越多越低）
        """
        if not content or not content.strip():
            return 0.0

        score = 0.0
        lines = content.split('\n')
        non_empty = [l for l in lines if l.strip()]

        # 1. 长度合理性 (0~0.3)
        length = len(content)
        if 50 < length < 5000:
            score += 0.3
        elif 20 < length <= 50 or 5000 <= length < self.config.max_content_length:
            score += 0.2
        elif length >= 10:
            score += 0.1

        # 2. 信息密度 (0~0.3)
        if lines:
            density = len(non_empty) / len(lines)
            score += density * 0.3

        # 3. 结构化程度 (0~0.2)
        if category == "code":
            # 代码应有缩进
            indented = sum(1 for l in non_empty if l.startswith(('    ', '\t')))
            if non_empty:
                score += (indented / len(non_empty)) * 0.2
        else:
            # 文档应有标题/列表
            structured = sum(1 for l in non_empty if l.startswith(('#', '-', '*', '1.', '2.')))
            if non_empty:
                score += min(structured / len(non_empty), 1.0) * 0.2

        # 4. 去重惩罚 (0~0.2)
        if non_empty:
            unique_ratio = len(set(l.strip() for l in non_empty)) / len(non_empty)
            score += unique_ratio * 0.2

        # 类别权重
        weight = self.config.category_weights.get(category, 0.5)
        score *= weight

        return min(1.0, score)

    def _generate_samples(self, content: str, category: str,
                          source_path: str) -> List[dict]:
        """将内容转换为训练样本"""
        samples = []

        if category == "code":
            # 代码 → ReAct 样本（"分析这段代码"）
            samples.extend(self._code_to_samples(content, source_path))
        elif category == "conversation":
            # 对话 → 对话样本
            samples.extend(self._conversation_to_samples(content))
        else:
            # 通用知识 → 问答样本
            samples.extend(self._knowledge_to_samples(content, source_path))

        return samples

    def _code_to_samples(self, code: str, source_path: str) -> List[dict]:
        """代码 → ReAct 训练样本"""
        samples = []
        fname = os.path.basename(source_path)

        # 按函数/类分段
        segments = self._split_code_segments(code)

        for seg_name, seg_code in segments[:5]:  # 最多 5 个
            if len(seg_code.strip()) < 20:
                continue

            # 生成"分析代码"的 ReAct 样本
            task = f"分析 {fname} 中的 {seg_name} 的功能和实现"
            thought = f"这是一个名为 {seg_name} 的代码段，我需要理解它的功能。"
            answer = f"这段代码实现了 {seg_name} 的功能。"

            samples.append({
                "type": "react",
                "task": task,
                "steps": [{
                    "thought": thought,
                    "action": "read_local_file",
                    "action_args": {"input": source_path},
                    "observation": seg_code[:500],
                    "final_answer": answer,
                }],
            })

        return samples

    def _conversation_to_samples(self, content: str) -> List[dict]:
        """对话文本 → 对话训练样本"""
        samples = []
        lines = content.strip().split('\n')

        messages = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 简单解析 "角色: 内容" 格式
            if ':' in line:
                parts = line.split(':', 1)
                role = parts[0].strip().lower()
                text = parts[1].strip()
                if role in ('user', 'human', '用户'):
                    messages.append({"role": "user", "content": text})
                elif role in ('assistant', 'ai', '助手'):
                    messages.append({"role": "assistant", "content": text})

        if len(messages) >= 2:
            samples.append({"type": "conversation", "messages": messages})

        return samples

    def _knowledge_to_samples(self, content: str, source_path: str) -> List[dict]:
        """知识文本 → 问答训练样本"""
        samples = []
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 30]

        for i, para in enumerate(paragraphs[:5]):  # 最多 5 段
            # 将每段转换为问答对
            title = para.split('\n')[0].strip('# -')
            task = f"解释以下概念: {title}"
            answer = para[:300]

            samples.append({
                "type": "react",
                "task": task,
                "steps": [{
                    "thought": f"让我查阅关于 {title} 的知识。",
                    "action": "query_knowledge",
                    "action_args": {"input": title},
                    "observation": answer,
                    "final_answer": answer,
                }],
            })

        return samples

    def _split_code_segments(self, code: str) -> List[tuple]:
        """将代码按函数/类分段，返回 [(name, code), ...]"""
        segments = []
        lines = code.split('\n')
        current_name = "module_level"
        current_lines = []

        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith(('def ', 'class ', 'async def ')):
                # 保存上一段
                if current_lines:
                    segments.append((current_name, '\n'.join(current_lines)))
                # 提取名称
                parts = stripped.split('(')[0].split()
                current_name = parts[-1] if len(parts) > 1 else "unknown"
                current_lines = [line]
            else:
                current_lines.append(line)

        if current_lines:
            segments.append((current_name, '\n'.join(current_lines)))

        return segments

    def _feed_from_conversations(self, report: FeedReport):
        """从用户对话历史中进食"""
        try:
            from taiji.agent_ext.data_collector import get_collector
            collector = get_collector()
            react_data, conv_data = collector.load_as_training_data()

            for item in conv_data[:self.config.max_items_per_feed]:
                content = json.dumps(item, ensure_ascii=False)
                feed_item = self._process_content(
                    content=content,
                    source="conversation",
                    source_path="data_collector",
                    category="conversation",
                )
                if feed_item:
                    report.items_fed += 1 if feed_item.status == "digested" else 0
                    report.items_rejected += 1 if feed_item.status == "rejected" else 0
                    report.samples_generated += feed_item.sample_count

        except (ImportError, Exception) as e:
            logger.debug(f"Feed from conversations skipped: {e}")

    def _feed_from_knowledge_store(self, report: FeedReport):
        """从知识库中进食"""
        knowledge_dir = os.path.join("knowledge_store", "entries")
        if not os.path.isdir(knowledge_dir):
            return

        for fname in os.listdir(knowledge_dir)[:self.config.max_items_per_feed]:
            if fname.endswith('.json'):
                fpath = os.path.join(knowledge_dir, fname)
                item = self.feed_file(fpath, category="knowledge")
                if item:
                    report.items_fed += 1 if item.status == "digested" else 0
                    report.items_rejected += 1 if item.status == "rejected" else 0
                    report.samples_generated += item.sample_count

    def _feed_from_data_collector(self, report: FeedReport):
        """从数据收集器中进食 ReAct 数据"""
        try:
            from taiji.agent_ext.data_collector import get_collector
            collector = get_collector()
            react_data, _ = collector.load_as_training_data()

            for item in react_data[:self.config.max_items_per_feed]:
                content = json.dumps(item, ensure_ascii=False)
                feed_item = self._process_content(
                    content=content,
                    source="react_collector",
                    source_path="data_collector",
                    category="code",
                )
                if feed_item:
                    report.items_fed += 1 if feed_item.status == "digested" else 0
                    report.items_rejected += 1 if feed_item.status == "rejected" else 0
                    report.samples_generated += feed_item.sample_count

        except (ImportError, Exception) as e:
            logger.debug(f"Feed from data collector skipped: {e}")

    def _digest_and_filter(self, report: FeedReport):
        """消化和过滤：统计类别、计算平均质量"""
        categories = {}
        total_quality = 0.0
        count = 0

        for item in self._feed_items:
            if item.status == "digested":
                cat = item.category
                categories[cat] = categories.get(cat, 0) + 1
                total_quality += item.quality_score
                count += 1

        report.categories = categories
        report.avg_quality = round(total_quality / max(count, 1), 3)

    def _generate_recommendations(self, report: FeedReport):
        """根据进食情况生成建议"""
        # 检查能力评估器，找出薄弱环节
        try:
            from taiji.infra.auto_upgrade import CapabilityEvaluator
            eval_path = os.path.join(self.data_dir, "..", "sleep_data", "capability_scores.json")
            if os.path.exists(eval_path):
                evaluator = CapabilityEvaluator(save_path=eval_path)
                scores = evaluator.scores
                weak_categories = [c for c, s in scores.items() if s < 0.4]
                if weak_categories:
                    report.recommendations.append(
                        f"能力薄弱领域: {', '.join(weak_categories)}，建议多吃相关数据"
                    )
        except Exception as e:
            logger.debug("feed_engine: non-critical %s", e, exc_info=True)

        # 检查数据量
        if report.samples_generated < 10:
            report.recommendations.append("进食数据量不足，建议手动喂养更多知识")
        if report.avg_quality < 0.4:
            report.recommendations.append("平均数据质量偏低，建议提供更高质量的知识源")

    # ─── 持久化 ─────────────────────────────────────

    def _ensure_data_dir(self):
        """延迟创建数据目录（只在首次写入时创建）"""
        if not self._data_dir_ready:
            os.makedirs(self.data_dir, exist_ok=True)
            self._data_dir_ready = True

    def _append_samples(self, samples: List[dict]):
        """追加训练样本到待消化队列"""
        self._ensure_data_dir()
        samples_path = os.path.join(self.data_dir, "pending_samples.jsonl")
        with open(samples_path, "a", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    def _save_pending_items(self):
        """保存待处理的进食项"""
        items_path = os.path.join(self.data_dir, "feed_items.json")
        recent_items = self._feed_items[-200:]  # 只保留最近 200 项
        try:
            data = []
            for item in recent_items:
                data.append({
                    "source": item.source,
                    "source_path": item.source_path,
                    "content_hash": item.content_hash,
                    "quality_score": item.quality_score,
                    "category": item.category,
                    "sample_count": item.sample_count,
                    "timestamp": item.timestamp,
                    "status": item.status,
                })
            with open(items_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save feed items: {e}")

    def _save_history(self):
        """保存进食历史"""
        path = os.path.join(self.data_dir, "feed_history.json")
        try:
            data = []
            for report in self._feed_history[-50:]:
                data.append({
                    "timestamp": report.timestamp,
                    "duration_seconds": report.duration_seconds,
                    "items_fed": report.items_fed,
                    "items_rejected": report.items_rejected,
                    "samples_generated": report.samples_generated,
                    "categories": report.categories,
                    "avg_quality": report.avg_quality,
                    "recommendations": report.recommendations,
                })
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save feed history: {e}")

    def _load_history(self):
        """加载进食历史"""
        path = os.path.join(self.data_dir, "feed_history.json")
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                self._feed_history.append(FeedReport(**item))
        except Exception as e:
            logger.warning(f"Failed to load feed history: {e}")

        # 加载去重哈希
        items_path = os.path.join(self.data_dir, "feed_items.json")
        if os.path.exists(items_path):
            try:
                with open(items_path, "r", encoding="utf-8") as f:
                    items_data = json.load(f)
                for item_data in items_data:
                    self._content_hashes.add(item_data.get("content_hash", ""))
            except Exception as e:
                logger.debug("feed_engine: non-critical %s", e, exc_info=True)

    # ─── 状态查询 ───────────────────────────────────

    def get_status(self) -> dict:
        """获取喂养引擎状态"""
        return {
            "last_feed": self._last_feed_time.isoformat() if self._last_feed_time else None,
            "total_feeds": len(self._feed_history),
            "total_items_fed": sum(r.items_fed for r in self._feed_history),
            "total_samples": sum(r.samples_generated for r in self._feed_history),
            "auto_feed_enabled": self.config.auto_feed_enabled,
            "known_content_hashes": len(self._content_hashes),
        }

    def get_summary(self) -> str:
        """获取人类可读的状态摘要"""
        status = self.get_status()
        last_feed = status["last_feed"] or "从未进食"

        lines = [
            "🍚 喂养引擎状态",
            "━━━━━━━━━━━━━━━━",
            f"上次进食: {last_feed}",
            f"总进食次数: {status['total_feeds']}",
            f"总摄入数据: {status['total_items_fed']} 条",
            f"总生成样本: {status['total_samples']} 条",
            f"已知内容: {status['known_content_hashes']} 条（去重用）",
            f"自动喂养: {'✅ 开启' if status['auto_feed_enabled'] else '❌ 关闭'}",
        ]

        if self._feed_history:
            last = self._feed_history[-1]
            lines.append(f"\n最近一次进食报告:")
            lines.append(f"  时长: {last.duration_seconds}s")
            lines.append(f"  进食: {last.items_fed} 条, 拒绝: {last.items_rejected} 条")
            lines.append(f"  生成样本: {last.samples_generated} 条")
            lines.append(f"  平均质量: {last.avg_quality:.2f}")
            if last.recommendations:
                lines.append(f"  建议: {last.recommendations[0]}")

        return "\n".join(lines)

    def get_feed_plan(self) -> Dict[str, Any]:
        """
        生成进食计划 — 根据能力短板推荐吃什么。

        Returns:
            {
                "weak_categories": [...],
                "recommended_sources": [...],
                "priority": "high" / "normal",
            }
        """
        plan = {
            "weak_categories": [],
            "recommended_sources": [],
            "priority": "normal",
        }

        try:
            from taiji.infra.auto_upgrade import CapabilityEvaluator
            eval_path = os.path.join(self.data_dir, "..", "sleep_data", "capability_scores.json")
            if os.path.exists(eval_path):
                evaluator = CapabilityEvaluator(save_path=eval_path)
                for cat, score in evaluator.scores.items():
                    if score < 0.4:
                        plan["weak_categories"].append({"category": cat, "score": score})
                        plan["priority"] = "high"

                        # 推荐对应的数据源
                        if cat == "code":
                            plan["recommended_sources"].append("喂养更多代码文件 (.py, .js)")
                        elif cat == "knowledge":
                            plan["recommended_sources"].append("喂养知识文档 (.md, .txt)")
                        elif cat == "creative":
                            plan["recommended_sources"].append("喂养创意文本和对话")
                        elif cat == "math":
                            plan["recommended_sources"].append("喂养数学题目和解答")
        except Exception as e:
            logger.debug("feed_engine: non-critical %s", e, exc_info=True)

        if not plan["weak_categories"]:
            plan["recommended_sources"].append("当前能力均衡，继续保持多样化的数据摄入")

        return plan


# ─── 全局实例 ─────────────────────────────────────

_global_feed: Optional[FeedEngine] = None


def get_feed_engine(config: Optional[FeedConfig] = None) -> FeedEngine:
    """获取全局喂养引擎实例"""
    global _global_feed
    if _global_feed is None:
        _global_feed = FeedEngine(config)
    return _global_feed