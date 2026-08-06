"""
态极递归改进系统 (Recursive Improver)
======================================

基于 Gödel Agent (ACL 2025) 和 Continual Harness (2026) 的思想：
态极可以改进自己的行为策略，并生成下一轮训练数据。

改进层次（神经元架构下的形态）：
1. 策略改进 — 优化 prompt、工具选择、反思模板（推理时记录，睡眠时分析）
2. 数据改进 — 生成下一轮训练数据建议（睡眠时；消费方：跨域协作层训练）

废弃（2026-08-06 清理）：
- 架构改进（design_next_generation）：神经元架构通过 neurogenesis 动态新增
  神经元，非代际变大——相关 ~300 行死代码已删除。

核心哲学：
态极不能改自己的权重，但可以设计更好的自己。
"""
import os
import json
import time
import logging
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

logger = logging.getLogger("RecursiveImprover")


@dataclass
class StrategyRecord:
    """一次策略使用记录"""
    strategy_type: str      # "prompt" | "tool_choice" | "reflection" | "planning"
    strategy_content: str   # 策略内容
    task: str               # 任务描述
    success: bool           # 是否成功
    quality_score: float    # 质量评分 0-1
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ImprovementProposal:
    """一次改进提案"""
    proposal_type: str      # "prompt" | "tool" | "reflection" | "architecture"
    description: str        # 改进描述
    old_value: str          # 旧值
    new_value: str          # 新值
    confidence: float       # 置信度 0-1
    evidence_count: int     # 支持证据数量
    # 神经元架构扩展：标记是否需要新建神经元 + 目标域
    needs_new_neuron: bool = False   # 是否需要 neurogenesis 创建新神经元
    target_domain: str = ""          # 目标域（zh/en/code/math/general）


class RecursiveImprover:
    """
    态极递归改进系统

    不修改模型权重，而是改进模型的行为策略。
    每次改进都基于实际任务数据，不是凭空想象。
    """

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            try:
                from taiji.config import get_taiji_data_path
                data_dir = get_taiji_data_path("improvement_data")
            except ImportError:
                data_dir = "taiji_data/improvement_data"
        self.data_dir = data_dir
        self._data_dir_ready = False

        # 策略记录
        self._strategy_records: List[StrategyRecord] = []
        self._load_records()

        # 当前最优策略
        self._best_strategies: Dict[str, str] = {
            "system_prompt": "",
            "tool_priority": "",
            "reflection_template": "",
            "planning_template": "",
        }
        self._load_best_strategies()

        # 改进历史
        self._improvements: List[ImprovementProposal] = []

        logger.info(f"RecursiveImprover initialized, records={len(self._strategy_records)}")

    # ─── 策略记录 ─────────────────────────────────────

    def record_strategy(self, strategy_type: str, strategy_content: str,
                       task: str, success: bool, quality_score: float):
        """记录一次策略使用（推理时由 chat_strategies 等调用方喂数据）"""
        record = StrategyRecord(
            strategy_type=strategy_type,
            strategy_content=strategy_content,
            task=task,
            success=success,
            quality_score=quality_score,
        )
        self._strategy_records.append(record)
        self._save_records()

    # ─── 策略分析与改进 ───────────────────────────────

    def analyze_and_improve(self) -> List[ImprovementProposal]:
        """
        分析历史策略数据，生成改进提案。
        在睡眠时调用（sleep_engine Phase 5）。
        """
        proposals = []

        # 1. Prompt 改进
        prompt_proposals = self._analyze_prompt_strategies()
        proposals.extend(prompt_proposals)

        # 2. 工具选择改进
        tool_proposals = self._analyze_tool_strategies()
        proposals.extend(tool_proposals)

        # 3. 反思模板改进
        reflection_proposals = self._analyze_reflection_strategies()
        proposals.extend(reflection_proposals)

        # 4. 保留高质量改进
        for p in proposals:
            if p.confidence >= 0.7:
                self._apply_improvement(p)
                self._improvements.append(p)

        logger.info(f"Generated {len(proposals)} improvement proposals, "
                    f"{len([p for p in proposals if p.confidence >= 0.7])} applied")
        return proposals

    def _analyze_prompt_strategies(self) -> List[ImprovementProposal]:
        """分析 prompt 策略，找出最有效的模式"""
        proposals = []
        prompt_records = [r for r in self._strategy_records if r.strategy_type == "prompt"]

        if len(prompt_records) < 10:
            return proposals

        # 按成功率分组
        high_quality = [r for r in prompt_records if r.quality_score >= 0.8]
        low_quality = [r for r in prompt_records if r.quality_score < 0.4]

        if len(high_quality) >= 3 and len(low_quality) >= 3:
            # 找出高分 prompt 的共同特征
            high_patterns = self._extract_patterns([r.strategy_content for r in high_quality])
            low_patterns = self._extract_patterns([r.strategy_content for r in low_quality])

            # 高分独有模式（过滤超长整句/噪声：中文无空格分词时整句会成为单个"词"）
            unique_patterns = {
                p for p in (high_patterns - low_patterns) if len(p) <= 80
            }
            if unique_patterns:
                proposals.append(ImprovementProposal(
                    proposal_type="prompt",
                    description=f"发现 {len(unique_patterns)} 个高效 prompt 模式",
                    old_value=self._best_strategies.get("system_prompt", ""),
                    new_value=f"建议加入: {', '.join(list(unique_patterns)[:3])}",
                    confidence=min(len(high_quality) / 10, 1.0),
                    evidence_count=len(high_quality),
                ))

        return proposals

    def _analyze_tool_strategies(self) -> List[ImprovementProposal]:
        """分析工具使用策略"""
        proposals = []
        tool_records = [r for r in self._strategy_records if r.strategy_type == "tool_choice"]

        if len(tool_records) < 5:
            return proposals

        # 统计每种工具的成功率
        tool_stats: Dict[str, Dict] = {}
        for r in tool_records:
            tool = r.strategy_content
            if tool not in tool_stats:
                tool_stats[tool] = {"success": 0, "total": 0, "quality_sum": 0}
            tool_stats[tool]["total"] += 1
            if r.success:
                tool_stats[tool]["success"] += 1
            tool_stats[tool]["quality_sum"] += r.quality_score

        # 找出高效和低效工具
        for tool, stats in tool_stats.items():
            if stats["total"] >= 3:
                success_rate = stats["success"] / stats["total"]
                avg_quality = stats["quality_sum"] / stats["total"]

                if success_rate < 0.3:
                    proposals.append(ImprovementProposal(
                        proposal_type="tool",
                        description=f"工具 {tool} 成功率仅 {success_rate:.0%}",
                        old_value=f"当前使用频率: {stats['total']}次",
                        new_value="建议降低优先级或寻找替代工具",
                        confidence=0.8,
                        evidence_count=stats["total"],
                    ))

        return proposals

    def _analyze_reflection_strategies(self) -> List[ImprovementProposal]:
        """分析反思策略是否有效"""
        proposals = []
        reflection_records = [r for r in self._strategy_records if r.strategy_type == "reflection"]

        if len(reflection_records) < 5:
            return proposals

        # 检查反思后的行为是否改善
        # 简化实现：检查反思后的任务成功率是否更高
        success_after_reflection = [r for r in reflection_records if r.success]
        if len(success_after_reflection) < len(reflection_records) * 0.5:
            proposals.append(ImprovementProposal(
                proposal_type="reflection",
                description="反思后的行为改善率不足 50%",
                old_value="当前反思模板",
                new_value="建议：增加具体行动步骤，减少泛泛而谈",
                confidence=0.7,
                evidence_count=len(reflection_records),
            ))

        return proposals

    def _extract_patterns(self, texts: List[str]) -> set:
        """提取文本中的共同模式"""
        patterns = set()
        for text in texts:
            # 提取关键词（中文无空格分词时整句成为一个"词"，由调用方过滤超长项）
            words = text.split()
            for word in words:
                if len(word) > 3:
                    patterns.add(word)
        return patterns

    def _apply_improvement(self, proposal: ImprovementProposal):
        """应用改进提案"""
        if proposal.proposal_type == "prompt":
            self._best_strategies["system_prompt"] = proposal.new_value
        elif proposal.proposal_type == "reflection":
            self._best_strategies["reflection_template"] = proposal.new_value

        self._save_best_strategies()
        logger.info(f"Applied improvement: {proposal.proposal_type} - {proposal.description}")

    # ─── 持久化 ───────────────────────────────────────

    def _load_records(self):
        """加载策略记录"""
        path = os.path.join(self.data_dir, "strategy_records.jsonl")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        self._strategy_records.append(StrategyRecord(**data))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass

    def _ensure_data_dir(self):
        """延迟创建数据目录（只在首次写入时创建）"""
        if not self._data_dir_ready:
            os.makedirs(self.data_dir, exist_ok=True)
            self._data_dir_ready = True

    def _save_records(self):
        """保存策略记录"""
        self._ensure_data_dir()
        path = os.path.join(self.data_dir, "strategy_records.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in self._strategy_records[-1000:]:  # 只保留最近 1000 条
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    def _load_best_strategies(self):
        """加载最优策略"""
        path = os.path.join(self.data_dir, "best_strategies.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self._best_strategies.update(json.load(f))

    def _save_best_strategies(self):
        """保存最优策略"""
        self._ensure_data_dir()
        path = os.path.join(self.data_dir, "best_strategies.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._best_strategies, f, ensure_ascii=False, indent=2)

    # ─── 状态查询 ─────────────────────────────────────

    def get_status(self) -> dict:
        """获取改进系统状态"""
        return {
            "total_records": len(self._strategy_records),
            "total_improvements": len(self._improvements),
            "best_strategies": self._best_strategies,
            "recent_improvements": [
                {
                    "type": p.proposal_type,
                    "description": p.description,
                    "confidence": p.confidence,
                }
                for p in self._improvements[-5:]
            ],
        }


# B4 修复：全局单例，避免每次新建实例丢失历史记录
_recursive_improver: Optional[RecursiveImprover] = None


def get_recursive_improver() -> RecursiveImprover:
    global _recursive_improver
    if _recursive_improver is None:
        _recursive_improver = RecursiveImprover()
    return _recursive_improver
