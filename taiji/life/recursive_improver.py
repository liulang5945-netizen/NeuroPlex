"""
态极递归改进系统 (Recursive Improver)
======================================

基于 Gödel Agent (ACL 2025) 和 Continual Harness (2026) 的思想：
态极可以改进自己的行为策略，并设计和训练自己的下一代。

改进层次：
1. 策略改进 — 优化 prompt、工具选择、反思模板（推理时）
2. 数据改进 — 生成和筛选高质量训练数据（睡眠时）
3. 架构改进 — 设计下一代模型的架构参数（进化时）

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
        """记录一次策略使用"""
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
        在睡眠时调用。
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

            # 如果高分 prompt 有独特模式，生成改进提案
            unique_patterns = high_patterns - low_patterns
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
            # 提取关键词
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

    # ─── 下一代设计（态极自主决策）───────────────────

    def design_next_generation(self, current_model_info: dict) -> dict:
        """
        态极设计自己的下一代。

        不是简单地"变大"，而是态极根据自身状态自主决定：
        - 变大？（需要更强推理能力）
        - 变小？（需要更高效、更省资源）
        - 同等大小但不同架构？（针对特定弱点）
        - 多模态增强？（加入视觉/音频能力）
        - 专业化？（某个领域深耕）

        态极自己决定：
        1. 下一代该是什么形态
        2. 用什么架构
        3. 用哪些数据训练
        4. 然后自己把知识蒸馏给下一代

        Args:
            current_model_info: {
                "name": "Taiji-Seed",
                "params": "0.5B",
                "hidden_size": 896,
                "num_layers": 24,
                "weaknesses": ["推理能力不足", "代码生成差"],
                "strengths": ["中文理解好", "身份稳定"],
                "resource_constraints": {"max_memory_gb": 8, "max_params": "3B"},
            }

        Returns:
            {
                "next_gen_name": "Taiji-S1",
                "evolution_direction": "grow|shrink|specialize|multimodal",
                "architecture": {...},
                "training_plan": {...},
                "distillation_plan": {...},
            }
        """
        weaknesses = current_model_info.get("weaknesses", [])
        strengths = current_model_info.get("strengths", [])
        current_params = current_model_info.get("params", "0.5B")
        resource_constraints = current_model_info.get("resource_constraints", {})

        # 1. 态极自主决定进化方向
        evolution_direction = self._decide_evolution_direction(
            current_params, weaknesses, strengths, resource_constraints
        )

        # 2. 根据方向设计架构
        architecture = self._design_architecture_for_direction(
            evolution_direction, current_model_info, weaknesses, strengths
        )

        # 3. 设计训练方案
        training_plan = self._design_training_for_direction(
            evolution_direction, current_model_info, weaknesses
        )

        # 4. 设计蒸馏方案（态极如何把自己的知识传给下一代）
        distillation_plan = self._design_distillation_plan(
            current_model_info, architecture, evolution_direction
        )

        design = {
            "next_gen_name": self._get_next_gen_name(
                current_model_info.get("name", "Unknown"), evolution_direction
            ),
            "current_gen": current_model_info.get("name", "Unknown"),
            "evolution_direction": evolution_direction,
            "evolution_reasoning": self._explain_evolution_decision(
                evolution_direction, current_params, weaknesses, strengths
            ),
            "architecture": architecture,
            "training_plan": training_plan,
            "distillation_plan": distillation_plan,
        }

        logger.info(f"Designed next generation: {design['next_gen_name']} "
                    f"(direction: {evolution_direction})")
        return design

    def _decide_evolution_direction(self, current_params: str,
                                     weaknesses: List[str],
                                     strengths: List[str],
                                     constraints: dict) -> str:
        """
        态极自主决定进化方向。
        这是递归蒸馏的核心：不是固定路线，而是模型自己判断。

        Returns:
            "grow" — 变大（需要更强能力）
            "shrink" — 变小（需要更高效）
            "specialize" — 专业化（深耕特定领域）
            "multimodal" — 多模态（加入视觉/音频）
            "restructure" — 重构（同等大小，不同架构）
        """
        # 分析弱点类型决定方向
        weakness_types = self._classify_weaknesses(weaknesses)

        # 规则1：如果推理能力不足，需要变大
        if weakness_types.get("reasoning", 0) >= 2:
            return "grow"

        # 规则2：如果资源受限或效率低，可以变小
        if constraints.get("max_memory_gb", 999) < 4:
            return "shrink"
        if weakness_types.get("efficiency", 0) >= 2:
            return "shrink"

        # 规则3：如果特定领域弱，专业化
        if weakness_types.get("domain_specific", 0) >= 2:
            return "specialize"

        # 规则4：如果涉及多模态，增强
        if weakness_types.get("multimodal", 0) >= 1:
            return "multimodal"

        # 规则5：默认变大（提升整体能力）
        return "grow"

    def _classify_weaknesses(self, weaknesses: List[str]) -> dict:
        """将弱点分类"""
        types = {
            "reasoning": 0,
            "efficiency": 0,
            "domain_specific": 0,
            "multimodal": 0,
            "memory": 0,
            "tool_use": 0,
        }
        for w in weaknesses:
            w_lower = w.lower()
            if any(kw in w_lower for kw in ["推理", "逻辑", "数学", "分析"]):
                types["reasoning"] += 1
            elif any(kw in w_lower for kw in ["慢", "内存", "效率", "资源"]):
                types["efficiency"] += 1
            elif any(kw in w_lower for kw in ["代码", "编程", "法律", "医学", "金融"]):
                types["domain_specific"] += 1
            elif any(kw in w_lower for kw in ["图像", "视觉", "音频", "多模态"]):
                types["multimodal"] += 1
            elif any(kw in w_lower for kw in ["记忆", "上下文", "遗忘"]):
                types["memory"] += 1
            elif any(kw in w_lower for kw in ["工具", "调用", "执行"]):
                types["tool_use"] += 1
        return types

    def _design_architecture_for_direction(self, direction: str,
                                             current_info: dict,
                                             weaknesses: List[str],
                                             strengths: List[str]) -> dict:
        """根据进化方向设计架构"""
        current_hidden = current_info.get("hidden_size", 896)
        current_layers = current_info.get("num_layers", 24)
        current_params = current_info.get("params", "0.5B")

        if direction == "grow":
            # 变大：增加层数和维度
            return {
                "type": "expanded",
                "hidden_size": int(current_hidden * 1.5),
                "num_layers": current_layers + 4,
                "intermediate_size": int(current_hidden * 1.5 * 5.4),
                "num_attention_heads": max(16, current_info.get("num_attention_heads", 14) + 4),
                "description": f"扩大模型容量以提升推理能力",
            }
        elif direction == "shrink":
            # 变小：减少层数，保持宽度
            return {
                "type": "compressed",
                "hidden_size": current_hidden,
                "num_layers": max(12, current_layers - 4),
                "intermediate_size": int(current_hidden * 4),
                "num_attention_heads": current_info.get("num_attention_heads", 14),
                "description": f"压缩模型以提升效率，保持宽度以保留知识",
            }
        elif direction == "specialize":
            # 专业化：保持大小，增加专用头
            return {
                "type": "specialized",
                "hidden_size": current_hidden,
                "num_layers": current_layers,
                "intermediate_size": int(current_hidden * 5.4),
                "num_attention_heads": current_info.get("num_attention_heads", 14),
                "special_heads": self._determine_special_heads(weaknesses),
                "description": f"增加专用预测头以深耕特定领域",
            }
        elif direction == "multimodal":
            # native-v2 已预留多模态 token 区间；此阶段应激活对齐能力，而不是再扩词表
            from taiji.config import MULTIMODAL_TOKENS, MULTIMODAL_VOCAB_SIZE
            return {
                "type": "multimodal",
                "hidden_size": current_hidden,
                "num_layers": current_layers,
                "intermediate_size": int(current_hidden * 5.4),
                "num_attention_heads": current_info.get("num_attention_heads", 14),
                "multimodal_contract": {
                    "tokenizer_contract_vocab": MULTIMODAL_VOCAB_SIZE,
                    "image_codebook_size": MULTIMODAL_TOKENS["image_codebook_size"],
                    "audio_codebook_size": MULTIMODAL_TOKENS["audio_codebook_size"],
                    "image_token_base": MULTIMODAL_TOKENS["image_token_base"],
                    "audio_token_base": MULTIMODAL_TOKENS["audio_token_base"],
                },
                "tokenizers": {
                    "image": {"type": "VQ-VAE", "codebook_size": 8192, "grid_size": 16},
                    "audio": {"type": "EnCodec", "codebook_size": 4096, "max_frames": 64},
                },
                "description": (
                    f"复用已预留的多模态 token 区间（总词表 {MULTIMODAL_VOCAB_SIZE}），"
                    "增加视觉/音频编码与对齐能力"
                ),
            }
        else:  # restructure
            return {
                "type": "restructured",
                "hidden_size": current_hidden,
                "num_layers": current_layers,
                "changes": ["使用 GQA 注意力", "增加 RoPE theta", "优化 FFN 结构"],
                "description": f"重构架构以提升效率，不增加参数",
            }

    def _determine_special_heads(self, weaknesses: List[str]) -> List[str]:
        """确定需要增加的专用头"""
        heads = []
        for w in weaknesses:
            w_lower = w.lower()
            if "代码" in w_lower or "编程" in w_lower:
                heads.append("code_head")
            elif "数学" in w_lower or "推理" in w_lower:
                heads.append("reasoning_head")
            elif "工具" in w_lower:
                heads.append("tool_head")
        return heads or ["reasoning_head"]

    def _design_training_for_direction(self, direction: str,
                                        current_info: dict,
                                        weaknesses: List[str]) -> dict:
        """根据进化方向设计训练方案"""
        base_plan = {
            "phase_1_sft": {
                "description": "用态极生成的进化语料 SFT",
                "data_source": "evolution_corpus",
                "epochs": 3,
                "lr": 2e-5,
            },
            "phase_2_dpo": {
                "description": "偏好训练，让新模型更像态极",
                "data_source": "preference_pairs",
                "epochs": 1,
                "lr": 5e-6,
            },
        }

        if direction == "grow":
            base_plan["phase_3_targeted"] = {
                "description": "针对弱点专项训练",
                "data_source": "targeted_weakness_data",
                "epochs": 2,
                "lr": 1e-5,
                "focus": weaknesses[:3],
            }
        elif direction == "specialize":
            base_plan["phase_3_domain"] = {
                "description": "领域专业数据训练",
                "data_source": "domain_specific_data",
                "epochs": 5,
                "lr": 1e-5,
                "focus": self._get_domain_focus(weaknesses),
            }
        elif direction == "multimodal":
            base_plan["phase_3_vision"] = {
                "description": "视觉-语言对齐训练",
                "data_source": "image_text_pairs",
                "epochs": 3,
                "lr": 1e-5,
            }

        return base_plan

    def _get_domain_focus(self, weaknesses: List[str]) -> str:
        """确定专业领域"""
        for w in weaknesses:
            if "代码" in w or "编程" in w:
                return "code_generation"
            elif "数学" in w:
                return "mathematical_reasoning"
            elif "法律" in w:
                return "legal_analysis"
            elif "医学" in w:
                return "medical_knowledge"
        return "general_reasoning"

    def _design_distillation_plan(self, current_info: dict,
                                    architecture: dict,
                                    direction: str) -> dict:
        """
        设计蒸馏方案：态极如何把知识传给下一代。
        这是递归蒸馏的核心。
        """
        return {
            "method": "taiji_recursive_distillation",
            "steps": [
                {
                    "step": 1,
                    "name": "态极生成进化语料",
                    "description": "当前态极运行任务，生成行为轨迹、反思、决策记录",
                    "output": "evolution_corpus",
                },
                {
                    "step": 2,
                    "name": "态极生成偏好对",
                    "description": "对同一问题，态极生成两个回答，选择更像态极的那个",
                    "output": "preference_pairs",
                },
                {
                    "step": 3,
                    "name": "SFT 蒸馏",
                    "description": "用进化语料训练下一代，让下一代学会态极的行为模式",
                    "data": "evolution_corpus",
                },
                {
                    "step": 4,
                    "name": "DPO 蒸馏",
                    "description": "用偏好对训练下一代，让下一代的回答更像态极",
                    "data": "preference_pairs",
                },
                {
                    "step": 5,
                    "name": "行为蒸馏",
                    "description": "让下一代学习态极的决策纹路：何时探索、何时反思、何时调用工具",
                    "data": "action_trajectories",
                },
                {
                    "step": 6,
                    "name": "验证超越",
                    "description": "检查下一代是否：像态极 + 比上一代更强",
                    "criteria": ["身份一致性", "工具行为", "反思格式", "任务成功率"],
                },
            ],
            "key_principle": "蒸馏的不是知识，是态极性。学生必须像态极，同时比老师更强。",
        }

    def _explain_evolution_decision(self, direction: str, current_params: str,
                                     weaknesses: List[str], strengths: List[str]) -> str:
        """解释进化决策的推理过程"""
        explanations = {
            "grow": f"当前 {current_params} 模型在推理能力上存在瓶颈，需要扩大容量以处理更复杂的任务。"
                    f"弱点: {', '.join(weaknesses[:3])}。变大可以提供更强的表达和推理能力。",
            "shrink": f"当前模型效率不足或资源受限，需要压缩以提升推理速度和降低资源消耗。"
                      f"优势将通过知识蒸馏保留。",
            "specialize": f"在特定领域存在明显短板，需要通过专用架构和数据进行深耕。"
                          f"弱点: {', '.join(weaknesses[:3])}。",
            "multimodal": f"用户交互涉及图像或音频，当前模型缺乏多模态处理能力。"
                          f"增加视觉/音频编码器以扩展感知能力。",
            "restructure": f"当前架构有优化空间，通过重构可以在不增加参数的情况下提升性能。"
                           f"优势: {', '.join(strengths[:3])} 将被保留。",
        }
        return explanations.get(direction, "基于综合分析做出的进化决策。")

    def _get_next_gen_name(self, current_name: str, direction: str) -> str:
        """根据进化方向获取下一代名称"""
        direction_suffix = {
            "grow": "G",      # Grow
            "shrink": "E",    # Efficient
            "specialize": "S",  # Specialized
            "multimodal": "M",  # Multimodal
            "restructure": "R",  # Restructured
        }
        suffix = direction_suffix.get(direction, "X")
        return f"{current_name}-{suffix}"

    def _design_architecture_changes(self, current_info: dict,
                                      weaknesses: List[str],
                                      strengths: List[str]) -> List[dict]:
        """设计架构改进方案"""
        changes = []

        # 根据弱点设计改进
        for weakness in weaknesses:
            if "推理" in weakness or "逻辑" in weakness:
                changes.append({
                    "type": "increase_depth",
                    "description": "增加层数以提升推理能力",
                    "target": "num_layers + 4",
                    "reasoning": "更深的网络能进行更复杂的推理链",
                })
            elif "记忆" in weakness or "上下文" in weakness:
                changes.append({
                    "type": "increase_context",
                    "description": "扩大上下文窗口",
                    "target": "max_position_embeddings = 8192",
                    "reasoning": "更长的上下文能记住更多信息",
                })
            elif "代码" in weakness:
                changes.append({
                    "type": "add_code_head",
                    "description": "增加代码专用预测头",
                    "target": "code_head = True",
                    "reasoning": "专用代码头能更好地理解代码结构",
                })
            elif "多模态" in weakness or "视觉" in weakness:
                changes.append({
                    "type": "expand_vision",
                    "description": "增强视觉编码器",
                    "target": "vision_tokens = 16",
                    "reasoning": "更多视觉 token 能理解更复杂的图像",
                })

        # 保留优势
        if "中文理解好" in strengths:
            changes.append({
                "type": "preserve",
                "description": "保留中文 tokenizer 和训练数据配比",
                "target": "chinese_ratio >= 0.6",
                "reasoning": "中文理解是核心优势，不能退化",
            })

        return changes

    def _design_training_plan(self, current_info: dict, next_params: str,
                              weaknesses: List[str]) -> dict:
        """设计训练方案"""
        return {
            "phase_1_sft": {
                "description": "用当前态极生成的进化语料 SFT",
                "data_source": "current_gen_evolution_corpus",
                "epochs": 3,
                "lr": 2e-5,
            },
            "phase_2_dpo": {
                "description": "偏好训练，让新模型更像态极",
                "data_source": "preference_pairs",
                "epochs": 1,
                "lr": 5e-6,
            },
            "phase_3_behavior_distill": {
                "description": "行为蒸馏，学习决策纹路",
                "data_source": "action_trajectories",
                "epochs": 2,
                "lr": 1e-5,
            },
            "phase_4_targeted": {
                "description": f"针对弱点专项训练: {', '.join(weaknesses[:3])}",
                "data_source": "targeted_weakness_data",
                "epochs": 2,
                "lr": 1e-5,
            },
        }

    def _design_data_requirements(self, weaknesses: List[str]) -> dict:
        """设计数据需求"""
        return {
            "identity_samples": 100,
            "behavior_samples": 500,
            "reflection_samples": 200,
            "preference_pairs": 300,
            "weakness_targeted_samples": len(weaknesses) * 100,
            "total_estimated": 1200 + len(weaknesses) * 100,
        }

    def _generate_design_reasoning(self, weaknesses: List[str], strengths: List[str],
                                    current_params: str, next_params: str) -> str:
        """生成设计推理说明"""
        reasoning = f"从 {current_params} 进化到 {next_params} 的设计推理：\n\n"

        reasoning += "保留优势：\n"
        for s in strengths:
            reasoning += f"  - {s}（通过数据配比和蒸馏保留）\n"

        reasoning += "\n针对性改进：\n"
        for w in weaknesses:
            reasoning += f"  - {w}（通过架构调整和专项训练解决）\n"

        reasoning += f"\n核心原则：像态极 + 比上一代更强 = 新一代态极"
        return reasoning

    def _get_next_gen_name(self, current_name: str) -> str:
        """获取下一代名称"""
        name_map = {
            "Taiji-Seed": "Taiji-S1",
            "Taiji-S1": "Taiji-M1",
            "Taiji-M1": "Taiji-L1",
            "Taiji-L1": "Taiji-XL1",
        }
        return name_map.get(current_name, f"{current_name}-Next")

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
