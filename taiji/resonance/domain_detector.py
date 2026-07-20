"""域检测器 (DomainDetector) - 决策点 1 实现

人脑启发：前额叶预处理感知输入，决定路由到哪个皮层区域。
态极实现：规则启发式粗筛 + general 神经元细调（待训练）。

决策方案 A+C 混合：
- 规则启发式作为 fallback（快速、零参数、可解释）
- general 神经元输出路由分数（未来训练后启用，能学习进化）
- 当 general 神经元未训练时，自动回退到规则
"""

from __future__ import annotations

import logging
import re
from typing import Optional, Tuple

logger = logging.getLogger("Taiji.DomainDetector")


# 五个域的标识
DOMAINS = ("zh", "en", "code", "math", "general")


class RuleBasedDetector:
    """规则启发式域检测器（方案 A）。

    快速、零参数、可解释。作为 fallback 和 general 神经元未训练时的默认路由。

    规则优先级（按检测难度）：
    1. code: 包含代码块/函数定义/语法关键字
    2. math: 包含数学公式符号/LaTeX
    3. zh: 中文字符占比 > 30%
    4. en: 英文字母为主
    5. general: 无法判断
    """

    # 代码块特征
    CODE_PATTERNS = [
        r"```",
        r"\bdef\s+\w+\s*\(",
        r"\bclass\s+\w+\s*[\(:]",
        r"\bimport\s+\w+",
        r"\bfrom\s+\w+\s+import\b",
        r"\b(public|private|protected)\s+",
        r"\bfunction\s+\w+\s*\(",
        r"\b(var|let|const)\s+\w+",
        r"\bif\s*\(.*\)\s*\{",
        r"\bfor\s*\(.*\)\s*\{",
        r"^\s*#\s*(include|include_once|require)",
        r"\bprint\s*\(",
        r"\bconsole\.\w+\s*\(",
        r";\s*$",  # 语句结尾分号
    ]

    # 数学公式特征
    MATH_PATTERNS = [
        r"\\frac\{",
        r"\\sum_",
        r"\\int_",
        r"\\sqrt\{",
        r"\\alpha|\\beta|\\gamma|\\theta|\\lambda|\\mu|\\sigma|\\omega",
        r"\$\$.*\$\$",
        r"\$[^$]+\$",
        r"\b[∫∑∏√≈≤≥≠±×÷∞]\b",
        r"\b\d+\s*[+\-*/]\s*\d+\s*=",
        r"\bx\^\d+\b",
        r"\bax\^2\s*[+\-]\s*bx\s*[+\-]\s*c\b",
    ]

    # 中文 Unicode 范围
    CJK_PATTERN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")

    def __init__(self):
        # 预编译正则
        self._code_re = [re.compile(p, re.MULTILINE) for p in self.CODE_PATTERNS]
        self._math_re = [re.compile(p, re.MULTILINE) for p in self.MATH_PATTERNS]

    def detect(self, text: str) -> Tuple[str, float]:
        """检测输入文本的域。

        Args:
            text: 输入文本

        Returns:
            (domain, confidence) 元组
            domain ∈ {"zh", "en", "code", "math", "general"}
            confidence ∈ [0, 1]，1=非常确定
        """
        if not text or not text.strip():
            return "general", 0.5

        text_lower = text.lower()
        scores = {d: 0.0 for d in DOMAINS}

        # 1. 检测 code
        code_matches = sum(1 for r in self._code_re if r.search(text))
        if code_matches >= 2:
            scores["code"] = 0.9
        elif code_matches == 1:
            scores["code"] = 0.6

        # 2. 检测 math
        math_matches = sum(1 for r in self._math_re if r.search(text))
        if math_matches >= 2:
            scores["math"] = 0.9
        elif math_matches == 1:
            scores["math"] = 0.6

        # 3. 检测 zh vs en
        cjk_chars = len(self.CJK_PATTERN.findall(text))
        total_chars = len(text)
        if total_chars > 0:
            cjk_ratio = cjk_chars / total_chars
            if cjk_ratio > 0.3:
                scores["zh"] = min(0.95, 0.5 + cjk_ratio)
            elif cjk_ratio < 0.05:
                # 中文字符极少，检查是否以英文为主
                alpha_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
                if alpha_chars / total_chars > 0.4:
                    scores["en"] = 0.7

        # 4. 选最高分
        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]

        # 如果最高分 < 0.5，归为 general
        if best_score < 0.5:
            return "general", 0.5

        return best_domain, best_score

    def detect_multi(self, text: str) -> dict:
        """返回所有域的置信度分数。"""
        if not text or not text.strip():
            return {d: 0.2 for d in DOMAINS}

        scores = {d: 0.0 for d in DOMAINS}

        # 1. code
        code_matches = sum(1 for r in self._code_re if r.search(text))
        if code_matches >= 2:
            scores["code"] = 0.9
        elif code_matches == 1:
            scores["code"] = 0.6

        # 2. math
        math_matches = sum(1 for r in self._math_re if r.search(text))
        if math_matches >= 2:
            scores["math"] = 0.9
        elif math_matches == 1:
            scores["math"] = 0.6

        # 3. zh vs en
        cjk_chars = len(self.CJK_PATTERN.findall(text))
        total_chars = len(text)
        if total_chars > 0:
            cjk_ratio = cjk_chars / total_chars
            if cjk_ratio > 0.3:
                scores["zh"] = min(0.95, 0.5 + cjk_ratio)
            elif cjk_ratio < 0.05:
                alpha_chars = sum(1 for c in text if c.isalpha() and ord(c) < 128)
                if alpha_chars / total_chars > 0.4:
                    scores["en"] = 0.7

        # 归一化
        total = sum(scores.values())
        if total > 0:
            scores = {k: v / total for k, v in scores.items()}

        return scores


class DomainDetector:
    """域检测器（方案 A+C 混合）。

    主路由：general 神经元输出路由分数（如果可用）
    Fallback：规则启发式（RuleBasedDetector）

    使用方式：
        detector = DomainDetector()

        # 纯规则模式
        domain, conf = detector.detect("def hello(): pass")
        # -> ("code", 0.9)

        # 启用 general 路由（需要训练后的 general 神经元）
        detector.set_general_neuron(cortex.neurons["general"])
        domain, conf = detector.detect("def hello(): pass")
        # -> general 神经元的判断 + 规则 fallback
    """

    def __init__(self):
        self.rule_detector = RuleBasedDetector()
        # general 神经元引用（可选，未来训练后启用）
        self._general_neuron = None
        self._general_router_enabled = False

    def set_general_neuron(self, neuron) -> None:
        """注入 general 神经元用于细调路由。"""
        self._general_neuron = neuron
        # 检查神经元是否已训练（有 field_vector 输出）
        # 简单启发式：如果 fingerprint 已冻结，视为已训练
        if hasattr(neuron, "fingerprint") and neuron.fingerprint.norm() > 1e-6:
            self._general_router_enabled = True
            logger.info("DomainDetector: general 神经元路由已启用")
        else:
            logger.info("DomainDetector: general 神经元未训练，使用规则路由")

    def detect(self, text: str) -> Tuple[str, float]:
        """检测输入文本的域。

        策略：
        1. 如果 general 神经元路由启用，使用它
        2. 否则用规则启发式
        3. 两者都失败时返回 "general"

        Returns:
            (domain, confidence)
        """
        # 规则结果作为基础
        rule_domain, rule_conf = self.rule_detector.detect(text)

        # 如果 general 路由未启用，直接返回规则结果
        if not self._general_router_enabled or self._general_neuron is None:
            return rule_domain, rule_conf

        # general 路由（待实现：需要 general 神经元训练出路由分数输出）
        # 目前先返回规则结果，待训练管线支持后扩展
        # TODO: 当 general 神经元能输出 domain_routing_scores 时启用
        return rule_domain, rule_conf

    def detect_multi(self, text: str) -> dict:
        """返回所有域的置信度分数。"""
        return self.rule_detector.detect_multi(text)

    def get_status(self) -> dict:
        return {
            "general_router_enabled": self._general_router_enabled,
            "has_general_neuron": self._general_neuron is not None,
            "fallback": "rule_based",
        }


# 全局单例（可选使用）
_global_detector: Optional[DomainDetector] = None


def get_detector() -> DomainDetector:
    """获取全局 DomainDetector 实例。"""
    global _global_detector
    if _global_detector is None:
        _global_detector = DomainDetector()
    return _global_detector
