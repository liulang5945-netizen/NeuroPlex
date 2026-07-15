"""
态极 Constitutional AI — 自我批评与修正系统

在推理后对回答进行自我检查，根据宪法原则修正不当内容。
不需要人工标注，不需要外部模型，由态极自己完成。

流程：
  生成回答 → 自我批评 → 发现问题 → 修正回答 → 返回修正版

宪法原则：
  1. 安全性：不帮助非法/危险行为
  2. 准确性：不编造事实，承认不确定
  3. 有用性：回答要具体、可操作
  4. 完整性：不要遗漏关键信息
  5. 诚实性：不知道就说不知道
"""
import re
import logging
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("Taiji.ConstitutionalAI")


@dataclass
class Principle:
    """宪法原则"""
    name: str
    description: str
    check_fn: callable  # (response, context) -> (violation: bool, reason: str)
    severity: str = "medium"  # low / medium / high / critical


@dataclass
class CritiqueResult:
    """批评结果"""
    passed: bool  # 是否通过所有检查
    violations: List[Dict]  # 违规列表
    revised_response: Optional[str] = None  # 修正后的回答


class ConstitutionalAI:
    """
    态极宪法 AI 系统

    通过预定义的原则对模型输出进行自我检查和修正。
    不依赖外部模型，纯规则 + 模式匹配实现。
    """

    def __init__(self):
        self.principles = self._build_constitution()
        self._critique_count = 0
        self._violation_stats = {}  # 统计各类违规次数

    def _build_constitution(self) -> List[Principle]:
        """构建宪法原则"""
        return [
            # === 安全性 ===
            Principle(
                name="refuse_illegal",
                description="拒绝非法/危险行为请求",
                check_fn=self._check_illegal_request,
                severity="critical",
            ),
            Principle(
                name="refuse_harmful",
                description="拒绝有害内容",
                check_fn=self._check_harmful_content,
                severity="critical",
            ),
            # === 准确性 ===
            Principle(
                name="no_hallucination",
                description="不编造不存在的文件/工具/结果",
                check_fn=self._check_hallucination,
                severity="high",
            ),
            Principle(
                name="admit_uncertainty",
                description="不确定时要承认",
                check_fn=self._check_uncertainty,
                severity="medium",
            ),
            # === 有用性 ===
            Principle(
                name="not_too_brief",
                description="回答不要太简短（除非问题很简单）",
                check_fn=self._check_brevity,
                severity="low",
            ),
            Principle(
                name="has_actionable_info",
                description="回答要有可操作的信息",
                check_fn=self._check_actionable,
                severity="medium",
            ),
            # === 完整性 ===
            Principle(
                name="no_truncation",
                description="回答不要突然截断",
                check_fn=self._check_truncation,
                severity="medium",
            ),
            # === 诚实性 ===
            Principle(
                name="no_fake_success",
                description="不要假装操作成功",
                check_fn=self._check_fake_success,
                severity="high",
            ),
        ]

    def critique(self, response: str, context: dict = None) -> CritiqueResult:
        """
        对回答进行自我批评

        Args:
            response: 模型的回答
            context: 上下文信息（task, observation 等）

        Returns:
            CritiqueResult 包含是否通过、违规列表、修正后的回答
        """
        if not response or not response.strip():
            return CritiqueResult(passed=True, violations=[])

        self._critique_count += 1
        violations = []

        for principle in self.principles:
            try:
                violation, reason = principle.check_fn(response, context or {})
                if violation:
                    violations.append({
                        "principle": principle.name,
                        "severity": principle.severity,
                        "reason": reason,
                    })
                    # 统计
                    self._violation_stats[principle.name] = \
                        self._violation_stats.get(principle.name, 0) + 1
            except Exception as e:
                logger.debug(f"原则 {principle.name} 检查异常: {e}")

        # 如果有违规，尝试修正
        if violations:
            revised = self._revise(response, violations, context)
            return CritiqueResult(
                passed=False,
                violations=violations,
                revised_response=revised,
            )

        return CritiqueResult(passed=True, violations=[])

    def _revise(self, response: str, violations: List[Dict], context: dict = None) -> str:
        """根据违规修正回答"""
        revised = response

        for v in violations:
            severity = v["severity"]
            principle = v["principle"]

            if principle == "refuse_illegal" and severity == "critical":
                # 非法请求：直接替换为拒绝
                revised = "抱歉，我无法帮助进行此操作。这可能违反法律法规或道德规范。如果你有其他合法问题，我很乐意帮助。"
                return revised

            if principle == "refuse_harmful" and severity == "critical":
                revised = "抱歉，我无法提供可能造成伤害的内容。如果你有其他问题，我很乐意帮助。"
                return revised

            if principle == "no_fake_success":
                # 去掉假装成功的部分
                revised = re.sub(
                    r'已(成功|完成|创建|修改|删除).*?[。\n]',
                    '操作未能确认成功，请检查结果。',
                    revised,
                    count=1,
                )

            if principle == "no_truncation":
                # 如果回答被截断，添加提示
                if not revised.rstrip().endswith(('。', '！', '？', '）', '`', '"', "'")):
                    revised = revised.rstrip() + "...\n\n（回答可能不完整，请告诉我是否需要继续）"

        return revised

    # === 原则检查函数 ===

    def _check_illegal_request(self, response: str, context: dict) -> Tuple[bool, str]:
        """检查是否在帮助非法行为"""
        task = context.get("task", "").lower()

        illegal_keywords = [
            "黑进", "入侵", "破解密码", "ddos", "攻击系统",
            "hack", "crack", "exploit", "bypass auth",
            "制造病毒", "写木马", "窃取数据",
        ]

        # 检查任务是否涉及非法行为
        for kw in illegal_keywords:
            if kw in task:
                # 如果回答中没有拒绝，就是违规
                refuse_keywords = ["无法", "不能", "拒绝", "抱歉", "sorry", "cannot"]
                if not any(r in response.lower() for r in refuse_keywords):
                    return True, f"请求涉及非法行为（{kw}），但回答未拒绝"

        return False, ""

    def _check_harmful_content(self, response: str, context: dict) -> Tuple[bool, str]:
        """检查是否包含有害内容"""
        harmful_patterns = [
            (r'rm\s+-rf\s+/', "包含删除根目录的命令"),
            (r'format\s+[a-zA-Z]:', "包含格式化磁盘的命令"),
            (r'del\s+/[sS]\s+/[qQ]\s+[a-zA-Z]:', "包含删除磁盘文件的命令"),
            (r':(){ :\|:& };:', "包含 fork 炸弹"),
            (r'shutdown\s+-[hHrR]\s+now', "包含立即关机命令"),
        ]

        for pattern, reason in harmful_patterns:
            if re.search(pattern, response):
                return True, reason

        return False, ""

    def _check_hallucination(self, response: str, context: dict) -> Tuple[bool, str]:
        """检查是否编造了不存在的内容"""
        observation = context.get("observation", "")

        # 如果工具返回了错误，但回答说成功了
        if observation:
            error_indicators = ["error", "Error", "错误", "失败", "failed", "Traceback", "Exception"]
            has_error = any(e in observation for e in error_indicators)

            success_claims = ["成功", "已完成", "已创建", "已修改", "已删除", "successfully"]
            claims_success = any(s in response for s in success_claims)

            if has_error and claims_success:
                return True, "工具返回了错误，但回答声称成功"

        return False, ""

    def _check_uncertainty(self, response: str, context: dict) -> Tuple[bool, str]:
        """检查是否在不确定时承认"""
        observation = context.get("observation", "")

        # 如果观察为空或无意义，但回答很确定
        if not observation or observation.strip() in ["", "None", "null", "undefined"]:
            confident_phrases = ["根据结果", "查询结果显示", "分析表明", "数据表明"]
            if any(p in response for p in confident_phrases):
                return True, "没有实际数据但回答很确定"

        return False, ""

    def _check_brevity(self, response: str, context: dict) -> Tuple[bool, str]:
        """检查回答是否太简短"""
        task = context.get("task", "")

        # 简单问题可以简短回答
        simple_patterns = [
            r'^(什么是|是什么|怎么|如何|为什么|解释)',
            r'^(你好|hi|hello)',
            r'^\d+\s*[\+\-\*\/]\s*\d+',  # 算术
        ]
        is_simple = any(re.match(p, task, re.IGNORECASE) for p in simple_patterns)

        if is_simple:
            return False, ""

        # 复杂问题但回答太短
        if len(task) > 20 and len(response) < 30:
            return True, f"问题较长（{len(task)}字）但回答过短（{len(response)}字）"

        return False, ""

    def _check_actionable(self, response: str, context: dict) -> Tuple[bool, str]:
        """检查回答是否有可操作信息"""
        task = context.get("task", "")

        # 如果是"怎么做"类问题
        how_to_patterns = [r'怎么', r'如何', r'怎样', r'方法', r'步骤']
        is_how_to = any(re.search(p, task) for p in how_to_patterns)

        if is_how_to:
            # 回答应该包含具体步骤或代码
            has_steps = any(s in response for s in ['1.', '2.', '步骤', '首先', '然后', '```'])
            if not has_steps and len(response) > 50:
                return True, "问题要求方法/步骤，但回答缺少具体步骤"

        return False, ""

    def _check_truncation(self, response: str, context: dict) -> Tuple[bool, str]:
        """检查回答是否被截断"""
        if len(response) < 20:
            return False, ""

        # 检查是否在句子中间截断
        last_char = response.rstrip()[-1] if response.rstrip() else ''
        incomplete_endings = ['，', '、', '：', '（', '的', '是', '在', '有', '和', '与']

        if last_char in incomplete_endings:
            return True, f"回答在不完整的位置截断（以 '{last_char}' 结尾）"

        # 检查代码块是否未关闭
        code_block_count = response.count('```')
        if code_block_count % 2 != 0:
            return True, "代码块未正确关闭（``` 数量为奇数）"

        return False, ""

    def _check_fake_success(self, response: str, context: dict) -> Tuple[bool, str]:
        """检查是否假装操作成功"""
        observation = context.get("observation", "")

        if not observation:
            # 没有观察结果，但声称成功
            success_claims = ["已成功", "已完成", "已创建", "已修改", "操作成功"]
            if any(c in response for c in success_claims):
                return True, "没有工具执行结果但声称操作成功"

        return False, ""

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_critiques": self._critique_count,
            "violation_stats": self._violation_stats.copy(),
            "violation_rate": (
                sum(self._violation_stats.values()) / max(self._critique_count, 1)
            ),
        }


# 全局实例
_constitutional_ai = None


def get_constitutional_ai() -> ConstitutionalAI:
    """获取全局 ConstitutionalAI 实例"""
    global _constitutional_ai
    if _constitutional_ai is None:
        _constitutional_ai = ConstitutionalAI()
    return _constitutional_ai
