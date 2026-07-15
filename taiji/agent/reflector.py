"""
ModelSelf 反思系统 v2
镜像神经元 — 让模型拥有自我评估和纠错能力

检测执行错误，分析原因，生成纠正方案。
v2 新增：执行前自验证、多路径探索、工具成功率统计。
"""
import os
import re
import logging
from typing import Any, Optional, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("ModelSelf.Reflector")


class ReflectionType(Enum):
    """反思类型"""
    CONFIRM = "confirm"     # 确认成功
    DETECT = "detect"       # 检测到问题
    CAUSE = "cause"         # 分析原因
    CORRECT = "correct"     # 纠正方案


@dataclass
class ReflectionResult:
    """反思结果"""
    type: ReflectionType
    message: str
    confidence: float = 0.0   # 0-1 置信度
    should_retry: bool = False
    correction_hint: Optional[str] = None

    def to_token_text(self) -> str:
        if self.type == ReflectionType.CONFIRM:
            return f"<reflect><confirm>{self.message}</confirm></reflect>"
        elif self.type == ReflectionType.DETECT:
            return f"<reflect><detect>{self.message}</detect></reflect>"
        elif self.type == ReflectionType.CAUSE:
            return f"<reflect><cause>{self.message}</cause></reflect>"
        elif self.type == ReflectionType.CORRECT:
            return f"<reflect><correct>{self.message}</correct></reflect>"
        return f"<reflect>{self.message}</reflect>"


class ReflectorSystem:
    """
    反思系统 — 错误检测、自验证与纠正

    职责:
    1. 检测工具执行结果中的错误
    2. 分析错误原因
    3. 生成纠正建议
    4. 确认成功执行
    5. 【v2】执行前自验证：预测动作成功率，低置信度时建议替代方案
    6. 【v2】多路径探索：为高风险动作生成备选方案

    与模型的交互通过特殊 token:
    - <reflect><detect>错误描述</detect></reflect> → 检测到错误
    - <reflect><cause>原因分析</cause></reflect> → 分析原因
    - <reflect><correct>纠正方案</correct></reflect> → 纠正建议
    - <reflect><confirm>成功确认</confirm></reflect> → 确认成功
    - <reflect><verify>自验证结果</verify></reflect> → 执行前验证
    """

    # 常见错误模式
    ERROR_PATTERNS = [
        # Python 错误
        (r"SyntaxError", "语法错误", "检查代码语法，注意括号、缩进、冒号"),
        (r"NameError.*'(\w+)'", "变量未定义", "变量 '{0}' 未定义，检查拼写或是否需要 import"),
        (r"TypeError", "类型错误", "检查函数参数类型是否正确"),
        (r"FileNotFoundError.*'([^']+)'", "文件不存在", "文件 '{0}' 不存在，检查路径是否正确"),
        (r"PermissionError", "权限不足", "没有文件访问权限，检查文件是否被占用"),
        (r"ModuleNotFoundError.*'(\w+)'", "模块未安装", "模块 '{0}' 未安装，使用 install_dependency 安装"),
        (r"IndentationError", "缩进错误", "检查代码缩进，确保使用一致的空格或 Tab"),
        (r"ZeroDivisionError", "除零错误", "代码中存在除以零的操作，添加零值检查"),
        (r"IndexError", "索引越界", "列表索引超出范围，检查列表长度"),
        (r"KeyError.*'(\w+)'", "键不存在", "字典中不存在键 '{0}'，检查键名或使用 .get()"),
        # 工具错误
        (r"command not found", "命令未找到", "系统命令不存在，检查命令拼写"),
        (r"ConnectionError|TimeoutError|超时", "网络错误", "网络连接失败，检查网络或 URL"),
        (r"403|401|Unauthorized", "认证失败", "API 认证失败，检查 API Key"),
        (r"404|Not Found", "资源不存在", "请求的资源不存在，检查 URL 或文件路径"),
        # 编译错误
        (r"error:|Error:|ERROR:", "执行错误", "工具执行返回错误，检查输入参数"),
    ]

    # 高风险工具（执行前需要额外验证）
    HIGH_RISK_TOOLS = {
        "run_command", "execute_code", "delete_file", "write_file",
        "install_dependency", "git_push", "send_email",
    }

    # 常见工具的典型失败模式（用于自验证）
    TOOL_ANTIPATTERNS = {
        "read_file": [
            (lambda a: not a.get("input"), "缺少文件路径参数"),
            (lambda a: a.get("input", "").startswith("~"), "路径含 ~，可能无法解析"),
        ],
        "run_command": [
            (lambda a: not a.get("input"), "缺少命令参数"),
            (lambda a: "rm -rf /" in str(a.get("input", "")), "危险的递归删除命令"),
            (lambda a: "sudo" in str(a.get("input", "")), "需要 sudo 权限，可能失败"),
        ],
        "write_file": [
            (lambda a: not a.get("input"), "缺少文件路径"),
            (lambda a: not a.get("content") and len(str(a.get("input", "")).split()) < 2, "缺少写入内容"),
        ],
        "search": [
            (lambda a: not a.get("input"), "缺少搜索关键词"),
            (lambda a: len(str(a.get("input", ""))) < 3, "搜索词太短，结果可能不精准"),
        ],
    }

    def __init__(self, save_dir: str = None):
        self.reflection_history: List[ReflectionResult] = []
        self.consecutive_errors: int = 0
        self.max_retries: int = 3
        # v2: 工具成功率统计（用于自验证置信度）
        self._tool_stats: Dict[str, Dict[str, int]] = {}  # tool -> {"success": N, "fail": N}
        self._save_path = os.path.join(save_dir, "tool_stats.json") if save_dir else None
        self._load_stats()

    def _load_stats(self):
        """从磁盘加载工具成功率统计"""
        if not self._save_path or not os.path.exists(self._save_path):
            return
        try:
            import json
            with open(self._save_path, "r", encoding="utf-8") as f:
                self._tool_stats = json.load(f)
        except Exception:
            pass

    def _save_stats(self):
        """保存工具成功率统计到磁盘"""
        if not self._save_path:
            return
        try:
            import json
            os.makedirs(os.path.dirname(self._save_path), exist_ok=True)
            with open(self._save_path, "w", encoding="utf-8") as f:
                json.dump(self._tool_stats, f, indent=2)
        except Exception:
            pass

    def evaluate_result(self, tool_name: str, result: str) -> ReflectionResult:
        """
        评估工具执行结果

        Returns:
            ReflectionResult 包含评估结果和建议
        """
        # 检查是否为错误
        error_match = self._detect_error(result)
        if error_match:
            self.consecutive_errors += 1
            self.record_outcome(tool_name, success=False)
            reflection = ReflectionResult(
                type=ReflectionType.DETECT,
                message=error_match[1],
                confidence=0.9,
                should_retry=self.consecutive_errors <= self.max_retries,
                correction_hint=error_match[2],
            )
            self.reflection_history.append(reflection)
            return reflection

        # 成功
        self.consecutive_errors = 0
        self.record_outcome(tool_name, success=True)
        reflection = ReflectionResult(
            type=ReflectionType.CONFIRM,
            message=f"{tool_name} 执行成功",
            confidence=0.8,
        )
        self.reflection_history.append(reflection)
        return reflection

    def verify_before_act(
        self,
        tool_name: str,
        action_args: dict,
        context: str = "",
    ) -> Tuple[float, List[str]]:
        """
        【v2】执行前自验证：在调用工具前预测成功率。

        结合三种信号计算置信度：
        1. 历史成功率：该工具过去的成功/失败统计
        2. 反模式检测：检查已知的典型错误模式
        3. 上下文合理性：工具是否与当前任务上下文匹配

        Args:
            tool_name: 工具名
            action_args: 工具参数
            context: 当前任务上下文（可选）

        Returns:
            (confidence: float 0~1, warnings: List[str])
            confidence < 0.5 时建议重新考虑
        """
        warnings = []
        confidence = 0.7  # 基线置信度

        # ── 信号1: 历史成功率 ──
        stats = self._tool_stats.get(tool_name)
        if stats:
            total = stats["success"] + stats["fail"]
            if total >= 3:
                success_rate = stats["success"] / total
                confidence = confidence * 0.4 + success_rate * 0.6
                if success_rate < 0.3:
                    warnings.append(f"工具 {tool_name} 历史成功率仅 {success_rate:.0%}")

        # ── 信号2: 反模式检测 ──
        antipatterns = self.TOOL_ANTIPATTERNS.get(tool_name, [])
        for check_fn, msg in antipatterns:
            try:
                if check_fn(action_args):
                    warnings.append(msg)
                    confidence -= 0.2
            except Exception:
                pass

        # ── 信号3: 高风险工具标记 ──
        if tool_name in self.HIGH_RISK_TOOLS:
            confidence -= 0.1
            warnings.append(f"{tool_name} 是高风险工具，建议确认参数")

        # ── 信号4: 参数完整性 ──
        if not action_args or all(not v for v in action_args.values()):
            warnings.append("参数为空，工具可能无法正确执行")
            confidence -= 0.15

        confidence = max(0.0, min(1.0, confidence))

        # 记录验证结果
        self.reflection_history.append(ReflectionResult(
            type=ReflectionType.CORRECT if confidence >= 0.5 else ReflectionType.DETECT,
            message=f"验证 {tool_name}: 置信度 {confidence:.2f}, {len(warnings)} 个警告",
            confidence=confidence,
            should_retry=confidence < 0.5,
        ))

        return confidence, warnings

    def record_outcome(self, tool_name: str, success: bool):
        """
        【v2】记录工具执行结果，用于更新历史成功率。

        Args:
            tool_name: 工具名
            success: 是否成功
        """
        if tool_name not in self._tool_stats:
            self._tool_stats[tool_name] = {"success": 0, "fail": 0}
        if success:
            self._tool_stats[tool_name]["success"] += 1
        else:
            self._tool_stats[tool_name]["fail"] += 1
        self._save_stats()

    def suggest_alternatives(
        self,
        tool_name: str,
        action_args: dict,
        available_tools: List[str],
    ) -> List[Dict[str, Any]]:
        """
        【v2】多路径探索：当主方案置信度低时，生成备选方案。

        策略：
        1. 同类工具替换（如 read_file → read_file with different path）
        2. 降级工具（如 execute_code → run_command）
        3. 拆分步骤（如 "搜索并读取" → 先搜索，再读取）

        Args:
            tool_name: 原始工具名
            action_args: 原始参数
            available_tools: 可用工具列表

        Returns:
            备选方案列表 [{tool, args, reason}, ...]
        """
        alternatives = []

        # 替换策略映射
        replacements = {
            "execute_code": ["run_command"],
            "run_command": ["execute_code"],
            "read_file": ["smart_fetch", "read_webpage"],
            "smart_fetch": ["read_webpage", "browse_web"],
            "search": ["smart_fetch"],
            "write_file": ["execute_code"],
        }

        alts = replacements.get(tool_name, [])
        for alt_tool in alts:
            if alt_tool in available_tools:
                alternatives.append({
                    "tool": alt_tool,
                    "args": action_args.copy(),
                    "reason": f"{tool_name} 置信度低，尝试 {alt_tool} 作为替代",
                })

        # 参数修正建议
        if tool_name == "read_file" and action_args.get("input"):
            path = action_args["input"]
            # 尝试绝对路径
            if not os.path.isabs(path):
                alternatives.append({
                    "tool": tool_name,
                    "args": {**action_args, "input": os.path.abspath(path)},
                    "reason": "使用绝对路径代替相对路径",
                })

        return alternatives[:3]  # 最多 3 个备选

    def analyze_error(self, tool_name: str, error: str) -> ReflectionResult:
        """
        深入分析错误原因

        Returns:
            包含原因分析和纠正建议的结果
        """
        error_match = self._detect_error(error)
        if error_match:
            message = f"{tool_name}: {error_match[1]}"
            if error_match[0]:
                message += f" ({error_match[0]})"
        else:
            message = f"{tool_name}: 未知错误"

        reflection = ReflectionResult(
            type=ReflectionType.CAUSE,
            message=message,
            confidence=0.7,
            correction_hint=error_match[2] if error_match else "检查工具输入参数",
        )
        self.reflection_history.append(reflection)
        return reflection

    def generate_correction(self, error_reflection: ReflectionResult,
                            original_tool: str, original_args: dict) -> ReflectionResult:
        """
        基于错误分析生成纠正方案

        Returns:
            包含纠正建议的结果
        """
        hint = error_reflection.correction_hint or "检查参数并重试"

        correction_text = f"建议: {hint}"
        if original_args:
            correction_text += f"\n原始参数: {original_args}"

        reflection = ReflectionResult(
            type=ReflectionType.CORRECT,
            message=correction_text,
            confidence=0.6,
            should_retry=True,
        )
        self.reflection_history.append(reflection)
        return reflection

    def should_abort(self) -> bool:
        """判断是否应放弃任务（连续错误过多）"""
        return self.consecutive_errors > self.max_retries

    def reset(self):
        """重置反思状态（保留工具统计，跨任务累积）"""
        self.consecutive_errors = 0
        self.reflection_history.clear()

    def get_context_tokens(self, tokenizer) -> list:
        """获取最近的反思上下文"""
        if not self.reflection_history:
            return []
        # 只取最近 3 条反思
        recent = self.reflection_history[-3:]
        text = "".join(r.to_token_text() for r in recent)
        return tokenizer._encode(text)

    def _detect_error(self, text: str) -> Optional[Tuple[str, str, str]]:
        """
        检测文本中的错误模式

        Returns:
            (matched_pattern, error_desc, correction_hint) 或 None
        """
        if not text:
            return None

        for pattern, desc, hint in self.ERROR_PATTERNS:
            match = re.search(pattern, text)
            if match:
                # 尝试提取匹配的捕获组作为补充信息
                try:
                    extra = match.group(1) if match.lastindex else ""
                except (IndexError, AttributeError):
                    extra = ""
                full_desc = desc.format(extra) if extra else desc
                return (pattern, full_desc, hint)

        return None

    def get_stats(self) -> dict:
        """获取反思统计（含 v2 工具成功率）"""
        total = len(self.reflection_history)
        errors = sum(1 for r in self.reflection_history
                     if r.type == ReflectionType.DETECT)
        confirms = sum(1 for r in self.reflection_history
                       if r.type == ReflectionType.CONFIRM)
        # v2: 工具成功率汇总
        tool_success_rates = {}
        for tool, stats in self._tool_stats.items():
            total_calls = stats["success"] + stats["fail"]
            if total_calls > 0:
                tool_success_rates[tool] = round(stats["success"] / total_calls, 2)

        return {
            "total_reflections": total,
            "errors_detected": errors,
            "confirms": confirms,
            "consecutive_errors": self.consecutive_errors,
            "should_abort": self.should_abort(),
            "tool_success_rates": tool_success_rates,
        }