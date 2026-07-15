"""
态极免疫系统 (Safety Guard)
============================

态极的自我保护机制 — 防止被恶意输入伤害，防止异常扩散。

安全规则：
1. 输入过滤：过滤注入攻击、恶意指令
2. 输出审查：防止输出系统路径、密钥等敏感信息
3. 资源限制：CPU/内存过高时暂停自动任务
4. 异常隔离：一个引擎崩溃不影响其他引擎
5. 频率限制：防止疯狂调用
"""
import re
import time
import logging
import functools
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger("Taiji.Safety")


class SafetyGuard:
    """
    态极免疫系统

    保护态极不被恶意输入伤害，防止异常扩散。
    """

    # 敏感信息模式（不应出现在输出中）
    SENSITIVE_PATTERNS = [
        r'(?:password|passwd|pwd)\s*[=:]\s*\S+',
        r'(?:api[_-]?key|apikey)\s*[=:]\s*\S+',
        r'(?:secret|token)\s*[=:]\s*\S+',
        r'(?:ssh-rsa|ssh-ed25519)\s+\S+',
        r'-----BEGIN\s+(?:RSA|EC|OPENSSH)\s+PRIVATE\s+KEY-----',
    ]

    # 恶意输入模式
    MALICIOUS_PATTERNS = [
        r'ignore\s+(?:all\s+)?previous\s+instructions',
        r'you\s+are\s+now\s+(?:a|an)\s+',
        r'system\s*:\s*you\s+are',
        r'forget\s+(?:everything|all)',
        r'override\s+(?:safety|security)',
    ]

    def __init__(self, max_events_per_minute: int = 60):
        self._rate_limits: Dict[str, List[float]] = defaultdict(list)
        self._max_events_per_minute = max_events_per_minute
        self._blocked_count = 0
        self._alert_count = 0
        self._sensitive_regex = [re.compile(p, re.IGNORECASE) for p in self.SENSITIVE_PATTERNS]
        self._malicious_regex = [re.compile(p, re.IGNORECASE) for p in self.MALICIOUS_PATTERNS]

    def validate_input(self, text: str) -> tuple:
        """
        验证输入是否安全。

        Returns:
            (is_safe: bool, reason: str)
        """
        if not text or not text.strip():
            return True, ""

        # 检查恶意模式
        for pattern in self._malicious_regex:
            if pattern.search(text):
                self._blocked_count += 1
                logger.warning(f"Blocked malicious input: {text[:100]}...")
                return False, "检测到恶意指令模式"

        # 检查输入长度（防止超长输入攻击）
        if len(text) > 100000:
            return False, "输入过长"

        return True, ""

    def validate_output(self, text: str) -> str:
        """
        审查输出，脱敏敏感信息。

        Returns:
            脱敏后的文本
        """
        if not text:
            return text

        result = text
        for pattern in self._sensitive_regex:
            result = pattern.sub("[REDACTED]", result)

        return result

    def check_resources(self, body) -> bool:
        """
        检查资源是否充足（可以继续执行自动任务）。

        Args:
            body: BodyCore 实例

        Returns:
            True = 资源充足，可以继续
        """
        try:
            resources = body.check_resources()
            if resources.get("cpu_percent", 0) > 90:
                logger.warning("CPU usage too high, pausing auto tasks")
                return False
            if resources.get("memory_percent", 0) > 90:
                logger.warning("Memory usage too high, pausing auto tasks")
                return False
        except Exception as e:
            logger.debug("safety: non-critical %s", e, exc_info=True)
        return True

    def rate_limit(self, action: str, max_per_minute: int = None) -> bool:
        """
        频率限制检查。

        Returns:
            True = 允许执行, False = 被限制
        """
        limit = max_per_minute or self._max_events_per_minute
        now = time.time()
        cutoff = now - 60

        # 清理过期记录
        self._rate_limits[action] = [
            t for t in self._rate_limits[action] if t > cutoff
        ]

        if len(self._rate_limits[action]) >= limit:
            self._blocked_count += 1
            logger.warning(f"Rate limit exceeded for {action}: {len(self._rate_limits[action])}/{limit} per minute")
            return False

        self._rate_limits[action].append(now)
        return True

    def isolate_execution(self, fn: Callable, *args, default=None, **kwargs) -> Any:
        """
        异常隔离执行 — 一个引擎崩溃不影响其他引擎。

        Args:
            fn: 要执行的函数
            default: 异常时的默认返回值

        Returns:
            函数返回值，或异常时的默认值
        """
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            self._alert_count += 1
            logger.error(f"Isolated execution error in {fn.__name__}: {e}")
            return default

    def get_threat_level(self) -> str:
        """获取当前威胁级别"""
        if self._blocked_count > 10:
            return "high"
        elif self._blocked_count > 3:
            return "medium"
        return "low"

    def get_status(self) -> dict:
        """获取免疫系统状态"""
        return {
            "threat_level": self.get_threat_level(),
            "blocked_count": self._blocked_count,
            "alert_count": self._alert_count,
            "active_rate_limits": len(self._rate_limits),
        }

    def reset(self):
        """重置计数器"""
        self._blocked_count = 0
        self._alert_count = 0
        self._rate_limits.clear()