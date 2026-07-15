"""
PII（个人身份信息）脱敏工具

在写入训练数据前调用，防止 API Key、密码、邮箱、手机号等
敏感信息泄露到微调数据集中。
"""
import re

# 常见 PII 模式（顺序敏感：先匹配长模式，避免子串误匹配）
_PII_PATTERNS = [
    (re.compile(r'\b(sk-[A-Za-z0-9]{20,})', re.IGNORECASE), '[API_KEY]'),
    (re.compile(r'\b(ghp_[A-Za-z0-9]{36,})', re.IGNORECASE), '[GITHUB_TOKEN]'),
    (re.compile(r'\b(key-[A-Za-z0-9]{20,})', re.IGNORECASE), '[API_KEY]'),
    (re.compile(r'(Bearer\s+[A-Za-z0-9._\-]{20,})', re.IGNORECASE), '[BEARER_TOKEN]'),
    (re.compile(r'(Authorization:\s*[A-Za-z0-9._\-+/=]{20,})', re.IGNORECASE), '[AUTH_HEADER]'),
    (re.compile(r'\b(AKIA[0-9A-Z]{16})\b'), '[AWS_ACCESS_KEY]'),
    (re.compile(r'\b([A-Za-z0-9/+=]{40})\b(?=.*(?:aws|secret))', re.IGNORECASE), '[AWS_SECRET]'),
    (re.compile(r'-----BEGIN\s+(RSA\s+)?PRIVATE\s+KEY-----[\s\S]*?-----END\s+(RSA\s+)?PRIVATE\s+KEY-----'), '[PRIVATE_KEY]'),
    (re.compile(r'\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b'), '[EMAIL]'),
    (re.compile(r'\b1[3-9]\d{9}\b'), '[PHONE]'),
    (re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'), '[IP_ADDR]'),
    (re.compile(r'((?:password|passwd|pwd|secret|token)\s*[=:]\s*)\S+', re.IGNORECASE), r'\1[PASSWORD]'),
    (re.compile(r'\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b'), '[CARD_NUM]'),
    (re.compile(r'\b\d{6}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b'), '[ID_CARD]'),
]


def sanitize_pii(text: str) -> str:
    """对文本进行 PII 脱敏。

    Args:
        text: 原始文本

    Returns:
        脱敏后的文本（PII 替换为标记）
    """
    if not text:
        return text
    result = text
    for pattern, replacement in _PII_PATTERNS:
        result = pattern.sub(replacement, result)
    return result
