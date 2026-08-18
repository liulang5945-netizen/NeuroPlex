"""对话训练/评估口径契约（2026-08-12 机制化）。

统一对话数据的 prompt 构造与守卫判断，作为「单一真相源」放在核心库，
供 cortex 守卫与回归测试共用，杜绝散落脚本各自为政导致的口径漂移。

背景：
- dialogue neuron（zh_aug*_dialogue / zh_std0_dialogue）用 "问：{question}\n答："
  格式训练（SFT answer masking，只对 "答：" 之后计算 loss）。
- 裸 prompt 下 50K 模型陷入换行/空格死循环（top1 恒为 ▁ token）→ 假退化。
- 历史同类口径错误（07-29 评估集分布失真、07-31 token ID 错位、08-12 裸 prompt）
  均靠人工发现——本模块把契约固化，配合 tests/test_dialogue_format.py 回归防复发。
"""
from __future__ import annotations

from typing import Iterable, Optional

SFT_ANSWER_MARKER = "答："
Q_MARKER = "问："


def build_dialogue_prompt(question: str) -> str:
    """构造与训练数据格式严格一致的对话 prompt。

    Args:
        question: 用户问题（不含 "问：" 前缀）。

    Returns:
        "问：{question}\n答："——与 finetune 训练数据逐字符一致。
    """
    return f"{Q_MARKER}{question}\n{SFT_ANSWER_MARKER}"


def dialogue_prompt_requires_guard(
    prompt: str,
    domain: str,
    active_nids: Optional[Iterable[str]],
    allow_plain_prompt: bool = False,
) -> bool:
    """判断给定 prompt 是否触发对话口径守卫（硬失败）。

    Args:
        prompt: 原始输入文本。
        domain: 目标域（"zh" 等）。
        active_nids: 已归一化的激活神经元 ID 列表（None 视为无神经元）。
        allow_plain_prompt: 例外开关——base/域 neuron 评估用纯问题 prompt 时传 True。

    Returns:
        True 表示应抛 ValueError（裸 prompt + zh 域 + 激活 dialogue neuron）。
    """
    if domain != "zh":
        return False
    if prompt.rstrip().endswith(SFT_ANSWER_MARKER):
        return False
    if allow_plain_prompt:
        return False
    if not active_nids:
        return False
    return any(nid.endswith("_dialogue") for nid in active_nids)
