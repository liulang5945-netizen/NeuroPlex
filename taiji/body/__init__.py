"""
态极身体系统 (Body System)
==========================

态极的肉体 — 天生一体的四肢、内脏、感官。

- core:       躯干（BodyCore）— 统一资源管理器
- limbs:      四肢 — 行动系统（工具调用、文件操作）
- metabolism: 代谢 — 硬件感知、资源管理
- senses:     感知 — 接收外部输入
"""

from taiji.body.core import BodyCore

__all__ = ["BodyCore"]