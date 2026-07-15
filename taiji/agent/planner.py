"""
ModelSelf 规划系统
前额叶 — 让模型拥有自主任务规划能力

将复杂任务分解为可执行步骤，跟踪进度，处理失败和重新规划。
"""
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field
from enum import IntEnum

logger = logging.getLogger("ModelSelf.Planner")


class StepStatus(IntEnum):
    PENDING = 0     # 待执行
    ACTIVE = 1      # 正在执行
    DONE = 2        # 已完成
    FAILED = 3      # 失败
    SKIPPED = 4     # 跳过


class PlanAction(IntEnum):
    """规划头输出的动作类型"""
    NEW_PLAN = 0    # 创建新计划
    NEXT_STEP = 1   # 执行下一步
    REPLAN = 2      # 重新规划
    SKIP_STEP = 3   # 跳过当前步骤
    DONE = 4        # 任务完成
    WAIT = 5        # 等待外部输入
    ABORT = 6       # 放弃任务
    CONTINUE = 7    # 继续当前步骤


@dataclass
class PlanStep:
    """单个计划步骤"""
    step_id: int
    description: str
    status: StepStatus = StepStatus.PENDING
    tool_name: Optional[str] = None
    result_summary: Optional[str] = None
    error: Optional[str] = None

    def to_token_text(self) -> str:
        status_map = {
            StepStatus.PENDING: "pending",
            StepStatus.ACTIVE: "active",
            StepStatus.DONE: "done",
            StepStatus.FAILED: "failed",
            StepStatus.SKIPPED: "skipped",
        }
        return f'<step id="{self.step_id}" status="{status_map[self.status]}">{self.description}</step>'

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "description": self.description,
            "status": int(self.status),
            "tool_name": self.tool_name,
            "result_summary": self.result_summary,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PlanStep":
        return cls(
            step_id=d["step_id"],
            description=d["description"],
            status=StepStatus(d.get("status", 0)),
            tool_name=d.get("tool_name"),
            result_summary=d.get("result_summary"),
            error=d.get("error"),
        )


class Plan:
    """完整计划"""
    def __init__(self, task: str, steps: List[PlanStep] = None):
        self.task = task
        self.steps: List[PlanStep] = steps or []
        self.current_step_idx: int = 0
        self.replan_count: int = 0

    @property
    def current_step(self) -> Optional[PlanStep]:
        if 0 <= self.current_step_idx < len(self.steps):
            return self.steps[self.current_step_idx]
        return None

    @property
    def progress(self) -> float:
        if not self.steps:
            return 0.0
        done = sum(1 for s in self.steps if s.status == StepStatus.DONE)
        return done / len(self.steps)

    @property
    def is_complete(self) -> bool:
        return all(s.status in (StepStatus.DONE, StepStatus.SKIPPED) for s in self.steps)

    def advance(self) -> Optional[PlanStep]:
        """前进到下一步"""
        if self.current_step:
            self.current_step.status = StepStatus.DONE
        self.current_step_idx += 1
        if self.current_step:
            self.current_step.status = StepStatus.ACTIVE
            return self.current_step
        return None

    def mark_failed(self, error: str = ""):
        if self.current_step:
            self.current_step.status = StepStatus.FAILED
            self.current_step.error = error

    def to_token_text(self) -> str:
        """编码为 token 文本"""
        lines = [f'<plan task="{self.task}">']
        for step in self.steps:
            lines.append(f"  {step.to_token_text()}")
        lines.append("</plan>")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "steps": [s.to_dict() for s in self.steps],
            "current_step_idx": self.current_step_idx,
            "replan_count": self.replan_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        p = cls(d["task"], [PlanStep.from_dict(s) for s in d.get("steps", [])])
        p.current_step_idx = d.get("current_step_idx", 0)
        p.replan_count = d.get("replan_count", 0)
        return p


class PlannerSystem:
    """
    规划系统 — 任务分解与执行跟踪

    与模型的交互通过特殊 token:
    - <plan task="..."><step>N. description</step>...</plan> → 创建计划
    - <plan_done step="N"/> → 标记步骤完成
    - <replan> → 重新规划

    规划头输出 PlanAction，驱动规划状态机。
    """

    def __init__(self):
        self.current_plan: Optional[Plan] = None
        self.plan_history: List[Dict] = []  # 历史计划记录

    def create_plan(self, task: str, step_descriptions: List[str]) -> Plan:
        """创建新计划"""
        steps = [
            PlanStep(step_id=i + 1, description=desc)
            for i, desc in enumerate(step_descriptions)
        ]
        if steps:
            steps[0].status = StepStatus.ACTIVE

        self.current_plan = Plan(task, steps)
        logger.info(f"Plan created: {len(steps)} steps for '{task[:50]}'")
        return self.current_plan

    def get_current_step(self) -> Optional[PlanStep]:
        """获取当前应执行的步骤"""
        if self.current_plan:
            return self.current_plan.current_step
        return None

    def complete_current_step(self, result_summary: str = "") -> Optional[PlanStep]:
        """完成当前步骤，前进到下一步"""
        if not self.current_plan:
            return None
        if self.current_plan.current_step:
            self.current_plan.current_step.result_summary = result_summary
        return self.current_plan.advance()

    def fail_current_step(self, error: str) -> None:
        """标记当前步骤失败"""
        if self.current_plan:
            self.current_plan.mark_failed(error)

    def replan(self, new_steps: List[str]) -> Plan:
        """重新规划（保留已完成步骤）"""
        if not self.current_plan:
            return self.create_plan("replan", new_steps)

        self.current_plan.replan_count += 1
        # 保留已完成的步骤
        done_steps = [s for s in self.current_plan.steps
                      if s.status == StepStatus.DONE]
        new_plan_steps = [
            PlanStep(step_id=len(done_steps) + i + 1, description=desc)
            for i, desc in enumerate(new_steps)
        ]

        all_steps = done_steps + new_plan_steps
        if new_plan_steps:
            new_plan_steps[0].status = StepStatus.ACTIVE

        self.current_plan = Plan(self.current_plan.task, all_steps)
        logger.info(f"Replan #{self.current_plan.replan_count}: {len(new_steps)} new steps")
        return self.current_plan

    def handle_action(self, action: PlanAction, step_descs: List[str] = None,
                      error: str = "") -> Optional[str]:
        """
        处理规划头输出的动作

        Returns:
            模型应该看到的反馈文本
        """
        if action == PlanAction.NEW_PLAN and step_descs:
            self.create_plan("(自动规划)", step_descs)
            return self.current_plan.to_token_text()

        elif action == PlanAction.NEXT_STEP:
            step = self.complete_current_step()
            if step:
                return f"步骤完成。下一步: {step.description}"
            return "所有步骤已完成。"

        elif action == PlanAction.REPLAN:
            if step_descs:
                self.replan(step_descs)
                return self.current_plan.to_token_text()
            return "需要重新规划。"

        elif action == PlanAction.SKIP_STEP:
            if self.current_plan and self.current_plan.current_step:
                self.current_plan.current_step.status = StepStatus.SKIPPED
                step = self.current_plan.advance()
                if step:
                    return f"已跳过。下一步: {step.description}"
            return "已跳过。"

        elif action == PlanAction.DONE:
            if self.current_plan:
                if self.current_plan.current_step:
                    self.current_plan.current_step.status = StepStatus.DONE
                self.plan_history.append(self.current_plan.to_dict())
                self.current_plan = None
            return "任务完成。"

        elif action == PlanAction.ABORT:
            self.current_plan = None
            return "任务已放弃。"

        return None

    def get_context_tokens(self, tokenizer) -> list:
        """获取当前计划的上下文 token"""
        if not self.current_plan:
            return []
        return tokenizer._encode(self.current_plan.to_token_text())

    def get_status(self) -> dict:
        """获取规划状态"""
        if not self.current_plan:
            return {"has_plan": False}
        return {
            "has_plan": True,
            "task": self.current_plan.task,
            "progress": f"{self.current_plan.progress:.0%}",
            "current_step": self.current_plan.current_step.description if self.current_plan.current_step else None,
            "total_steps": len(self.current_plan.steps),
            "replan_count": self.current_plan.replan_count,
        }

    def parse_plan_tokens(self, token_ids: list, tokenizer) -> Optional[List[str]]:
        """从 token 序列解析计划步骤"""
        from taiji.config import SPECIAL_TOKENS
        ids = token_ids if isinstance(token_ids, list) else token_ids.tolist()

        steps = []
        in_plan = False
        step_parts = []

        for tid in ids:
            if tid == SPECIAL_TOKENS["plan_start"]:
                in_plan = True
                continue
            if tid == SPECIAL_TOKENS["plan_end"]:
                if step_parts:
                    steps.append("".join(step_parts).strip())
                return steps if steps else None
            if tid == SPECIAL_TOKENS["plan_step"]:
                if step_parts:
                    steps.append("".join(step_parts).strip())
                step_parts = []
                continue
            if tid == SPECIAL_TOKENS["plan_step_end"]:
                if step_parts:
                    steps.append("".join(step_parts).strip())
                    step_parts = []
                continue

            if in_plan:
                text = tokenizer.decode([tid], skip_special_tokens=True)
                if text:
                    step_parts.append(text)

        return steps if steps else None