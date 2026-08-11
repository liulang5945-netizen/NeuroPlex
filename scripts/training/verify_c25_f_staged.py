#!/usr/bin/env python3
"""C25-F 多阶段任务模式链冒烟验证（2026-08-11）。

对比文档 2.11"回合级路由替代连续任务切换（多阶段任务留 v2）"修复：
- generate_staged：task-set 序列驱动——每阶段 = 激活模式 + 判定约束，
  阶段间显式传递中间输出（{prev} 模板或自动拼接）
- 本验证用 mock cortex 验证阶段链编排逻辑（模板/拼接/透传/异常），
  完整 9 神经元端到端待 C20 重训完成后执行（verify_c25_f_staged_e2e.py）

运行：python -u scripts/training/verify_c25_f_staged.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.brain.cortex import Cortex

passed = 0
failed = 0


def check(name: str, cond: bool, extra: str = "") -> None:
    global passed, failed
    if cond:
        passed += 1
        print(f"  [PASS] {name} {extra}", flush=True)
    else:
        failed += 1
        print(f"  [FAIL] {name} {extra}", flush=True)


class FakeCortex:
    """记录 generate 调用的 mock（验证阶段链编排，不真实 forward）。"""

    def __init__(self):
        self.calls: list = []
        self.replies = {"s1": "[s1输出]", "s2": "[s2输出]", "s3": "[s3输出]"}

    def generate(self, prompt: str, max_tokens: int = 60, temperature: float = 0.55,
                 top_k: int = 15, domain=None, repetition_penalty: float = 1.4,
                 collab_mode: str = "executive", fusion_mode: str = "soft",
                 **kwargs) -> str:
        self.calls.append({
            "prompt": prompt, "max_tokens": max_tokens, "domain": domain,
            "collab_mode": collab_mode, "fusion_mode": fusion_mode,
        })
        return self.replies.get(f"s{len(self.calls)}", "[out]")


def test_1_template_prev():
    """{prev} 模板填充：阶段输出注入下一阶段指令。"""
    fc = FakeCortex()
    stages = [
        {"prompt": "阶段1"},
        {"prompt": "阶段2：对上一阶段结果分析\n{prev}"},
    ]
    out = Cortex.generate_staged(fc, stages)
    check("两阶段输出", len(out) == 2, f"n={len(out)}")
    check("阶段2 prompt 含 prev", "{prev}" not in fc.calls[1]["prompt"]
          and "[s1输出]" in fc.calls[1]["prompt"],
          f"prompt={fc.calls[1]['prompt']!r}")
    check("阶段2 生成输入含上一阶段输出", "[s1输出]" in fc.calls[1]["prompt"])


def test_2_auto_concat():
    """无 {prev} 自动拼接：prompt\n上一阶段输出。"""
    fc = FakeCortex()
    stages = [{"prompt": "阶段1"}, {"prompt": "阶段2"}]
    out = Cortex.generate_staged(fc, stages)
    check("自动拼接", fc.calls[1]["prompt"].endswith("[s1输出]"),
          f"prompt={fc.calls[1]['prompt']!r}")


def test_3_first_stage_no_concat():
    """首阶段无 prev：prompt 原样。"""
    fc = FakeCortex()
    stages = [{"prompt": "首阶段指令"}]
    out = Cortex.generate_staged(fc, stages)
    check("首阶段 prompt 原样", fc.calls[0]["prompt"] == "首阶段指令",
          f"prompt={fc.calls[0]['prompt']!r}")


def test_4_task_set_params():
    """task-set 参数透传：domain/mode/max_tokens 正确传给 generate。"""
    fc = FakeCortex()
    stages = [
        {"prompt": "理解", "mode": "executive", "domain": "zh", "max_tokens": 40},
        {"prompt": "生成", "mode": "continuous", "domain": "code"},
    ]
    out = Cortex.generate_staged(fc, stages)
    c1, c2 = fc.calls[0], fc.calls[1]
    check("阶段1 domain=zh", c1["domain"] == "zh", f"domain={c1['domain']}")
    check("阶段1 mode=executive", c1["collab_mode"] == "executive",
          f"mode={c1['collab_mode']}")
    check("阶段1 max_tokens=40", c1["max_tokens"] == 40, f"max={c1['max_tokens']}")
    check("阶段2 domain=code", c2["domain"] == "code", f"domain={c2['domain']}")
    check("阶段2 mode=continuous（task-set 切换）", c2["collab_mode"] == "continuous",
          f"mode={c2['collab_mode']}")


def test_5_empty_prompt_skipped():
    """空 prompt 阶段跳过（输出 ""，不调 generate）。"""
    fc = FakeCortex()
    stages = [{"prompt": ""}, {"prompt": "有效阶段"}]
    out = Cortex.generate_staged(fc, stages)
    check("空阶段输出空串", out[0] == "", f"out={out[0]!r}")
    check("有效阶段照常调用", len(fc.calls) == 1, f"calls={len(fc.calls)}")


def test_6_exception_handled():
    """generate 异常 → 阶段输出 "" 且继续后续阶段。"""

    class Boom(FakeCortex):
        def generate(self, prompt=None, **k):
            if prompt == "失败阶段":
                raise RuntimeError("boom")
            return super().generate(prompt, **k)

    fc = Boom()
    stages = [{"prompt": "失败阶段"}, {"prompt": "后续阶段"}]
    out = Cortex.generate_staged(fc, stages)
    check("异常阶段输出空串", out[0] == "", f"out={out[0]!r}")
    check("后续阶段继续", len(fc.calls) == 1 and out[1] != "", f"out1={out[1]!r}")


def test_7_zh_code_zh_example():
    """示例任务链（zh 理解→code 生成→zh 表达）：三阶段编排。"""
    fc = FakeCortex()
    stages = [
        {"prompt": "请理解以下需求：写一个斐波那契函数", "mode": "executive", "domain": "zh"},
        {"prompt": "根据上面的理解生成 Python 代码：\n{prev}", "mode": "executive", "domain": "code"},
        {"prompt": "请用中文解释这段代码的作用：\n{prev}", "mode": "continuous", "domain": "zh"},
    ]
    out = Cortex.generate_staged(fc, stages, max_tokens_per_stage=20)
    check("三阶段输出", len(out) == 3, f"n={len(out)}")
    check("阶段间输出传递（阶段2 含阶段1 输出）", "[s1输出]" in fc.calls[1]["prompt"])
    check("阶段3 含阶段2 输出", "[s2输出]" in fc.calls[2]["prompt"])
    check("task-set 切换（zh→code→zh）",
          [c["domain"] for c in fc.calls] == ["zh", "code", "zh"],
          f"domains={[c['domain'] for c in fc.calls]}")


if __name__ == "__main__":
    print("=" * 60, flush=True)
    print("C25-F 多阶段任务模式链冒烟验证（task-set 序列）", flush=True)
    print("=" * 60, flush=True)
    for fn in [test_1_template_prev, test_2_auto_concat, test_3_first_stage_no_concat,
               test_4_task_set_params, test_5_empty_prompt_skipped,
               test_6_exception_handled, test_7_zh_code_zh_example]:
        fn()
    print("=" * 60, flush=True)
    print(f"结果: {passed}/{passed + failed} PASS", flush=True)
    sys.exit(0 if failed == 0 else 1)
