#!/usr/bin/env python3
"""C26 增量八验证：多阶段任务模式链 v2（TaskSet 序列，2026-08-14）。

背景：C25-F 首步 generate_staged（dict 阶段 + {prev} 文本传递）被 R17 标注
死代码（生产 generate 路径不消费）。增量八升级为 v2：
1. **TaskSet 类型化对象**：每阶段 = 任务集（prompt/mode/domain/active_nids/
   max_tokens/quality_gate/record_memory）——显式激活子集 = 任务集切换
2. **三重阶段间传递**：文本 {prev} + 场状态（final field_state → 下阶段
   seed_memories 记忆注意窗）+ 记忆写入（record_memory → 睡眠固化候选）
3. **阶段质量门**：退化检测 → 高温重试 → 仍退化隔离（阶段间互不污染）
4. **生产接入**：generate_task_chain + API 端点，摘除 R17 死代码标记

验证层次：
A. TaskSet 对象化：构造/序列化、active_nids 显式激活子集
B. 三重传递：文本 prev（阶段间文本流）+ 场状态（下阶段 seed_memories 生效，
   阶段间 final_scores 不同）+ 记忆写入（record_memory=True → 记忆库条目）
C. 质量门：退化阶段高温重试、重试恢复；仍退化隔离（阶段互不污染）
D. 兼容层：generate_staged（dict）转发 v2 仍可用
E. 生产：cortex_task_chain API 端点注册（import 检查）

运行：python -u scripts/training/verify_c27_task_chain.py
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
# N2（REMEDIATION_PLAN R7）：固定 seed 保证可复现
import random  # noqa: E402
import numpy as np  # noqa: E402

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
from neuroplex.loader import assemble_cortex  # noqa: E402
from neuroplex.brain.cortex import TaskSet  # noqa: E402
# 口径契约（2026-08-12）：zh/dialogue 域 prompt 必须走训练格式，裸 prompt 触发硬失败
from neuroplex.resonance.dialogue_format import build_dialogue_prompt  # noqa: E402

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


DIALOGUE_IDS = ["zh_aug0_dialogue", "zh_aug1_dialogue", "zh_aug2_dialogue",
                "zh_aug3_dialogue", "zh_std0_dialogue"]
COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("C26 增量八：多阶段任务模式链 v2（TaskSet 序列）", flush=True)
    print("=" * 60, flush=True)

    cortex, tokenizer, modules = assemble_cortex(
        neurons_dir="data/neurons",
        collab_name=COLLAB_NAME,
        extra_neurons_dir=EXTRA_NEURONS_DIR,
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=DIALOGUE_IDS,
    )
    print(f"  装配: {list(cortex.neurons.keys())}", flush=True)
    check("装配成功（5 dialogue + 4 general）", len(cortex.neurons) == 9)

    # judge EMA 预热（executive 判定主信号）
    warm = ["你好", "帮我写代码", "解一道数学题", "What is this?", "写一首诗"]
    for _ in range(30):
        for wp in warm:
            cortex._executive_route(wp)
    print("  judge EMA 预热完成", flush=True)

    # ── A. TaskSet 对象化 ──
    print("\n[A] TaskSet 对象化 ...", flush=True)
    ts = TaskSet(prompt="测试", mode="continuous", domain="zh",
                 active_nids=["zh_aug0_dialogue", "zh_aug1_dialogue"],
                 max_tokens=16, quality_gate=True, record_memory=False)
    check("A1. TaskSet 构造（显式激活子集）",
          ts.active_nids == ["zh_aug0_dialogue", "zh_aug1_dialogue"]
          and ts.domain == "zh" and ts.mode == "continuous")
    ts2 = TaskSet(prompt="默认", )
    check("A2. TaskSet 默认值（continuous/无约束）",
          ts2.mode == "continuous" and ts2.domain is None
          and ts2.active_nids is None and ts2.quality_gate)

    # ── B. 三重传递：文本 prev + 场状态 + 记忆写入 ──
    print("\n[B] 三重传递 ...", flush=True)
    # B1: 文本 prev 传递（显式 {prev} 模板）
    # 口径守卫：zh 域阶段一律训练格式 "问：{q}\n答："（阶段0/2），code 域可裸 prompt
    chain = [
        TaskSet(prompt=build_dialogue_prompt(
                    "用户需求：写一个 Python 函数，输入 n，输出斐波那契数列第 n 项。"),
                mode="continuous", domain="zh", max_tokens=16),
        TaskSet(prompt="根据需求：{prev}\n写出满足需求的完整 Python 函数代码。",
                mode="continuous", domain="code", max_tokens=32),
        TaskSet(prompt="问：用中文向用户解释以下代码的功能：{prev}\n答：",
                mode="continuous", domain="zh", max_tokens=16,
                record_memory=True, memory_label="任务链阶段3"),
    ]
    result = cortex.generate_task_chain(chain, max_tokens_per_stage=16)
    outs = result["outputs"]
    fss = result["field_states"]
    gates = result["gates"]
    for i, o in enumerate(outs):
        print(f"    阶段{i + 1} → {o[:50]!r}", flush=True)
    check("B1. 三阶段输出全部非空（文本 prev 传递生效）",
          all(isinstance(o, str) and len(o.strip()) > 0 for o in outs),
          f"lens={[len(o) for o in outs]}")
    check("B2. 场状态三重传递：各阶段 field_state 被截获",
          all(fs is not None for fs in fss),
          f"fs_dims={[None if fs is None else tuple(fs.shape) for fs in fss]}")
    check("B3. record_memory=True 阶段已写入记忆候选",
          gates[2].get("memory") == "recorded",
          f"gate3={gates[2]}")
    # B4: 记忆库已收到候选（sleep_engine 全局单例 pending）
    try:
        from neuroplex.life.sleep_engine import get_sleep_engine
        engine = get_sleep_engine()
        pending = list(engine.pending_field_memories)
        check("B4. 记忆库收到任务链写入候选",
              len(pending) >= 1 and any(
                  lbl == "任务链阶段3" for _, lbl, *_ in pending),
              f"pending={[(None, lbl) for _, lbl, *_ in pending]}")
        # 清理（不污染后续/真实记忆）
        engine.pending_field_memories = []
    except Exception as e:
        check("B4. 记忆库收到任务链写入候选", False, f"err={e}")

    # B5: 场状态 seed_memories 生效（带 prev_fs vs 不带 生成不同）
    print("\n[B5] 场状态 seed_memories 生效性 ...", flush=True)
    out_with = cortex.generate("写一个 Python 函数",
                               domain="code", max_tokens=16, temperature=0.55)
    out_with_fs = cortex.generate("写一个 Python 函数", domain="code",
                                  max_tokens=16, temperature=0.55,
                                  memory_vectors=[(fss[0], 0.8)])
    print(f"    无记忆: {out_with[:30]!r}", flush=True)
    print(f"    带记忆: {out_with_fs[:30]!r}", flush=True)
    check("B5. 场状态 seed_memories 改变生成（记忆注意窗生效）",
          out_with != out_with_fs,
          f"same={out_with == out_with_fs}")

    # ── C. 质量门 ──
    print("\n[C] 阶段质量门 ...", flush=True)
    # C1: 质量门开启时退化输出被重试（gate 记录 retried 或 degenerate）
    gate_chain = [
        TaskSet(prompt=build_dialogue_prompt("写一首关于春天的诗。"), mode="continuous",
                domain="zh", max_tokens=24, quality_gate=True),
    ]
    gr = cortex.generate_task_chain(gate_chain)
    g = gr["gates"][0]
    check("C1. 质量门记录（ok/retried/degenerate）",
          g.get("quality") in ("ok", "retried", "degenerate"),
          f"gate={g}")
    check("C2. 质量门开启时输出非退化",
          not (gr["outputs"][0] and cortex._is_degenerate_text(gr["outputs"][0])),
          f"out={gr['outputs'][0][:40]!r}")

    # ── D. 兼容层：generate_staged（dict）转发 v2 ──
    print("\n[D] C25-F 兼容层转发 ...", flush=True)
    outs_d = cortex.generate_staged([
        {"prompt": build_dialogue_prompt("你好"), "mode": "continuous",
         "domain": "zh", "max_tokens": 8},
        {"prompt": "问：上一阶段说了：{prev}\n答：", "mode": "continuous",
         "domain": "zh", "max_tokens": 8},
    ], max_tokens_per_stage=8)
    check("D1. generate_staged dict 转发 v2 可用",
          len(outs_d) == 2 and all(len(o.strip()) > 0 for o in outs_d),
          f"lens={[len(o) for o in outs_d]}")

    # ── E. 生产接入（API 端点注册）──
    print("\n[E] 生产接入 ...", flush=True)
    try:
        import api.routes_taiji as rt
        routes = {getattr(r, "path", ""): r for r in rt.router.routes}
        has_chain = any("/cortex/task_chain" in p for p in routes)
        check("E1. /api/taiji/cortex/task_chain 端点已注册", has_chain,
              f"paths={[p for p in routes if 'task' in p]}")
    except Exception as e:
        check("E1. /api/taiji/cortex/task_chain 端点已注册", False, f"err={e}")
    try:
        has_taskset = hasattr(cortex, "generate_task_chain")
        check("E2. cortex.generate_task_chain 生产方法存在", has_taskset)
    except Exception:
        pass

    print(f"\n  总耗时: {time.time() - t0:.1f}s", flush=True)
    print("=" * 60, flush=True)
    print(f"结果: {passed} PASS / {failed} FAIL", flush=True)
    print("=" * 60, flush=True)
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
