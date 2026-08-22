#!/usr/bin/env python3
"""C28 Gap 1 + Gap 2 回归守卫（无需真实 checkpoint）。

Gap 1（场记忆自动捕获）：
  G1.1  _capture_field_memory() 直接调用 → record_field_memory 收到
        (field_state, label, text, phase) 正确参数
  G1.2  generate(auto_capture=True) 默认 → _capture_field_memory 被调用
  G1.3  generate(auto_capture=False) → _capture_field_memory 不被调用
        （向后兼容 / 隔离验证脚本）

Gap 2（coaction 连续路径补全）：
  G2.1  continuous_forward 源码中存在 coaction.update 调用（与离散 forward 一致）
  G2.2  调用带 round_num=t+1 + try/except 非致命包装
  G2.3  仅在 len(active_this) >= 2 时调用（与 CoactivationTracker 单 neuron 短路一致）

运行：python -u scripts/training/verify_c28_gap1_gap2_contract.py
"""
from __future__ import annotations

import os
import sys
import inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

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


def main() -> None:
    print("=" * 64, flush=True)
    print("C28 Gap 1（场记忆自动捕获）+ Gap 2（coaction 连续路径）回归守卫",
          flush=True)
    print("=" * 64, flush=True)

    # ════════════════════════════════════════════════════════════
    # Gap 1: 场记忆自动捕获
    # ════════════════════════════════════════════════════════════
    print("\n[Gap 1] 场记忆自动捕获（auto_capture）", flush=True)

    # G1.1: _capture_field_memory 直接调用
    print("\n  G1.1 _capture_field_memory() 直接调用验证...", flush=True)
    import neuroplex.brain.cortex as cortex_mod  # noqa: E402

    # 构造一个最小 mock Cortex（只暴露 _capture_field_memory 需要的接口）
    class _MockCortexForCapture:
        def __init__(self, field_state, phase):
            self._fs = field_state
            self._phase = phase

        def get_last_field_state(self):
            return self._fs

        def get_last_phase(self):
            return self._phase

    fs = torch.ones(3072) * 0.7
    mock_cortex = _MockCortexForCapture(fs, phase=1.234)

    # Mock get_sleep_engine 返回记录调用的假 engine
    record_calls: list = []
    original_get_sleep = cortex_mod.get_sleep_engine if hasattr(
        cortex_mod, "get_sleep_engine") else None

    class _FakeSleepEngine:
        def record_field_memory(self, vector, label, text=None, phase=None):
            record_calls.append({
                "vector_norm": float(vector.norm()) if torch.is_tensor(vector) else None,
                "label": label,
                "text": text,
                "phase": phase,
            })

    fake_engine = _FakeSleepEngine()
    # 通过 monkeypatch sleep_engine 模块的 get_sleep_engine
    import neuroplex.life.sleep_engine as sleep_mod  # noqa: E402
    original_global_get = sleep_mod.get_sleep_engine
    sleep_mod.get_sleep_engine = lambda config=None: fake_engine

    try:
        # 调用真实 Cortex._capture_field_memory（绑定到 mock 对象）
        cortex_mod.Cortex._capture_field_memory(
            mock_cortex, "测试 prompt 内容", "生成的回复文本"
        )
    finally:
        sleep_mod.get_sleep_engine = original_global_get

    check("G1.1a record_field_memory 被调用 1 次",
          len(record_calls) == 1, f"calls={len(record_calls)}")
    if record_calls:
        c = record_calls[0]
        check("G1.1b field_state 正确传入（非空张量）",
              c["vector_norm"] is not None and c["vector_norm"] > 0,
              f"norm={c['vector_norm']:.4f}")
        check("G1.1c label 来自 prompt[:40]",
              c["label"] == "测试 prompt 内容",
              f"label={c['label']!r}")
        check("G1.1d text 是生成结果",
              c["text"] == "生成的回复文本",
              f"text={c['text']!r}")
        check("G1.1e phase 来自 get_last_phase()",
              c["phase"] == 1.234,
              f"phase={c['phase']}")
    else:
        check("G1.1b-e record_call 存在", False, "no calls")

    # G1.2/G1.3: auto_capture 门控
    # 验证两层：(a) generate() 把 auto_capture 透传给 _generate_p7
    #          (b) _generate_p7 源码确实在 auto_capture=True 且 text 非空时
    #              调 _capture_field_memory（源码级 inspect，避开真实 _generate_p7
    #              的重装配依赖）
    print("\n  G1.2/G1.3 auto_capture 门控验证...", flush=True)

    generate_p7_calls: list = []

    class _StubCortex:
        """最小 stub：让 generate() 走 n_candidates<=1 单候选路径。
        仅记录 _generate_p7 收到的 auto_capture，不模拟内部 capture。"""
        _tokenizer_hub = "set"  # 绕过 generate() 顶部的 None 检查

        def _generate_p7(self, *args, **kwargs):
            generate_p7_calls.append(kwargs.get("auto_capture"))
            return "stub generated text"

        def _is_degenerate_text(self, text):
            return False  # 不触发退化重试

    stub = _StubCortex()
    # 绑定真实 Cortex.generate 到 stub
    bound_generate = cortex_mod.Cortex.generate.__get__(stub, _StubCortex)

    # G1.2: auto_capture=True（默认）— 透传验证
    bound_generate("hello", auto_capture=True)
    check("G1.2 generate(auto_capture=True) → _generate_p7 收到 auto_capture=True",
          len(generate_p7_calls) >= 1 and generate_p7_calls[-1] is True,
          f"p7_calls={generate_p7_calls}")

    # G1.3: auto_capture=False — 透传验证
    generate_p7_calls.clear()
    bound_generate("hello", auto_capture=False)
    check("G1.3a generate(auto_capture=False) → _generate_p7 收到 auto_capture=False",
          len(generate_p7_calls) >= 1 and generate_p7_calls[-1] is False,
          f"p7_calls={generate_p7_calls}")

    # G1.4: 默认参数（不传 auto_capture）= True
    generate_p7_calls.clear()
    bound_generate("hello")  # 不传 auto_capture
    check("G1.4 generate() 默认 → auto_capture=True 透传给 _generate_p7",
          len(generate_p7_calls) >= 1 and generate_p7_calls[-1] is True,
          f"p7_auto_capture={generate_p7_calls[-1] if generate_p7_calls else None}")

    # G1.5: 源码级 — _generate_p7 含 auto_capture 门控 + _capture_field_memory 调用
    p7_src = inspect.getsource(cortex_mod.Cortex._generate_p7)
    check("G1.5a _generate_p7 源码含 auto_capture 参数",
          "auto_capture" in p7_src, "")
    check("G1.5b _generate_p7 源码含 'if auto_capture and result_text.strip()' 门控",
          "if auto_capture and result_text.strip()" in p7_src, "")
    check("G1.5c _generate_p7 源码含 _capture_field_memory 调用",
          "self._capture_field_memory(" in p7_src, "")

    # G1.6: 源码级 — generate() 含 auto_capture 默认 True
    gen_src = inspect.getsource(cortex_mod.Cortex.generate)
    check("G1.6a generate() 源码含 auto_capture: bool = True 默认",
          "auto_capture: bool = True" in gen_src, "")
    check("G1.6b generate() 把 auto_capture 透传给 _generate_p7（单候选路径）",
          "auto_capture=auto_capture" in gen_src, "")

    # G1.7: 源码级 — _capture_field_memory 方法定义存在
    cap_src = inspect.getsource(cortex_mod.Cortex._capture_field_memory)
    check("G1.7 _capture_field_memory 方法定义存在且调 record_field_memory",
          "record_field_memory" in cap_src
          and "get_last_field_state" in cap_src
          and "get_last_phase" in cap_src, "")

    # ════════════════════════════════════════════════════════════
    # Gap 2: coaction 连续路径补全
    # ════════════════════════════════════════════════════════════
    print("\n[Gap 2] coaction 连续路径补全（continuous_forward）", flush=True)

    import neuroplex.resonance.ensemble as ens_mod  # noqa: E402

    cf_src = inspect.getsource(ens_mod.ResonanceEnsemble.continuous_forward)
    fwd_src = inspect.getsource(ens_mod.ResonanceEnsemble.forward)

    # G2.1: continuous_forward 源码中存在 coaction.update
    check("G2.1 continuous_forward 源码包含 coaction.update 调用",
          "self.coaction.update(" in cf_src,
          "（修复前缺失 → §4.2/§11 ⚠️ 缺口）")

    # G2.2: 带 round_num + try/except 非致命包装
    check("G2.2a continuous_forward 的 coaction.update 带 round_num=t+1",
          "round_num=t + 1" in cf_src or "round_num=t+1" in cf_src,
          "")
    check("G2.2b continuous_forward 的 coaction 带 try/except 非致命包装",
          "非致命" in cf_src or "logger.warning" in cf_src,
          "")

    # G2.3: 仅 len(active_this) >= 2 时调用
    check("G2.3 continuous_forward coaction 带 len(active_this) >= 2 门控",
          "len(active_this) >= 2" in cf_src or "len(active_this)>=2" in cf_src,
          "（与 CoactivationTracker.update 单 neuron 短路一致）")

    # G2.4: 确认这是新增的（离散 forward 已有，continuous 此前缺失）
    # 离散 forward 应该也有 coaction.update（作为对照）
    check("G2.4a 离散 forward 也有 coaction.update（对照基准）",
          "self.coaction.update(" in fwd_src,
          "（离散路径本就有，作为对照）")

    # G2.5: 两条路径的 coaction 调用数量（continuous 应 ≥1，discrete 应 ≥1）
    cf_coaction_count = cf_src.count("self.coaction.update(")
    fwd_coaction_count = fwd_src.count("self.coaction.update(")
    check("G2.5 continuous_forward 至少 1 处 coaction.update",
          cf_coaction_count >= 1,
          f"cf_count={cf_coaction_count}")
    check("G2.5b 离散 forward 至少 1 处 coaction.update（对照）",
          fwd_coaction_count >= 1,
          f"fwd_count={fwd_coaction_count}")

    # ════════════════════════════════════════════════════════════
    # 总结
    # ════════════════════════════════════════════════════════════
    print("\n" + "=" * 64, flush=True)
    if failed == 0:
        print(f"判定: 全部 {passed} 维 PASS — Gap 1 + Gap 2 契约闭合",
              flush=True)
        print("  Gap 1: generate() 默认自动沉淀场记忆；auto_capture=False 可隔离",
              flush=True)
        print("  Gap 2: continuous_forward 每步补全 coaction 统计",
              flush=True)
    else:
        print(f"判定: {failed} 维 FAIL（{passed} 维 PASS）", flush=True)
    print("=" * 64, flush=True)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
