#!/usr/bin/env python3
"""PlayEngine 自由共振运行契约回归（2026-08-20 机制审计修复）。

背景：
    NEUROPLEX_MECHANISM_RUNTIME_MAP_20260820.md §12.2 实测发现 PlayEngine
    `_free_resonance_session()` 在 line 211 抛 `TypeError: 'dict_values'
    object is not an iterator`，被外层 except 吞掉 → 永远返回 None。同时
    §3.3 / §12.1 确认 ResonanceNeuron.forward() 不产出 `resonance_score`
    和 `_last_field_state`，所以即便绕过迭代器 bug，原实现的"激活/coaction/
    高共振 replay"链也从未真正闭合。

    本脚本回归三件修复（R-PE-1/2/3）：
      R-PE-1  迭代器 bug：next(dict.values()) → next(iter(dict.values()))
      R-PE-2  走真实 Cortex.think(collab_mode="continuous") 而非直调
              neuron.forward()；共振分来自 final_scores（时间平均激活）
      R-PE-3  field_state 来自 cortex.get_last_field_state()（真实任务场），
              而非 neuron._last_field_state（永不写入）

判据（5/5 全过 = 契约闭合）：
    C1  _free_resonance_session 行级 trace 不出现 TypeError 异常事件
    C2  执行到 R-PE-2 修复块（cortex.think 被调用 ≥ 1 次）
    C3  think_result["final_scores"] 非空（群体真实产生共振分）
    C4  返回非 None 的 PlayActivity（play 路径不再被外层 except 吞掉）
    C5  若 max norm_score > 0.5：record_high_resonance_state 被调用，
        且传入的 field_state 是非空张量（真实任务场，非 _last_field_state）

约束：
    - 冻结 9 成员 production weights（不写 checkpoint，不训练）
    - 仅运行 1 次 _free_resonance_session（契约回归，非长跑）
    - 复用 assemble_cortex 生产装配

运行：python -u scripts/training/verify_play_engine_runtime_contract.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

from neuroplex.loader import assemble_cortex  # noqa: E402
from neuroplex.life.play_engine import PlayEngine  # noqa: E402
from neuroplex.resonance.neuro_modulation import SleepConsolidator  # noqa: E402

COLLAB_NAME = "collab_v3_c24v2.ckpt.pt"
EXTRA_NEURONS_DIR = "data/foundation_v1_dual"
NEURONS_DIR = "data/neurons"

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
    t0 = time.time()
    today = time.strftime("%Y%m%d")
    print("=" * 64, flush=True)
    print("PlayEngine 自由共振运行契约回归（R-PE-1/2/3）", flush=True)
    print("=" * 64, flush=True)

    print("\n[1/4] 装配 9 成员 production cortex（冻结，不写 checkpoint）...",
          flush=True)
    cortex, _tok, modules = assemble_cortex(
        neurons_dir=NEURONS_DIR,
        collab_name=COLLAB_NAME,
        extra_neurons_dir=EXTRA_NEURONS_DIR,
        device="cpu",
        max_rounds=3,
        wire_bio_modules=True,
        neuron_ids=None,
    )
    play_engine: PlayEngine = modules["play_engine"]
    sleep_consolidator = SleepConsolidator(replay_buffer_size=64)
    # 注入一个真实的 SleepConsolidator（modules 中的可能为 None 或未接线）
    play_engine.set_brain_interfaces(
        cortex=cortex,
        coactivation=getattr(cortex, "coaction", None),
        sleep_consolidator=sleep_consolidator,
    )
    print(f"  装配 {len(cortex.neurons)} 神经元，PlayEngine 已接线",
          flush=True)

    # ── 安装 instrumentation ───────────────────────────────────────
    # 行级 trace：记录 _free_resonance_session 执行的所有行号 + 异常事件
    line_trace: list = []
    exception_events: list = []
    play_file = play_engine.__module__
    # 取 play_engine.py 的绝对路径用于行号匹配
    import neuroplex.life.play_engine as pe_mod
    play_path = pe_mod.__file__

    original_think = cortex.think
    think_calls: list = []

    def traced_think(*args, **kwargs):
        result = original_think(*args, **kwargs)
        think_calls.append({
            "collab_mode": kwargs.get("collab_mode"),
            "final_scores_len": len(result.get("final_scores") or {}),
            "field_state_present": result.get("field_state") is not None,
            "final_scores_sample": dict(list((result.get("final_scores") or {}).items())[:3]),
        })
        return result

    cortex.think = traced_think

    original_record = sleep_consolidator.record_high_resonance_state
    record_calls: list = []

    def traced_record(*args, **kwargs):
        fs = kwargs.get("field_state")
        record_calls.append({
            "resonance_score": kwargs.get("resonance_score"),
            "active_nids": kwargs.get("active_nids"),
            "text": kwargs.get("text"),
            "field_state_shape": list(fs.shape) if torch.is_tensor(fs) else None,
            "field_state_norm": float(fs.norm()) if torch.is_tensor(fs) else None,
        })
        return original_record(*args, **kwargs)

    sleep_consolidator.record_high_resonance_state = traced_record

    previous_trace = sys.gettrace()

    def _trace_play(frame, event, arg):
        if frame.f_code.co_filename == play_path and \
                frame.f_code.co_name == "_free_resonance_session":
            if event == "line":
                line_trace.append(frame.f_lineno)
            elif event == "exception":
                exc_type, exc_value, _ = arg
                exception_events.append({
                    "type": exc_type.__name__,
                    "message": str(exc_value),
                    "line": frame.f_lineno,
                })
            return _trace_play
        return _trace_play

    # ── 执行 _free_resonance_session ──────────────────────────────
    print("\n[2/4] 行级 trace + 执行 _free_resonance_session() ...",
          flush=True)
    sys.settrace(_trace_play)
    play_result = None
    play_error: str | None = None
    try:
        play_result = play_engine._free_resonance_session()
    except Exception as exc:
        play_error = f"{type(exc).__name__}: {exc}"
    finally:
        sys.settrace(previous_trace)
        cortex.think = original_think
        sleep_consolidator.record_high_resonance_state = original_record

    print(f"  行级 trace: {len(line_trace)} 行", flush=True)
    print(f"  异常事件: {len(exception_events)} 个", flush=True)
    print(f"  think 调用: {len(think_calls)} 次", flush=True)
    print(f"  record_high_resonance_state 调用: {len(record_calls)} 次",
          flush=True)
    print(f"  play_result: {play_result!r}", flush=True)
    if play_error:
        print(f"  play_error: {play_error}", flush=True)

    # ── 判据 ───────────────────────────────────────────────────────
    print("\n" + "=" * 64, flush=True)
    print("5 维判据：", flush=True)
    print("=" * 64, flush=True)

    # C1: 无 TypeError 异常事件（R-PE-1 修复点）
    type_errors = [e for e in exception_events if e["type"] == "TypeError"]
    check("C1 行级 trace 无 TypeError 异常（R-PE-1 迭代器修复）",
          len(type_errors) == 0,
          f"type_errors={len(type_errors)}" +
          (f" first={type_errors[0]}" if type_errors else ""))

    # C2: cortex.think 被调用（R-PE-2 走真实 ensemble 路径）
    check("C2 cortex.think 被调用 ≥ 1 次（R-PE-2 走 ensemble 路径）",
          len(think_calls) >= 1,
          f"think_calls={len(think_calls)}")

    # C3: final_scores 非空（群体真实产生共振分）
    final_scores_nonempty = (
        len(think_calls) >= 1 and think_calls[0]["final_scores_len"] > 0
    )
    check("C3 think_result['final_scores'] 非空（群体真实共振分）",
          final_scores_nonempty,
          f"final_scores_len={think_calls[0]['final_scores_len'] if think_calls else 0}")

    # C4: 返回非 None 的 PlayActivity（play 路径不再被外层 except 吞掉）
    check("C4 _free_resonance_session 返回非 None PlayActivity",
          play_result is not None and play_error is None,
          f"result={type(play_result).__name__ if play_result else 'None'} "
          f"error={play_error}")

    # C5: 若 max norm_score > 0.5 → record_high_resonance_state 被调用且
    # field_state 是非空张量（R-PE-3 真实任务场）
    # 由于 max_score 由 final_scores 归一化得到，至少有一个 nid score=1.0，
    # 所以阈值 0.5 必然满足 → record 应被调用（除非 sleep_consolidator 未接线）
    record_called_with_real_field = (
        len(record_calls) >= 1
        and record_calls[0]["field_state_shape"] is not None
        and record_calls[0]["field_state_norm"] is not None
        and record_calls[0]["field_state_norm"] > 0.0
    )
    check("C5 record_high_resonance_state 被调用且 field_state 非空张量"
          "（R-PE-3 真实任务场）",
          record_called_with_real_field,
          f"record_calls={len(record_calls)}" +
          (f" first_field_norm={record_calls[0]['field_state_norm']:.4f}"
           if record_calls and record_calls[0]['field_state_norm'] else ""))

    contract_pass = (failed == 0)

    print("\n" + "=" * 64, flush=True)
    if contract_pass:
        print("判定: PlayEngine 运行契约 PASS — 自由共振→高共振 replay 链闭合",
              flush=True)
        print("  R-PE-1 迭代器 bug 修复 + R-PE-2 走 ensemble 路径 + "
              "R-PE-3 真实任务场", flush=True)
        print("下一步: 重新跑 diag_runtime_mechanism_trace.py 确认生产路径"
              "play_result 不再为 None；再决定场记忆自动捕获 + coaction "
              "连续路径补全", flush=True)
    else:
        print(f"判定: PlayEngine 运行契约 FAIL ({failed} 维不过)", flush=True)
        print("下一步: 检查 think() 是否在 play 上下文成功返回；"
              "检查 get_last_field_state() 是否在 think() 后仍可读",
              flush=True)
    print("=" * 64, flush=True)

    report_obj = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task": "PlayEngine 自由共振运行契约回归（R-PE-1/2/3）",
        "cortex": {
            "n_neurons": len(cortex.neurons),
            "population": sorted(cortex.neurons.keys()),
            "collab_name": COLLAB_NAME,
        },
        "play_line_trace_count": len(line_trace),
        "play_exception_events": exception_events,
        "think_calls": think_calls,
        "record_high_resonance_state_calls": record_calls,
        "play_result_repr": repr(play_result),
        "play_error": play_error,
        "checks": {
            "passed": passed,
            "failed": failed,
        },
        "verdict": ("PlayEngine 契约 PASS" if contract_pass
                    else f"PlayEngine 契约 FAIL ({failed} 维不过)"),
        "elapsed_seconds": time.time() - t0,
    }
    out_path = f"reports/play_engine_runtime_contract_{today}.json"
    os.makedirs("reports", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {out_path}", flush=True)
    print(f"总耗时: {time.time() - t0:.1f}s", flush=True)

    if not contract_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
