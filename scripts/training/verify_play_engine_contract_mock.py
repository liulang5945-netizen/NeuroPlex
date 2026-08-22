#!/usr/bin/env python3
"""PlayEngine 契约修复的纯逻辑回归（无需真实 checkpoint）。

verify_play_engine_runtime_contract.py 走真实 assemble_cortex（需要 data/
下的 production checkpoint）。本脚本用最小 mock cortex 验证 R-PE-1/2/3 的
控制流契约在以下边界条件下都成立：

  1. 群体 final_scores 全正 → record_high_resonance_state 必被调用，
     field_state 来自 mock 的 get_last_field_state()
  2. 群体 final_scores 为空 dict → 不抛异常，返回低质量 PlayActivity
  3. 迭代器 bug 不再出现（next(iter(...)) 路径）

运行：python -u scripts/training/verify_play_engine_contract_mock.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

from neuroplex.life.play_engine import PlayEngine, PlayActivity, PlayConfig  # noqa: E402
from neuroplex.resonance.neuro_modulation import SleepConsolidator  # noqa: E402

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


class _MockNeuron(torch.nn.Module):
    """最小可参数化的 neuron mock（用于 device 推断路径）。"""

    def __init__(self, nid: str):
        super().__init__()
        self.neuron_id = nid
        self._dummy = torch.nn.Parameter(torch.zeros(4))


class _MockCortex:
    """最小 Cortex mock：暴露 _free_resonance_session 真正消费的属性/方法。

    关键：think() 与 get_last_field_state() 是 R-PE-2/3 修复点的入口。
    """

    def __init__(self, final_scores: dict, field_state: torch.Tensor | None):
        self.neurons = {
            "zh_std0_dialogue": _MockNeuron("zh_std0_dialogue"),
            "zh_aug0_dialogue": _MockNeuron("zh_aug0_dialogue"),
            "en": _MockNeuron("en"),
        }
        self._tokenizer_hub = None  # 走旧 tokenizer 分支
        self._tokenizer = _MockTokenizer()
        self._shared_embedding = _MockSharedEmbedding()
        self._final_scores = final_scores
        self._field_state = field_state
        self.think_call_count = 0

    def think(self, shared_embeddings=None, collab_mode="fusion", **kwargs):
        self.think_call_count += 1
        return {
            "field_state": self._field_state,
            "final_scores": dict(self._final_scores),
            "n_rounds": 1,
        }

    def get_last_field_state(self):
        return self._field_state


class _MockTokenizer:
    def encode(self, text: str):
        # 返回足够长的 id 列表（>=3 触发 P7 旧路径分支）
        return [1, 2, 3, 4, 5]


class _MockSharedEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._w = torch.nn.Parameter(torch.zeros(32, 16))

    def forward(self, ids):
        return torch.zeros(ids.shape[0], ids.shape[1], 16)


def _make_engine(cortex: _MockCortex) -> tuple[PlayEngine, SleepConsolidator]:
    pe = PlayEngine.__new__(PlayEngine)
    # 绕过 __init__（避免触发持久化路径），手工设置必要字段
    pe._cortex = cortex
    pe._coaction = None  # 不测 coaction 路径
    pe._play_step = 0
    pe.config = PlayConfig()  # _free_resonance_session 末尾读 min_quality_to_keep
    sc = SleepConsolidator(replay_buffer_size=16)
    pe._sleep_consolidator = sc
    # 包装 record_high_resonance_state 以观察调用
    pe._record_calls = []
    original_record = sc.record_high_resonance_state

    def traced_record(*args, **kwargs):
        pe._record_calls.append({
            "resonance_score": kwargs.get("resonance_score"),
            "field_state_norm": (
                float(kwargs["field_state"].norm())
                if torch.is_tensor(kwargs.get("field_state")) else None
            ),
            "text": kwargs.get("text"),
            "active_nids": kwargs.get("active_nids"),
        })
        return original_record(*args, **kwargs)

    sc.record_high_resonance_state = traced_record
    return pe, sc


def main() -> None:
    print("=" * 64, flush=True)
    print("PlayEngine 契约修复纯逻辑回归（mock cortex，无需 checkpoint）",
          flush=True)
    print("=" * 64, flush=True)

    # ── 场景 1：final_scores 全正 → record 必被调用 ──────────────
    print("\n[场景 1] 群体 final_scores 全正（max>0.5 阈值必满足）",
          flush=True)
    fs = {"zh_std0_dialogue": 2.5, "zh_aug0_dialogue": 1.2, "en": 0.4}
    field_state = torch.ones(3072) * 0.5
    cortex = _MockCortex(fs, field_state)
    pe, sc = _make_engine(cortex)

    result = pe._free_resonance_session()

    check("S1.1 cortex.think 被调用 ≥ 1 次",
          cortex.think_call_count >= 1,
          f"think_calls={cortex.think_call_count}")
    check("S1.2 返回非 None PlayActivity",
          isinstance(result, PlayActivity),
          f"type={type(result).__name__}")
    check("S1.3 record_high_resonance_state 被调用",
          len(pe._record_calls) >= 1,
          f"record_calls={len(pe._record_calls)}")
    check("S1.4 record 传入的 field_state 是非空张量",
          len(pe._record_calls) >= 1
          and pe._record_calls[0]["field_state_norm"] is not None
          and pe._record_calls[0]["field_state_norm"] > 0.0,
          f"first_norm={pe._record_calls[0]['field_state_norm']}"
          if pe._record_calls else "no calls")
    check("S1.5 record 传入的 text 是 topic",
          len(pe._record_calls) >= 1
          and pe._record_calls[0]["text"] is not None,
          f"text={pe._record_calls[0]['text']!r}"
          if pe._record_calls else "no calls")
    check("S1.6 迭代器 bug 不出现（无 TypeError）",
          result is not None,
          "（result 非 None 表示 line 211 路径未抛 TypeError）")

    # ── 场景 2：final_scores 为空 → 不抛、返回低质量活动 ─────────
    print("\n[场景 2] 群体 final_scores 为空（极端情况，群体未激活）",
          flush=True)
    cortex2 = _MockCortex({}, field_state)
    pe2, sc2 = _make_engine(cortex2)

    result2 = pe2._free_resonance_session()

    check("S2.1 不抛异常，返回 PlayActivity",
          isinstance(result2, PlayActivity),
          f"type={type(result2).__name__}")
    check("S2.2 quality_score == 0.0（群体未激活）",
          isinstance(result2, PlayActivity) and result2.quality_score == 0.0,
          f"quality={result2.quality_score}"
          if isinstance(result2, PlayActivity) else "no result")
    check("S2.3 record_high_resonance_state 未被调用",
          len(pe2._record_calls) == 0,
          f"record_calls={len(pe2._record_calls)}")

    # ── 场景 3：field_state 为 None → record 跳过、不抛 ─────────
    print("\n[场景 3] get_last_field_state() 返回 None（任务场为空）",
          flush=True)
    cortex3 = _MockCortex(fs, None)
    pe3, sc3 = _make_engine(cortex3)

    result3 = pe3._free_resonance_session()

    check("S3.1 不抛异常，返回 PlayActivity",
          isinstance(result3, PlayActivity),
          f"type={type(result3).__name__}")
    check("S3.2 record_high_resonance_state 未被调用（field_state=None 跳过）",
          len(pe3._record_calls) == 0,
          f"record_calls={len(pe3._record_calls)}")

    # ── 场景 4：active_nids 注入 coaction → coaction.update 被调用 ─
    print("\n[场景 4] coaction 接线 + ≥2 神经元激活 → coaction.update 调用",
          flush=True)

    class _MockCoaction:
        def __init__(self):
            self.update_calls = []

        def update(self, ids):
            self.update_calls.append(list(ids))

    cortex4 = _MockCortex(fs, field_state)
    pe4, sc4 = _make_engine(cortex4)
    pe4._coaction = _MockCoaction()

    pe4._free_resonance_session()

    check("S4.1 coaction.update 被调用 ≥ 1 次",
          len(pe4._coaction.update_calls) >= 1,
          f"update_calls={len(pe4._coaction.update_calls)}")
    check("S4.2 coaction.update 收到 ≥ 2 个 active_ids",
          len(pe4._coaction.update_calls) >= 1
          and len(pe4._coaction.update_calls[0]) >= 2,
          f"first_ids={pe4._coaction.update_calls[0]}"
          if pe4._coaction.update_calls else "no calls")

    print("\n" + "=" * 64, flush=True)
    if failed == 0:
        print(f"判定: 全部 {passed} 维 PASS — R-PE-1/2/3 控制流契约闭合",
              flush=True)
    else:
        print(f"判定: {failed} 维 FAIL（{passed} 维 PASS）", flush=True)
    print("=" * 64, flush=True)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
