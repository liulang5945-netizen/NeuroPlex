#!/usr/bin/env python3
"""C25-D 睡眠重放真重放 + 突触稳态下调冒烟验证（2026-08-09）。

对比文档 2.6/2.11/2.12 弱项："态极 sleep 是'拿累积样本离线训练'，重放/下调是
方向性借鉴，未实现生物意义上的'逐条回放 + 全局缩放'"。C25-D 修复：
1. 真重放：replay 记录 active_nids，consolidate 重放时再激活共激活统计
   （人脑海马回放 → 皮层再激活 → 突触巩固），而非纯统计占位
2. 突触稳态下调（downscaling）：全局 side_channels ×0.98（NREM 慢波全局缩放）

验证目标：
1. record_high_resonance_state 带 active_nids → 重放驱动共激活（get_strong_pairs 可见）
2. 重放驱动的强 pair 通道被强化（×1.1）
3. downscaling 全局缩放生效（channels_downscaled > 0）
4. 既有强化/修剪/fingerprint/遗忘路径保留

运行：python -u scripts/training/verify_c25_d_replay.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.resonance.neuro_modulation import SleepConsolidator
from taiji.resonance.tribal import CoactivationTracker

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


class FakeNeuron:
    """带 excite_channels + 弱通道修剪的最小 neuron 替身。"""

    def __init__(self, nid: str, channel_keys: list, weight: float = 0.5):
        self.config = type("Cfg", (), {"neuron_id": nid})()
        self.excite_channels = {
            k: torch.nn.Linear(8, 8, bias=False) for k in channel_keys
        }
        for lin in self.excite_channels.values():
            lin.weight.data.fill_(weight)

    def prune_weak_channels(self, threshold: float = 0.01) -> int:
        """修剪权重绝对值均值 < threshold 的通道（模拟真 neuron 语义）。"""
        pruned = 0
        for key in list(self.excite_channels.keys()):
            if self.excite_channels[key].weight.abs().mean().item() < threshold:
                del self.excite_channels[key]
                pruned += 1
        return pruned


def main() -> None:
    print("=" * 60)
    print("C25-D 睡眠重放真重放 + 突触稳态下调冒烟验证")
    print("=" * 60)

    # ---- 装配 ----
    coaction = CoactivationTracker()
    sc = SleepConsolidator(downscale_factor=0.98)
    neurons = {
        "code": FakeNeuron("code", ["math", "zh"], weight=0.5),
        "math": FakeNeuron("math", ["code", "en"], weight=0.5),
        "zh": FakeNeuron("zh", ["code", "math", "en"], weight=0.05),
    }

    # ---- 1. 记录高共振状态 ----
    # 6 次 (code,math) 重放（EMA 累积超过 strong 阈值）+ 1 次旧格式（无 active_nids）
    print("\n[1] 高共振状态记录", flush=True)
    for i in range(6):
        sc.record_high_resonance_state(
            field_state=torch.randn(8), resonance_score=0.9, step=100 + i,
            active_nids=["code", "math"],  # C25-D：重放将驱动 (code,math) 共激活
        )
    sc.record_high_resonance_state(
        field_state=torch.randn(8), resonance_score=0.8, step=200,
        active_nids=None,  # 旧格式记录：仅计数，不驱动
    )
    check("replay buffer 含 7 条记录", len(sc._replay_buffer) == 7)
    check("记录含 active_nids", sc._replay_buffer[0]["active_nids"] == ["code", "math"])
    check("旧记录 active_nids=None", sc._replay_buffer[6]["active_nids"] is None)

    # ---- 2. consolidate：真重放 + 强化 + downscaling + 修剪 ----
    print("\n[2] consolidate 真重放", flush=True)
    result = sc.consolidate(
        neurons=neurons, coactivation_tracker=coaction, current_step=1000,
    )
    print(f"  stats: {result}", flush=True)

    # 真重放：6 次 (code,math) 重放累积 → 共激活超过 strong 阈值
    co = coaction.get_coactivation("code", "math")
    check("重放累积驱动 (code,math) 共激活", co > 0.2, f"co={co:.4f}")
    check("旧记录不驱动共激活", coaction.get_coactivation("code", "en") == 0.0)
    check("replayed_states == 7", result["replayed_states"] == 7)

    # 重放驱动的强 pair 被强化（×1.1）：(code,math) 进入 strong_pairs
    strong = coaction.get_strong_pairs(threshold=0.2)
    check("重放 pair 进入 strong_pairs", ("code", "math") in strong, f"strong={strong}")
    # code→math 通道：初始 0.5 → 强化 ×1.1 → downscaling ×0.98 ≈ 0.539
    w_after = neurons["code"].excite_channels["math"].weight.abs().mean().item()
    check("强通道净保留（≈0.5×1.1×0.98=0.539）",
          abs(w_after - 0.5 * 1.1 * 0.98) < 1e-3, f"w={w_after:.4f}")

    # ---- 3. downscaling 全局缩放 ----
    print("\n[3] 突触稳态下调", flush=True)
    check("channels_downscaled > 0", result["channels_downscaled"] > 0,
          f"n={result['channels_downscaled']}")
    # 未强化通道（zh→code，weight=0.05）：×0.98 → ≈0.049（弱信号被压低）
    w_zh = neurons["zh"].excite_channels["code"].weight.abs().mean().item()
    check("弱通道整体下压（×0.98）", abs(w_zh - 0.05 * 0.98) < 1e-3, f"w={w_zh:.4f}")
    # 所有通道被缩放：zh 有 3 个 + code 2 个 + math 2 个 = 7
    check("缩放通道数 == 7", result["channels_downscaled"] == 7)

    # ---- 4. 修剪 / 遗忘路径保留 ----
    print("\n[4] 既有路径保留", flush=True)
    check("弱通道修剪仍执行", "channels_pruned" in result)
    check("指纹更新字段保留", "fingerprints_updated" in result)
    check("遗忘字段保留", "pairs_forgotten" in result)

    # ---- 5. 持久化兼容（replay 含 active_nids 可序列化）----
    print("\n[5] 持久化兼容", flush=True)
    sc.record_high_resonance_state(
        field_state=torch.randn(8), resonance_score=0.85, step=300,
        active_nids=["zh", "math"],
    )
    state = sc.get_state_dict()
    check("replay 序列化含 active_nids",
          any("active_nids" in r for r in state["replay_buffer"]))
    sc2 = SleepConsolidator()
    sc2.load_state_dict(state)
    check("load 还原 active_nids",
          sc2._replay_buffer[0].get("active_nids") == ["zh", "math"])
    check("downscale_factor 持久化", sc2.downscale_factor == 0.98)

    print("\n" + "=" * 60)
    print(f"结果: {passed} PASS / {failed} FAIL")
    print("=" * 60)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
