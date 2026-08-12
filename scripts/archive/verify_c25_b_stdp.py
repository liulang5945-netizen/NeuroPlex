#!/usr/bin/env python3
"""C25-B STDP 突触生长/修剪本体化冒烟验证（2026-08-10）。

对比文档 2.11/2.12（缺口 R）：态极 STDP 此前"只追踪不驱动"——权重缩放
[0.5,2.0] 有，但缺通道级结构可塑性（excite/inhibit_channels 条目修剪/生长）。
C25-B 修复：
1. STDPTracker 增共激活统计累积（(pre,post)→count/sim 持久化，跨会话）
2. apply_structure_updates：长期低共激活弱通道条目修剪 + 高共激活缺失
   通道生长（邻居相似初始化）——连接层"突触可塑性"从权重缩放升级为结构演化
3. SleepConsolidator.consolidate 接入（离线路径，不碰 forward_train 监督）

验证目标：
1. 共激活统计累积正确（有向 pre→post + sim）
2. 修剪：低共激活弱通道删除（含 scale/bias 清理）；强权重通道保留
3. 生长：高共激活缺失通道建立（维度正确 + 邻居相似初始化）
4. 持久化：get_state_dict/load_state_dict round-trip
5. sleep 接入：consolidate(stdp_tracker=...) 返回结构演化统计

运行：python -u scripts/training/verify_c25_b_stdp.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from taiji.resonance.stdp import STDPTracker, FiringRecord
from taiji.resonance.neuro_modulation import SleepConsolidator

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


class FakeNeuron(nn.Module):
    """带 excite/inhibit_channels + config 的最小 neuron 替身（对齐真 ResonanceNeuron 接口）。"""

    def __init__(self, nid: str, field_dim: int = 8, hidden: int = 8):
        super().__init__()
        self.config = type("Cfg", (), {"neuron_id": nid, "field_dim": field_dim,
                                       "hidden_size": hidden})()
        self.excite_channels = nn.ModuleDict()
        self.inhibit_channels = nn.ModuleDict()
        self._channel_usage = {}

    def establish(self, peer_id: str, peer_neuron: "FakeNeuron",
                  ctype: str = "excite", weight: float = 0.5) -> None:
        ch = nn.Linear(peer_neuron.config.field_dim, self.config.hidden_size, bias=False)
        ch.weight.data.fill_(weight)
        ch_dict = self.excite_channels if ctype == "excite" else self.inhibit_channels
        ch_dict[peer_id] = ch
        self.register_parameter(f"{ctype}_scale_{peer_id}",
                                nn.Parameter(torch.tensor(50.0)))
        self.register_buffer(f"{ctype}_bias_{peer_id}", torch.zeros(1))


def make_vec(dim: int = 8, seed: int = 0, scale: float = 1.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g) * scale
    return v / (v.norm() + 1e-8)  # 归一化，保证 sim 可预测


def main() -> None:
    print("=" * 60)
    print("C25-B STDP 突触生长/修剪本体化冒烟验证")
    print("=" * 60)

    # ---- 1. 共激活统计累积（有向 pre→post + sim）----
    tr = STDPTracker()
    # A 先于 B 发放（round 1→2），方向相似
    tr.record_firing("A", 1, make_vec(seed=1))
    tr.record_firing("B", 2, make_vec(seed=1))
    # B 先于 C 发放（round 2→3），但方向不同
    tr.record_firing("C", 3, make_vec(seed=2))
    added = tr.accumulate_coactivation()
    check("累积有向对", added >= 2, f"added={added}")

    s_ab = tr.get_coactivation_stats("A", "B")
    check("(A,B) count=1", s_ab["count"] == 1, f"count={s_ab['count']}")
    check("(A,B) avg_sim 高（同向）", s_ab["total_sim"] > 0.8,
          f"sim={s_ab['total_sim']:.2f}")
    # 反向 (B,A) 不累积（B 晚于 A 发放，非 pre→post）
    s_ba = tr.get_coactivation_stats("B", "A")
    check("(B,A) 不累积（有向）", s_ba["count"] == 0, f"count={s_ba['count']}")
    # (B,C) 方向不同 → count 有但 avg_sim 低
    s_bc = tr.get_coactivation_stats("B", "C")
    check("(B,C) count=1", s_bc["count"] == 1, f"count={s_bc['count']}")
    check("(B,C) avg_sim 低（异向）", s_bc["total_sim"] < 0.3,
          f"sim={s_bc['total_sim']:.2f}")
    tr.clear_history()

    # ---- 2. 修剪：低共激活弱通道删除 + scale/bias 清理 ----
    tr2 = STDPTracker()
    a, b, c = FakeNeuron("a"), FakeNeuron("b"), FakeNeuron("c")
    a.establish("b", b, weight=0.005)   # 弱通道，无共激活统计 → 应修剪
    a.establish("c", c, weight=0.8)     # 强权重通道，无统计 → 应保留（防误删）
    st = tr2.apply_structure_updates({"a": a, "b": b, "c": c})
    check("修剪 a<-b（弱+无统计）", st["pruned"] == 1,
          f"pruned={st['pruned']} keys={st['pruned_keys']}")
    check("a<-b 通道已删除", "b" not in a.excite_channels)
    check("a<-b scale param 已清理", "excite_scale_b" not in a._parameters)
    check("a<-b bias buffer 已清理", "excite_bias_b" not in a._buffers)
    check("a<-c 强权重保留", "c" in a.excite_channels,
          f"keys={list(a.excite_channels.keys())}")

    # ---- 3. 生长：高共激活缺失通道建立（邻居相似初始化）----
    tr3 = STDPTracker()
    x, y = FakeNeuron("x"), FakeNeuron("y", field_dim=8, hidden=8)
    y.establish("x", x, weight=0.5)  # y 已有 x 通道（作为"邻居"）
    # 构造 (n, y) 高共激活（count=6, sim=0.9）：y 应长出 <-n 通道
    tr3._coactivation_stats[("n", "y")] = {"count": 6, "total_sim": 6 * 0.9, "last_update": 1}
    n = FakeNeuron("n")
    st3 = tr3.apply_structure_updates({"x": x, "y": y, "n": n})
    check("生长 y<-n（高共激活缺失）", st3["grown"] == 1,
          f"grown={st3['grown']} keys={st3['grown_keys']}")
    check("y<-n 通道已建立", "n" in y.excite_channels)
    ch_n = y.excite_channels["n"]
    check("生长通道维度正确", ch_n.weight.shape == (8, n.config.field_dim),
          f"shape={tuple(ch_n.weight.shape)}")
    # 邻居相似初始化：y 已有 x 通道（weight=0.5），n 通道应接近 0.5+噪声
    w_n = float(ch_n.weight.data.mean().item())
    check("邻居相似初始化（≈0.5+小噪声）", abs(w_n - 0.5) < 0.05,
          f"mean_w={w_n:.3f}")

    # ---- 4. 持久化 round-trip ----
    tr4 = STDPTracker()
    tr4._coactivation_stats[("a", "b")] = {"count": 7, "total_sim": 5.6, "last_update": 3}
    state = tr4.get_state_dict()
    tr5 = STDPTracker()
    tr5.load_state_dict(state)
    s = tr5.get_coactivation_stats("a", "b")
    check("持久化 count 恢复", s["count"] == 7, f"count={s['count']}")
    check("持久化 sim 恢复", abs(s["total_sim"] - 5.6) < 1e-6,
          f"sim={s['total_sim']}")
    check("阈值持久化", tr5.grow_count_threshold == tr4.grow_count_threshold)

    # ---- 5. sleep 接入：consolidate(stdp_tracker=...) ----
    sc = SleepConsolidator()
    tr6 = STDPTracker()
    p, q, r = FakeNeuron("p"), FakeNeuron("q"), FakeNeuron("r")
    p.establish("q", q, weight=0.005)  # 弱通道，将结构修剪
    # 高共激活 (q,p)：将生长 p<-q 缺失方向? p 已有 q 通道 → 不会生长；
    # 用 (r,p) 高共激活 → 生长 p<-r
    tr6._coactivation_stats[("r", "p")] = {"count": 6, "total_sim": 5.0, "last_update": 1}
    result = sc.consolidate({"p": p, "q": q, "r": r}, stdp_tracker=tr6)
    check("consolidate 返回结构修剪统计",
          result["channels_struct_pruned"] >= 1,
          f"struct_pruned={result['channels_struct_pruned']}")
    check("consolidate 返回结构生长统计",
          result["channels_grown"] >= 1,
          f"grown={result['channels_grown']}")
    check("consolidate 保留既有统计字段",
          "channels_downscaled" in result and "replayed_states" in result)

    print(f"\n结果: {passed} PASS / {failed} FAIL")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
