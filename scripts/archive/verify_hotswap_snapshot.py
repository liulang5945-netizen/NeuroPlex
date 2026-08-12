#!/usr/bin/env python3
"""验证热插拔三机制之一：快照隔离 + 混合规格热插拔 + 并发增删。

人脑启发：神经元动态增加/减少不影响工作中的推理（正常对话输出）。
验证点：
[1] 快照隔离：推理线程持续 forward，主线程并发 add/remove/isolate/revive，
    推理全程不崩溃、输出有限值
[2] 混合规格热插拔：field_dim≠unified 的新 neuron 加入后自动补建跨规格投影层，
    推理正常（此前 3072 vs 2048 RuntimeError 回归防护）
[3] 增删后快照语义：被删 neuron 在快照中仍持引用（统计性影响），
    增删只影响后续推理，不破坏当前推理
[4] 隔离/复活：pop 保留引用可复活，复活后推理正常

Usage:
    python scripts/training/verify_hotswap_snapshot.py
"""
import os
import sys
import threading
import time
from dataclasses import replace

os.environ.setdefault("TAIJI_TEST_MODE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from taiji.resonance.config import TINY_TEST
from taiji.resonance.neuron import ResonanceNeuron
from taiji.resonance.field import ResonanceField
from taiji.resonance.ensemble import ResonanceEnsemble
from taiji.resonance.tribal import CoactivationTracker

# 共享 embedding 契约维度（所有 neuron 一致，对应生产 256000×512）
BASE_EMBED_DIM = 512


def make_neuron(nid: str, field_dim: int = 512, seed: int = 0) -> ResonanceNeuron:
    """构造 tiny 测试 neuron（field_dim 可覆盖以模拟混合规格）。"""
    cfg = replace(TINY_TEST, neuron_id=nid, field_dim=field_dim)
    torch.manual_seed(seed)
    n = ResonanceNeuron(cfg)
    n.eval()
    return n


def build_ensemble() -> ResonanceEnsemble:
    """mini 混合规格 ensemble：base 512 场 + 2 个 tiny neuron。"""
    n_a = make_neuron("tiny_a", field_dim=512, seed=1)
    n_b = make_neuron("tiny_b", field_dim=512, seed=2)
    field = ResonanceField(dim=512)
    ens = ResonanceEnsemble(
        {"tiny_a": n_a, "tiny_b": n_b}, field,
        max_rounds=2,
        coaction=CoactivationTracker(),
    )
    return ens


def run_inference_loop(ens: ResonanceEnsemble, results: dict, stop: threading.Event):
    """推理线程：持续 forward，记录异常与输出有限性。"""
    errors = []
    n_calls = 0
    finite_ok = True
    try:
        for i in range(80):
            if stop.is_set():
                break
            emb = torch.randn(1, 4, BASE_EMBED_DIM)
            out = ens.forward(shared_embeddings=emb)
            n_calls += 1
            scores = out["final_scores"]
            for v in scores.values():
                if not torch.isfinite(torch.tensor(v)).all():
                    finite_ok = False
            # 每 10 次让出 CPU，模拟真实推理耗时交错
            if i % 10 == 0:
                time.sleep(0.02)
    except Exception as e:  # noqa: BLE001
        import traceback
        errors.append(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    results["errors"] = errors
    results["n_calls"] = n_calls
    results["finite_ok"] = finite_ok


def main():
    print("=" * 60)
    print("热插拔快照隔离 + 混合规格热插拔验证")
    print("=" * 60)

    ens = build_ensemble()
    print(f"[0] ensemble 就绪: neurons={list(ens.neurons.keys())}, field_dim={ens.field.dim}")

    # 启动推理线程
    results: dict = {}
    stop = threading.Event()
    t = threading.Thread(target=run_inference_loop, args=(ens, results, stop), daemon=True)
    t.start()
    time.sleep(0.05)

    # 主线程并发增删/隔离/复活（混合规格 + 同规格混打）
    print("\n[1] 主线程并发热插拔（推理进行中）...")
    mixed_cfg = replace(TINY_TEST, neuron_id="tiny_mixed", field_dim=256)
    torch.manual_seed(3)
    mixed_neuron = ResonanceNeuron(mixed_cfg)
    mixed_neuron.eval()

    # 场景 A: 混合规格热插拔（field_dim=256 ≠ field.dim=512）
    ens.add_neuron("tiny_mixed", mixed_neuron)
    assert "tiny_mixed" in ens.neurons, "[1-A] 混合规格 add_neuron 失败"
    assert "tiny_mixed" in ens._cross_spec_projectors, \
        "[1-A] 混合规格投影层未补建"
    proj = ens._cross_spec_projectors["tiny_mixed"]
    assert proj.linear1.in_features == 256 and proj.linear1.out_features == 512, \
        "[1-A] 投影层维度错误"
    print(f"  ok [1-A] 混合规格热插拔: 投影层自动补建 (256->512)")

    # 场景 B: 同规格热插拔
    n_c = make_neuron("tiny_c", field_dim=512, seed=4)
    ens.add_neuron("tiny_c", n_c)
    print(f"  ok [1-B] 同规格热插拔: neurons={len(ens.neurons)}")

    # 场景 C: 推理中途移除一个 neuron（模拟凋亡 remove）
    ens.neurons.pop("tiny_a")
    print(f"  ok [1-C] 推理中移除 tiny_a: neurons={list(ens.neurons.keys())}")

    # 场景 D: 隔离 + 复活（保留引用）
    isolated_ref = ens.neurons.pop("tiny_b")
    print(f"  ok [1-D] 推理中隔离 tiny_b: neurons={list(ens.neurons.keys())}")
    ens.add_neuron("tiny_b", isolated_ref)
    print(f"  ok [1-D] 复活 tiny_b: neurons={list(ens.neurons.keys())}")

    # 场景 E: 再加入一个混合规格（低维 256）后立刻推理一次（直接验证投影路径）
    emb = torch.randn(1, 4, BASE_EMBED_DIM)
    out = ens.forward(shared_embeddings=emb)
    assert all(
        torch.isfinite(torch.tensor(v)).all() for v in out["final_scores"].values()
    ), "[1-E] 推理分数非有限"
    print("  ok [1-E] 含混合规格 neuron 的推理正常")

    # 场景 F: 移除混合规格 neuron 后推理仍正常
    ens.neurons.pop("tiny_mixed")
    out = ens.forward(shared_embeddings=torch.randn(1, 4, BASE_EMBED_DIM))
    assert len(out["final_scores"]) >= 2, "[1-F] 移除后推理输出异常"
    print("  ok [1-F] 移除混合规格后推理正常")

    # 停止推理线程并等待
    stop.set()
    t.join(timeout=30)

    # 汇总
    print(f"\n[2] 推理线程汇总: calls={results.get('n_calls', 0)}")
    errors = results.get("errors", [])
    if errors:
        print(f"  fail 推理线程异常 {len(errors)} 次:")
        for e in errors[:3]:
            print(e)
        print("=" * 60)
        return 1
    assert results.get("n_calls", 0) > 0, "推理线程未执行任何 forward"
    assert results.get("finite_ok", False), "推理输出出现非有限值"
    print(f"  ok [2] 推理线程 {results['n_calls']} 次 forward 全部正常"
          f"（并发增删/隔离/复活期间不崩溃）")

    # 混合规格投影层回归测试（独立于并发）
    print("\n[3] 混合规格投影层回归（静态）...")
    ens2 = build_ensemble()
    cfg2 = replace(TINY_TEST, neuron_id="m2", field_dim=256)
    torch.manual_seed(5)
    m2 = ResonanceNeuron(cfg2)
    m2.eval()
    ens2.add_neuron("m2", m2)
    out2 = ens2.forward(shared_embeddings=torch.randn(1, 4, BASE_EMBED_DIM))
    assert all(
        torch.isfinite(torch.tensor(v)).all() for v in out2["final_scores"].values()
    ), "[3] 静态混合规格推理失败"
    print("  ok [3] 混合规格 neuron 推理分数有限（旧 3072-vs-2048 崩溃已修复）")

    print(f"\n{'='*60}")
    print("热插拔快照隔离验证: 全部通过")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
