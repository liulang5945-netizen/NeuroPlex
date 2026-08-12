#!/usr/bin/env python3
"""验证热插拔三机制之二：训练/推理分离（影子权重 COW）。

人脑启发：学习/训练期间正常对话输出不受影响。
核心契约：
[1] 训练在 deepcopy 影子权重上进行 → live 权重训练全程稳定
[2] 推理（快照隔离读 dict）训练期间读到稳定权重，不崩溃
[3] 训练结束写回 live ← 影子（per-tensor copy_）+ 恢复 live 引用
[4] dict 引用不变（cortex.neurons 与 ensemble.neurons 同引用），
    内容替换对推理原子可见
[5] 训练期间被移除的 neuron 写回时跳过（不复活）

Usage:
    python scripts/training/verify_train_infer_separation.py
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

BASE_EMBED_DIM = 512
GENERAL_VOCAB = 4096  # 迷你 shared_embedding


def make_neuron(nid: str, seed: int = 0) -> ResonanceNeuron:
    cfg = replace(TINY_TEST, neuron_id=nid)
    torch.manual_seed(seed)
    n = ResonanceNeuron(cfg)
    n.eval()
    return n


class MiniCortex:
    """模拟 cortex 的最小接口：neurons dict（与 ensemble 同引用）+ shared_embedding。"""

    def __init__(self):
        self.neurons = {
            "tiny_a": make_neuron("tiny_a", seed=1),
            "tiny_b": make_neuron("tiny_b", seed=2),
        }
        torch.manual_seed(7)
        self._shared_embedding = torch.nn.Embedding(GENERAL_VOCAB, BASE_EMBED_DIM)
        self.ensemble = ResonanceEnsemble(
            self.neurons, ResonanceField(dim=512),
            max_rounds=2,
            coaction=CoactivationTracker(),
        )


def snapshot_weights(modules: dict) -> dict:
    """返回 {nid: {param_name: tensor 快照}}，用于断言 live 是否被改动。"""
    out = {}
    for nid, m in modules.items():
        out[nid] = {
            k: v.data.detach().clone() for k, v in m.state_dict().items()
            if v.dtype.is_floating_point
        }
    return out


def weights_equal(snap: dict, modules: dict) -> bool:
    for nid, params in snap.items():
        m = modules.get(nid)
        if m is None:
            return False
        sd = m.state_dict()
        for k, v in params.items():
            if k not in sd or not torch.equal(v, sd[k].data):
                return False
    return True


def run_inference_loop(ens: ResonanceEnsemble, results: dict, stop: threading.Event):
    errors = []
    n_calls = 0
    finite_ok = True
    try:
        for i in range(120):
            if stop.is_set():
                break
            emb = torch.randn(1, 4, BASE_EMBED_DIM)
            out = ens.forward(shared_embeddings=emb)
            n_calls += 1
            for v in out["final_scores"].values():
                if not torch.isfinite(torch.tensor(v)).all():
                    finite_ok = False
            if i % 8 == 0:
                time.sleep(0.01)
    except Exception as e:  # noqa: BLE001
        import traceback
        errors.append(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    results["errors"] = errors
    results["n_calls"] = n_calls
    results["finite_ok"] = finite_ok


def main():
    from taiji.life.sleep_engine import SleepEngine, _clone_module

    print("=" * 60)
    print("训练/推理分离（影子权重 COW）验证")
    print("=" * 60)

    cx = MiniCortex()
    ens = cx.ensemble
    nids_before = set(cx.neurons.keys())

    # 启动推理线程（训练周期全程运行）
    results: dict = {}
    stop = threading.Event()
    t = threading.Thread(target=run_inference_loop, args=(ens, results, stop), daemon=True)
    t.start()
    time.sleep(0.05)

    # ── 模拟睡眠训练的 COW 周期（与 _train_cortex_neurons 完全一致）──
    print("\n[1] 模拟训练 COW 周期（推理进行中）...")
    live_modules = dict(cx.neurons)
    live_emb = cx._shared_embedding
    live_snap = snapshot_weights(live_modules)

    shadow_modules = {nid: _clone_module(m) for nid, m in live_modules.items()}
    shadow_emb = _clone_module(live_emb)
    # dict 引用不变（同引用同步可见），内容换影子
    assert ens.neurons is cx.neurons, "[1] dict 引用断裂（ensemble 与 cortex 不同引用）"
    cx.neurons.update(shadow_modules)
    cx._shared_embedding = shadow_emb
    assert ens.neurons is cx.neurons, "[1] update 后 dict 引用断裂"

    # 训练影子（扰动 lm_head + embed_adapter + shared_embedding 权重）
    with torch.no_grad():
        for nid, m in shadow_modules.items():
            m.lm_head.weight.data.add_(torch.randn_like(m.lm_head.weight) * 0.1)
            if hasattr(m, "embed_adapter"):
                m.embed_adapter.weight.data.add_(torch.randn_like(m.embed_adapter.weight) * 0.1)
        shadow_emb.weight.data.add_(torch.randn_like(shadow_emb.weight) * 0.05)
    time.sleep(0.05)  # 让推理线程在训练期间跑几轮

    # [2] 训练期间 live 权重必须稳定
    assert weights_equal(live_snap, live_modules), \
        "[2] 训练期间 live 权重被改动（影子隔离失效）"
    print("  ok [2] 训练期间 live 权重完全稳定（推理读到稳定权重）")

    # [3] 训练期间推理线程不崩溃（后续汇总断言）

    # 训练期间移除一个 neuron（模拟凋亡），写回应跳过
    removed_nid = "tiny_b"
    cx.neurons.pop(removed_nid)

    # 写回 live ← 影子 + 恢复 live 引用（与 sleep_engine finally 块一致：
    # 只恢复仍在 dict 中的 nid，被移除的不复活）
    SleepEngine._copy_shadow_back(live_modules, live_emb, shadow_modules, shadow_emb)
    for nid in list(cx.neurons.keys()):
        live_n = live_modules.get(nid)
        if live_n is not None:
            cx.neurons[nid] = live_n
    cx._shared_embedding = live_emb
    time.sleep(0.05)

    # [4] 写回后 live == 影子（训练生效）；被移除的 nid 未复活
    assert weights_equal(
        {nid: {k: v.data.detach().clone() for k, v in shadow_modules[nid].state_dict().items()
               if v.dtype.is_floating_point}
         for nid in shadow_modules},
        live_modules,
    ), "[4] 写回后 live != 影子"
    print("  ok [4] 写回后 live == 影子（训练生效）")
    assert removed_nid not in cx.neurons, "[4] 训练期间移除的 neuron 被复活"
    assert removed_nid not in ens.neurons, "[4] ensemble 侧仍含被移除 neuron"
    print(f"  ok [4] 训练期间移除的 {removed_nid} 未复活（写回跳过）")
    assert ens.neurons is cx.neurons, "[4] 恢复后 dict 引用断裂"
    assert cx._shared_embedding is live_emb, "[4] shared_embedding 引用未恢复"

    # 恢复后推理正常（一次完整 forward）
    out = ens.forward(shared_embeddings=torch.randn(1, 4, BASE_EMBED_DIM))
    assert len(out["final_scores"]) >= 1, "[5] 恢复后推理异常"
    print("  ok [5] 恢复 live 引用后推理正常")

    stop.set()
    t.join(timeout=30)

    # ── 汇总 ──
    print(f"\n[6] 推理线程汇总: calls={results.get('n_calls', 0)}")
    errors = results.get("errors", [])
    if errors:
        for e in errors[:3]:
            print(e)
        print("=" * 60)
        return 1
    assert results.get("n_calls", 0) > 0, "推理线程未执行任何 forward"
    assert results.get("finite_ok", False), "推理输出出现非有限值"
    print(f"  ok [6] 推理线程 {results['n_calls']} 次 forward 全部正常"
          f"（训练周期全程不崩溃、分数有限）")

    print(f"\n{'='*60}")
    print("训练/推理分离验证: 全部通过")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
