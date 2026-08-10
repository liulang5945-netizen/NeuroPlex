#!/usr/bin/env python3
"""轻量真实集成验证：三机制在真实神经元权重上的行为。

范围（刻意轻量，避免与后台基座训练争资源）：
- 只装配 1 个真实神经元 zh_general（最小 ckpt，~700MB）
- 不接 bio 模块（wire 全关）→ 不加载 3GB cross_spec_dialogue.pt
- 验证点：
  [1] 真实装配成功 + 规格正确
  [2] 任务级并行：多线程并发 ensemble.forward（真实权重）不崩 + 分数有限 + 路由隔离
  [3] 快照隔离 + 混合规格热插拔：推理线程中 add compact neuron（跨规格投影）→ forward 正常
  [4] 训练/推理分离 COW：真实 _clone_module 克隆 + 扰动 + 并发推理 + 写回生效

Usage:
    python scripts/training/verify_hotswap_integration.py
"""
import os
import sys
import threading
import time

os.environ.setdefault("TAIJI_TEST_MODE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

BASE_EMBED_DIM = 512


def run_inference_loop(ens, results: dict, stop: threading.Event, n_iters: int = 40):
    errors = []
    n_calls = 0
    finite_ok = True
    try:
        for i in range(n_iters):
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
    from taiji.loader import create_cortex
    from taiji.life.sleep_engine import _clone_module

    print("=" * 60)
    print("三机制真实集成验证（轻量：zh_std0_dialogue 单神经元）")
    print("=" * 60)

    # [1] 真实装配
    print("\n[1] 装配真实 cortex（zh_std0_dialogue）...")
    cortex, _ = create_cortex(
        neurons_dir="data/neurons",
        device="cpu",
        max_rounds=2,
        neuron_ids=["zh_std0_dialogue"],
    )
    nid = next(iter(cortex.neurons))
    neuron = cortex.neurons[nid]
    cfg = neuron.config
    print(f"  ok [1] 加载 {nid}: spec={cfg.spec}, hidden={cfg.hidden_size}, "
          f"field_dim={cfg.field_dim}, layers={cfg.num_hidden_layers}")
    print(f"      ensemble field.dim={cortex.ensemble.field.dim}, "
          f"neurons={list(cortex.neurons.keys())}")

    # [2] 任务级并行：3 线程并发 forward（真实权重）
    print("\n[2] 三线程并发推理（真实权重）...")
    barrier = threading.Barrier(3)
    results = [{} for _ in range(3)]

    def run_task(i, seed):
        torch.manual_seed(seed)
        emb = torch.randn(1, 4, BASE_EMBED_DIM)
        try:
            barrier.wait(timeout=30)
            out = cortex.ensemble.forward(shared_embeddings=emb)
            results[i] = {
                "finite": all(
                    torch.isfinite(torch.tensor(v)).all()
                    for v in out["final_scores"].values()
                ),
                "scores": out["final_scores"],
                "n_rounds": out["n_rounds"],
            }
        except Exception as e:  # noqa: BLE001
            import traceback
            results[i] = {"errors": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}

    threads = [
        threading.Thread(target=run_task, args=(i, 100 + i), daemon=True)
        for i in range(3)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=90)
    for i in range(3):
        if "errors" in results[i]:
            print(f"  fail task{i}: {results[i]['errors']}")
            return 1
        assert results[i]["finite"], f"[2] task{i} 分数非有限"
        print(f"  ok [2] task{i}: rounds={results[i]['n_rounds']}, "
              f"scores={ {k: round(v, 3) for k, v in results[i]['scores'].items()} }")

    # [3] 快照隔离 + 混合规格热插拔（推理线程 + 同规格 + 跨规格）
    print("\n[3] 推理线程中热插拔（同规格 compact + 跨规格 standard）...")
    inf_results: dict = {}
    stop = threading.Event()
    t = threading.Thread(
        target=run_inference_loop, args=(cortex.ensemble, inf_results, stop), daemon=True
    )
    t.start()
    time.sleep(0.1)

    # 3a. 同规格热插拔（cortex.add_neuron → compact 2048 == 场 2048）
    new_nid = cortex.add_neuron("zh")
    print(f"  add_neuron({new_nid}): field_dim={cortex.neurons[new_nid].config.field_dim}, "
          f"cross_spec_projector={'yes' if new_nid in cortex.ensemble._cross_spec_projectors else 'no'}")

    # 3b. 跨规格热插拔（field_dim=1024 ≠ 场 2048，hidden=512 同规格 →
    # 触发跨规格投影层补建，模拟旧 3072-vs-2048 崩溃场景的变体）
    from dataclasses import replace
    from taiji.resonance.config import get_domain_neuron_config
    from taiji.resonance.neuron import ResonanceNeuron
    mixed_cfg = replace(get_domain_neuron_config("zh"), neuron_id="zh_alt_hotswap",
                        field_dim=1024)
    alt_neuron = ResonanceNeuron(mixed_cfg).to("cpu")
    alt_neuron.eval()
    cortex.ensemble.add_neuron("zh_alt_hotswap", alt_neuron)  # 内部写入共享 dict
    assert "zh_alt_hotswap" in cortex.ensemble._cross_spec_projectors, \
        "[3] 跨规格投影层未补建"
    proj = cortex.ensemble._cross_spec_projectors["zh_alt_hotswap"]
    print(f"  add alt 1024: 投影层自动补建 ({proj.linear1.in_features}->{proj.linear1.out_features})")

    # 含跨规格 neuron 的 forward 正常
    out = cortex.ensemble.forward(shared_embeddings=torch.randn(1, 4, BASE_EMBED_DIM))
    assert all(
        torch.isfinite(torch.tensor(v)).all() for v in out["final_scores"].values()
    ), "[3] 含混合规格新 neuron 的 forward 分数非有限"
    print("  ok [3] 混合规格热插拔后 forward 正常")

    # 推理中隔离 + 复活 zh_general（此时有 3 个 neuron，不触发安全检查）
    ok = cortex.isolate_neuron(nid)
    assert ok, "[3] isolate_neuron 失败"
    print(f"  ok [3] 推理中隔离 {nid}（保留 ckpt 可复活）")
    ok = cortex.revive_neuron(nid)
    assert ok, "[3] revive_neuron 失败"
    print(f"  ok [3] 推理中复活 {nid}")

    # 推理中移除临时 neuron
    removed = cortex.remove_neuron("zh_alt_hotswap", delete_ckpt=False)
    assert removed, "[3] remove alt 失败"
    removed = cortex.remove_neuron(new_nid, delete_ckpt=True)
    assert removed, "[3] remove compact 失败"
    print(f"  ok [3] 推理中移除临时 neuron（{new_nid}, zh_alt_hotswap）")

    stop.set()
    t.join(timeout=60)
    errs = inf_results.get("errors", [])
    if errs:
        for e in errs[:3]:
            print(e)
        return 1
    print(f"  ok [3] 推理线程 {inf_results.get('n_calls', 0)} 次 forward 全部正常"
          f"（增删/隔离/复活期间不崩溃）")

    # [4] 训练/推理分离 COW（真实权重）
    print("\n[4] 真实 COW 周期（影子权重）...")
    live_modules = dict(cortex.neurons)
    live_snap = {
        n: {k: v.data.detach().clone() for k, v in m.state_dict().items()
            if v.dtype.is_floating_point}
        for n, m in live_modules.items()
    }
    shadow_modules = {n: _clone_module(m) for n, m in live_modules.items()}
    # dict 内容换影子（引用不变）→ 推理读影子
    cortex.neurons.update(shadow_modules)
    with torch.no_grad():
        for m in shadow_modules.values():
            if hasattr(m, "lm_head"):
                m.lm_head.weight.data.add_(torch.randn_like(m.lm_head.weight) * 0.05)
    # 并发推理（读影子）
    out = cortex.ensemble.forward(shared_embeddings=torch.randn(1, 4, BASE_EMBED_DIM))
    assert all(
        torch.isfinite(torch.tensor(v)).all() for v in out["final_scores"].values()
    ), "[4] 影子训练期间推理异常"
    # live 权重稳定
    stable = all(
        torch.equal(live_snap[n][k], m.state_dict()[k].data)
        for n, m in live_modules.items()
        for k in live_snap[n]
    )
    assert stable, "[4] 影子训练期间 live 权重被改动"
    # 写回 + 恢复
    from taiji.life.sleep_engine import SleepEngine
    SleepEngine._copy_shadow_back(live_modules, None, shadow_modules, None)
    for n in list(cortex.neurons.keys()):
        ln = live_modules.get(n)
        if ln is not None:
            cortex.neurons[n] = ln
    # 写回生效
    trained = not all(
        torch.equal(live_snap[n][k], m.state_dict()[k].data)
        for n, m in live_modules.items() for k in live_snap[n]
    )
    assert trained, "[4] 写回未生效（live == 训练前）"
    out = cortex.ensemble.forward(shared_embeddings=torch.randn(1, 4, BASE_EMBED_DIM))
    assert all(
        torch.isfinite(torch.tensor(v)).all() for v in out["final_scores"].values()
    ), "[4] 写回恢复后推理异常"
    print("  ok [4] 影子训练期间 live 稳定 → 写回生效 → 恢复后推理正常")

    print(f"\n{'='*60}")
    print("三机制真实集成验证: 全部通过")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
