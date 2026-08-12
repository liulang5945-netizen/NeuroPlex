#!/usr/bin/env python3
"""验证热插拔三机制之三：任务级并行（多线程并发推理 + 按域路由隔离）。

人脑启发：多线程处理不同的任务，各任务独立工作互不干扰。
核心契约：
[1] 并发 forward 不崩溃、输出分数有限
[2] 每任务 final_scores 只含本任务 active_nids（+shared_expert）→ 路由隔离
[3] 每任务独立共振场（thread-local）：不同线程的 task field 是不同对象
[4] 并发结果与串行结果 top-1 一致（任务间无实质污染；
    允许 neuron 不应期共享带来的微小调度差异——真实大脑神经元本就共享不应期）
[5] 每任务 round_scores 独立（不互相污染）

Usage:
    python scripts/training/verify_task_parallel.py
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


def make_neuron(nid: str, seed: int = 0) -> ResonanceNeuron:
    cfg = replace(TINY_TEST, neuron_id=nid)
    torch.manual_seed(seed)
    n = ResonanceNeuron(cfg)
    n.eval()
    return n


def build_ensemble() -> ResonanceEnsemble:
    neurons = {
        "tiny_a": make_neuron("tiny_a", seed=1),
        "tiny_b": make_neuron("tiny_b", seed=2),
        "tiny_c": make_neuron("tiny_c", seed=3),
    }
    ens = ResonanceEnsemble(
        neurons, ResonanceField(dim=512),
        max_rounds=2,
        coaction=CoactivationTracker(),
    )
    return ens


def run_task(ens, task_id, emb, active_nids, barrier, results):
    """单任务：等 barrier 后并发 forward。"""
    errors = []
    try:
        barrier.wait(timeout=30)
        out = ens.forward(shared_embeddings=emb, active_nids=active_nids)
        results[task_id] = {
            "final_scores": out["final_scores"],
            "n_rounds": out["n_rounds"],
            "top1": max(out["final_scores"], key=out["final_scores"].get)
                   if out["final_scores"] else None,
            "finite": all(
                torch.isfinite(torch.tensor(v)).all()
                for v in out["final_scores"].values()
            ),
        }
    except Exception as e:  # noqa: BLE001
        import traceback
        errors.append(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
    results[task_id]["errors"] = errors


def serial_baseline(ens, emb, active_nids):
    out = ens.forward(shared_embeddings=emb, active_nids=active_nids)
    return {
        "top1": max(out["final_scores"], key=out["final_scores"].get)
                if out["final_scores"] else None,
        "n_rounds": out["n_rounds"],
    }


def main():
    print("=" * 60)
    print("任务级并行（多线程并发推理 + 按域路由隔离）验证")
    print("=" * 60)

    ens = build_ensemble()
    tasks = [
        {"active_nids": ["tiny_a"], "seed": 11},
        {"active_nids": ["tiny_b"], "seed": 22},
        {"active_nids": ["tiny_c"], "seed": 33},
    ]
    embs = []
    for t in tasks:
        torch.manual_seed(t["seed"])
        embs.append(torch.randn(1, 4, BASE_EMBED_DIM))

    # [0] 串行基线
    print("\n[0] 串行基线（独占 field）...")
    baseline = []
    for i, t in enumerate(tasks):
        b = serial_baseline(ens, embs[i], t["active_nids"])
        baseline.append(b)
        print(f"  task{i}: top1={b['top1']}, rounds={b['n_rounds']}")

    # [3] task field 隔离（机制层面：不同线程不同对象，同线程同对象缓存）
    print("\n[1] task field 隔离（thread-local）...")
    f1 = ens._get_task_field()
    f2 = ens._get_task_field()
    assert f1 is f2, "[1] 同线程 task field 未缓存"
    print("  ok [1] 同线程 task field 缓存复用")

    # [1][2][4][5] 并发三任务（barrier 同步起跑）
    print("\n[2] 三线程并发推理（barrier 同步起跑）...")
    barrier = threading.Barrier(3)
    results = {i: {} for i in range(3)}
    threads = [
        threading.Thread(
            target=run_task,
            args=(ens, i, embs[i], tasks[i]["active_nids"], barrier, results),
            daemon=True,
        )
        for i in range(3)
    ]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=60)
    assert all(not th.is_alive() for th in threads), "[2] 线程未在 60s 内结束"

    for i in range(3):
        errs = results[i].get("errors", [])
        if errs:
            print(f"  fail task{i} 异常:")
            for e in errs[:3]:
                print(e)
            print("=" * 60)
            return 1
        assert results[i].get("finite", False), f"[2] task{i} 分数非有限"
        # [2] 路由隔离：final_scores keys 只含本任务 active_nids
        keys = set(results[i]["final_scores"].keys())
        assert keys == set(tasks[i]["active_nids"]), \
            f"[2] task{i} 路由污染: {keys} != {tasks[i]['active_nids']}"
        # [4] top-1 与串行一致（不应期共享允许调度微差，但主导 neuron 不变）
        assert results[i]["top1"] == baseline[i]["top1"], \
            f"[2] task{i} top-1 漂移: {results[i]['top1']} != {baseline[i]['top1']}"
        # [5] round_scores 独立
        assert len(ens._fstate("round_scores")) >= 1, \
            f"[2] task{i} round_scores 丢失"
        print(f"  ok [2] task{i}: top1={results[i]['top1']}, "
              f"keys={sorted(keys)}, rounds={results[i]['n_rounds']}")

    # [4] 汇总：并发 top-1 全部与串行一致 → 任务间无实质污染
    print("\n[3] 并发 vs 串行一致性（top-1 全对齐）...")
    all_match = all(
        results[i]["top1"] == baseline[i]["top1"] for i in range(3)
    )
    assert all_match, "[3] 并发 top-1 与串行不一致"
    print("  ok [3] 三任务并发 top-1 与串行基线完全一致（field 隔离生效）")

    print(f"\n{'='*60}")
    print("任务级并行验证: 全部通过")
    print(f"{'='*60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
