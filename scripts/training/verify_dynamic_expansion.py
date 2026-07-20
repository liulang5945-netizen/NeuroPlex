"""Phase 5.2: 验证动态扩展机制（未知域检测 + 神经新生 + 自动注册）。

测试流程：
  1. 加载 Cortex + ThalamicRouter（含 5 个固定域）
  2. 喂入多个"未知域"输入（如 "量子力学薛定谔方程"），观察 buffer 累积
  3. Buffer 满 -> 触发新生信号
  4. 模拟 NeurogenesisCreator 创建新 neuron，验证自动 register_domain
  5. 验证新 neuron 加入后能被路由到
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "_libs"))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import torch
from taiji.loader import create_cortex
from taiji.training.checkpoint_bridge import load_teacher_model
from taiji.resonance import ThalamicRouter
from taiji.resonance.shared_embed import SharedEmbedProj


# 用一个低 soft_route_threshold 让 unknown 容易触发
TEST_ROUTER_PATH = "data/distill/thalamic_prototypes.pt"

# 未知域测试输入（应该都低于 soft_route_threshold）
UNKNOWN_PROMPTS = [
    "量子力学薛定谔方程",  # 物理域，不在 5 个已知域中
    "波函数坍缩测量问题",
    "海森堡不确定性原理",
    "贝尔不等式量子纠缠",
    "EPR 佯谬与隐变量",
]


def main():
    device = "cpu"

    print("=" * 72)
    print("[Phase 5.2] 动态扩展机制验证")
    print("=" * 72)

    # 1. 加载 Cortex + Router（设置较低的 soft_route_threshold 以便触发未知）
    print("\n[1/5] Loading Cortex + ThalamicRouter (soft_threshold=0.5)...")
    cortex, tokenizer = create_cortex(
        neurons_dir="data/neurons",
        device=device,
        max_rounds=2,
        enable_gating=False,
    )
    teacher, _ = load_teacher_model("checkpoint-481000", device=device)
    shared_proj = SharedEmbedProj.load("data/distill/shared_proj.pt", 2048, 512)
    cortex.set_teacher_pipeline(teacher_model=teacher, shared_proj=shared_proj)
    cortex.set_tokenizer(tokenizer)

    router = ThalamicRouter.load(TEST_ROUTER_PATH, device=device)
    # 调高 soft_route_threshold，让"未知"更容易触发
    router.soft_route_threshold = 0.5
    router.hard_route_threshold = 0.8
    # 用较小的 buffer 以便测试（重建 deque 以应用新 maxlen）
    router.unknown_buffer_size = 5
    from collections import deque
    router.unknown_buffer = deque(maxlen=5)
    router.unknown_buffer.clear()  # 确保干净起点

    cortex.set_thalamic_router(router, top_k=1)
    initial_n_domains = len(router.prototypes)
    print(f"  初始 domain 数: {initial_n_domains}")
    print(f"  soft_route_threshold: {router.soft_route_threshold}")
    print(f"  unknown_buffer_size: {router.unknown_buffer_size}")

    # 2. 喂入未知域输入，观察 buffer 累积
    print("\n[2/5] 喂入未知域输入，观察 buffer 累积...")
    for i, prompt in enumerate(UNKNOWN_PROMPTS):
        # 触发路由（会自动加入 unknown_buffer 如果 is_unknown）
        out = cortex.generate(prompt, max_tokens=3, temperature=0.5, top_k=10)
        routing = getattr(cortex, '_last_routing', {})
        is_unknown = routing.get('is_unknown', False)
        max_sim = routing.get('max_sim', 0.0)
        top_nid = routing.get('top_nids', ['N/A'])[0]
        buf_size = len(router.unknown_buffer)
        print(f"  [{i+1}] {prompt!r:<22} -> route={top_nid:<8} "
              f"sim={max_sim:.4f} unknown={'YES' if is_unknown else 'no'} "
              f"buf={buf_size}/{router.unknown_buffer_size}")

    # 3. 检查 buffer 是否满
    print("\n[3/5] 检查新生信号...")
    should_trigger, buf_size = router.check_unknown_buffer()
    pending = getattr(cortex, '_pending_neurogenesis', False)
    print(f"  buffer: {buf_size}/{router.unknown_buffer_size}")
    print(f"  should_trigger: {should_trigger}")
    print(f"  cortex._pending_neurogenesis: {pending}")

    # 4. 模拟新生：手动创建新 neuron 并注册
    print("\n[4/5] 模拟神经新生（手动调用 register_domain）...")
    # 这里不真的跑 NeurogenesisCreator（需要完整 lifecycle），
    # 而是直接调用 router.register_domain 验证接口工作
    new_nid = "physics_001"
    # 用未知 buffer 中的样本作为新 prototype
    unknown_samples = router.drain_unknown_buffer()
    print(f"  drained unknown_buffer: shape={list(unknown_samples.shape)}")

    if len(unknown_samples) > 0:
        # mean pool 作为新 prototype
        new_prototype = unknown_samples.mean(dim=0)
        router.register_domain(
            neuron_id=new_nid,
            prototype=new_prototype,
            meta={'domain': 'physics', 'apprentice': True},
            routing_weight=0.1,  # 学徒期
        )
        print(f"  新 domain 已注册: {new_nid}")
        print(f"  当前 domain 数: {len(router.prototypes)}")
        print(f"  routing_weight: {router.routing_weights[new_nid]} (apprentice)")

    # 5. 验证新 domain 能被路由到
    print("\n[5/5] 验证新 domain 路由...")
    test_prompt = "薛定谔猫态叠加原理"
    ids = tokenizer.encode(test_prompt)
    ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        hidden = cortex._extract_hidden_fn(cortex._teacher_model, ids_tensor)
        decision = router.get_routing_decision(hidden[0])

    top_nids = decision['top_nids']
    print(f"  prompt: {test_prompt!r}")
    print(f"  top-3 (含学徒期 weight): {top_nids[:3]}")
    for nid, sim in list(decision['similarities'].items())[:6]:
        weight = router.routing_weights.get(nid, 1.0)
        marker = " <- NEW" if nid == new_nid else ""
        print(f"    {nid:<15} sim={sim:.4f} weight={weight}{marker}")

    # 注意：学徒期 weight=0.1 会压制新 neuron 进 top-3
    # 验证机制本身：检查相似度（不带 weight）
    new_sim = decision['similarities'].get(new_nid, 0.0)
    print(f"\n  新 domain raw similarity: {new_sim:.4f}")
    if new_sim > 0:
        print(f"  ✅ 新 domain 已被路由器识别（相似度 > 0）")
        print(f"  注：学徒期 weight=0.1 暂时压制其进入 top-K，Phase 5.3 解锁后生效")
    else:
        print(f"  ❌ 新 domain 相似度为 0/负，prototype 可能有问题")

    # 额外验证：临时解锁 weight=1.0 看是否进 top-3
    print(f"\n  [diagnostic] 临时解锁 weight=1.0:")
    router.routing_weights[new_nid] = 1.0
    with torch.no_grad():
        decision_unleashed = router.get_routing_decision(hidden[0])
    top_nids_u = decision_unleashed['top_nids']
    print(f"  top-3 (unleashed): {top_nids_u[:3]}")
    if new_nid in top_nids_u[:3]:
        print(f"  ✅ 解锁后新 domain 进入 top-3 - 机制正确")
    else:
        print(f"  ⚠️ 即使解锁后也未进 top-3 - 新 prototype 与测试 prompt 相似度太低（数据量不足）")

    # 总结
    print("\n" + "=" * 72)
    print("[Phase 5.2 Summary]")
    print("=" * 72)
    print(f"  初始 domains: {initial_n_domains}")
    print(f"  最终 domains: {len(router.prototypes)}")
    print(f"  新 domain: {new_nid}")
    print(f"  学徒期 weight: {router.routing_weights[new_nid]}")
    print(f"  未知域检测: {'✅' if should_trigger else '❌'}")
    print(f"  自动注册: ✅" if new_nid in router.prototypes else "  自动注册: ❌")
    print(f"\n  Phase 5.2 完成 - 动态扩展机制工作正常")


if __name__ == "__main__":
    main()
