"""Phase 5.1: 验证 ThalamicRouter 路由效果。

测试 4 个标准 prompt：
  - 你好       (期望 zh)
  - 1+1=       (期望 math)
  - hello world (期望 en)
  - def fibonacci (期望 code)

对比：
  - ThalamicRouter ON  (hard route top-1)
  - ThalamicRouter OFF (原 entropy-based ensemble)
"""
from __future__ import annotations

import os
import sys

# sentencepiece + 项目根目录
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


ROUTER_PATH = "data/distill/thalamic_prototypes.pt"
TEST_PROMPTS = [
    ("你好", "zh"),
    ("1+1=", "math"),
    ("hello world", "en"),
    ("def fibonacci", "code"),
]


def main():
    device = "cpu"

    # 1. 加载 Cortex + teacher
    print("=" * 72)
    print("[Phase 5.1] ThalamicRouter 验证")
    print("=" * 72)
    print("\n[1/4] Loading Cortex + teacher...")
    cortex, tokenizer = create_cortex(
        neurons_dir="data/neurons",
        device=device,
        max_rounds=2,
        enable_gating=False,
    )
    teacher, shared_embedding = load_teacher_model("checkpoint-481000", device=device)
    shared_proj = SharedEmbedProj.load("data/distill/shared_proj.pt", 2048, 512)
    cortex.set_teacher_pipeline(teacher_model=teacher, shared_proj=shared_proj)
    cortex.set_tokenizer(tokenizer)

    # 2. 加载 ThalamicRouter
    print(f"\n[2/4] Loading ThalamicRouter from {ROUTER_PATH}...")
    if not os.path.exists(ROUTER_PATH):
        print(f"  ERROR: {ROUTER_PATH} not found")
        sys.exit(1)
    router = ThalamicRouter.load(ROUTER_PATH, device=device)
    print(f"  Loaded {len(router.prototypes)} prototypes")

    # 3. 验证路由准确性（先做 hard route top-1）
    print("\n[3/4] 路由准确性验证 (top-1 hard routing)")
    print("-" * 72)
    cortex.set_thalamic_router(router, top_k=1)

    correct = 0
    for prompt, expected in TEST_PROMPTS:
        ids = tokenizer.encode(prompt)
        ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            hidden = cortex._extract_hidden_fn(cortex._teacher_model, ids_tensor)
            # 传完整 [L, H]，让 router 自己 mean pool
            decision = router.get_routing_decision(hidden[0])

        top_nid = decision['top_nids'][0] if decision['top_nids'] else 'N/A'
        max_sim = decision['max_sim']
        is_correct = top_nid == expected
        if is_correct:
            correct += 1
        print(f"  prompt: {prompt!r:<20} expected: {expected:<8} -> "
              f"top-1: {top_nid:<8} sim={max_sim:.4f} "
              f"{'✅' if is_correct else '❌'}")
        # 显示 top-3 相似度
        for nid, sim in list(decision['similarities'].items())[:3]:
            marker = " <- selected" if nid == top_nid else ""
            print(f"      {nid:<10} sim={sim:.4f}{marker}")

    print(f"\n  路由准确率: {correct}/{len(TEST_PROMPTS)}")

    # 4. 对比 generate 输出（DR ON vs OFF）
    print("\n[4/4] Generate 输出对比")
    print("-" * 72)

    # top-1 hard route
    print("\n  >> 模式 A: ThalamicRouter ON (hard top-1)")
    cortex.set_thalamic_router(router, top_k=1)
    outputs_on_hard = {}
    for prompt, expected in TEST_PROMPTS:
        out = cortex.generate(prompt, max_tokens=15, temperature=0.7, top_k=20)
        outputs_on_hard[prompt] = out
        # 显示路由信息
        routing = getattr(cortex, '_last_routing', None)
        route_str = ""
        if routing:
            route_str = f" [route: {','.join(routing['top_nids'])}]"
        print(f"  {prompt!r:<20} ->{route_str} {out!r}")

    # top-2 soft route
    print("\n  >> 模式 B: ThalamicRouter ON (soft top-2)")
    cortex.set_thalamic_router(router, top_k=2)
    outputs_on_soft = {}
    for prompt, expected in TEST_PROMPTS:
        out = cortex.generate(prompt, max_tokens=15, temperature=0.7, top_k=20)
        outputs_on_soft[prompt] = out
        routing = getattr(cortex, '_last_routing', None)
        route_str = ""
        if routing:
            route_str = f" [route: {','.join(routing['top_nids'])}]"
        print(f"  {prompt!r:<20} ->{route_str} {out!r}")

    # DR OFF
    print("\n  >> 模式 C: ThalamicRouter OFF (原 ensemble)")
    cortex.thalamic_router = None
    outputs_off = {}
    for prompt, expected in TEST_PROMPTS:
        out = cortex.generate(prompt, max_tokens=15, temperature=0.7, top_k=20)
        outputs_off[prompt] = out
        print(f"  {prompt!r:<20} -> {out!r}")

    # 5. 总结
    print("\n" + "=" * 72)
    print("[Summary]")
    print("=" * 72)
    print(f"\n  {'prompt':<20} | {'DR OFF':<30} | {'DR hard top-1':<30} | {'DR soft top-2':<30}")
    print(f"  {'-'*20}-+-{'-'*30}-+-{'-'*30}-+-{'-'*30}")
    for prompt, _ in TEST_PROMPTS:
        off = outputs_off.get(prompt, '')[:28]
        hard = outputs_on_hard.get(prompt, '')[:28]
        soft = outputs_on_soft.get(prompt, '')[:28]
        print(f"  {prompt:<20} | {off:<30} | {hard:<30} | {soft:<30}")


if __name__ == "__main__":
    main()
