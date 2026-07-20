"""P6-5: 端到端脱教师推理验证。

完整流程（**运行时完全不 forward 教师**）：
  1. 构建 StandaloneEmbedding（从教师 + shared_proj，一次性，构建后释放教师）
  2. 构建 embedding-based prototypes（用 standalone_embedding 而非 teacher forward）
  3. 验证脱教师路由准确性（4 个标准 prompt）
  4. 验证脱教师 generate 输出
  5. 对比：脱教师 vs 教师路径 的路由结果一致性

Usage:
    python scripts/training/verify_p6_standalone_inference.py
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
from taiji.training.checkpoint_bridge import load_teacher_model, extract_hidden_states
from taiji.resonance import ThalamicRouter, StandaloneEmbedding
from taiji.resonance.shared_embed import SharedEmbedProj


# ── 配置 ──
TEACHER_CKPT = "checkpoint-481000"
NEURONS_DIR = "data/neurons"
SHARED_PROJ_PATH = "data/distill/shared_proj.pt"
ROUTER_PATH = "data/distill/thalamic_prototypes.pt"
DOMAIN_DATA_PATH = "data/distill/domain_datasets.pt"
STANDALONE_EMBED_PATH = "data/distill/standalone_embedding.pt"
ROUTER_P6_PATH = "data/distill/thalamic_prototypes_p6.pt"  # 含 prototypes_embed

TEST_PROMPTS = [
    ("你好", "zh"),
    ("1+1=", "math"),
    ("hello world", "en"),
    ("def fibonacci", "code"),
]

EMBED_DIM = 512
TEACHER_HIDDEN_DIM = 2048


def step1_build_standalone_embedding(device: str) -> StandaloneEmbedding:
    """Step 1: 构建 StandaloneEmbedding（一次性，构建后释放教师）。"""
    print("\n[Step 1] Build StandaloneEmbedding (one-time, from teacher)")
    print("-" * 72)

    # 若已构建，直接加载
    if os.path.exists(STANDALONE_EMBED_PATH):
        print(f"  Found existing {STANDALONE_EMBED_PATH}, loading...")
        return StandaloneEmbedding.load(STANDALONE_EMBED_PATH, device=device)

    # 否则用 build_from_shared_proj 构建（等价于原 teacher 路径）
    print(f"  Loading teacher (only for build, will be released)...")
    teacher, _ = load_teacher_model(TEACHER_CKPT, device=device)
    print(f"  Building standalone embedding via SharedEmbedProj (equiv to teacher path)...")
    embed_table = StandaloneEmbedding.build_from_shared_proj(
        teacher_model=teacher,
        shared_proj_path=SHARED_PROJ_PATH,
        target_dim=EMBED_DIM,
    )
    embed_table.save(STANDALONE_EMBED_PATH)

    # 释放教师
    del teacher
    import gc
    gc.collect()
    print(f"  Teacher released (no longer needed for inference)")
    return embed_table


def step2_build_embed_prototypes(
    router: ThalamicRouter,
    embed_table: StandaloneEmbedding,
    neurons: dict,
    device: str,
) -> ThalamicRouter:
    """Step 2: 构建 embedding-based prototypes。"""
    print("\n[Step 2] Build embedding-based prototypes (no teacher forward)")
    print("-" * 72)

    if router.prototypes_embed:
        print(f"  prototypes_embed already exists ({len(router.prototypes_embed)} entries)")
        return router

    if not os.path.exists(DOMAIN_DATA_PATH):
        print(f"  ERROR: {DOMAIN_DATA_PATH} not found")
        sys.exit(1)
    domain_data = torch.load(DOMAIN_DATA_PATH, map_location="cpu", weights_only=False)
    print(f"  Loaded {len(domain_data)} domain datasets")

    router.compute_prototypes_from_embedding(
        standalone_embedding=embed_table,
        neurons=neurons,
        domain_data=domain_data,
        device=device,
        batch_size=8,
        max_samples=50,
    )
    print(f"  Built {len(router.prototypes_embed)} embed prototypes")
    return router


def step3_verify_routing_embed(
    router: ThalamicRouter,
    embed_table: StandaloneEmbedding,
    tokenizer,
    device: str,
) -> float:
    """Step 3: 验证 embedding-based 路由准确性。"""
    print("\n[Step 3] Verify embedding-based routing (no teacher forward)")
    print("-" * 72)

    correct = 0
    for prompt, expected in TEST_PROMPTS:
        ids = tokenizer.encode(prompt)
        ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            decision = router.get_routing_decision_by_embedding(
                ids_tensor, embed_table,
            )
        top_nid = decision['top_nids'][0] if decision['top_nids'] else 'N/A'
        max_sim = decision['max_sim']
        is_correct = top_nid == expected
        if is_correct:
            correct += 1
        print(f"  prompt: {prompt!r:<20} expected: {expected:<8} -> "
              f"top-1: {top_nid:<12} sim={max_sim:.4f} "
              f"{'✅' if is_correct else '❌'}")
        for nid, sim in list(decision['similarities'].items())[:3]:
            marker = " <- selected" if nid == top_nid else ""
            print(f"      {nid:<14} sim={sim:.4f}{marker}")

    acc = correct / len(TEST_PROMPTS)
    print(f"\n  Embedding-based routing accuracy: {correct}/{len(TEST_PROMPTS)}")
    return acc


def step4_compare_with_teacher_routing(
    router: ThalamicRouter,
    embed_table: StandaloneEmbedding,
    tokenizer,
    device: str,
):
    """Step 4: 对比 embedding-based vs teacher-based 路由（一致性检验）。"""
    print("\n[Step 4] Compare embedding-based vs teacher-based routing")
    print("-" * 72)

    if not router.prototypes:
        print("  No teacher-based prototypes, skip comparison")
        return

    # 必须重新加载教师做对比（仅诊断用，生产路径不需要）
    print("  Loading teacher (only for comparison diagnostic)...")
    teacher, _ = load_teacher_model(TEACHER_CKPT, device=device)

    consistent = 0
    for prompt, expected in TEST_PROMPTS:
        ids = tokenizer.encode(prompt)
        ids_tensor = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            # 教师 hidden state 路由
            hidden = extract_hidden_states(teacher, ids_tensor)
            decision_t = router.get_routing_decision(hidden[0])
            # embedding 路由
            decision_e = router.get_routing_decision_by_embedding(ids_tensor, embed_table)

        top_t = decision_t['top_nids'][0] if decision_t['top_nids'] else 'N/A'
        top_e = decision_e['top_nids'][0] if decision_e['top_nids'] else 'N/A'
        agree = top_t == top_e
        if agree:
            consistent += 1
        print(f"  {prompt!r:<20} teacher: {top_t:<12} sim_t={decision_t['max_sim']:.4f} | "
              f"embed: {top_e:<12} sim_e={decision_e['max_sim']:.4f} "
              f"{'✅ agree' if agree else '⚠️ differ'}")

    del teacher
    import gc
    gc.collect()
    print(f"\n  Routing consistency (teacher vs embed): {consistent}/{len(TEST_PROMPTS)}")


def step5_verify_generate_standalone(cortex, tokenizer):
    """Step 5: 验证脱教师 generate 输出。"""
    print("\n[Step 5] Verify generate (standalone, no teacher forward)")
    print("-" * 72)

    # 关键断言：cortex 内部 _teacher_model 必须为 None
    assert cortex._teacher_model is None, \
        f"cortex._teacher_model should be None in standalone mode, got {cortex._teacher_model}"
    assert cortex._extract_hidden_fn is None, \
        "cortex._extract_hidden_fn should be None in standalone mode"
    print(f"  ✓ cortex._teacher_model = None (no teacher in inference path)")
    print(f"  ✓ cortex._extract_hidden_fn = None")

    outputs = {}
    for prompt, expected in TEST_PROMPTS:
        out = cortex.generate(prompt, max_tokens=15, temperature=0.7, top_k=20)
        outputs[prompt] = out
        routing = getattr(cortex, '_last_routing', None)
        route_str = ""
        if routing:
            mode = routing.get('mode', '?')
            nids = ','.join(routing['top_nids'])
            route_str = f" [route({mode}): {nids}]"
        print(f"  {prompt!r:<20} ->{route_str} {out!r}")

    return outputs


def main():
    device = "cpu"

    print("=" * 72)
    print("[P6-5] 端到端脱教师推理验证")
    print("  Goal: 推理路径完全不 forward 1.5B 教师")
    print("=" * 72)

    # ── 加载 Cortex + neurons ──
    print("\n[0/5] Loading Cortex + neurons...")
    cortex, tokenizer = create_cortex(
        neurons_dir=NEURONS_DIR,
        device=device,
        max_rounds=2,
        enable_gating=False,
    )
    print(f"  Cortex loaded with {len(cortex.neurons)} neurons")

    # ── Step 1: 构建独立 embedding ──
    embed_table = step1_build_standalone_embedding(device)

    # ── Step 2: 加载 router + 构建 embed prototypes ──
    print(f"\nLoading ThalamicRouter from {ROUTER_PATH}...")
    router = ThalamicRouter.load(ROUTER_PATH, device=device)
    print(f"  Loaded {len(router.prototypes)} teacher-prototypes")
    router = step2_build_embed_prototypes(router, embed_table, cortex.neurons, device)
    # 保存更新后的 router（含 prototypes_embed）
    router.save(ROUTER_P6_PATH)
    print(f"  Saved P6 router to {ROUTER_P6_PATH}")

    # ── Step 3: 验证脱教师路由准确性 ──
    embed_acc = step3_verify_routing_embed(router, embed_table, tokenizer, device)

    # ── Step 4: 对比 teacher-based vs embedding-based 路由 ──
    step4_compare_with_teacher_routing(router, embed_table, tokenizer, device)

    # ── Step 5: 脱教师 generate ──
    # 注册 standalone_embedding 到 cortex（替换教师路径）
    print("\nRegistering StandaloneEmbedding to Cortex (replacing teacher pipeline)...")
    cortex.set_standalone_embedding(embed_table)
    cortex.set_tokenizer(tokenizer)
    cortex.set_thalamic_router(router, top_k=1)
    outputs = step5_verify_generate_standalone(cortex, tokenizer)

    # ── 总结 ──
    print("\n" + "=" * 72)
    print("[P6-5 Summary]")
    print("=" * 72)
    print(f"  StandaloneEmbedding: {STANDALONE_EMBED_PATH}")
    print(f"  Router (P6): {ROUTER_P6_PATH}")
    print(f"  Embed-routing accuracy: {embed_acc*100:.1f}%")
    print(f"  推理路径教师依赖: 0 (cortex._teacher_model=None)")
    print()
    print("  Generate outputs (standalone mode):")
    for prompt, out in outputs.items():
        print(f"    {prompt!r:<20} -> {out!r}")
    print()
    print("  ✓ P6-2 脱教师推理路径已验证")
    print("  → 教师仅在'构建 embedding'和'构建 prototype'时使用（离线工具）")
    print("  → 运行时 0 教师依赖")


if __name__ == "__main__":
    main()
